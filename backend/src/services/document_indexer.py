"""
src/document_indexer.py

Document Ingestion, Chunking, Embedding & Vector Indexing Engine.
Provides runtime document intake for the Knovera RAG Backend API Service.
Enables new content to be uploaded, ingested, chunked, embedded, and indexed
into the vector database dynamically so it becomes immediately searchable
without restarting the server application.
"""

import os
import sys
import time
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from fastapi import UploadFile, HTTPException, status

# Ensure Knovera root is in sys.path
_root_dir = Path(__file__).resolve().parent.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

from src.document_loader import DocumentLoader
from src.text_cleaner import TextCleaner
from src.token_chunker import TokenChunker
from src.chunk_tagger import ChunkTagger
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import VectorDatabase
from src.api_config import get_config, APIConfig

logger = logging.getLogger("knovera.indexer")

# Supported file extensions for document ingestion
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".html", ".htm"}

# Default upload configuration
DEFAULT_UPLOAD_DIR = _root_dir / "uploads"
DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit


# ============================================================================
# UPLOAD VALIDATION & SAFE PERSISTENCE
# ============================================================================

def validate_upload_metadata(
    filename: Optional[str],
    file_size: Optional[int] = None,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
) -> str:
    """
    Validates uploaded file metadata (filename, extension, size).
    
    Args:
        filename: Name of the uploaded file.
        file_size: Optional measured size of the file in bytes.
        max_size_bytes: Maximum permitted file size in bytes.
        
    Returns:
        str: Validated lowercase file extension (e.g. '.md').
        
    Raises:
        HTTPException:
            - 400 Bad Request if filename is missing or empty.
            - 415 Unsupported Media Type if file extension is not supported.
            - 413 Payload Too Large if file exceeds size limit.
    """
    if not filename or not filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename cannot be empty."
        )

    # Sanitize filename
    clean_name = Path(filename).name.strip()
    if not clean_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename specified."
        )

    suffix = Path(clean_name).suffix.lower()
    if not suffix or suffix not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Supported formats are: {supported_str}"
        )

    if file_size is not None:
        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)."
            )
        if file_size > max_size_bytes:
            mb_limit = max_size_bytes / (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({round(file_size / (1024 * 1024), 2)} MB) exceeds maximum allowed limit ({mb_limit} MB)."
            )

    return suffix


async def store_upload(
    file: UploadFile,
    upload_dir: Optional[Path] = None,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
) -> Tuple[Path, int]:
    """
    Safely stores an uploaded file to disk after strict validation.
    
    Args:
        file: FastAPI UploadFile object.
        upload_dir: Directory path where uploaded files are stored.
        max_size_bytes: Maximum permitted size in bytes.
        
    Returns:
        Tuple[Path, int]: Tuple of (saved file path, total bytes written).
    """
    # 1. Validate metadata
    raw_filename = file.filename or ""
    validate_upload_metadata(raw_filename, max_size_bytes=max_size_bytes)

    # 2. Read content & validate size
    content = await file.read()
    file_size = len(content)
    validate_upload_metadata(raw_filename, file_size=file_size, max_size_bytes=max_size_bytes)

    # 3. Create target directory
    target_dir = upload_dir or DEFAULT_UPLOAD_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # 4. Safe destination path (prevent directory traversal)
    safe_name = Path(raw_filename).name
    dest_path = target_dir / safe_name

    # 5. Persist binary content to disk
    dest_path.write_bytes(content)
    logger.info(f"Stored uploaded file '{safe_name}' ({file_size} bytes) to {dest_path}")

    return dest_path, file_size


# ============================================================================
# DOCUMENT INGESTION, CHUNKING & INDEXING PIPELINE
# ============================================================================

def process_uploaded_document(
    path: Path,
    vector_db: Optional[VectorDatabase] = None,
    embedding_generator: Optional[EmbeddingGenerator] = None,
    collection_name: Optional[str] = None,
    chunk_size: int = 400,
    overlap: int = 60
) -> Dict[str, Any]:
    """
    Executes the complete document ingestion pipeline on an uploaded file:
    Load Text -> Clean Text -> Token Chunk -> Tag Metadata -> Batch Embed -> Upsert into ChromaDB.
    
    Args:
        path: Path to the stored document on disk.
        vector_db: Optional VectorDatabase instance.
        embedding_generator: Optional EmbeddingGenerator instance.
        collection_name: Target vector store collection name.
        chunk_size: Token chunk capacity (default: 400 tokens).
        overlap: Token chunk overlap (default: 60 tokens).
        
    Returns:
        Dict[str, Any]: Detailed indexing summary payload.
    """
    start_time = time.perf_counter()
    latencies: Dict[str, float] = {}

    cfg = get_config()
    target_collection = collection_name or cfg.collection_name
    active_db = vector_db or VectorDatabase(
        persist_dir=cfg.vector_db_url,
        embedding_generator=embedding_generator
    )
    active_generator = embedding_generator or EmbeddingGenerator(
        model_name=cfg.embedding_model,
        api_key=cfg.active_api_key or None
    )

    # ------------------------------------------------------------------------
    # Stage 1: Multi-Format Document Loading
    # ------------------------------------------------------------------------
    t0 = time.perf_counter()
    loader = DocumentLoader(verbose=False)
    try:
        raw_text = loader.load_text(path)
    except Exception as e:
        logger.error(f"Failed to load document text from '{path}': {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to extract text from '{path.name}': {str(e)}"
        )
    latencies["load_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if not raw_text or not raw_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{path.name}' contains no readable textual content."
        )

    # ------------------------------------------------------------------------
    # Stage 2: Text Cleaning & Normalization
    # ------------------------------------------------------------------------
    t0 = time.perf_counter()
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean(raw_text)
    latencies["clean_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ------------------------------------------------------------------------
    # Stage 3: Token-Aware Chunking
    # ------------------------------------------------------------------------
    t0 = time.perf_counter()
    chunker = TokenChunker(
        default_chunk_size=chunk_size,
        default_overlap=overlap
    )
    raw_chunks = chunker.token_chunks_with_offsets(cleaned_text)
    latencies["chunk_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if not raw_chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No viable chunks could be created from document '{path.name}'."
        )

    # ------------------------------------------------------------------------
    # Stage 4: Metadata Tagging
    # ------------------------------------------------------------------------
    t0 = time.perf_counter()
    tagger = ChunkTagger()
    doc_title = path.stem.replace("-", " ").replace("_", " ").title()

    tagged_chunks: List[Dict[str, Any]] = []
    for idx, c in enumerate(raw_chunks):
        chunk_text = c.get("text", "")
        # Attempt to infer section header from text
        first_line = chunk_text.strip().split("\n")[0] if chunk_text else ""
        section_heading = first_line[:80].strip("# ") if first_line.startswith("#") else f"Section {idx+1}"

        meta = tagger.create_metadata(
            source=path.name,
            chunk_index=idx,
            char_start=c.get("char_start", 0),
            char_end=c.get("char_end", len(chunk_text)),
            file_type=path.suffix,
            doc_title=doc_title,
            section_heading=section_heading
        )
        tagged_chunks.append({
            "id": f"{path.name}:{idx}",
            "text": chunk_text,
            "metadata": meta,
            "chunk_index": idx
        })
    latencies["tag_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ------------------------------------------------------------------------
    # Stage 5: Dense Vector Embedding Generation
    # ------------------------------------------------------------------------
    t0 = time.perf_counter()
    texts_to_embed = [c["text"] for c in tagged_chunks]
    embeddings = active_generator.embed(texts_to_embed)
    latencies["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if len(embeddings) != len(tagged_chunks):
        raise RuntimeError(f"Embedding count mismatch: expected {len(tagged_chunks)}, got {len(embeddings)}")

    # ------------------------------------------------------------------------
    # Stage 6: Vector Database Upsert & Indexing
    # ------------------------------------------------------------------------
    t0 = time.perf_counter()
    records: List[Dict[str, Any]] = []
    for chunk_dict, emb in zip(tagged_chunks, embeddings):
        records.append({
            "id": chunk_dict["id"],
            "vector": emb,
            "text": chunk_dict["text"],
            "metadata": chunk_dict["metadata"]
        })

    upsert_ok = active_db.upsert_records(records, collection_name=target_collection)
    latencies["index_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if not upsert_ok:
        raise RuntimeError(f"Vector store upsert failed for collection '{target_collection}'")

    total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    latencies["total_ms"] = total_time_ms

    logger.info(
        f"Successfully indexed '{path.name}': {len(records)} chunks into '{target_collection}' in {total_time_ms} ms."
    )

    return {
        "document": str(path),
        "filename": path.name,
        "raw_characters": len(raw_text),
        "cleaned_characters": len(cleaned_text),
        "chunks": len(records),
        "indexed": len(records),
        "collection_name": target_collection,
        "stage_latencies_ms": latencies,
        "status": "indexed"
    }
