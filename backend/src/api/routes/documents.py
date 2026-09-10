"""
src/api/routes/documents.py

Document upload, processing, indexing, and knowledge base management endpoints.
"""

import datetime
import logging
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Depends

from src.config import get_config, APIConfig
from src.models.schemas import (
    DocumentUploadResponse,
    DocumentIndexingSummary,
    DocumentListResponse,
    DocumentInfo
)
from src.services.document_indexer import store_upload, process_uploaded_document
from src.services.vector_store import VectorDatabase
from src.services.embedding_generator import EmbeddingGenerator

logger = logging.getLogger("knovera.api.documents")
router = APIRouter(tags=["Document Ingestion"])


def get_vector_db(config: APIConfig = Depends(get_config)) -> VectorDatabase:
    generator = EmbeddingGenerator(
        model_name=config.embedding_model,
        api_key=config.active_api_key or None,
        base_url=config.openrouter_base_url if config.openrouter_api_key else config.openai_base_url
    )
    return VectorDatabase(
        persist_dir=config.vector_db_url,
        embedding_generator=generator
    )


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"description": "Bad Request (empty file or invalid document)"},
        413: {"description": "Payload Too Large (file exceeds size limit)"},
        415: {"description": "Unsupported Media Type (unsupported file extension)"},
        500: {"description": "Internal Server Error (indexing failure)"}
    }
)
async def upload_document(
    file: UploadFile = File(...),
    config: APIConfig = Depends(get_config),
    vector_db: VectorDatabase = Depends(get_vector_db)
) -> DocumentUploadResponse:
    """
    Document Upload & Indexing Endpoint.
    Accepts a multi-format document file (.txt, .md, .pdf, .html), validates metadata,
    stores it safely, executes text cleaning, token chunking, metadata tagging,
    dense vector embedding, and indexes it into the ChromaDB vector database so
    it becomes immediately searchable at runtime without restarting the application.
    """
    try:
        upload_dir = Path(config.upload_dir)
        max_bytes = config.max_upload_size_mb * 1024 * 1024

        # Step 1: Validate and safely store upload
        stored_path, file_size = await store_upload(
            file=file,
            upload_dir=upload_dir,
            max_size_bytes=max_bytes
        )

        # Step 2: Ingest, clean, chunk, embed, and index
        summary_dict = process_uploaded_document(
            path=stored_path,
            vector_db=vector_db,
            embedding_generator=vector_db.embedding_generator,
            collection_name=config.collection_name
        )

        summary_model = DocumentIndexingSummary(
            document=summary_dict["document"],
            filename=summary_dict["filename"],
            chunks=summary_dict["chunks"],
            indexed=summary_dict["indexed"],
            raw_characters=summary_dict.get("raw_characters"),
            cleaned_characters=summary_dict.get("cleaned_characters"),
            collection_name=summary_dict.get("collection_name"),
            stage_latencies_ms=summary_dict.get("stage_latencies_ms")
        )

        return DocumentUploadResponse(
            status="indexed",
            filename=file.filename or stored_path.name,
            summary=summary_model,
            message=f"Successfully indexed {summary_model.chunks} chunks from '{file.filename}' into collection '{config.collection_name}'.",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )

    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"Document upload/indexing failed for '{file.filename}': {err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document indexing failed: {str(err)}"
        )


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False
)
async def upload_document_alias(
    file: UploadFile = File(...),
    config: APIConfig = Depends(get_config),
    vector_db: VectorDatabase = Depends(get_vector_db)
) -> DocumentUploadResponse:
    """Alias for /documents endpoint."""
    return await upload_document(file=file, config=config, vector_db=vector_db)


@router.get(
    "/documents",
    response_model=DocumentListResponse
)
def list_documents(
    config: APIConfig = Depends(get_config),
    vector_db: VectorDatabase = Depends(get_vector_db)
) -> DocumentListResponse:
    """List documents stored in the uploads directory and collection chunk count."""
    upload_dir = Path(config.upload_dir)
    docs_info: List[DocumentInfo] = []
    
    total_chunks = 0
    try:
        if vector_db.is_reachable():
            total_chunks = vector_db.count(config.collection_name)
    except Exception:
        pass

    if upload_dir.exists():
        for item in upload_dir.glob("*.*"):
            if item.is_file() and not item.name.startswith("."):
                docs_info.append(
                    DocumentInfo(
                        filename=item.name,
                        chunk_count=0,
                        collection_name=config.collection_name,
                        indexed_at=datetime.datetime.fromtimestamp(
                            item.stat().st_mtime, tz=datetime.timezone.utc
                        ).isoformat()
                    )
                )

    return DocumentListResponse(
        documents=docs_info,
        total_documents=len(docs_info),
        total_chunks=total_chunks
    )
