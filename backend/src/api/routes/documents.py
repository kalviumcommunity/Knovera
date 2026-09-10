"""
src/api/routes/documents.py

Document upload, processing, indexing, and knowledge base management endpoints.
Persists document metadata in MongoDB Atlas with multi-tenant admin isolation.
"""

import datetime
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Depends, Query, Header

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
from src.services.mongo_storage import get_mongo_storage

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
    admin_id: Optional[str] = Query(None),
    x_admin_id: Optional[str] = Header(None, alias="X-Admin-Id"),
    config: APIConfig = Depends(get_config),
    vector_db: VectorDatabase = Depends(get_vector_db)
) -> DocumentUploadResponse:
    """
    Document Upload & Indexing Endpoint.
    Stores file, chunks and indexes into MongoDB Atlas vector storage tagged with admin_id.
    """
    target_admin = admin_id or x_admin_id or "admin@knovera.ai"
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
        active_gen = getattr(vector_db, "embedding_generator", getattr(vector_db, "generator", None))
        summary_dict = process_uploaded_document(
            path=stored_path,
            vector_db=vector_db,
            embedding_generator=active_gen,
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

        # Step 3: Record in MongoDB documents collection
        storage = get_mongo_storage()
        storage.record_document({
            "filename": file.filename or stored_path.name,
            "admin_id": target_admin,
            "chunk_count": summary_model.chunks,
            "file_size": file_size,
            "collection_name": config.collection_name,
            "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "indexed"
        }, admin_id=target_admin)

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
    admin_id: Optional[str] = Query(None),
    x_admin_id: Optional[str] = Header(None, alias="X-Admin-Id"),
    config: APIConfig = Depends(get_config),
    vector_db: VectorDatabase = Depends(get_vector_db)
) -> DocumentUploadResponse:
    """Alias for /documents endpoint."""
    return await upload_document(file=file, admin_id=admin_id, x_admin_id=x_admin_id, config=config, vector_db=vector_db)


@router.get(
    "/documents",
    response_model=DocumentListResponse
)
def list_documents(
    admin_id: Optional[str] = Query(None),
    x_admin_id: Optional[str] = Header(None, alias="X-Admin-Id"),
    config: APIConfig = Depends(get_config),
    vector_db: VectorDatabase = Depends(get_vector_db)
) -> DocumentListResponse:
    """List documents stored in MongoDB Atlas for the requesting admin."""
    target_admin = admin_id or x_admin_id or "admin@knovera.ai"
    storage = get_mongo_storage()
    mongo_docs = storage.list_documents(admin_id=target_admin)

    total_chunks = 0
    try:
        if vector_db.is_reachable():
            total_chunks = vector_db.count(config.collection_name)
    except Exception:
        pass

    docs_info: List[DocumentInfo] = []
    
    if mongo_docs:
        for doc in mongo_docs:
            docs_info.append(
                DocumentInfo(
                    filename=doc.get("filename", "unknown"),
                    chunk_count=doc.get("chunk_count", 0),
                    collection_name=doc.get("collection_name", config.collection_name),
                    indexed_at=doc.get("uploaded_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
                )
            )
    else:
        # Fallback to files on disk and sample directory
        upload_dir = Path(config.upload_dir)
        seen = set()
        if upload_dir.exists():
            for item in upload_dir.glob("*.*"):
                if item.is_file() and not item.name.startswith("."):
                    seen.add(item.name)
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


@router.delete(
    "/documents/{filename}",
    status_code=status.HTTP_200_OK
)
def delete_document(
    filename: str,
    admin_id: Optional[str] = Query(None),
    x_admin_id: Optional[str] = Header(None, alias="X-Admin-Id"),
    config: APIConfig = Depends(get_config),
    vector_db: VectorDatabase = Depends(get_vector_db)
):
    """Delete a document record from MongoDB Atlas, disk, and vector collection."""
    target_admin = admin_id or x_admin_id or "admin@knovera.ai"
    storage = get_mongo_storage()
    storage.delete_document(filename, admin_id=target_admin)

    # Delete from vector store
    try:
        if vector_db.is_reachable():
            vector_db.delete_by_source(config.collection_name, filename)
    except Exception as e:
        logger.warning(f"Could not purge vectors for '{filename}': {e}")

    # Delete from uploads directory if exists
    upload_dir = Path(config.upload_dir)
    target_path = upload_dir / filename
    if target_path.exists() and target_path.is_file():
        try:
            target_path.unlink()
        except Exception:
            pass

    return {"status": "deleted", "filename": filename}
