"""
src/api/routes/chunks.py

Knowledge Base Chunks Inspection API.
Queries actual indexed passages and vector chunks from ChromaDB and SQLite.
"""

import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Query, Depends
from pydantic import BaseModel

from src.db import get_db_connection
from src.config import get_config, APIConfig

router = APIRouter(prefix="/api/chunks", tags=["Knowledge Base Chunks"])


class ChunkModel(BaseModel):
    id: str
    sourceDoc: str
    section: Optional[str] = ""
    chunkIndex: int = 0
    tokenCount: int = 0
    content: str
    embeddingModel: Optional[str] = "BAAI/bge-small-en-v1.5"
    indexedAt: str
    metadata: Dict[str, Any] = {}


class ChunkListResponse(BaseModel):
    chunks: List[ChunkModel]
    totalChunks: int
    totalDocuments: int
    embeddingModel: str
    dimension: int


@router.get("", response_model=ChunkListResponse)
def get_chunks(
    search: Optional[str] = Query(None, description="Search chunk text or section"),
    doc: Optional[str] = Query(None, description="Filter by source document filename"),
    config: APIConfig = Depends(get_config)
):
    """Retrieve indexed passages and vector chunks from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query_parts = ["SELECT * FROM knowledge_chunks WHERE 1=1"]
    params = []

    if doc and doc != "all":
        query_parts.append("AND source_doc = ?")
        params.append(doc)

    if search:
        search_like = f"%{search}%"
        query_parts.append("AND (content LIKE ? OR section LIKE ? OR source_doc LIKE ?)")
        params.extend([search_like, search_like, search_like])

    query_parts.append("ORDER BY chunk_index ASC")

    cursor.execute(" ".join(query_parts), params)
    rows = cursor.fetchall()

    cursor.execute("SELECT COUNT(DISTINCT source_doc) FROM knowledge_chunks")
    doc_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM knowledge_chunks")
    total_chunks = cursor.fetchone()[0]

    conn.close()

    chunks = []
    for r in rows:
        meta = {}
        if r["metadata_json"]:
            try:
                meta = json.loads(r["metadata_json"])
            except Exception:
                pass

        chunks.append(
            ChunkModel(
                id=r["id"],
                sourceDoc=r["source_doc"],
                section=r["section"] or "",
                chunkIndex=r["chunk_index"],
                tokenCount=r["token_count"],
                content=r["content"],
                embeddingModel=r["embedding_model"] or config.embedding_model,
                indexedAt=r["indexed_at"],
                metadata=meta
            )
        )

    return ChunkListResponse(
        chunks=chunks,
        totalChunks=total_chunks,
        totalDocuments=doc_count,
        embeddingModel=config.embedding_model,
        dimension=384
    )
