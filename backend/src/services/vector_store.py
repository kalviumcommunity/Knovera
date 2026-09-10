"""
src/vector_store.py

Enterprise Vector Database Management & Collection Design Engine.
Designed for Knovera RAG Assistant using ChromaDB (Persistent & In-Memory).

Key Capabilities:
1. Reachable Vector Store Connection: Connects to local persistent storage or in-memory instance via environment configuration.
2. Dimension-Guarded Collection Design: Configures collections with precise vector dimensions (1536 for text-embedding-3-small) and distance metrics (cosine/l2/ip).
3. Grounded Record Schema: Binds high-dimensional vector embeddings with source text and rich hierarchical metadata.
4. CRUD & Readback Verification: Provides atomic upserts, exact ID lookups, metadata filtering, and nearest-neighbor search.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dotenv import load_dotenv

# Python 3.8 compatibility shim for chromadb product telemetry
if sys.version_info < (3, 9):
    from unittest.mock import MagicMock
    if "posthog" not in sys.modules:
        sys.modules["posthog"] = MagicMock()

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

from src.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


# Canonical Record Schema Definition for Knovera RAG Collections
STORED_RECORD_SCHEMA: Dict[str, Any] = {
    "id": "str - Unique deterministic chunk identifier (e.g., 'account-guide.md#chunk_0')",
    "vector": "List[float] - Embedding vector of length VECTOR_DIMENSION (1536)",
    "text": "str - Original raw text chunk utilized for LLM context grounding",
    "metadata": {
        "source": "str - Document file path or source URI",
        "chunk_index": "int - 0-indexed position of chunk within the source document",
        "section": "str - Heading or semantic section identifier",
        "doc_title": "str - Title of the source document",
        "category": "str - Domain category (e.g., 'Authentication', 'Billing', 'Campus Life')"
    }
}


class VectorDatabase:
    """
    Manages Vector Database connections, collection lifecycles, and grounded record operations.
    """

    DEFAULT_COLLECTION_NAME = "rag_chunks"
    DEFAULT_DIMENSION = 1536
    DEFAULT_METRIC = "cosine"
    DEFAULT_PERSIST_DIR = os.path.join(".", "data", "chroma_db")

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        in_memory: bool = False,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """
        Initialize the Vector Database connection.
        
        Args:
            persist_dir: Directory path for ChromaDB on-disk persistence.
            in_memory: If True, uses ephemeral in-memory storage (ideal for ephemeral testing).
            embedding_generator: Optional EmbeddingGenerator for dimension resolution.
        """
        load_dotenv()
        if not HAS_CHROMADB:
            raise ImportError("chromadb is not installed. Please install via 'pip install chromadb'.")

        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", self.DEFAULT_PERSIST_DIR)
        self.in_memory = in_memory or (os.getenv("VECTOR_DB_IN_MEMORY", "false").lower() == "true")
        self.generator = embedding_generator or EmbeddingGenerator()
        self.expected_dimension = self.generator.dimension or self.DEFAULT_DIMENSION

        # Initialize ChromaDB Client
        if self.in_memory:
            self.client = chromadb.EphemeralClient()
            self.storage_mode = "Ephemeral (In-Memory)"
            logger.info("Initialized ephemeral in-memory ChromaDB client.")
        else:
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.storage_mode = f"Persistent Storage ({os.path.abspath(self.persist_dir)})"
            logger.info(f"Initialized persistent ChromaDB client at: {self.persist_dir}")

        self.active_collection = None

    def is_reachable(self) -> bool:
        """
        Confirms whether the Vector Database client is connected and reachable.
        """
        if self.client is None:
            return False
        try:
            self.client.heartbeat()
            return True
        except Exception as e:
            logger.warning(f"Heartbeat check failed: {e}")
            try:
                # Fallback connectivity check
                self.client.list_collections()
                return True
            except Exception:
                return False

    def create_or_get_collection(
        self,
        name: str = DEFAULT_COLLECTION_NAME,
        dimension: Optional[int] = None,
        metric: str = DEFAULT_METRIC,
        metadata: Optional[Dict[str, Any]] = None
    ) -> chromadb.Collection:
        """
        Creates or accesses a Chroma collection with explicit vector space parameters.
        
        Args:
            name: Name of the collection.
            dimension: Target vector dimension (defaults to generator dimension, 1536).
            metric: Distance metric ('cosine', 'l2', or 'ip').
            metadata: Optional collection-level metadata.
        """
        dim = dimension or self.expected_dimension
        metric_clean = "cosine" if metric.lower() in ("cosine", "cos") else ("l2" if metric.lower() in ("l2", "euclidean") else "ip")

        collection_metadata = {
            "dimension": dim,
            "hnsw:space": metric_clean,
            "description": "Knovera RAG Grounded Document Chunks Collection",
            **(metadata or {})
        }

        self.active_collection = self.client.get_or_create_collection(
            name=name,
            metadata=collection_metadata
        )
        logger.info(f"Connected to collection '{name}' (Dimension: {dim}, Metric: {metric_clean})")
        return self.active_collection

    def validate_record_schema(self, record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validates that a record satisfies the required schema fields and dimension constraints.
        """
        if not isinstance(record, dict):
            return False, "Record must be a dictionary."

        # Check required fields
        if "id" not in record or not record["id"]:
            return False, "Record must contain a non-empty 'id' string."
        if "vector" not in record and "embedding" not in record:
            return False, "Record must contain 'vector' or 'embedding' list of floats."
        if "text" not in record:
            return False, "Record must contain 'text' string."

        vector = record.get("vector") if "vector" in record else record.get("embedding")
        if not isinstance(vector, (list, tuple)) or len(vector) == 0:
            return False, "Vector must be a non-empty list of floats."

        if len(vector) != self.expected_dimension:
            return False, f"Vector dimension mismatch: got {len(vector)}, expected {self.expected_dimension}."

        return True, None

    def insert_record(
        self,
        id: str,
        vector: List[float],
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Inserts or updates a single grounded record into the vector database.
        """
        record = {
            "id": id,
            "vector": vector,
            "text": text,
            "metadata": metadata or {}
        }
        return self.upsert_records([record], collection_name=collection_name)

    def upsert_records(
        self,
        records: List[Dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Batch upserts structured records (vector + text + metadata) into the collection.
        """
        if not records:
            return False

        col = self.create_or_get_collection(collection_name) if collection_name else (self.active_collection or self.create_or_get_collection())

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for r in records:
            is_valid, err = self.validate_record_schema(r)
            if not is_valid:
                raise ValueError(f"Invalid record schema: {err}")

            vec = r.get("vector") if "vector" in r else r.get("embedding")
            ids.append(str(r["id"]))
            embeddings.append(list(vec))
            documents.append(str(r.get("text", "")))
            
            # ChromaDB requires primitive metadata values (str, int, float, bool)
            meta = dict(r.get("metadata", {}))
            sanitized_meta = {
                k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                for k, v in meta.items()
            }
            metadatas.append(sanitized_meta)

        col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Successfully upserted {len(records)} records into collection '{col.name}'")
        return True

    def get_record(
        self,
        id: str,
        collection_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Reads back a stored record by its ID, returning its ID, vector, source text, and metadata.
        """
        col = self.create_or_get_collection(collection_name) if collection_name else (self.active_collection or self.create_or_get_collection())
        
        result = col.get(
            ids=[id],
            include=["embeddings", "documents", "metadatas"]
        )

        if not result["ids"] or len(result["ids"]) == 0:
            return None

        emb = result["embeddings"][0] if result["embeddings"] is not None and len(result["embeddings"]) > 0 else []
        doc = result["documents"][0] if result["documents"] is not None and len(result["documents"]) > 0 else ""
        meta = result["metadatas"][0] if result["metadatas"] is not None and len(result["metadatas"]) > 0 else {}

        return {
            "id": result["ids"][0],
            "vector": list(emb) if emb is not None else [],
            "vector_length": len(emb) if emb is not None else 0,
            "text": doc,
            "metadata": meta
        }

    def count(self, collection_name: Optional[str] = None) -> int:
        """Returns the total number of items stored in the collection."""
        col = self.create_or_get_collection(collection_name) if collection_name else (self.active_collection or self.create_or_get_collection())
        return col.count()

    def query_similar(
        self,
        query_vector: List[float],
        top_k: int = 3,
        where: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes vector nearest-neighbor similarity search with optional metadata filtering.
        """
        col = self.create_or_get_collection(collection_name) if collection_name else (self.active_collection or self.create_or_get_collection())
        
        query_params: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["embeddings", "documents", "metadatas", "distances"]
        }
        if where:
            query_params["where"] = where

        results = col.query(**query_params)
        
        formatted = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for idx in range(len(results["ids"][0])):
                # Distance for cosine in chroma is (1 - cosine_sim)
                dist = results["distances"][0][idx] if results.get("distances") else 0.0
                score = round(1.0 - dist, 4)
                formatted.append({
                    "id": results["ids"][0][idx],
                    "text": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "distance": round(dist, 4),
                    "similarity": score,
                    "rank": idx + 1
                })
        return formatted

    def delete_collection(self, name: Optional[str] = None):
        """Deletes a collection by name."""
        target_name = name or (self.active_collection.name if self.active_collection else self.DEFAULT_COLLECTION_NAME)
        try:
            self.client.delete_collection(target_name)
            if self.active_collection and self.active_collection.name == target_name:
                self.active_collection = None
            logger.info(f"Deleted collection '{target_name}'")
        except Exception as e:
            logger.warning(f"Failed to delete collection '{target_name}': {e}")

    def create_collection(self, name: str, **kwargs):
        """Alias for create_or_get_collection."""
        return self.create_or_get_collection(name=name, **kwargs)

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        try:
            cols = self.client.list_collections()
            col_names = [c.name for c in cols]
            return name in col_names
        except Exception:
            return False

    def upsert(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Convenience method to upsert arrays of ids, embeddings, documents, and metadatas."""
        meta_list = metadatas if metadatas else [{} for _ in ids]
        records = [
            {"id": cid, "vector": emb, "text": doc, "metadata": meta}
            for cid, emb, doc, meta in zip(ids, embeddings, documents, meta_list)
        ]
        return self.upsert_records(records, collection_name=collection_name)

    def query(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 3,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Alias for query_similar."""
        return self.query_similar(query_vector=query_vector, top_k=top_k, where=where, collection_name=collection_name)
