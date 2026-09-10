"""
src/services/vector_store.py

Enterprise Vector Database Management & Collection Design Engine.
Designed for Knovera RAG Assistant using MongoDB & MongoDB Atlas Vector Search.

Key Capabilities:
1. Reachable Vector Store Connection: Connects to MongoDB cluster or in-memory instance via environment configuration.
2. Dimension-Guarded Collection Design: Configures collections with precise vector dimensions (1536 for text-embedding-3-small) and cosine distance metrics.
3. Grounded Record Schema: Binds high-dimensional vector embeddings with source text and rich hierarchical metadata in MongoDB.
4. CRUD & Readback Verification: Provides atomic upserts, exact ID lookups, metadata filtering, and nearest-neighbor search.
5. Hybrid Vector Search: Utilizes MongoDB Atlas $vectorSearch when an index is available, with high-speed NumPy cosine similarity fallback.
"""

import os
import re
import sys
import logging
import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from dotenv import load_dotenv
import numpy as np

try:
    import pymongo
    from pymongo import MongoClient, ReplaceOne
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False

from src.services.embedding_generator import EmbeddingGenerator

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


def _mask_mongo_url(url: str) -> str:
    """Mask credentials in MongoDB connection string for safe logging."""
    if not url:
        return "Not Configured"
    return re.sub(r'(mongodb(?:\+srv)?://[^:]+:)([^@]+)(@.+)', r'\1***\3', url)


def _build_mongo_filter(where: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Translates Chroma-style metadata where filters to MongoDB field queries."""
    if not where:
        return {}
    query_filter: Dict[str, Any] = {}
    for k, v in where.items():
        if k in ("_id", "id", "text"):
            query_filter[k] = v
        elif k.startswith("metadata."):
            query_filter[k] = v
        else:
            query_filter[f"metadata.{k}"] = v
    return query_filter


class MongoCollectionWrapper:
    """
    Wrapper providing Chroma-compatible API methods over a MongoDB collection.
    """

    def __init__(
        self,
        name: str,
        collection: Any,
        vector_db: "VectorDatabase",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self._col = collection
        self.vector_db = vector_db
        self.metadata = metadata or {}

    def count(self) -> int:
        """Returns total document count in this collection."""
        return self._col.count_documents({})

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """ChromaDB-compatible get() returning ids, embeddings, documents, and metadatas."""
        filter_query: Dict[str, Any] = {}
        if ids:
            filter_query["_id"] = {"$in": [str(i) for i in ids]}
        if where:
            filter_query.update(_build_mongo_filter(where))

        docs = list(self._col.find(filter_query))
        ret_ids = [d.get("id") or str(d.get("_id")) for d in docs]
        ret_embeddings = [d.get("vector", []) for d in docs]
        ret_documents = [d.get("text", "") for d in docs]
        ret_metadatas = [d.get("metadata", {}) for d in docs]

        return {
            "ids": ret_ids,
            "embeddings": ret_embeddings,
            "documents": ret_documents,
            "metadatas": ret_metadatas
        }

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None
    ):
        """ChromaDB-compatible delete() by IDs or metadata conditions."""
        filter_query: Dict[str, Any] = {}
        if ids:
            filter_query["_id"] = {"$in": [str(i) for i in ids]}
        if where:
            filter_query.update(_build_mongo_filter(where))

        if filter_query:
            result = self._col.delete_many(filter_query)
            logger.info(f"Deleted {result.deleted_count} documents from MongoDB collection '{self.name}'")

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """ChromaDB-compatible upsert() method."""
        meta_list = metadatas if metadatas else [{} for _ in ids]
        records = [
            {"id": cid, "vector": emb, "text": doc, "metadata": meta}
            for cid, emb, doc, meta in zip(ids, embeddings, documents, meta_list)
        ]
        return self.vector_db.upsert_records(records, collection_name=self.name)

    def query(self, **kwargs) -> Dict[str, Any]:
        """ChromaDB-compatible query method."""
        query_embeddings = kwargs.get("query_embeddings", [[]])
        q_vec = query_embeddings[0] if query_embeddings else []
        top_k = kwargs.get("n_results", 3)
        where = kwargs.get("where")

        results = self.vector_db.query_similar(
            query_vector=q_vec,
            top_k=top_k,
            where=where,
            collection_name=self.name
        )

        ids = [r["id"] for r in results]
        docs = [r["text"] for r in results]
        metas = [r["metadata"] for r in results]
        distances = [r["distance"] for r in results]

        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [distances]
        }

    # Proxy common pymongo collection methods
    def find(self, *args, **kwargs):
        return self._col.find(*args, **kwargs)

    def find_one(self, *args, **kwargs):
        return self._col.find_one(*args, **kwargs)

    def replace_one(self, *args, **kwargs):
        return self._col.replace_one(*args, **kwargs)

    def delete_many(self, *args, **kwargs):
        return self._col.delete_many(*args, **kwargs)

    def count_documents(self, *args, **kwargs):
        return self._col.count_documents(*args, **kwargs)

    def bulk_write(self, *args, **kwargs):
        return self._col.bulk_write(*args, **kwargs)


class InMemoryCollection:
    """Mock PyMongo collection used for offline/in-memory unit testing."""

    def __init__(self, name: str):
        self.name = name
        self._docs: Dict[str, Dict[str, Any]] = {}

    def count_documents(self, filter_query: Optional[Dict[str, Any]] = None) -> int:
        if not filter_query:
            return len(self._docs)
        return len(list(self.find(filter_query)))

    def find_one(self, filter_query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        results = list(self.find(filter_query))
        return results[0] if results else None

    def find(
        self,
        filter_query: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, int]] = None
    ) -> List[Dict[str, Any]]:
        results = []
        filter_query = filter_query or {}

        def match_single_cond(doc: Dict[str, Any], k: str, v: Any) -> bool:
            if k == "$or" and isinstance(v, list):
                return any(all(match_single_cond(doc, sub_k, sub_v) for sub_k, sub_v in sub_dict.items()) for sub_dict in v)
            if k == "_id" and isinstance(v, dict) and "$in" in v:
                return (doc.get("_id") in v["$in"]) or (doc.get("id") in v["$in"])
            if k.startswith("metadata."):
                sub_k = k[len("metadata."):]
                return doc.get("metadata", {}).get(sub_k) == v
            return doc.get(k) == v

        for d in self._docs.values():
            if all(match_single_cond(d, k, v) for k, v in filter_query.items()):
                results.append(dict(d))
        return results

    def replace_one(self, filter_query: Dict[str, Any], doc: Dict[str, Any], upsert: bool = False):
        key = str(doc.get("_id") or doc.get("id") or filter_query.get("_id"))
        self._docs[key] = dict(doc)

    def delete_many(self, filter_query: Dict[str, Any]):
        class DeleteResult:
            def __init__(self, count: int):
                self.deleted_count = count

        matching = self.find(filter_query)
        for m in matching:
            key = str(m.get("_id") or m.get("id"))
            self._docs.pop(key, None)
        return DeleteResult(len(matching))

    def bulk_write(self, operations: List[Any]):
        for op in operations:
            if hasattr(op, "_filter") and hasattr(op, "_doc"):
                key = str(op._doc.get("_id") or op._doc.get("id"))
                self._docs[key] = dict(op._doc)
            elif isinstance(op, ReplaceOne):
                key = str(op._doc.get("_id") or op._doc.get("id"))
                self._docs[key] = dict(op._doc)

    def drop(self):
        self._docs.clear()


class VectorDatabase:
    """
    Manages Vector Database connections, collection lifecycles, and grounded record operations
    using MongoDB (Atlas Vector Search & exact Cosine Similarity).
    """

    DEFAULT_COLLECTION_NAME = "rag_chunks"
    DEFAULT_DIMENSION = 1536
    DEFAULT_METRIC = "cosine"
    DEFAULT_DB_NAME = "Knovera"

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        in_memory: bool = False,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        mongodb_url: Optional[str] = None,
        db_name: Optional[str] = None
    ):
        """
        Initialize the MongoDB Vector Database connection.
        
        Args:
            persist_dir: Optional URL or directory (fallback).
            in_memory: If True, uses ephemeral in-memory storage (ideal for unit testing).
            embedding_generator: Optional EmbeddingGenerator for dimension resolution.
            mongodb_url: Optional explicit MongoDB connection string.
            db_name: Optional target MongoDB database name.
        """
        load_dotenv()
        self.in_memory = in_memory or (os.getenv("VECTOR_DB_IN_MEMORY", "false").lower() == "true")
        self.generator = embedding_generator or EmbeddingGenerator()
        self.embedding_generator = self.generator
        self.expected_dimension = self.generator.dimension or self.DEFAULT_DIMENSION
        self.db_name = db_name or os.getenv("MONGODB_DB_NAME", self.DEFAULT_DB_NAME)

        # Resolve MongoDB URL
        resolved_url = (
            mongodb_url
            or os.getenv("MONGODB_URL")
            or os.getenv("Mongodb_url")
            or os.getenv("MONGODB_URI")
            or os.getenv("VECTOR_DB_URL")
            or persist_dir
            or ""
        ).strip()

        self.mongodb_url = resolved_url
        self.persist_dir = persist_dir or self.mongodb_url

        self.client: Any = None
        self.db: Any = None
        self.active_collection: Optional[MongoCollectionWrapper] = None
        self._collections: Dict[str, MongoCollectionWrapper] = {}
        self._in_memory_collections: Dict[str, InMemoryCollection] = {}

        if self.in_memory or not HAS_PYMONGO or not (self.mongodb_url.startswith("mongodb://") or self.mongodb_url.startswith("mongodb+srv://")):
            if not self.in_memory and not (self.mongodb_url.startswith("mongodb://") or self.mongodb_url.startswith("mongodb+srv://")):
                logger.warning("No valid MongoDB URL configured. Operating in ephemeral in-memory mode.")
            self.in_memory = True
            self.storage_mode = "Ephemeral (In-Memory)"
            logger.info("Initialized ephemeral in-memory Vector Database.")
        else:
            try:
                self.client = MongoClient(self.mongodb_url, serverSelectionTimeoutMS=5000)
                try:
                    self.db = self.client.get_default_database()
                except Exception:
                    self.db = None
                if self.db is None:
                    self.db = self.client[self.db_name]

                self.storage_mode = f"MongoDB Cluster ({_mask_mongo_url(self.mongodb_url)}) [DB: {self.db.name}]"
                logger.info(f"Initialized MongoDB Vector Database at {self.storage_mode}")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB cluster: {e}. Falling back to in-memory mode.")
                self.in_memory = True
                self.storage_mode = "Ephemeral (In-Memory Fallback)"

    def is_reachable(self) -> bool:
        """Confirms whether the Vector Database is connected and reachable."""
        if self.in_memory:
            return True
        if self.client is None:
            return False
        try:
            self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.warning(f"MongoDB ping failed: {e}")
            try:
                if self.db is not None:
                    self.db.command('ping')
                    return True
            except Exception:
                pass
            return False

    def create_or_get_collection(
        self,
        name: str = DEFAULT_COLLECTION_NAME,
        dimension: Optional[int] = None,
        metric: str = DEFAULT_METRIC,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MongoCollectionWrapper:
        """
        Creates or accesses a MongoDB vector collection with schema metadata.
        """
        dim = dimension or self.expected_dimension
        collection_metadata = {
            "dimension": dim,
            "metric": metric,
            "description": "Knovera RAG Grounded Document Chunks Collection",
            **(metadata or {})
        }

        if self.in_memory:
            if name not in self._in_memory_collections:
                self._in_memory_collections[name] = InMemoryCollection(name)
            wrapper = MongoCollectionWrapper(
                name=name,
                collection=self._in_memory_collections[name],
                vector_db=self,
                metadata=collection_metadata
            )
        else:
            mongo_col = self.db[name]
            # Create indexing if connection is healthy
            try:
                mongo_col.create_index("id")
                mongo_col.create_index("metadata.source")
            except Exception:
                pass

            wrapper = MongoCollectionWrapper(
                name=name,
                collection=mongo_col,
                vector_db=self,
                metadata=collection_metadata
            )

        self.active_collection = wrapper
        self._collections[name] = wrapper
        logger.info(f"Connected to MongoDB collection '{name}' (Dimension: {dim}, Metric: {metric})")
        return wrapper

    def validate_record_schema(self, record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validates that a record satisfies the required schema fields and dimension constraints.
        """
        if not isinstance(record, dict):
            return False, "Record must be a dictionary."

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
        """Inserts or updates a single grounded record into the vector database."""
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
        Batch upserts structured records (vector + text + metadata) into MongoDB collection.
        """
        if not records:
            return False

        col = self.create_or_get_collection(collection_name) if collection_name else (self.active_collection or self.create_or_get_collection())

        prepared_operations = []
        for r in records:
            is_valid, err = self.validate_record_schema(r)
            if not is_valid:
                raise ValueError(f"Invalid record schema: {err}")

            vec = r.get("vector") if "vector" in r else r.get("embedding")
            doc = {
                "_id": str(r["id"]),
                "id": str(r["id"]),
                "vector": [float(x) for x in vec],
                "text": str(r.get("text", "")),
                "metadata": dict(r.get("metadata", {})),
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            if HAS_PYMONGO and not self.in_memory:
                prepared_operations.append(ReplaceOne({"_id": doc["_id"]}, doc, upsert=True))
            else:
                col._col.replace_one({"_id": doc["_id"]}, doc, upsert=True)

        if HAS_PYMONGO and not self.in_memory and prepared_operations:
            col._col.bulk_write(prepared_operations)

        logger.info(f"Successfully upserted {len(records)} records into MongoDB collection '{col.name}'")
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
        doc = col.find_one({"$or": [{"_id": str(id)}, {"id": str(id)}]})
        if not doc:
            return None

        vec = doc.get("vector", [])
        return {
            "id": doc.get("id", str(doc.get("_id"))),
            "vector": list(vec),
            "vector_length": len(vec),
            "text": doc.get("text", ""),
            "metadata": doc.get("metadata", {})
        }

    def count(self, collection_name: Optional[str] = None) -> int:
        """Returns total number of items stored in the collection."""
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
        Attempts MongoDB Atlas $vectorSearch first, falling back to exact NumPy cosine similarity.
        """
        col = self.create_or_get_collection(collection_name) if collection_name else (self.active_collection or self.create_or_get_collection())
        mongo_filter = _build_mongo_filter(where)

        # 1. Attempt Atlas $vectorSearch if running against live MongoDB Atlas
        if not self.in_memory and self.db is not None:
            try:
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "path": "vector",
                            "queryVector": query_vector,
                            "numCandidates": max(top_k * 10, 20),
                            "limit": top_k,
                            **( {"filter": mongo_filter} if mongo_filter else {} )
                        }
                    },
                    {
                        "$project": {
                            "_id": 1,
                            "id": 1,
                            "text": 1,
                            "metadata": 1,
                            "score": {"$meta": "vectorSearchScore"}
                        }
                    }
                ]
                cursor = col._col.aggregate(pipeline)
                results = list(cursor)
                if results:
                    formatted = []
                    for idx, doc in enumerate(results):
                        sim = float(doc.get("score", 0.0))
                        formatted.append({
                            "id": str(doc.get("id") or doc.get("_id")),
                            "text": doc.get("text", ""),
                            "metadata": doc.get("metadata", {}),
                            "distance": round(max(0.0, 1.0 - sim), 4),
                            "similarity": round(sim, 4),
                            "rank": idx + 1
                        })
                    return formatted
            except Exception as e:
                # $vectorSearch index not configured yet or not supported on this tier; fallback to exact cosine
                logger.debug(f"Atlas $vectorSearch not available ({e}); using exact cosine similarity fallback.")

        # 2. Exact Cosine Similarity Retrieval over MongoDB Documents
        docs = list(col.find(mongo_filter))
        if not docs:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm == 0.0:
            q_norm = 1e-10

        scored_docs = []
        for d in docs:
            v_raw = d.get("vector")
            if not v_raw:
                continue
            v = np.array(v_raw, dtype=np.float32)
            v_norm = float(np.linalg.norm(v))
            if v_norm == 0.0:
                sim = 0.0
            else:
                sim = float(np.dot(v, q_vec) / (v_norm * q_norm))
            scored_docs.append((sim, d))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = scored_docs[:top_k]

        formatted = []
        for idx, (sim, doc) in enumerate(top_docs):
            dist = max(0.0, 1.0 - sim)
            formatted.append({
                "id": str(doc.get("id") or doc.get("_id")),
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
                "distance": round(dist, 4),
                "similarity": round(sim, 4),
                "rank": idx + 1
            })

        return formatted

    def delete_collection(self, name: Optional[str] = None):
        """Deletes/drops a collection by name."""
        target_name = name or (self.active_collection.name if self.active_collection else self.DEFAULT_COLLECTION_NAME)
        try:
            if self.in_memory:
                if target_name in self._in_memory_collections:
                    self._in_memory_collections[target_name].drop()
                    self._in_memory_collections.pop(target_name, None)
            else:
                if self.db is not None:
                    self.db.drop_collection(target_name)

            if self.active_collection and self.active_collection.name == target_name:
                self.active_collection = None
            self._collections.pop(target_name, None)
            logger.info(f"Deleted MongoDB collection '{target_name}'")
        except Exception as e:
            logger.warning(f"Failed to delete collection '{target_name}': {e}")

    def create_collection(self, name: str, **kwargs) -> MongoCollectionWrapper:
        """Alias for create_or_get_collection."""
        return self.create_or_get_collection(name=name, **kwargs)

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        if self.in_memory:
            return name in self._in_memory_collections
        try:
            if self.db is None:
                return False
            return name in self.db.list_collection_names()
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
