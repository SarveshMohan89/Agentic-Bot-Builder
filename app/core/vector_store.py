import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

settings = get_settings()


class VectorStoreManager:

    def __init__(self):
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        print(f"Loading the embedding model: {settings.embedding_model}")
        self._embedder = SentenceTransformer(settings.embedding_model)
        print("Embedding Model Loaded")

    def _collection_name(self, bot_id: str) -> str:
        return f"{settings.chroma_collection_prefix}{bot_id.replace('-', '_')}"

    def _get_or_create_collection(self, bot_id: str):
        return self._client.get_or_create_collection(
            name=self._collection_name(bot_id),
            metadata={"hnsw:space": "cosine"},
        )

    def delete_collection(self, bot_id: str):
        try:
            self._client.delete_collection(self._collection_name(bot_id))
        except Exception:
            pass

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, query: str) -> List[float]:
        return self._embedder.encode([query], convert_to_numpy=True)[0].tolist()

    def add_documents(
        self,
        bot_id: str,
        chunks: List[str],
        metadatas: List[Dict[str, Any]],
        source_id: str,
    ) -> int:
        if not chunks:
            return 0
        collection = self._get_or_create_collection(bot_id)
        ids = [f"{source_id}_{i}" for i in range(len(chunks))]
        for meta in metadatas:
            meta["source_id"] = source_id
        embeddings = self.embed(chunks)
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def delete_source(self, bot_id: str, source_id: str):
        collection = self._get_or_create_collection(bot_id)
        collection.delete(where={"source_id": source_id})

    def similarity_search(
        self,
        bot_id: str,
        query: str,
        top_k: int = None,
        score_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        top_k = top_k or settings.top_k_results
        score_threshold = score_threshold or settings.similarity_threshold
        collection = self._get_or_create_collection(bot_id)
        if collection.count() == 0:
            return []
        query_embedding = self.embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        for doc, meta, dist in zip(documents, metadatas, distances):
            score = 1.0 - (dist / 2.0)
            if score >= score_threshold:
                docs.append({
                    "text": doc,
                    "metadata": meta,
                    "score": round(score, 4),
                })
        docs.sort(key=lambda x: x["score"], reverse=True)
        return docs

    def get_collection_stats(self, bot_id: str) -> Dict[str, Any]:
        collection = self._get_or_create_collection(bot_id)
        return {
            "bot_id": bot_id,
            "collection_name": self._collection_name(bot_id),
            "total_chunks": collection.count(),
        }


_vector_store_instance: Optional[VectorStoreManager] = None


def get_vector_store() -> VectorStoreManager:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreManager()
    return _vector_store_instance