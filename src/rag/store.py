from __future__ import annotations

import uuid
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.rag.retriever import K_RETRIEVAL
from src.rag.embeddings import get_embeddings


class DocumentStore:
    def __init__(self, persist_dir: str = "./chroma_db", embeddings: Embeddings | None = None):
        self._chroma = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings or get_embeddings(),
        )

    def add_documents(self, chunks: list[Document], doc_id: str) -> None:
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
        self._chroma.add_documents(chunks)

    def _all_docs(self) -> list[Document]:
        result = self._chroma.get()
        return [
            Document(page_content=content, metadata=meta or {})
            for content, meta in zip(result["documents"], result["metadatas"])
        ]

    def get_retriever(self) -> EnsembleRetriever | None:
        all_docs = self._all_docs()
        if not all_docs:
            return None

        bm25 = BM25Retriever.from_documents(all_docs)
        bm25.k = K_RETRIEVAL

        vector = self._chroma.as_retriever(search_kwargs={"k": K_RETRIEVAL})

        return EnsembleRetriever(
            retrievers=[bm25, vector],
            weights=[0.5, 0.5],
        )

    def list_documents(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for doc in self._all_docs():
            doc_id = doc.metadata.get("doc_id", "")
            if doc_id not in seen:
                seen[doc_id] = {
                    "id": doc_id,
                    "title": doc.metadata.get("title", ""),
                    "doc_type": doc.metadata.get("doc_type", ""),
                    "chunk_count": 0,
                }
            seen[doc_id]["chunk_count"] += 1
        return list(seen.values())

    def get_chunks(self, doc_id: str) -> list[str]:
        return [
            doc.page_content
            for doc in self._all_docs()
            if doc.metadata.get("doc_id") == doc_id
        ]

    def search_with_l2(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        """(doc, l2_distance) 반환. 낮을수록 유사."""
        return self._chroma.similarity_search_with_score(query, k=k)

    def delete_document(self, doc_id: str) -> None:
        result = self._chroma.get()
        ids_to_delete = [
            id_ for id_, meta in zip(result["ids"], result["metadatas"])
            if (meta or {}).get("doc_id") == doc_id
        ]
        if ids_to_delete:
            self._chroma.delete(ids=ids_to_delete)


_store: DocumentStore | None = None


def get_store() -> DocumentStore:
    global _store
    if _store is None:
        _store = DocumentStore()
    return _store
