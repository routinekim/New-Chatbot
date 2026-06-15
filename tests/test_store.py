import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from src.rag.store import DocumentStore
from src.rag.ingest import ingest


class FakeEmbeddings(Embeddings):
    """테스트용 — API 키 없이 고정 벡터 반환"""
    DIM = 128

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i % 10) / 10] * self.DIM for i, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [0.5] * self.DIM


@pytest.fixture
def store(tmp_path):
    return DocumentStore(persist_dir=str(tmp_path), embeddings=FakeEmbeddings())


SAMPLE_CONTENT = (
    "제1조 (목적) 이 규정은 한세대학교 학생의 학사 운영에 관한 사항을 규정함을 목적으로 한다.\n\n" * 30
)


def test_add_and_list(store):
    chunks = ingest("학칙", "교무처", SAMPLE_CONTENT)
    store.add_documents(chunks, doc_id="doc-001")

    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["title"] == "학칙"
    assert docs[0]["department"] == "교무처"
    assert docs[0]["chunk_count"] == len(chunks)


def test_get_retriever_returns_results(store):
    chunks = ingest("학칙", "교무처", SAMPLE_CONTENT)
    store.add_documents(chunks, doc_id="doc-001")

    retriever = store.get_retriever()
    assert retriever is not None

    results = retriever.invoke("목적")
    assert len(results) > 0
    assert all(isinstance(r, Document) for r in results)


def test_get_retriever_empty_store_returns_none(store):
    assert store.get_retriever() is None


def test_delete_document(store):
    chunks = ingest("학칙", "교무처", SAMPLE_CONTENT)
    store.add_documents(chunks, doc_id="doc-001")
    assert len(store.list_documents()) == 1

    store.delete_document("doc-001")
    assert store.list_documents() == []
