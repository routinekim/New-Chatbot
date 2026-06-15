import pytest
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.embeddings import Embeddings
from src.rag.store import DocumentStore
from src.rag.ingest import ingest
from src.rag import chain as chain_module


class FakeEmbeddings(Embeddings):
    DIM = 128
    def embed_documents(self, texts):
        return [[0.5] * self.DIM for _ in texts]
    def embed_query(self, text):
        return [0.5] * self.DIM


SAMPLE_CONTENT = (
    "제1조 (목적) 이 규정은 한세대학교 학생의 학사 운영에 관한 사항을 규정함을 목적으로 한다.\n\n" * 30
)

fake_llm = FakeListChatModel(responses=["제1조에 따르면 이 규정은 학사 운영에 관한 사항을 목적으로 합니다."])


@pytest.fixture
def populated_store(tmp_path, monkeypatch):
    store = DocumentStore(persist_dir=str(tmp_path), embeddings=FakeEmbeddings())
    chunks = ingest("학칙", "교무처", SAMPLE_CONTENT)
    store.add_documents(chunks, doc_id="doc-001")
    monkeypatch.setattr(chain_module, "get_store", lambda: store)
    return store


@pytest.mark.asyncio
async def test_run_rag_returns_answer(populated_store):
    result = await chain_module.run_rag("목적이 뭔가요?", llm=fake_llm)
    assert result["answer"]
    assert isinstance(result["sources"], list)
    assert len(result["sources"]) > 0


@pytest.mark.asyncio
async def test_run_rag_sources_have_metadata(populated_store):
    result = await chain_module.run_rag("목적이 뭔가요?", llm=fake_llm)
    src = result["sources"][0]
    assert src["title"] == "학칙"
    assert src["department"] == "교무처"
    assert "preview" in src


@pytest.mark.asyncio
async def test_run_rag_empty_store_returns_no_documents_message(tmp_path, monkeypatch):
    empty_store = DocumentStore(persist_dir=str(tmp_path), embeddings=FakeEmbeddings())
    monkeypatch.setattr(chain_module, "get_store", lambda: empty_store)
    result = await chain_module.run_rag("휴학 기한?", llm=fake_llm)
    assert "등록된 문서가 없습니다" in result["answer"]
    assert result["sources"] == []
