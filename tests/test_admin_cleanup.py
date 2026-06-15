import pytest
from httpx import AsyncClient, ASGITransport
from langchain_core.language_models.fake_chat_models import FakeListChatModel
import src.rag.api.admin as admin_module


ADMIN_TOKEN = "testtoken"
FAKE_LLM = FakeListChatModel(responses=["| 구분 | 내용 |\n|---|---|\n| A | B |"] * 10)


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    monkeypatch.setattr(admin_module, "get_llm", lambda: FAKE_LLM)


@pytest.mark.asyncio
async def test_cleanup_requires_auth():
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post("/admin/cleanup", json={"content": "| a | b |"})
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_cleanup_returns_cleaned_content():
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post(
            "/admin/cleanup",
            json={"content": "| 구분 | 내용 |\n| 복잡한 병합 셀 내용 |"},
            auth=("admin", ADMIN_TOKEN),
        )
    assert res.status_code == 200
    data = res.json()
    assert "content" in data
    assert len(data["content"]) > 0


@pytest.mark.asyncio
async def test_cleanup_empty_content_rejected():
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post(
            "/admin/cleanup",
            json={"content": "   "},
            auth=("admin", ADMIN_TOKEN),
        )
    assert res.status_code == 400
