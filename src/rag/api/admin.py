import os
import uuid
import secrets
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.rag.chain import get_llm

router = APIRouter()
templates = Jinja2Templates(directory="templates")
security = HTTPBasic()

DEPARTMENTS = ["학생처", "교무처", "입학처", "총무처", "도서관", "기타"]
DOC_TYPES = ["규정", "지침", "업무매뉴얼", "결재공문/기안문", "계획서/결과보고서", "회의록"]


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    admin_token = os.getenv("ADMIN_TOKEN", "")
    ok = secrets.compare_digest(credentials.password.encode(), admin_token.encode())
    if not ok:
        raise HTTPException(status_code=401, detail="인증 실패")
    return credentials.username


class IngestRequest(BaseModel):
    title: str
    content: str
    doc_type: str | None = None


class IngestResponse(BaseModel):
    chunk_count: int
    message: str


class CleanupRequest(BaseModel):
    content: str


class CleanupResponse(BaseModel):
    content: str


@router.get("/", response_class=HTMLResponse)
async def admin_page(request: Request, _: str = Depends(verify_admin)):
    return templates.TemplateResponse(
        request, "admin.html", {"doc_types": DOC_TYPES}
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(req: IngestRequest, _: str = Depends(verify_admin)):
    from src.rag.ingest import ingest
    from src.rag.store import get_store

    chunks = ingest(req.title, req.content, doc_type=req.doc_type)
    if not chunks:
        raise HTTPException(status_code=400, detail="내용이 비어 있습니다.")

    doc_id = str(uuid.uuid4())
    get_store().add_documents(chunks, doc_id)

    return IngestResponse(chunk_count=len(chunks), message="인덱싱 완료")


@router.post("/cleanup", response_model=CleanupResponse)
async def cleanup_table(req: CleanupRequest, _: str = Depends(verify_admin)):
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="내용이 비어 있습니다.")

    from langchain_core.messages import HumanMessage
    prompt = (
        "다음 마크다운 표를 정리해주세요. "
        "병합 셀이 있으면 내용을 명확히 구분하고 빈 셀은 위 셀 내용으로 채워주세요. "
        "마크다운만 반환하고 설명은 제외하세요.\n\n"
        + req.content
    )
    result = await get_llm().ainvoke([HumanMessage(content=prompt)])
    return CleanupResponse(content=result.content)


@router.get("/documents")
async def list_documents(_: str = Depends(verify_admin)):
    from src.rag.store import get_store
    return {"documents": get_store().list_documents()}


@router.get("/documents/{doc_id}/chunks")
async def get_chunks(doc_id: str, _: str = Depends(verify_admin)):
    from src.rag.store import get_store
    return {"chunks": get_store().get_chunks(doc_id)}


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, _: str = Depends(verify_admin)):
    from src.rag.store import get_store
    get_store().delete_document(doc_id)
    return {"ok": True}
