from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

router = APIRouter()
templates = Jinja2Templates(directory="templates")


class QueryRequest(BaseModel):
    question: str
    voice: bool = False
    doc_type: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@router.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    from src.rag.chain import run_rag
    result = await run_rag(req.question, voice=req.voice, doc_type=req.doc_type)
    return QueryResponse(**result)
