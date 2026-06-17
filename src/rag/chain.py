from __future__ import annotations

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from src.rag.store import get_store

SYSTEM_PROMPT = """당신은 한세대학교 규정 안내 챗봇입니다.
아래 문서 내용만을 근거로 답변하세요.
문서에 없는 내용은 반드시 "해당 규정을 찾지 못했습니다."라고 답하세요.
답변 시 관련 조항이나 항목을 명시하세요.

[문서 내용]
{context}"""

VOICE_SYSTEM_PROMPT = """당신은 한세대학교 학생을 돕는 친절한 AI 비서입니다.
아래 문서 내용만을 근거로 답변하세요.
문서에 없는 내용은 반드시 "해당 규정을 찾지 못했습니다."라고 답하세요.

답변 형식:
1. 질문 내용에 공감하거나 가볍게 반응하는 한 문장으로 시작하세요. (예: "장학금에 관심이 있으시군요!", "중요한 사항을 물어봐 주셨네요.")
2. 규정 내용을 친절하고 자연스럽게 설명하세요. 조항 번호는 유지하세요.
3. 마지막에 짧은 응원 또는 안내 한 문장으로 마무리하세요.

[문서 내용]
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

voice_prompt = ChatPromptTemplate.from_messages([
    ("system", VOICE_SYSTEM_PROMPT),
    ("human", "{question}"),
])


def get_llm() -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )


def _scored_to_sources(scored: list[tuple[Document, float]], rel_threshold: float = 1.3, doc_type: str | None = None) -> list[dict]:
    """L2 거리 기준 상대 임계값 내의 고유 문서만 반환. 낮은 거리 = 높은 유사도."""
    if not scored:
        return []
    best = scored[0][1]
    seen: set[str] = set()
    sources = []
    for doc, score in scored:
        if score > best * rel_threshold:
            break
        if doc_type and doc.metadata.get("doc_type") != doc_type:
            continue
        key = doc.metadata.get("doc_id") or doc.metadata.get("title", "")
        if key in seen:
            continue
        seen.add(key)
        relevance = round(max(0, (1 - (score / best - 1) / (rel_threshold - 1))) * 100)
        sources.append({
            "title": doc.metadata.get("title", ""),
            "doc_type": doc.metadata.get("doc_type", ""),
            "preview": doc.page_content[:300].replace("\n", " "),
            "relevance": relevance,
        })
    return sources


async def run_rag(question: str, llm: BaseChatModel | None = None, voice: bool = False, doc_type: str | None = None) -> dict:
    store = get_store()
    retriever = store.get_retriever()
    if retriever is None:
        return {"answer": "등록된 문서가 없습니다.", "sources": []}

    docs: list[Document] = retriever.invoke(question)

    if doc_type:
        docs = [d for d in docs if d.metadata.get("doc_type") == doc_type]

    if not docs:
        return {"answer": "해당 규정을 찾지 못했습니다.", "sources": []}

    context = "\n\n".join(doc.page_content for doc in docs)
    selected_prompt = voice_prompt if voice else prompt
    chain = selected_prompt | (llm or get_llm()) | StrOutputParser()
    answer = await chain.ainvoke({"context": context, "question": question})

    if "찾지 못했습니다" in answer:
        return {"answer": answer, "sources": []}

    scored = store.search_with_l2(question, k=10)
    sources = _scored_to_sources(scored, rel_threshold=1.3, doc_type=doc_type)
    return {"answer": answer, "sources": sources}
