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

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def get_llm() -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )


def _docs_to_sources(docs: list[Document]) -> list[dict]:
    return [
        {
            "title": doc.metadata.get("title", ""),
            "department": doc.metadata.get("department", ""),
            "preview": doc.page_content[:120].replace("\n", " "),
        }
        for doc in docs
    ]


async def run_rag(question: str, llm: BaseChatModel | None = None) -> dict:
    retriever = get_store().get_retriever()
    if retriever is None:
        return {"answer": "등록된 문서가 없습니다.", "sources": []}

    docs: list[Document] = retriever.invoke(question)
    if not docs:
        return {"answer": "해당 규정을 찾지 못했습니다.", "sources": []}

    context = "\n\n".join(doc.page_content for doc in docs)
    chain = prompt | (llm or get_llm()) | StrOutputParser()
    answer = await chain.ainvoke({"context": context, "question": question})

    return {"answer": answer, "sources": _docs_to_sources(docs)}
