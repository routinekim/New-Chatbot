from typing import Protocol
from langchain_core.documents import Document

K_RETRIEVAL = 5


class DocumentRetriever(Protocol):
    def retrieve(self, query: str) -> list[Document]: ...
