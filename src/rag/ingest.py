from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

SEPARATORS = ["\n제", "\n조", "\n항", "\n\n", "\n", " ", ""]

splitter = RecursiveCharacterTextSplitter(
    separators=SEPARATORS,
    chunk_size=800,
    chunk_overlap=200,
)


def ingest(title: str, content: str, doc_type: str | None = None) -> list[Document]:
    meta: dict = {"title": title}
    if doc_type:
        meta["doc_type"] = doc_type
    chunks = splitter.create_documents(texts=[content], metadatas=[meta])
    return chunks
