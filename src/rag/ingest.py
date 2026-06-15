from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

SEPARATORS = ["\n제", "\n조", "\n항", "\n\n", "\n", " ", ""]

splitter = RecursiveCharacterTextSplitter(
    separators=SEPARATORS,
    chunk_size=800,
    chunk_overlap=200,
)


def ingest(title: str, department: str, content: str) -> list[Document]:
    """텍스트를 청킹하고 메타데이터를 부착한 Document 리스트를 반환한다."""
    chunks = splitter.create_documents(
        texts=[content],
        metadatas=[{"title": title, "department": department}],
    )
    return chunks
