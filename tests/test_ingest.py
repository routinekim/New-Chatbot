from src.rag.ingest import ingest


def test_ingest_returns_chunks():
    # 800자 청크 크기를 초과하도록 충분히 긴 콘텐츠 사용
    content = "제1조 (목적) 이 규정은 한세대학교 학생의 학사 운영에 관한 사항을 규정함을 목적으로 한다.\n\n" * 30
    chunks = ingest("학칙", "교무처", content)
    assert len(chunks) > 1


def test_ingest_chunk_has_metadata():
    content = "제1조 (목적) 이 규정은 한세대학교 학생의 학사 운영에 관한 사항을 규정함을 목적으로 한다."
    chunks = ingest("학칙", "교무처", content)
    assert chunks[0].metadata["title"] == "학칙"
    assert chunks[0].metadata["department"] == "교무처"


def test_ingest_empty_content_returns_empty():
    chunks = ingest("빈 문서", "학생처", "")
    assert chunks == []
