from app.rag.chunker import TextChunker


def test_empty_text_returns_no_chunks():
    assert TextChunker().split("") == []
    assert TextChunker().split("   \n  ") == []


def test_plain_text_is_split_like_fixed_size_chunks():
    chunker = TextChunker(chunk_size=3, overlap=0)

    result = [chunk.text for chunk in chunker.split("abcdefg")]

    assert result == ["abc", "def", "g"]


def test_overlap_is_clamped_to_chunk_size():
    assert TextChunker(chunk_size=3, overlap=80).overlap == 0
    assert TextChunker(chunk_size=100, overlap=80).overlap == 25


def test_chunks_never_exceed_chunk_size():
    text = "差旅报销标准是一线城市每天500元，二线城市每天350元。" * 20

    chunks = TextChunker(chunk_size=120, overlap=30).split(text)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 120 for chunk in chunks)


def test_chunk_indexes_are_contiguous():
    text = "\n\n".join(f"段落内容第{index}段。" * 40 for index in range(6))

    chunks = TextChunker(chunk_size=100, overlap=20).split(text)

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_paragraphs_are_kept_inside_a_chunk():
    text = (
        "第一段介绍公司差旅报销的整体原则。\n\n"
        "第二段介绍具体的住宿和交通标准。"
    )

    chunks = TextChunker(chunk_size=100, overlap=0).split(text)

    assert len(chunks) == 1
    assert "第一段介绍公司差旅报销的整体原则。" in chunks[0].text
    assert "第二段介绍具体的住宿和交通标准。" in chunks[0].text


def test_no_paragraph_is_lost_when_text_is_split():
    paragraphs = [
        "差旅报销标准是一线城市每天500元。" * 10,
        "市内交通费每天上限是100元。" * 10,
        "超标部分需要部门经理审批。" * 10,
    ]
    text = "\n\n".join(paragraphs)
    sentences = [
        sentence
        for paragraph in paragraphs
        for sentence in paragraph.split("。")
        if sentence
    ]

    chunks = TextChunker(chunk_size=150, overlap=30).split(text)

    assert len(chunks) > 1

    for sentence in sentences:
        assert any(sentence in chunk.text for chunk in chunks)


def test_consecutive_chunks_share_overlap():
    text = "ABCDEFGHIJ" * 30

    chunks = TextChunker(chunk_size=60, overlap=15).split(text)

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:]):
        assert current.text.startswith(previous.text[-15:])


def test_long_sentence_without_boundaries_is_split():
    text = "A1B2C3D4E5" * 30

    chunks = TextChunker(chunk_size=100, overlap=20).split(text)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)
    assert text.startswith(chunks[0].text)
    assert text.endswith(chunks[-1].text)


def test_crlf_is_normalised():
    text = "第一段。\r\n\r\n第二段。"

    chunks = TextChunker(chunk_size=100, overlap=0).split(text)

    assert "\r" not in chunks[0].text


def test_invalid_chunk_size_is_rejected():
    try:
        TextChunker(chunk_size=0)
    except ValueError as exc:
        assert "chunk_size" in str(exc)
    else:
        raise AssertionError("expected ValueError")
