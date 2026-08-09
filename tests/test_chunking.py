from app.rag.chunking import MAX_CHARS, TARGET_CHARS, chunk_markdown


def test_small_doc_single_chunk():
    chunks = chunk_markdown("Just a short paragraph.")
    assert chunks == ["Just a short paragraph."]


def test_empty():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  ") == []


def test_headings_carried_into_chunks():
    md = "# Title\n\n" + ("word " * 100).strip() + \
         "\n\n## Section Two\n\n" + ("data " * 100).strip()
    chunks = chunk_markdown(md)
    assert any(c.startswith("# Title") for c in chunks)
    assert any(c.startswith("## Section Two") for c in chunks)


def test_long_section_splits_with_overlap():
    paras = [f"Paragraph {i}. " + "content " * 60 for i in range(8)]
    md = "# Big\n\n" + "\n\n".join(paras)
    chunks = chunk_markdown(md)
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHARS + 200 for c in chunks)
    # one-paragraph overlap: last para of chunk N appears in chunk N+1
    assert "Paragraph 1." in chunks[0]
    joined_rest = "".join(chunks[1:])
    assert "Paragraph 1." in joined_rest or "Paragraph 2." in joined_rest


def test_giant_paragraph_hard_split():
    md = "x" * (TARGET_CHARS * 3)
    chunks = chunk_markdown(md)
    assert len(chunks) >= 3
    assert all(len(c) <= MAX_CHARS for c in chunks)
