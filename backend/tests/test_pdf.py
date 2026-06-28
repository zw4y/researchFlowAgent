from app.rag.pdf import PdfPage, chunk_pages, parse_pdf


def test_parse_pdf_retains_page_numbers(sample_pdf):
    parsed = parse_pdf(sample_pdf, max_pages=10)
    assert parsed.title == "Transformer Study"
    assert [page.page for page in parsed.pages] == [1, 2]
    assert "Attention" in parsed.pages[0].text


def test_chunking_never_crosses_pages():
    pages = [
        PdfPage(page=1, text="alpha " * 100),
        PdfPage(page=2, text="beta " * 100),
    ]
    chunks = chunk_pages(pages, max_tokens=30, overlap_tokens=5)
    assert {item.page for item in chunks} == {1, 2}
    assert all(not ("alpha" in item.text and "beta" in item.text) for item in chunks)
