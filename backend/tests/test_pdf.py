from app.rag.pdf import (
    PdfPage,
    _group_ocr_lines,
    _select_table_lines,
    _summarize_table_rows,
    chunk_pages,
    parse_pdf,
)


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

def test_ocr_lines_preserve_table_row_values():
    result = [
        ([[0, 0], [80, 0], [80, 20], [0, 20]], "ISFM(Ours)", 0.99),
        ([[100, 1], [140, 1], [140, 21], [100, 21]], "6.70", 0.99),
        ([[160, 0], [210, 0], [210, 20], [160, 20]], "11.42", 0.99),
        ([[0, 50], [80, 50], [80, 70], [0, 70]], "ordinary prose", 0.99),
        ([[100, 50], [140, 50], [140, 70], [100, 70]], "ignored", 0.2),
    ]

    lines = _group_ocr_lines(result, min_confidence=0.5)

    assert lines == ["ISFM(Ours) | 6.70 | 11.42", "ordinary prose"]
    assert _select_table_lines(lines) == ["ISFM(Ours) | 6.70 | 11.42"]

def test_table_summary_keeps_dataset_with_ours_row():
    lines = [
        "Dataset | Method | EN | SF | Avg.R",
        "MSRS | DATFuse | 6.48 | 10.93 | 4.57",
        "ISFM(Ours) | 6.70 | 11.42 | 1.14",
        "FMB | DATFuse | 6.32 | 10.85 | 7.43",
        "ISFM(Ours) | 6.76 | 13.65 | 1.71",
        "TABLE III",
        "Ours | 9.99 | 9.99 | 9.99",
    ]

    summary = _summarize_table_rows(lines)

    assert "MSRS | ISFM(Ours) | 6.70 | 11.42 | 1.14" in summary
    assert "FMB | ISFM(Ours) | 6.76 | 13.65 | 1.71" in summary
    assert "9.99" not in summary
