from dataclasses import dataclass
from pathlib import Path

import tiktoken
from pypdf import PdfReader

from app.core.errors import AppError


@dataclass(slots=True)
class PdfPage:
    page: int
    text: str


@dataclass(slots=True)
class ParsedPdf:
    title: str | None
    pages: list[PdfPage]


@dataclass(slots=True)
class TextChunk:
    page: int
    chunk_index: int
    text: str
    token_count: int


def parse_pdf(path: Path, max_pages: int) -> ParsedPdf:
    try:
        reader = PdfReader(path)
    except Exception as exc:
        raise AppError("无法解析 PDF 文件。", code="invalid_pdf") from exc
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise AppError("暂不支持加密 PDF。", code="encrypted_pdf") from exc
    if len(reader.pages) > max_pages:
        raise AppError(f"PDF 超过 {max_pages} 页限制。", code="pdf_page_limit_exceeded")
    pages = [
        PdfPage(page=index, text=(page.extract_text() or "").strip())
        for index, page in enumerate(reader.pages, start=1)
    ]
    if len("".join(page.text for page in pages).strip()) < 50:
        raise AppError(
            "未提取到足够文本，该文件可能是扫描件；当前版本尚未启用 OCR。",
            code="ocr_required",
        )
    title = None
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip()
    return ParsedPdf(title=title, pages=pages)


def chunk_pages(
    pages: list[PdfPage],
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[TextChunk]:
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks: list[TextChunk] = []
    step = max(max_tokens - overlap_tokens, 1)
    for page in pages:
        token_ids = encoding.encode(page.text)
        for chunk_index, start in enumerate(range(0, len(token_ids), step)):
            current = token_ids[start : start + max_tokens]
            if not current:
                continue
            text = encoding.decode(current).strip()
            if text:
                chunks.append(
                    TextChunk(
                        page=page.page,
                        chunk_index=chunk_index,
                        text=text,
                        token_count=len(current),
                    )
                )
            if start + max_tokens >= len(token_ids):
                break
    return chunks
