import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import tiktoken
from pypdf import PdfReader

from app.core.errors import AppError

logger = logging.getLogger(__name__)
_TABLE_HEADING = re.compile(r"\bTABLE\s+(?:[IVXLCDM]+|\d+)\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:\.\d+)?(?!\w)")
_TABLE_KEYWORDS = (
    "table",
    "dataset",
    "method",
    "quantitative",
    "configurations",
    "avg.r",
    "avg.rank",
    "isfm",
)


@dataclass(slots=True)
class PdfPage:
    page: int
    text: str


@dataclass(slots=True)
class ParsedPdf:
    title: str | None
    pages: list[PdfPage]
    table_ocr_pages: list[int] = field(default_factory=list)


@dataclass(slots=True)
class TextChunk:
    page: int
    chunk_index: int
    text: str
    token_count: int


def parse_pdf(
    path: Path,
    max_pages: int,
    *,
    table_ocr_enabled: bool = True,
    table_ocr_dpi: int = 300,
    table_ocr_min_confidence: float = 0.5,
) -> ParsedPdf:
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
            "未提取到足够文本，该文件可能是扫描件；当前版本尚未启用全文 OCR。",
            code="ocr_required",
        )

    table_ocr_pages: list[int] = []
    if table_ocr_enabled:
        candidates = [page.page for page in pages if _TABLE_HEADING.search(page.text)]
        if candidates:
            try:
                table_text = _extract_table_ocr(
                    path,
                    candidates,
                    dpi=table_ocr_dpi,
                    min_confidence=table_ocr_min_confidence,
                )
            except Exception as exc:
                logger.warning("Table OCR failed for %s: %s", path.name, exc)
            else:
                for page in pages:
                    extracted = table_text.get(page.page)
                    if not extracted:
                        continue
                    page.text = (
                        f"[Table OCR page {page.page}]\n{extracted}"
                        f"\n\n[Extracted page text]\n{page.text}"
                    )
                    table_ocr_pages.append(page.page)
                logger.info("Table OCR completed paper=%s pages=%s", path.name, table_ocr_pages)

    title = None
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip()
    return ParsedPdf(title=title, pages=pages, table_ocr_pages=table_ocr_pages)


def _extract_table_ocr(
    path: Path,
    page_numbers: list[int],
    *,
    dpi: int,
    min_confidence: float,
) -> dict[int, str]:
    import pypdfium2 as pdfium  # type: ignore[import-untyped]
    from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-untyped]

    document = pdfium.PdfDocument(path)
    engine = RapidOCR()
    extracted: dict[int, str] = {}
    try:
        for page_number in page_numbers:
            image = document[page_number - 1].render(scale=dpi / 72).to_pil()
            result, _ = engine(image)
            lines = _group_ocr_lines(result or [], min_confidence=min_confidence)
            table_lines = _select_table_lines(lines)
            if table_lines:
                summary = _summarize_table_rows(table_lines)
                full_table = "\n".join(table_lines)
                extracted[page_number] = (
                    f"[Table key rows]\n{summary}\n\n[Full table OCR]\n{full_table}"
                    if summary
                    else full_table
                )
    finally:
        document.close()
    return extracted


def _group_ocr_lines(result: list[Any], *, min_confidence: float) -> list[str]:
    items: list[dict[str, float | str]] = []
    heights: list[float] = []
    for raw in result:
        if len(raw) < 3:
            continue
        box, raw_text, raw_score = raw[0], raw[1], raw[2]
        text = " ".join(str(raw_text).split())
        score = float(raw_score)
        if not text or score < min_confidence or len(box) < 4:
            continue
        x_values = [float(point[0]) for point in box]
        y_values = [float(point[1]) for point in box]
        height = max(y_values) - min(y_values)
        heights.append(height)
        items.append(
            {
                "x": sum(x_values) / len(x_values),
                "y": sum(y_values) / len(y_values),
                "text": text,
            }
        )
    if not items:
        return []

    tolerance = max(8.0, median(heights) * 0.6 if heights else 8.0)
    rows: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda value: (float(value["y"]), float(value["x"]))):
        target = next(
            (
                row
                for row in reversed(rows[-3:])
                if abs(float(row["y"]) - float(item["y"])) <= tolerance
            ),
            None,
        )
        if target is None:
            target = {"y": float(item["y"]), "items": []}
            rows.append(target)
        target["items"].append(item)
        count = len(target["items"])
        target["y"] = (float(target["y"]) * (count - 1) + float(item["y"])) / count

    lines: list[str] = []
    for row in sorted(rows, key=lambda value: float(value["y"])):
        ordered = sorted(row["items"], key=lambda value: float(value["x"]))
        line = " | ".join(str(item["text"]) for item in ordered)
        if line:
            lines.append(line)
    return lines


def _summarize_table_rows(lines: list[str]) -> str:
    header: str | None = None
    header_columns = 0
    current_dataset: str | None = None
    rows: list[str] = []

    for line in lines:
        cells = [cell.strip() for cell in line.split("|")]
        lowered_cells = [cell.lower() for cell in cells]
        if header is None:
            if "dataset" in lowered_cells and "method" in lowered_cells:
                header = line
                header_columns = len(cells)
            continue
        if cells[0].lower().startswith("table"):
            break
        is_target = cells[0].lower().startswith(("isfm", "ours"))
        if len(cells) >= header_columns and not is_target:
            current_dataset = cells[0]
        if is_target and current_dataset:
            rows.append(f"{current_dataset} | {line}")

    if not header or not rows:
        return ""
    return "\n".join([header, *rows])


def _select_table_lines(lines: list[str]) -> list[str]:
    selected: list[str] = []
    for line in lines:
        lowered = line.lower()
        has_table_keyword = any(keyword in lowered for keyword in _TABLE_KEYWORDS)
        if len(_NUMBER.findall(line)) >= 2 or has_table_keyword:
            selected.append(line)
    return selected


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
