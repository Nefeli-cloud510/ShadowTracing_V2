"""Extract searchable text from uploaded scientific PDFs.

The live workspace keeps raw uploads under ``uploads/knowledge`` and exposes
plain-text copies under ``outputs/pdf_texts``. ``pdfplumber`` is preferred for
layout-aware extraction; ``pypdf`` is used as a fallback when the first engine
raises or returns almost no usable text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MIN_USABLE_TEXT_CHARS = 120


def _clean_text_fragment(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines() if line.strip())


def _extract_with_pdfplumber(pdf_path: Path) -> tuple[str, int, str]:
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for index, page in enumerate(pdf.pages, start=1):
            fragment = _clean_text_fragment(page.extract_text() or "")
            if fragment:
                pages.append(f"--- page {index} ---\n{fragment}")
    return "\n\n".join(pages), page_count, "pdfplumber"


def _extract_with_pypdf(pdf_path: Path) -> tuple[str, int, str]:
    from pypdf import PdfReader

    pages: list[str] = []
    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)
    for index, page in enumerate(reader.pages, start=1):
        fragment = _clean_text_fragment(page.extract_text() or "")
        if fragment:
            pages.append(f"--- page {index} ---\n{fragment}")
    return "\n\n".join(pages), page_count, "pypdf"


def extract_pdf_text(pdf_path: Path) -> dict[str, Any]:
    """Return extracted text plus provenance metadata.

    A successful result has ``error is None`` and non-empty ``text``. When every
    engine fails or yields only a near-empty string, the same structure is
    returned with an error message so callers can keep an auditable placeholder.
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path.name}")

    engines = (
        ("pdfplumber", _extract_with_pdfplumber),
        ("pypdf", _extract_with_pypdf),
    )
    errors: list[str] = []
    last_page_count = 0
    for engine_name, extractor in engines:
        try:
            text, page_count, engine = extractor(pdf_path)
        except Exception as exc:  # noqa: BLE001 - keep one extraction path alive
            errors.append(f"{engine_name}: {exc!r}")
            continue
        clean = text.strip()
        if len(clean) >= MIN_USABLE_TEXT_CHARS:
            return {
                "text": clean,
                "page_count": page_count,
                "character_count": len(clean),
                "engine": engine,
                "error": None,
            }
        errors.append(f"{engine_name}: extracted only {len(clean)} usable chars")
        last_page_count = max(last_page_count, page_count)

    return {
        "text": "",
        "page_count": last_page_count,
        "character_count": 0,
        "engine": None,
        "error": "; ".join(errors) or "no extraction engine succeeded",
    }
