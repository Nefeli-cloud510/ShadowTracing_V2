"""Render selected PDF pages to PNG for report visual QA."""

import sys
from pathlib import Path

import pypdfium2 as pdfium


def render(pdf_path: Path, page_numbers: list[int], scale: float = 1.6) -> list[Path]:
    pdf = pdfium.PdfDocument(str(pdf_path))
    out_paths: list[Path] = []
    for page_number in page_numbers:
        page = pdf[page_number - 1]
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        out = pdf_path.with_name(
            f"{pdf_path.stem}_page_{page_number:02d}.png"
        )
        pil_image.save(out)
        out_paths.append(out)
    return out_paths


def ocr_pages(pdf_path: Path, page_numbers: list[int], scale: float = 2.0) -> None:
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    rendered = render(pdf_path, page_numbers, scale=scale)
    for image_path in rendered:
        print(f"== OCR {image_path.name}")
        result, _ = engine(str(image_path))
        lines: list[str] = []
        for item in result or []:
            box = item[0]
            text = str(item[1])
            score = float(item[2]) if len(item) > 2 else 0.0
            top = min(point[1] for point in box)
            lines.append(f"{top:05.1f}|{score:.2f}|{text}")
        for line in sorted(lines):
            print(line)
        print(f"-- text count: {len(lines)}")


def info(pdf_path: Path) -> None:
    pdf = pdfium.PdfDocument(str(pdf_path))
    print(f"{pdf_path} pages={len(pdf)}")


def show_lines(path_str: str, start: int, end: int) -> None:
    lines = Path(path_str).read_text(encoding="utf-8").splitlines()
    for index in range(start - 1, min(end, len(lines))):
        print(f"{index + 1:5d}  {lines[index]}")


def scan(pdf_path: Path) -> None:
    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            words = page.extract_words() or []
            char_count = sum(1 for ch in text if not ch.isspace())
            image_count = len(page.images)
            blank = "BLANK" if char_count == 0 else ""
            tofu = "TOFU?" if text.count("□") > 0 or text.count("\ufffd") > 0 else ""
            clipped = ""
            for word in words:
                if word["x1"] > page.width - 2 or word["x0"] < 2:
                    clipped = "CLIPPED"
                    break
            if image_count or blank or tofu or clipped or index in (24, 27):
                print(
                    f"p{index:02d} images={image_count} chars={char_count} "
                    f"words={len(words)} {blank}{tofu}{clipped}"
                )


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--ocr":
        pdf_file = Path(sys.argv[2])
        ocr_pages(pdf_file, [int(item) for item in sys.argv[3:]])
    elif len(sys.argv) > 2 and sys.argv[1] == "--info":
        info(Path(sys.argv[2]))
    elif len(sys.argv) > 2 and sys.argv[1] == "--scan":
        scan(Path(sys.argv[2]))
    elif len(sys.argv) > 3 and sys.argv[1] == "--lines":
        show_lines(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        pdf_file = Path(sys.argv[1])
        for path in render(pdf_file, [int(item) for item in sys.argv[2:]]):
            print(path)
