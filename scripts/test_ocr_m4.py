#!/usr/bin/env python3
"""
Тестовый OCR на MacBook Air M4 — несколько страниц Сапина.

Запуск:
    python scripts/test_ocr_m4.py
    python scripts/test_ocr_m4.py --pages 3       # первые 3 страницы
    python scripts/test_ocr_m4.py --page 50        # одна конкретная страница
    python scripts/test_ocr_m4.py --vol 2          # второй том
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import ROOT

# PyMuPDF экспортирует себя то как 'fitz', то как 'pymupdf' в зависимости от версии
try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore[no-redef]

# ──────────────────────────────────────────────
#  Зависимости
# ──────────────────────────────────────────────
def check_deps() -> None:
    missing = []
    for pkg in ("docling", "easyocr"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Установите: pip install {' '.join(missing)}")
        sys.exit(1)


# ──────────────────────────────────────────────
#  Тест 1: Docling (полный пайплайн — layout + OCR)
# ──────────────────────────────────────────────
def test_docling(pdf_path: Path, pages: list[int]) -> None:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    print("\n" + "═" * 60)
    print("ТЕСТ: Docling (layout detection + EasyOCR)")
    print("═" * 60)

    ocr_opts = EasyOcrOptions(lang=["ru", "en"])
    try:
        ocr_opts.force_full_page_ocr = True
    except AttributeError:
        pass

    pipe_opts = PdfPipelineOptions()
    pipe_opts.do_ocr = True
    pipe_opts.ocr_options = ocr_opts
    pipe_opts.do_table_structure = True
    pipe_opts.generate_picture_images = True
    pipe_opts.images_scale = 2.0

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipe_opts)}
    )

    # Ограничить диапазон страниц
    # Docling v2 принимает page_range через kwargs или через DocumentStream
    # Самый простой способ — конвертировать весь файл и взять нужные страницы из результата
    print(f"Конвертирую страницы {pages} из {pdf_path.name}...")
    t0 = time.perf_counter()

    result = converter.convert(str(pdf_path))

    elapsed = time.perf_counter() - t0
    doc = result.document

    # Вывести markdown первых страниц
    md = doc.export_to_markdown()
    lines = md.splitlines()

    print(f"\n[Время: {elapsed:.1f} с]\n")
    print("─" * 60)
    print("Первые 100 строк распознанного текста:")
    print("─" * 60)
    for line in lines[:100]:
        print(line)

    # Статистика
    total_chars = len(md)
    n_figures = sum(1 for el, _ in doc.iterate_items()
                    if el.__class__.__name__ == "PictureItem")

    print("\n─" * 60)
    print(f"Символов распознано:  {total_chars:,}")
    print(f"Рисунков найдено:     {n_figures}")

    # Сохранить результат
    out_dir = ROOT / "data" / "test_ocr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / f"{pdf_path.stem}_test.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"Markdown сохранён:    {out_md}")

    if total_chars < 200:
        print("\n⚠  Мало текста — возможно, OCR не сработал или страницы пустые")
    else:
        print("\n✓  OCR работает корректно")


# ──────────────────────────────────────────────
#  Тест 2: Прямой EasyOCR без Docling (для диагностики)
# ──────────────────────────────────────────────
def test_easyocr_direct(pdf_path: Path, page_no: int = 5) -> None:
    import easyocr

    print("\n" + "═" * 60)
    print(f"ТЕСТ: EasyOCR напрямую (страница {page_no})")
    print("═" * 60)

    # Рендер страницы через PyMuPDF
    doc = fitz.open(str(pdf_path))
    if page_no >= len(doc):
        page_no = min(5, len(doc) - 1)

    page = doc[page_no]
    mat = fitz.Matrix(2.0, 2.0)       # 144 dpi
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")

    print(f"Страница {page_no}: {pix.width}×{pix.height} px")

    # OCR
    t0 = time.perf_counter()
    reader = easyocr.Reader(["ru", "en"], gpu=False)  # M4 — нет CUDA, MPS через CPU fallback
    results = reader.readtext(img_bytes)
    elapsed = time.perf_counter() - t0

    print(f"Время OCR: {elapsed:.1f} с")
    print(f"Блоков текста: {len(results)}\n")

    for _, text, conf in results[:20]:
        print(f"  [{conf:.2f}] {text}")

    if results:
        print("\n✓  EasyOCR распознаёт текст")
    else:
        print("\n⚠  Текст не найден на этой странице")


# ──────────────────────────────────────────────
#  Точка входа
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Тестовый OCR на M4")
    parser.add_argument("--vol", type=int, default=1, choices=[1, 2], help="Том Сапина (1 или 2)")
    parser.add_argument("--pages", type=int, default=2, help="Кол-во страниц для теста Docling")
    parser.add_argument("--page", type=int, default=5, help="Страница для прямого теста EasyOCR")
    parser.add_argument("--mode", choices=["docling", "easyocr", "both"], default="both")
    args = parser.parse_args()

    check_deps()

    pdf_path = ROOT / f"books/sapin-vol-{args.vol}.pdf"
    if not pdf_path.exists():
        print(f"Файл не найден: {pdf_path}")
        sys.exit(1)

    n_pages = len(fitz.open(str(pdf_path)))
    print(f"Файл: {pdf_path.name}  ({n_pages} страниц)")

    pages = list(range(1, args.pages + 1))

    if args.mode in ("easyocr", "both"):
        test_easyocr_direct(pdf_path, page_no=args.page)

    if args.mode in ("docling", "both"):
        test_docling(pdf_path, pages=pages)


if __name__ == "__main__":
    main()
