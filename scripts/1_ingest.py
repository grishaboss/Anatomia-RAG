#!/usr/bin/env python3
"""
Шаг 1: Ингестия документов через Docling.

Что делает:
  • Конвертирует PDF / DOCX / PPTX в markdown + JSON
  • Для сканированных PDF (Сапин) запускает полное OCR на каждой странице
  • Извлекает рисунки в data/figures/
  • Нарезает документ на чанки (HybridChunker) → data/chunks/*.jsonl
  • Ведёт манифест — повторно обрабатывает только изменившиеся файлы

Использование:
    python scripts/1_ingest.py                         # обработать новые/изменённые
    python scripts/1_ingest.py --force                 # переобработать всё
    python scripts/1_ingest.py --file books/sapin-vol-1.pdf  # один файл
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".pdf", ".docx", ".pptx"}


# ──────────────────────────────────────────────
#  Вспомогательные функции
# ──────────────────────────────────────────────

def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def collect_source_files(config: dict) -> list[Path]:
    files: list[Path] = []
    for folder in [config["paths"]["books_dir"], config["paths"]["questions_dir"]]:
        p = ROOT / folder
        if p.exists():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXT:
                    files.append(f)
    return files


# ──────────────────────────────────────────────
#  Docling конвертер
# ──────────────────────────────────────────────

def build_converter(config: dict, force_full_ocr: bool = False):
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions

    ocr_opts = EasyOcrOptions(lang=config["ingestion"]["ocr_languages"])

    # Некоторые версии Docling поддерживают force_full_page_ocr
    if force_full_ocr:
        try:
            ocr_opts.force_full_page_ocr = True
        except AttributeError:
            pass  # не критично: у сканированных PDF всё равно нет текстового слоя

    pipe_opts = PdfPipelineOptions()
    pipe_opts.do_ocr = True
    pipe_opts.ocr_options = ocr_opts
    pipe_opts.do_table_structure = True
    pipe_opts.images_scale = float(config["ingestion"]["figures_scale"])
    pipe_opts.generate_picture_images = bool(config["ingestion"]["extract_figures"])

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipe_opts)}
    )


# ──────────────────────────────────────────────
#  Извлечение рисунков
# ──────────────────────────────────────────────

def extract_figures(doc, stem: str, figures_dir: Path, min_px: int = 80) -> list[str]:
    try:
        from docling.datamodel.document import PictureItem
    except ImportError:
        logger.warning("PictureItem недоступен — рисунки не будут извлечены")
        return []

    saved: list[str] = []
    idx = 0
    for element, _ in doc.iterate_items():
        if not isinstance(element, PictureItem):
            continue
        try:
            img = element.get_image(doc)
            if img and img.width >= min_px and img.height >= min_px:
                name = f"{stem}_fig_{idx:03d}.png"
                path = figures_dir / name
                img.save(path)
                saved.append(str(path.relative_to(ROOT)))
                idx += 1
        except Exception as e:
            logger.debug(f"Рисунок {idx} пропущен: {e}")
    return saved


# ──────────────────────────────────────────────
#  Чанкинг
# ──────────────────────────────────────────────

def chunk_document(doc, source_path: Path, config: dict) -> list[dict]:
    try:
        from docling.chunking import HybridChunker
    except ImportError:
        try:
            from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        except ImportError:
            raise ImportError(
                "HybridChunker не найден. Установите docling >= 2.5.0"
            )

    chunk_cfg = config["chunking"]
    chunker = HybridChunker(
        tokenizer=chunk_cfg["tokenizer"],
        max_tokens=chunk_cfg["max_tokens"],
    )

    chunks: list[dict] = []
    for chunk in chunker.chunk(doc):
        text = chunk.text.strip()
        if not text or len(text) < 30:
            continue

        headings = list(getattr(chunk.meta, "headings", None) or [])

        page: int | None = None
        doc_items = getattr(chunk.meta, "doc_items", None) or []
        if doc_items:
            prov = getattr(doc_items[0], "prov", None) or []
            if prov:
                page = getattr(prov[0], "page_no", None)

        chunks.append(
            {
                "text": text,
                "source": str(source_path.relative_to(ROOT)),
                "source_name": source_path.name,
                "headings": headings,
                "page": page,
            }
        )
    return chunks


# ──────────────────────────────────────────────
#  Обработка одного файла
# ──────────────────────────────────────────────

def process_file(source_path: Path, config: dict, dirs: dict[str, Path]) -> tuple[int, int]:
    force_ocr_set = {ROOT / f for f in config["ingestion"].get("force_ocr_for_files", [])}
    force_full_ocr = source_path in force_ocr_set

    logger.info(f"Конвертирую: {source_path.name}  (full_ocr={force_full_ocr})")

    converter = build_converter(config, force_full_ocr=force_full_ocr)
    result = converter.convert(str(source_path))
    doc = result.document
    stem = source_path.stem

    # Сохранить markdown
    md_path = dirs["processed"] / f"{stem}.md"
    md_path.write_text(doc.export_to_markdown(), encoding="utf-8")

    # Извлечь рисунки
    n_figures = 0
    if config["ingestion"]["extract_figures"]:
        min_px = int(config["ingestion"].get("min_figure_px", 80))
        fig_paths = extract_figures(doc, stem, dirs["figures"], min_px=min_px)
        n_figures = len(fig_paths)

    # Нарезать на чанки
    chunks = chunk_document(doc, source_path, config)

    chunks_path = dirs["chunks"] / f"{stem}_chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"  → {len(chunks)} чанков, {n_figures} рисунков")
    return len(chunks), n_figures


# ──────────────────────────────────────────────
#  Точка входа
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestion: обработка документов через Docling")
    parser.add_argument("--force", action="store_true", help="Переобработать все файлы")
    parser.add_argument("--file", help="Путь к конкретному файлу (абсолютный или от корня проекта)")
    args = parser.parse_args()

    config = load_config()

    dirs: dict[str, Path] = {
        "processed": ROOT / config["paths"]["processed_dir"],
        "figures":   ROOT / config["paths"]["figures_dir"],
        "chunks":    ROOT / config["paths"]["chunks_dir"],
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT / "data" / "manifest.json"
    manifest = load_manifest(manifest_path)

    if args.file:
        p = Path(args.file)
        files = [p if p.is_absolute() else ROOT / p]
    else:
        files = collect_source_files(config)
        logger.info(f"Найдено {len(files)} файлов")

    processed = skipped = failed = 0

    for source_path in tqdm(files, desc="Ингестия"):
        rel = str(source_path.relative_to(ROOT))
        current_hash = file_hash(source_path)

        if not args.force and rel in manifest:
            if manifest[rel].get("hash") == current_hash:
                skipped += 1
                continue

        try:
            n_chunks, n_figs = process_file(source_path, config, dirs)
            manifest[rel] = {
                "hash":         current_hash,
                "processed_at": datetime.now().isoformat(),
                "chunks":       n_chunks,
                "figures":      n_figs,
            }
            save_manifest(manifest_path, manifest)
            processed += 1
        except Exception as e:
            logger.error(f"ОШИБКА [{rel}]: {e}")
            failed += 1

    logger.info(
        f"\nГотово.  Обработано: {processed}  |  Пропущено (без изменений): {skipped}  |  Ошибок: {failed}"
    )


if __name__ == "__main__":
    main()
