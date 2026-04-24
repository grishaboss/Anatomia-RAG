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
    python scripts/1_ingest.py --rechunk               # пересчитать чанки с figures
    python scripts/1_ingest.py --force                 # переобработать всё
    python scripts/1_ingest.py --file books/sapin-vol-1.pdf  # один файл
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Показываем внутренние логи Docling (page-level прогресс, pipeline stages)
logging.getLogger("docling").setLevel(logging.INFO)
logging.getLogger("docling_core").setLevel(logging.INFO)


def _gpu_vram_info() -> str:
    """Вернуть строку с текущим использованием VRAM или пустую строку."""
    try:
        import torch
        if torch.cuda.is_available():
            used  = torch.cuda.memory_allocated(0) / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return f"VRAM {used:.1f}/{total:.1f} GB"
    except Exception:
        pass
    return ""


def _pdf_page_count(path: Path) -> int | None:
    """Быстро получить количество страниц PDF через PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        with fitz.open(str(path)) as doc:
            return len(doc)
    except Exception:
        return None

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

    ing = config["ingestion"]
    ocr_batch_size  = int(ing.get("ocr_batch_size",  8))
    num_cpu_threads = int(ing.get("num_cpu_threads", 4))

    ocr_opts = EasyOcrOptions(lang=ing["ocr_languages"])

    # Размер батча для EasyOCR: GPU обрабатывает N текстовых регионов за раз
    # Pydantic v2 бросает ValidationError (не AttributeError) на неизвестное поле
    try:
        ocr_opts.batch_size = ocr_batch_size
    except Exception:
        pass  # эта версия EasyOcrOptions не поддерживает batch_size

    # Некоторые версии Docling поддерживают force_full_page_ocr
    if force_full_ocr:
        try:
            ocr_opts.force_full_page_ocr = True
        except Exception:
            pass

    pipe_opts = PdfPipelineOptions()
    pipe_opts.do_ocr = True
    pipe_opts.ocr_options = ocr_opts
    pipe_opts.do_table_structure = True
    pipe_opts.images_scale = float(ing["figures_scale"])
    pipe_opts.generate_picture_images = bool(ing["extract_figures"])

    # AcceleratorOptions: больше CPU-потоков для препроцессинга → меньше простоев GPU
    try:
        from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice
        acc_opts = AcceleratorOptions(
            num_threads=num_cpu_threads,
            device=AcceleratorDevice.AUTO,
        )
        pipe_opts.accelerator_options = acc_opts
        logger.info(
            f"AcceleratorOptions: num_threads={num_cpu_threads}, device=AUTO, "
            f"ocr_batch_size={ocr_batch_size}"
        )
    except (ImportError, AttributeError):
        logger.debug("AcceleratorOptions недоступен в этой версии Docling — пропускаем")

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
#  Очистка артефактов PDF/Docling
# ──────────────────────────────────────────────

# Имена символов из PostScript/PDF spec, которые Docling оставляет как текст
_PDF_ENTITY_MAP = {
    "/hyphenminus": "-", "/hyphen": "-", "/minus": "-",
    "/endash": "\u2013", "/emdash": "\u2014",
    "/space": " ", "/period": ".", "/comma": ",",
    "/colon": ":", "/semicolon": ";",
    "/slash": "/", "/backslash": "\\",
    "/parenleft": "(", "/parenright": ")",
    "/bracketleft": "[", "/bracketright": "]",
    "/quotedbl": '"', "/quoteright": "\u2019", "/quoteleft": "\u2018",
    "/guillemotleft": "\u00AB", "/guillemotright": "\u00BB",
    "/bullet": "\u2022",
    "/multiply": "\u00D7", "/divide": "\u00F7",
    "/plus": "+", "/equal": "=", "/greater": ">", "/less": "<",
    "/percent": "%", "/numbersign": "#", "/at": "@", "/ampersand": "&",
    "/asterisk": "*", "/circumflex": "^", "/asciitilde": "~",
    "/asciicircum": "^", "/underscore": "_",
    "/fi": "fi", "/fl": "fl", "/ff": "ff", "/ffi": "ffi", "/ffl": "ffl",
}


def clean_md_text(text: str) -> str:
    """Устраняет артефакты Docling/PDF: заменяет /hyphenminus и аналогичные имена символов."""
    for entity, char in _PDF_ENTITY_MAP.items():
        if entity in text:
            text = text.replace(entity, char)
    return text


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

    # Определить автора по подпапке в books/
    books_root = ROOT / config["paths"]["books_dir"]
    author: str | None = None
    try:
        rel_to_books = source_path.relative_to(books_root)
        if rel_to_books.parts:
            author = rel_to_books.parts[0]
    except ValueError:
        pass  # файл не из books/ (например, questions/)

    chunks: list[dict] = []
    for chunk in chunker.chunk(doc):
        text = clean_md_text(chunk.text.strip())
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
                "author": author,
            }
        )
    return chunks


# ──────────────────────────────────────────────
#  Heading-based chunking (strategy: heading)
# ──────────────────────────────────────────────

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
_IMAGE_TAG_RE = re.compile(r'<!--\s*image\s*-->', re.IGNORECASE)


def chunk_by_headings_md(md_path: Path, source_path: Path, config: dict) -> list[dict]:
    """Чанкинг по заголовкам markdown.

    Каждая секция (заголовок + тело) становится одним чанком:
      - embed_text = только текст заголовка  (короткий, точный → хорошо совпадает с вопросами)
      - text      = heading + полное тело (весь контекст для LLM)
    """
    chunk_cfg = config.get("chunking", {})
    min_chars = int(chunk_cfg.get("min_section_chars", 100))
    max_chars = int(chunk_cfg.get("max_section_chars", 8000))

    text = md_path.read_text(encoding="utf-8")
    text = clean_md_text(text)
    figures_dir = ROOT / config["paths"].get("figures_dir", "data/figures")
    stem = source_path.stem

    books_root = ROOT / config["paths"]["books_dir"]
    author: str | None = None
    try:
        rel = source_path.relative_to(books_root)
        if rel.parts:
            author = rel.parts[0]
    except ValueError:
        pass

    base_meta = {
        "source":      str(source_path.relative_to(ROOT)),
        "source_name": source_path.name,
        "page":        None,
        "author":      author,
    }

    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        stripped = text.strip()
        if len(stripped) >= min_chars:
            return [{
                **base_meta,
                "text":          stripped[:max_chars],
                "embed_text":    source_path.stem,
                "headings":      [],
                "heading_level": 0,
            }]
        return []

    chunks: list[dict] = []
    fig_counter = 0  # running figure index across all sections
    for i, m in enumerate(matches):
        level        = len(m.group(1))
        heading_text = m.group(2).strip()

        body_start = m.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body       = text[body_start:body_end].strip()

        full_text = f"{heading_text}\n\n{body}" if body else heading_text

        # Collect figure paths for this section (count even if section is skipped)
        n_images = len(_IMAGE_TAG_RE.findall(body))
        section_figures: list[str] = []
        for j in range(n_images):
            fig_path = figures_dir / f"{stem}_fig_{fig_counter + j:03d}.png"
            if fig_path.exists():
                section_figures.append(str(fig_path.relative_to(ROOT)))
        fig_counter += n_images

        if len(full_text) < min_chars:
            continue

        if len(full_text) > max_chars:
            full_text = full_text[:max_chars]

        chunks.append({
            **base_meta,
            "text":          full_text,
            "embed_text":    heading_text,
            "headings":      [heading_text],
            "heading_level": level,
            "figures":       section_figures,
        })

    return chunks

# ──────────────────────────────────────────────
#  Обработка одного файла
# ──────────────────────────────────────────────

def process_file(
    source_path: Path,
    config: dict,
    dirs: dict[str, Path],
    rechunk_only: bool = False,
) -> tuple[int, int]:
    stem     = source_path.stem
    md_path  = dirs["processed"] / f"{stem}.md"
    strategy = config.get("chunking", {}).get("strategy", "fixed")

    if rechunk_only:
        if not md_path.exists():
            raise FileNotFoundError(
                f"MD не найден (сначала запустите без --rechunk): {md_path}"
            )
        if strategy != "heading":
            raise ValueError(
                "--rechunk работает только с strategy=heading в config.yaml"
            )
        logger.info(f"Перечанкую из MD: {source_path.name}")
        doc       = None
        n_figures = 0
    else:
        force_ocr_set  = {ROOT / f for f in config["ingestion"].get("force_ocr_for_files", [])}
        force_full_ocr = source_path in force_ocr_set
        logger.info(f"Конвертирую: {source_path.name}  (full_ocr={force_full_ocr})")

        converter = build_converter(config, force_full_ocr=force_full_ocr)
        result    = converter.convert(str(source_path))
        doc       = result.document

        md_path.write_text(doc.export_to_markdown(), encoding="utf-8")

        n_figures = 0
        if config["ingestion"]["extract_figures"]:
            min_px    = int(config["ingestion"].get("min_figure_px", 80))
            fig_paths = extract_figures(doc, stem, dirs["figures"], min_px=min_px)
            n_figures = len(fig_paths)

    # Чанкинг
    if strategy == "heading":
        chunks = chunk_by_headings_md(md_path, source_path, config)
    else:
        chunks = chunk_document(doc, source_path, config)  # type: ignore[arg-type]

    chunks_path = dirs["chunks"] / f"{stem}_chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    n_chunks_with_figs = sum(1 for c in chunks if c.get("figures"))
    fig_info = f"{n_figures} извлечено" if not rechunk_only else f"{n_chunks_with_figs} чанков с рисунками"
    logger.info(f"  → {len(chunks)} чанков, рисунков: {fig_info}")
    return len(chunks), n_figures


# ──────────────────────────────────────────────
#  Точка входа
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestion: обработка документов через Docling")
    parser.add_argument("--force",   action="store_true", help="Переобработать все файлы")
    parser.add_argument("--file",    help="Путь к конкретному файлу (абсолютный или от корня проекта)")
    parser.add_argument(
        "--rechunk",
        action="store_true",
        help=(
            "Перечанковать из готовых MD-файлов (без OCR). "
            "Требует strategy=heading в config.yaml. "
            "Используется после смены стратегии чанкинга."
        ),
    )
    parser.add_argument(
        "--chunks-dir",
        default=None,
        help=(
            "Переопределить директорию для сохранения чанков (относительно корня проекта). "
            "Например: data/chunks_no_gocr. По умолчанию — значение из config.yaml."
        ),
    )
    parser.add_argument(
        "--skip-authors",
        nargs="*",
        metavar="AUTHOR",
        default=[],
        help=(
            "Пропустить файлы из указанных подпапок books/ (по имени автора). "
            "Например: --skip-authors Гайворонский"
        ),
    )
    parser.add_argument(
        "--no-force-ocr-authors",
        nargs="*",
        metavar="AUTHOR",
        default=[],
        help=(
            "Не применять force OCR для файлов этих авторов — Docling будет использовать "
            "нативный текстовый слой PDF (если есть). "
            "Например: --no-force-ocr-authors Гайворонский"
        ),
    )
    args = parser.parse_args()

    config = load_config()

    # Переопределение chunks_dir через CLI
    chunks_dir_path = ROOT / (args.chunks_dir if args.chunks_dir else config["paths"]["chunks_dir"])

    dirs: dict[str, Path] = {
        "processed": ROOT / config["paths"]["processed_dir"],
        "figures":   ROOT / config["paths"]["figures_dir"],
        "chunks":    chunks_dir_path,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    if args.chunks_dir:
        logger.info(f"Chunks dir переопределён: {chunks_dir_path}")

    # Нормализуем авторов для фильтрации
    skip_authors = {a.strip().lower() for a in (args.skip_authors or [])}
    no_force_ocr_authors = {a.strip().lower() for a in (args.no_force_ocr_authors or [])}

    # Изменяем force_ocr_for_files в конфиге — убираем файлы для no_force_ocr_authors
    if no_force_ocr_authors:
        original_force_ocr = config["ingestion"].get("force_ocr_for_files", [])
        books_root = ROOT / config["paths"]["books_dir"]
        filtered_force_ocr = []
        for p_str in original_force_ocr:
            p = Path(p_str)
            # Определить автора по первой части пути внутри books/
            try:
                rel = p.relative_to(books_root) if p.is_absolute() else p
                author_folder = rel.parts[0].lower() if rel.parts else ""
            except Exception:
                parts = Path(p_str).parts
                author_folder = ""
                for i, part in enumerate(parts):
                    if part.lower() in ("books",):
                        author_folder = parts[i + 1].lower() if i + 1 < len(parts) else ""
                        break
            if author_folder not in no_force_ocr_authors:
                filtered_force_ocr.append(p_str)
            else:
                logger.info(f"Force OCR отключён для: {p_str}")
        config["ingestion"]["force_ocr_for_files"] = filtered_force_ocr

    # Имя манифеста зависит от chunks_dir (разные пресеты → разные манифесты)
    manifest_name = "manifest.json" if not args.chunks_dir else f"manifest_{chunks_dir_path.name}.json"
    manifest_path = ROOT / "data" / manifest_name
    manifest = load_manifest(manifest_path)

    if args.file:
        p = Path(args.file)
        files = [p if p.is_absolute() else ROOT / p]
    else:
        files = collect_source_files(config)
        # Фильтрация по авторам
        if skip_authors:
            books_root = ROOT / config["paths"]["books_dir"]
            before = len(files)
            filtered = []
            for f in files:
                try:
                    rel = f.relative_to(books_root)
                    author_folder = rel.parts[0].lower() if rel.parts else ""
                except ValueError:
                    author_folder = ""
                if author_folder not in skip_authors:
                    filtered.append(f)
            files = filtered
            logger.info(f"После фильтрации --skip-authors {skip_authors}: {len(files)}/{before} файлов")
        logger.info(f"Найдено {len(files)} файлов")

    processed = skipped = failed = 0

    for source_path in tqdm(files, desc="Ингестия"):
        rel = str(source_path.relative_to(ROOT))
        current_hash = file_hash(source_path)

        if not args.force and not args.rechunk and rel in manifest:
            if manifest[rel].get("hash") == current_hash:
                skipped += 1
                continue

        try:
            n_chunks, n_figs = process_file(source_path, config, dirs, rechunk_only=args.rechunk)
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
    logger.info(f"Чанки сохранены в: {chunks_dir_path}")


if __name__ == "__main__":
    main()
