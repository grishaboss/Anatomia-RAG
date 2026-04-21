#!/usr/bin/env python3
"""
Шаг 0: Предварительная загрузка всех ML-моделей.

Запустите один раз перед основным пайплайном — особенно важно на
вычислительных кластерах, где compute-нода может быть без интернета.

Использование:
    python scripts/0_download_models.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def download_docling_models(config: dict) -> None:
    """Инициализируем DocumentConverter — это запускает скачивание моделей Docling."""
    logger.info("Инициализация Docling (layout, table-structure модели)...")
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions

    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.ocr_options = EasyOcrOptions(lang=config["ingestion"]["ocr_languages"])
    opts.do_table_structure = True
    opts.generate_picture_images = True

    # Создание конвертера запускает загрузку моделей
    DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    logger.info("  ✓ Docling модели готовы")


def download_embedding_model(config: dict, device: str) -> None:
    model_name = config["embedding"]["model"]
    logger.info(f"Загрузка embedding модели: {model_name}  (device={device})")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    # Тестовый прогон
    model.encode(["тест анатомия"], normalize_embeddings=True)
    logger.info("  ✓ Embedding модель готова")


def download_easyocr_models(config: dict) -> None:
    langs = config["ingestion"]["ocr_languages"]
    logger.info(f"Загрузка EasyOCR моделей для языков: {langs}")
    import easyocr

    easyocr.Reader(langs, gpu=False)
    logger.info("  ✓ EasyOCR модели готовы")


def download_chunker_tokenizer(config: dict) -> None:
    tok_name = config["chunking"]["tokenizer"]
    logger.info(f"Загрузка токенизатора чанкера: {tok_name}")
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(tok_name)
    logger.info("  ✓ Токенизатор готов")


def main() -> None:
    config = load_config()
    device = get_device(config)
    logger.info(f"Устройство: {device}\n")

    download_docling_models(config)
    download_embedding_model(config, device)
    download_easyocr_models(config)
    download_chunker_tokenizer(config)

    logger.info("\nВсе модели загружены и готовы к работе!")


if __name__ == "__main__":
    main()
