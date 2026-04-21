#!/usr/bin/env python3
"""
Шаг 2: Построение векторного индекса ChromaDB из чанков.

Что делает:
  • Читает все *.jsonl из data/chunks/
  • Генерирует embeddings батчами (поддержка GPU)
  • Записывает в ChromaDB с метаданными (источник, страница, раздел)
  • Идемпотентен: повторный запуск добавляет только новые чанки
  • --reset пересобирает индекс с нуля

Использование:
    python scripts/2_build_index.py
    python scripts/2_build_index.py --reset
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import ROOT, get_device, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_all_chunks(chunks_dir: Path) -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(chunks_dir.glob("*_chunks.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    logger.info(f"Загружено чанков: {len(chunks)} из {chunks_dir}")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Построить векторный индекс")
    parser.add_argument("--reset", action="store_true", help="Удалить коллекцию и пересобрать")
    args = parser.parse_args()

    config = load_config()
    device = get_device(config)

    chunks_dir = ROOT / config["paths"]["chunks_dir"]
    chroma_dir = ROOT / config["paths"]["chroma_dir"]
    chroma_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_all_chunks(chunks_dir)
    if not chunks:
        logger.error("Чанки не найдены — сначала запустите 1_ingest.py")
        sys.exit(1)

    # ── Embedding модель ────────────────────────────────
    embed_cfg = config["embedding"]
    logger.info(f"Загружаю embedding модель: {embed_cfg['model']}  (device={device})")
    embed_model = SentenceTransformer(embed_cfg["model"], device=device)

    # ── ChromaDB ────────────────────────────────────────
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if args.reset:
        try:
            client.delete_collection("anatomy")
            logger.info("Существующая коллекция удалена")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name="anatomy",
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids: set[str] = set(collection.get(include=[])["ids"])
    logger.info(f"Уже в индексе: {len(existing_ids)} чанков")

    # Присвоить стабильные ID и оставить только новые
    new_chunks: list[dict] = []
    for i, chunk in enumerate(chunks):
        cid = f"chunk_{i:07d}"
        if cid not in existing_ids:
            chunk["_id"] = cid
            new_chunks.append(chunk)

    if not new_chunks:
        logger.info("Индекс актуален. Используйте --reset для перестройки.")
        return

    logger.info(f"Генерирую embeddings для {len(new_chunks)} новых чанков...")

    batch_size = embed_cfg["batch_size"]
    texts = [c["text"] for c in new_chunks]
    all_embeddings: list[list[float]] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        embs = embed_model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(embs.tolist())

    # ── Запись в ChromaDB ────────────────────────────────
    logger.info("Записываю в ChromaDB...")
    insert_batch = 500
    for i in tqdm(range(0, len(new_chunks), insert_batch), desc="Индексация"):
        batch = new_chunks[i : i + insert_batch]
        collection.add(
            ids=[c["_id"] for c in batch],
            embeddings=all_embeddings[i : i + insert_batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "source":      c["source"],
                    "source_name": c["source_name"],
                    "headings":    " > ".join(c.get("headings") or []),
                    "page":        c.get("page") or 0,
                }
                for c in batch
            ],
        )

    logger.info(f"Готово. Всего в коллекции: {collection.count()} чанков")


if __name__ == "__main__":
    main()
