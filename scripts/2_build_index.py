#!/usr/bin/env python3
"""
Шаг 2: Построение векторного индекса ChromaDB из чанков.

Что делает:
  • Читает все *.jsonl из data/chunks/
  • Генерирует embeddings батчами (поддержка GPU)
  • Записывает в ChromaDB с метаданными (источник, страница, раздел)
  • Идемпотентен: повторный запуск добавляет только новые чанки
  • --reset пересобирает индекс с нуля

Поддержка нескольких ChromaDB-коллекций по имени эмбеддера:
  • --collection-name    — явное имя коллекции (по умолчанию: anatomy)
  • --embedding-model    — переопределить модель из config.yaml
  • --embed-strategy     — heading_only | text_only | heading_plus_text
  • --embed-chars        — кол-во символов текста для heading_plus_text
  • --merge-window       — склеить N соседних чанков из одного файла
  • --max-chunk-chars    — разбить чанки длиннее N символов

Использование:
    python scripts/2_build_index.py
    python scripts/2_build_index.py --reset
    python scripts/2_build_index.py --collection-name anatomy__bge-m3__text_only \\
        --embedding-model BAAI/bge-m3 --embed-strategy text_only
"""
from __future__ import annotations

import argparse
import json
import logging
import re
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


def merge_chunks(chunks: list[dict], window: int) -> list[dict]:
    """Склеить до `window` последовательных чанков из одного файла."""
    if window <= 1:
        return chunks
    merged: list[dict] = []
    i = 0
    while i < len(chunks):
        group = [chunks[i]]
        source = chunks[i].get("source")
        j = i + 1
        while j < i + window and j < len(chunks) and chunks[j].get("source") == source:
            group.append(chunks[j])
            j += 1
        if len(group) == 1:
            merged.append(chunks[i])
        else:
            base = dict(group[0])
            base["text"] = "\n\n".join(c["text"] for c in group)
            merged.append(base)
        i += len(group)
    return merged


def split_large_chunks(chunks: list[dict], max_chars: int) -> list[dict]:
    """Разбить чанки длиннее `max_chars` символов по границам предложений."""
    if max_chars <= 0:
        return chunks
    result: list[dict] = []
    sent_re = re.compile(r"(?<=[.!?])\s+")
    for chunk in chunks:
        text = chunk["text"]
        if len(text) <= max_chars:
            result.append(chunk)
            continue
        sentences = sent_re.split(text)
        current = ""
        part = 0
        for sent in sentences:
            if current and len(current) + len(sent) + 1 > max_chars:
                new_c = dict(chunk)
                new_c["text"] = current.strip()
                new_c["_split_part"] = part
                result.append(new_c)
                current = sent
                part += 1
            else:
                current = f"{current} {sent}".strip() if current else sent
        if current:
            new_c = dict(chunk)
            new_c["text"] = current.strip()
            new_c["_split_part"] = part
            result.append(new_c)
    return result


def get_embed_text(chunk: dict, strategy: str, chars: int) -> str:
    """Сформировать строку для эмбеддинга по заданной стратегии."""
    heading = chunk.get("embed_text") or " > ".join(chunk.get("headings") or [])
    text = chunk.get("text", "")
    if strategy == "heading_only":
        return heading or text[:200]
    elif strategy == "text_only":
        return text if chars == 0 else text[:chars]
    elif strategy == "heading_plus_text":
        snippet = text if chars == 0 else text[:chars]
        return f"{heading}. {snippet}" if heading else snippet
    else:
        raise ValueError(f"Неизвестная embed_strategy: {strategy!r}")


def _derive_collection_name(model: str, strategy: str, chars: int, merge: int, split: int) -> str:
    """Автоматически вывести имя коллекции из параметров.

    Пример: anatomy__bge-m3__text_only__c300_m1_s0
    ChromaDB допускает только [a-zA-Z0-9_-] и длину 3–63.
    """
    slug = model.split("/")[-1]
    slug = re.sub(r"multilingual-?", "m", slug)
    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", slug)[:20]
    strat = strategy.replace("_", "-")[:12]
    name = f"anatomy__{slug}__{strat}__c{chars}_m{merge}_s{split}"
    # ChromaDB: max 63 символа
    return name[:63]


def main() -> None:
    parser = argparse.ArgumentParser(description="Построить векторный индекс ChromaDB")
    parser.add_argument("--reset", action="store_true",
                        help="Удалить коллекцию и пересобрать с нуля")

    # Именование коллекции
    parser.add_argument("--collection-name", default=None,
                        help="Имя ChromaDB-коллекции (по умолчанию: anatomy или авто-derived)")

    # Переопределение embedding-модели
    parser.add_argument("--embedding-model", default=None,
                        help="HuggingFace модель для эмбеддингов (переопределяет config.yaml)")

    # Стратегия формирования embed_text
    parser.add_argument("--embed-strategy", default="heading_only",
                        choices=["heading_only", "text_only", "heading_plus_text"],
                        help="Что эмбеддируем: только заголовок, только текст, или заголовок+текст")
    parser.add_argument("--embed-chars", type=int, default=300,
                        help="Для heading_plus_text: первые N символов текста (0=всё)")

    # Пре-обработка чанков
    parser.add_argument("--merge-window", type=int, default=1,
                        help="Склеить N соседних чанков из одного источника (1=не мержить)")
    parser.add_argument("--max-chunk-chars", type=int, default=0,
                        help="Разбить чанки длиннее N символов (0=не разбивать)")
    args = parser.parse_args()

    config = load_config()
    device = get_device(config)

    chunks_dir = ROOT / config["paths"]["chunks_dir"]
    chroma_dir = ROOT / config["paths"]["chroma_dir"]
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # ── Параметры ────────────────────────────────────────
    embed_model_name = args.embedding_model or config["embedding"]["model"]
    strategy = args.embed_strategy
    embed_chars = args.embed_chars
    merge_window = args.merge_window
    max_chunk_chars = args.max_chunk_chars

    # Имя коллекции
    if args.collection_name:
        coll_name = args.collection_name
    elif args.embedding_model or strategy != "heading_only" or merge_window > 1 or max_chunk_chars > 0:
        # Не стандартный конфиг → автоматическое имя
        coll_name = _derive_collection_name(
            embed_model_name, strategy, embed_chars, merge_window, max_chunk_chars
        )
    else:
        coll_name = "anatomy"  # обратная совместимость

    logger.info(f"ChromaDB коллекция: '{coll_name}'")
    logger.info(f"Embedding модель:   {embed_model_name}")
    logger.info(f"Стратегия:          {strategy} (chars={embed_chars})")
    logger.info(f"Merge window:       {merge_window} | Max chunk chars: {max_chunk_chars}")

    chunks = load_all_chunks(chunks_dir)
    if not chunks:
        logger.error("Чанки не найдены — сначала запустите 1_ingest.py")
        sys.exit(1)

    # Пре-обработка
    if merge_window > 1:
        chunks = merge_chunks(chunks, merge_window)
        logger.info(f"После merge ({merge_window}): {len(chunks)} чанков")
    if max_chunk_chars > 0:
        chunks = split_large_chunks(chunks, max_chunk_chars)
        logger.info(f"После split ({max_chunk_chars}): {len(chunks)} чанков")

    # ── Embedding модель ────────────────────────────────
    logger.info(f"Загружаю embedding модель: {embed_model_name}  (device={device})")
    fp16 = bool(int(__import__("os").environ.get("EMBED_FP16", "0")))
    embed_model = SentenceTransformer(embed_model_name, device=device)
    if fp16 and device != "cpu":
        embed_model = embed_model.half()
        logger.info("FP16 включён (EMBED_FP16=1)")

    # ── ChromaDB ────────────────────────────────────────
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if args.reset:
        try:
            client.delete_collection(coll_name)
            logger.info(f"Существующая коллекция '{coll_name}' удалена")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=coll_name,
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
    logger.info(f"Стратегия embed_text: {strategy}")

    batch_size = config["embedding"]["batch_size"]
    texts = [get_embed_text(c, strategy, embed_chars) for c in new_chunks]
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
                    "source":         c["source"],
                    "source_name":    c["source_name"],
                    "headings":       " > ".join(c.get("headings") or []),
                    "page":           c.get("page") or 0,
                    "author":         c.get("author") or "",
                    "figures":        json.dumps(c.get("figures") or [], ensure_ascii=False),
                    "embed_strategy": strategy,
                    "embed_model":    embed_model_name,
                }
                for c in batch
            ],
        )

    logger.info(f"Готово. Коллекция '{coll_name}': {collection.count()} чанков")


if __name__ == "__main__":
    main()
