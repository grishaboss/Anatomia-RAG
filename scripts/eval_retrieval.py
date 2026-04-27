#!/usr/bin/env python3
"""
Ручная оценка качества retrieval.

Показывает топ-N чанков для заданного запроса с полным текстом —
удобно для сравнения лучше/хуже при разных стратегиях индексирования.

Использование:
    python scripts/eval_retrieval.py "позвонки строение"
    python scripts/eval_retrieval.py "позвонки строение" --top-k 10
    python scripts/eval_retrieval.py "позвонки строение" --author Сапин
    python scripts/eval_retrieval.py "позвонки строение" --hyde
    python scripts/eval_retrieval.py --inspect-chunk  # показать структуру первого чанка
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import ROOT, get_device, load_config, retrieve


def inspect_chunk() -> None:
    """Показать структуру и поля одного чанка из индекса."""
    cfg = load_config()
    chunks_dir = ROOT / cfg["paths"]["chunks_dir"]
    first_file = next(chunks_dir.glob("*_chunks.jsonl"), None)
    if not first_file:
        print("Чанки не найдены")
        return
    with open(first_file, encoding="utf-8") as f:
        d = json.loads(f.readline())
    print(f"Файл: {first_file.name}")
    print(f"Поля: {list(d.keys())}")
    print(f"\nembed_text ({len(d.get('embed_text',''))} симв.):")
    print(" ", repr(d.get("embed_text", "<нет>")[:400]))
    print(f"\ntext ({len(d.get('text',''))} симв.):")
    print(" ", repr(d.get("text", "")[:400]))
    meta = {k: v for k, v in d.items() if k not in ("text", "embed_text")}
    print(f"\nМетаданные: {meta}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ручная оценка retrieval")
    parser.add_argument("query", nargs="?", help="Поисковый запрос")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--author", default=None, help="Фильтр по автору")
    parser.add_argument("--hyde", action="store_true", help="Включить HyDE")
    parser.add_argument("--text-len", type=int, default=300,
                        help="Сколько символов текста показывать (0 = весь)")
    parser.add_argument("--inspect-chunk", action="store_true",
                        help="Показать структуру чанка и выйти")
    args = parser.parse_args()

    if args.inspect_chunk:
        inspect_chunk()
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    cfg = load_config()
    cfg["retrieval"]["top_k"] = args.top_k
    cfg["retrieval"]["score_threshold"] = 0.0  # показываем всё

    device = get_device(cfg)
    print(f"Загружаю embedding модель ({device})…", flush=True)

    from sentence_transformers import SentenceTransformer
    import chromadb

    embed_model = SentenceTransformer(cfg["embedding"]["model"], device=device)
    chroma_dir = ROOT / cfg["paths"]["chroma_dir"]
    client = chromadb.PersistentClient(path=str(chroma_dir))
    coll_name = cfg.get("retrieval", {}).get("collection_name", "anatomy")
    collection = client.get_collection(coll_name)

    print(f"\nЗапрос: «{args.query}»")
    print(f"HyDE: {'да' if args.hyde else 'нет'}  |  top_k: {args.top_k}  |  автор: {args.author or 'все'}")
    print("─" * 80)

    chunks = retrieve(
        args.query,
        collection,
        embed_model,
        cfg,
        author_filter=args.author,
        hyde=args.hyde,
    )

    if not chunks:
        print("Ничего не найдено (все чанки ниже score_threshold).")
        return

    for i, chunk in enumerate(chunks, 1):
        meta = chunk["meta"]
        score = chunk["score"]
        source = meta.get("source_name", "—")
        page = meta.get("page", "")
        headings = meta.get("headings", "")
        author = meta.get("author", "")
        embed_text = meta.get("embed_text", "")  # если сохранено в мета

        print(f"\n[{i}] score={score:.3f}  |  {author}  |  {source}  стр.{page}")
        if headings:
            print(f"     headings: {headings}")
        if embed_text:
            print(f"     embed_text: {embed_text[:120]}")

        text = chunk["text"]
        if args.text_len and len(text) > args.text_len:
            text = text[: args.text_len] + "…"
        print(f"     текст: {text}")

    print("\n" + "─" * 80)
    print(f"Итого: {len(chunks)} чанков")


if __name__ == "__main__":
    main()
