#!/usr/bin/env python3
"""
Шаг 3: Интерактивный CLI для вопросов к базе знаний.

Использование:
    python scripts/3_query.py
    python scripts/3_query.py --backend ollama --model llama3.1:8b
    python scripts/3_query.py --backend openai --model gpt-4o
    python scripts/3_query.py --top-k 8 --no-llm   # только показать найденный контекст
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from utils import ROOT, call_llm, format_context, get_device, load_config, retrieve

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты эксперт-анатом. Отвечай строго на основе предоставленного контекста из учебников "
    "по анатомии. Структурируй ответ: определение → строение → функции → особенности. "
    "При первом упоминании анатомического термина давай латинский эквивалент в скобках. "
    "Если информации в контексте недостаточно, прямо укажи на это."
)


def print_sources(chunks: list[dict]) -> None:
    print("\nИсточники:")
    seen: set[str] = set()
    for chunk in chunks:
        m = chunk["meta"]
        key = f"{m.get('source_name', '—')}_p{m.get('page', 0)}"
        if key in seen:
            continue
        seen.add(key)
        line = f"  • {m.get('source_name', '—')}"
        if m.get("page"):
            line += f", стр. {m['page']}"
        if m.get("headings"):
            line += f"  [{m['headings']}]"
        line += f"  (score: {chunk['score']:.2f})"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Интерактивный RAG-чат по анатомии")
    parser.add_argument("--backend", choices=["ollama", "openai"], help="LLM backend")
    parser.add_argument("--model", help="Название модели")
    parser.add_argument("--top-k", type=int, help="Кол-во чанков для поиска")
    parser.add_argument("--no-llm", action="store_true", help="Показать только найденный контекст без генерации")
    args = parser.parse_args()

    config = load_config()
    if args.top_k:
        config["retrieval"]["top_k"] = args.top_k

    device = get_device(config)
    print(f"[device={device}]")

    print(f"Загружаю embedding модель {config['embedding']['model']}...")
    embed_model = SentenceTransformer(config["embedding"]["model"], device=device)

    client = chromadb.PersistentClient(path=str(ROOT / config["paths"]["chroma_dir"]))
    try:
        collection = client.get_collection("anatomy")
    except Exception:
        print("ОШИБКА: Индекс не найден. Запустите 2_build_index.py")
        sys.exit(1)

    print(f"Индекс: {collection.count()} чанков\n")
    print("═" * 60)
    print("Анатомия RAG │ Введите вопрос  ('выход' для выхода)")
    print("═" * 60)

    while True:
        try:
            query = input("\nВопрос: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query or query.lower() in ("выход", "exit", "quit"):
            break

        chunks = retrieve(query, collection, embed_model, config)

        if not chunks:
            print("Релевантных фрагментов не найдено в базе знаний.")
            continue

        if args.no_llm:
            print("\n" + format_context(chunks))
            print_sources(chunks)
            continue

        context = format_context(chunks)
        prompt = (
            f"Вопрос: {query}\n\n"
            f"Контекст из учебников:\n{context}\n\n"
            "Дай развёрнутый академический ответ."
        )

        print("\nГенерирую ответ...")
        try:
            answer = call_llm(prompt, SYSTEM_PROMPT, config, args.backend, args.model)
            print(f"\n{answer}")
        except Exception as e:
            print(f"\nОшибка LLM: {e}")
            print("\nКонтекст из базы знаний:")
            print(format_context(chunks))

        print_sources(chunks)


if __name__ == "__main__":
    main()
