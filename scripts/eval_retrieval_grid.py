#!/usr/bin/env python3
"""
Сравнительный grid-тест стратегий retrieval.

Перебирает эксперименты из experiments/retrieval_experiments.yaml:
  • разные embed_strategy (heading_only / text_only / heading_plus_text)
  • HyDE vs обычный поиск
  • разные top_k

Не пересобирает ChromaDB — вычисляет эмбеддинги в памяти.
Кэширует эмбеддинги чанков между экспериментами с одинаковой стратегией.

Использование:
    python scripts/eval_retrieval_grid.py
    python scripts/eval_retrieval_grid.py --experiments heading_only text_only
    python scripts/eval_retrieval_grid.py --no-hyde
    python scripts/eval_retrieval_grid.py --query "строение позвонка"
    python scripts/eval_retrieval_grid.py --top-k 10
    python scripts/eval_retrieval_grid.py --config experiments/retrieval_experiments.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from rich import box as rich_box
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from utils import ROOT, get_device, load_config, _generate_hypothetical_answer

console = Console()


# ── Embed strategies ──────────────────────────────────────────────────────────

def get_embed_text(chunk: dict, strategy: str, text_chars: int = 0) -> str:
    """Сформировать строку для эмбеддинга из чанка по заданной стратегии.

    chunk["embed_text"] — заголовок раздела (выставлен при ингестии).
    chunk["text"]       — полный текст чанка.
    """
    heading = chunk.get("embed_text") or " > ".join(chunk.get("headings") or [])
    text = chunk.get("text", "")

    if strategy == "heading_only":
        # Текущий baseline: embed = заголовок раздела
        return heading or text[:200]

    elif strategy == "text_only":
        # Весь текст (или первые N символов если задано)
        return text if text_chars == 0 else text[:text_chars]

    elif strategy == "heading_plus_text":
        # Заголовок + текст (или первые N символов текста)
        snippet = text if text_chars == 0 else text[:text_chars]
        return f"{heading}. {snippet}" if heading else snippet

    else:
        raise ValueError(
            f"Unknown embed_strategy: {strategy!r}. "
            f"Доступные: heading_only, text_only, heading_plus_text"
        )


# ── Relevance ─────────────────────────────────────────────────────────────────

def is_relevant(chunk: dict, keywords: list[str]) -> bool:
    """Проверить релевантность чанка — хотя бы одно слово из keywords в тексте."""
    if not keywords:
        return False
    haystack = (
        chunk.get("text", "") + " " +
        (chunk.get("embed_text") or "") + " " +
        " ".join(chunk.get("headings") or [])
    ).lower()
    return any(kw.lower() in haystack for kw in keywords)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_chunks(cfg: dict) -> list[dict]:
    chunks_dir = ROOT / cfg["paths"]["chunks_dir"]
    chunks: list[dict] = []
    for path in sorted(chunks_dir.glob("*_chunks.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


# ── Encoding ──────────────────────────────────────────────────────────────────

_embed_cache: dict[tuple, np.ndarray] = {}


def get_chunk_embeddings(
    chunks: list[dict],
    embed_model,
    strategy: str,
    text_chars: int,
    batch_size: int = 128,
) -> np.ndarray:
    """Вернуть матрицу эмбеддингов всех чанков, кэшируя по (strategy, text_chars)."""
    key = (strategy, text_chars)
    if key not in _embed_cache:
        texts = [get_embed_text(c, strategy, text_chars) for c in chunks]
        console.print(f"  Кодирую {len(texts)} чанков (strategy={strategy}, chars={text_chars})…")
        t0 = time.perf_counter()
        embs = embed_model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        _embed_cache[key] = np.array(embs, dtype=np.float32)
        console.print(f"  Готово за {time.perf_counter() - t0:.1f}s → {_embed_cache[key].shape}")
    else:
        console.print(f"  [dim]Эмбеддинги из кэша (strategy={strategy}, chars={text_chars})[/dim]")
    return _embed_cache[key]


# ── In-memory retrieval ───────────────────────────────────────────────────────

def retrieve_inmemory(
    query_emb: np.ndarray,
    chunk_embs: np.ndarray,
    chunks: list[dict],
    top_k: int,
) -> list[dict]:
    """Cosine similarity (vectors already normalized) → top-k."""
    scores = chunk_embs @ query_emb
    idx = np.argsort(scores)[::-1][:top_k]
    return [{"chunk": chunks[i], "score": float(scores[i])} for i in idx]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-тест стратегий retrieval")
    parser.add_argument(
        "--config", default="experiments/retrieval_experiments.yaml",
        help="Путь к YAML-конфигу экспериментов (относительно корня проекта)",
    )
    parser.add_argument(
        "--experiments", nargs="*", metavar="NAME",
        help="Запустить только эти эксперименты (по полю name)",
    )
    parser.add_argument(
        "--query", metavar="TEXT",
        help="Запустить для одного произвольного запроса вместо queries из конфига",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Переопределить top_k из конфига",
    )
    parser.add_argument(
        "--no-hyde", action="store_true",
        help="Пропустить эксперименты с hyde: true",
    )
    parser.add_argument(
        "--show-chars", type=int, default=None,
        help="Сколько символов текста чанка показывать (переопределяет eval.show_text_chars)",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config
    if not config_path.exists():
        console.print(f"[red]Конфиг не найден: {config_path}[/red]")
        console.print(f"[yellow]Создай файл или укажи --config путь/к/конфигу.yaml[/yellow]")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    queries: list[dict] = eval_cfg.get("queries", [])
    if args.query:
        queries = [{"id": "cli", "text": args.query, "relevant_keywords": []}]

    experiments: list[dict] = eval_cfg.get("experiments", [])
    if args.experiments:
        experiments = [e for e in experiments if e["name"] in args.experiments]
    if args.no_hyde:
        experiments = [e for e in experiments if not e.get("hyde", False)]

    if not experiments:
        console.print("[red]Нет экспериментов для запуска.[/red]")
        sys.exit(1)

    top_k: int = args.top_k or eval_cfg.get("eval", {}).get("top_k", 7)
    show_chars: int = args.show_chars or eval_cfg.get("eval", {}).get("show_text_chars", 200)

    cfg = load_config()
    device = get_device(cfg)

    console.rule("[bold]eval_retrieval_grid[/bold]")
    console.print(
        f"  Запросов: [cyan]{len(queries)}[/cyan]  |  "
        f"Экспериментов: [cyan]{len(experiments)}[/cyan]  |  "
        f"top_k: [cyan]{top_k}[/cyan]"
    )

    # ── Load embedding model ──
    with console.status("Загружаю embedding модель…"):
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(cfg["embedding"]["model"], device=device)
    console.print(f"  Модель: [cyan]{cfg['embedding']['model']}[/cyan] на {device}")

    # ── Load chunks ──
    with console.status("Загружаю чанки…"):
        chunks = load_all_chunks(cfg)
    console.print(f"  Чанков всего: [cyan]{len(chunks)}[/cyan]")

    # ── Run experiments ──
    # Summary: exp_name -> query_id -> n_relevant
    summary: list[dict] = []

    for exp in experiments:
        name = exp["name"]
        desc = exp.get("description", "")
        strategy = exp["embed_strategy"]
        text_chars = exp.get("heading_plus_text_chars", 300)
        hyde = exp.get("hyde", False)

        console.rule(f"[bold cyan]{name}[/bold cyan]  —  {desc}")

        # Chunk embeddings (cached per strategy+chars)
        chunk_embs = get_chunk_embeddings(chunks, embed_model, strategy, text_chars)

        exp_summary: dict[str, int] = {}

        for q in queries:
            qid = q["id"]
            qtext = q["text"]
            keywords = q.get("relevant_keywords", [])

            if hyde:
                console.print(f"\n  [bold]HyDE:[/bold] генерирую ответ для «{qtext[:60]}»…")
                qtext_for_embed = _generate_hypothetical_answer(qtext, cfg)
                console.print(f"  → [dim]{qtext_for_embed[:120]}[/dim]")
            else:
                qtext_for_embed = qtext

            q_emb = embed_model.encode(
                [qtext_for_embed], normalize_embeddings=True
            )[0]

            hits = retrieve_inmemory(q_emb, chunk_embs, chunks, top_k)

            # Count relevant
            n_rel = sum(1 for h in hits if is_relevant(h["chunk"], keywords))
            exp_summary[qid] = n_rel

            # Print results for this query
            rel_color = "green" if n_rel > 0 else "red"
            console.print(
                f"\n  [bold]Запрос {qid}[/bold]: {qtext[:70]}\n"
                f"  Релевантных в top-{top_k}: [{rel_color}]{n_rel}/{top_k}[/{rel_color}]"
            )

            for rank, hit in enumerate(hits, 1):
                chunk = hit["chunk"]
                score = hit["score"]
                heading = (chunk.get("embed_text") or "").replace("\n", " ")[:60]
                snippet = chunk.get("text", "").replace("\n", " ")[:show_chars]
                author = chunk.get("author") or "—"
                rel = is_relevant(chunk, keywords)
                mark = "[green]✓[/green]" if rel else "[dim]·[/dim]"
                console.print(f"    {mark} [{rank:2d}] score={score:.3f}  {author}  «{heading}»")
                if show_chars > 0:
                    console.print(f"         [dim]{snippet}[/dim]")

        summary.append({
            "name": name,
            "strategy": strategy,
            "text_chars": text_chars,
            "hyde": hyde,
            "query_results": exp_summary,
        })

    # ── Summary table ──
    console.rule("[bold]Итог: релевантных чанков в top-K[/bold]")

    table = Table(box=rich_box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Эксперимент", style="bold")
    table.add_column("Стратегия", style="cyan")
    table.add_column("chars", justify="right")
    table.add_column("HyDE", justify="center")

    scored_queries = [q for q in queries if q.get("relevant_keywords")]
    for q in scored_queries:
        table.add_column(f"{q['id']}  (/{top_k})", justify="center")

    for row_data in summary:
        hyde_mark = "[yellow]да[/yellow]" if row_data["hyde"] else "нет"
        chars_str = str(row_data["text_chars"]) if row_data["strategy"] == "heading_plus_text" else "—"
        row = [
            row_data["name"],
            row_data["strategy"],
            chars_str,
            hyde_mark,
        ]
        for q in scored_queries:
            n = row_data["query_results"].get(q["id"], 0)
            color = "green" if n >= top_k // 2 else ("yellow" if n > 0 else "red")
            row.append(f"[{color}]{n}[/{color}]")
        table.add_row(*row)

    console.print(table)
    console.print(f"[dim]Ключевые слова для оценки релевантности: {[q.get('relevant_keywords') for q in scored_queries]}[/dim]")


if __name__ == "__main__":
    main()
