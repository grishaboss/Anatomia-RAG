"""
test_02_retrieval.py — тестирование и сравнение retrieval-стратегий

Запуск:
    # 5 дефолтных запросов
    python tests/test_02_retrieval.py

    # Произвольный запрос
    python tests/test_02_retrieval.py --query "Рёбра и грудина, их соединения. Мышцы груди. Васкуляризация и иннервация стенок грудной полости."

    # С фильтром по автору
    python tests/test_02_retrieval.py --query "позвонок" --author Гайворонский

    # Сравнение baseline vs HyDE
    python tests/test_02_retrieval.py --query "строение позвонка" --compare-hyde

    # Матрица top_k × threshold
    python tests/test_02_retrieval.py --query "строение позвонка" --compare-params

    # Сравнение: embed по заголовку vs по полному тексту
    python tests/test_02_retrieval.py --query "строение позвонка" --compare-embed-mode

    # Показать полный текст чанков (не только превью)
    python tests/test_02_retrieval.py --query "позвонок" --full-text
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from conftest import ROOT, get_config, get_collection, get_embed_model

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console(width=160)

# ── Дефолтные запросы для smoke-теста ──────────────────────────────────────
DEFAULT_QUERIES = [
    "Какие части выделяют у типичного позвонка?",
    "В чём особенности шейных позвонков?",
    "Строение атланта — первого шейного позвонка",
    "Особенности поясничных позвонков",
    "Соединения позвонков — межпозвоночный диск",
]


# ── Core helpers ────────────────────────────────────────────────────────────

def retrieve_raw(
    query: str,
    top_k: int,
    threshold: float,
    author_filter: str | None = None,
    embed_text: str | None = None,  # override what gets embedded
    hyde: bool = False,
) -> tuple[list[dict], float]:
    """
    Returns (chunks, elapsed_sec).
    chunks: [{"text", "meta", "score"}]
    """
    cfg = get_config()
    embed_model = get_embed_model()
    collection = get_collection()

    t0 = time.perf_counter()

    if hyde:
        from utils import _generate_hypothetical_answer
        embed_text = _generate_hypothetical_answer(query, cfg)

    text_to_embed = embed_text if embed_text is not None else query
    embedding = embed_model.encode([text_to_embed], normalize_embeddings=True)[0].tolist()

    kwargs: dict = dict(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if author_filter:
        kwargs["where"] = {"author": author_filter}

    results = collection.query(**kwargs)
    elapsed = time.perf_counter() - t0

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1.0 - float(dist)
        if score >= threshold:
            chunks.append({"text": doc, "meta": meta, "score": score})

    return chunks, elapsed


def make_results_table(
    chunks: list[dict],
    title: str,
    full_text: bool = False,
    preview_len: int = 120,
) -> Table:
    tbl = Table(
        title=title,
        box=box.ROUNDED,
        show_lines=True,
        expand=False,
        min_width=60,
    )
    tbl.add_column("#", style="dim", width=3)
    tbl.add_column("Score", justify="right", width=6)
    tbl.add_column("Author", style="cyan", max_width=14)
    tbl.add_column("Headings", max_width=35)
    tbl.add_column("Text preview" if not full_text else "Full text", max_width=80 if not full_text else 120)

    for i, c in enumerate(chunks, 1):
        score = c["score"]
        score_color = "green" if score >= 0.5 else "yellow" if score >= 0.35 else "red"
        meta = c["meta"]
        headings = meta.get("headings", "")
        text = c["text"]
        preview = text if full_text else (text[:preview_len] + "…" if len(text) > preview_len else text)
        # clean newlines for table display
        preview = preview.replace("\n", " ")

        tbl.add_row(
            str(i),
            f"[{score_color}]{score:.3f}[/{score_color}]",
            meta.get("author", "—"),
            headings[:60],
            preview,
        )

    if not chunks:
        tbl.add_row("—", "—", "—", "—", "[dim](нет результатов)[/dim]")

    return tbl


def save_results(data: dict, label: str) -> Path:
    results_dir = ROOT / "tests" / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"retrieval_{label}_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out


# ── Modes ───────────────────────────────────────────────────────────────────

def run_single(query: str, cfg: dict, author: str | None, full_text: bool) -> list[dict]:
    top_k = cfg["retrieval"]["top_k"]
    threshold = cfg["retrieval"]["score_threshold"]
    chunks, elapsed = retrieve_raw(query, top_k, threshold, author_filter=author)

    tbl = make_results_table(
        chunks,
        title=f"Query: \"{query[:70]}\"  |  top_k={top_k}  threshold={threshold}",
        full_text=full_text,
    )
    console.print(tbl)
    console.print(
        f"  [dim]Найдено: {len(chunks)} чанков  |  время: {elapsed:.2f}s"
        + (f"  |  автор: {author}" if author else "")
        + "[/dim]"
    )
    return chunks


def run_compare_hyde(query: str, cfg: dict, author: str | None) -> None:
    top_k = cfg["retrieval"]["top_k"]
    threshold = cfg["retrieval"]["score_threshold"]

    console.rule(f"[bold]Сравнение: Baseline vs HyDE[/bold]  |  query: \"{query[:60]}\"")

    with console.status("Baseline retrieval…"):
        chunks_base, t_base = retrieve_raw(query, top_k, threshold, author_filter=author)
    with console.status("HyDE: генерирую гипотетический ответ…"):
        chunks_hyde, t_hyde = retrieve_raw(query, top_k, threshold, author_filter=author, hyde=True)

    tbl_base = make_results_table(chunks_base, f"BASELINE  ({t_base:.2f}s)")
    tbl_hyde = make_results_table(chunks_hyde, f"HyDE  ({t_hyde:.2f}s)")

    console.print(Columns([tbl_base, tbl_hyde], expand=True))

    # diff
    base_headings = {c["meta"].get("headings", "") for c in chunks_base}
    hyde_headings = {c["meta"].get("headings", "") for c in chunks_hyde}
    only_base = base_headings - hyde_headings
    only_hyde = hyde_headings - base_headings

    if only_base or only_hyde:
        console.print("\n[yellow]Различия в найденных чанках:[/yellow]")
        for h in only_base:
            console.print(f"  [red]- только Baseline:[/red] {h[:80]}")
        for h in only_hyde:
            console.print(f"  [green]+ только HyDE:[/green] {h[:80]}")
    else:
        console.print("\n[green]Оба режима вернули одинаковый набор чанков.[/green]")


def run_compare_params(query: str, author: str | None) -> None:
    top_ks = [3, 5, 7, 10]
    thresholds = [0.15, 0.20, 0.25, 0.30]

    console.rule(f"[bold]Матрица параметров[/bold]  |  query: \"{query[:60]}\"")

    tbl = Table(title="top_k × threshold → (чанков, avg_score)", box=box.ROUNDED)
    tbl.add_column("top_k \\ threshold", style="bold")
    for th in thresholds:
        tbl.add_column(f"th={th}", justify="center")

    records = []
    for tk in top_ks:
        row_vals = [str(tk)]
        for th in thresholds:
            with console.status(f"top_k={tk}, threshold={th}…"):
                chunks, elapsed = retrieve_raw(query, tk, th, author_filter=author)
            n = len(chunks)
            avg = sum(c["score"] for c in chunks) / n if n else 0.0
            color = "green" if n >= 3 else "yellow" if n >= 1 else "red"
            row_vals.append(f"[{color}]{n}[/{color}] ({avg:.3f})")
            records.append({"top_k": tk, "threshold": th, "n_chunks": n, "avg_score": avg, "elapsed": elapsed})
        tbl.add_row(*row_vals)

    console.print(tbl)
    out = save_results({"query": query, "matrix": records}, "params_matrix")
    console.print(f"[dim]Результаты сохранены: {out}[/dim]")


def run_compare_embed_mode(query: str, cfg: dict, author: str | None) -> None:
    """
    Сравнивает два способа эмбеддинга:
    - 'heading': ChromaDB содержит embed_text = только заголовок (текущий режим)
    - 'full_query': запрос остаётся как есть (текущий режим)
    Показывает разницу при использовании query = полный вопрос vs query = короткий термин.
    """
    top_k = cfg["retrieval"]["top_k"]
    threshold = cfg["retrieval"]["score_threshold"]

    console.rule("[bold]Сравнение: короткий запрос vs развёрнутый вопрос[/bold]")

    short_query = query.split()[0] if query else query  # первое слово
    long_query = query

    for label, q in [("Короткий запрос", short_query), ("Полный вопрос", long_query)]:
        chunks, elapsed = retrieve_raw(q, top_k, threshold, author_filter=author)
        tbl = make_results_table(chunks, f"{label}: \"{q[:60]}\"  ({elapsed:.2f}s)")
        console.print(tbl)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval tester")
    parser.add_argument("--query", "-q", help="Поисковый запрос")
    parser.add_argument("--author", "-a", help="Фильтр по автору (Гайворонский/Сапин/Якимов)")
    parser.add_argument("--compare-hyde", action="store_true", help="Сравнить baseline vs HyDE")
    parser.add_argument("--compare-params", action="store_true", help="Матрица top_k × threshold")
    parser.add_argument("--compare-embed-mode", action="store_true",
                        help="Сравнить короткий vs длинный запрос")
    parser.add_argument("--full-text", action="store_true", help="Показать полный текст чанков")
    parser.add_argument("--save", action="store_true", help="Сохранить результаты в tests/results/")
    args = parser.parse_args()

    console.rule("[bold cyan]test_02_retrieval.py[/bold cyan]")
    cfg = get_config()
    console.print(
        f"  Model: [bold]{cfg['llm']['model']}[/bold]  |  "
        f"top_k={cfg['retrieval']['top_k']}  |  "
        f"threshold={cfg['retrieval']['score_threshold']}  |  "
        f"HyDE={'on' if cfg['retrieval'].get('hyde') else 'off'}"
    )

    if args.compare_hyde:
        query = args.query or DEFAULT_QUERIES[0]
        run_compare_hyde(query, cfg, args.author)

    elif args.compare_params:
        query = args.query or DEFAULT_QUERIES[0]
        run_compare_params(query, args.author)

    elif args.compare_embed_mode:
        query = args.query or DEFAULT_QUERIES[0]
        run_compare_embed_mode(query, cfg, args.author)

    elif args.query:
        chunks = run_single(args.query, cfg, args.author, args.full_text)
        if args.save:
            out = save_results(
                {"query": args.query, "chunks": [{"score": c["score"], "headings": c["meta"].get("headings", "")} for c in chunks]},
                "single",
            )
            console.print(f"[dim]Сохранено: {out}[/dim]")

    else:
        # Smoke test: 5 дефолтных запросов
        console.print("[bold]Smoke test: 5 дефолтных запросов[/bold]\n")
        summary = []
        for q in DEFAULT_QUERIES:
            console.rule(f"[dim]{q[:80]}[/dim]", style="dim")
            top_k = cfg["retrieval"]["top_k"]
            threshold = cfg["retrieval"]["score_threshold"]
            chunks, elapsed = retrieve_raw(q, top_k, threshold)
            tbl = make_results_table(chunks, f"\"{q[:60]}\"", full_text=False)
            console.print(tbl)
            avg_score = sum(c["score"] for c in chunks) / len(chunks) if chunks else 0.0
            console.print(f"  [dim]Найдено: {len(chunks)} | avg_score: {avg_score:.3f} | {elapsed:.2f}s[/dim]\n")
            summary.append({"query": q, "n_chunks": len(chunks), "avg_score": avg_score, "elapsed": elapsed})

        # Summary table
        sum_tbl = Table(title="Сводка smoke-теста", box=box.SIMPLE)
        sum_tbl.add_column("Запрос", max_width=60)
        sum_tbl.add_column("Чанков", justify="right")
        sum_tbl.add_column("Avg score", justify="right")
        sum_tbl.add_column("Время", justify="right")
        for s in summary:
            color = "green" if s["avg_score"] >= 0.4 else "yellow" if s["avg_score"] >= 0.3 else "red"
            sum_tbl.add_row(
                s["query"][:60],
                str(s["n_chunks"]),
                f"[{color}]{s['avg_score']:.3f}[/{color}]",
                f"{s['elapsed']:.2f}s",
            )
        console.print(sum_tbl)

        if args.save:
            out = save_results({"queries": summary}, "smoke")
            console.print(f"[dim]Сохранено: {out}[/dim]")


if __name__ == "__main__":
    main()
