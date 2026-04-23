"""
test_01_chunks.py — статистика по чанкам из data/chunks/*.jsonl

Запуск:
    python tests/test_01_chunks.py
    python tests/test_01_chunks.py --author Гайворонский
    python tests/test_01_chunks.py --sample 5
    python tests/test_01_chunks.py --author Сапин --sample 3 --show-artifacts
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# Allow import from tests/
sys.path.insert(0, str(Path(__file__).parent))
from conftest import ROOT, get_config

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

# PDF-артефакты, которые должны были быть почищены
_ARTIFACTS = [
    "/hyphenminus", "/hyphen", "/minus", "/fi", "/fl", "/ff",
    "/endash", "/emdash", "/bullet", "/period", "/comma",
]


def load_chunks(chunks_dir: Path, author_filter: str | None) -> list[dict]:
    chunks = []
    for jl in sorted(chunks_dir.glob("*.jsonl")):
        with open(jl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                if author_filter and c.get("author", "") != author_filter:
                    continue
                chunks.append(c)
    return chunks


def chunk_stats(chunks: list[dict]) -> dict:
    by_author: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_author[c.get("author", "(нет автора)")].append(c)
    return dict(by_author)


def artifact_count(text: str) -> int:
    return sum(text.count(a) for a in _ARTIFACTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk statistics")
    parser.add_argument("--author", help="Фильтр по автору")
    parser.add_argument("--sample", type=int, default=0, metavar="N",
                        help="Показать N случайных чанков полностью")
    parser.add_argument("--show-artifacts", action="store_true",
                        help="Показать чанки с PDF-артефактами")
    args = parser.parse_args()

    cfg = get_config()
    chunks_dir = ROOT / cfg["paths"]["chunks_dir"]

    console.rule("[bold cyan]test_01_chunks.py — статистика чанков")
    console.print(f"Директория: [dim]{chunks_dir}[/dim]")

    with console.status("Загрузка чанков…"):
        all_chunks = load_chunks(chunks_dir, args.author)

    if not all_chunks:
        console.print("[red]Чанки не найдены. Запусти: python scripts/1_ingest.py")
        sys.exit(1)

    console.print(f"Всего чанков: [bold]{len(all_chunks):,}[/bold]\n")

    by_author = chunk_stats(all_chunks)

    # ── Таблица по авторам ──────────────────────────────────────────────────
    tbl = Table(title="Статистика по авторам", box=box.ROUNDED, show_lines=True)
    tbl.add_column("Автор", style="cyan")
    tbl.add_column("Чанков", justify="right")
    tbl.add_column("Ср. длина", justify="right")
    tbl.add_column("Мин", justify="right")
    tbl.add_column("Макс", justify="right")
    tbl.add_column("С рисунками", justify="right")
    tbl.add_column("Артефакты", justify="right")

    for author, chunks in sorted(by_author.items(), key=lambda x: x[0] or ""):
        lengths = [len(c["text"]) for c in chunks]
        n_figs = sum(1 for c in chunks if c.get("figures"))
        n_art = sum(1 for c in chunks if artifact_count(c["text"]) > 0)
        tbl.add_row(
            author,
            f"{len(chunks):,}",
            f"{sum(lengths) // len(lengths):,}",
            f"{min(lengths):,}",
            f"{max(lengths):,}",
            f"{n_figs} ({100 * n_figs // len(chunks)}%)",
            f"[red]{n_art}[/red]" if n_art else "[green]0[/green]",
        )

    console.print(tbl)

    # ── Распределение по длинам ─────────────────────────────────────────────
    lengths = [len(c["text"]) for c in all_chunks]
    buckets = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 99999)]
    dist_tbl = Table(title="Распределение по длине текста (символов)", box=box.SIMPLE)
    dist_tbl.add_column("Диапазон", style="dim")
    dist_tbl.add_column("Кол-во", justify="right")
    dist_tbl.add_column("Доля", justify="right")
    for lo, hi in buckets:
        cnt = sum(lo <= l < hi for l in lengths)
        label = f"{lo:,}–{hi:,}" if hi < 99999 else f"{lo:,}+"
        dist_tbl.add_row(label, f"{cnt:,}", f"{100 * cnt // len(lengths)}%")
    console.print(dist_tbl)

    # ── Топ-10 самых длинных заголовков ────────────────────────────────────
    heading_chunks = [(c, len(c["text"])) for c in all_chunks if c.get("headings")]
    heading_chunks.sort(key=lambda x: x[1], reverse=True)
    top_tbl = Table(title="Топ-10 самых длинных чанков", box=box.SIMPLE)
    top_tbl.add_column("Автор", style="cyan", max_width=15)
    top_tbl.add_column("Длина", justify="right")
    top_tbl.add_column("Заголовок")
    for c, ln in heading_chunks[:10]:
        h = " > ".join(c["headings"])[:80]
        top_tbl.add_row(c.get("author", "—"), f"{ln:,}", h)
    console.print(top_tbl)

    # ── Артефакты ───────────────────────────────────────────────────────────
    if args.show_artifacts:
        art_chunks = [c for c in all_chunks if artifact_count(c["text"]) > 0]
        console.print(f"\n[yellow]Чанки с PDF-артефактами: {len(art_chunks)}[/yellow]")
        for c in art_chunks[:10]:
            h = " > ".join(c.get("headings", []))[:60]
            found = [a for a in _ARTIFACTS if a in c["text"]]
            console.print(f"  [{c.get('author','?')}] {h} → {found}")

    # ── Случайные примеры ───────────────────────────────────────────────────
    if args.sample > 0:
        sample = random.sample(all_chunks, min(args.sample, len(all_chunks)))
        for i, c in enumerate(sample, 1):
            h = " > ".join(c.get("headings", ["(нет заголовка)"]))
            figs = c.get("figures") or []
            console.print(
                Panel(
                    c["text"],
                    title=f"[{i}/{args.sample}] {c.get('author','?')} | {h[:70]}",
                    subtitle=f"len={len(c['text'])} | figures={len(figs)}",
                    border_style="dim",
                )
            )


if __name__ == "__main__":
    main()
