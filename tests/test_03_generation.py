"""
test_03_generation.py — тестирование генерации LLM: промпт → ответ

Запуск:
    python tests/test_03_generation.py --question-num 3
    python tests/test_03_generation.py --question-num 3 --no-llm
    python tests/test_03_generation.py --question-num 3 --author Гайворонский
    python tests/test_03_generation.py --query "атлант первый шейный позвонок"
    python tests/test_03_generation.py --question-num 3 --show-prompt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from conftest import ROOT, get_config, get_collection, get_embed_model

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich import box
from rich.table import Table

console = Console(width=140)


def load_questions(cfg: dict) -> list[dict]:
    """Load enriched questions using the same priority as 6_generate_answers_v2.py."""
    enrich_cfg = cfg.get("enrichment", {})
    llm_file = ROOT / enrich_cfg.get("llm_file", "")
    fallback = ROOT / enrich_cfg.get("output_file", "data/enriched_questions.json")

    for path in [llm_file, fallback]:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            # Normalize from ChatGPT format
            if data and "exam_num" in data[0]:
                normalized = []
                for i, item in enumerate(data, 1):
                    normalized.append({
                        "number": i,
                        "exam_num": item["exam_num"],
                        "question": item["exam_text"],
                        "sub_questions": item.get("sub_questions", []),
                    })
                return normalized
            return data
    raise FileNotFoundError("Enriched questions not found. Run step 5 or place LLM file.")


def retrieve_for_author(
    queries: list[str],
    author: str,
    cfg: dict,
    max_chunks: int,
    top_k_per_query: int,
) -> list[dict]:
    """Deduplicated retrieval for one author."""
    from utils import retrieve

    seen: set[str] = set()
    all_chunks: list[dict] = []
    override_cfg = {**cfg, "retrieval": {**cfg["retrieval"], "top_k": top_k_per_query}}

    for query in queries:
        chunks = retrieve(query, get_collection(), get_embed_model(), override_cfg, author_filter=author)
        for c in chunks:
            if c["text"] not in seen:
                seen.add(c["text"])
                all_chunks.append(c)

    all_chunks.sort(key=lambda c: c["score"], reverse=True)
    return all_chunks[:max_chunks]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generation tester")
    parser.add_argument("--question-num", "-n", type=int, default=3,
                        help="Номер вопроса из enriched_questions (1-based, default=3)")
    parser.add_argument("--query", "-q", help="Произвольный запрос вместо вопроса из файла")
    parser.add_argument("--author", "-a", help="Только один автор (иначе все primary)")
    parser.add_argument("--no-llm", action="store_true", help="Только показать контекст, без LLM")
    parser.add_argument("--show-prompt", action="store_true", help="Показать полный промпт")
    args = parser.parse_args()

    console.rule("[bold cyan]test_03_generation.py — тест генерации[/bold cyan]")

    cfg = get_config()
    authors_cfg = cfg.get("authors", [])

    # ── Определяем вопрос ─────────────────────────────────────────────────
    if args.query:
        question = {
            "number": 0,
            "exam_num": "custom",
            "question": args.query,
            "sub_questions": [],
        }
    else:
        with console.status("Загрузка вопросов…"):
            questions = load_questions(cfg)
        idx = args.question_num - 1
        if not 0 <= idx < len(questions):
            console.print(f"[red]Вопрос #{args.question_num} не найден (всего {len(questions)})[/red]")
            sys.exit(1)
        question = questions[idx]

    gen_cfg = cfg.get("generation_v2", {})
    max_chunks = gen_cfg.get("max_chunks_per_author", 5)
    top_k_per_query = gen_cfg.get("top_k_per_query", 2)

    console.print(Panel(
        f"[bold]{question.get('exam_num', '')}[/bold] {question['question']}",
        title=f"Вопрос #{question.get('number', '—')}",
        border_style="cyan",
    ))

    sub_qs = question.get("sub_questions", [])
    if sub_qs:
        console.print(f"  Подвопросы ({len(sub_qs)}):")
        for sq in sub_qs:
            console.print(f"    [dim]• {sq['text']}[/dim]")

    # ── Retrieval ─────────────────────────────────────────────────────────
    queries = [question["question"]] + [sq["text"] for sq in sub_qs]
    target_authors = [a["name"] for a in authors_cfg] if not args.author else [args.author]

    chunks_by_author: dict[str, list[dict]] = {}
    ctx_by_author: dict[str, str] = {}

    for author in target_authors:
        with console.status(f"Retrieval: {author}…"):
            chunks = retrieve_for_author(queries, author, cfg, max_chunks, top_k_per_query)
        chunks_by_author[author] = chunks

        # Show retrieval table
        tbl = Table(title=f"Найдено для '{author}': {len(chunks)} чанков", box=box.SIMPLE, show_lines=False)
        tbl.add_column("Score", justify="right", width=6)
        tbl.add_column("Headings", max_width=60)
        tbl.add_column("Chars", justify="right", width=6)
        for c in chunks:
            score = c["score"]
            color = "green" if score >= 0.5 else "yellow" if score >= 0.35 else "red"
            tbl.add_row(
                f"[{color}]{score:.3f}[/{color}]",
                c["meta"].get("headings", "")[:60],
                str(len(c["text"])),
            )
        console.print(tbl)

        from utils import format_context
        ctx = format_context(chunks)
        ctx_by_author[author] = ctx

    if args.no_llm:
        console.rule("[dim]--no-llm: пропускаем генерацию[/dim]", style="dim")
        if args.show_prompt:
            for author, ctx in ctx_by_author.items():
                console.print(Panel(ctx, title=f"Контекст: {author}", border_style="dim"))
        return

    # ── Build prompt ──────────────────────────────────────────────────────
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_v2", ROOT / "scripts" / "6_generate_answers_v2.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore

    system_prompt = mod.SYSTEM_PROMPT_V2
    user_prompt = mod.build_user_prompt(question, ctx_by_author)

    if args.show_prompt:
        console.print(Panel(system_prompt, title="SYSTEM PROMPT", border_style="yellow"))
        console.print(Panel(user_prompt, title="USER PROMPT", border_style="yellow"))

    console.print(f"\n  [dim]Промпт: {len(user_prompt):,} символов[/dim]")

    # ── LLM call ─────────────────────────────────────────────────────────
    console.rule(f"[bold]Генерация: {cfg['llm']['model']}[/bold]")
    t0 = time.perf_counter()
    with console.status(f"Ожидание ответа от {cfg['llm']['model']}…"):
        from utils import call_llm
        answer = call_llm(user_prompt, system_prompt, cfg)
    elapsed = time.perf_counter() - t0

    console.print(Panel(
        answer,
        title=f"Ответ LLM ({cfg['llm']['model']}, {elapsed:.1f}s)",
        border_style="green",
    ))

    # ── Stats ─────────────────────────────────────────────────────────────
    total_ctx_chars = sum(len(ctx) for ctx in ctx_by_author.values())
    console.print(
        f"  [dim]Контекст: {total_ctx_chars:,} символов  |  "
        f"Ответ: {len(answer):,} символов  |  "
        f"Время LLM: {elapsed:.1f}s[/dim]"
    )


if __name__ == "__main__":
    main()
