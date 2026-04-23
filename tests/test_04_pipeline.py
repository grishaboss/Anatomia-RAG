"""
test_04_pipeline.py — полный пайплайн для одного вопроса, шаг за шагом

Запуск:
    python tests/test_04_pipeline.py --question-num 3
    python tests/test_04_pipeline.py --question-num 3 --no-llm
    python tests/test_04_pipeline.py --question-num 3 --save
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from conftest import ROOT, get_config, get_collection, get_embed_model

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

console = Console(width=140)

STEP_STYLE = "bold magenta"


def step(n: int, title: str) -> None:
    console.rule(f"[{STEP_STYLE}]Шаг {n}: {title}[/{STEP_STYLE}]")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("_mod", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline step-by-step")
    parser.add_argument("--question-num", "-n", type=int, default=3,
                        help="Номер вопроса (1-based)")
    parser.add_argument("--no-llm", action="store_true", help="Пропустить шаг LLM")
    parser.add_argument("--save", action="store_true", help="Сохранить результат в tests/results/")
    args = parser.parse_args()

    console.rule("[bold cyan]test_04_pipeline.py — полный пайплайн[/bold cyan]")

    # ─────────────────────────────────────────────────────────────────────
    # ШАГ 1: Конфигурация
    # ─────────────────────────────────────────────────────────────────────
    step(1, "Конфигурация")
    cfg = get_config()
    gen_cfg = cfg.get("generation_v2", {})
    ret_cfg = cfg["retrieval"]
    llm_cfg = cfg["llm"]

    cfg_tbl = Table(box=box.SIMPLE, show_header=False)
    cfg_tbl.add_column(style="dim")
    cfg_tbl.add_column(style="bold")
    cfg_tbl.add_row("LLM model", llm_cfg["model"])
    cfg_tbl.add_row("LLM backend", llm_cfg["backend"])
    cfg_tbl.add_row("temperature", str(llm_cfg["temperature"]))
    cfg_tbl.add_row("top_k", str(ret_cfg["top_k"]))
    cfg_tbl.add_row("score_threshold", str(ret_cfg["score_threshold"]))
    cfg_tbl.add_row("HyDE", str(ret_cfg.get("hyde", False)))
    cfg_tbl.add_row("max_chunks_per_author", str(gen_cfg.get("max_chunks_per_author", 5)))
    cfg_tbl.add_row("top_k_per_query", str(gen_cfg.get("top_k_per_query", 2)))
    console.print(cfg_tbl)

    # ─────────────────────────────────────────────────────────────────────
    # ШАГ 2: Загрузка вопроса
    # ─────────────────────────────────────────────────────────────────────
    step(2, "Загрузка вопроса")
    enrich_cfg = cfg.get("enrichment", {})
    llm_file = ROOT / enrich_cfg.get("llm_file", "")
    fallback = ROOT / enrich_cfg.get("output_file", "data/enriched_questions.json")

    for path in [llm_file, fallback]:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            if raw and "exam_num" in raw[0]:
                questions = []
                for i, item in enumerate(raw, 1):
                    questions.append({
                        "number": i,
                        "exam_num": item["exam_num"],
                        "question": item["exam_text"],
                        "sub_questions": item.get("sub_questions", []),
                    })
            else:
                questions = raw
            console.print(f"  Источник: [dim]{path.name}[/dim] ({len(questions)} вопросов)")
            break
    else:
        console.print("[red]Файл вопросов не найден.[/red]")
        sys.exit(1)

    idx = args.question_num - 1
    if not 0 <= idx < len(questions):
        console.print(f"[red]Вопрос #{args.question_num} не найден.[/red]")
        sys.exit(1)

    question = questions[idx]
    console.print(Panel(
        f"[bold]{question.get('exam_num', '')}[/bold]  {question['question']}",
        title=f"Вопрос #{question['number']}",
        border_style="cyan",
    ))
    sub_qs = question.get("sub_questions", [])
    if sub_qs:
        console.print(f"  Подвопросы: {len(sub_qs)}")
        for sq in sub_qs:
            console.print(f"    [dim]• {sq['text']}[/dim]")

    # ─────────────────────────────────────────────────────────────────────
    # ШАГ 3: Формирование запросов
    # ─────────────────────────────────────────────────────────────────────
    step(3, "Формирование запросов")
    queries = [question["question"]] + [sq["text"] for sq in sub_qs]
    console.print(f"  Всего запросов: {len(queries)}")
    for i, q in enumerate(queries):
        prefix = "[bold]главный[/bold]" if i == 0 else f"[dim]подвопрос {i}[/dim]"
        console.print(f"  {prefix}: {q[:100]}")

    # ─────────────────────────────────────────────────────────────────────
    # ШАГ 4: Retrieval по каждому автору
    # ─────────────────────────────────────────────────────────────────────
    step(4, "Retrieval")
    from utils import retrieve, format_context

    authors_cfg = cfg.get("authors", [])
    max_chunks = gen_cfg.get("max_chunks_per_author", 5)
    top_k_per_query = gen_cfg.get("top_k_per_query", 2)
    override_cfg = {**cfg, "retrieval": {**ret_cfg, "top_k": top_k_per_query}}

    chunks_by_author: dict[str, list[dict]] = {}
    ctx_by_author: dict[str, str] = {}
    t_retrieval_total = 0.0

    for author_info in authors_cfg:
        author = author_info["name"]
        t0 = time.perf_counter()

        seen: set[str] = set()
        all_chunks: list[dict] = []
        for query in queries:
            cs = retrieve(query, get_collection(), get_embed_model(), override_cfg, author_filter=author)
            for c in cs:
                if c["text"] not in seen:
                    seen.add(c["text"])
                    all_chunks.append(c)

        all_chunks.sort(key=lambda c: c["score"], reverse=True)
        all_chunks = all_chunks[:max_chunks]
        elapsed = time.perf_counter() - t0
        t_retrieval_total += elapsed

        chunks_by_author[author] = all_chunks
        ctx = format_context(all_chunks)
        ctx_by_author[author] = ctx

        tbl = Table(
            title=f"{author}: {len(all_chunks)} чанков ({elapsed:.2f}s)",
            box=box.SIMPLE,
        )
        tbl.add_column("Score", justify="right", width=6)
        tbl.add_column("Page", justify="right", width=5)
        tbl.add_column("Headings", max_width=55)
        tbl.add_column("Chars", justify="right", width=6)
        for c in all_chunks:
            score = c["score"]
            color = "green" if score >= 0.5 else "yellow" if score >= 0.35 else "red"
            tbl.add_row(
                f"[{color}]{score:.3f}[/{color}]",
                str(c["meta"].get("page") or "—"),
                c["meta"].get("headings", "")[:55],
                str(len(c["text"])),
            )
        console.print(tbl)

    console.print(f"  [dim]Retrieval total: {t_retrieval_total:.2f}s[/dim]")

    # ─────────────────────────────────────────────────────────────────────
    # ШАГ 5: Формирование промпта
    # ─────────────────────────────────────────────────────────────────────
    step(5, "Формирование промпта")
    mod = load_module(ROOT / "scripts" / "6_generate_answers_v2.py")
    system_prompt = mod.SYSTEM_PROMPT_V2
    user_prompt = mod.build_user_prompt(question, ctx_by_author)

    total_ctx = sum(len(c) for c in ctx_by_author.values())
    console.print(f"  System prompt: {len(system_prompt):,} символов")
    console.print(f"  User prompt:   {len(user_prompt):,} символов  (контекст: {total_ctx:,})")
    console.print(
        Panel(
            system_prompt,
            title="System Prompt",
            border_style="yellow",
            expand=False,
        )
    )

    if args.no_llm:
        console.print(Panel(user_prompt, title="User Prompt (full)", border_style="dim"))
        console.rule("[dim]--no-llm: пропускаем генерацию[/dim]", style="dim")
        return

    # ─────────────────────────────────────────────────────────────────────
    # ШАГ 6: LLM-генерация
    # ─────────────────────────────────────────────────────────────────────
    step(6, f"LLM генерация ({llm_cfg['model']})")
    from utils import call_llm

    t0 = time.perf_counter()
    with console.status(f"Ожидание {llm_cfg['model']}…"):
        answer = call_llm(user_prompt, system_prompt, cfg)
    t_llm = time.perf_counter() - t0

    console.print(Panel(
        answer,
        title=f"Ответ LLM ({llm_cfg['model']}, {t_llm:.1f}s)",
        border_style="green",
    ))

    # ─────────────────────────────────────────────────────────────────────
    # ШАГ 7: Итог
    # ─────────────────────────────────────────────────────────────────────
    step(7, "Итог")
    summary_tbl = Table(box=box.SIMPLE, show_header=False)
    summary_tbl.add_column(style="dim")
    summary_tbl.add_column(style="bold")
    summary_tbl.add_row("Retrieval time", f"{t_retrieval_total:.2f}s")
    summary_tbl.add_row("LLM time", f"{t_llm:.1f}s")
    summary_tbl.add_row("Total time", f"{t_retrieval_total + t_llm:.1f}s")
    summary_tbl.add_row("Context chars", f"{total_ctx:,}")
    summary_tbl.add_row("Answer chars", f"{len(answer):,}")
    console.print(summary_tbl)

    if args.save:
        out_dir = ROOT / "tests" / "results"
        out_dir.mkdir(exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"pipeline_q{args.question_num}_{ts}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "question": question,
                    "chunks_by_author": {
                        a: [{"score": c["score"], "headings": c["meta"].get("headings", ""), "chars": len(c["text"])}
                            for c in cs]
                        for a, cs in chunks_by_author.items()
                    },
                    "answer": answer,
                    "t_retrieval": t_retrieval_total,
                    "t_llm": t_llm,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        console.print(f"[dim]Сохранено: {out}[/dim]")


if __name__ == "__main__":
    main()
