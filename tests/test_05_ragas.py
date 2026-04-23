"""
test_05_ragas.py — RAGAS-оценка RAG-пайплайна

Метрики:
  - LLMContextPrecisionWithReference  — насколько возвращённые чанки релевантны
  - LLMContextRecall                  — всё ли нужное попало в контекст
  - Faithfulness                      — модель не добавляет информацию вне контекста
  - ResponseRelevancy                 — ответ раскрывает вопрос

Требования:
  - Заполни tests/ground_truth.json (поле "reference") для Precision/Recall
  - Без reference работают только Faithfulness + ResponseRelevancy

Запуск:
    python tests/test_05_ragas.py
    python tests/test_05_ragas.py --ids I.3.1 I.3.2
    python tests/test_05_ragas.py --no-reference          # только Faithfulness + Relevancy
    python tests/test_05_ragas.py --no-llm-answer         # только метрики на контексте
    python tests/test_05_ragas.py --save
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

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console(width=140)


def load_ground_truth(ids: list[str] | None = None) -> list[dict]:
    path = Path(__file__).parent / "ground_truth.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    return questions


def retrieve_context(question_text: str, author_filter: str | None, cfg: dict) -> tuple[list[str], list[dict]]:
    """Returns (context_strings, raw_chunks)."""
    from utils import retrieve, format_context

    top_k = cfg["retrieval"]["top_k"]
    threshold = cfg["retrieval"]["score_threshold"]
    chunks = retrieve(
        question_text,
        get_collection(),
        get_embed_model(),
        cfg,
        author_filter=author_filter,
    )
    return [c["text"] for c in chunks], chunks


def generate_answer(question_text: str, contexts: list[str], cfg: dict) -> str:
    from utils import call_llm
    import importlib.util

    mod_path = ROOT / "scripts" / "6_generate_answers_v2.py"
    spec = importlib.util.spec_from_file_location("gen_v2", mod_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore

    ctx_block = "\n\n---\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    pseudo_question = {"question": question_text, "sub_questions": []}
    pseudo_ctx = {"Контекст": ctx_block}

    user_prompt = mod.build_user_prompt(pseudo_question, pseudo_ctx)
    return call_llm(user_prompt, mod.SYSTEM_PROMPT_V2, cfg)


def build_ragas_llm():
    """Build LangChain-wrapped Ollama LLM for RAGAS."""
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    cfg = get_config()
    model = cfg["llm"]["model"]
    base_url = cfg["llm"].get("api_base", "http://localhost:11434")
    console.print(f"  [dim]RAGAS judge LLM: {model} @ {base_url}[/dim]")
    llm = ChatOllama(model=model, temperature=0, base_url=base_url)
    return LangchainLLMWrapper(llm)


def build_ragas_embeddings():
    """Build LangChain-wrapped HuggingFace embeddings for RAGAS."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from utils import get_device

    cfg = get_config()
    model_name = cfg["embedding"]["model"]
    device = get_device(cfg)
    console.print(f"  [dim]RAGAS embed: {model_name} on {device}[/dim]")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation")
    parser.add_argument("--ids", nargs="+", metavar="ID",
                        help="ID из ground_truth.json (напр. I.3.1 I.3.2)")
    parser.add_argument("--no-reference", action="store_true",
                        help="Использовать только Faithfulness + ResponseRelevancy (без эталонов)")
    parser.add_argument("--no-llm-answer", action="store_true",
                        help="Не генерировать ответы LLM (только контекстные метрики)")
    parser.add_argument("--save", action="store_true",
                        help="Сохранить отчёт в tests/results/ragas_TIMESTAMP.json")
    args = parser.parse_args()

    console.rule("[bold cyan]test_05_ragas.py — RAGAS evaluation[/bold cyan]")

    cfg = get_config()
    questions = load_ground_truth(args.ids)

    if not questions:
        console.print("[red]Нет вопросов для оценки.[/red]")
        sys.exit(1)

    has_reference = any(q.get("reference", "").strip() for q in questions) and not args.no_reference

    console.print(f"  Вопросов: {len(questions)}")
    console.print(f"  Эталонные ответы: {'да' if has_reference else 'нет'}")
    console.print(f"  Генерировать ответы LLM: {'нет' if args.no_llm_answer else 'да'}")

    # ── Подготовка RAGAS ──────────────────────────────────────────────────
    with console.status("Инициализация RAGAS…"):
        ragas_llm = build_ragas_llm()
        ragas_emb = build_ragas_embeddings()

    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    # Выбираем метрики
    try:
        try:
            from ragas.metrics.collections import (
                LLMContextPrecisionWithReference,
                LLMContextRecall,
                Faithfulness,
                ResponseRelevancy,
            )
        except ImportError:
            from ragas.metrics import (  # type: ignore[no-redef]
                LLMContextPrecisionWithReference,
                LLMContextRecall,
                Faithfulness,
                ResponseRelevancy,
            )
        metrics_with_ref = [
            LLMContextPrecisionWithReference(llm=ragas_llm),
            LLMContextRecall(llm=ragas_llm),
            Faithfulness(llm=ragas_llm),
            ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ]
        metrics_no_ref = [
            Faithfulness(llm=ragas_llm),
            ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ]
    except ImportError as e:
        console.print(f"[red]Ошибка импорта RAGAS: {e}[/red]")
        console.print("[yellow]Установи: pip install ragas langchain-ollama langchain-huggingface[/yellow]")
        sys.exit(1)

    metrics = metrics_with_ref if has_reference else metrics_no_ref
    console.print(f"  Метрики: {[m.__class__.__name__ for m in metrics]}")

    # ── Сборка датасета ───────────────────────────────────────────────────
    samples: list[SingleTurnSample] = []
    raw_records = []

    for q in questions:
        qid = q["id"]
        question_text = q["question"]
        reference = q.get("reference", "").strip() or None
        author_filter = q.get("author_filter")

        console.rule(f"[dim]{qid}: {question_text[:70]}[/dim]", style="dim")

        # Retrieval
        with console.status(f"Retrieval: {qid}…"):
            t0 = time.perf_counter()
            contexts, raw_chunks = retrieve_context(question_text, author_filter, cfg)
            t_ret = time.perf_counter() - t0

        console.print(f"  Найдено чанков: {len(contexts)} ({t_ret:.2f}s)")
        for i, c in enumerate(raw_chunks):
            score = c["score"]
            color = "green" if score >= 0.5 else "yellow" if score >= 0.35 else "red"
            console.print(
                f"    [{color}]{score:.3f}[/{color}] {c['meta'].get('headings','')[:60]}"
            )

        # LLM answer
        if args.no_llm_answer:
            answer = "N/A"
        else:
            with console.status(f"LLM answer: {qid}…"):
                t0 = time.perf_counter()
                answer = generate_answer(question_text, contexts, cfg)
                t_llm = time.perf_counter() - t0
            console.print(f"  LLM ({t_llm:.1f}s): {answer[:150]}…" if len(answer) > 150 else f"  LLM: {answer}")

        sample_kwargs = dict(
            user_input=question_text,
            retrieved_contexts=contexts,
            response=answer,
        )
        if has_reference and reference:
            sample_kwargs["reference"] = reference

        samples.append(SingleTurnSample(**sample_kwargs))
        raw_records.append({
            "id": qid,
            "question": question_text,
            "n_contexts": len(contexts),
            "context_chars": sum(len(c) for c in contexts),
            "answer_chars": len(answer),
        })

    # ── Запуск оценки ─────────────────────────────────────────────────────
    from ragas import evaluate

    dataset = EvaluationDataset(samples=samples)
    console.rule("[bold]Запуск RAGAS evaluate…[/bold]")

    t0 = time.perf_counter()
    with console.status("Оценка (может занять несколько минут для каждого вопроса)…"):
        result = evaluate(dataset=dataset, metrics=metrics)
    t_eval = time.perf_counter() - t0

    console.print(f"  Оценка завершена за {t_eval:.1f}s")

    # ── Результаты ────────────────────────────────────────────────────────
    result_df = result.to_pandas()
    metric_cols = [c for c in result_df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]

    res_tbl = Table(title="RAGAS Results", box=box.ROUNDED, show_lines=True)
    res_tbl.add_column("Question ID", style="cyan", max_width=10)
    res_tbl.add_column("Question", max_width=45)
    for col in metric_cols:
        res_tbl.add_column(col.replace("_", "\n"), justify="right", min_width=8)

    for i, (_, row) in enumerate(result_df.iterrows()):
        vals = [raw_records[i]["id"], raw_records[i]["question"][:45]]
        for col in metric_cols:
            v = row.get(col)
            if v is None or (isinstance(v, float) and v != v):  # NaN
                vals.append("[dim]N/A[/dim]")
            else:
                color = "green" if float(v) >= 0.7 else "yellow" if float(v) >= 0.4 else "red"
                vals.append(f"[{color}]{float(v):.3f}[/{color}]")
        res_tbl.add_row(*vals)

    console.print(res_tbl)

    # Aggregate
    agg_tbl = Table(title="Aggregate Scores", box=box.SIMPLE)
    agg_tbl.add_column("Metric", style="bold")
    agg_tbl.add_column("Mean", justify="right")
    agg_tbl.add_column("Min", justify="right")
    agg_tbl.add_column("Max", justify="right")
    for col in metric_cols:
        series = result_df[col].dropna()
        if series.empty:
            continue
        mean_v, min_v, max_v = series.mean(), series.min(), series.max()
        color = "green" if mean_v >= 0.7 else "yellow" if mean_v >= 0.4 else "red"
        agg_tbl.add_row(
            col,
            f"[{color}]{mean_v:.3f}[/{color}]",
            f"{min_v:.3f}",
            f"{max_v:.3f}",
        )
    console.print(agg_tbl)

    # ── Save ──────────────────────────────────────────────────────────────
    if args.save:
        out_dir = ROOT / "tests" / "results"
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out_dir / f"ragas_{ts}.json"
        report = {
            "timestamp": ts,
            "model": cfg["llm"]["model"],
            "retrieval": {
                "top_k": cfg["retrieval"]["top_k"],
                "threshold": cfg["retrieval"]["score_threshold"],
                "hyde": cfg["retrieval"].get("hyde", False),
            },
            "per_question": [],
            "aggregate": {},
        }
        for i, (_, row) in enumerate(result_df.iterrows()):
            qrec = dict(raw_records[i])
            for col in metric_cols:
                v = row.get(col)
                qrec[col] = round(float(v), 4) if v is not None and v == v else None
            report["per_question"].append(qrec)
        for col in metric_cols:
            series = result_df[col].dropna()
            report["aggregate"][col] = round(series.mean(), 4) if not series.empty else None
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        console.print(f"\n[dim]Отчёт сохранён: {out}[/dim]")
        console.print("[dim]Сравни с предыдущими запусками в tests/results/[/dim]")

    # ── Интерпретация ────────────────────────────────────────────────────
    console.print(Panel(
        "[bold]Как интерпретировать:[/bold]\n"
        "  [cyan]ContextPrecision[/cyan] — доля релевантных чанков среди найденных (выше = лучший retrieval)\n"
        "  [cyan]ContextRecall[/cyan]    — доля нужной информации, попавшей в контекст (выше = меньше пропусков)\n"
        "  [cyan]Faithfulness[/cyan]     — модель отвечает ТОЛЬКО из контекста (выше = меньше галлюцинаций)\n"
        "  [cyan]ResponseRelevancy[/cyan]— ответ раскрывает вопрос (выше = более полный ответ)\n\n"
        "  [green]≥0.7[/green] хорошо  [yellow]0.4–0.7[/yellow] приемлемо  [red]<0.4[/red] плохо",
        title="Справка",
        border_style="dim",
    ))


if __name__ == "__main__":
    main()
