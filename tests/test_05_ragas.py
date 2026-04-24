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


def retrieve_context(question: dict, cfg: dict) -> tuple[list[str], list[dict]]:
    """Retrieval для RAGAS-оценки.

    Ищет по каждому автору отдельно (без HyDE — точнее для коротких запросов),
    объединяет, дедуплицирует и возвращает top_k лучших чанков суммарно.
    Кол-во чанков ограничено top_k из config (default 7), иначе RAGAS
    делает по LLM-вызову на каждый чанк и выбивается из timeout.
    """
    from utils import retrieve

    question_text = question["question"]
    # Для RAGAS evaluation берём не более 3 чанков: каждый чанк = отдельный LLM-вызов
    # (~70s × 7 чанков = 490s только для precision), что гарантированно выбивает timeout.
    # 3 чанка × 70s ≈ 210s — укладывается в timeout.
    top_k = min(cfg["retrieval"]["top_k"], 3)

    # HyDE отключаем для evaluation: прямой запрос точнее для специфичных вопросов,
    # и экономит ~30s на LLM-вызов гипотетического документа.
    eval_cfg = {**cfg, "retrieval": {**cfg["retrieval"], "hyde": False}}

    authors = [a["name"] for a in cfg.get("authors", [])] or [None]

    all_chunks: list[dict] = []
    seen: set[str] = set()
    for author in authors:
        chunks = retrieve(
            question_text,
            get_collection(),
            get_embed_model(),
            eval_cfg,
            author_filter=author,
        )
        for c in chunks:
            if c["text"] not in seen:
                seen.add(c["text"])
                all_chunks.append(c)

    all_chunks.sort(key=lambda c: c["score"], reverse=True)
    best = all_chunks[:top_k]
    return [c["text"] for c in best], best


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


def build_ragas_llm(judge_model: str | None = None):
    """Build LangChain-wrapped Ollama LLM for RAGAS."""
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    cfg = get_config()
    # Judge LLM отдельно от answer LLM — должен продуцировать чёткий JSON.
    # gemma3:4b хорошо генерирует ответы, но плохо следует JSON-формату RAGAS-промптов.
    # llama3:8b, mistral:7b — значительно надёжнее выдают JSON.
    model = judge_model or cfg["llm"].get("ragas_judge_model") or cfg["llm"]["model"]
    base_url = cfg["llm"].get("api_base", "http://localhost:11434")
    console.print(f"  [dim]RAGAS judge LLM: {model} @ {base_url}[/dim]")
    # НЕ используем format="json" — Ollama JSON-режим ломает fix_output_format:
    # модель начинает оборачивать произвольный текст в {} вместо нужной RAGAS-схемы.
    # RAGAS сам справляется с парсингом при достаточном timeout.
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
    parser.add_argument("--judge", metavar="MODEL", default=None,
                        help="Ollama модель для RAGAS judge (default: значение llm.model из config.yaml)")
    parser.add_argument("--only", metavar="METRIC",
                        choices=["precision", "recall", "faithfulness", "relevancy"],
                        help="Запустить только одну метрику (для быстрой проверки)")
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
        ragas_llm = build_ragas_llm(args.judge)
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

    # --only: ограничить одной метрикой
    if getattr(args, "only", None):
        _map = {
            "precision": "LLMContextPrecisionWithReference",
            "recall": "LLMContextRecall",
            "faithfulness": "Faithfulness",
            "relevancy": "ResponseRelevancy",
        }
        target = _map[args.only]
        metrics = [m for m in metrics if m.__class__.__name__ == target]
        if not metrics:
            console.print(f"[red]Метрика '{args.only}' недоступна с текущими настройками[/red]")
            sys.exit(1)

    console.print(f"  Метрики: {[m.__class__.__name__ for m in metrics]}")

    # ── Сборка датасета ───────────────────────────────────────────────────
    samples: list[SingleTurnSample] = []
    raw_records = []
    contexts_by_qid: dict[str, list[str]] = {}
    answers_by_qid: dict[str, str] = {}

    for q in questions:
        qid = q["id"]
        question_text = q["question"]
        reference = q.get("reference", "").strip() or None
        author_filter = q.get("author_filter")

        console.rule(f"[dim]{qid}: {question_text[:70]}[/dim]", style="dim")

        # Retrieval — используем тот же пайплайн что и 6_generate_answers_v2
        with console.status(f"Retrieval: {qid}…"):
            t0 = time.perf_counter()
            contexts, raw_chunks = retrieve_context(q, cfg)
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

        contexts_by_qid[qid] = contexts
        answers_by_qid[qid] = answer

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
    from ragas import evaluate, RunConfig

    # ВАЖНО: запускаем evaluate() отдельно для каждой метрики.
    # Если передать все 4 сразу — asyncio создаёт все Job-ы одновременно,
    # таймеры тикают параллельно, и Jobs 1-3 падают пока ждут Job 0.
    # При отдельных вызовах каждая метрика получает полный timeout.
    # llama3:8b: ~70s / вызов. 3 чанка × ~5 вызовов = ~350s — нужен запас ≥ 600s.
    run_cfg = RunConfig(max_workers=1, timeout=1800)

    dataset = EvaluationDataset(samples=samples)
    console.rule("[bold]Запуск RAGAS evaluate…[/bold]")

    import pandas as pd

    all_metric_dfs: list[pd.DataFrame] = []
    t_total = 0.0
    for metric in metrics:
        metric_name = metric.__class__.__name__
        with console.status(f"  {metric_name}…"):
            t0 = time.perf_counter()
            try:
                r = evaluate(dataset=dataset, metrics=[metric], run_config=run_cfg)
                df = r.to_pandas()
                all_metric_dfs.append(df)
                elapsed = time.perf_counter() - t0
                t_total += elapsed
                console.print(f"  [green]✓[/green] {metric_name}: {elapsed:.0f}s")
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                t_total += elapsed
                console.print(f"  [red]✗[/red] {metric_name}: {exc} ({elapsed:.0f}s)")

    console.print(f"  Всего: {t_total:.0f}s")

    # Объединяем результаты в один DataFrame
    if all_metric_dfs:
        base_cols = ["user_input", "response", "retrieved_contexts", "reference"]
        result_df = all_metric_dfs[0].copy()
        for df in all_metric_dfs[1:]:
            metric_cols_extra = [c for c in df.columns if c not in base_cols]
            for col in metric_cols_extra:
                result_df[col] = df[col].values
    else:
        result_df = pd.DataFrame()

    t_eval = t_total

    # ── Результаты ────────────────────────────────────────────────────────
    if result_df.empty:
        console.print("[red]Нет результатов — все метрики упали.[/red]")
        sys.exit(1)
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

    # ── Покрытие аспектов (sub_aspects → ContextRecall) ─────────────────────
    aspect_meta: list[dict] = []
    for q in questions:
        qid = q["id"]
        sub_aspects = q.get("sub_aspects", [])
        filled = [a for a in sub_aspects if a.get("text", "").strip()]
        if not filled:
            continue
        for aspect in filled:
            aspect_meta.append({
                "qid": qid,
                "aid": aspect["id"],
                "label": aspect.get("label", aspect["id"]),
                "text": aspect["text"],
            })

    # ── Покрытие аспектов через embedding cosine-similarity (без LLM) ──────
    # Для каждого sub_aspect проверяем, покрывает ли хотя бы один из
    # retrieved_contexts этот аспект семантически (cosine sim > threshold).
    # Используем BAAI/bge-m3 напрямую — быстро, без LLM-вызовов.
    aspect_coverage: list[dict] = []
    if aspect_meta:
        console.rule("[bold]Покрытие аспектов — Cosine Similarity (bge-m3)[/bold]")
        console.print(f"  Аспектов для оценки: {len(aspect_meta)}")
        try:
            import numpy as np

            COVERAGE_THRESHOLD = 0.45  # cosine sim для BAAI/bge-m3

            t0 = time.perf_counter()
            embed_model = get_embed_model()

            for meta in aspect_meta:
                qid = meta["qid"]
                ctxs = contexts_by_qid.get(qid, [])
                if not ctxs:
                    aspect_coverage.append({**meta, "max_sim": None, "covered": False})
                    continue

                # Encode aspect text и все chunks
                asp_emb = embed_model.encode(
                    [meta["text"]], normalize_embeddings=True, show_progress_bar=False
                )[0]
                ctx_embs = embed_model.encode(
                    ctxs, normalize_embeddings=True, batch_size=16, show_progress_bar=False
                )
                sims = ctx_embs @ asp_emb  # cosine similarity (vectors уже нормализованы)
                max_sim = float(np.max(sims))
                covered = max_sim >= COVERAGE_THRESHOLD
                aspect_coverage.append({**meta, "max_sim": round(max_sim, 4), "covered": covered})

            t_asp = time.perf_counter() - t0
            console.print(f"  Завершено за {t_asp:.1f}s")

            asp_tbl = Table(
                title=f"Покрытие аспектов (cosine_sim ≥ {COVERAGE_THRESHOLD})",
                box=box.ROUNDED, show_lines=True,
            )
            asp_tbl.add_column("ID", style="cyan", min_width=7)
            asp_tbl.add_column("Аспект", max_width=45)
            asp_tbl.add_column("Max Sim", justify="right", min_width=8)
            asp_tbl.add_column("Покрыт?", justify="center", min_width=8)
            for ac in aspect_coverage:
                sim_v = ac["max_sim"]
                if sim_v is None:
                    sim_str = "[dim]N/A[/dim]"
                    cov_str = "[dim]—[/dim]"
                else:
                    color = "green" if sim_v >= 0.6 else "yellow" if sim_v >= COVERAGE_THRESHOLD else "red"
                    sim_str = f"[{color}]{sim_v:.3f}[/{color}]"
                    cov_str = "[green]✓[/green]" if ac["covered"] else "[red]✗[/red]"
                asp_tbl.add_row(ac["aid"], ac["label"][:45], sim_str, cov_str)
            console.print(asp_tbl)

            covered_count = sum(1 for ac in aspect_coverage if ac["covered"])
            total_count = len(aspect_coverage)
            pct = 100 * covered_count / total_count if total_count else 0
            color = "green" if pct >= 70 else "yellow" if pct >= 40 else "red"
            console.print(
                f"  Покрытых аспектов: [{color}]{covered_count}/{total_count} ({pct:.0f}%)[/{color}]"
            )
        except Exception as exc:
            console.print(f"[red]Ошибка оценки аспектов: {exc}[/red]")
            import traceback; traceback.print_exc()

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
        if aspect_coverage:
            report["aspect_coverage"] = [
                {
                    "id": ac["aid"],
                    "label": ac["label"],
                    "max_cosine_sim": ac["max_sim"],
                    "covered": ac["covered"],
                }
                for ac in aspect_coverage
            ]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        console.print(f"\n[dim]Отчёт сохранён: {out}[/dim]")
        console.print("[dim]Сравни с предыдущими запусками в tests/results/[/dim]")

    # ── Интерпретация ────────────────────────────────────────────────────
    console.print(Panel(
        "[bold]Как интерпретировать:[/bold]\n"
        "  [cyan]SemanticSimilarity[/cyan] — cosine-sim между LLM-ответом и эталоном (\u2265 0.7 = хорошо)\n"
        "  [cyan]ResponseRelevancy[/cyan] — ответ раскрывает вопрос (выше = более полный ответ)\n"
        "  [cyan]Покрытие аспектов[/cyan] — доля аспектов с cosine_sim \u2265 0.45 хотя бы с одним чанком\n\n"
        "  [green]≥0.7[/green] хорошо  [yellow]0.4–0.7[/yellow] приемлемо  [red]<0.4[/red] плохо",
        title="Справка",
        border_style="dim",
    ))


if __name__ == "__main__":
    main()
