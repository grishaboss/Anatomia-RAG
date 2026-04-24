#!/usr/bin/env python3
"""
Комплексный grid-эксперимент для оптимизации RAG-пайплайна.

Перебирает комбинации из experiments/full_grid.yaml:
  • embed_strategy    — стратегия формирования текста для эмбеддинга
  • embedding_model   — HuggingFace модель для векторизации
  • merge_window      — склейка N соседних чанков из одного источника
  • max_chunk_chars   — сплит длинных чанков
  • top_k             — количество чанков в контексте
  • hyde              — гипотетический документ (HyDE)
  • llm_model         — Ollama-модель для генерации ответов
  • prompt_template   — шаблон промпта
  • ragas.judge_model — Ollama-модель для RAGAS-судьи
  • ragas.metrics     — набор метрик RAGAS

После каждого эксперимента запускает RAGAS.
Результаты сохраняет в experiments/results/TIMESTAMP_EXPNAME.json
Итоговую сравнительную таблицу выводит в конце.

Использование:
    python scripts/run_experiment_grid.py
    python scripts/run_experiment_grid.py --experiments baseline text_only
    python scripts/run_experiment_grid.py --ids I.3
    python scripts/run_experiment_grid.py --skip-ragas
    python scripts/run_experiment_grid.py --skip-generation
    python scripts/run_experiment_grid.py --no-hyde
    python scripts/run_experiment_grid.py --config experiments/full_grid.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from rich import box as rich_box
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from utils import ROOT, get_device, load_config, _generate_hypothetical_answer, call_llm

console = Console()

METRIC_ALIASES = {
    "relevancy":   "ResponseRelevancy",
    "faithfulness": "Faithfulness",
    "precision":   "LLMContextPrecisionWithReference",
    "recall":      "LLMContextRecall",
}


# ══════════════════════════════════════════════════════════════════════════════
# Пре-обработка чанков
# ══════════════════════════════════════════════════════════════════════════════

def load_all_chunks(main_cfg: dict) -> list[dict]:
    """Загрузить все чанки из data/chunks/*.jsonl."""
    chunks_dir = ROOT / main_cfg["paths"]["chunks_dir"]
    chunks: list[dict] = []
    for path in sorted(chunks_dir.glob("*_chunks.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


def merge_chunks(chunks: list[dict], window: int) -> list[dict]:
    """Склеить до `window` последовательных чанков из одного файла.

    Не перекрывающиеся окна: чанки [0..W-1], [W..2W-1], ...
    Итоговый чанк наследует метаданные первого в группе.
    """
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


def preprocess_chunks(chunks: list[dict], exp: dict) -> list[dict]:
    """Применить merge_window и max_chunk_chars из конфига эксперимента."""
    chunks = merge_chunks(chunks, exp.get("merge_window", 1))
    chunks = split_large_chunks(chunks, exp.get("max_chunk_chars", 0))
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Формирование embed_text из чанка
# ══════════════════════════════════════════════════════════════════════════════

def get_embed_text(chunk: dict, strategy: str, chars: int = 0) -> str:
    """Сформировать строку для эмбеддинга по заданной стратегии.

    Стратегии:
      heading_only      — только заголовок (текущий baseline)
      text_only         — только текст чанка
      heading_plus_text — заголовок + первые `chars` символов текста (0=всё)
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# Embedding модели (кэш по имени)
# ══════════════════════════════════════════════════════════════════════════════

_model_cache: dict[str, object] = {}


def get_embed_model(model_name: str, device: str):
    """Загрузить/вернуть из кэша SentenceTransformer."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        console.print(f"  [dim]Загружаю embedding модель: {model_name} на {device}[/dim]")
        _model_cache[model_name] = SentenceTransformer(model_name, device=device)
    return _model_cache[model_name]


_emb_cache: dict[tuple, np.ndarray] = {}


def get_chunk_embeddings(
    chunks: list[dict],
    model,
    model_name: str,
    strategy: str,
    chars: int,
    merge_window: int,
    max_chars: int,
    batch_size: int = 128,
) -> np.ndarray:
    """Закодировать все чанки; кэш по (model_name, strategy, chars, merge, split)."""
    key = (model_name, strategy, chars, merge_window, max_chars)
    if key not in _emb_cache:
        texts = [get_embed_text(c, strategy, chars) for c in chunks]
        console.print(
            f"  Кодирую {len(texts)} чанков "
            f"(model={model_name.split('/')[-1]}, strategy={strategy}, chars={chars})…"
        )
        t0 = time.perf_counter()
        embs = model.encode(
            texts, normalize_embeddings=True,
            batch_size=batch_size, show_progress_bar=True,
        )
        _emb_cache[key] = np.array(embs, dtype=np.float32)
        console.print(f"  Готово за {time.perf_counter() - t0:.1f}s")
    else:
        console.print(f"  [dim]Chunk embeddings из кэша[/dim]")
    return _emb_cache[key]


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval (in-memory cosine similarity)
# ══════════════════════════════════════════════════════════════════════════════

def encode_query(
    query: str,
    model,
    hyde: bool,
    main_cfg: dict,
) -> np.ndarray:
    """Закодировать запрос (опционально через HyDE)."""
    if hyde:
        console.print("  [dim]HyDE: генерирую гипотетический ответ…[/dim]")
        hypo = _generate_hypothetical_answer(query, main_cfg)
        console.print(f"  [dim]→ {hypo[:100]}[/dim]")
        text = hypo
    else:
        text = query
    emb = model.encode([text], normalize_embeddings=True)
    return np.array(emb[0], dtype=np.float32)


def retrieve_inmemory(
    q_emb: np.ndarray,
    chunk_embs: np.ndarray,
    chunks: list[dict],
    top_k: int,
) -> list[dict]:
    """Cosine similarity → top-k чанков (векторы уже нормализованы)."""
    scores = chunk_embs @ q_emb
    idx = np.argsort(scores)[::-1][:top_k]
    return [{"chunk": chunks[i], "score": float(scores[i])} for i in idx]


def format_context(hits: list[dict]) -> str:
    """Отформатировать найденные чанки в текстовый блок для LLM."""
    parts = []
    for i, h in enumerate(hits, 1):
        c = h["chunk"]
        heading = (c.get("embed_text") or "").replace("\n", " ")
        author = c.get("author") or "—"
        header = f"[{i}] {author}"
        if heading:
            header += f" | {heading}"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Генерация ответа
# ══════════════════════════════════════════════════════════════════════════════

def generate_answer(
    question: str,
    context: str,
    llm_model: str,
    prompt_template: str,
    main_cfg: dict,
) -> str:
    """Вызвать Ollama с нужной моделью и промптом."""
    prompt = prompt_template.format(context=context, question=question)
    system = "Ты эксперт-анатом. Отвечай на русском языке."
    return call_llm(prompt, system, main_cfg, model_name=llm_model)


# ══════════════════════════════════════════════════════════════════════════════
# RAGAS
# ══════════════════════════════════════════════════════════════════════════════

def _import_metric_class(metric_name: str):
    """Импортировать класс метрики из ragas (новый API → fallback к старому)."""
    class_name = METRIC_ALIASES.get(metric_name, metric_name)
    try:
        import importlib
        mod = importlib.import_module("ragas.metrics.collections")
        return getattr(mod, class_name)
    except (ImportError, AttributeError):
        import importlib
        mod = importlib.import_module("ragas.metrics")
        return getattr(mod, class_name)


def run_ragas(
    samples_data: list[dict],      # [{question, answer, contexts, reference}]
    ragas_cfg: dict,               # из эксперимента
    ragas_defaults: dict,
    main_cfg: dict,
) -> dict[str, float | None]:
    """Запустить RAGAS для списка вопросов. Возвращает dict metric_name → mean score."""
    from ragas import evaluate, RunConfig
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings

    merged_cfg = {**ragas_defaults, **ragas_cfg}
    judge_model = merged_cfg.get("judge_model", main_cfg["llm"]["model"])
    timeout = merged_cfg.get("timeout", 900)
    metric_names: list[str] = merged_cfg.get("metrics", ["relevancy"])

    base_url = main_cfg["llm"].get("api_base", "http://localhost:11434")
    llm = LangchainLLMWrapper(ChatOllama(model=judge_model, temperature=0, base_url=base_url))

    device = get_device(main_cfg)
    embed_name = main_cfg["embedding"]["model"]
    hf_embs = HuggingFaceEmbeddings(model_name=embed_name, model_kwargs={"device": device})
    ragas_emb = LangchainEmbeddingsWrapper(hf_embs)

    samples = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s.get("reference"),
        )
        for s in samples_data
    ]
    dataset = EvaluationDataset(samples=samples)
    run_cfg = RunConfig(max_workers=1, timeout=timeout)

    results: dict[str, float | None] = {}
    for metric_name in metric_names:
        try:
            MetricClass = _import_metric_class(metric_name)
            if metric_name == "relevancy":
                metric = MetricClass(llm=llm, embeddings=ragas_emb)
            else:
                metric = MetricClass(llm=llm)
            t0 = time.perf_counter()
            r = evaluate(dataset=dataset, metrics=[metric], run_config=run_cfg)
            df = r.to_pandas()
            col = [c for c in df.columns if metric_name in c.lower() or METRIC_ALIASES.get(metric_name, "").lower() in c.lower()]
            val = float(df[col[0]].mean()) if col else None
            results[metric_name] = val
            console.print(
                f"  [green]✓[/green] RAGAS {metric_name} ({judge_model}): "
                f"{val:.3f} ({time.perf_counter()-t0:.0f}s)"
            )
        except Exception as exc:
            console.print(f"  [red]✗[/red] RAGAS {metric_name}: {exc}")
            results[metric_name] = None

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Оценка релевантности (by keywords, без LLM)
# ══════════════════════════════════════════════════════════════════════════════

def count_relevant(hits: list[dict], keywords: list[str]) -> int:
    if not keywords:
        return 0
    result = 0
    for h in hits:
        c = h["chunk"]
        haystack = (
            c.get("text", "") + " " +
            (c.get("embed_text") or "") + " " +
            " ".join(c.get("headings") or [])
        ).lower()
        if any(kw.lower() in haystack for kw in keywords):
            result += 1
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Сохранение результатов
# ══════════════════════════════════════════════════════════════════════════════

def save_results(results: dict, exp_name: str) -> Path:
    results_dir = ROOT / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"{ts}_{exp_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Сводная таблица результатов (CSV — для последующего анализа)
# ══════════════════════════════════════════════════════════════════════════════

SUMMARY_CSV = ROOT / "experiments" / "results_summary.csv"

# Фиксированные колонки конфига — порядок важен для читаемости
_CFG_COLS = [
    "embedding_model",
    "embed_strategy",
    "embed_chars",
    "merge_window",
    "max_chunk_chars",
    "top_k",
    "hyde",
    "llm_model",
    "prompt_template",
]


def _model_slug(name: str) -> str:
    """'BAAI/bge-m3' → 'bge-m3'."""
    return name.split("/")[-1]


def update_summary_csv(exp_result: dict, result_file: Path) -> None:
    """Добавить/обновить строку в experiments/results_summary.csv.

    Структура строки:
      run_ts | exp_name | [cfg columns] | n_questions | avg_rel_pct |
      rel_{qid}... | avg_score_top1 | avg_time_ret | avg_time_gen |
      ragas_relevancy | ragas_faithfulness | ragas_precision | ragas_recall |
      result_file

    Если строка с таким (exp_name + run_ts) уже есть — она перезаписывается.
    """
    cfg = exp_result.get("config", {})
    qr_list: list[dict] = exp_result.get("query_results", [])
    ragas: dict = exp_result.get("ragas", {})

    # ── Вычислить агрегаты по запросам ──────────────────────────────────────
    n_q = len(qr_list)
    rel_pcts: list[float] = []
    avg_scores: list[float] = []
    ret_times: list[float] = []
    gen_times: list[float] = []

    per_q_rel: dict[str, str] = {}
    for qr in qr_list:
        qid = qr["id"]
        n_rel = qr.get("n_relevant", 0)
        tk = qr.get("top_k", 1) or 1
        pct = round(n_rel / tk, 3)
        rel_pcts.append(pct)
        per_q_rel[f"rel_{qid}"] = str(n_rel)
        per_q_rel[f"rel_pct_{qid}"] = f"{pct:.3f}"
        # Средний score top-1 чанка
        hits = qr.get("hits", [])
        if hits:
            avg_scores.append(hits[0].get("score", 0.0))
        ret_times.append(qr.get("time_retrieval", 0.0))
        gen_times.append(qr.get("time_generation", 0.0))

    avg_rel_pct = round(sum(rel_pcts) / len(rel_pcts), 3) if rel_pcts else 0.0
    avg_top1_score = round(sum(avg_scores) / len(avg_scores), 4) if avg_scores else 0.0
    avg_ret = round(sum(ret_times) / len(ret_times), 2) if ret_times else 0.0
    avg_gen = round(sum(gen_times) / len(gen_times), 2) if gen_times else 0.0

    # ── Собрать строку ───────────────────────────────────────────────────────
    row: dict[str, str] = {
        "run_ts":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exp_name":  exp_result.get("name", ""),
        "description": exp_result.get("description", ""),
    }
    for col in _CFG_COLS:
        val = cfg.get(col, "")
        if col == "embedding_model":
            val = _model_slug(str(val))
        row[col] = str(val)

    row["n_questions"]   = str(n_q)
    row["avg_rel_pct"]   = f"{avg_rel_pct:.3f}"
    row["avg_top1_score"] = f"{avg_top1_score:.4f}"
    row["avg_time_ret_s"] = f"{avg_ret:.2f}"
    row["avg_time_gen_s"] = f"{avg_gen:.2f}"

    # Колонки per-question
    row.update(per_q_rel)

    # RAGAS
    for metric in ("relevancy", "faithfulness", "precision", "recall"):
        val = ragas.get(metric)
        row[f"ragas_{metric}"] = f"{val:.4f}" if val is not None else ""

    row["result_file"] = str(result_file.relative_to(ROOT))

    # ── Читаем существующий CSV ──────────────────────────────────────────────
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if SUMMARY_CSV.exists():
        try:
            with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
                existing = list(csv.DictReader(f))
        except Exception:
            existing = []

    # Удалить старую строку с тем же exp_name + result_file (idempotent overwrite)
    existing = [
        r for r in existing
        if not (r.get("exp_name") == row["exp_name"] and r.get("result_file") == row["result_file"])
    ]
    existing.append(row)

    # ── Пересобрать заголовок (union всех ключей, сохраняя порядок) ──────────
    all_keys: list[str] = []
    seen: set[str] = set()
    priority = (
        ["run_ts", "exp_name", "description"]
        + _CFG_COLS
        + ["n_questions", "avg_rel_pct", "avg_top1_score", "avg_time_ret_s", "avg_time_gen_s"]
        + sorted(k for k in row if k.startswith("rel_"))
        + ["ragas_relevancy", "ragas_faithfulness", "ragas_precision", "ragas_recall"]
        + ["result_file"]
    )
    for k in priority:
        if k not in seen:
            all_keys.append(k)
            seen.add(k)
    for r in existing:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # ── Записать ─────────────────────────────────────────────────────────────
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)

    console.print(
        f"  [dim]Сводная таблица обновлена: "
        f"[cyan]experiments/results_summary.csv[/cyan]  "
        f"({len(existing)} строк)[/dim]"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-эксперимент RAG-пайплайна")
    parser.add_argument("--config", default="experiments/full_grid.yaml")
    parser.add_argument("--experiments", nargs="*", metavar="NAME",
                        help="Запустить только эти эксперименты")
    parser.add_argument("--ids", nargs="*", metavar="ID",
                        help="Только эти ID из ground_truth.json (напр. I.3)")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Пропустить RAGAS (только retrieval + generation)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Пропустить генерацию ответа (только retrieval)")
    parser.add_argument("--no-hyde", action="store_true",
                        help="Пропустить эксперименты с hyde: true")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Переопределить top_k для всех экспериментов")
    args = parser.parse_args()

    # ── Загрузка конфигов ──
    config_path = ROOT / args.config
    if not config_path.exists():
        console.print(f"[red]Конфиг не найден: {config_path}[/red]")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        grid_cfg = yaml.safe_load(f)

    main_cfg = load_config()
    device = get_device(main_cfg)
    ragas_defaults = grid_cfg.get("ragas_defaults", {})
    prompt_templates: dict[str, str] = grid_cfg.get("prompt_templates", {})

    # ── Загрузка ground truth ──
    gt_path = ROOT / grid_cfg.get("ground_truth_file", "tests/ground_truth.json")
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    queries = gt["questions"]
    if args.ids:
        queries = [q for q in queries if q["id"] in args.ids]
    if not queries:
        console.print("[red]Нет вопросов для тестирования.[/red]")
        sys.exit(1)

    # ── Фильтрация экспериментов ──
    experiments: list[dict] = grid_cfg.get("experiments", [])
    if args.experiments:
        experiments = [e for e in experiments if e["name"] in args.experiments]
    if args.no_hyde:
        experiments = [e for e in experiments if not e.get("hyde", False)]
    if not experiments:
        console.print("[red]Нет экспериментов для запуска.[/red]")
        sys.exit(1)

    # ── Загрузка чанков ──
    console.rule("[bold]run_experiment_grid[/bold]")
    console.print(f"  Вопросов: {len(queries)} | Экспериментов: {len(experiments)}")
    with console.status("Загружаю чанки…"):
        raw_chunks = load_all_chunks(main_cfg)
    console.print(f"  Чанков из jsonl: {len(raw_chunks)}")

    # ── Запуск экспериментов ──
    all_results: list[dict] = []

    for exp in experiments:
        exp_name = exp["name"]
        desc = exp.get("description", "")
        console.rule(f"[bold cyan]{exp_name}[/bold cyan]  —  {desc}")

        strategy = exp["embed_strategy"]
        embed_chars = exp.get("embed_chars", 300)
        merge_w = exp.get("merge_window", 1)
        max_chars = exp.get("max_chunk_chars", 0)
        top_k = args.top_k or exp.get("top_k", 7)
        hyde = exp.get("hyde", False)
        llm_model = exp.get("llm_model", main_cfg["llm"]["model"])
        prompt_key = exp.get("prompt_template", "standard")
        prompt_template = prompt_templates.get(prompt_key, "{context}\n\n{question}")

        # Пре-обработка чанков
        chunks = preprocess_chunks(raw_chunks, exp)
        console.print(f"  Чанков после merge/split: {len(chunks)}")

        # Embedding модель
        emb_model_name = exp.get("embedding_model", main_cfg["embedding"]["model"])
        embed_model = get_embed_model(emb_model_name, device)

        # Эмбеддинги чанков
        chunk_embs = get_chunk_embeddings(
            chunks, embed_model, emb_model_name,
            strategy, embed_chars, merge_w, max_chars,
        )

        exp_result = {
            "name": exp_name,
            "description": desc,
            "config": {
                "embedding_model": emb_model_name,
                "embed_strategy": strategy,
                "embed_chars": embed_chars,
                "merge_window": merge_w,
                "max_chunk_chars": max_chars,
                "top_k": top_k,
                "hyde": hyde,
                "llm_model": llm_model,
                "prompt_template": prompt_key,
            },
            "query_results": [],
        }

        ragas_samples: list[dict] = []

        for q in queries:
            qid = q["id"]
            qtext = q["question"]
            reference = q.get("reference")
            keywords = [
                a.get("text", "")[:30]
                for a in q.get("sub_aspects", [])
            ] + [qtext[:20]]
            kw_simple = [w for a in q.get("sub_aspects", []) for w in
                         (a.get("label") or "").lower().split()]

            console.print(f"\n  [bold]{qid}[/bold]: {qtext[:70]}")

            # Retrieval
            t0 = time.perf_counter()
            q_emb = encode_query(qtext, embed_model, hyde, main_cfg)
            hits = retrieve_inmemory(q_emb, chunk_embs, chunks, top_k)
            t_ret = time.perf_counter() - t0

            n_rel = count_relevant(hits, kw_simple)
            console.print(f"  Retrieval: {t_ret:.1f}s | релевантных: {n_rel}/{top_k}")
            for rank, h in enumerate(hits, 1):
                c = h["chunk"]
                heading = (c.get("embed_text") or "").replace("\n", " ")[:60]
                score = h["score"]
                console.print(f"    [{rank:2d}] {score:.3f}  «{heading}»")

            context = format_context(hits)
            answer = ""

            # Generation
            if not args.skip_generation:
                t0 = time.perf_counter()
                answer = generate_answer(qtext, context, llm_model, prompt_template, main_cfg)
                t_gen = time.perf_counter() - t0
                console.print(f"  Generation ({llm_model}): {t_gen:.1f}s")
                console.print(f"  [dim]{answer[:150]}…[/dim]")
            else:
                t_gen = 0.0

            ragas_samples.append({
                "question": qtext,
                "answer": answer,
                "contexts": [h["chunk"]["text"] for h in hits],
                "reference": reference,
            })

            exp_result["query_results"].append({
                "id": qid,
                "question": qtext,
                "n_relevant": n_rel,
                "top_k": top_k,
                "answer_preview": answer[:300],
                "time_retrieval": round(t_ret, 2),
                "time_generation": round(t_gen, 2),
                "hits": [
                    {
                        "score": round(h["score"], 4),
                        "heading": (h["chunk"].get("embed_text") or "")[:80],
                        "author": h["chunk"].get("author"),
                    }
                    for h in hits
                ],
            })

        # RAGAS
        ragas_scores: dict[str, float | None] = {}
        if not args.skip_ragas and ragas_defaults.get("enabled", True):
            ragas_exp_cfg = exp.get("ragas", {})
            if ragas_exp_cfg.get("enabled", True) is not False:
                console.print("\n  [bold]RAGAS…[/bold]")
                ragas_scores = run_ragas(
                    ragas_samples, ragas_exp_cfg, ragas_defaults, main_cfg
                )

        exp_result["ragas"] = ragas_scores

        # Сохранение
        saved = save_results(exp_result, exp_name)
        console.print(f"\n  Сохранено: [cyan]{saved.relative_to(ROOT)}[/cyan]")
        update_summary_csv(exp_result, saved)
        all_results.append(exp_result)

    # ══ Итоговая сводная таблица ══════════════════════════════════════════════
    console.rule("[bold]Итоговая сводная таблица[/bold]")

    all_metric_names = sorted({m for r in all_results for m in r.get("ragas", {})})
    query_ids = [q["id"] for q in queries]

    table = Table(box=rich_box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Эксперимент", style="bold", min_width=22)
    table.add_column("Embed", style="cyan", min_width=12)
    table.add_column("Merge", justify="center")
    table.add_column("top_k", justify="center")
    table.add_column("HyDE", justify="center")
    table.add_column("LLM", style="dim")

    # Колонки: релевантность по вопросам
    for qid in query_ids:
        table.add_column(f"Rel/{qid}", justify="center")

    # Колонки: RAGAS метрики
    for m in all_metric_names:
        table.add_column(f"RAGAS\n{m}", justify="center")

    for r in all_results:
        cfg = r["config"]
        strategy_short = cfg["embed_strategy"].replace("heading_plus_text", "h+t").replace("heading_only", "head").replace("text_only", "text")
        hyde_mark = "[yellow]✓[/yellow]" if cfg.get("hyde") else "·"
        merge_str = str(cfg.get("merge_window", 1))

        rel_cells = []
        for qid in query_ids:
            qr = next((x for x in r["query_results"] if x["id"] == qid), None)
            if qr:
                n = qr["n_relevant"]
                tk = qr["top_k"]
                color = "green" if n >= tk // 2 else ("yellow" if n > 0 else "red")
                rel_cells.append(f"[{color}]{n}/{tk}[/{color}]")
            else:
                rel_cells.append("—")

        ragas_cells = []
        for m in all_metric_names:
            val = r.get("ragas", {}).get(m)
            if val is None:
                ragas_cells.append("[dim]N/A[/dim]")
            else:
                color = "green" if val >= 0.7 else ("yellow" if val >= 0.4 else "red")
                ragas_cells.append(f"[{color}]{val:.3f}[/{color}]")

        table.add_row(
            r["name"],
            strategy_short,
            merge_str,
            str(cfg.get("top_k", 7)),
            hyde_mark,
            cfg.get("llm_model", "—"),
            *rel_cells,
            *ragas_cells,
        )

    console.print(table)
    console.print(f"[dim]Результаты сохранены в experiments/results/[/dim]")


if __name__ == "__main__":
    main()
