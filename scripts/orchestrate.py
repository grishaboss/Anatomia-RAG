#!/usr/bin/env python3
"""
SLURM-совместимый оркестратор RAG-экспериментов.

Читает experiments/full_grid.yaml, запускает каждый эксперимент через srun
(или локально если SLURM не обнаружен). Поддерживает:

  • Пропуск уже выполненных экспериментов по fingerprint-у конфига
  • Несколько ChromaDB-коллекций по имени embedding-модели
  • Ресурсные профили GPU (gpu1 / gpu2 / cpu) для SLURM
  • --dry-run, --force, --resume, --stop-on-error
  • Настройки GPU-утилизации (num_workers, batch_size, fp16, tf32, ...)

Порядок работы:
  1. Разбирает full_grid.yaml и вычисляет fingerprint каждого эксперимента
  2. Если --build-index: для каждого уникального (model, strategy, ...) строит
     отдельную ChromaDB-коллекцию через 2_build_index.py
  3. Запускает run_experiment_grid.py --experiments NAME для каждого эксперимента
  4. После каждого прогона сохраняет статус в experiments/orchestrator_state.json

Использование:
    python scripts/orchestrate.py                       # все эксперименты
    python scripts/orchestrate.py --dry-run             # показать команды без запуска
    python scripts/orchestrate.py --force               # игнорировать state, перезапустить всё
    python scripts/orchestrate.py --build-index         # сначала построить ChromaDB-коллекции
    python scripts/orchestrate.py --experiments baseline text_only
    python scripts/orchestrate.py --profile gpu1 --partition a100
    python scripts/orchestrate.py --skip-ragas --ids I.3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
STATE_FILE = ROOT / "experiments" / "orchestrator_state.json"
GRID_CONFIG = ROOT / "experiments" / "full_grid.yaml"


# ══════════════════════════════════════════════════════════════════════════════
# Ресурсные профили (для SLURM)
# ══════════════════════════════════════════════════════════════════════════════

RESOURCE_PROFILES: dict[str, dict[str, Any]] = {
    # Только CPU (не нужен GPU)
    "cpu": {
        "gpu_num": 0,
        "cpus": 4,
        "mem_gb": 16,
        "time_min": 8,
        "partition": None,
    },
    # 1 GPU, малая VRAM (bge-m3 ~2.4 GB, MiniLM ~0.1 GB)
    "gpu1_small": {
        "gpu_num": 1,
        "cpus": 4,
        "mem_gb": 24,
        "time_min": 8,
    },
    # 1 GPU, стандартный (любой embedding-model до ~8 GB)
    "gpu1": {
        "gpu_num": 1,
        "cpus": 8,
        "mem_gb": 40,
        "time_min": 8,
    },
    # 2 GPU (большие модели или параллельное embedding + Ollama)
    "gpu2": {
        "gpu_num": 2,
        "cpus": 16,
        "mem_gb": 80,
        "time_min": 8,
    },
    # Полный узел с 8 GPU
    "gpu8": {
        "gpu_num": 8,
        "cpus": 96,
        "mem_gb": 320,
        "time_min": 8,
        "exclusive": True,
    },
    # Кластер hiperf, узел tesla-a101
    "hiperf": {
        "gpu_num": 1,
        "cpus": 16,
        "mem_gb": 80,
        "time_min": 8,
        "partition": "hiperf",
        "nodelist": "tesla-a101",
    },
}

_SRUN_PARSABLE_SUPPORTED: bool | None = None
_SUCCESS_STATES = {"COMPLETED"}
_JOB_ID_PATTERNS = [
    re.compile(r"\bjob\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"^(\d+)(?:;[^\s]+)?$"),
]


# ══════════════════════════════════════════════════════════════════════════════
# SLURM helpers
# ══════════════════════════════════════════════════════════════════════════════

def is_slurm() -> bool:
    """True если доступен srun (выполняется в среде SLURM)."""
    return shutil.which("srun") is not None


def _supports_parsable() -> bool:
    global _SRUN_PARSABLE_SUPPORTED
    if _SRUN_PARSABLE_SUPPORTED is not None:
        return _SRUN_PARSABLE_SUPPORTED
    if not is_slurm():
        _SRUN_PARSABLE_SUPPORTED = False
        return False
    try:
        out = subprocess.check_output(
            ["srun", "--help"], text=True, stderr=subprocess.STDOUT, timeout=10
        )
        _SRUN_PARSABLE_SUPPORTED = "--parsable" in out
    except Exception:
        _SRUN_PARSABLE_SUPPORTED = False
    return _SRUN_PARSABLE_SUPPORTED


def build_srun_prefix(
    job_name: str | None = None,
    gpu_num: int = 0,
    cpus: int = 4,
    mem_gb: int = 16,
    time_min: int = 120,
    gpu_type: str | None = None,
    partition: str | None = None,
    nodelist: str = "",
    nodes: int = 1,
    nice: int = 0,
    exclusive: bool = False,
    dependency: str | None = None,
    export_env: str = "ALL",
    **_,
) -> str:
    """Построить строку префикса srun с нужными ресурсами."""
    if not is_slurm():
        return ""

    parts = ["srun"]
    if _supports_parsable():
        parts.append("--parsable")
    if partition:
        parts += ["-p", partition]
    if exclusive:
        parts.append("--exclusive")
    parts += ["-c", str(cpus)]
    parts += ["--mem", f"{mem_gb}G"]
    if gpu_num > 0:
        gtype = f":{gpu_type}" if gpu_type else ""
        parts.append(f"--gres=gpu{gtype}:{gpu_num}")
    parts += ["-N", str(nodes)]
    if nodelist:
        parts += ["-w", nodelist]
    parts += ["-t", f"{time_min}:0:0"]
    if job_name:
        parts += ["-J", job_name]
    if nice:
        parts += ["--nice", str(nice)]
    if dependency:
        parts.append(f"--dependency={dependency}")
    parts.append(f"--export={export_env}")
    return " ".join(parts)


def make_run_cmd(
    cmd: str,
    profile: dict,
    job_name: str | None = None,
    dependency: str | None = None,
) -> str:
    """Обернуть команду в srun-префикс (или оставить без него)."""
    prefix = build_srun_prefix(
        job_name=job_name,
        dependency=dependency,
        **profile,
    )
    return f"{prefix} {cmd}".strip() if prefix else cmd


def _extract_job_id(line: str) -> str | None:
    stripped = line.strip()
    for p in _JOB_ID_PATTERNS:
        m = p.search(stripped)
        if m:
            return m.group(1)
    return None


def _normalize_slurm_state(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip().split("+")[0].split(" ")[0].upper()
    return s or None


def query_slurm_state(job_id: str) -> str | None:
    """Запросить статус задачи через sacct/scontrol."""
    if shutil.which("sacct"):
        try:
            out = subprocess.check_output(
                ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
                text=True, stderr=subprocess.DEVNULL, timeout=15,
            )
            for line in out.splitlines():
                norm = _normalize_slurm_state(line.split("|", 1)[0])
                if norm:
                    return norm
        except Exception:
            pass
    if shutil.which("scontrol"):
        try:
            out = subprocess.check_output(
                ["scontrol", "show", "job", job_id],
                text=True, stderr=subprocess.DEVNULL, timeout=15,
            )
            m = re.search(r"JobState=([^\s]+)", out)
            if m:
                return _normalize_slurm_state(m.group(1))
        except Exception:
            pass
    return None


def run_streaming(
    cmd: str,
    cwd: str | None = None,
    env: dict | None = None,
) -> tuple[int, str | None]:
    """Запустить команду, стримить stdout/stderr, вернуть (exit_code, slurm_job_id)."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    kwargs: dict[str, Any] = dict(
        shell=True,
        cwd=cwd,
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if hasattr(os, "setsid"):
        kwargs["preexec_fn"] = os.setsid

    process = subprocess.Popen(cmd, **kwargs)
    job_id: str | None = None
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            if job_id is None:
                job_id = _extract_job_id(line)
    except KeyboardInterrupt:
        with suppress(Exception):
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
        raise
    finally:
        rc = process.wait()
    return rc, job_id


# ══════════════════════════════════════════════════════════════════════════════
# State management
# ══════════════════════════════════════════════════════════════════════════════

def load_state(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "experiments": {}, "indices": {}}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("experiments", {})
        data.setdefault("indices", {})
        return data
    except Exception:
        return {"version": 1, "experiments": {}, "indices": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)


def entry_ok(entry: dict) -> bool:
    """True если задача успешно завершена (exit_code=0 + SLURM COMPLETED)."""
    if int(entry.get("exit_code", 1)) != 0:
        return False
    job_id = entry.get("job_id")
    if not job_id:
        return True
    state = query_slurm_state(str(job_id))
    if state is None:
        return True  # sacct лаг → доверяем exit_code
    return state in _SUCCESS_STATES


# ══════════════════════════════════════════════════════════════════════════════
# Fingerprinting
# ══════════════════════════════════════════════════════════════════════════════

# Параметры, которые влияют на результат эксперимента
_FP_KEYS = [
    "embedding_model", "embed_strategy", "embed_chars",
    "merge_window", "max_chunk_chars",
    "top_k", "hyde",
    "llm_model", "prompt_template",
    "ragas",
]

# Параметры, которые влияют на ChromaDB-коллекцию
_IDX_KEYS = [
    "embedding_model", "embed_strategy", "embed_chars",
    "merge_window", "max_chunk_chars",
]


def fingerprint(d: dict, keys: list[str]) -> str:
    stable = {k: d[k] for k in keys if k in d}
    canon = json.dumps(stable, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(canon.encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════════
# ChromaDB collection naming
# ══════════════════════════════════════════════════════════════════════════════

def model_slug(model_name: str) -> str:
    """'BAAI/bge-m3' → 'bge-m3',  'intfloat/multilingual-e5-large' → 'me5-large'."""
    slug = model_name.split("/")[-1]
    # Сокращаем длинные имена (не более 20 символов)
    slug = re.sub(r"multilingual-?", "m", slug)
    return slug[:20]


def derive_collection_name(exp: dict, main_cfg: dict) -> str:
    """Имя ChromaDB-коллекции по параметрам эксперимента.

    Пример: anatomy__bge-m3__text_only__c300_m1_s0
    """
    model = exp.get("embedding_model") or main_cfg.get("embedding", {}).get("model", "unknown")
    strategy = exp.get("embed_strategy", "heading_only")
    chars = exp.get("embed_chars", 300)
    merge = exp.get("merge_window", 1)
    split = exp.get("max_chunk_chars", 0)
    slug = model_slug(model)
    strat = strategy.replace("heading_plus_text", "h+text").replace("heading_only", "head").replace("text_only", "text")
    return f"anatomy__{slug}__{strat}__c{chars}_m{merge}_s{split}"


def collection_exists(collection_name: str, main_cfg: dict) -> bool:
    """True если ChromaDB-коллекция с таким именем уже существует и не пустая."""
    try:
        import chromadb
        chroma_path = ROOT / main_cfg.get("paths", {}).get("chroma_db", "chroma_db")
        client = chromadb.PersistentClient(path=str(chroma_path))
        existing = {c.name for c in client.list_collections()}
        if collection_name not in existing:
            return False
        col = client.get_collection(collection_name)
        return col.count() > 0
    except Exception:
        return False


def chunks_dir_has_data(chunks_dir: str) -> bool:
    """True если директория с чанками существует и содержит хотя бы один .jsonl файл."""
    path = ROOT / chunks_dir
    if not path.exists():
        return False
    return any(path.glob("*_chunks.jsonl"))


# ══════════════════════════════════════════════════════════════════════════════
# GPU / env настройки для хорошей утилизации
# ══════════════════════════════════════════════════════════════════════════════

def build_gpu_env(
    gpu_ids: str | None = None,
    fp16: bool = True,
    tf32: bool = True,
    tokenizers_parallelism: bool = False,
    num_workers: int = 4,
) -> dict[str, str]:
    """Переменные окружения для максимальной утилизации GPU.

    fp16=True      → SentenceTransformer будет использовать half-precision
    tf32=True       → разрешает TF32 на Ampere (A100, A10) — быстрее без потери точности
    tokenizers_parallelism=False → отключает предупреждения HuggingFace о fork
    num_workers     → для DataLoader в SentenceTransformer
    """
    env: dict[str, str] = {}
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    if tf32:
        # PyTorch: разрешить TF32 для matmul на Ampere+
        env["TORCH_ALLOW_TF32"] = "1"
    if fp16:
        # Подсказка для нашего кода (читается в run_experiment_grid.py если добавить)
        env["EMBED_FP16"] = "1"
    if not tokenizers_parallelism:
        env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = str(max(1, num_workers // 2))
    env["MKL_NUM_THREADS"] = str(max(1, num_workers // 2))
    # Настройки NCCL (для мульти-GPU — нет лишних предупреждений)
    env["NCCL_DEBUG"] = "WARN"
    # Отключить HuggingFace прогресс-бары в SLURM (нет TTY)
    if is_slurm():
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        env["TQDM_DISABLE"] = "1"
    return env


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:  # noqa: C901 — намеренно монолитная, чтобы поток был виден
    parser = argparse.ArgumentParser(
        description="SLURM-оркестратор RAG-экспериментов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default=str(GRID_CONFIG),
                        help="Путь к full_grid.yaml")
    parser.add_argument("--experiments", nargs="*", metavar="NAME",
                        help="Запустить только эти эксперименты (по имени)")
    parser.add_argument("--grid-only", action="store_true",
                        help="Использовать только grid_axes (декартово произведение)")
    parser.add_argument("--grid", action="store_true",
                        help="Объединить experiments[] и grid_axes")
    parser.add_argument("--ids", nargs="*", metavar="ID",
                        help="Только эти ID вопросов из ground_truth.json")
    parser.add_argument("--state-file", default=str(STATE_FILE),
                        help="Путь к файлу состояния")

    # Режимы запуска
    parser.add_argument("--dry-run", action="store_true",
                        help="Показать команды без выполнения")
    parser.add_argument("--force", action="store_true",
                        help="Игнорировать state, перезапустить всё")
    parser.add_argument("--no-resume", action="store_true",
                        help="Не восстанавливать состояние (аналог --force для state)")
    parser.add_argument("--stop-on-error", action="store_true", default=True,
                        help="Остановить при первой ошибке (по умолчанию: True)")
    parser.add_argument("--no-stop-on-error", dest="stop_on_error", action="store_false")

    # Шаги пайплайна
    parser.add_argument("--force-index", action="store_true",
                        help="Пересобрать ChromaDB-коллекции даже если уже есть")
    parser.add_argument("--index-only", action="store_true",
                        help="Только построить ChromaDB-коллекции (и ингестию если надо), не запускать эксперименты")
    parser.add_argument("--ingest-force", action="store_true",
                        help="Передать --force в 1_ingest.py (переобработать все файлы)")
    parser.add_argument("--ingest-only", action="store_true",
                        help="Только ингестия для всех уникальных chunks_dir, без индексирования и экспериментов")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Пропустить RAGAS в экспериментах")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Пропустить генерацию ответов")

    # SLURM / ресурсы
    parser.add_argument("--profile", default="hiperf", choices=list(RESOURCE_PROFILES),
                        help=f"Ресурсный профиль SLURM: {list(RESOURCE_PROFILES)}")
    parser.add_argument("--index-profile", default="hiperf", choices=list(RESOURCE_PROFILES),
                        help="Ресурсный профиль для 2_build_index.py")
    parser.add_argument("--partition", default="hiperf",
                        help="SLURM partition (например: v100, a100)")
    parser.add_argument("--nodelist", default="tesla-a101",
                        help="Запустить на конкретных узлах: node1,node2")
    parser.add_argument("--time", type=int, default=None, metavar="MIN",
                        help="Переопределить время (минут) для srun")
    parser.add_argument("--gpu-num", type=int, default=None,
                        help="Переопределить число GPU")
    parser.add_argument("--gpu-ids", default=None,
                        help="CUDA_VISIBLE_DEVICES (локально без SLURM)")

    # GPU-утилизация
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Передать EMBED_FP16=1 (half-precision embeddings)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--tf32", action="store_true", default=True,
                        help="Разрешить TF32 на Ampere GPU (TORCH_ALLOW_TF32=1)")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="OMP/MKL потоки")

    args = parser.parse_args()

    # ── Загрузка конфигов ──────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        print(f"[ERROR] Конфиг не найден: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        grid_cfg = yaml.safe_load(f)

    # Читаем основной config.yaml для дефолтных значений
    main_cfg_path = ROOT / "config.yaml"
    with open(main_cfg_path, encoding="utf-8") as f:
        main_cfg = yaml.safe_load(f)

    # ── Сборка списка экспериментов ───────────────────────────────────────────
    from run_experiment_grid import expand_grid_axes  # noqa: E402

    manual_exps: list[dict] = grid_cfg.get("experiments", [])
    grid_exps: list[dict] = expand_grid_axes(grid_cfg) if (args.grid_only or args.grid or not manual_exps) else []

    if args.grid_only:
        all_experiments = grid_exps
    elif args.grid:
        all_experiments = manual_exps + grid_exps
    else:
        # Обратная совместимость: если есть experiments[] — берём их;
        # если нет (pure grid yaml) — берём grid_axes
        all_experiments = manual_exps if manual_exps else grid_exps

    if args.experiments:
        all_experiments = [e for e in all_experiments if e["name"] in args.experiments]
    if not all_experiments:
        print("[ERROR] Нет экспериментов для запуска.")
        sys.exit(1)

    print(f"[GRID] Экспериментов: {len(all_experiments)}")

    # ── State ──────────────────────────────────────────────────────────────────
    state_path = Path(args.state_file)
    state = load_state(state_path)
    exp_state = state["experiments"]
    idx_state = state["indices"]

    # ── Ресурсный профиль ──────────────────────────────────────────────────────
    profile = dict(RESOURCE_PROFILES[args.profile])
    if args.partition:
        profile["partition"] = args.partition
    if args.nodelist:
        profile["nodelist"] = args.nodelist
    if args.time is not None:
        profile["time_min"] = args.time
    if args.gpu_num is not None:
        profile["gpu_num"] = args.gpu_num

    idx_profile = dict(RESOURCE_PROFILES[args.index_profile])
    if args.partition:
        idx_profile["partition"] = args.partition
    if args.nodelist:
        idx_profile["nodelist"] = args.nodelist

    # ── Env для GPU-утилизации ─────────────────────────────────────────────────
    gpu_env = build_gpu_env(
        gpu_ids=args.gpu_ids,
        fp16=args.fp16,
        tf32=args.tf32,
        num_workers=args.num_workers,
    )

    python_exe = shlex.quote(sys.executable)
    cwd = str(ROOT)

    # ──────────────────────────────────────────────────────────────────────────
    # ШАГ 0: --ingest-only — принудительная ингестия всех chunks_dir
    # (обычная ингестия встроена в ШАГ 1 автоматически)
    # ──────────────────────────────────────────────────────────────────────────

    if args.ingest_only:
        default_chunks_dir = main_cfg["paths"]["chunks_dir"]
        ingest_variants: dict[str, dict] = {}
        for exp in all_experiments:
            cd = exp.get("chunks_dir") or default_chunks_dir
            if cd not in ingest_variants:
                ingest_variants[cd] = {
                    "skip_authors":         exp.get("skip_authors", []),
                    "no_force_ocr_authors": exp.get("no_force_ocr_authors", []),
                }
        print(f"\n[INGEST-ONLY] Подготовить {len(ingest_variants)} набор(ов) чанков")
        for cd, opts in ingest_variants.items():
            ingest_cmd = f"{python_exe} scripts/1_ingest.py"
            if cd != default_chunks_dir:
                ingest_cmd += f" --chunks-dir {shlex.quote(cd)}"
            if opts.get("skip_authors"):
                ingest_cmd += " --skip-authors " + " ".join(shlex.quote(a) for a in opts["skip_authors"])
            if opts.get("no_force_ocr_authors"):
                ingest_cmd += " --no-force-ocr-authors " + " ".join(shlex.quote(a) for a in opts["no_force_ocr_authors"])
            if args.ingest_force:
                ingest_cmd += " --force"
            cpu_profile = dict(RESOURCE_PROFILES["cpu"])
            if args.partition:
                cpu_profile["partition"] = args.partition
            full_ingest_cmd = make_run_cmd(ingest_cmd, cpu_profile, job_name=f"ingest_{Path(cd).name[:20]}")
            print(f"\n[INGEST] {cd}\n  cmd: {full_ingest_cmd}")
            if not args.dry_run:
                rc_i, _ = run_streaming(full_ingest_cmd, cwd=cwd, env=gpu_env)
                if rc_i != 0:
                    print(f"[ERROR] Ingest failed (rc={rc_i})")
                    if args.stop_on_error:
                        sys.exit(rc_i)
        print("\n[DONE] Ингестия завершена.")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # ШАГ 1: Автоматически проверить и построить ChromaDB-коллекции
    # Всегда выполняется — пропускает уже готовые (state + проверка ChromaDB)
    # --force-index: пересобрать всё даже если уже есть
    # ──────────────────────────────────────────────────────────────────────────

    # Уникальные комбинации параметров индексирования
    seen_idx: dict[str, dict] = {}
    for exp in all_experiments:
        fp = fingerprint(exp, _IDX_KEYS)
        if fp not in seen_idx:
            seen_idx[fp] = exp

    needs_build: dict[str, dict] = {}
    for fp, exp in seen_idx.items():
        coll_name = derive_collection_name(exp, main_cfg)
        entry = idx_state.get(fp, {})
        already_done = (
            not args.force_index
            and entry.get("status") == "done"
            and entry_ok(entry)
            and collection_exists(coll_name, main_cfg)
        )
        if already_done:
            print(f"[OK]    Index {coll_name}  (уже построен, пропуск)")
        else:
            needs_build[fp] = exp

    if needs_build:
        print(f"\n[INDEX] Нужно построить {len(needs_build)} уникальных ChromaDB-коллекций")
    else:
        print(f"\n[INDEX] Все {len(seen_idx)} коллекций уже готовы.")

    for fp, exp in needs_build.items():
        coll_name = derive_collection_name(exp, main_cfg)

        # ── Автоматическая ингестия если chunks_dir не готов ─────────────────
        cd = exp.get("chunks_dir") or main_cfg.get("paths", {}).get("chunks_dir", "data/chunks")
        if not chunks_dir_has_data(cd):
            print(f"\n[INGEST-AUTO] chunks_dir={cd} пуст или не существует — запускаю ингестию")
            ingest_cmd = f"{python_exe} scripts/1_ingest.py"
            default_chunks_dir = main_cfg.get("paths", {}).get("chunks_dir", "data/chunks")
            if cd != default_chunks_dir:
                ingest_cmd += f" --chunks-dir {shlex.quote(cd)}"
            opts = exp
            if opts.get("skip_authors"):
                ingest_cmd += " --skip-authors " + " ".join(shlex.quote(a) for a in opts["skip_authors"])
            if opts.get("no_force_ocr_authors"):
                ingest_cmd += " --no-force-ocr-authors " + " ".join(shlex.quote(a) for a in opts["no_force_ocr_authors"])
            if args.ingest_force:
                ingest_cmd += " --force"

            cpu_profile = dict(RESOURCE_PROFILES["cpu"])
            if args.partition:
                cpu_profile["partition"] = args.partition
            full_ingest_cmd = make_run_cmd(ingest_cmd, cpu_profile, job_name=f"ingest_{Path(cd).name[:20]}")
            print(f"  cmd: {full_ingest_cmd}")

            if not args.dry_run:
                rc_i, _ = run_streaming(full_ingest_cmd, cwd=cwd, env=gpu_env)
                if rc_i != 0:
                    print(f"[ERROR] Ingest failed (rc={rc_i})")
                    if args.stop_on_error:
                        sys.exit(rc_i)
            else:
                print("  [dry-run] пропускаем реальный запуск")

        model = exp.get("embedding_model") or main_cfg["embedding"]["model"]
        strategy = exp.get("embed_strategy", "heading_only")
        embed_chars = exp.get("embed_chars", 300)
        merge_w = exp.get("merge_window", 1)
        max_chars = exp.get("max_chunk_chars", 0)
        cd = exp.get("chunks_dir") or main_cfg.get("paths", {}).get("chunks_dir", "data/chunks")

        idx_cmd = (
            f"{python_exe} scripts/2_build_index.py"
            f" --embedding-model {shlex.quote(model)}"
            f" --embed-strategy {strategy}"
            f" --embed-chars {embed_chars}"
            f" --merge-window {merge_w}"
            f" --max-chunk-chars {max_chars}"
            f" --collection-name {shlex.quote(coll_name)}"
        )

        full_idx_cmd = make_run_cmd(
            idx_cmd, idx_profile,
            job_name=f"idx_{coll_name[:20]}",
        )
        print(f"\n[INDEX] {coll_name}")
        print(f"  fingerprint = {fp}")
        print(f"  cmd: {full_idx_cmd}")

        if args.dry_run:
            idx_state[fp] = {
                "collection": coll_name,
                "status": "done",
                "exit_code": 0,
                "dry_run": True,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            save_state(state_path, state)
            continue

        t0 = time.perf_counter()
        rc, job_id = run_streaming(full_idx_cmd, cwd=cwd, env=gpu_env)
        elapsed = time.perf_counter() - t0

        entry_new = {
            "collection": coll_name,
            "status": "done" if rc == 0 else "failed",
            "exit_code": rc,
            "job_id": job_id,
            "elapsed_s": round(elapsed, 1),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        idx_state[fp] = entry_new
        save_state(state_path, state)

        if rc != 0:
            print(f"[ERROR] Index build failed (rc={rc}): {coll_name}")
            if args.stop_on_error:
                sys.exit(rc)

    if args.index_only:
        print("\n[DONE] Индексирование завершено.")
        _print_index_summary(idx_state)
        return

    # ──────────────────────────────────────────────────────────────────────────
    # ШАГ 2: Запуск экспериментов
    # ──────────────────────────────────────────────────────────────────────────

    print(f"\n[GRID] Запуск {len(all_experiments)} экспериментов")
    print(f"       SLURM: {'ДА (' + profile.get('partition', 'default') + ')' if is_slurm() else 'НЕТ (локально)'}")
    print(f"       GPU:   {profile.get('gpu_num', 0)}  |  CPUs: {profile.get('cpus', 4)}  |  MEM: {profile.get('mem_gb', 16)}G")
    print(f"       State: {state_path}")

    results: list[dict] = []

    for exp in all_experiments:
        exp_name = exp["name"]
        fp = fingerprint(exp, _FP_KEYS)
        entry = exp_state.get(fp, {})

        # ── Проверка — уже сделано? ────────────────────────────────────────
        if not args.force and not args.no_resume:
            if entry.get("status") == "done" and entry_ok(entry):
                print(f"\n[SKIP]  {exp_name}  (fingerprint={fp}, уже выполнен {entry.get('ts', '')[:10]})")
                results.append({"name": exp_name, "status": "skipped", "fp": fp})
                continue

        # ── Построить команду ──────────────────────────────────────────────
        exp_cmd = (
            f"{python_exe} scripts/run_experiment_grid.py"
            f" --config {shlex.quote(args.config)}"
            f" --experiments {shlex.quote(exp_name)}"
        )
        if args.skip_ragas:
            exp_cmd += " --skip-ragas"
        if args.skip_generation:
            exp_cmd += " --skip-generation"
        if args.ids:
            exp_cmd += " --ids " + " ".join(shlex.quote(i) for i in args.ids)

        full_cmd = make_run_cmd(exp_cmd, profile, job_name=f"exp_{exp_name[:20]}")

        print(f"\n[EXP]  {exp_name}  (fingerprint={fp})")
        print(f"  embed: {exp.get('embed_strategy', '?')}  "
              f"model: {model_slug(exp.get('embedding_model', '?'))}  "
              f"top_k: {exp.get('top_k', '?')}  "
              f"llm: {exp.get('llm_model', '?')}")
        print(f"  cmd: {full_cmd}")

        if args.dry_run:
            exp_state[fp] = {
                "name": exp_name,
                "status": "done",
                "exit_code": 0,
                "dry_run": True,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            save_state(state_path, state)
            results.append({"name": exp_name, "status": "dry_run", "fp": fp})
            continue

        t0 = time.perf_counter()
        rc, job_id = run_streaming(full_cmd, cwd=cwd, env=gpu_env)
        elapsed = time.perf_counter() - t0

        status = "done" if rc == 0 else "failed"
        exp_state[fp] = {
            "name": exp_name,
            "fingerprint": fp,
            "status": status,
            "exit_code": rc,
            "job_id": job_id,
            "elapsed_s": round(elapsed, 1),
            "config_snapshot": {k: exp.get(k) for k in _FP_KEYS if k in exp},
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        save_state(state_path, state)

        results.append({"name": exp_name, "status": status, "fp": fp, "elapsed": elapsed})

        if rc != 0:
            print(f"[ERROR] Эксперимент '{exp_name}' завершился с кодом {rc}")
            if args.stop_on_error:
                _print_summary(results)
                sys.exit(rc)

    # ── Итоговая сводка ────────────────────────────────────────────────────────
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    print("\n" + "═" * 60)
    print("ИТОГ ОРКЕСТРАТОРА")
    print("═" * 60)
    statuses: dict[str, int] = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        elapsed = f"  {r.get('elapsed', 0):.0f}s" if r.get("elapsed") else ""
        print(f"  [{r['status'].upper():8s}]  {r['name']}{elapsed}")
    print("─" * 60)
    for st, count in statuses.items():
        print(f"  {st}: {count}")
    print(f"\nState:  {STATE_FILE}")
    print(f"Results: {ROOT / 'experiments' / 'results'}/")


def _print_index_summary(idx_state: dict) -> None:
    print("\n[ИНДЕКСЫ]")
    for fp, entry in idx_state.items():
        status = entry.get("status", "?")
        coll = entry.get("collection", fp)
        ts = entry.get("ts", "")[:10]
        print(f"  [{status.upper():6s}]  {coll}  ({ts})")


if __name__ == "__main__":
    main()
