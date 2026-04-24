#исходный рабочий шаблон
'''
from contextlib import suppress
import os
import signal
import shutil
import subprocess
import time


def is_slurm():
    return shutil.which('srun') is not None


def get_cmd_prefix(
    job_name: str = None, 
    gpu_num: int = 0, 
    time: int = 20, 
    # cpu_per_gpu=2, # for shard
    cpu_per_gpu: int = 9, 
    # mem_per_gpu=7, # for shard
    mem_per_gpu: int = 47, 
    gpu_type: str = None, 
    partition: str = None,
    max_gpus_per_node: int = 8,
    exclusive: bool = False,
    shard: bool = False,
    nodelist: str = '',
    nodes: int = 1,
    nice: int = 0,
    **kwargs
):
    cmd_prefix = ''
    if is_slurm():
        cpus = int(max(1, gpu_num) * cpu_per_gpu)
        cpus = cpus - cpus % 2
        mems = int(max(1, gpu_num) * mem_per_gpu)

        if gpu_type is not None:
            gpu_type = f':{gpu_type}'
        else:
            gpu_type = ''
        
        gres_type = 'shard' if shard else 'gpu'
        exclusive = ((gpu_num == max_gpus_per_node) or exclusive) and not shard
        
        partition = f'-p {partition}' if partition is not None else ''
        gres = f'--gres={gres_type}{gpu_type}:{gpu_num}' if gpu_num else ''
        cpus = '--exclusive' if exclusive else f'-c {cpus}'
        mems = '--mem=0' if exclusive else f'--mem={mems}G'
        job_name = f"-J '{job_name}'" if job_name is not None else ''
        nl = f'-w {nodelist}' if len(nodelist) else ''
        N = f'-N {nodes}'
        nice = f'--nice={nice}' if nice else ''
        cmd_prefix = f'srun {nice} {partition} {cpus} {mems} {gres} {N} {nl} -t {time}:0:0 {job_name}'
    return cmd_prefix


def get_cmd(cmd, **kwargs):
    cmd_prefix = get_cmd_prefix(**kwargs)
    run_commmand = f"{cmd_prefix} {cmd} "
    return run_commmand


def run_tasks(
    cmd_list,
    iterations_per_task=1,
    remain_iterations_to_job_name=True,
    **kwargs
):
    schedule_plan = []
    for cmd in cmd_list:
        if isinstance(cmd, str):
            task = dict(
                remain_iterations=iterations_per_task,
                cmd=cmd,
                **kwargs
            )
        elif isinstance(cmd, dict):
            remain_iterations = cmd.get('iterations', iterations_per_task)
            k = kwargs.copy()
            k.update(cmd)
            task = dict(
                remain_iterations=remain_iterations,
                **k
            )
        else:
            raise NotImplementedError()
        schedule_plan.append(task)

    try:
        keep_run = True
        while keep_run:
            keep_run = False
            for task in schedule_plan:
                process = task.get('process')
                if task['remain_iterations'] >= 0 and not (process is not None and process.poll() is None):
                    if task['remain_iterations'] > 0:
                        t = task.copy()
                        job_name = t.pop('job_name', '')
                        if remain_iterations_to_job_name:
                            job_name = f'{task['remain_iterations'] - 1} {job_name}'

                        cmd2run = get_cmd(job_name=job_name, **t)

                        print(task['remain_iterations'], cmd2run)
                        process = subprocess.Popen(
                            cmd2run, 
                            shell=True,
                            env=task.get('env'),
                            cwd=task.get('cwd'),
                        )
                        task['process'] = process
                        if not is_slurm():
                            process.wait()
                    task['remain_iterations'] -= 1
                if task['remain_iterations'] >= 0:
                    keep_run = True
            if is_slurm():
                time.sleep(1)            
    finally:
        for task in schedule_plan:
            process = task.get('process')
            if process is not None:
                with suppress(Exception):
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
'''
from contextlib import suppress
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.model_registry import LLM_MODEL_REGISTRY, resolve_llm_model_name


# Resource presets
RESOURCE_PROFILES = {
    "cpu": {
        "partition": "apollo",
        "gpu_num": 0,
        "exclusive": True,
        "mem_gb": 0,
    },
    "gpu1": {
        "partition": "v100",
        "gpu_num": 1,
        "cpus": 8,
        "mem_gb": 40,
    },
    "gpu2": {
        "partition": "v100",
        "gpu_num": 2,
        "cpus": 8,
        "mem_gb": 80,
    },
    "gpu8": {
        "partition": "hiperf",
        "gpu_num": 8,
        "cpus": 96,
        "mem_gb": 300,
    },
}


DEFAULT_TARGETS = ["main_diag", "mkb_rubric3", "mkb_block", "mkb_chapter"]
DEFAULT_ML_MODELS = ["xgboost", "catboost", "random_forest", "logreg"]
DEFAULT_LLM_KEYS = list(LLM_MODEL_REGISTRY.keys())
if not DEFAULT_LLM_KEYS:
    DEFAULT_LLM_KEYS = ["qwen2.5-7b"]
DEFAULT_RAG_LLM_KEYS = list(DEFAULT_LLM_KEYS)
DEFAULT_HYBRID_LLM_KEYS = list(DEFAULT_LLM_KEYS)
DEFAULT_STATE_REL_PATH = Path("outputs") / "results" / "orchestrator_state.json"
DEFAULT_DATASET_DIR = str(
    Path("/home1")
    / (((os.getenv("USER") or "").strip()) or Path.home().name)
    / "datasets"
)
_SRUN_PARSABLE_SUPPORTED: bool | None = None

_SUCCESS_STATES = {"COMPLETED"}
_JOB_ID_PATTERNS = [
    re.compile(r"\bjob\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"^(\d+)(?:;[^\s]+)?$"),
]


def _csv(values):
    return ",".join(str(v) for v in values)


def _resolve_hf_model_name(llm_key: str, env: dict) -> str:
    return resolve_llm_model_name(
        llm_key,
        backend="transformers",
        env=env,
    )


def _hf_token(env: dict) -> str | None:
    for key in (
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
    ):
        value = (env.get(key) or "").strip()
        if value:
            return value
    return None


def _read_hf_token_from_file(path: Path) -> str | None:
    try:
        if not path.exists() or not path.is_file():
            return None
        token = path.read_text(encoding="utf-8").strip()
        return token or None
    except Exception:
        return None


def _hydrate_hf_token_env(env: dict) -> bool:
    token = _hf_token(env)
    if not token:
        candidates: list[Path] = []

        explicit_token_file = (env.get("HF_TOKEN_FILE") or "").strip()
        if explicit_token_file:
            candidates.append(Path(explicit_token_file).expanduser())

        hf_home_raw = (env.get("HF_HOME") or "").strip()
        if hf_home_raw:
            candidates.append(Path(hf_home_raw).expanduser() / "token")

        candidates.extend(
            [
                Path.home() / ".cache" / "huggingface" / "token",
                Path.home() / ".huggingface" / "token",
            ]
        )

        for candidate in candidates:
            token = _read_hf_token_from_file(candidate)
            if token:
                break

    if not token:
        return False

    env["HF_TOKEN"] = token
    env["HUGGINGFACE_HUB_TOKEN"] = token
    env["HUGGING_FACE_HUB_TOKEN"] = token
    env.setdefault("HUGGINGFACE_TOKEN", token)
    return True


def _assert_accessible_llm_keys(llm_keys: list[str], env: dict, stage: str) -> None:
    inaccessible: list[tuple[str, str]] = []
    for key in llm_keys:
        model_name = _resolve_hf_model_name(key, env)
        if not _hf_repo_accessible(model_name, env):
            inaccessible.append((key, model_name))

    if not inaccessible:
        return

    details = ", ".join(f"{k} -> {m}" for k, m in inaccessible)
    raise RuntimeError(
        f"{stage}: inaccessible HF model(s): {details}. "
        "Authenticate with huggingface-cli login (or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN) "
        "and ensure your account has access to gated repos."
    )


def _hf_repo_accessible(repo_name: str, env: dict) -> bool:
    # Non-HF names (local aliases, OpenAI compat names) are considered accessible.
    if "/" not in repo_name:
        return True

    token = _hf_token(env)

    # Primary check: use huggingface_hub API (same path as runtime user verification).
    try:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi()
        api.model_info(repo_name, token=token)
        return True
    except Exception as exc:
        status_code = None
        response = getattr(exc, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", None)
        if status_code is None:
            status_code = getattr(exc, "status_code", None)
        if status_code in (401, 403):
            return False
        # Continue with HTTP fallback for non-auth related exceptions.

    url = f"https://huggingface.co/{repo_name}/resolve/main/config.json"
    req = urllib_request.Request(url, method="HEAD")
    req.add_header("User-Agent", "neonatal-icd10-orchestrator/1.0")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib_request.urlopen(req, timeout=10):
            return True
    except urllib_error.HTTPError as exc:
        if exc.code in (401, 403):
            return False
        return True
    except Exception:
        # Do not block pipeline on transient network errors.
        return True


def _filter_inaccessible_llm_keys(llm_keys: list[str], env: dict, stage: str) -> list[str]:
    backend = (env.get("LLM_BACKEND") or "transformers").strip().lower()
    if backend != "transformers":
        return llm_keys

    filtered: list[str] = []
    for key in llm_keys:
        model_name = _resolve_hf_model_name(key, env)
        if _hf_repo_accessible(model_name, env):
            filtered.append(key)
            continue
        print(
            f"[WARN] Skip {stage} llm '{key}' (HF model '{model_name}') due to access 401/403. "
            "Set HF_TOKEN with approved access or remove this model from llm list."
        )
    return filtered


def _supports_srun_parsable() -> bool:
    global _SRUN_PARSABLE_SUPPORTED
    if _SRUN_PARSABLE_SUPPORTED is not None:
        return _SRUN_PARSABLE_SUPPORTED

    if not is_slurm():
        _SRUN_PARSABLE_SUPPORTED = False
        return False

    try:
        out = subprocess.check_output(["srun", "--help"], text=True, stderr=subprocess.STDOUT)
        _SRUN_PARSABLE_SUPPORTED = "--parsable" in out
    except Exception:
        _SRUN_PARSABLE_SUPPORTED = False
    return _SRUN_PARSABLE_SUPPORTED


def _resolve_task_env(task: dict, base_env: dict | None) -> dict:
    # Keep full process environment so shell, conda and Slurm tools work predictably.
    env = dict(os.environ)
    if base_env:
        env.update(base_env)
    task_env = task.get("env")
    if isinstance(task_env, dict):
        env.update(task_env)
    return env


def _resolve_task_cwd(task: dict, base_cwd: str | None, env: dict) -> str:
    if task.get("cwd"):
        return str(task["cwd"])
    if base_cwd:
        return str(base_cwd)
    if env.get("PROJECT_DIR"):
        return str(env["PROJECT_DIR"])
    return os.getcwd()


def _resolve_python_executable(task: dict, env: dict) -> str:
    for key in ("python_executable", "python_bin"):
        value = task.get(key)
        if value:
            return str(value)

    for key in ("PYTHON_EXECUTABLE", "PYTHON_BIN"):
        value = env.get(key)
        if value:
            return str(value)

    exe = sys.executable
    if exe:
        return str(exe)

    return shutil.which("python3") or shutil.which("python") or "python"


def _rewrite_python_command(cmd: str, python_executable: str) -> str:
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return cmd

    if not tokens:
        return cmd

    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok in {"python", "python3"}:
            tokens[idx] = python_executable
            return shlex.join(tokens)

        if "=" in tok and not tok.startswith("-"):
            key, _sep, _val = tok.partition("=")
            if key.isidentifier():
                idx += 1
                continue
        break

    return cmd


def _resolve_state_path(state_path: str | None, cwd: str | None, env: dict | None) -> Path:
    if state_path:
        return Path(state_path).expanduser().resolve()

    if env and env.get("PROJECT_DIR"):
        return (Path(env["PROJECT_DIR"]) / DEFAULT_STATE_REL_PATH).resolve()

    if cwd:
        return (Path(cwd) / DEFAULT_STATE_REL_PATH).resolve()

    return (Path.cwd() / DEFAULT_STATE_REL_PATH).resolve()


def _parse_env_line(raw_line: str) -> tuple[str | None, str | None]:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None, None
    if line.startswith("export "):
        line = line[len("export ") :].strip()
    if "=" not in line:
        return None, None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None, None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
        value = value[1:-1]
    return key, value


def _load_project_env_files(project_dir: str, env: dict, overwrite: bool = False) -> int:
    root = Path(project_dir).expanduser()
    loaded = 0
    for name in (".env.uran", ".env"):
        path = root / name
        if not path.exists() or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for raw in lines:
            key, value = _parse_env_line(raw)
            if key is None:
                continue
            if overwrite or key not in env:
                env[key] = value or ""
                loaded += 1
    return loaded


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "tasks": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"version": 1, "tasks": {}}

    if not isinstance(data, dict):
        return {"version": 1, "tasks": {}}
    if "tasks" not in data or not isinstance(data["tasks"], dict):
        data["tasks"] = {}
    if "version" not in data:
        data["version"] = 1
    return data


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def _arg_value(args: list[str], option: str, default: str | None = None) -> str | None:
    prefix = f"{option}="
    for i, tok in enumerate(args):
        if tok == option and i + 1 < len(args):
            return args[i + 1]
        if tok.startswith(prefix):
            return tok[len(prefix):]
    return default


def _extract_stage_from_cmd(cmd: str) -> tuple[str | None, list[str]]:
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return None, []

    for i, tok in enumerate(tokens):
        if tok.endswith("scripts/run_pipeline.py") and i + 1 < len(tokens):
            stage = tokens[i + 1]
            args = tokens[i + 2 :]
            if args and args[0] == "--":
                args = args[1:]
            return stage, args

    direct_map = {
        "run_clean_data.py": "clean",
        "run_build_kb.py": "kb",
        "run_ml.py": "ml",
        "run_rag.py": "rag",
        "run_hybrid.py": "hybrid",
        "run_statia_profile.py": "statia_profile",
        "run_article_tables.py": "article_tables",
        "run_holdout.py": "holdout",
        "parity_check.py": "parity",
    }
    for i, tok in enumerate(tokens):
        base = os.path.basename(tok)
        if base in direct_map:
            return direct_map[base], tokens[i + 1 :]

    return None, []


def _extract_job_id_from_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    for pattern in _JOB_ID_PATTERNS:
        m = pattern.search(stripped)
        if m:
            return m.group(1)
    return None


def _normalize_slurm_state(state: str | None) -> str | None:
    if not state:
        return None
    s = state.strip()
    if not s:
        return None
    s = s.split("+", 1)[0]
    s = s.split(" ", 1)[0]
    return s.upper()


def _query_slurm_state(job_id: str | None) -> str | None:
    if not job_id:
        return None

    if shutil.which("sacct"):
        try:
            out = subprocess.check_output(
                ["sacct", "-j", str(job_id), "--format=State", "--noheader", "--parsable2"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            states = []
            for line in out.splitlines():
                raw = line.split("|", 1)[0].strip()
                norm = _normalize_slurm_state(raw)
                if norm:
                    states.append(norm)
            if states:
                if "COMPLETED" in states:
                    return "COMPLETED"
                return states[0]
        except Exception:
            pass

    if shutil.which("scontrol"):
        try:
            out = subprocess.check_output(
                ["scontrol", "show", "job", str(job_id)],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            m = re.search(r"JobState=([^\s]+)", out)
            if m:
                return _normalize_slurm_state(m.group(1))
        except Exception:
            pass

    return None


def _task_status_success(entry: dict[str, Any], require_success_status: bool) -> bool:
    last_exit_code = entry.get("last_exit_code", 1)
    try:
        exit_code = int(last_exit_code)
    except (TypeError, ValueError):
        exit_code = 1

    if exit_code != 0:
        return False

    if not require_success_status:
        return True

    job_id = entry.get("last_job_id")
    if not job_id:
        # Older Slurm builds or fast-starting jobs may not print a job id in srun output.
        # Fall back to successful process exit in this case.
        return True

    state = _query_slurm_state(str(job_id))
    if state is None:
        # Accounting can lag; keep the successful process exit as fallback.
        return True
    return state in _SUCCESS_STATES


def _import_contracts():
    try:
        from _contracts import (  # type: ignore
            _csv_has_data,
            verify_clean_outputs,
            verify_hybrid_outputs,
            verify_kb_outputs,
            verify_ml_outputs,
            verify_rag_outputs,
        )
    except ModuleNotFoundError:
        from scripts._contracts import (  # type: ignore
            _csv_has_data,
            verify_clean_outputs,
            verify_hybrid_outputs,
            verify_kb_outputs,
            verify_ml_outputs,
            verify_rag_outputs,
        )
    return {
        "_csv_has_data": _csv_has_data,
        "verify_clean_outputs": verify_clean_outputs,
        "verify_kb_outputs": verify_kb_outputs,
        "verify_ml_outputs": verify_ml_outputs,
        "verify_rag_outputs": verify_rag_outputs,
        "verify_hybrid_outputs": verify_hybrid_outputs,
    }


def _guess_llm_key(job_name: str, args: list[str], prefix: str) -> str | None:
    llm = _arg_value(args, "--llm")
    if llm:
        return llm
    if job_name.startswith(prefix):
        guessed = job_name[len(prefix) :].strip()
        if guessed:
            return guessed
    return None


def _is_task_artifacts_complete(task: dict, env: dict, cwd: str) -> bool:
    contracts = _import_contracts()

    stage, stage_args = _extract_stage_from_cmd(str(task.get("cmd", "")))
    if not stage:
        return False

    project_dir = Path(env.get("PROJECT_DIR") or cwd).resolve()
    results_dir = project_dir / "outputs" / "results"
    clean_data_path = project_dir / "outputs" / "clean_data.csv"
    kb_dir = project_dir / "outputs" / "kb"

    try:
        if stage == "clean":
            expect_clean = "--no-save" not in stage_args
            expect_splits = "--no-split" not in stage_args
            targets = []
            if expect_splits:
                split_dir = results_dir / "splits"
                targets = [p.stem.replace("split_", "", 1) for p in split_dir.glob("split_*.csv")]
                if not targets:
                    return False

            contracts["verify_clean_outputs"](
                clean_data_path=clean_data_path,
                results_dir=results_dir,
                targets=targets,
                expect_clean=expect_clean,
                expect_splits=expect_splits,
            )
            return True

        if stage == "kb":
            kb_values = _parse_csv_arg(_arg_value(stage_args, "--kb"))
            variants = kb_values
            if not variants:
                variants = [p.name for p in kb_dir.iterdir() if p.is_dir()] if kb_dir.exists() else []
                if not variants:
                    return False
            contracts["verify_kb_outputs"](kb_dir, variants)
            return True

        if stage == "ml":
            contracts["verify_ml_outputs"](results_dir)
            return True

        if stage == "rag":
            llm_key = _guess_llm_key(str(task.get("job_name", "")), stage_args, "rag_")
            split = _arg_value(stage_args, "--split", default="test") or "test"
            if not llm_key:
                rag_candidates = list(results_dir.glob("rag_full_results__*.csv"))
                return any(contracts["_csv_has_data"](p) for p in rag_candidates)

            llm_result_key = llm_key if split == "all" else f"{llm_key}__{split}"
            try:
                contracts["verify_rag_outputs"](results_dir, llm_result_key)
            except SystemExit:
                if split == "test":
                    contracts["verify_rag_outputs"](results_dir, llm_key)
                else:
                    raise
            return True

        if stage == "hybrid":
            llm_key = _guess_llm_key(str(task.get("job_name", "")), stage_args, "hybrid_")
            if not llm_key:
                hyb_candidates = list(results_dir.glob("hybrid_results__*.csv"))
                return any(contracts["_csv_has_data"](p) for p in hyb_candidates)
            contracts["verify_hybrid_outputs"](results_dir, llm_key)
            return True

        if stage == "article_tables":
            required = [
                results_dir / "table2_model_comparison.csv",
                results_dir / "table5_rag_experiments.csv",
                results_dir / "table6_hybrid_results.csv",
                results_dir / "table7_final_comparison.csv",
                results_dir / "comparison_summary.csv",
            ]
            return all(contracts["_csv_has_data"](p) for p in required)

        if stage == "statia_profile":
            skip_rag = "--skip-rag" in stage_args
            skip_hybrid = "--skip-hybrid" in stage_args
            skip_tables = "--skip-tables" in stage_args

            required = []
            if not skip_rag:
                required.append(results_dir / "rag_full_results__statia_profile.csv")
            if not skip_hybrid:
                required.append(results_dir / "hybrid_results__statia_profile.csv")
            if not skip_tables:
                required.extend(
                    [
                        results_dir / "table2_model_comparison.csv",
                        results_dir / "table5_rag_experiments.csv",
                        results_dir / "table6_hybrid_results.csv",
                        results_dir / "table7_final_comparison.csv",
                        results_dir / "comparison_summary.csv",
                    ]
                )

            return bool(required) and all(contracts["_csv_has_data"](p) for p in required)

        if stage == "holdout":
            candidates = list(results_dir.glob("unified_holdout_results__*.csv"))
            return any(contracts["_csv_has_data"](p) for p in candidates)

        if stage == "parity":
            out = results_dir / "parity_summary.csv"
            return contracts["_csv_has_data"](out)

    except SystemExit:
        return False

    return False


def _has_known_stage(task: dict) -> bool:
    stage, _ = _extract_stage_from_cmd(str(task.get("cmd", "")))
    return stage is not None


def _run_command_with_streaming(cmd: str, env: dict | None, cwd: str | None) -> tuple[int, str | None]:
    process = subprocess.Popen(
        cmd,
        shell=True,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )

    job_id = None
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            if job_id is None:
                job_id = _extract_job_id_from_line(line)
    except KeyboardInterrupt:
        with suppress(Exception):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        raise
    finally:
        rc = process.wait()
    return rc, job_id


def _build_task_key(task: dict, index: int, cmd: str) -> str:
    base = str(task.get("task_key") or task.get("job_name") or f"task_{index}")
    return f"{index}:{base}:{cmd}"


def is_slurm():
    return shutil.which('srun') is not None


def get_cmd_prefix(
    job_name: str = None,
    gpu_num: int = 0,
    time: int = 20,
    cpus: int = None,
    mem_gb: int = None,
    cpu_per_gpu: int = 9,
    mem_per_gpu: int = 47,
    gpu_type: str = None,
    partition: str = None,
    max_gpus_per_node: int = 8,
    exclusive: bool = False,
    shard: bool = False,
    nodelist: str = '',
    nodes: int = 1,
    nice: int = 0,
    dependency: str = None,
    parsable: bool = False,
    export_env: str = "ALL",
    **kwargs
):
    cmd_prefix = ''
    if is_slurm():
        if cpus is None:
            cpus = int(max(1, gpu_num) * cpu_per_gpu)
            cpus = cpus - cpus % 2
        if mem_gb is None:
            mem_gb = int(max(1, gpu_num) * mem_per_gpu)

        if gpu_type is not None:
            gpu_type = f':{gpu_type}'
        else:
            gpu_type = ''

        gres_type = 'shard' if shard else 'gpu'
        exclusive = ((gpu_num == max_gpus_per_node) or exclusive) and not shard

        partition = f'-p {partition}' if partition is not None else ''
        gres = f'--gres={gres_type}{gpu_type}:{gpu_num}' if gpu_num else ''
        cpus_opt = f'--exclusive -c {cpus}' if exclusive else f'-c {cpus}'
        mem_opt = '--mem=0' if mem_gb == 0 else f'--mem={mem_gb}G'
        job_name = f"-J '{job_name}'" if job_name is not None else ''
        nl = f'-w {nodelist}' if len(nodelist) else ''
        N = f'-N {nodes}'
        nice = f'--nice={nice}' if nice else ''
        dep = f'--dependency={dependency}' if dependency else ''
        parsable_opt = '--parsable' if parsable and _supports_srun_parsable() else ''
        export_opt = f'--export={export_env}' if export_env else ''
        cmd_prefix = f'srun {parsable_opt} {nice} {dep} {export_opt} {partition} {cpus_opt} {mem_opt} {gres} {N} {nl} -t {time}:0:0 {job_name}'
    return cmd_prefix


def get_cmd(cmd, **kwargs):
    cmd_prefix = get_cmd_prefix(**kwargs)
    run_commmand = f"{cmd_prefix} {cmd} "
    return run_commmand


def _merge_task_resources(task: dict, default_profile: str = None, **kwargs) -> dict:
    merged = kwargs.copy()

    profile_name = task.get('profile', default_profile)
    if profile_name:
        if profile_name not in RESOURCE_PROFILES:
            raise ValueError(f"Unknown resource profile: {profile_name}")
        merged.update(RESOURCE_PROFILES[profile_name])

    # Per-task overrides should win over profile values.
    for key, value in task.items():
        if key in {
            'job_name', 'gpu_num', 'time', 'cpus', 'mem_gb', 'cpu_per_gpu',
            'mem_per_gpu', 'gpu_type', 'partition', 'max_gpus_per_node',
            'exclusive', 'shard', 'nodelist', 'nodes', 'nice', 'dependency',
            'parsable', 'export_env',
        }:
            merged[key] = value
    return merged


def build_default_pipeline_tasks(
    llm_key=None,
    ml_targets=None,
    ml_models=None,
    ml_gpu_profile='cpu',
    rag_llm_keys=None,
    rag_split='all',
    rag_kb='html_matched_icd',
    rag_retrieval_methods='dense',
    rag_index_types='flat',
    rag_similarities='cosine',
    rag_strategies='plain_topk',
    rag_max_samples=200,
    hybrid_llm_keys=None,
    hybrid_targets=None,
    hybrid_ml_model='xgboost',
    hybrid_kb='html_matched_icd',
    hybrid_max_samples=None,
    rag_gpu_profile='gpu1',
    hybrid_gpu_profile='gpu1',
    include_holdout=False,
    include_parity=False,
):
    if llm_key:
        rag_llm_keys = [llm_key]
        hybrid_llm_keys = [llm_key]

    ml_targets = list(ml_targets) if ml_targets is not None else list(DEFAULT_TARGETS)
    ml_models = list(ml_models) if ml_models is not None else list(DEFAULT_ML_MODELS)
    rag_llm_keys = list(rag_llm_keys) if rag_llm_keys is not None else list(DEFAULT_RAG_LLM_KEYS)
    hybrid_targets = list(hybrid_targets) if hybrid_targets is not None else list(DEFAULT_TARGETS)
    hybrid_llm_keys = list(hybrid_llm_keys) if hybrid_llm_keys is not None else list(DEFAULT_HYBRID_LLM_KEYS)

    ml_targets_csv = shlex.quote(_csv(ml_targets))
    ml_models_csv = shlex.quote(_csv(ml_models))
    rag_targets_csv = shlex.quote(_csv(DEFAULT_TARGETS))
    hybrid_targets_csv = shlex.quote(_csv(hybrid_targets))
    rag_kb_q = shlex.quote(rag_kb)
    rag_split_q = shlex.quote(rag_split)
    rag_methods_q = shlex.quote(rag_retrieval_methods)
    rag_index_q = shlex.quote(rag_index_types)
    rag_sim_q = shlex.quote(rag_similarities)
    rag_strat_q = shlex.quote(rag_strategies)
    hybrid_kb_q = shlex.quote(hybrid_kb)
    hybrid_ml_q = shlex.quote(hybrid_ml_model)

    tasks = [
        {'job_name': 'clean', 'cmd': 'python scripts/run_pipeline.py clean', 'profile': 'cpu'},
        {'job_name': 'kb', 'cmd': 'python scripts/run_pipeline.py kb', 'profile': 'cpu'},
        {
            'job_name': 'ml',
            'cmd': (
                'python scripts/run_pipeline.py ml -- '
                f'--targets {ml_targets_csv} --models {ml_models_csv}'
            ),
            'profile': ml_gpu_profile,
        },
    ]

    for rag_llm in rag_llm_keys:
        rag_llm_q = shlex.quote(rag_llm)
        rag_cmd = (
            'python scripts/run_pipeline.py rag -- '
            f'--llm {rag_llm_q} '
            f'--split {rag_split_q} '
            f'--targets {rag_targets_csv} '
            f'--kb {rag_kb_q} '
            f'--retrieval-methods {rag_methods_q} '
            f'--index-types {rag_index_q} '
            f'--similarities {rag_sim_q} '
            f'--strategies {rag_strat_q}'
        )
        if rag_max_samples is not None:
            rag_cmd += f' --max-samples {int(rag_max_samples)}'
        tasks.append(
            {
                'job_name': f'rag_{rag_llm}',
                'cmd': rag_cmd,
                'profile': rag_gpu_profile,
            }
        )

    for hyb_llm in hybrid_llm_keys:
        hyb_llm_q = shlex.quote(hyb_llm)
        hybrid_cmd = (
            'python scripts/run_pipeline.py hybrid -- '
            f'--llm {hyb_llm_q} '
            f'--targets {hybrid_targets_csv} '
            f'--ml-model {hybrid_ml_q} '
            f'--kb {hybrid_kb_q}'
        )
        if hybrid_max_samples is not None:
            hybrid_cmd += f' --max-samples {int(hybrid_max_samples)}'
        tasks.append(
            {
                'job_name': f'hybrid_{hyb_llm}',
                'cmd': hybrid_cmd,
                'profile': hybrid_gpu_profile,
            }
        )

    tasks.extend([
        {'job_name': 'article_tables', 'cmd': 'python scripts/run_pipeline.py article_tables', 'profile': 'cpu'},
    ])

    if include_holdout:
        holdout_llm = hybrid_llm_keys[0] if hybrid_llm_keys else DEFAULT_HYBRID_LLM_KEYS[0]
        holdout_llm_q = shlex.quote(holdout_llm)
        tasks.append(
            {
                'job_name': 'holdout',
                'cmd': f'python scripts/run_pipeline.py holdout -- --mode both --llm {holdout_llm_q}',
                'profile': 'cpu',
            }
        )
    if include_parity:
        tasks.append(
            {
                'job_name': 'parity',
                'cmd': 'python scripts/run_pipeline.py parity -- --threshold 0.05',
                'profile': 'cpu',
            }
        )
    return tasks


def build_statia_exact_pipeline_tasks(
    llm_qwen='qwen2.5-7b',
    llm_gemma='gemma2-9b',
    ml_targets=None,
    ml_models=None,
    ml_gpu_profile='cpu',
    hybrid_target='mkb_block',
    statia_gpu_profile='gpu1',
    include_holdout=False,
    include_parity=False,
):
    ml_targets = list(ml_targets) if ml_targets is not None else list(DEFAULT_TARGETS)
    ml_models = list(ml_models) if ml_models is not None else list(DEFAULT_ML_MODELS)

    ml_targets_csv = shlex.quote(_csv(ml_targets))
    ml_models_csv = shlex.quote(_csv(ml_models))
    llm_qwen_q = shlex.quote(llm_qwen)
    llm_gemma_q = shlex.quote(llm_gemma)
    hybrid_target_q = shlex.quote(hybrid_target)

    tasks = [
        {'job_name': 'clean', 'cmd': 'python scripts/run_pipeline.py clean', 'profile': 'cpu'},
        {'job_name': 'kb', 'cmd': 'python scripts/run_pipeline.py kb', 'profile': 'cpu'},
        {
            'job_name': 'ml',
            'cmd': (
                'python scripts/run_pipeline.py ml -- '
                f'--targets {ml_targets_csv} --models {ml_models_csv}'
            ),
            'profile': ml_gpu_profile,
        },
        {
            'job_name': 'statia_profile',
            'cmd': (
                'python scripts/run_pipeline.py statia_profile -- '
                f'--llm-qwen {llm_qwen_q} '
                f'--llm-gemma {llm_gemma_q} '
                f'--target {hybrid_target_q}'
            ),
            'profile': statia_gpu_profile,
        },
    ]

    if include_holdout:
        holdout_llm_q = shlex.quote(llm_gemma)
        tasks.append(
            {
                'job_name': 'holdout',
                'cmd': f'python scripts/run_pipeline.py holdout -- --mode both --llm {holdout_llm_q}',
                'profile': 'cpu',
            }
        )
    if include_parity:
        tasks.append(
            {
                'job_name': 'parity',
                'cmd': 'python scripts/run_pipeline.py parity -- --threshold 0.05',
                'profile': 'cpu',
            }
        )

    return tasks


def submit_default_pipeline(
    project_dir,
    dataset_dir,
    llm_key=None,
    ml_targets=None,
    ml_models=None,
    ml_gpu_profile='cpu',
    rag_llm_keys=None,
    rag_split='all',
    rag_kb='html_matched_icd',
    rag_retrieval_methods='dense',
    rag_index_types='flat',
    rag_similarities='cosine',
    rag_strategies='elimination',
    rag_max_samples=200,
    hybrid_llm_keys=None,
    hybrid_targets=None,
    hybrid_ml_model='xgboost',
    hybrid_kb='html_matched_icd',
    hybrid_max_samples=None,
    rag_gpu_profile='gpu1',
    hybrid_gpu_profile='gpu1',
    include_holdout=False,
    include_parity=False,
    env_extra=None,
    dry_run=False,
    resume=True,
    force=False,
    state_path=None,
    require_success_status=True,
    use_dependency=False,
    stop_on_error=True,
    auto_skip_inaccessible_llms=False,
):
    env = {
        'PROJECT_DIR': project_dir,
        'DATASET_DIR': dataset_dir,
    }
    if env_extra:
        env.update(env_extra)

    env_loaded_count = _load_project_env_files(project_dir, env, overwrite=False)

    if dataset_dir:
        env['DATASET_DIR'] = str(Path(str(dataset_dir)).expanduser())

    hf_token_loaded = _hydrate_hf_token_env(env)

    print(f"[DATA] DATASET_DIR={env.get('DATASET_DIR', '')}")
    print(f"[ENV] loaded_vars={env_loaded_count}")
    print(f"[HF] token={'loaded' if hf_token_loaded else 'missing'}")

    rag_keys = list(rag_llm_keys) if rag_llm_keys is not None else list(DEFAULT_RAG_LLM_KEYS)
    hyb_keys = list(hybrid_llm_keys) if hybrid_llm_keys is not None else list(DEFAULT_HYBRID_LLM_KEYS)
    if llm_key:
        rag_keys = [llm_key]
        hyb_keys = [llm_key]

    if auto_skip_inaccessible_llms:
        rag_keys = _filter_inaccessible_llm_keys(rag_keys, env, stage="rag")
        hyb_keys = _filter_inaccessible_llm_keys(hyb_keys, env, stage="hybrid")
        if not rag_keys:
            raise RuntimeError(
                "No accessible RAG LLMs after HF access check. "
                "Provide valid HF token access or disable inaccessible models explicitly."
            )
        if not hyb_keys:
            raise RuntimeError(
                "No accessible Hybrid LLMs after HF access check. "
                "Provide valid HF token access or disable inaccessible models explicitly."
            )
    else:
        _assert_accessible_llm_keys(rag_keys, env, stage="rag")
        _assert_accessible_llm_keys(hyb_keys, env, stage="hybrid")

    rag_llm_keys = rag_keys
    hybrid_llm_keys = hyb_keys

    tasks = build_default_pipeline_tasks(
        llm_key=llm_key,
        ml_targets=ml_targets,
        ml_models=ml_models,
        ml_gpu_profile=ml_gpu_profile,
        rag_llm_keys=rag_llm_keys,
        rag_split=rag_split,
        rag_kb=rag_kb,
        rag_retrieval_methods=rag_retrieval_methods,
        rag_index_types=rag_index_types,
        rag_similarities=rag_similarities,
        rag_strategies=rag_strategies,
        rag_max_samples=rag_max_samples,
        hybrid_llm_keys=hybrid_llm_keys,
        hybrid_targets=hybrid_targets,
        hybrid_ml_model=hybrid_ml_model,
        hybrid_kb=hybrid_kb,
        hybrid_max_samples=hybrid_max_samples,
        rag_gpu_profile=rag_gpu_profile,
        hybrid_gpu_profile=hybrid_gpu_profile,
        include_holdout=include_holdout,
        include_parity=include_parity,
    )

    return run_tasks(
        tasks,
        default_profile='cpu',
        cwd=project_dir,
        env=env,
        dry_run=dry_run,
        resume=resume,
        force=force,
        state_path=state_path,
        require_success_status=require_success_status,
        use_dependency=use_dependency,
        stop_on_error=stop_on_error,
    )


def submit_statia_exact_pipeline(
    project_dir,
    dataset_dir,
    llm_qwen='qwen2.5-7b',
    llm_gemma='gemma2-9b',
    ml_targets=None,
    ml_models=None,
    ml_gpu_profile='cpu',
    hybrid_target='mkb_block',
    statia_gpu_profile='gpu1',
    include_holdout=False,
    include_parity=False,
    env_extra=None,
    dry_run=False,
    resume=True,
    force=False,
    state_path=None,
    require_success_status=True,
    use_dependency=False,
    stop_on_error=True,
    auto_skip_inaccessible_llms=False,
):
    env = {
        'PROJECT_DIR': project_dir,
        'DATASET_DIR': dataset_dir,
    }
    if env_extra:
        env.update(env_extra)

    env_loaded_count = _load_project_env_files(project_dir, env, overwrite=False)

    if dataset_dir:
        env['DATASET_DIR'] = str(Path(str(dataset_dir)).expanduser())

    hf_token_loaded = _hydrate_hf_token_env(env)

    print(f"[DATA] DATASET_DIR={env.get('DATASET_DIR', '')}")
    print(f"[ENV] loaded_vars={env_loaded_count}")
    print(f"[HF] token={'loaded' if hf_token_loaded else 'missing'}")

    llm_keys = [llm_qwen, llm_gemma]
    if auto_skip_inaccessible_llms:
        filtered = _filter_inaccessible_llm_keys(llm_keys, env, stage="statia_exact")
        if len(filtered) < 2:
            raise RuntimeError(
                "statia_exact requires both Qwen and Gemma for 1:1 reproducibility. "
                "Provide valid HF token access for both models."
            )
    else:
        _assert_accessible_llm_keys(llm_keys, env, stage="statia_exact")

    tasks = build_statia_exact_pipeline_tasks(
        llm_qwen=llm_qwen,
        llm_gemma=llm_gemma,
        ml_targets=ml_targets,
        ml_models=ml_models,
        ml_gpu_profile=ml_gpu_profile,
        hybrid_target=hybrid_target,
        statia_gpu_profile=statia_gpu_profile,
        include_holdout=include_holdout,
        include_parity=include_parity,
    )

    return run_tasks(
        tasks,
        default_profile='cpu',
        cwd=project_dir,
        env=env,
        dry_run=dry_run,
        resume=resume,
        force=force,
        state_path=state_path,
        require_success_status=require_success_status,
        use_dependency=use_dependency,
        stop_on_error=stop_on_error,
    )


def run_tasks(
    cmd_list,
    iterations_per_task=1,
    remain_iterations_to_job_name=True,
    default_profile='cpu',
    resume=True,
    force=False,
    dry_run=False,
    state_path=None,
    require_success_status=True,
    use_dependency=False,
    stop_on_error=True,
    **kwargs
):
    schedule_plan = []
    for cmd in cmd_list:
        if isinstance(cmd, str):
            task = dict(
                remain_iterations=iterations_per_task,
                cmd=cmd,
                profile=default_profile,
                **kwargs
            )
        elif isinstance(cmd, dict):
            remain_iterations = cmd.get('iterations', iterations_per_task)
            k = kwargs.copy()
            k.update(cmd)
            task = dict(
                remain_iterations=remain_iterations,
                **k
            )
        else:
            raise NotImplementedError()
        schedule_plan.append(task)

    base_env = kwargs.get('env') if isinstance(kwargs.get('env'), dict) else None
    base_cwd = kwargs.get('cwd')
    state_file = _resolve_state_path(state_path, base_cwd, base_env)
    state = _load_state(state_file)
    task_state = state.setdefault('tasks', {})

    print(f"[STATE] {state_file}")
    results = []
    previous_job_id = None

    for index, task in enumerate(schedule_plan, start=1):
        cmd = str(task.get('cmd', '')).strip()
        if not cmd:
            raise ValueError(f"Task #{index} has empty cmd")

        total_iterations = int(task.get('remain_iterations', iterations_per_task) or 0)
        if total_iterations <= 0:
            results.append({'task': task.get('job_name', f'task_{index}'), 'status': 'skip_empty'})
            continue

        env = _resolve_task_env(task, base_env)
        cwd = _resolve_task_cwd(task, base_cwd, env)
        task_key = _build_task_key(task, index, cmd)
        python_executable = _resolve_python_executable(task, env)

        entry = task_state.get(task_key, {})
        completed_iterations = int(entry.get('completed_iterations', 0) or 0)
        completed_iterations = min(max(completed_iterations, 0), total_iterations)

        known_stage = _has_known_stage(task)
        artifacts_ready = _is_task_artifacts_complete(task, env=env, cwd=cwd) if known_stage else (
            completed_iterations >= total_iterations
        )
        status_ready = _task_status_success(entry, require_success_status=require_success_status)

        if force:
            completed_iterations = 0
        elif not resume:
            completed_iterations = 0
        elif completed_iterations >= total_iterations and artifacts_ready and status_ready:
            print(f"[SKIP] {task.get('job_name', f'task_{index}')}: already completed")
            results.append(
                {
                    'task': task.get('job_name', f'task_{index}'),
                    'status': 'skipped_done',
                    'iterations': total_iterations,
                    'job_id': entry.get('last_job_id'),
                    'task_key': task_key,
                }
            )
            continue
        elif completed_iterations >= total_iterations and (not artifacts_ready or not status_ready):
            print(
                f"[RERUN] {task.get('job_name', f'task_{index}')}: "
                "state indicates completion but artifacts/status are not valid"
            )
            completed_iterations = 0

        if completed_iterations > 0:
            print(
                f"[RESUME] {task.get('job_name', f'task_{index}')} "
                f"from iteration {completed_iterations}/{total_iterations}"
            )

        task_entry = {
            'task_key': task_key,
            'job_name': task.get('job_name', f'task_{index}'),
            'cmd': cmd,
            'cwd': cwd,
            'iterations_total': total_iterations,
            'completed_iterations': completed_iterations,
            'status': 'partial' if completed_iterations > 0 else 'pending',
            'last_exit_code': entry.get('last_exit_code'),
            'last_job_id': entry.get('last_job_id'),
            'updated_at': int(time.time()),
        }

        failed = False
        for iteration_idx in range(completed_iterations, total_iterations):
            t = task.copy()
            t.pop('remain_iterations', None)
            t.pop('process', None)
            t.pop('cmd', None)
            t.pop('job_name', None)
            t.pop('env', None)
            t.pop('cwd', None)
            t.pop('task_key', None)
            t.pop('iterations', None)

            base_job_name = str(task.get('job_name', f'task_{index}')).strip()
            job_name = base_job_name
            if remain_iterations_to_job_name:
                job_name = f"{iteration_idx} {base_job_name}".strip()

            profile_name = t.pop('profile', default_profile)
            if profile_name:
                t = _merge_task_resources({}, default_profile=profile_name, **t)

            if use_dependency and previous_job_id and is_slurm():
                t['dependency'] = f"afterok:{previous_job_id}"

            t['parsable'] = bool(is_slurm() and _supports_srun_parsable())
            t.setdefault('export_env', 'ALL')
            run_cmd = _rewrite_python_command(cmd, python_executable)
            cmd2run = get_cmd(cmd=run_cmd, job_name=job_name, **t).strip()

            if dry_run:
                print(f"[DRY-RUN] {job_name}: {cmd2run}")
                task_entry['completed_iterations'] = iteration_idx + 1
                task_entry['last_exit_code'] = 0
                task_entry['status'] = 'done' if (iteration_idx + 1) >= total_iterations else 'partial'
                task_entry['updated_at'] = int(time.time())
                task_state[task_key] = task_entry
                _save_state(state_file, state)
                continue

            print(f"[RUN] {job_name}: {cmd2run}")
            rc, job_id = _run_command_with_streaming(cmd2run, env=env or None, cwd=cwd)

            if job_id:
                previous_job_id = job_id
                print(f"[JOB] {job_name}: job_id={job_id}")

            task_entry['last_exit_code'] = rc
            task_entry['last_job_id'] = job_id or task_entry.get('last_job_id')
            task_entry['completed_iterations'] = iteration_idx + 1 if rc == 0 else iteration_idx
            task_entry['updated_at'] = int(time.time())

            if rc != 0:
                failed = True
                task_entry['status'] = 'failed' if task_entry['completed_iterations'] == 0 else 'partial'
                task_state[task_key] = task_entry
                _save_state(state_file, state)
                msg = f"Task '{base_job_name}' failed at iteration {iteration_idx + 1}/{total_iterations} (rc={rc})"
                print(f"[ERROR] {msg}")
                if stop_on_error:
                    raise RuntimeError(msg)
                break

            task_state[task_key] = task_entry
            _save_state(state_file, state)

        if dry_run:
            stage_complete = task_entry.get('completed_iterations', 0) >= total_iterations
            status_ok = True
        else:
            stage_complete = _is_task_artifacts_complete(task, env=env, cwd=cwd) if known_stage else (
                task_entry.get('completed_iterations', 0) >= total_iterations
                and int(task_entry.get('last_exit_code', 1)) == 0
            )
            status_ok = _task_status_success(task_entry, require_success_status=require_success_status)

        if not failed and task_entry.get('completed_iterations', 0) >= total_iterations and stage_complete and status_ok:
            task_entry['status'] = 'done'
        elif failed:
            task_entry['status'] = task_entry.get('status', 'failed')
        else:
            task_entry['status'] = 'partial'

        task_entry['updated_at'] = int(time.time())
        task_state[task_key] = task_entry
        _save_state(state_file, state)

        results.append(
            {
                'task': task_entry['job_name'],
                'status': task_entry['status'],
                'iterations': task_entry['completed_iterations'],
                'iterations_total': total_iterations,
                'job_id': task_entry.get('last_job_id'),
                'task_key': task_key,
            }
        )

        if task_entry['status'] != 'done' and stop_on_error:
            raise RuntimeError(f"Task '{task_entry['job_name']}' not completed (status={task_entry['status']})")

    return results


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    return parts or None


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orchestrator for full pipeline runs with resource profiles",
    )
    parser.add_argument(
        "--project-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root path",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Dataset root (default: /home1/$USER/datasets)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--state-path",
        default=None,
        help="Path to orchestrator state JSON",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore previous state and rerun tasks",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun even if artifacts already exist",
    )
    parser.add_argument(
        "--use-dependency",
        action="store_true",
        help="Use Slurm afterok dependency for sequential tasks",
    )
    parser.add_argument(
        "--no-require-success-status",
        action="store_true",
        help="Do not require COMPLETED status in state checks",
    )
    parser.add_argument(
        "--no-stop-on-error",
        action="store_true",
        help="Continue remaining tasks when one fails",
    )
    parser.add_argument(
        "--auto-skip-inaccessible-llms",
        action="store_true",
        help="Skip LLMs without HF access instead of failing",
    )

    subparsers = parser.add_subparsers(dest="pipeline")
    subparsers.required = True

    p_default = subparsers.add_parser(
        "default",
        help="Run default clean->kb->ml->rag->hybrid pipeline",
    )
    p_default.add_argument("--llm-key", default=None)
    p_default.add_argument("--ml-targets", default=None)
    p_default.add_argument("--ml-models", default=None)
    p_default.add_argument("--ml-gpu-profile", default="cpu")
    p_default.add_argument("--rag-llm-keys", default=None)
    p_default.add_argument("--rag-split", default="all")
    p_default.add_argument("--rag-kb", default="html_matched_icd")
    p_default.add_argument("--rag-retrieval-methods", default="dense")
    p_default.add_argument("--rag-index-types", default="flat")
    p_default.add_argument("--rag-similarities", default="cosine")
    p_default.add_argument("--rag-strategies", default="plain_topk")
    p_default.add_argument("--rag-max-samples", type=int, default=200)
    p_default.add_argument("--hybrid-llm-keys", default=None)
    p_default.add_argument("--hybrid-targets", default=None)
    p_default.add_argument("--hybrid-ml-model", default="xgboost")
    p_default.add_argument("--hybrid-kb", default="html_matched_icd")
    p_default.add_argument("--hybrid-max-samples", type=int, default=None)
    p_default.add_argument("--rag-gpu-profile", default="gpu1")
    p_default.add_argument("--hybrid-gpu-profile", default="gpu1")
    p_default.add_argument("--include-holdout", action="store_true")
    p_default.add_argument("--include-parity", action="store_true")

    p_statia = subparsers.add_parser(
        "statia_exact",
        help="Run statia-exact profile with Qwen + Gemma",
    )
    p_statia.add_argument("--llm-qwen", default="qwen2.5-7b")
    p_statia.add_argument("--llm-gemma", default="gemma2-9b")
    p_statia.add_argument("--ml-targets", default=None)
    p_statia.add_argument("--ml-models", default=None)
    p_statia.add_argument("--ml-gpu-profile", default="cpu")
    p_statia.add_argument("--hybrid-target", default="mkb_block")
    p_statia.add_argument("--statia-gpu-profile", default="gpu1")
    p_statia.add_argument("--include-holdout", action="store_true")
    p_statia.add_argument("--include-parity", action="store_true")

    return parser


def main() -> int:
    parser = _build_cli_parser()
    args = parser.parse_args()

    pipeline = args.pipeline
    project_dir = str(Path(args.project_dir).expanduser())
    dataset_dir = str(Path(args.dataset_dir).expanduser())

    common_kwargs = dict(
        project_dir=project_dir,
        dataset_dir=dataset_dir,
        dry_run=bool(args.dry_run),
        resume=not bool(args.no_resume),
        force=bool(args.force),
        state_path=args.state_path,
        require_success_status=not bool(args.no_require_success_status),
        use_dependency=bool(args.use_dependency),
        stop_on_error=not bool(args.no_stop_on_error),
        auto_skip_inaccessible_llms=bool(args.auto_skip_inaccessible_llms),
    )

    if pipeline == "statia_exact":
        result = submit_statia_exact_pipeline(
            llm_qwen=args.llm_qwen,
            llm_gemma=args.llm_gemma,
            ml_targets=_parse_csv_arg(args.ml_targets),
            ml_models=_parse_csv_arg(args.ml_models),
            ml_gpu_profile=args.ml_gpu_profile,
            hybrid_target=args.hybrid_target,
            statia_gpu_profile=args.statia_gpu_profile,
            include_holdout=bool(args.include_holdout),
            include_parity=bool(args.include_parity),
            **common_kwargs,
        )
    else:
        result = submit_default_pipeline(
            llm_key=args.llm_key,
            ml_targets=_parse_csv_arg(args.ml_targets),
            ml_models=_parse_csv_arg(args.ml_models),
            ml_gpu_profile=args.ml_gpu_profile,
            rag_llm_keys=_parse_csv_arg(args.rag_llm_keys),
            rag_split=args.rag_split,
            rag_kb=args.rag_kb,
            rag_retrieval_methods=args.rag_retrieval_methods,
            rag_index_types=args.rag_index_types,
            rag_similarities=args.rag_similarities,
            rag_strategies=args.rag_strategies,
            rag_max_samples=args.rag_max_samples,
            hybrid_llm_keys=_parse_csv_arg(args.hybrid_llm_keys),
            hybrid_targets=_parse_csv_arg(args.hybrid_targets),
            hybrid_ml_model=args.hybrid_ml_model,
            hybrid_kb=args.hybrid_kb,
            hybrid_max_samples=args.hybrid_max_samples,
            rag_gpu_profile=args.rag_gpu_profile,
            hybrid_gpu_profile=args.hybrid_gpu_profile,
            include_holdout=bool(args.include_holdout),
            include_parity=bool(args.include_parity),
            **common_kwargs,
        )

    print(f"[DONE] Orchestrator tasks: {len(result)}")
    for item in result:
        print(
            f"[TASK] {item.get('task')} status={item.get('status')} "
            f"iters={item.get('iterations')}/{item.get('iterations_total')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())