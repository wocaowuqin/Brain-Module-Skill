#!/usr/bin/env python3
"""Run A2C, PPO, and ordinary macro-action DDQN on one request corpus."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_ROOT = Path.home() / "Desktop" / "hrl"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "rl_arrival_sweep_seed42"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "runs" / "rl_arrival_sweep_seed2026"

ALGORITHM_SCRIPTS = {
    "a2c": Path("model/A2C/a2ctrain.py"),
    "ppo": Path("model/PPO/ppotrain.py"),
    "dqn": Path("model/DQN/Dqntrain_macro_legacy_backup.py"),
}
ENGINE_TOPOLOGY_NAMES = {"us_backbone": "us_backbone", "germany50": "50node"}


def rate_key(value: float) -> str:
    return str(value).replace(".", "p")


def data_path(root: Path, topology: str, rate: float) -> Path:
    return root / topology / f"per_node_rate_{rate_key(rate)}" / "phase3_requests.pkl"


def limited_data(source: Path, output_root: Path, limit: int) -> Path:
    if limit <= 0:
        return source
    target = output_root / "_inputs" / f"{source.parent.parent.name}_{source.parent.name}_first{limit}.pkl"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream:
        rows = pickle.load(stream)
    if not isinstance(rows, list):
        raise TypeError(f"expected list dataset: {source}")
    with target.open("wb") as stream:
        pickle.dump(rows[:limit], stream, protocol=pickle.HIGHEST_PROTOCOL)
    return target


def parse_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) >= 2 and row[0] != "metric":
                result[row[0]] = row[1]
    return result


def build_command(
    python: Path,
    legacy_root: Path,
    algorithm: str,
    topology: str,
    dataset: Path,
    output: Path,
    seed: int,
    gpu: int,
) -> list[str]:
    script = legacy_root / ALGORITHM_SCRIPTS[algorithm]
    command = [
        str(python), str(script),
        "--topo", ENGINE_TOPOLOGY_NAMES[topology],
        "--data", str(dataset),
        "--rate", "8",
        "--seed", str(seed),
        "--gpu", str(gpu),
        "--output_dir", str(output),
        "--passes", "1",
    ]
    if algorithm == "dqn":
        command += ["--legacy_macro_ddqn"]
    return command


def append_csv(path: Path, row: dict) -> None:
    fields = [
        "started_at", "algorithm", "algorithm_semantics", "topology",
        "per_source_rate", "dataset", "request_limit", "status", "returncode",
        "elapsed_s", "acceptance", "success", "tree_len", "cpu_cumsum",
        "mem_cumsum", "bw_cumsum", "output_dir", "log_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", choices=sorted(ALGORITHM_SCRIPTS), default=["a2c", "ppo"])
    parser.add_argument("--topologies", nargs="+", choices=sorted(ENGINE_TOPOLOGY_NAMES), default=sorted(ENGINE_TOPOLOGY_NAMES))
    parser.add_argument("--rates", nargs="+", type=float, default=[0.5, 1.5, 2.5])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=Path.home() / ".conda/envs/sfc_ppo/python.exe")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit-requests", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output.resolve()
    legacy_root = args.legacy_root.resolve()
    data_root = args.data_root.resolve()
    python = args.python.resolve()
    if not python.exists():
        raise FileNotFoundError(f"Python environment not found: {python}")
    for algorithm in args.algorithms:
        script = legacy_root / ALGORITHM_SCRIPTS[algorithm]
        if not script.exists():
            raise FileNotFoundError(f"baseline script not found: {script}")

    summary_csv = output_root / "sweep_summary.csv"
    state_path = output_root / "sweep_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    failures = 0
    for topology in args.topologies:
        for rate in args.rates:
            source_data = data_path(data_root, topology, rate)
            if not source_data.exists():
                raise FileNotFoundError(
                    f"dataset not found: {source_data}; run generate_rl_arrival_sweep.py first"
                )
            dataset = limited_data(source_data, output_root, args.limit_requests)
            for algorithm in args.algorithms:
                task_id = f"{algorithm}_{topology}_per_node_{rate_key(rate)}"
                if args.limit_requests > 0:
                    task_id += f"_first{args.limit_requests}"
                run_dir = output_root / topology / f"per_node_rate_{rate_key(rate)}" / algorithm
                result_file = run_dir / "dataset" / "cost_summary.csv"
                if result_file.exists() and not args.force:
                    print(f"skip existing {task_id}: {result_file}")
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                log_file = run_dir / "run.log"
                command = build_command(
                    python, legacy_root, algorithm, topology, dataset,
                    run_dir, args.seed, args.gpu,
                )
                print(f"[{task_id}] {' '.join(command)}")
                if args.dry_run:
                    continue
                started = datetime.now().isoformat(timespec="seconds")
                start = time.perf_counter()
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                with log_file.open("w", encoding="utf-8", newline="") as log:
                    log.write("COMMAND:\n" + " ".join(command) + "\n\n")
                    log.flush()
                    process = subprocess.run(
                        command,
                        cwd=str(legacy_root),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                    )
                elapsed = time.perf_counter() - start
                metrics = parse_summary(result_file)
                semantics = "ordinary_macro_action_ddqn" if algorithm == "dqn" else f"flat_{algorithm}"
                row = {
                    "started_at": started,
                    "algorithm": algorithm,
                    "algorithm_semantics": semantics,
                    "topology": topology,
                    "per_source_rate": rate,
                    "dataset": str(dataset),
                    "request_limit": args.limit_requests or "all",
                    "status": "ok" if process.returncode == 0 else "failed",
                    "returncode": process.returncode,
                    "elapsed_s": f"{elapsed:.3f}",
                    "acceptance": metrics.get("acc", ""),
                    "success": metrics.get("success", ""),
                    "tree_len": metrics.get("tree_len", ""),
                    "cpu_cumsum": metrics.get("cpu_cumsum", ""),
                    "mem_cumsum": metrics.get("mem_cumsum", ""),
                    "bw_cumsum": metrics.get("bw_cumsum", ""),
                    "output_dir": str(run_dir),
                    "log_file": str(log_file),
                }
                append_csv(summary_csv, row)
                state[task_id] = row
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                print(
                    f"[{task_id}] {row['status']} elapsed={elapsed:.1f}s "
                    f"acceptance={row['acceptance'] or 'n/a'}"
                )
                if process.returncode != 0:
                    failures += 1
                    if not args.continue_on_error:
                        return process.returncode or 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
