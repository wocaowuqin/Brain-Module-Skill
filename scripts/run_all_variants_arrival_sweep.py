#!/usr/bin/env python3
"""Run all requested baselines on the two-topology arrival-rate sweep.

The script is deliberately an experiment orchestrator rather than another
trainer.  A2C/PPO/DDQN are launched through their existing entry points and
the migrated Python MSFC-CE solver is evaluated in-process with an explicit
arrival/lifetime resource ledger.  Every task has its own directory, so old
results under ``Desktop/结果`` are never overwritten.

Examples (PowerShell)::

    python scripts/run_all_variants_arrival_sweep.py --dry-run
    python scripts/run_all_variants_arrival_sweep.py --limit-requests 100 \
        --algorithms all --continue-on-error
    python scripts/run_all_variants_arrival_sweep.py --algorithms all \
        --topologies us_backbone germany50 --rates 0.5 1.5 2.5

The default sweep is intentionally resumable.  Use ``--force`` only when a
completed task really needs to be recomputed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# When this file is executed as ``python scripts/<file>.py``, Python puts
# ``scripts`` (rather than the project root) on sys.path.  The migrated
# MSFC-CE adapter imports the local ``core`` package, so make that package
# location explicit and independent of the caller's working directory.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LEGACY_ROOT_DEFAULT = Path.home() / "Desktop" / "hrl"
DATA_ROOT_DEFAULT = PROJECT_ROOT / "data" / "rl_arrival_sweep_seed42"
OUTPUT_DEFAULT = PROJECT_ROOT / "artifacts" / "runs" / "all_variants_arrival_sweep"

TOPOLOGY = {
    "us_backbone": {
        "engine_name": "us_backbone",
        "mat": "US_Backbone_path.mat",
        "path_db": "US_Backbone_path.mat",
        "dc_nodes": [1, 3, 4, 7, 8, 9, 11, 12, 13, 14, 17, 18, 19, 20, 22, 23, 25, 26, 27, 28],
        "capacity": {"cpu": 55.0, "mem": 45.0, "bw": 90.0},
    },
    "germany50": {
        "engine_name": "50node",
        "mat": "50node.mat",
        "path_db": "path_db_50node.mat",
        # Keep the runner consistent with configs/topology.yaml (35 Germany50 DCs).
        "dc_nodes": [3, 4, 5, 6, 9, 11, 12, 14, 15, 17, 19, 20, 21, 22,
                      23, 24, 25, 26, 28, 29, 32, 33, 34, 35, 38, 39,
                      40, 42, 44, 45, 46, 47, 48, 49, 50],
        "capacity": {"cpu": 55.0, "mem": 45.0, "bw": 90.0},
    },
}

ALGORITHMS = (
    "msft_hirl",
    "msft_hrl",
    "msft_ilrl",
    "msft_hirl_gat",
    "msft_hirl_mlp",
    "msfc_ce",
    "ppo",
    "a2c",
    "ddqn",
    "d3qn",
)
HRL_VARIANTS = {
    # Figure labels mapped to the actual TA-HRL switches.
    "msft_hirl": ("full", "MSFT-HIRL"),
    "msft_hrl": ("no_il", "MSFT-HRL"),
    "msft_ilrl": ("single_dqn", "MSFT-ILRL"),
    "msft_hirl_gat": ("gat", "MSFT-HIRL-GAT"),
    "msft_hirl_mlp": ("mlp", "MSFT-HIRL-MLP"),
}
ALGORITHM_LABELS = {
    **{key: value[1] for key, value in HRL_VARIANTS.items()},
    "msfc_ce": "MSFC-CE",
    "ppo": "PPO",
    "a2c": "A2C",
    "ddqn": "DDQN",
    "d3qn": "D3QN",
}
# Backward-compatible CLI alias used by earlier smoke commands.
ALGORITHM_ALIASES = {"hrl": "msft_hirl"}
RATE_LABELS = {0.5: 4, 1.0: 8, 1.5: 12, 2.0: 16, 2.5: 20, 3.0: 24, 3.5: 28}
SCRIPT_BY_ALGORITHM = {
    "a2c": Path("model/A2C/a2ctrain.py"),
    "ppo": Path("model/PPO/ppotrain.py"),
    "ddqn": Path("model/DQN/Dqntrain_macro_legacy_backup.py"),
    "d3qn": Path("model/DQN/Dqntrain_macro_legacy_backup.py"),
}


def build_hrl_command(
    python: Path,
    algorithm: str,
    topology: str,
    rate: float,
    data: Path,
    output: Path,
    seed: int,
    gpu: int,
    keep_checkpoints: bool = False,
) -> list[str]:
    """Run one of the five current TA-HRL variants."""
    variant, _ = HRL_VARIANTS[algorithm]
    command = [
        str(python),
        str((PROJECT_ROOT / "train_tahrl.py").resolve()),
        "--phase", "phase3",
        "--ablation_variant", variant,
        "--topo", TOPOLOGY[topology]["engine_name"],
        "--data", str(data.resolve()),
        "--seed", str(seed),
        "--gpu", str(gpu),
        "--output_dir", str(output.resolve()),
        "--stable-output-dir",
    ]
    if not keep_checkpoints:
        command.append("--disable-checkpoint")
    return command


def rate_key(rate: float) -> str:
    # Dataset directories preserve the decimal part for integer-valued rates
    # (e.g. ``per_node_rate_1p0``).  Stripping trailing zeroes would turn
    # ``1.0`` into ``1`` and make otherwise existing datasets unreachable.
    value = float(rate)
    if value.is_integer():
        return f"{int(value)}p0"
    return str(value).rstrip("0").rstrip(".").replace(".", "p")


def same_rate(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) < 1e-9


def dataset_path(data_root: Path, topology: str, rate: float) -> Path:
    return data_root / topology / f"per_node_rate_{rate_key(rate)}" / "phase3_requests.pkl"


def limited_dataset(source: Path, output_root: Path, limit: int) -> Path:
    if limit <= 0:
        return source
    target = output_root / "_inputs" / f"{source.parent.parent.name}_{source.parent.name}_first{limit}.pkl"
    if target.exists():
        return target
    with source.open("rb") as stream:
        rows = pickle.load(stream)
    if not isinstance(rows, list):
        raise TypeError(f"expected a list dataset: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        pickle.dump(rows[:limit], stream, protocol=pickle.HIGHEST_PROTOCOL)
    return target


def parse_cost_summary(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) >= 2 and row[0].strip().lower() not in {"metric", ""}:
                result[row[0].strip()] = row[1].strip()
    return result


def parse_number(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip().replace("%", "")
        return float(text)
    except (TypeError, ValueError):
        return default


def failure_reason(log_file: Path, returncode: int | None = None) -> str:
    """Extract a concise actionable reason when a task has no metrics."""
    if not log_file.exists():
        return f"process_returncode={returncode}; log_missing"
    try:
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"process_returncode={returncode}; cannot_read_log={exc}"
    markers = (
        "Traceback", "Error", "Exception", "FileNotFound", "RuntimeError",
        "ValueError", "PermissionError", "FAILED", "失败", "不存在", "拒绝访问",
    )
    matches = [line.strip() for line in lines if any(marker in line for marker in markers)]
    if matches:
        return f"process_returncode={returncode}; " + " | ".join(matches[-3:])
    tail = [line.strip() for line in lines if line.strip()]
    if tail:
        return f"process_returncode={returncode}; log_tail=" + " | ".join(tail[-3:])
    return f"process_returncode={returncode}; empty_log"


def run_logged_process(
    command: list[str],
    cwd: Path,
    log,
    total_requests: int,
    env: dict[str, str],
    progress_root: Path | None = None,
) -> int:
    """Stream child output to the log and display a live progress bar."""
    # Windows PowerShell may expose a GBK stdout while child logs contain
    # UTF-8/replacement characters.  Printing the raw line can therefore
    # abort the whole sweep before metrics are written.  Keep the full line
    # in the UTF-8 log and make console rendering loss-tolerant.
    def console_line(value: str) -> str:
        try:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            return value.encode(encoding, errors="replace").decode(encoding, errors="replace")
        except Exception:
            return value.encode("ascii", errors="replace").decode("ascii", errors="replace")

    process = subprocess.Popen(
        command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def pump() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    started = time.perf_counter()
    completed = 0
    observed_total = max(0, int(total_requests))
    finished_reading = False
    last_render = 0.0

    def render(force: bool = False) -> None:
        nonlocal last_render
        now = time.perf_counter()
        if not force and now - last_render < 1.0:
            return
        last_render = now
        total = observed_total
        ratio = min(1.0, completed / total) if total else 0.0
        width = 28
        filled = int(width * ratio) if total else 0
        bar = "#" * filled + "." * (width - filled)
        count = f"{completed}/{total}" if total else f"{completed}/?"
        print(f"\rProgress [{bar}] {ratio * 100:6.2f}% {count} | elapsed {now - started:7.1f}s", end="", flush=True)

    def read_csv_progress() -> int:
        if progress_root is None or not progress_root.exists():
            return 0
        best = 0
        for path in progress_root.rglob("episodes.csv"):
            try:
                # The trainer flushes this file after each episode.
                rows = max(0, sum(1 for _ in path.open("r", encoding="utf-8", errors="replace")) - 1)
                best = max(best, min(rows, observed_total or rows))
            except (OSError, UnicodeError):
                continue
        return best

    while not finished_reading or process.poll() is None or not lines.empty():
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            completed = max(completed, read_csv_progress())
            render()
            continue
        if line is None:
            finished_reading = True
            render()
            continue
        log.write(line)
        log.flush()
        match = re.search(r"(?:Episode|episode|ep)\s*(\d+)\s*/\s*(\d+)", line)
        if match:
            completed = int(match.group(1))
            observed_total = int(match.group(2))
        else:
            # TA-HRL's periodic line is ``Ep 10: ...`` and has no denominator.
            match = re.search(r"\bEp\s*[:=]?\s*(\d+)\s*:", line, re.IGNORECASE)
            if match:
                completed = max(completed, int(match.group(1)))
            completed = max(completed, read_csv_progress())
        if "训练完成" in line or "程序执行完成" in line:
            completed = observed_total or completed
        if any(marker in line for marker in ("Episode", "episode", "训练完成", "ERROR", "WARNING", "失败")):
            print(f"\n{console_line(line.rstrip())}", flush=True)
        render()
    render(force=True)
    print(flush=True)
    return int(process.wait())


def load_topology_matrix(path: Path, topology: str) -> np.ndarray:
    import scipy.io as sio

    mat = sio.loadmat(path)
    if "adjacency" in mat:
        matrix = np.asarray(mat["adjacency"], dtype=float)
    elif "Adjacency" in mat:
        matrix = np.asarray(mat["Adjacency"], dtype=float)
    elif "topo" in mat:
        matrix = np.asarray(mat["topo"], dtype=float)
    elif "Paths" in mat:
        paths = mat["Paths"]
        n = paths.shape[0]
        matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                try:
                    distance = int(paths[i, j]["pathsdistance"].flat[0])
                    matrix[i, j] = float(distance == 1)
                except Exception:
                    pass
    else:
        raise ValueError(f"cannot find topology matrix in {path}")
    matrix = (matrix > 0).astype(float)
    np.fill_diagonal(matrix, 0.0)
    if topology == "us_backbone":
        matrix = np.maximum(matrix, matrix.T)
    return matrix


def build_subprocess_command(
    python: Path,
    legacy_root: Path,
    algorithm: str,
    topology: str,
    rate: float,
    data: Path,
    output: Path,
    seed: int,
    gpu: int,
) -> list[str]:
    command = [
        str(python),
        str((legacy_root / SCRIPT_BY_ALGORITHM[algorithm]).resolve()),
        "--topo", TOPOLOGY[topology]["engine_name"],
        "--data", str(data.resolve()),
        # The legacy CLIs validate this field against their historical labels
        # (8, 16, ...), while --data is authoritative for the half-rate
        # corpus.  Keep the compatibility label at 8 for every explicit file.
        "--rate", "8",
        "--seed", str(seed),
        "--gpu", str(gpu),
        "--output_dir", str(output.resolve()),
        "--passes", "1",
    ]
    if algorithm in ("ddqn", "d3qn"):
        command.append("--legacy_macro_ddqn")
        if algorithm == "d3qn":
            command.extend([
                "--dueling", "--safe_bias", "8.0",
                "--epsilon_start", "0.35", "--epsilon_end", "0.02",
                "--decay_fraction", "0.35", "--guided_exploration",
                "--future_feasibility_mask",
            ])
    return command


def task_complete(run_dir: Path, algorithm: str) -> bool:
    if algorithm in HRL_VARIANTS:
        run_dir = normalise_hrl_output(run_dir)
    dataset_dir = run_dir / "dataset"
    return (dataset_dir / "episodes.csv").exists() and (dataset_dir / "cost_summary.csv").exists()


def normalise_hrl_output(run_dir: Path) -> Path:
    """Restore the stable task directory after train_tahrl finalization.

    ``train_tahrl.py`` renames its phase-3 directory to include acceptance and
    tree length.  The sweep needs a stable path for resume/aggregation, so the
    newest finalized sibling is moved back to the requested task directory.
    """
    if (run_dir / "dataset" / "episodes.csv").exists():
        return run_dir
    nested = [
        item for item in run_dir.rglob("*")
        if item.is_dir() and (item / "dataset" / "episodes.csv").exists()
    ] if run_dir.exists() else []
    if nested:
        return max(nested, key=lambda item: item.stat().st_mtime)
    candidates = [
        item for item in run_dir.parent.glob(f"{run_dir.name}-*")
        if item.is_dir() and (item / "dataset" / "episodes.csv").exists()
    ]
    if not candidates:
        return run_dir
    source = max(candidates, key=lambda item: item.stat().st_mtime)
    try:
        shutil.move(str(source), str(run_dir))
        return run_dir
    except Exception:
        return source


def extract_tree_deltas(
    tree: dict,
    trajectory: list,
    request: dict,
    n_nodes: int,
    n_links: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract MATLAB-compatible resource deltas from a solved tree.

    The original ``serve_leave_request.m`` releases one instance of each VNF
    type per node by looking at ``hvt``.  Summing every trajectory snapshot
    counts the same placement once per multicast branch and artificially
    lowers acceptance.  Prefer the final hvt ledger and retain the trajectory
    path only as a compatibility fallback for old trees without hvt.
    """
    cpu = np.zeros(n_nodes, dtype=float)
    mem = np.zeros(n_nodes, dtype=float)
    hvt = np.asarray((tree or {}).get("hvt", np.zeros((0, 0))))
    vnf_types = [int(value) for value in request.get("vnf", [])]
    cpu_req = [float(value) for value in request.get("cpu_origin", [])]
    mem_req = [float(value) for value in request.get("memory_origin", [])]
    if hvt.ndim == 2 and hvt.size:
        for node_idx in range(min(n_nodes, hvt.shape[0])):
            for type_idx in range(min(hvt.shape[1], 8)):
                if hvt[node_idx, type_idx] == 0:
                    continue
                vnf_type = type_idx + 1
                try:
                    req_idx = vnf_types.index(vnf_type)
                except ValueError:
                    continue
                if req_idx < len(cpu_req):
                    cpu[node_idx] += cpu_req[req_idx]
                if req_idx < len(mem_req):
                    mem[node_idx] += mem_req[req_idx]
    else:
        for item in trajectory or []:
            if not isinstance(item, (tuple, list)) or len(item) < 3:
                continue
            state = item[2]
            if isinstance(state, dict):
                if "cpu" in state:
                    cpu += np.asarray(state["cpu"], dtype=float).reshape(-1)[:n_nodes]
                if "mem" in state:
                    mem += np.asarray(state["mem"], dtype=float).reshape(-1)[:n_nodes]
    links = np.asarray((tree or {}).get("tree", np.zeros(n_links)), dtype=float).reshape(-1)
    links = (links[:n_links] > 0).astype(float)
    # The solver reports sum_bw for the multicast tree; one request reserves
    # one bandwidth unit on every distinct tree edge.
    return cpu, mem, links, np.array([float(cpu.sum()), float(mem.sum()),
                                      float((tree or {}).get("sum_bw", links.sum()))])


def write_msfc_summary(run_dir: Path, rows: list[dict], accepted: int, elapsed: float) -> None:
    dataset_dir = run_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (dataset_dir / "episodes.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    total = len(rows)
    successes = [row for row in rows if int(row.get("success", 0)) == 1]
    metrics = {
        "total": total,
        "success": accepted,
        "acc": f"{accepted / max(1, total) * 100:.3f}%",
        # Keep blocking rate as a plain numeric percentage in CSV output.
        "block_rate": f"{(1 - accepted / max(1, total)) * 100:.3f}",
        "tree_len": f"{np.mean([row['tree_len'] for row in successes]):.3f}" if successes else "0",
        "cpu_cumsum": f"{sum(row['cpu_used_abs'] for row in successes):.3f}",
        "mem_cumsum": f"{sum(row['mem_used_abs'] for row in successes):.3f}",
        "bw_cumsum": f"{sum(row['bw_used_abs'] for row in successes):.3f}",
        "elapsed_s": f"{elapsed:.3f}",
    }
    with (dataset_dir / "cost_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())
    (run_dir / "msfc_ce_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run_msfc_ce(data: Path, run_dir: Path, topology: str, legacy_root: Path, limit: int) -> dict:
    """Run the migrated MSFC-CE with lifecycle-aware resource accounting."""
    from core.expert.expert_msfce.core.solver import MSFCE_Solver, SolverConfig

    with data.open("rb") as stream:
        requests = pickle.load(stream)
    if limit > 0:
        requests = requests[:limit]
    topo_cfg = TOPOLOGY[topology]
    topo_file = legacy_root / "topo" / topo_cfg["mat"]
    path_file = legacy_root / "topo" / topo_cfg["path_db"]
    matrix = load_topology_matrix(topo_file, topology)
    capacities = topo_cfg["capacity"]
    solver = MSFCE_Solver(
        path_db_file=path_file,
        topology_matrix=matrix,
        dc_nodes=topo_cfg["dc_nodes"],
        capacities={"cpu": capacities["cpu"], "memory": capacities["mem"], "bandwidth": capacities["bw"]},
        config=SolverConfig(candidate_set_size=8, k_path=5, max_time_seconds=2.0),
    )
    n_nodes = matrix.shape[0]
    link_count = int(np.triu(matrix > 0, 1).sum())
    cpu_avail = np.full(n_nodes, capacities["cpu"], dtype=float)
    mem_avail = np.full(n_nodes, capacities["mem"], dtype=float)
    bw_avail = np.full(link_count, capacities["bw"], dtype=float)
    # Solver's link IDs enumerate upper-triangle edges in row-major order.
    link_id: dict[tuple[int, int], int] = {}
    lid = 0
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            if matrix[u, v] > 0:
                link_id[(u + 1, v + 1)] = lid
                link_id[(v + 1, u + 1)] = lid
                lid += 1
    release_events: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    rows: list[dict] = []
    accepted = 0
    started = time.perf_counter()
    for request in sorted(requests, key=lambda item: float(item.get("arrival_time", 0.0))):
        arrival = float(request.get("arrival_time", 0.0))
        while release_events and release_events[0][0] <= arrival + 1e-9:
            _, rel_cpu, rel_mem, rel_bw = release_events.pop(0)
            cpu_avail = np.minimum(capacities["cpu"], cpu_avail + rel_cpu)
            mem_avail = np.minimum(capacities["mem"], mem_avail + rel_mem)
            bw_avail = np.minimum(capacities["bw"], bw_avail + rel_bw)
        # The migrated solver consumes zero-based requests and converts them
        # internally to its one-based path representation.
        solver_req = dict(request)
        solver_req["source"] = int(request["source"]) - 1
        solver_req["dest"] = [int(node) - 1 for node in request.get("dest", [])]
        net_state = {"cpu": cpu_avail.copy(), "mem": mem_avail.copy(), "bw": bw_avail.copy()}
        t0 = time.perf_counter()
        tree = None
        trajectory: list = []
        try:
            tree, trajectory = solver.solve_request_for_expert(solver_req, network_state=net_state)
        except Exception:
            tree = None
        cpu_delta, mem_delta, links, sums = extract_tree_deltas(
            tree or {}, trajectory, request, n_nodes, link_count
        )
        bw_delta = links * float(request.get("bw_origin", 0.0))
        covered_count = len((tree or {}).get("covered_dests", request.get("dest", [])))
        if tree is None:
            reason = "solver_no_tree"
        elif covered_count < len(request.get("dest", [])):
            reason = "destination_not_covered"
        elif not np.all(cpu_delta <= cpu_avail + 1e-6):
            reason = "cpu_capacity"
        elif not np.all(mem_delta <= mem_avail + 1e-6):
            reason = "memory_capacity"
        elif not np.all(bw_delta <= bw_avail + 1e-6):
            reason = "bandwidth_capacity"
        else:
            reason = "accepted"
        feasible = reason == "accepted"
        if feasible:
            cpu_avail -= cpu_delta
            mem_avail -= mem_delta
            bw_avail -= bw_delta
            leave = float(request.get("leave_time", arrival))
            release_events.append((leave, cpu_delta, mem_delta, bw_delta))
            release_events.sort(key=lambda item: item[0])
            accepted += 1
        used_cpu = float(cpu_delta.sum()) if feasible else 0.0
        used_mem = float(mem_delta.sum()) if feasible else 0.0
        used_bw = float(bw_delta.sum()) if feasible else 0.0
        row = {
            "request_id": request.get("id", len(rows) + 1),
            "success": int(feasible),
            "reason": reason,
            "tree_len": int(links.sum()) if feasible else 0,
            "cpu_used_abs": used_cpu,
            "mem_used_abs": used_mem,
            "bw_used_abs": used_bw,
            "cpu_util_pct": (1 - cpu_avail.mean() / capacities["cpu"]) * 100,
            "mem_util_pct": (1 - mem_avail.mean() / capacities["mem"]) * 100,
            "bw_util_pct": (1 - bw_avail.mean() / capacities["bw"]) * 100,
            "res_util": np.mean([
                1 - cpu_avail.mean() / capacities["cpu"],
                1 - mem_avail.mean() / capacities["mem"],
                1 - bw_avail.mean() / capacities["bw"],
            ]),
            "decision_ms": (time.perf_counter() - t0) * 1000,
            "arrival_time": arrival,
            "leave_time": float(request.get("leave_time", arrival)),
        }
        rows.append(row)
    elapsed = time.perf_counter() - started
    write_msfc_summary(run_dir, rows, accepted, elapsed)
    return {
        "status": "ok",
        "returncode": 0,
        "elapsed_s": elapsed,
        "success": accepted,
        "acceptance": accepted / max(1, len(rows)) * 100,
        "tree_len": np.mean([row["tree_len"] for row in rows if row["success"]]) if accepted else 0.0,
        "cpu_cumsum": sum(row["cpu_used_abs"] for row in rows),
        "mem_cumsum": sum(row["mem_used_abs"] for row in rows),
        "bw_cumsum": sum(row["bw_used_abs"] for row in rows),
    }


def append_summary(path: Path, row: dict) -> None:
    fields = [
        "started_at", "algorithm", "algorithm_label", "algorithm_semantics", "topology", "per_source_rate",
        "dataset", "request_limit", "status", "returncode", "elapsed_s", "success",
        "acceptance", "tree_len", "cpu_cumsum", "mem_cumsum", "bw_cumsum", "source",
        "output_dir", "log_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    # A task can first fail (for example because an environment is missing)
    # and later succeed after the caller fixes it.  Replace that task's old
    # row instead of appending duplicates, which keeps curves and reports
    # numerically reproducible across resumed runs.
    existing: list[dict] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            existing = list(csv.DictReader(stream))
    task_key = (str(row.get("topology")), str(row.get("per_source_rate")), str(row.get("algorithm")))
    existing = [item for item in existing if (str(item.get("topology")), str(item.get("per_source_rate")), str(item.get("algorithm"))) != task_key]
    existing.append(row)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: item.get(key, "") for key in fields} for item in existing)


def load_episode_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    # The three legacy baselines use slightly different column names.  Keep
    # one canonical schema for the aggregate plots without rewriting their
    # original files.
    for row in rows:
        row.setdefault("cpu_util_pct", row.get("cpu_used_pct", ""))
        row.setdefault("mem_util_pct", row.get("mem_used_pct", ""))
        if not row.get("bw_util_pct"):
            bw_req = parse_number(row.get("bw_req"))
            bw_cap = 90.0
            row["bw_util_pct"] = str(bw_req / bw_cap * 100.0) if bw_req else ""
        row.setdefault("cpu_used_abs", row.get("cpu_resource_comp", ""))
        row.setdefault("mem_used_abs", row.get("mem_resource_comp", ""))
        row.setdefault("bw_used_abs", row.get("bw_resource_comp", ""))
        if not row.get("res_util"):
            cpu = parse_number(row.get("cpu_util_pct")) / 100.0
            mem = parse_number(row.get("mem_util_pct")) / 100.0
            bw = parse_number(row.get("bw_util_pct")) / 100.0
            row["res_util"] = str(float(np.mean([cpu, mem, bw])))
    return rows


def build_plots(output_root: Path) -> None:
    """Aggregate per-run episode rows and write tables/PNG figures."""
    summary = output_root / "summary.csv"
    if not summary.exists():
        return
    with summary.open("r", encoding="utf-8-sig", newline="") as stream:
        runs = list(csv.DictReader(stream))
    curve_rows: list[dict] = []
    grouped: defaultdict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for run in runs:
        if run.get("status") != "ok":
            continue
        episode_path = Path(run["output_dir"]) / "dataset" / "episodes.csv"
        for index, item in enumerate(load_episode_rows(episode_path), 1):
            grouped[(run["topology"], run["per_source_rate"], run["algorithm"])].append(item)
    for (topology, rate, algorithm), rows in sorted(grouped.items()):
        def mean_key(name: str) -> float:
            values = [parse_number(row.get(name)) for row in rows if row.get(name, "") != ""]
            return float(np.mean(values)) if values else 0.0
        curve_rows.append({
            "topology": topology,
            "per_source_rate": rate,
            "algorithm": algorithm,
            "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm),
            "requests": len(rows),
            "mean_cpu_util_pct": mean_key("cpu_util_pct"),
            "mean_mem_util_pct": mean_key("mem_util_pct"),
            "mean_bw_util_pct": mean_key("bw_util_pct"),
            "mean_res_util": mean_key("res_util"),
            "mean_cpu_used_abs": mean_key("cpu_used_abs"),
            "mean_mem_used_abs": mean_key("mem_used_abs"),
            "mean_bw_used_abs": mean_key("bw_used_abs"),
        })
    curve_path = output_root / "average_resource_curves.csv"
    if curve_rows:
        with curve_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(curve_rows[0]))
            writer.writeheader()
            writer.writerows(curve_rows)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for topology in sorted({row["topology"] for row in curve_rows}):
        subset = [row for row in curve_rows if row["topology"] == topology]
        for metric, title, filename in [
            ("mean_cpu_util_pct", "Average CPU utilization (%)", "average_cpu"),
            ("mean_mem_util_pct", "Average MEM utilization (%)", "average_mem"),
            ("mean_bw_util_pct", "Average BW utilization (%)", "average_bw"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            for algorithm in ALGORITHMS:
                points = sorted((float(row["per_source_rate"]), float(row[metric])) for row in subset if row["algorithm"] == algorithm)
                if points:
                    ax.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker="o",
                        label=ALGORITHM_LABELS.get(algorithm, algorithm),
                    )
            ax.set_xlabel("Per-source arrival rate (req/s)")
            ax.set_ylabel(title)
            ax.set_title(f"{topology}: {title}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.savefig(output_root / f"{topology}_{filename}.png", dpi=180)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--algorithms", nargs="+", default=["all"], choices=[*ALGORITHMS, *ALGORITHM_ALIASES, "all"])
    parser.add_argument("--topologies", nargs="+", default=list(TOPOLOGY), choices=list(TOPOLOGY))
    parser.add_argument("--rates", nargs="+", type=float, default=[0.5, 1.5, 2.5])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--legacy-root", type=Path, default=LEGACY_ROOT_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--python", type=Path, default=None, help="Python executable for legacy baselines")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit-requests", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--keep-checkpoints", action="store_true",
        help="保留 HRL checkpoint；默认关闭以避免大量 .pth 占用磁盘",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    algorithms = list(ALGORITHMS) if "all" in args.algorithms else list(dict.fromkeys(
        ALGORITHM_ALIASES.get(item, item) for item in args.algorithms
    ))
    output_root = args.output.resolve()
    data_root = args.data_root.resolve()
    legacy_root = args.legacy_root.resolve()
    python = (args.python or (Path.home() / ".conda" / "envs" / "sfc_ppo" / "python.exe")).resolve()
    if not python.exists():
        python = Path(sys.executable).resolve()
    for rate in args.rates:
        if not any(same_rate(rate, known) for known in RATE_LABELS):
            raise ValueError(f"unsupported rate {rate}; choose from {sorted(RATE_LABELS)}")
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.csv"
    failures = 0
    for topology in args.topologies:
        for rate in args.rates:
            data = dataset_path(data_root, topology, rate)
            if not data.exists():
                raise FileNotFoundError(f"dataset not found: {data}; run generate_rl_arrival_sweep.py first")
            limited = limited_dataset(data, output_root, args.limit_requests)
            for algorithm in algorithms:
                run_dir = output_root / topology / f"per_node_rate_{rate_key(rate)}" / algorithm
                if task_complete(run_dir, algorithm) and not args.force:
                    print(f"skip complete: {topology} rate={rate} {algorithm}")
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                task_id = f"{topology}/rate={rate}/{algorithm}"
                print(f"[{task_id}] requests={'all' if not args.limit_requests else args.limit_requests}")
                with limited.open("rb") as dataset_stream:
                    request_total = len(pickle.load(dataset_stream))
                if args.dry_run:
                    continue
                started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                log_file = run_dir / "run.log"
                start = time.perf_counter()
                try:
                    if algorithm in HRL_VARIANTS:
                        command = build_hrl_command(
                            python, algorithm, topology, rate, limited, run_dir,
                            args.seed, args.gpu, args.keep_checkpoints,
                        )
                        with log_file.open("w", encoding="utf-8", newline="") as log:
                            log.write("COMMAND:\n" + " ".join(command) + "\n\n")
                            log.flush()
                            process_returncode = run_logged_process(
                command, PROJECT_ROOT, log, request_total,
                                {**os.environ, "PYTHONUNBUFFERED": "1"}, run_dir,
                            )
                        # train_tahrl finalizes by renaming the directory.
                        actual_run_dir = normalise_hrl_output(run_dir)
                        metrics = parse_cost_summary(actual_run_dir / "dataset" / "cost_summary.csv")
                        result = {
                            "status": "ok" if process_returncode == 0 and metrics else "failed",
                            "returncode": process_returncode,
                            "elapsed_s": time.perf_counter() - start,
                            "success": metrics.get("success", ""),
                            "acceptance": metrics.get("acc", ""),
                            "tree_len": metrics.get("tree_len", ""),
                            "cpu_cumsum": metrics.get("cpu_cumsum", ""),
                            "mem_cumsum": metrics.get("mem_cumsum", ""),
                            "bw_cumsum": metrics.get("bw_cumsum", ""),
                            "source": f"current_tahrl_{HRL_VARIANTS[algorithm][0]}_entrypoint",
                            "output_dir": str(actual_run_dir),
                        }
                        if result["status"] != "ok":
                            result["failure_reason"] = failure_reason(log_file, process_returncode)
                    elif algorithm == "msfc_ce":
                        result = run_msfc_ce(limited, run_dir, topology, legacy_root, 0)
                        result["source"] = "current_migrated_python_msfc_ce"
                    else:
                        command = build_subprocess_command(python, legacy_root, algorithm, topology, rate, limited, run_dir, args.seed, args.gpu)
                        with log_file.open("w", encoding="utf-8", newline="") as log:
                            log.write("COMMAND:\n" + " ".join(command) + "\n\n")
                            log.flush()
                            process_returncode = run_logged_process(
                                command, legacy_root, log, request_total,
                                {**os.environ, "PYTHONUNBUFFERED": "1"}, run_dir,
                            )
                        elapsed = time.perf_counter() - start
                        metrics = parse_cost_summary(run_dir / "dataset" / "cost_summary.csv")
                        result = {
                            "status": "ok" if process_returncode == 0 else "failed",
                            "returncode": process_returncode,
                            "elapsed_s": elapsed,
                            "success": metrics.get("success", ""),
                            "acceptance": metrics.get("acc", ""),
                            "tree_len": metrics.get("tree_len", ""),
                            "cpu_cumsum": metrics.get("cpu_cumsum", ""),
                            "mem_cumsum": metrics.get("mem_cumsum", ""),
                            "bw_cumsum": metrics.get("bw_cumsum", ""),
                            "source": f"current_{algorithm}_entrypoint",
                        }
                    result.update({
                        "started_at": started_at,
                        "algorithm": algorithm,
                        "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm),
                        "algorithm_semantics": (
                            f"tahrl_{HRL_VARIANTS[algorithm][0]}"
                            if algorithm in HRL_VARIANTS
                            else "migrated_python_msfce"
                            if algorithm == "msfc_ce"
                            else "ordinary_macro_action_d3qn"
                            if algorithm == "d3qn"
                            else "ordinary_macro_action_ddqn"
                            if algorithm == "ddqn"
                            else f"flat_{algorithm}"
                        ),
                        "topology": topology,
                        "per_source_rate": rate,
                        "dataset": str(limited),
                        "request_limit": args.limit_requests or "all",
                        "output_dir": result.get("output_dir", str(run_dir)),
                            "log_file": str(log_file) if algorithm != "msfc_ce" else "",
                    })
                    append_summary(summary_path, result)
                    print(f"[{task_id}] {result['status']} elapsed={float(result['elapsed_s']):.1f}s acceptance={result.get('acceptance', 'n/a')}")
                    if result.get("failure_reason"):
                        print(f"[{task_id}] failure_reason={result['failure_reason']}")
                    if result["status"] != "ok":
                        failures += 1
                        if not args.continue_on_error:
                            return int(result.get("returncode") or 1)
                except Exception as error:
                    failures += 1
                    row = {"started_at": started_at, "algorithm": algorithm, "algorithm_label": ALGORITHM_LABELS.get(algorithm, algorithm), "algorithm_semantics": "error", "topology": topology, "per_source_rate": rate, "dataset": str(limited), "request_limit": args.limit_requests or "all", "status": "failed", "returncode": 1, "elapsed_s": time.perf_counter() - start, "source": "error", "output_dir": str(run_dir), "log_file": str(log_file), "error": repr(error)}
                    append_summary(summary_path, row)
                    print(f"[{task_id}] failed: {error}")
                    if not args.continue_on_error:
                        raise
    build_plots(output_root)
    state = {"finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "algorithms": algorithms, "topologies": args.topologies, "rates": args.rates, "seed": args.seed, "request_limit": args.limit_requests or "all"}
    (output_root / "run_manifest.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
