"""Run curated project workflows through one safe command-line interface."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Iterable, Sequence

from . import __version__


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOWS = ROOT / "configs" / "workflows.yaml"
VALID_RISKS = {"read_only", "compute", "system"}
VALID_PLATFORMS = {"windows", "linux", "wsl"}
PRESET_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class WorkflowConfigError(ValueError):
    """Raised when the workflow catalog is malformed."""


@dataclass(frozen=True)
class Preset:
    name: str
    category: str
    description: str
    risk: str
    command: tuple[str, ...]
    required_paths: tuple[str, ...]
    required_modules: tuple[str, ...]
    platforms: tuple[str, ...]


def _string_list(value: Any, field: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if value is None and allow_empty:
        return ()
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a non-empty" if not allow_empty else "a"
        raise WorkflowConfigError(f"{field} must be {qualifier} list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise WorkflowConfigError(f"{field} must contain non-empty strings")
    return tuple(item.strip() for item in value)


def load_workflows(path: Path) -> dict[str, Preset]:
    try:
        import yaml
    except ImportError as exc:
        raise WorkflowConfigError(
            "PyYAML is required to read configs/workflows.yaml"
        ) from exc

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WorkflowConfigError(f"cannot read workflow config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise WorkflowConfigError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise WorkflowConfigError("workflow config must be a mapping with version: 1")
    raw_presets = raw.get("presets")
    if not isinstance(raw_presets, dict) or not raw_presets:
        raise WorkflowConfigError("workflow config must define at least one preset")

    presets: dict[str, Preset] = {}
    for name, value in raw_presets.items():
        if not isinstance(name, str) or not PRESET_NAME.fullmatch(name):
            raise WorkflowConfigError(f"invalid preset name: {name!r}")
        if not isinstance(value, dict):
            raise WorkflowConfigError(f"preset {name!r} must be a mapping")
        category = value.get("category")
        description = value.get("description")
        risk = value.get("risk", "compute")
        if not isinstance(category, str) or not category.strip():
            raise WorkflowConfigError(f"preset {name!r} needs a category")
        if not isinstance(description, str) or not description.strip():
            raise WorkflowConfigError(f"preset {name!r} needs a description")
        if risk not in VALID_RISKS:
            raise WorkflowConfigError(
                f"preset {name!r} risk must be one of {sorted(VALID_RISKS)}"
            )
        command = _string_list(
            value.get("command"), f"presets.{name}.command", allow_empty=False
        )
        required_paths = _string_list(
            value.get("required_paths"), f"presets.{name}.required_paths"
        )
        required_modules = _string_list(
            value.get("required_modules"), f"presets.{name}.required_modules"
        )
        platforms = _string_list(
            value.get("platforms"), f"presets.{name}.platforms"
        )
        unknown_platforms = set(platforms) - VALID_PLATFORMS
        if unknown_platforms:
            raise WorkflowConfigError(
                f"preset {name!r} has unknown platforms: {sorted(unknown_platforms)}"
            )
        presets[name] = Preset(
            name=name,
            category=category.strip(),
            description=" ".join(description.split()),
            risk=risk,
            command=command,
            required_paths=required_paths,
            required_modules=required_modules,
            platforms=platforms,
        )
    return presets


def host_platform() -> str:
    if os.name == "nt":
        return "windows"
    try:
        release = platform.release().lower()
    except OSError:
        release = ""
    return "wsl" if "microsoft" in release else "linux"


def resolve_project_path(value: str) -> Path:
    expanded = value.replace("{root}", str(ROOT))
    path = Path(expanded).expanduser()
    return path if path.is_absolute() else ROOT / path


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def preset_problems(preset: Preset) -> list[str]:
    problems: list[str] = []
    current = host_platform()
    if preset.platforms and current not in preset.platforms:
        problems.append(
            f"host platform {current!r} is not in {', '.join(preset.platforms)}"
        )
    for module in preset.required_modules:
        if not module_available(module):
            problems.append(f"missing Python module: {module}")
    for raw_path in preset.required_paths:
        path = resolve_project_path(raw_path)
        if not path.exists():
            problems.append(f"missing path: {path}")
    if preset.category == "mininet" and host_platform() == "windows":
        if shutil.which("wsl") is None:
            problems.append("wsl.exe is not available on PATH")
    return problems


def preset_status(preset: Preset) -> str:
    current = host_platform()
    if preset.platforms and current not in preset.platforms:
        return "unavailable"
    return "ready" if not preset_problems(preset) else "blocked"


def expand_command(command: Iterable[str], extra_args: Sequence[str] = ()) -> list[str]:
    replacements = {
        "{python}": sys.executable,
        "{root}": str(ROOT),
    }
    expanded = []
    for item in command:
        for marker, replacement in replacements.items():
            item = item.replace(marker, replacement)
        expanded.append(item)
    expanded.extend(extra_args)
    return expanded


def format_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def command_list(args: argparse.Namespace, presets: dict[str, Preset]) -> int:
    selected = [
        preset
        for preset in presets.values()
        if args.category is None or preset.category == args.category
    ]
    selected.sort(key=lambda item: (item.category, item.name))
    if args.json:
        payload = [
            {
                "name": preset.name,
                "category": preset.category,
                "risk": preset.risk,
                "status": preset_status(preset),
                "ready": preset_status(preset) == "ready",
                "problems": preset_problems(preset),
                "description": preset.description,
            }
            for preset in selected
        ]
        print(json.dumps(payload, indent=2))
        return 0
    if not selected:
        print("No matching presets.")
        return 0
    rows = []
    for preset in selected:
        rows.append(
            [
                preset.name,
                preset.category,
                preset.risk,
                preset_status(preset),
                preset.description,
            ]
        )
    _print_table(rows, ["PRESET", "CATEGORY", "RISK", "STATUS", "DESCRIPTION"])
    return 0


def _deep_wsl_check() -> tuple[bool, str]:
    if host_platform() != "windows":
        return True, "not needed on this host"
    if shutil.which("wsl") is None:
        return False, "wsl.exe not found"
    try:
        result = subprocess.run(
            ["wsl", "--status"],
            check=False,
            capture_output=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    raw_detail = result.stdout or result.stderr
    if b"\x00" in raw_detail:
        detail = raw_detail.decode("utf-16-le", errors="replace")
    else:
        detail = raw_detail.decode("utf-8", errors="replace")
    detail = detail.strip().replace("\x00", " ")
    detail = " ".join(detail.split())
    if len(detail) > 160:
        detail = detail[:157] + "..."
    return result.returncode == 0, detail or f"exit code {result.returncode}"


def command_doctor(
    args: argparse.Namespace,
    presets: dict[str, Preset],
    config_path: Path,
) -> int:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        checks.append(
            {"name": name, "ok": bool(ok), "detail": detail, "required": required}
        )

    version_ok = sys.version_info >= (3, 10)
    add("python", version_ok, f"{platform.python_version()} at {sys.executable}")
    add("project_root", ROOT.is_dir(), str(ROOT))
    add("workflow_config", config_path.is_file(), str(config_path))

    modules = sorted(
        {"yaml"}.union(
            module for preset in presets.values() for module in preset.required_modules
        )
    )
    for module in modules:
        add(f"module:{module}", module_available(module), "importable")

    status_counts = {"ready": 0, "blocked": 0, "unavailable": 0}
    for preset in sorted(presets.values(), key=lambda item: item.name):
        problems = preset_problems(preset)
        supported = not preset.platforms or host_platform() in preset.platforms
        status_counts[preset_status(preset)] += 1
        add(
            f"preset:{preset.name}",
            not problems,
            "; ".join(problems) if problems else "ready",
            required=supported,
        )

    if args.deep and any(preset.category == "mininet" for preset in presets.values()):
        ok, detail = _deep_wsl_check()
        add("wsl_status", ok, detail, required=False)

    failed_required = [row for row in checks if row["required"] and not row["ok"]]
    payload = {
        "ok": not failed_required,
        "version": __version__,
        "host_platform": host_platform(),
        "checks": checks,
        "preset_count": len(presets),
        "preset_status": status_counts,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for row in checks:
            if row["ok"]:
                marker = "PASS"
            elif row["required"]:
                marker = "FAIL"
            else:
                marker = "SKIP"
            print(f"[{marker}] {row['name']}: {row['detail']}")
        summary = "ready" if payload["ok"] else "needs attention"
        print(
            f"Doctor result: {summary}; {status_counts['ready']}/{len(presets)} "
            "presets ready, "
            f"{status_counts['blocked']} blocked, "
            f"{status_counts['unavailable']} unavailable on this host."
        )
    return 0 if payload["ok"] else 1


def command_run(args: argparse.Namespace, presets: dict[str, Preset]) -> int:
    preset = presets.get(args.preset)
    if preset is None:
        available = ", ".join(sorted(presets))
        print(f"Unknown preset {args.preset!r}. Available: {available}", file=sys.stderr)
        return 2

    extra_args = list(args.extra_args)
    command = expand_command(preset.command, extra_args)
    problems = preset_problems(preset)

    print(f"Preset: {preset.name}")
    print(f"Category: {preset.category}")
    print(f"Risk: {preset.risk}")
    print(f"Working directory: {ROOT}")
    print(f"Command: {format_command(command)}")
    sys.stdout.flush()
    if problems:
        print("Blocked:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    if not args.execute:
        print("DRY RUN: no process was started.")
        suffix = " --confirm-system-changes" if preset.risk == "system" else ""
        print(f"Execute with: python -m sfc_project run {preset.name} --execute{suffix}")
        return 0

    if preset.risk == "system" and not args.confirm_system_changes:
        print(
            "Refusing to start a system workflow without --confirm-system-changes.",
            file=sys.stderr,
        )
        return 2

    print("Starting workflow...", flush=True)
    try:
        completed = subprocess.run(command, cwd=ROOT, check=False)
    except FileNotFoundError as exc:
        print(f"Failed to start workflow: {exc}", file=sys.stderr)
        return 127
    except KeyboardInterrupt:
        print("Workflow interrupted.", file=sys.stderr)
        return 130
    print(f"Workflow exited with code {completed.returncode}.")
    return int(completed.returncode)


def command_snapshot(
    args: argparse.Namespace, presets: dict[str, Preset], config_path: Path
) -> int:
    """Write an immutable-ish manifest of code/config state for an experiment."""
    name = str(args.name).strip()
    if not PRESET_NAME.fullmatch(name):
        print("snapshot name must match [a-z0-9][a-z0-9_-]*", file=sys.stderr)
        return 2
    output = args.output.expanduser()
    if not output.is_absolute():
        output = ROOT / output
    target = output / name
    target.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    manifest = {
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "project_root": str(ROOT),
        "workflow_config": str(config_path),
        "workflow_config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "presets": {key: {"category": value.category, "risk": value.risk, "command": list(value.command)} for key, value in presets.items()},
        "stage": "migration_exp_v1",
        "wqmix_safety_guard": False,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (target / "workflows.yaml").write_text(config_text, encoding="utf-8")
    print(json.dumps({"snapshot": str(target), **manifest}, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sfc_project",
        description="Inspect and run curated HRL/SFC workflows safely.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m sfc_project doctor\n"
            "  python -m sfc_project list\n"
            "  python -m sfc_project run simulation-smoke\n"
            "  python -m sfc_project run simulation-smoke --execute -- --max-requests 10\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_WORKFLOWS,
        help="workflow catalog (default: configs/workflows.yaml)",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    doctor = subparsers.add_parser("doctor", help="check runtime and preset inputs")
    doctor.add_argument("--deep", action="store_true", help="also query WSL status")
    doctor.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    listing = subparsers.add_parser("list", help="list curated workflow presets")
    listing.add_argument("--category", help="show only one category")
    listing.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    snapshot = subparsers.add_parser("snapshot", help="freeze a reproducible experiment snapshot")
    snapshot.add_argument("--name", required=True, help="snapshot name")
    snapshot.add_argument("--output", type=Path, default=ROOT / "artifacts" / "snapshots")

    run = subparsers.add_parser("run", help="show or execute one workflow preset")
    run.add_argument("preset", help="preset name from the list command")
    run.add_argument(
        "--execute",
        action="store_true",
        help="start the process; without this flag run is always a dry run",
    )
    run.add_argument(
        "--confirm-system-changes",
        action="store_true",
        help="required with --execute for Mininet/WSL service workflows",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw_args:
        delimiter = raw_args.index("--")
        cli_args = raw_args[:delimiter]
        extra_args = raw_args[delimiter + 1 :]
    else:
        cli_args = raw_args
        extra_args = []
    args = parser.parse_args(cli_args)
    if extra_args and args.subcommand != "run":
        parser.error("arguments after '--' are supported only by the run command")
    args.extra_args = extra_args
    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        presets = load_workflows(config_path)
    except WorkflowConfigError as exc:
        print(f"Workflow configuration error: {exc}", file=sys.stderr)
        return 2

    if args.subcommand == "doctor":
        return command_doctor(args, presets, config_path)
    if args.subcommand == "list":
        return command_list(args, presets)
    if args.subcommand == "snapshot":
        return command_snapshot(args, presets, config_path)
    if args.subcommand == "run":
        return command_run(args, presets)
    parser.error(f"unknown subcommand: {args.subcommand}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
