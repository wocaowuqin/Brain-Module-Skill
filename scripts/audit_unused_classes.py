#!/usr/bin/env python3
"""Statically classify Python classes by project-wide identifier references."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git",
    ".idea",
    "__pycache__",
    "data",
    "artifacts",
    "logs",
    "outputs",
    "tmp",
    "work",
}


@dataclass(frozen=True)
class ClassDefinition:
    name: str
    path: str
    line: int
    bases: tuple[str, ...]


def python_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.py")
        if not any(part in EXCLUDED_DIRS for part in path.relative_to(ROOT).parts)
    )


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def is_test_path(path: str) -> bool:
    name = Path(path).name.lower()
    return name.startswith(("test_", "check_")) or "backup" in name or "备份" in name


def audit() -> dict[str, Any]:
    definitions: list[ClassDefinition] = []
    references: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parse_errors = []
    for path in python_files():
        relative = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (SyntaxError, UnicodeDecodeError) as exc:
            parse_errors.append({"path": relative, "error": str(exc)})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                definitions.append(
                    ClassDefinition(
                        name=node.name,
                        path=relative,
                        line=node.lineno,
                        bases=tuple(filter(None, (dotted_name(value) for value in node.bases))),
                    )
                )
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                references[node.id].append({"path": relative, "line": node.lineno})
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                references[node.attr].append({"path": relative, "line": node.lineno})

    rows = []
    framework_bases = {"ControllerBase", "SimpleSwitch13", "RyuApp"}
    for definition in sorted(definitions, key=lambda value: (value.path, value.line)):
        refs = references.get(definition.name, [])
        external = [value for value in refs if value["path"] != definition.path]
        non_test = [value for value in refs if not is_test_path(value["path"])]
        if any(base.rsplit(".", 1)[-1] in framework_bases for base in definition.bases):
            category = "framework_entry"
        elif not refs:
            category = "unreferenced_candidate"
        elif not non_test:
            category = "test_only_reference"
        elif not external:
            category = "same_file_only"
        else:
            category = "referenced"
        rows.append(
            {
                "class": definition.name,
                "path": definition.path,
                "line": definition.line,
                "bases": list(definition.bases),
                "category": category,
                "reference_count": len(refs),
                "external_reference_count": len(external),
                "reference_preview": refs[:5],
            }
        )
    counts = defaultdict(int)
    for row in rows:
        counts[row["category"]] += 1
    return {
        "files_scanned": len(python_files()),
        "classes": len(rows),
        "category_counts": dict(sorted(counts.items())),
        "parse_errors": parse_errors,
        "results": rows,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Static Class Usage Audit",
        "",
        "This is a conservative static-reference audit. Dynamic imports, framework",
        "discovery, serialization, and external scripts can make an unreferenced class",
        "necessary. Review candidates before deletion.",
        "",
        f"- Python files scanned: {report['files_scanned']}",
        f"- Classes found: {report['classes']}",
        f"- Parse errors: {len(report['parse_errors'])}",
        "",
    ]
    labels = (
        ("unreferenced_candidate", "Unreferenced Candidates"),
        ("test_only_reference", "Referenced Only By Checks/Backups"),
        ("same_file_only", "Same-File-Only Classes"),
        ("framework_entry", "Framework Entry Classes (Keep)"),
    )
    for category, title in labels:
        values = [row for row in report["results"] if row["category"] == category]
        lines.extend((f"## {title}", ""))
        if not values:
            lines.extend(("None.", ""))
            continue
        lines.extend(("| Class | Location | References |", "|---|---|---:|"))
        for row in values:
            lines.append(
                f"| `{row['class']}` | `{row['path']}:{row['line']}` | {row['reference_count']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        default="artifacts/reports/maintenance/unused_class_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="artifacts/reports/maintenance/unused_class_audit.md",
    )
    args = parser.parse_args()
    report = audit()
    json_path = ROOT / args.json_output
    markdown_path = ROOT / args.markdown_output
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(markdown_path),
                "category_counts": report["category_counts"],
                "parse_errors": report["parse_errors"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
