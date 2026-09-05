from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "CLASS_CATALOG.md"
SKIP = {"__pycache__", ".venv", "artifacts", "legacy_hrl_runtime", "tests"}

def sentence(node: ast.ClassDef) -> str:
    doc = ast.get_docstring(node)
    if doc:
        return " ".join(doc.split()).split(".")[0].strip() + "."
    name = node.name
    return f"代码类 `{name}`，用于承载该文件中的相关数据或逻辑。"

rows = []
for path in sorted(ROOT.rglob("*.py")):
    if any(part in SKIP for part in path.parts):
        continue
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception:
        continue
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            rows.append((str(path.relative_to(ROOT)).replace("\\", "/"), node.lineno, node.name, sentence(node)))

groups = [
    ("大脑与编排", lambda p: "/orchestration/" in p or p.endswith("role_coordinator.py") or p.endswith("online_parallel_pipeline.py")),
    ("动态环境与执行", lambda p: p.startswith("envs/") or p.startswith("sdn/") or p.endswith("migration_scheduler.py")),
    ("迁移、资源账本与候选", lambda p: "/marl/" in p and any(x in p for x in ["migration", "deployment_topk", "batch_deployment", "joint_candidate", "online_parallel"])),
    ("强化学习与神经网络", lambda p: "/hrl/" in p or "/gnn/" in p or p.endswith("qmix.py") or p.endswith("trainable_role_agents.py") or p.endswith("migration_baseline_learners.py")),
    ("数据、预测与评估", lambda p: "/marl/" in p or p.startswith("trainer/") or p.startswith("experiments/")),
    ("传统部署与求解器", lambda p: p.startswith("core/expert/") or p.startswith("core/baselines/")),
    ("工具与项目入口", lambda p: p.startswith("sfc_project/") or p.startswith("scripts/")),
]
used=set(); out=["# 当前项目类目录\n", "生成范围：排除 `legacy_hrl_runtime/`、`tests/`、`artifacts/` 和虚拟环境；共列出当前主代码类。\n", "## 阅读方式\n", "大脑负责调度模块；模块组织多个技能；技能调用预测、候选生成、资源校验、执行和验证能力。\n"]
for title, pred in groups:
    selected=[r for r in rows if r[0] not in used and pred(r[0])]
    if not selected: continue
    out += [f"## {title}\n", "| 文件 | 行 | 类 | 作用简介 |\n|---|---:|---|---|\n"]
    for p,l,n,d in selected:
        used.add(p)
        out.append(f"| `{p}` | {l} | `{n}` | {d.replace('|','/')} |\n")

remaining=[r for r in rows if r[0] not in used]
if remaining:
    out += ["## 其他当前代码类\n", "| 文件 | 行 | 类 | 作用简介 |\n|---|---:|---|---|\n"]
    for p,l,n,d in remaining:
        out.append(f"| `{p}` | {l} | `{n}` | {d.replace('|','/')} |\n")
OUT.write_text("".join(out), encoding="utf-8")
print(f"wrote {OUT} ({len(rows)} classes)")
