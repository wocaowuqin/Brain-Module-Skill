"""Prepare, verify and replace the specifically requested portable archive."""
import argparse
import ast
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "legacy_hrl_runtime_dual_topology_no_checkpoint_20260904_v2.zip"
STAGE = ROOT / "artifacts" / "runtime_package_repair_20260905"


def prepare():
    STAGE.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(ARCHIVE) as source:
        for item in source.infolist():
            target = (STAGE / item.filename).resolve()
            if not target.is_relative_to(STAGE.resolve()):
                raise ValueError(f"Unsafe archive entry: {item.filename}")
        source.extractall(STAGE)
    for folder in ("core", "envs", "trainer", "utils", "configs", "topo"):
        for src in (ROOT / folder).rglob("*"):
            if not src.is_file() or "__pycache__" in src.parts:
                continue
            if src.suffix not in {".py", ".yaml", ".yml", ".json", ".mat", ".pkl", ".txt", ".npy", ".npz"}:
                continue
            dst = STAGE / src.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for name in (
        "train_tahrl.py", "requirements.txt", "run_germany_ablation_4rates.bat",
        "scripts/run_germany_ablation_4rates.py",
        "scripts/run_all_variants_arrival_sweep.py",
        "scripts/generate_rl_arrival_sweep.py",
    ):
        dst = STAGE / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dst)
    with (STAGE / "requirements.txt").open("a", encoding="utf-8") as stream:
        stream.write("\n# Also supports the retained legacy runtime.\ngym>=0.26,<1\n")
    for requirement in (STAGE / 'requirements.txt', STAGE / 'legacy_hrl_runtime/requirements.txt'):
        requirement.write_text(requirement.read_text(encoding='utf-8-sig').replace(
            'torch>=2.2,<3', 'torch>=2.2,<2.6'), encoding='utf-8')
    (STAGE / "START_HERE_GERMANY.md").write_text(
        """# Germany50 五个 HRL 变体：启动说明

此包于 2026-09-05 修复。新增根目录当前训练代码及 Germany 批量入口；原 legacy_hrl_runtime 目录保留。

## 安装与运行

完整解压压缩包，进入包含本说明、scripts、train_tahrl.py、data、ilModel 的根目录。
建议 Python 3.10 或 3.11，先激活自己的环境，再执行：

```powershell
python -m pip install -r requirements.txt
python scripts/run_germany_ablation_4rates.py --gpu 0
```

无 NVIDIA CUDA 时使用 `--gpu -1`。所有子进程使用当前 Python，无需原电脑的用户名或 conda 环境路径。
也可以在已激活环境的终端运行 `run_germany_ablation_4rates.bat --gpu -1`。

## 范围

默认跑 Germany50 的 rate=1、1.5、2、2.5、3、3.5，共 30 个任务。

| 名称 | 实际变体 |
|---|---|
| msft_hirl | full |
| msft_hrl | no_il |
| msft_ilrl | single_dqn |
| msft_hirl_gat | gat |
| msft_hirl_mlp | mlp |

使用 Germany50 的 35 个 DC，CPU=55、MEM=45、BW=90。
结果：`artifacts/runs/germany_ablation_4rates/`，总表为其中的 `summary.csv`。
默认跳过已完成任务；用 `--force` 重跑。失败任务仍会继续其他任务，最终返回非零退出码。
Phase3 禁用周期 checkpoint 和 final_model.pth，保留 CSV、JSON、日志与图表。
包内 ilModel/50/il_model_best.pth 是输入 IL 权重，请保留。

## 短测试

```powershell
python scripts/run_germany_ablation_4rates.py --dry-run
python scripts/run_germany_ablation_4rates.py --rates 1.0 --limit-requests 2 --gpu -1
```

短测试默认写入独立的 `artifacts/runs/germany_ablation_4rates_smoke/`，不会被全量实验当作已完成结果。
`run_50node_phase1_phase2.py` 仍在 `legacy_hrl_runtime/scripts/`；它是专家数据/IL 预训练入口，不是本次消融入口。

## 权重与实验解释

此包沿用已有算法实现。MSFT-ILRL 当前实际映射 single_dqn。
GAT/MLP 若与现有 TreeTransformer IL 权重不匹配，当前训练器会跳过该权重并从随机初始化训练；
因此在获得对应编码器的 IL 预训练权重前，不能将其解释为只更换编码器、其他条件相同的严格消融。
小样本运行仅验证可启动、数据接线和结果输出，不证明收敛或最终性能。

## 原有 US 入口

`python legacy_hrl_runtime/run_us_legacy_dc20_sweep.py --gpu 0`
该目录保留旧版 HRL 和旧 US DC 配置；Germany 上述入口使用根目录当前版本。
""", encoding="utf-8")
    print(STAGE)


def pack():
    files = [p for p in STAGE.rglob("*") if p.is_file()
             and "__pycache__" not in p.parts
             and "artifacts" not in p.relative_to(STAGE).parts
             and "outputs" not in p.relative_to(STAGE).parts
             and "output" not in p.relative_to(STAGE).parts
             and p.name not in {"PACKAGE_SHA256.json", "verification_smoke_console.log"}]
    for path in files:
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    manifest = {str(p.relative_to(STAGE)).replace("\\", "/"):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest_path = STAGE / "PACKAGE_SHA256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    candidate = ARCHIVE.with_suffix(".replacement.zip")
    with zipfile.ZipFile(candidate, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for src in [*files, manifest_path]:
            output.write(src, src.relative_to(STAGE).as_posix())
    with zipfile.ZipFile(candidate) as check:
        assert check.testzip() is None
        names = check.namelist()
        assert len(names) == len(set(names))
        for name, digest in manifest.items():
            assert hashlib.sha256(check.read(name)).hexdigest() == digest, name
    candidate.replace(ARCHIVE)
    print(json.dumps({"archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
                      "files": len(files) + 1, "sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "pack"])
    args = parser.parse_args()
    {"prepare": prepare, "pack": pack}[args.action]()
