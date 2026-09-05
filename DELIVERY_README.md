# HRL-MARL SFT Minimal Delivery

This package is the minimal runnable handoff of the current project. It keeps
the active implementation and bounded fixtures, while excluding virtual
environments, IDE files, experiment logs, generated plots, paper PDFs, and
large historical outputs.

## Included workflows

- Hierarchical RL for multicast SFT placement and routing.
- Request-level Top-K candidates, WQMIX ranking, joint feasibility decoding,
  and atomic CPU/MEM/directed-bandwidth accounting.
- Central-brain, role-agent, and skill orchestration.
- Predictive VNF migration planning and make-before-break transaction checks.
- Germany50 `rate=0.5` five-variant HRL ablation entry point.
- A bounded rate8/seed7071 SFT fixture for reproducible regression tests.

## Setup

Python 3.9 or 3.10 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the complete package self-check:

```powershell
.\.venv\Scripts\python.exe scripts\check_delivery_bundle.py
```

The check covers multi-agent orchestration, migration transaction semantics,
the original rate8 SFT candidates, and the migration WQMIX loading/selection
pipeline. A successful run ends with `DELIVERY BUNDLE: PASS`.

## Checkpoint policy

Arrival-rate HRL sweep scripts do not write periodic or final `.pth` training
checkpoints by default. They keep the CSV/JSON metrics, logs, and plots needed
for evaluation. Add `--keep-checkpoints` to
`scripts/run_all_variants_arrival_sweep.py` only when resumable checkpoints
are needed.

## Germany50 rate=0.5 ablation

The no-argument entry point runs these variants in order:

1. `msft_hirl`
2. `msft_hrl`
3. `msft_ilrl`
4. `msft_hirl_gat`
5. `msft_hirl_mlp`

```powershell
.\.venv\Scripts\python.exe scripts\run_germany_ablation_4rates.py
```

Despite the historical filename, the retained entry currently runs only
Germany50 `rate=0.5`. Results are written to
`artifacts/runs/germany_ablation_4rates/` and completed variants are resumed
from their existing CSV outputs.

## Useful direct checks

```powershell
python scripts\check_multiagent_orchestration.py --json
python scripts\check_atomic_ledger_replacement.py
python scripts\check_migration_runtime_transaction.py
python scripts\check_rate8_sft_migration_regression.py
```

## Ryu/Mininet boundary

The Python orchestration and SDN source code are included, but a real Mininet
run additionally requires Linux/WSL with Ryu, Open vSwitch, Mininet, GCC, and
appropriate privileges. Those system packages are not embedded in this zip.
Install `requirements-sdn.txt` inside that environment after the base Python
requirements.

## Evidence boundary

The included migration WQMIX checkpoint is a small interface fixture. It
proves that loading, masking, candidate materialization, joint feasibility,
and monitoring work; it is not a final paper-performance checkpoint. The
included rate8 plans are bounded to the first 100 requests for regression.
