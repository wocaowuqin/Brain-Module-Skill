# Project baseline

Date: 2026-09-05

## Environment

- OS: Windows
- Python: 3.12.7 (`E:\software\ana\python.exe`)
- PyTorch: 2.6.0+cu124
- Project root: `C:\Users\11353\Desktop\hrl_marl_reconfig_starter`
- Git: repository initialized for the refactor baseline

## Baseline checks

- `python -m compileall -q core envs trainer sdn scripts bms sfc_project`: passed
- `scripts/check_bms_smoke.py`: passed
- `scripts/check_multiagent_orchestration.py --json`: passed
- `scripts/check_hrl_adapter.py`: passed
- `python -m sfc_project doctor --json`: 11/13 presets ready

## Known environment limitation

The doctor reports `torch_geometric` unavailable. This blocks the
`hrl-rate24-resume` and `hrl-smoke` presets. It is an environment dependency
issue and is recorded separately from the directory refactor.

The delivery bundle check also has a known checkpoint compatibility issue:
older WQMIX checkpoints do not contain the current
`BatchCandidateQNetwork.no_migration_head.*` parameters. The checkpoint must
be migrated or retrained before treating that bundle as a release artifact.

## Reference assets

Long-running outputs and large checkpoints remain outside the initial source
commit. They are referenced by path and are not imported as Python modules.
No directory was moved or deleted in Phase 0.
