# HRL training compatibility path

The HRL implementation now has one source of truth at the project root:

```powershell
python train_tahrl.py --help
```

`hrl_training/train_tahrl.py` is retained only so older commands continue to
work. It delegates directly to the root trainer. Use
`python train_tahrl.py --disable-checkpoint` instead of the removed duplicate
`train_tahrl_no_checkpoint.py` entry point.

Canonical modules are in `core/`, `envs/`, `trainer/`, `configs/`, and `topo/`.
Training outputs belong under `artifacts/runs/hrl/`.
