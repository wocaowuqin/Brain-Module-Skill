# RL arrival-rate sweep

This experiment compares RL-only initial SFC mapping baselines on identical
request traces. The two topologies are `us_backbone` and `germany50`; per-source
arrival rates are `0.5, 1, 1.5, 2, 2.5, 3 req/s`.

The historical rate labels `4, 8, 12, 16, 20, 24` are retained only for plot
compatibility. The actual aggregate rate is always:

```text
aggregate_rate = per_source_rate * number_of_source_nodes
```

US Backbone has 8 historical source nodes and the 50-node corpus has 15.

Generate or collect all six data points:

```powershell
$PY = 'C:\Users\11353\.conda\envs\sfc_ppo\python.exe'
& $PY scripts\generate_rl_arrival_sweep.py
```

Run the missing A2C/PPO points:

```powershell
& $PY scripts\run_rl_arrival_sweep.py `
  --algorithms a2c ppo `
  --topologies us_backbone germany50 `
  --rates 0.5 1.5 2.5
```

Run the ordinary macro-action DDQN baseline at per-source rate 1:

```powershell
& $PY scripts\run_rl_arrival_sweep.py `
  --algorithms dqn `
  --topologies us_backbone `
  --rates 1
```

Use `--limit-requests 100` for a smoke test. All results are written under
`artifacts/runs/rl_arrival_sweep_seed2026/`; `sweep_summary.csv` is the unified
machine-readable result table.

`dqn` deliberately invokes the standalone historical macro-action DDQN. It is
an external RL baseline, distinct from the `single_dqn` internal HRL ablation.
`germany50` is the paper-facing name of the topology stored as `50node.mat` and
passed to the legacy baseline programs as `--topo 50node`.
