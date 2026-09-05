# Baseline Algorithm Suite

This project evaluates initial SFC deployment and post-deployment VNF
migration as two separate decisions.  Algorithms must not be compared across
different candidate sets, masks, resource snapshots, decoders, or ledgers.

## Migration baselines

The migration suite contains:

- `no_migration`: explicit noop lower bound.
- `random_feasible`: random masked candidate order.
- `reactive_greedy`: migrate only after the current utilization threshold.
- `predictive_greedy`: migrate when current or predicted utilization triggers.
- `delay_aware_greedy`: predicted trigger with delay and target-load scoring.
- `milp_oracle`: exact small-batch SciPy/HiGHS upper bound.
- `bc`: behavior cloning from joint-oracle actions.
- `idqn`: one-step fitted independent Q baseline.
- `qmix`: standard unweighted QMIX loss (`alpha=1`).
- `wqmix`: optimistic Weighted QMIX (`alpha<1`).
- `mappo`: shared actor and centralized critic with decoded one-step returns.

One active migration task is one temporary agent.  Its actions are Top-K
complete target plans plus noop.  A physical node is not an agent.

Train every learned migration baseline:

```powershell
C:\Users\11353\.conda\envs\sfc_ppo\python.exe `
  scripts\train_migration_baselines.py `
  --train-data data\migration_wqmix_v1_seed7071_train `
  --validation-data data\migration_wqmix_v1_seed7071_val `
  --algorithm all `
  --allow-source-overlap `
  --output artifacts\runs\migration\baselines
```

Evaluate all heuristics, the MILP oracle, and learned checkpoints:

```powershell
C:\Users\11353\.conda\envs\sfc_ppo\python.exe `
  scripts\eval_migration_baselines.py `
  --data data\migration_wqmix_v1_seed7071_test `
  --checkpoint bc=artifacts\runs\migration\baselines\migration_bc.pt `
  --checkpoint idqn=artifacts\runs\migration\baselines\migration_idqn.pt `
  --checkpoint qmix=artifacts\runs\migration\baselines\migration_qmix.pt `
  --checkpoint wqmix=artifacts\runs\migration\baselines\migration_wqmix.pt `
  --checkpoint mappo=artifacts\runs\migration\baselines\migration_mappo.pt `
  --allow-seen-source `
  --output artifacts\runs\migration\baseline_eval
```

The two allow flags above are required by the currently bundled smoke data
because its nominal splits reuse one seed-7071 runtime report.  Results from
that command are deliberately marked non-reportable.  Omit both flags when
using genuinely independent runtime traces.

Run the implementation smoke check:

```powershell
C:\Users\11353\.conda\envs\sfc_ppo\python.exe `
  scripts\check_migration_baselines.py
```

## Initial-deployment baselines

The unified initial-deployment evaluator contains reject-all, the serialized
original policy, independent Top-1, random feasible, objective greedy,
legacy-HRL-only, conflict-aware bounded joint greedy, precomputed MILP oracle labels, and any
compatible BC/QMIX/WQMIX actor checkpoint.

Train standard QMIX and Weighted QMIX with the same action-coupled deployment
environment by changing only `--algorithm`:

```powershell
C:\Users\11353\.conda\envs\sfc_ppo\python.exe `
  scripts\train_deployment_wqmix.py `
  --algorithm qmix `
  --bc-checkpoint artifacts\reference_runs\deployment\deployment_bc_v4_tree_bw_rate24_sla80\bc_pretrained.pt `
  --train-trace data\sdn_runtime_requests\seed_7071_rate24_duration100_lifetime50node\requests.jsonl `
  --output artifacts\runs\deployment\qmix
```

Use `--algorithm wqmix --alpha 0.1` for the weighted variant.  The standard
QMIX mode forces `alpha=1.0`, regardless of the `--alpha` argument.

```powershell
C:\Users\11353\.conda\envs\sfc_ppo\python.exe `
  scripts\eval_deployment_baselines.py `
  --data data\deployment_v4_tree_bw_rate24_sla80\test\seed_7301\mb5ms `
  --checkpoint bc=artifacts\reference_runs\deployment\deployment_bc_v4_tree_bw_rate24_sla80\bc_pretrained.pt `
  --checkpoint wqmix=artifacts\reference_runs\deployment\deployment_wqmix_v4_tree_bw_rate24_sla80_ep5\wqmix_final.pt `
  --output artifacts\runs\deployment\baseline_eval
```

The `legacy_hrl_only` result is meaningful only when a candidate has the exact
serialized source `legacy_hrl`.  Beam or heuristic candidates are never
renamed as HRL results.

## Scientific boundary

The current migration files contain real runtime-derived plans mixed with
synthetic overload snapshots.  They are suitable for implementation smoke
tests, not final paper claims.  Moreover, their stored next snapshots do not
change when a policy selects a different current action.  Therefore IDQN and
MAPPO are explicitly one-step/contextual baselines, while the QMIX variants use
the existing offline consecutive-trace TD surrogate.  A final temporal MARL
claim requires an action-coupled migration simulator and seed-separated train,
validation, and test traces.

`modeled_sla_safe_rate` is based on candidate delay estimates.  It must not be
reported as measured Mininet/Ryu strict SLA.  Real strict SLA still requires
the runtime probe pipeline.
