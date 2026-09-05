# Migration WQMIX Pipeline

The migration policy is separate from the deployment policy. The unified
trainer writes `checkpoint_type=migration_policy_v2`; the runtime also accepts
the retained legacy type `migration_wqmix_v1`. It rejects deployment
checkpoints passed as migration weights.

## Semantics

- One selected VNF migration task is one logical agent.
- At most eight agents share one parameterized Q network in a decision batch.
- Each action is one complete target-DC plan; the final action is no-migration.
- Action masks enforce CPU, memory, directed bandwidth, lifetime, and delay.
- The bounded joint decoder resolves cross-task resource conflicts.
- The runtime adds a make-before-break flow-overlap gate before reservation.
- At most four non-conflicting migration executions run concurrently.

## Build Data

```powershell
python scripts/build_migration_dataset.py `
  --runtime-report artifacts/runs/mininet/runtime_result.json `
  --profile sdn/topologies/us_backbone_28_bw90.json `
  --output data/migration_train `
  --samples-per-report 256 `
  --max-agents 8 `
  --top-k 8 `
  --seed 7101
```

Use different request seeds, not only different hotspot-injection seeds, for
the final train, validation, and test split.

## Train

```powershell
python scripts/train_migration_baselines.py `
  --train-data data/migration_train `
  --validation-data data/migration_validation `
  --algorithm wqmix `
  --output artifacts/runs/migration/wqmix `
  --bc-epochs 20 `
  --epochs 40
```

With `--algorithm wqmix`, the script writes `migration_bc.pt`,
`migration_wqmix.pt`, and `training_report.json`. The BC checkpoint is the
default warm start; pass `--no-bc-warm-start` if it is intentionally disabled.

## Evaluate

```powershell
python scripts/eval_migration_baselines.py `
  --data data/migration_test `
  --checkpoint wqmix=artifacts/runs/migration/wqmix/migration_wqmix.pt `
  --output artifacts/runs/migration/wqmix/eval_test
```

## Online Runtime

Add these options to the existing SDN replay command:

```text
--online-wqmix-auto-migration
--online-migration-wqmix-checkpoint artifacts/runs/migration/wqmix/migration_wqmix.pt
--online-migration-max-agents 8
--online-migration-max-inflight 4
```

For periodic prediction-driven triggers, also add:

```text
--predictive-migration-scan-seconds 0.2
--predictive-migration-overload-threshold 0.85
--predictive-migration-safe-utilization 0.75
--predictive-migration-horizon-seconds 1.0
--predictive-migration-cooldown-seconds 5.0
```

The predictor consumes the online deployment ledger. It applies EWMA trend
prediction, hysteresis, remaining-lifetime filtering, cooldown, and minimum
relief selection before tasks enter WQMIX.

## Validation

```powershell
python scripts/check_migration_wqmix_pipeline.py `
  --data data/migration_test `
  --checkpoint artifacts/runs/migration/wqmix/migration_wqmix.pt `
  --runtime-report artifacts/runs/mininet/runtime_result.json `
  --profile sdn/topologies/us_backbone_28_bw90.json
```

The repository also retains one bounded implementation fixture. It can be
checked without generating a new Mininet report:

```powershell
python scripts/check_migration_wqmix_pipeline.py `
  --data data/migration_wqmix_smoke_seed7071 `
  --checkpoint artifacts/reference_runs/migration/migration_baselines_smoke/migration_wqmix.pt `
  --runtime-report artifacts/reference_runs/migration/runtime_rate8_seed7071_first100.json `
  --profile sdn/topologies/us_backbone_28_bw90.json
```

This fixture combines a historical rate-8 runtime trace with synthetic
hotspot snapshots. It validates loading, masks, bounded batching, online
selection, and predictive monitoring only; it is not a final migration result.

Run the native state-continuity test inside WSL because it requires GCC and
Unix sockets:

```powershell
wsl.exe -d Ubuntu-22.04 -- bash -lc `
  "cd /mnt/c/Users/11353/Desktop/hrl_marl_reconfig_starter && python3 scripts/check_vnf_migration_protocol.py"
```
