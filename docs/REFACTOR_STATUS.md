# Refactor status

## Completed

- Git baseline and rollback point established.
- Top-level `bms/` public package added while preserving `core/bms` imports.
- Legacy runtime physically isolated under `legacy/legacy_hrl_runtime/`.
- Script namespace packages created for `train`, `evaluate`, `data`, `checks`
  and `runtime`; root scripts remain compatibility entry points.
- Domain contracts added under `core/domain/` without dependencies on scripts,
  trainers, SDN clients or experiment output directories.

## Compatibility policy

No legacy implementation or large data/output directory is deleted during the
refactor. Every migration step is committed separately so a single Git revert
restores the previous layout.

## Known remaining work

The largest runtime script still needs extraction into tested service modules;
old `core` imports of script helpers need replacement; and the legacy tree can
only be deleted after two releases without external imports. These are tracked
as the next refactor commits rather than being represented as completed.
