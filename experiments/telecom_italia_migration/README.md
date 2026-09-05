# Telecom Italia Dynamic SFC Migration Experiment

This experiment is isolated from the existing rate8 and Mininet/Ryu runs.

## Data provenance

- Dataset: Telecom Italia, *Telecommunications - SMS, Call, Internet - MI*.
- DOI: `10.7910/DVN/EGZHFV`.
- License: ODbL 1.0; follow the attribution terms on the dataset page.
- Raw files require the Harvard Dataverse guestbook to be completed in a
  browser. Do not rename downloaded files because MD5 verification is keyed by
  the official filenames.

Put official files in `data/telecom_italia/raw/`, then run:

```powershell
C:\Users\11353\.conda\envs\DL\python.exe scripts\prepare_telecom_italia.py
C:\Users\11353\.conda\envs\DL\python.exe scripts\run_telecom_italia_migration.py
```

For a quick check:

```powershell
C:\Users\11353\.conda\envs\DL\python.exe scripts\run_telecom_italia_migration.py --sfc-counts 10 --max-slots 24
```

`experiment_spec.json` separates paper-provided parameters from assumptions.
The first implementation is a controlled reproduction scaffold. It does not
claim to reproduce the thesis MNTP-TTM predictor or its TRPO checkpoint.

Traffic handling is causal. Each grid profile is normalized at time `t` from
the strictly preceding sliding history window; later trace values cannot alter
earlier normalized values. Forecasts use only that same pre-slot history and
are bounded by historical peaks. A finite trace never wraps with modulo
indexing: once a profile is exhausted, its flow is zero. The
`brain_rule_marl` policy keeps its public name for compatibility, but its peak
risk signal is historical and causal rather than a future oracle.
All generated SFCs use the global trace slot (`profile_offset=0`); random phase
offsets are not used because they would expose future global slots at creation.
