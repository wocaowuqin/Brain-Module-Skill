"""Create calendar-based train/test splits and a baseline manifest."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

def day(row):
    value = row.get("date", row.get("timestamp", row.get("arrival_time", 0)))
    if isinstance(value, str):
        return value[:10]
    return datetime.fromtimestamp(float(value), timezone.utc).date().isoformat()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--requests", required=True); p.add_argument("--output", required=True); p.add_argument("--train-days", type=int, default=7)
    a=p.parse_args(); rows=[json.loads(x) for x in Path(a.requests).read_text(encoding="utf-8").splitlines() if x.strip()]
    groups=defaultdict(list)
    for row in rows: groups[day(row)].append(row)
    days=sorted(groups); train=set(days[:a.train_days]); test=set(days[a.train_days:])
    out=Path(a.output); out.mkdir(parents=True, exist_ok=True)
    for name, selected in (("train", train),("test", test)):
        (out/f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for d in sorted(selected) for r in groups[d])+"\n", encoding="utf-8")
    manifest={"split_strategy":"calendar_day","train_days":sorted(train),"test_days":sorted(test),"baselines":["no_migration","reactive_threshold","oracle_future","lifecycle_aware_wqmix"],"metrics":["acceptance_rate","invalid_migration_rate_30s","migration_interruption_time_ms","cluster_load_std_reduction"],"wqmix_safety_guard":False}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
if __name__ == "__main__": main()
