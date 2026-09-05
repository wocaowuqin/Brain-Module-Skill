from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.telecom_italia_migration.simulator import (
    ExperimentConfig,
    SFC,
    _causal_scale,
    _forecast,
    run_sweep,
)
from experiments.telecom_italia_migration.telecom_data import aggregate_files


class TelecomItaliaMigrationTest(unittest.TestCase):
    def test_causal_scaling_and_forecast_are_prefix_invariant(self) -> None:
        prefix = [1.0, 2.0, 3.0, 2.0, 4.0]
        changed_suffix = prefix + [10_000.0, 20_000.0]
        scaled_a = _causal_scale(prefix, 2.0, 18.0)
        scaled_b = _causal_scale(changed_suffix, 2.0, 18.0)
        np.testing.assert_allclose(scaled_a, scaled_b[: len(prefix)])
        self.assertEqual(_forecast(np.asarray(prefix), 4, 3), _forecast(np.asarray(changed_suffix), 4, 3))
        self.assertEqual(_forecast(np.asarray(prefix), len(prefix), 3), 0.0)

    def test_flow_does_not_wrap_to_trace_tail(self) -> None:
        sfc = SFC(
            sfc_id=1,
            source=0,
            destination=1,
            profile=np.asarray([2.0, 3.0]),
            profile_offset=0,
            lifetime=10,
            age=0,
            delay_bound_ms=50.0,
        )
        self.assertEqual(sfc.flow(0), 2.0)
        self.assertEqual(sfc.flow(1), 3.0)
        self.assertEqual(sfc.flow(2), 0.0)

    def test_aggregate_and_simulate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "sms-call-internet-mi-2013-11-01.txt"
            raw.write_bytes(
                b"5050\t1383260400000\t39\t1\t2\t3\t4\t5\n"
                b"5050\t1383260400000\t44\t\t\t\t\t2.5\n"
                b"5051\t1383260400000\t39\t1\t2\t3\t4\t3\n"
                b"5050\t1383261000000\t39\t1\t2\t3\t4\t8\n"
                b"5051\t1383261000000\t39\t1\t2\t3\t4\t4\n"
            )
            activity = root / "activity.csv"
            report = aggregate_files([raw], activity, grid_ids=[5050, 5051])
            self.assertEqual(report["timestamps"], 2)
            with activity.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            first = next(
                row
                for row in rows
                if row["timestamp_ms"] == "1383260400000" and row["grid_id"] == "5050"
            )
            self.assertEqual(float(first["internet_activity"]), 7.5)

            summaries = run_sweep(
                activity,
                root / "result",
                config=ExperimentConfig(seed=7),
                sfc_counts=[2],
                policies=["no_migration", "reactive_mih"],
                max_slots=2,
            )
            self.assertEqual(len(summaries), 2)
            self.assertTrue((root / "result" / "summary.csv").exists())
            self.assertTrue((root / "result" / "experiment_spec.json").exists())


if __name__ == "__main__":
    unittest.main()
