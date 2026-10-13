import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.calendar_samples import (
    build_sample_plan,
    compute_default_test_start,
    freeze_evaluation_plan,
    is_blind_sealed,
    load_or_freeze_evaluation_plan,
    make_calendar_split,
)


def _retained_records(drop_day: str | None = None):
    dates = pd.date_range("2021-01-01", periods=44, freq="D")
    frame = pd.DataFrame(
        {
            "Vsw": 350 + np.arange(44) * 2.0,
            "shadow": np.linspace(0.1, 0.6, 44),
        }
    )
    if drop_day is not None:
        index = int(np.where(np.asarray(dates) == np.datetime64(drop_day))[0][0])
        frame.loc[index, "shadow"] = np.nan
    plan = build_sample_plan(
        frame,
        frame["Vsw"],
        pd.Series(dates.strftime("%Y%m%d")),
        feature_columns=["Vsw", "shadow"],
        target_name="Vsw",
        past_lag_days=3,
        forecast_horizon_days=3,
    )
    return plan.retained_records()


class FrozenEvaluationPlanTest(unittest.TestCase):
    def test_final_test_without_seal_is_recorded_as_dev_backtest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = load_or_freeze_evaluation_plan(
                root,
                requested_split_label="final_test_blind",
                sealed_final_test=False,
            )
            self.assertEqual(plan.split_label, "dev_backtest")
            self.assertTrue(
                any("没有封存" in note for note in plan.notes)
            )
            reloaded = load_or_freeze_evaluation_plan(root)
            self.assertEqual(reloaded.split_label, "dev_backtest")
            self.assertFalse(is_blind_sealed(root))

    def test_sealed_final_test_is_honored(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = load_or_freeze_evaluation_plan(
                root,
                requested_split_label="final_test_blind",
                sealed_final_test=True,
            )
            self.assertEqual(plan.split_label, "final_test_blind")
            self.assertTrue(is_blind_sealed(root))

    def test_frozen_plan_rejects_conflicting_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            load_or_freeze_evaluation_plan(root, test_ratio=0.2)
            with self.assertRaises(ValueError):
                load_or_freeze_evaluation_plan(root, test_ratio=0.3)

    def test_frozen_test_start_does_not_drift(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            full = _retained_records()
            first_split = make_calendar_split(
                full,
                test_ratio=0.2,
                n_internal_folds=2,
            )
            plan = load_or_freeze_evaluation_plan(root, test_ratio=0.2)
            plan = freeze_evaluation_plan(
                root,
                plan.with_test_start(first_split.test_start),
            )
            self.assertEqual(plan.test_start, first_split.test_start.date().isoformat())

            sparse = _retained_records(drop_day="2021-01-15")
            self.assertNotEqual(
                {record.issue_date for record in full},
                {record.issue_date for record in sparse},
            )
            second_split = make_calendar_split(
                sparse,
                test_ratio=0.2,
                n_internal_folds=2,
                test_start=plan.test_start,
            )
            self.assertEqual(second_split.test_start, first_split.test_start)
            self.assertTrue(
                all(record.issue_date < second_split.test_start for record in second_split.train)
            )
            self.assertTrue(
                all(record.issue_date >= second_split.test_start for record in second_split.test)
            )

    def test_compute_default_matches_first_split(self) -> None:
        records = _retained_records()
        split = make_calendar_split(records, test_ratio=0.2, n_internal_folds=2)
        derived = compute_default_test_start(
            records,
            test_ratio=0.2,
            n_internal_folds=2,
        )
        self.assertEqual(derived, split.test_start)


if __name__ == "__main__":
    unittest.main()
