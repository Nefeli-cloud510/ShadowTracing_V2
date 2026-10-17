import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.calendar_samples import build_sample_plan
from models.elasticnet_runner import (
    ElasticNetConfig,
    ExperimentConfig,
    FileConfig,
    TimeWindowConfig,
    build_sliding_window,
    load_and_merge_data,
)


def _run_config(omni_path: Path, lhaaso_path: Path, features: list[str], lag: int = 3, horizon: int = 3):
    return ExperimentConfig(
        file_config=FileConfig(
            omni_file_path=omni_path,
            lhaaso_file_path=lhaaso_path,
            feature_columns=features,
            time_column="TIME",
            target_column="Vsw",
        ),
        time_window_config=TimeWindowConfig(
            window_size=lag,
            test_split_ratio=0.2,
            forecast_horizon_days=horizon,
            past_lag_days=lag,
        ),
        elasticnet_config=ElasticNetConfig(model_family="ridge"),
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.write_text(frame.to_csv(index=False), encoding="utf-8")


def _record_state(plan):
    return {
        (record.issue_date, record.target_date, record.sample_condition, record.drop_reason)
        for record in plan.records
    }


class CalendarSamplePlanTest(unittest.TestCase):
    def test_reverse_input_order_is_identical(self) -> None:
        dates = pd.date_range("2021-01-01", periods=40, freq="D")
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(
            {
                "Vsw": 350 + rng.normal(0, 1, len(dates)),
                "shadow": rng.normal(0, 1, len(dates)),
            }
        )
        frame.loc[12, "shadow"] = np.nan
        time_raw = pd.Series(dates.strftime("%Y%m%d"))

        forward = build_sample_plan(
            frame,
            frame["Vsw"],
            time_raw,
            feature_columns=["Vsw", "shadow"],
            target_name="Vsw",
            past_lag_days=3,
            forecast_horizon_days=3,
        )
        reverse = build_sample_plan(
            frame.iloc[::-1].reset_index(drop=True),
            frame["Vsw"].iloc[::-1].reset_index(drop=True),
            pd.Series(dates.strftime("%Y%m%d")[::-1]),
            feature_columns=["Vsw", "shadow"],
            target_name="Vsw",
            past_lag_days=3,
            forecast_horizon_days=3,
        )

        self.assertEqual(forward.retained_count, reverse.retained_count)
        self.assertEqual(_record_state(forward), _record_state(reverse))

    def test_future_day_missing_feature_keeps_target_sample(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dates = pd.date_range("2021-01-01", periods=12, freq="D")
            omni = pd.DataFrame(
                {
                    "TIME": dates.strftime("%Y%m%d"),
                    "Vsw": [350 + i for i in range(12)],
                }
            )
            lhaaso = pd.DataFrame(
                {
                    "TIME": dates.strftime("%Y%m%d"),
                    "shadow": [1.0] * 12,
                }
            )
            lhaaso.loc[4, "shadow"] = np.nan
            omni_path = root / "omni.csv"
            lhaaso_path = root / "lhaaso.csv"
            _write_csv(omni_path, omni)
            _write_csv(lhaaso_path, lhaaso)

            merged = load_and_merge_data(
                _run_config(omni_path, lhaaso_path, ["Vsw", "shadow"], lag=2, horizon=2)
            )
            self.assertEqual(len(merged.X), 12)
            plan = build_sample_plan(
                merged.X,
                merged.y,
                merged.time,
                feature_columns=["Vsw", "shadow"],
                target_name="Vsw",
                past_lag_days=2,
                forecast_horizon_days=2,
            )
            target = pd.Timestamp("2021-01-05")
            retained = [
                record
                for record in plan.records
                if record.target_date == target and record.drop_reason is None
            ]
            self.assertTrue(retained, "target-day feature NaN count does not need to drop the sample")

    def test_outer_merge_keeps_missing_source_day_as_nan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dates = pd.date_range("2021-01-01", periods=12, freq="D")
            omni = pd.DataFrame(
                {
                    "TIME": dates.strftime("%Y%m%d"),
                    "Vsw": [350 + i for i in range(12)],
                }
            )
            lhaaso = pd.DataFrame(
                {
                    "TIME": dates.strftime("%Y%m%d"),
                    "shadow": [1.0] * 12,
                }
            )
            lhaaso = lhaaso.drop(index=lhaaso.index[2])
            omni_path = root / "omni.csv"
            lhaaso_path = root / "lhaaso.csv"
            _write_csv(omni_path, omni)
            _write_csv(lhaaso_path, lhaaso)

            merged = load_and_merge_data(
                _run_config(omni_path, lhaaso_path, ["Vsw", "shadow"], lag=2, horizon=2)
            )
            self.assertEqual(len(merged.X), 12)
            plan = build_sample_plan(
                merged.X,
                merged.y,
                merged.time,
                feature_columns=["Vsw", "shadow"],
                target_name="Vsw",
                past_lag_days=2,
                forecast_horizon_days=2,
            )
            dropped = [
                record
                for record in plan.records
                if record.issue_date == pd.Timestamp("2021-01-03")
                and record.drop_reason == "feature_missing"
            ]
            self.assertTrue(dropped, "source-missing day should survive merge and be judged at sample layer")

    def test_sparse_feature_lags_are_authoritative(self) -> None:
        dates = pd.date_range("2021-01-01", periods=16, freq="D")
        frame = pd.DataFrame(
            {
                "A": np.arange(16, dtype=float),
                "B": np.arange(16, dtype=float) * 2,
            }
        )
        plan = build_sample_plan(
            frame,
            frame["A"],
            pd.Series(dates.strftime("%Y%m%d")),
            feature_columns=["A", "B"],
            target_name="A",
            past_lag_days=3,
            forecast_horizon_days=1,
            feature_lags={"A": [3, 1], "B": [9]},
        )
        self.assertEqual(plan.feature_lags, {"A": [1, 3], "B": [9]})
        self.assertEqual(
            [spec.feature_name() for spec in plan.feature_specs],
            ["A_t-1", "A_t-3", "B_t-9"],
        )
        self.assertEqual(plan.past_lag_days, 3)

    def test_sparse_feature_lag_missing_follows_feature_date(self) -> None:
        dates = pd.date_range("2021-01-01", periods=15, freq="D")
        frame = pd.DataFrame(
            {
                "A": np.arange(15, dtype=float),
                "B": np.arange(15, dtype=float),
            }
        )
        frame.loc[3, "B"] = np.nan
        plan = build_sample_plan(
            frame,
            frame["A"],
            pd.Series(dates.strftime("%Y%m%d")),
            feature_columns=["A", "B"],
            target_name="A",
            past_lag_days=1,
            forecast_horizon_days=1,
            feature_lags={"A": [1], "B": [9]},
        )
        dropped = [
            record
            for record in plan.records
            if record.issue_date == pd.Timestamp("2021-01-13")
            and record.drop_reason == "feature_missing"
        ]
        self.assertTrue(dropped, "B 在对应特征日期缺失时应拒绝该样本")
        retained = [
            record
            for record in plan.records
            if record.issue_date == pd.Timestamp("2021-01-12")
            and record.drop_reason is None
        ]
        self.assertTrue(retained, "特征日期完整时应保留该样本")
        self.assertEqual(plan.field_drop_counts.get("B"), 1)

    def test_sparse_feature_lags_filter_keys_not_in_run_columns(self) -> None:
        dates = pd.date_range("2021-01-01", periods=12, freq="D")
        frame = pd.DataFrame({"A": np.arange(12, dtype=float)})
        plan = build_sample_plan(
            frame,
            frame["A"],
            pd.Series(dates.strftime("%Y%m%d")),
            feature_columns=["A"],
            target_name="A",
            past_lag_days=1,
            forecast_horizon_days=1,
            feature_lags={"A": [1], "B": [9]},
        )
        self.assertEqual(plan.feature_lags, {"A": [1]})
        self.assertEqual(
            [spec.feature_name() for spec in plan.feature_specs],
            ["A_t-1"],
        )

    def test_without_sparse_lags_keeps_uniform_window(self) -> None:
        dates = pd.date_range("2021-01-01", periods=12, freq="D")
        frame = pd.DataFrame({"A": np.arange(12, dtype=float)})
        plan = build_sample_plan(
            frame,
            frame["A"],
            pd.Series(dates.strftime("%Y%m%d")),
            feature_columns=["A"],
            target_name="A",
            past_lag_days=3,
            forecast_horizon_days=3,
        )
        self.assertEqual(plan.feature_lags, {"A": [0, 1, 2]})
        self.assertEqual(
            [spec.feature_name() for spec in plan.feature_specs],
            ["A_t", "A_t-1", "A_t-2"],
        )

    def test_sparse_feature_lags_require_every_run_column(self) -> None:
        dates = pd.date_range("2021-01-01", periods=12, freq="D")
        frame = pd.DataFrame(
            {
                "A": np.arange(12, dtype=float),
                "B": np.arange(12, dtype=float),
            }
        )
        with self.assertRaisesRegex(ValueError, "缺少"):
            build_sample_plan(
                frame,
                frame["A"],
                pd.Series(dates.strftime("%Y%m%d")),
                feature_columns=["A", "B"],
                target_name="A",
                past_lag_days=1,
                forecast_horizon_days=1,
                feature_lags={"A": [1]},
            )

    def test_build_sliding_window_rejects_sparse_lags_without_intersection(self) -> None:
        dates = pd.date_range("2021-01-01", periods=12, freq="D")
        frame = pd.DataFrame({"A": np.arange(12, dtype=float)})
        tw_cfg = TimeWindowConfig(
            window_size=2,
            test_split_ratio=0.2,
            forecast_horizon_days=1,
            past_lag_days=3,
            feature_lags={"A": [1, 3]},
        )
        with self.assertRaisesRegex(ValueError, "没有交集"):
            build_sliding_window(
                frame,
                frame["A"],
                pd.Series(dates.strftime("%Y%m%d")),
                tw_cfg,
                feature_columns=["B"],
            )


if __name__ == "__main__":
    unittest.main()
