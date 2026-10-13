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


if __name__ == "__main__":
    unittest.main()
