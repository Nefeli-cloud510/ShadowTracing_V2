import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.calendar_samples import CalendarSample, build_sample_plan, make_calendar_split
from models.elasticnet_runner import (
    ElasticNetConfig,
    ExperimentConfig,
    FileConfig,
    TimeWindowConfig,
    build_sliding_window,
    evaluate_model,
    load_and_merge_data,
    split_dataset_by_calendar,
    train_elasticnet_model,
)

EXCEL = ROOT / "data" / "2021_public_data.xlsx"
TARGET = "SW Plasma Speed, km/s"
SHADOW = "p2_y中心值"


def _make_config(feature_columns: list[str]) -> ExperimentConfig:
    return ExperimentConfig(
        file_config=FileConfig(
            omni_file_path=EXCEL,
            lhaaso_file_path=EXCEL,
            feature_columns=feature_columns,
            time_column="dataset_id",
            target_column=TARGET,
        ),
        time_window_config=TimeWindowConfig(
            window_size=3,
            test_split_ratio=0.2,
            forecast_horizon_days=3,
            past_lag_days=3,
        ),
        elasticnet_config=ElasticNetConfig(model_family="ridge"),
    )


def _run(feature_columns: list[str], label: str) -> None:
    cfg = _make_config(feature_columns)
    merged = load_and_merge_data(cfg)
    import core.calendar_samples as cs

    plan = build_sample_plan(
        merged.X,
        merged.y,
        merged.time,
        feature_columns=feature_columns,
        target_name=TARGET,
        past_lag_days=3,
        forecast_horizon_days=3,
    )
    print(
        f"[{label}] plan retained={plan.retained_count} "
        f"candidate={plan.candidate_count} dropped={plan.dropped_count}"
    )
    split_plan = make_calendar_split(plan.retained_records())
    print(
        f"[{label}] split train={len(split_plan.train)} test={len(split_plan.test)} "
        f"boundary={len(split_plan.boundary_excluded)} folds={len(split_plan.cv_folds)}"
    )
    window = build_sliding_window(
        merged.X,
        merged.y,
        merged.time,
        cfg.time_window_config,
        feature_columns=feature_columns,
    )
    split = split_dataset_by_calendar(
        window.X_all,
        window.y_all,
        window.time_all,
        window.origins_all,
        window.sample_ids_all,
        window.records_all or [],
        cfg.time_window_config.test_split_ratio,
        window.feature_names,
    )
    model = train_elasticnet_model(
        split.X_train,
        split.y_train,
        cfg.elasticnet_config,
        cv_folds=split.cv_folds,
    )
    metrics, _, _ = evaluate_model(
        model,
        split.X_train,
        split.X_test,
        split.y_train,
        split.y_test,
    )
    print(
        f"[{label}] metrics test_rmse={metrics['test_rmse']:.3f} "
        f"test_r={metrics['test_pearson_r']:.3f} "
        f"n_train={metrics['n_train']} n_test={metrics['n_test']}"
    )


if __name__ == "__main__":
    _run([TARGET], "AR-only")
    _run([TARGET, SHADOW], "AR+shadow")
