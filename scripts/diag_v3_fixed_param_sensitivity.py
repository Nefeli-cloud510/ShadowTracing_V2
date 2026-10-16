"""Stand-alone fixed-parameter sensitivity diagnostics for the v3 run.

This script deliberately does not run the closed-loop control flow. It reuses
the frozen dev_backtest partition from E_R02_03_r1 and fits fixed-parameter
Ridge / ElasticNet arms so we can separate "model family" from "auto search
hits the grid boundary" before recommending any next experiment.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.calendar_samples import make_calendar_split
from core.json_safe import json_dumps
from models.elasticnet_runner import (
    ElasticNetConfig,
    ExperimentConfig,
    FileConfig,
    TimeWindowConfig,
    build_sliding_window,
    load_and_merge_data,
    make_model,
    select_best_params,
    split_dataset_by_calendar,
)


SESSION_ROOT = (
    ROOT
    / "runtime"
    / "live_session"
    / "sessions"
    / "session_20261015_132955_705440"
)
RAW_DIR = SESSION_ROOT / "data" / "raw"
OMNI_CSV = RAW_DIR / "omni_2021_full_calendar_v3.csv"
LHAASO_CSV = RAW_DIR / "lhaaso_shadow_pfss_2021_full_calendar_v3.csv"

TARGET = "SW Plasma Speed, km/s"
SHADOW = "p2_y中心值"
BASE_FEATURES = ["ss_br_mean", "BX, nT (GSE, GSM)", SHADOW]
FULL_FEATURES = [TARGET] + BASE_FEATURES

TEST_START = pd.Timestamp("2021-08-26")
HORIZON_DAYS = 3
PAST_LAG_DAYS = 3
WINDOW_SIZE = 14
TEST_RATIO = 0.2

ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
L1_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
MAX_COMBINATIONS = 30

RIDGE_FIXED_ALPHAS = [10.0, 100.0, 1000.0]
ENET_FIXED = [
    {"alpha": 10.0, "l1_ratio": 0.1},
    {"alpha": 10.0, "l1_ratio": 0.3},
    {"alpha": 10.0, "l1_ratio": 0.7},
]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pearson_r: float | None = None
    unavailable_reason: str | None = None
    if len(y_true) > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = float(np.corrcoef(y_true, y_pred)[0, 1])
        if math.isfinite(raw):
            pearson_r = raw
        elif float(np.std(y_true)) <= 1e-12:
            unavailable_reason = "constant_target"
        elif float(np.std(y_pred)) <= 1e-12:
            unavailable_reason = "constant_prediction"
        else:
            unavailable_reason = "undefined"
    else:
        unavailable_reason = "insufficient_samples"
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson_r,
        "unavailable_reason": unavailable_reason,
        "prediction_std": float(np.std(y_pred)),
    }


def _config(
    family: str,
    alpha: float,
    l1_ratio: float | None,
    search_mode: str,
    alpha_grid: list[float],
    l1_grid: list[float],
) -> ElasticNetConfig:
    return ElasticNetConfig(
        model_family=family,
        alpha=alpha,
        l1_ratio=l1_ratio if l1_ratio is not None else 0.5,
        random_state=42,
        alpha_grid=list(alpha_grid),
        l1_ratio_grid=list(l1_grid),
        param_search_mode=search_mode,
        max_param_search_combinations=MAX_COMBINATIONS,
        fit_intercept=True,
        ridge_solver="svd",
    )


def _pipeline(family: str, search) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                make_model(
                    _config(
                        family,
                        search.selected_alpha,
                        search.selected_l1_ratio,
                        "fixed",
                        ALPHA_GRID,
                        L1_GRID,
                    ),
                    alpha_override=search.selected_alpha,
                    l1_ratio_override=search.selected_l1_ratio,
                ),
            ),
        ]
    )


def _lag_column(variable: str, lag: int) -> str:
    return f"{variable}_t" if lag == 0 else f"{variable}_t-{lag}"


def _feature_sets() -> dict[str, list[str]]:
    target_features = {
        _lag_column(TARGET, lag): lag for lag in range(0, PAST_LAG_DAYS)
    }
    shadow_features = {
        _lag_column(SHADOW, lag): lag for lag in range(0, PAST_LAG_DAYS)
    }
    return {
        "vsw_history": list(target_features),
        "vsw_history_shadow": list(target_features) + list(shadow_features),
    }


def _extract_coefficients(
    pipeline: Pipeline,
    feature_cols: list[str],
    family: str,
) -> dict:
    model = pipeline.named_steps["model"]
    coef = np.asarray(model.coef_, dtype=float).reshape(-1)
    shadow_cols = [
        column for column in feature_cols if column.startswith(SHADOW + "_")
    ]
    shadow_abs = float(np.abs(coef[[feature_cols.index(c) for c in shadow_cols]]).sum())
    total_abs = float(np.abs(coef).sum())
    intercept = float(model.intercept_) if hasattr(model, "intercept_") else float("nan")
    return {
        "model_family": family,
        "coef_shadow_abs_sum": shadow_abs,
        "coef_total_abs_sum": total_abs,
        "coef_shadow_abs_share": (
            (shadow_abs / total_abs) if total_abs > 1e-12 else 0.0
        ),
        "intercept": intercept,
        "coefficients_json": json.dumps(
            {
                column: float(coef[index])
                for index, column in enumerate(feature_cols)
            },
            ensure_ascii=False,
        ),
        "shadow_zeroed": {
            column: bool(abs(float(coef[feature_cols.index(column)])) <= 1e-10)
            for column in shadow_cols
        },
    }


def main() -> int:
    output_dir = ROOT / "artifacts" / "v3_fixed_param_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_config = ExperimentConfig(
        file_config=FileConfig(
            omni_file_path=OMNI_CSV,
            lhaaso_file_path=LHAASO_CSV,
            time_column="TIME",
            target_column=TARGET,
            feature_columns=BASE_FEATURES,
            sparse_sources=["omni_2021_full_calendar_v3"],
        ),
        time_window_config=TimeWindowConfig(
            window_size=WINDOW_SIZE,
            test_split_ratio=TEST_RATIO,
            use_lag_feature=True,
            max_lag_day=PAST_LAG_DAYS,
            forecast_horizon_days=HORIZON_DAYS,
            past_lag_days=PAST_LAG_DAYS,
        ),
        elasticnet_config=_config("ridge", 0.5, 0.5, "auto", ALPHA_GRID, L1_GRID),
    )
    merged = load_and_merge_data(shared_config)
    merged.frame = merged.X.copy()
    merged.frame[TARGET] = merged.y

    sliding = build_sliding_window(
        merged.frame,
        merged.y,
        merged.time,
        shared_config.time_window_config,
        feature_columns=FULL_FEATURES,
    )
    retained = sliding.sample_plan.retained_records()
    archive_manifest = (
        SESSION_ROOT
        / "results"
        / "round_02"
        / "archive"
        / "E_R02_03_r1"
        / "sample_manifest.csv"
    )
    if archive_manifest.exists():
        archive_ids = set(
            pd.read_csv(archive_manifest)
            .loc[lambda frame: frame["sample_condition"] == "complete_case", "sample_id"]
        )
        current_ids = {record.sample_id for record in retained}
        if current_ids != archive_ids:
            raise RuntimeError(
                "诊断样本与 E_R02_03_r1 存档不一致: "
                f"current={len(current_ids)} archive={len(archive_ids)}"
            )

    shared_split = make_calendar_split(
        retained,
        test_ratio=TEST_RATIO,
        n_internal_folds=2,
        test_start=TEST_START,
    )
    split = split_dataset_by_calendar(
        sliding.X_all,
        sliding.y_all,
        sliding.time_all,
        sliding.origins_all,
        sliding.sample_ids_all,
        sliding.records_all,
        test_rate=TEST_RATIO,
        feature_names=sliding.feature_names,
        calendar_split=shared_split,
        n_internal_folds=2,
    )

    X_train = split.X_train
    X_test = split.X_test
    y_train = np.asarray(split.y_train, dtype=float)
    y_test = np.asarray(split.y_test, dtype=float)
    feature_sets = _feature_sets()

    rows: list[dict] = []
    coefficient_rows: list[dict] = []
    daily_frames: list[pd.DataFrame] = []
    for family, arms in (
        ("ridge", [{"alpha": alpha, "l1_ratio": None} for alpha in RIDGE_FIXED_ALPHAS]),
        (
            "elasticnet",
            [
                {"alpha": item["alpha"], "l1_ratio": item["l1_ratio"]}
                for item in ENET_FIXED
            ],
        ),
    ):
        for arm_index, arm in enumerate(arms):
            alpha = float(arm["alpha"])
            l1_ratio = arm["l1_ratio"]
            label = f"alpha={alpha:g}" if family == "ridge" else f"alpha={alpha:g},l1={l1_ratio:g}"
            for benchmark_type, feature_cols in feature_sets.items():
                missing = [column for column in feature_cols if column not in X_train.columns]
                if missing:
                    raise KeyError(f"{family} {benchmark_type} 缺少特征列: {missing}")
                search = select_best_params(
                    X_train[feature_cols],
                    y_train,
                    _config(
                        family,
                        alpha,
                        l1_ratio,
                        "fixed",
                        ALPHA_GRID,
                        L1_GRID,
                    ),
                    cv_folds=split.cv_folds,
                )
                pipeline = _pipeline(family, search)
                pipeline.fit(X_train[feature_cols], y_train)
                y_pred = np.asarray(
                    pipeline.predict(X_test[feature_cols]), dtype=float
                )
                run_id = f"{family}_fixed_{arm_index:02d}_{benchmark_type}"
                coefficient_row = _extract_coefficients(
                    pipeline, feature_cols, family
                )
                row = {
                    "run_id": run_id,
                    "family": family,
                    "fixed_label": label,
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                    "benchmark_type": benchmark_type,
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    **{
                        f"eval_{key}": value
                        for key, value in _metrics(y_test, y_pred).items()
                    },
                    **coefficient_row,
                }
                rows.append(row)
                coefficient_rows.append(
                    {
                        "run_id": run_id,
                        "family": family,
                        "fixed_label": label,
                        "benchmark_type": benchmark_type,
                        **coefficient_row,
                    }
                )
                daily_frames.append(
                    pd.DataFrame(
                        {
                            "run_id": run_id,
                            "sample_id": list(split.sample_ids_test),
                            "issue_date": pd.to_datetime(
                                split.origin_test
                            ).strftime("%Y-%m-%d"),
                            "target_date": pd.to_datetime(
                                split.time_test
                            ).strftime("%Y-%m-%d"),
                            "y_true": y_test,
                            "y_pred": y_pred,
                        }
                    )
                )

    summary = pd.DataFrame(rows)
    # One row per fixed model: pair base / shadow gains at identical parameters.
    comparison_rows = []
    grouped = summary.groupby(["family", "fixed_label"], sort=False)
    for (family, fixed_label), group in grouped:
        base = group.loc[group["benchmark_type"] == "vsw_history"]
        shadow = group.loc[group["benchmark_type"] == "vsw_history_shadow"]
        if base.empty or shadow.empty:
            continue
        base_row = base.iloc[0]
        shadow_row = shadow.iloc[0]
        comparison_rows.append(
            {
                "family": family,
                "fixed_label": fixed_label,
                "alpha": float(base_row["alpha"]),
                "l1_ratio": (
                    float(base_row["l1_ratio"]) if family == "elasticnet" else None
                ),
                "rmse_base": float(base_row["eval_rmse"]),
                "rmse_shadow": float(shadow_row["eval_rmse"]),
                "rmse_gain": float(base_row["eval_rmse"] - shadow_row["eval_rmse"]),
                "mae_base": float(base_row["eval_mae"]),
                "mae_shadow": float(shadow_row["eval_mae"]),
                "mae_gain": float(base_row["eval_mae"] - shadow_row["eval_mae"]),
                "r2_base": float(base_row["eval_r2"]),
                "r2_shadow": float(shadow_row["eval_r2"]),
                "pearson_base": (
                    float(base_row["eval_pearson_r"])
                    if base_row["eval_pearson_r"] is not None
                    else None
                ),
                "pearson_shadow": (
                    float(shadow_row["eval_pearson_r"])
                    if shadow_row["eval_pearson_r"] is not None
                    else None
                ),
                "shadow_abs_share": float(shadow_row["coef_shadow_abs_share"]),
                "shadow_zeroed": bool(
                    all(shadow_row["shadow_zeroed"].values())
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    summary_csv = output_dir / "fixed_param_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    comparison_csv = output_dir / "fixed_param_gain_comparison.csv"
    comparison.to_csv(comparison_csv, index=False, encoding="utf-8-sig")
    coefficient_csv = output_dir / "fixed_param_coefficients.csv"
    pd.DataFrame(coefficient_rows).to_csv(
        coefficient_csv, index=False, encoding="utf-8-sig"
    )
    daily_csv = output_dir / "fixed_param_daily.csv"
    pd.concat(daily_frames, ignore_index=True).to_csv(
        daily_csv, index=False, encoding="utf-8-sig"
    )

    manifest = {
        "source_run": "E_R02_03_r1",
        "data_version": "shadow-calendar-v1",
        "split_label": "dev_backtest",
        "phase_label": "迭代评估段",
        "test_start": str(TEST_START.date()),
        "test_end": str(pd.to_datetime(split.time_test).max().normalize().date()),
        "sample": {
            "retained_count": int(len(retained)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        },
        "time_definition": (
            "features use issue-date history; target date = issue date + "
            f"{HORIZON_DAYS} calendar days"
        ),
        "method": (
            "fixed-parameter arms only; both families use the same frozen "
            "calendars, folds, scaling policy and 35-day dev_backtest segment; "
            "no closed-loop state update was performed"
        ),
        "expected_caveat": (
            "zero gain here is still 35-day dev evidence, not a sealed final test"
        ),
        "summary_files": [
            summary_csv.name,
            comparison_csv.name,
            coefficient_csv.name,
            daily_csv.name,
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json_dumps(manifest, indent=2), encoding="utf-8")

    print(f"fixed-parameter diagnostics -> {output_dir}")
    print("\n=== gain comparison (positive = shadow arm improves eval RMSE) ===")
    print(
        comparison[
            [
                "family",
                "fixed_label",
                "rmse_base",
                "rmse_shadow",
                "rmse_gain",
                "mae_gain",
                "shadow_abs_share",
                "shadow_zeroed",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
