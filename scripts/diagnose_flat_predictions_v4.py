"""Stand-alone v4 flat-prediction diagnosis for the ShadowTracing v3 case.

The script answers two questions without touching the closed-loop state:

1. Why are the Ridge / ElasticNet predictions so flat?  We report the actual
   selected training-internal parameters, search/boundary/convergence status,
   train vs fold vs 35-day dev-segment metrics, prediction std, and parameter
   curves on identical frozen calendar folds.
2. Does a richer wind-history representation improve the baseline, and does
   the shadow increment survive on a shared sample and shared split?

Feature variants are kept deliberately small:

* A: recent 3 calendar-day solar wind speed values available at issue time.
* B: A + trailing 7-day mean / sd / linear trend ending at the issue date.
* C: A + Vsw at target_date - 27 calendar days (known at issue time because
  target_date - 27 days == issue_date - 24 days for a 3-day horizon).

No value is interpolated, no test date is moved, and the 35-day segment is
labelled ``dev_backtest`` because it has already been inspected repeatedly.
Every result table includes Pearson r / reason, not only RMSE.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.calendar_samples import (
    CalendarSample,
    build_sample_plan,
    make_calendar_split,
)
from core.json_safe import json_dumps
from core.robustness_diagnostics import paired_calendar_block_bootstrap
from models.elasticnet_runner import (
    ElasticNetConfig,
    ExperimentConfig,
    FileConfig,
    TimeWindowConfig,
    load_and_merge_data,
    make_model,
    select_best_params,
    split_dataset_by_calendar,
)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


SESSION_ROOT = (
    ROOT
    / "runtime"
    / "live_session"
    / "sessions"
    / "session_20261016_235451_805278"
)
RAW_DIR = SESSION_ROOT / "data" / "raw"
OMNI_CSV = RAW_DIR / "omni_2021_full_calendar_v3.csv"
LHAASO_CSV = RAW_DIR / "lhaaso_shadow_pfss_2021_full_calendar_v3.csv"

TARGET = "SW Plasma Speed, km/s"
SHADOW = "p2_y中心值"
TIME_COL = "TIME"
SPARSE_SOURCES = [LHAASO_CSV.stem]

HORIZON_DAYS = 3
TEST_START = pd.Timestamp("2021-08-26")
TEST_RATIO = 0.2
N_INTERNAL_FOLDS = 2
ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
L1_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
RANDOM_STATE = 42
BLOCK_SIZE = 7
N_DRAWS = 4000
BOOTSTRAP_SEED = 20261017
MIN_BLOCKS = 2

OUTPUT_DIR = ROOT / "artifacts" / "v4_flat_prediction_diagnosis"


SCHEME_A_COLS = ["vsw_lag0", "vsw_lag1", "vsw_lag2"]
SCHEME_B_COLS = [
    "vsw_lag0",
    "vsw_lag1",
    "vsw_lag2",
    "vsw_mean7",
    "vsw_std7",
    "vsw_trend7",
]
SCHEME_C_COLS = [
    "vsw_lag0",
    "vsw_lag1",
    "vsw_lag2",
    "vsw_recur_target_minus_27d",
]
SHADOW_COLS = ["shadow_lag0", "shadow_lag1", "shadow_lag2"]

SCHEMES = {
    "A": {
        "label": "A_3d_raw",
        "description": "最近3日历天原始风速",
        "cols": SCHEME_A_COLS,
    },
    "B": {
        "label": "B_3d_plus_7d_shape",
        "description": "最近3天 + 过去7天形态（均值/波动/趋势）",
        "cols": SCHEME_B_COLS,
    },
    "C": {
        "label": "C_3d_plus_27d_recur",
        "description": "最近3天 + 目标日前27天重现风速",
        "cols": SCHEME_C_COLS,
    },
}
ARM_COLS = {
    "base": {
        "label": "vsw_history_base",
        "description": "风速历史基线",
    },
    "shadow": {
        "label": "vsw_history_shadow",
        "description": "风速历史 + 日影",
    },
}
FAMILIES = ["ridge", "elasticnet"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pearson_r: float | None = None
    pearson_reason: str | None = None
    if len(y_true) > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = float(np.corrcoef(y_true, y_pred)[0, 1])
        if np.isfinite(raw):
            pearson_r = raw
        elif float(np.std(y_true)) <= 1e-12:
            pearson_reason = "constant_target"
        elif float(np.std(y_pred)) <= 1e-12:
            pearson_reason = "constant_prediction"
        else:
            pearson_reason = "undefined"
    else:
        pearson_reason = "insufficient_samples"
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson_r,
        "pearson_reason": pearson_reason,
        "y_std": float(np.std(y_true)),
        "pred_std": float(np.std(y_pred)),
        "n": int(len(y_true)),
    }


def _model_config(family: str) -> ElasticNetConfig:
    return ElasticNetConfig(
        model_family=family,
        alpha=1.0,
        l1_ratio=0.5,
        random_state=RANDOM_STATE,
        alpha_grid=list(ALPHA_GRID),
        l1_ratio_grid=list(L1_GRID),
        param_search_mode="auto",
        max_param_search_combinations=64,
        fit_intercept=True,
        ridge_solver="svd",
    )


def _refit_selected_pipeline(
    family: str,
    search,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Pipeline:
    cfg = _model_config(family)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                make_model(
                    cfg,
                    alpha_override=search.selected_alpha,
                    l1_ratio_override=search.selected_l1_ratio,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def _feature_prediction(
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    time_raw: pd.Series,
    records: list[CalendarSample],
    columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Extract issue-date available features for already-built calendar records.

    All columns here are explicitly constructed so they are known at the issue
    date; no lag of a target is created automatically by the caller.
    """

    frame = X_raw.copy()
    if "_hp_y" in frame.columns:
        frame["_hp_y"] = pd.to_numeric(frame["_hp_y"], errors="coerce")
    else:
        frame["_hp_y"] = pd.to_numeric(
            pd.Series(y_raw, index=X_raw.index), errors="coerce"
        )
    frame["_hp_time"] = pd.to_datetime(
        pd.Series(time_raw, index=X_raw.index), errors="coerce"
    ).dt.normalize()
    frame = frame.sort_values("_hp_time", kind="mergesort").reset_index(drop=True)
    date_to_index = {
        pd.Timestamp(value).normalize(): index
        for index, value in enumerate(frame["_hp_time"])
    }
    values: list[list[float]] = []
    target_values: list[float] = []
    target_dates: list[pd.Timestamp] = []
    origins: list[pd.Timestamp] = []
    sample_ids: list[str] = []
    for record in records:
        row: list[float] = []
        for column in columns:
            feature_date = record.issue_date
            row_index = date_to_index[pd.Timestamp(feature_date).normalize()]
            row.append(float(frame.iloc[row_index][column]))
        target_index = date_to_index[
            pd.Timestamp(record.target_date).normalize()
        ]
        values.append(row)
        target_values.append(float(frame.iloc[target_index]["_hp_y"]))
        target_dates.append(record.target_date)
        origins.append(record.issue_date)
        sample_ids.append(record.sample_id)
    feature_names = [f"{column}_t" for column in columns]
    matrix = pd.DataFrame(values, columns=feature_names)
    return (
        matrix,
        np.asarray(target_values, dtype=float),
        target_dates,
        feature_names,
    )


def _persistence_predictions(
    omni: pd.DataFrame,
    issue_dates: Iterable[pd.Timestamp],
) -> tuple[list[float], int]:
    omni = omni.copy()
    omni["_date"] = pd.to_datetime(omni[TIME_COL], format="%Y%m%d")
    omni["_vsw"] = pd.to_numeric(omni[TARGET], errors="coerce")
    omni = omni.dropna(subset=["_vsw"]).sort_values("_date")
    valid_dates = omni["_date"].to_numpy()
    valid_values = omni["_vsw"].to_numpy(dtype=float)
    values: list[float] = []
    fallback = 0
    for issue in issue_dates:
        available = np.flatnonzero(valid_dates <= np.datetime64(issue))
        if len(available) == 0:
            raise RuntimeError(f"发报日 {issue} 之前没有风速记录")
        index = int(available[-1])
        values.append(float(valid_values[index]))
        if pd.Timestamp(valid_dates[index]).normalize() != pd.Timestamp(issue).normalize():
            fallback += 1
    return values, fallback


def _build_omnibus_features(merged: pd.DataFrame) -> pd.DataFrame:
    frame = merged.copy()
    frame["_date"] = pd.to_datetime(frame[TIME_COL], errors="coerce").dt.normalize()
    frame["_vsw"] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame["_shadow"] = pd.to_numeric(frame[SHADOW], errors="coerce")
    frame = frame.sort_values("_date").reset_index(drop=True)
    date_values = {
        pd.Timestamp(row["_date"]).normalize(): float(row["_vsw"])
        for _, row in frame.iterrows()
    }
    shadow_values = {
        pd.Timestamp(row["_date"]).normalize(): (
            float(row["_shadow"]) if pd.notna(row["_shadow"]) else np.nan
        )
        for _, row in frame.iterrows()
    }

    def vsw_at(value: pd.Timestamp) -> float | None:
        value = pd.Timestamp(value).normalize()
        if value not in date_values:
            return None
        return date_values[value]

    def shadow_at(value: pd.Timestamp) -> float | None:
        value = pd.Timestamp(value).normalize()
        if value not in shadow_values:
            return None
        result = shadow_values[value]
        return None if pd.isna(result) else float(result)

    rows: list[dict] = []
    for _, row in frame.iterrows():
        issue = pd.Timestamp(row["_date"]).normalize()
        base: dict = {
            "vsw_lag0": vsw_at(issue),
            "vsw_lag1": vsw_at(issue - pd.Timedelta(days=1)),
            "vsw_lag2": vsw_at(issue - pd.Timedelta(days=2)),
        }

        trailing = [vsw_at(issue - pd.Timedelta(days=offset)) for offset in range(7)]
        if any(value is None or not math.isfinite(value) for value in trailing):
            base["vsw_mean7"] = np.nan
            base["vsw_std7"] = np.nan
            base["vsw_trend7"] = np.nan
        else:
            values = np.asarray(trailing, dtype=float)
            days = np.arange(7, dtype=float)
            base["vsw_mean7"] = float(values.mean())
            base["vsw_std7"] = float(values.std(ddof=1))
            slope = float(np.polyfit(days, values, deg=1)[0])
            base["vsw_trend7"] = slope

        target_date = issue + pd.Timedelta(days=HORIZON_DAYS)
        recur_date = target_date - pd.Timedelta(days=27)
        base["vsw_recur_target_minus_27d"] = vsw_at(recur_date)

        for offset, name in enumerate(("shadow_lag0", "shadow_lag1", "shadow_lag2")):
            base[name] = shadow_at(issue - pd.Timedelta(days=offset))
        rows.append(base)

    feature_frame = pd.DataFrame(rows)
    feature_frame[TARGET] = frame["_vsw"].to_numpy()
    return feature_frame


def _param_curve_table(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_folds: list[tuple[np.ndarray, np.ndarray]],
    family: str,
) -> pd.DataFrame:
    cfg = _model_config(family)
    l1_values: list[float | None] = [None] if family == "ridge" else list(L1_GRID)
    records: list[dict] = []
    for alpha in ALPHA_GRID:
        for l1_ratio in l1_values:
            fold_rows: list[dict] = []
            for fold_index, (train_index, eval_index) in enumerate(cv_folds):
                pipeline = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            make_model(
                                cfg,
                                alpha_override=alpha,
                                l1_ratio_override=l1_ratio,
                            ),
                        ),
                    ]
                )
                pipeline.fit(X_train.iloc[train_index], y_train[train_index])
                train_pred = pipeline.predict(X_train.iloc[train_index])
                eval_pred = pipeline.predict(X_train.iloc[eval_index])
                fold_rows.append(
                    {
                        "fold": fold_index + 1,
                        "fold_train_rmse": float(
                            np.sqrt(
                                np.mean(
                                    np.square(
                                        y_train[train_index] - train_pred
                                    )
                                )
                            )
                        ),
                        "fold_eval_rmse": float(
                            np.sqrt(
                                np.mean(
                                    np.square(
                                        y_train[eval_index] - eval_pred
                                    )
                                )
                            )
                        ),
                    }
                )
            fold_df = pd.DataFrame(fold_rows)
            records.append(
                {
                    "alpha": float(alpha),
                    "l1_ratio": l1_ratio,
                    "mean_train_rmse": float(fold_df["fold_train_rmse"].mean()),
                    "mean_eval_rmse": float(fold_df["fold_eval_rmse"].mean()),
                    "fold1_eval_rmse": float(fold_df.iloc[0]["fold_eval_rmse"]),
                    "fold2_eval_rmse": float(fold_df.iloc[1]["fold_eval_rmse"]),
                }
            )
    return pd.DataFrame(records)


def _search_rows(search, family: str) -> list[dict]:
    rows = list(getattr(search, "rows", []) or [])
    for row in rows:
        row.setdefault("family", family)
        for key in (
            "split0_test_neg_mse",
            "split1_test_neg_mse",
            "mean_test_neg_mse",
            "mean_test_rmse",
        ):
            if key in row:
                row[key] = (
                    float(row[key]) if row[key] is not None else None
                )
    return rows


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(
        file_config=FileConfig(
            omni_file_path=OMNI_CSV,
            lhaaso_file_path=LHAASO_CSV,
            time_column=TIME_COL,
            target_column=TARGET,
            feature_columns=[SHADOW],
            sparse_sources=SPARSE_SOURCES,
        ),
        time_window_config=TimeWindowConfig(
            window_size=14,
            test_split_ratio=TEST_RATIO,
            use_lag_feature=True,
            max_lag_day=3,
            forecast_horizon_days=HORIZON_DAYS,
            past_lag_days=3,
        ),
        elasticnet_config=_model_config("ridge"),
    )
    merged = load_and_merge_data(config)
    feature_frame = _build_omnibus_features(
        merged.X.join(merged.y.rename(TARGET)).join(
            pd.Series(merged.time, name=TIME_COL)
        )
    )
    y_series = merged.y.astype(float).rename(TARGET)
    time_series = pd.Series(
        pd.to_datetime(merged.time, errors="coerce"),
        name=TIME_COL,
    )

    all_columns = (
        SCHEME_A_COLS + SCHEME_B_COLS + SCHEME_C_COLS + SHADOW_COLS
    )
    common_plan = build_sample_plan(
        feature_frame[all_columns],
        y_series,
        time_series,
        feature_columns=all_columns,
        target_name=TARGET,
        past_lag_days=1,
        forecast_horizon_days=HORIZON_DAYS,
    )
    common_records = common_plan.retained_records()
    common_keys = {
        (
            record.issue_date.date().isoformat(),
            record.target_date.date().isoformat(),
            int(record.horizon_days),
        )
        for record in common_records
    }
    common_split = make_calendar_split(
        common_records,
        test_ratio=TEST_RATIO,
        n_internal_folds=N_INTERNAL_FOLDS,
        test_start=TEST_START,
    )
    test_count = len(common_split.test)
    if test_count != 35:
        raise RuntimeError(
            f"冻结测试段应包含 35 个 issue 日期，实际 {test_count}"
        )

    scheme_audit_rows: list[dict] = []
    plan_objects: dict[str, dict] = {}
    for scheme_key, scheme in SCHEMES.items():
        plan_cols = list(dict.fromkeys(scheme["cols"] + SHADOW_COLS))
        plan = build_sample_plan(
            feature_frame[plan_cols],
            y_series,
            time_series,
            feature_columns=plan_cols,
            target_name=TARGET,
            past_lag_days=1,
            forecast_horizon_days=HORIZON_DAYS,
        )
        plan_objects[scheme_key] = plan
        scheme_audit_rows.append(
            {
                "scheme": scheme_key,
                "candidate_count": plan.candidate_count,
                "retained_with_shadow": plan.retained_count,
                "dropped_count": plan.dropped_count,
                "retained_common_after_all_schemes": len(common_records),
                "drop_reasons": json.dumps(
                    plan.drop_reason_counts(), ensure_ascii=False
                ),
            }
        )
    audit_df = pd.DataFrame(scheme_audit_rows)
    audit_csv = OUTPUT_DIR / "common_sample_audit.csv"
    audit_df.to_csv(audit_csv, index=False, encoding="utf-8-sig")

    # Common arrays for all schemes (same records, same rows order).
    matrices: dict[str, pd.DataFrame] = {}
    target_arrays: dict[str, np.ndarray] = {}
    for scheme_key, scheme in SCHEMES.items():
        matrix, target_values, _, feature_names = _feature_prediction(
            feature_frame,
            y_series,
            time_series,
            common_records,
            scheme["cols"],
        )
        matrices[scheme_key] = matrix
        target_arrays[scheme_key] = target_values

    parameter_rows: list[dict] = []
    segmented_rows: list[dict] = []
    curve_frames: list[pd.DataFrame] = []
    search_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    coefficient_rows: list[dict] = []

    for scheme_key, scheme in SCHEMES.items():
        scheme_matrix = matrices[scheme_key]
        scheme_target = target_arrays[scheme_key]
        target_dates = pd.to_datetime(
            [record.target_date for record in common_records]
        )
        origins = pd.to_datetime(
            [record.issue_date for record in common_records]
        )
        sample_ids = [record.sample_id for record in common_records]
        feature_names = list(scheme_matrix.columns)
        for arm_key, arm in ARM_COLS.items():
            columns = (
                scheme["cols"]
                if arm_key == "base"
                else scheme["cols"] + SHADOW_COLS
            )
            arm_matrix, arm_target, _, arm_feature_names = _feature_prediction(
                feature_frame,
                y_series,
                time_series,
                common_records,
                columns,
            )
            split = split_dataset_by_calendar(
                arm_matrix.to_numpy(dtype=float),
                arm_target,
                target_dates.to_numpy(),
                origins.to_numpy(),
                sample_ids,
                common_records,
                test_rate=TEST_RATIO,
                feature_names=arm_feature_names,
                n_internal_folds=N_INTERNAL_FOLDS,
                calendar_split=common_split,
            )
            X_train = split.X_train
            y_train = np.asarray(split.y_train, dtype=float)
            X_test = split.X_test
            y_test = np.asarray(split.y_test, dtype=float)

            for family in FAMILIES:
                cfg = _model_config(family)
                search = select_best_params(
                    X_train,
                    y_train,
                    cfg,
                    cv_folds=split.cv_folds,
                )
                pipeline = _refit_selected_pipeline(
                    family, search, X_train, y_train
                )
                run_id = f"{scheme_key}_{arm_key}_{family}"
                fitted_model = pipeline.named_steps["model"]
                for feature_index, feature_name in enumerate(X_train.columns):
                    coefficient_rows.append(
                        {
                            "run_id": run_id,
                            "scheme": scheme_key,
                            "arm": arm_key,
                            "model_family": family,
                            "selected_alpha": search.selected_alpha,
                            "selected_l1_ratio": (
                                search.selected_l1_ratio
                                if search.selected_l1_ratio is not None
                                else None
                            ),
                            "feature": feature_name,
                            "coefficient": float(
                                fitted_model.coef_[feature_index]
                            ),
                            "intercept": float(fitted_model.intercept_),
                        }
                    )
                y_train_pred = np.asarray(
                    pipeline.predict(X_train), dtype=float
                )
                y_test_pred = np.asarray(
                    pipeline.predict(X_test), dtype=float
                )
                run_id = f"{scheme_key}_{arm_key}_{family}"
                train_metrics = _metrics(y_train, y_train_pred)
                test_metrics = _metrics(y_test, y_test_pred)
                parameter_rows.append(
                    {
                        "run_id": run_id,
                        "scheme": scheme_key,
                        "scheme_label": scheme["label"],
                        "arm": arm_key,
                        "arm_label": arm["label"],
                        "model_family": family,
                        "selected_alpha": float(search.selected_alpha),
                        "selected_l1_ratio": (
                            float(search.selected_l1_ratio)
                            if search.selected_l1_ratio is not None
                            else None
                        ),
                        "search_mode": search.search_mode,
                        "search_completed": search.search_completed,
                        "boundary_hit": search.boundary_hit,
                        "solver_converged": search.solver_converged,
                        "grid_size": search.grid_size,
                        "n_train": int(len(y_train)),
                        "n_test": int(len(y_test)),
                        "train_rmse": train_metrics["rmse"],
                        "train_mae": train_metrics["mae"],
                        "train_pearson_r": train_metrics["pearson_r"],
                        "train_r2": train_metrics["r2"],
                        "train_pred_std": train_metrics["pred_std"],
                        "train_y_std": train_metrics["y_std"],
                        "eval_rmse": test_metrics["rmse"],
                        "eval_mae": test_metrics["mae"],
                        "eval_pearson_r": test_metrics["pearson_r"],
                        "eval_r2": test_metrics["r2"],
                        "eval_pred_std": test_metrics["pred_std"],
                        "eval_y_std": test_metrics["y_std"],
                        "boundary_note": search.boundary_note(),
                    }
                )

                segmented_rows.extend(
                    [
                        {
                            "run_id": run_id,
                            "segment": "train_fit",
                            "date_start": str(
                                pd.to_datetime(
                                    split.origin_train
                                ).min().date()
                            ),
                            "date_end": str(
                                pd.to_datetime(
                                    split.origin_train
                                ).max().date()
                            ),
                            **{
                                f"metric_{key}": value
                                for key, value in train_metrics.items()
                            },
                        },
                        {
                            "run_id": run_id,
                            "segment": "dev_backtest",
                            "date_start": str(
                                pd.to_datetime(
                                    split.origin_test
                                ).min().date()
                            ),
                            "date_end": str(
                                pd.to_datetime(
                                    split.origin_test
                                ).max().date()
                            ),
                            **{
                                f"metric_{key}": value
                                for key, value in test_metrics.items()
                            },
                        },
                    ]
                )

                for fold_index, (train_index, eval_index) in enumerate(
                    split.cv_folds
                ):
                    fold_pipeline = _refit_selected_pipeline(
                        family, search, X_train, y_train
                    )
                    fold_pred = np.asarray(
                        fold_pipeline.predict(X_train.iloc[eval_index]),
                        dtype=float,
                    )
                    fold_metrics = _metrics(y_train[eval_index], fold_pred)
                    fold_frames.append(
                        {
                            "run_id": run_id,
                            "fold": fold_index + 1,
                            "n_train_fold": int(len(train_index)),
                            "n_eval_fold": int(len(eval_index)),
                            **{
                                f"metric_{key}": value
                                for key, value in fold_metrics.items()
                            },
                        }
                    )

                curve = _param_curve_table(
                    X_train,
                    y_train,
                    split.cv_folds,
                    family=family,
                )
                curve["run_id"] = run_id
                curve["scheme"] = scheme_key
                curve["arm"] = arm_key
                curve["model_family"] = family
                curve_frames.append(curve)

                search_rows = _search_rows(search, family)
                if search_rows:
                    for row in search_rows:
                        row["run_id"] = run_id
                        row["scheme"] = scheme_key
                        row["arm"] = arm_key
                    search_frames.append(pd.DataFrame(search_rows))

                test_daily = pd.DataFrame(
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
                        "y_pred": y_test_pred,
                    }
                )
                daily_frames.append(test_daily)

            # No-penalty reference, same frozen folds, no hyper-parameter search.
            family = "linear_unpenalized"
            pipeline = Pipeline(
                [("scaler", StandardScaler()), ("model", LinearRegression())]
            )
            pipeline.fit(X_train, y_train)
            run_id = f"{scheme_key}_{arm_key}_{family}"
            fitted_model = pipeline.named_steps["model"]
            for feature_index, feature_name in enumerate(X_train.columns):
                coefficient_rows.append(
                    {
                        "run_id": run_id,
                        "scheme": scheme_key,
                        "arm": arm_key,
                        "model_family": family,
                        "selected_alpha": None,
                        "selected_l1_ratio": None,
                        "feature": feature_name,
                        "coefficient": float(
                            fitted_model.coef_[feature_index]
                        ),
                        "intercept": float(fitted_model.intercept_),
                    }
                )
            y_train_pred = np.asarray(
                pipeline.predict(X_train), dtype=float
            )
            y_test_pred = np.asarray(
                pipeline.predict(X_test), dtype=float
            )
            train_metrics = _metrics(y_train, y_train_pred)
            test_metrics = _metrics(y_test, y_test_pred)
            parameter_rows.append(
                {
                    "run_id": run_id,
                    "scheme": scheme_key,
                    "scheme_label": scheme["label"],
                    "arm": arm_key,
                    "arm_label": arm["label"],
                    "model_family": family,
                    "selected_alpha": None,
                    "selected_l1_ratio": None,
                    "search_mode": "fixed",
                    "search_completed": None,
                    "boundary_hit": None,
                    "solver_converged": None,
                    "grid_size": 0,
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "train_rmse": train_metrics["rmse"],
                    "train_mae": train_metrics["mae"],
                    "train_pearson_r": train_metrics["pearson_r"],
                    "train_r2": train_metrics["r2"],
                    "train_pred_std": train_metrics["pred_std"],
                    "train_y_std": train_metrics["y_std"],
                    "eval_rmse": test_metrics["rmse"],
                    "eval_mae": test_metrics["mae"],
                    "eval_pearson_r": test_metrics["pearson_r"],
                    "eval_r2": test_metrics["r2"],
                    "eval_pred_std": test_metrics["pred_std"],
                    "eval_y_std": test_metrics["y_std"],
                    "boundary_note": "无惩罚 LinearRegression 参照",
                }
            )
            segmented_rows.extend(
                [
                    {
                        "run_id": run_id,
                        "segment": "train_fit",
                        "date_start": str(
                            pd.to_datetime(
                                split.origin_train
                            ).min().date()
                        ),
                        "date_end": str(
                            pd.to_datetime(
                                split.origin_train
                            ).max().date()
                        ),
                        **{
                            f"metric_{key}": value
                            for key, value in train_metrics.items()
                        },
                    },
                    {
                        "run_id": run_id,
                        "segment": "dev_backtest",
                        "date_start": str(
                            pd.to_datetime(
                                split.origin_test
                            ).min().date()
                        ),
                        "date_end": str(
                            pd.to_datetime(
                                split.origin_test
                            ).max().date()
                        ),
                        **{
                            f"metric_{key}": value
                            for key, value in test_metrics.items()
                        },
                    },
                ]
            )
            test_daily = pd.DataFrame(
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
                    "y_pred": y_test_pred,
                }
            )
            daily_frames.append(test_daily)

    parameter_df = pd.DataFrame(parameter_rows)
    parameter_csv = OUTPUT_DIR / "parameter_summary.csv"
    parameter_df.to_csv(parameter_csv, index=False, encoding="utf-8-sig")

    segmented_df = pd.DataFrame(segmented_rows)
    segmented_csv = OUTPUT_DIR / "segmented_metrics.csv"
    segmented_df.to_csv(segmented_csv, index=False, encoding="utf-8-sig")

    curve_df = pd.concat(curve_frames, ignore_index=True)
    curve_csv = OUTPUT_DIR / "parameter_curves.csv"
    curve_df.to_csv(curve_csv, index=False, encoding="utf-8-sig")

    if search_frames:
        search_df = pd.concat(search_frames, ignore_index=True)
        search_csv = OUTPUT_DIR / "param_search_all.csv"
        search_df.to_csv(search_csv, index=False, encoding="utf-8-sig")
    else:
        search_csv = None

    fold_df = pd.DataFrame(fold_frames)
    fold_csv = OUTPUT_DIR / "cv_fold_metrics.csv"
    fold_df.to_csv(fold_csv, index=False, encoding="utf-8-sig")

    daily_df = pd.concat(daily_frames, ignore_index=True)
    daily_csv = OUTPUT_DIR / "daily_predictions.csv"
    daily_df.to_csv(daily_csv, index=False, encoding="utf-8-sig")

    coefficient_df = pd.DataFrame(coefficient_rows)
    coefficient_csv = OUTPUT_DIR / "coefficients.csv"
    coefficient_df.to_csv(coefficient_csv, index=False, encoding="utf-8-sig")

    # Simple baselines on the same 35 dates.
    canonical = (
        daily_df.loc[daily_df["run_id"] == "A_base_ridge"]
        .sort_values("issue_date")
        .reset_index(drop=True)
    )
    y_test = canonical["y_true"].to_numpy(dtype=float)
    test_issues = pd.to_datetime(canonical["issue_date"])
    omni = pd.read_csv(OMNI_CSV)
    persistence_values, fallback_count = _persistence_predictions(
        omni, test_issues
    )
    train_mean_rows = {}
    for scheme_key in SCHEMES:
        split_for_mean = make_calendar_split(
            common_records,
            test_ratio=TEST_RATIO,
            n_internal_folds=N_INTERNAL_FOLDS,
            test_start=TEST_START,
        )
        train_ids = {record.sample_id for record in split_for_mean.train}
        train_y = np.asarray(
            [
                target_arrays[scheme_key][index]
                for index, record in enumerate(common_records)
                if record.sample_id in train_ids
            ]
        )
        train_mean_rows[scheme_key] = float(train_y.mean())

    baseline_rows: list[dict] = []
    for scheme_key in SCHEMES:
        for name, values in (
            ("train_mean", np.full(len(y_test), train_mean_rows[scheme_key])),
            ("persistence_latest_vsw", np.asarray(persistence_values)),
        ):
            baseline_rows.append(
                {
                    "scheme": scheme_key,
                    "baseline_name": name,
                    **{
                        f"eval_{key}": value
                        for key, value in _metrics(y_test, values).items()
                    },
                }
            )
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_csv = OUTPUT_DIR / "simple_baselines.csv"
    baseline_df.to_csv(baseline_csv, index=False, encoding="utf-8-sig")

    # Gain summary and paired bootstrap CI (RMSE + Pearson).
    eval_rows: list[dict] = []
    gain_rows: list[dict] = []
    for scheme_key in SCHEMES:
        for family in ["ridge", "elasticnet", "linear_unpenalized"]:
            base_run_id = f"{scheme_key}_base_{family}"
            treat_run_id = f"{scheme_key}_shadow_{family}"
            base_daily = (
                daily_df.loc[daily_df["run_id"] == base_run_id]
                .sort_values("issue_date")
                .reset_index(drop=True)
            )
            treat_daily = (
                daily_df.loc[daily_df["run_id"] == treat_run_id]
                .sort_values("issue_date")
                .reset_index(drop=True)
            )
            if base_daily.empty or treat_daily.empty:
                continue
            base_row = parameter_df.loc[
                parameter_df["run_id"] == base_run_id
            ].iloc[0]
            treat_row = parameter_df.loc[
                parameter_df["run_id"] == treat_run_id
            ].iloc[0]
            eval_rows.append(
                {
                    "scheme": scheme_key,
                    "model_family": family,
                    "run_id_base": base_run_id,
                    "run_id_treatment": treat_run_id,
                    "base_alpha": base_row["selected_alpha"],
                    "base_l1_ratio": base_row["selected_l1_ratio"],
                    "treat_alpha": treat_row["selected_alpha"],
                    "treat_l1_ratio": treat_row["selected_l1_ratio"],
                    "base_rmse": base_row["eval_rmse"],
                    "treat_rmse": treat_row["eval_rmse"],
                    "base_pearson_r": base_row["eval_pearson_r"],
                    "treat_pearson_r": treat_row["eval_pearson_r"],
                    "base_pred_std": base_row["eval_pred_std"],
                    "treat_pred_std": treat_row["eval_pred_std"],
                    "rmse_gain": float(base_row["eval_rmse"])
                    - float(treat_row["eval_rmse"]),
                    "mae_gain": float(base_row["eval_mae"])
                    - float(treat_row["eval_mae"]),
                    "pearson_gain": (
                        (
                            float(treat_row["eval_pearson_r"])
                            - float(base_row["eval_pearson_r"])
                        )
                        if (
                            base_row["eval_pearson_r"] is not None
                            and treat_row["eval_pearson_r"] is not None
                        )
                        else None
                    ),
                    "n_test": int(len(y_test)),
                }
            )
            bootstrap = paired_calendar_block_bootstrap(
                base_daily[["sample_id", "issue_date", "y_true", "y_pred"]],
                treat_daily[["sample_id", "issue_date", "y_true", "y_pred"]],
                block_size=BLOCK_SIZE,
                n_draws=N_DRAWS,
                seed=BOOTSTRAP_SEED,
                min_blocks=MIN_BLOCKS,
                verify_pearson=True,
            )
            gain_rows.append(
                {
                    "scheme": scheme_key,
                    "model_family": family,
                    "run_id_base": base_run_id,
                    "run_id_treatment": treat_run_id,
                    "rmse_gain": bootstrap["point_gain_rmse"],
                    "rmse_ci_low": bootstrap["rmse_ci"][0],
                    "rmse_ci_high": bootstrap["rmse_ci"][1],
                    "mae_gain": bootstrap["point_gain_mae"],
                    "pearson_gain": bootstrap["point_gain_pearson_r"],
                    "pearson_ci_low": bootstrap["pearson_r_ci"][0],
                    "pearson_ci_high": bootstrap["pearson_r_ci"][1],
                    "stable_rmse": bootstrap["stable_rmse"],
                    "n_nonempty_blocks": bootstrap["n_nonempty_blocks"],
                    "insufficient_blocks": bootstrap["insufficient_blocks"],
                    "n_valid_pearson_draws": bootstrap[
                        "n_valid_pearson_draws"
                    ],
                }
            )

    eval_df = pd.DataFrame(eval_rows)
    eval_csv = OUTPUT_DIR / "eval_comparison.csv"
    eval_df.to_csv(eval_csv, index=False, encoding="utf-8-sig")

    gain_df = pd.DataFrame(gain_rows)
    gain_csv = OUTPUT_DIR / "gain_ci.csv"
    gain_df.to_csv(gain_csv, index=False, encoding="utf-8-sig")

    # Figures.
    _plot_parameter_curves(curve_df)
    _plot_eval_comparison(eval_df)
    _plot_gain_ci(gain_df)

    # Machine-readable conclusions.
    conclusion_hints: list[str] = []
    for scheme_key in SCHEMES:
        subset = gain_df.loc[gain_df["scheme"] == scheme_key]
        for _, row in subset.iterrows():
            family = row["model_family"]
            if row["stable_rmse"]:
                conclusion_hints.append(
                    f"{scheme_key}/{family}: RMSE CI 不跨零"
                )
            elif (
                pd.isna(row["rmse_ci_low"])
                or pd.isna(row["rmse_ci_high"])
            ):
                conclusion_hints.append(
                    f"{scheme_key}/{family}: RMSE CI 不可用"
                )
            else:
                conclusion_hints.append(
                    f"{scheme_key}/{family}: RMSE CI 跨零（未检出稳定增益）"
                )

    manifest = {
        "task": "v4_flat_prediction_diagnosis",
        "session_root": str(SESSION_ROOT),
        "data_files": {
            "omni": str(OMNI_CSV),
            "omni_sha256": _sha256(OMNI_CSV),
            "lhaaso": str(LHAASO_CSV),
            "lhaaso_sha256": _sha256(LHAASO_CSV),
        },
        "split_label": "dev_backtest",
        "phase_label": "迭代评估段（已反复查看，不是独立盲测）",
        "test_start": str(TEST_START.date()),
        "test_end": "2021-09-29",
        "horizon_days": HORIZON_DAYS,
        "time_definition": (
            "特征在 issue_date 时已可获得；目标 = issue_date + "
            f"{HORIZON_DAYS} 日历天；C 方案复发特征 = target_date - 27 天"
        ),
        "method": (
            "同一冻结日历样本与同一 time-series folds；Ridge/ElasticNet 各自训练内"
            "网格选参，scaler 仅在每折训练部分拟合；另加无惩罚 LinearRegression 参照。"
            "35 天结果只按 dev_backtest 报告。"
        ),
        "feature_variants": {
            key: {
                "label": value["label"],
                "description": value["description"],
                "columns": value["cols"],
            }
            for key, value in SCHEMES.items()
        },
        "alpha_grid": ALPHA_GRID,
        "l1_ratio_grid": L1_GRID,
        "common_sample": {
            "retained_count": len(common_records),
            "train_count": len(common_split.train),
            "test_count": len(common_split.test),
            "boundary_excluded_count": len(common_split.boundary_excluded),
            "first_issue": str(common_records[0].issue_date.date()),
            "last_issue": str(common_records[-1].issue_date.date()),
        },
        "bootstrap": {
            "block_size": BLOCK_SIZE,
            "n_draws": N_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "min_blocks": MIN_BLOCKS,
        },
        "conclusion_hints": conclusion_hints,
        "summary_files": [
            audit_csv.name,
            parameter_csv.name,
            segmented_csv.name,
            curve_csv.name,
            (search_csv.name if search_csv is not None else None),
            fold_csv.name,
            daily_csv.name,
            coefficient_csv.name,
            baseline_csv.name,
            eval_csv.name,
            gain_csv.name,
            "parameter_curves.png",
            "eval_comparison.png",
            "gain_ci.png",
        ],
        "expected_caveat": (
            "dev_backtest 上任何改善都只能支持继续设计区分性实验，"
            "不能作为独立最终测试成绩。"
        ),
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json_dumps(manifest, indent=2), encoding="utf-8")

    print(f"diagnostics -> {OUTPUT_DIR}")
    print("\n=== selected parameters (training-internal search) ===")
    print(
        parameter_df[
            [
                "run_id",
                "scheme",
                "arm",
                "model_family",
                "selected_alpha",
                "selected_l1_ratio",
                "search_completed",
                "boundary_hit",
                "solver_converged",
                "train_rmse",
                "eval_rmse",
                "eval_pearson_r",
                "eval_pred_std",
            ]
        ].to_string(index=False)
    )
    print("\n=== A/B/C gain on the same 35 dev dates ===")
    print(
        gain_df[
            [
                "scheme",
                "model_family",
                "rmse_gain",
                "rmse_ci_low",
                "rmse_ci_high",
                "pearson_gain",
                "pearson_ci_low",
                "pearson_ci_high",
                "stable_rmse",
            ]
        ].to_string(index=False)
    )
    return 0


def _plot_parameter_curves(curve_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        constrained_layout=True,
        sharex=True,
    )
    for family, axis in zip(FAMILIES, axes):
        subset = curve_df.loc[curve_df["model_family"] == family]
        # Represent each scheme-arm by the selected-l1 agnostic alpha curve,
        # using the mean over available l1_ratio rows.
        agg = (
            subset.groupby(["scheme", "arm", "alpha"])
            .agg(
                mean_train_rmse=("mean_train_rmse", "mean"),
                mean_eval_rmse=("mean_eval_rmse", "mean"),
            )
            .reset_index()
        )
        for (scheme, arm), group in agg.groupby(["scheme", "arm"]):
            label = (
                f"{scheme}-{'基线' if arm == 'base' else '日影'}"
            )
            axis.plot(
                group["alpha"],
                group["mean_eval_rmse"],
                marker="o",
                label=f"{label} 训练内验证",
            )
            axis.plot(
                group["alpha"],
                group["mean_train_rmse"],
                marker="x",
                linestyle="--",
                alpha=0.7,
                label=f"{label} 折内训练",
            )
        axis.set_xscale("log")
        axis.set_ylabel("RMSE, km/s")
        axis.set_title(f"{family.title()} 参数曲线（共同日历折）")
        axis.grid(True, which="both", axis="both", alpha=0.3)
        axis.legend(fontsize=8)
    axes[1].set_xlabel("alpha")
    path = OUTPUT_DIR / "parameter_curves.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_eval_comparison(eval_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4.5),
        constrained_layout=True,
    )
    for axis, metric in zip(axes, ("rmse", "pearson_r", "pred_std")):
        for scheme in SCHEMES:
            subset = eval_df.loc[eval_df["scheme"] == scheme]
            x = np.arange(3) + {"A": -0.25, "B": 0.0, "C": 0.25}[
                scheme
            ]
            base_values = subset[
                f"base_{metric}" if metric in {"rmse", "pearson_r"} else "base_pred_std"
            ]
            treat_values = subset[
                f"treat_{metric}" if metric in {"rmse", "pearson_r"} else "treat_pred_std"
            ]
            axis.plot(
                x,
                base_values,
                marker="o",
                linestyle="",
                label=f"{scheme} 基线",
            )
            axis.plot(
                x,
                treat_values,
                marker="s",
                linestyle="",
                label=f"{scheme} 日影",
            )
        axis.set_xticks(np.arange(len(eval_df.loc[eval_df["scheme"] == "A"])))
        axis.set_xticklabels(
            [
                "Ridge",
                "ElasticNet",
                "Linear",
            ]
        )
        axis.set_title(metric.replace("_", " ").title())
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    path = OUTPUT_DIR / "eval_comparison.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_gain_ci(gain_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        constrained_layout=True,
    )
    for axis, metric in zip(axes, ("rmse", "pearson_r")):
        if metric == "pearson_r":
            low = "pearson_ci_low"
            high = "pearson_ci_high"
            point = "pearson_gain"
        else:
            low = f"{metric}_ci_low"
            high = f"{metric}_ci_high"
            point = f"{metric}_gain"
        for scheme in SCHEMES:
            subset = gain_df.loc[gain_df["scheme"] == scheme]
            y = np.arange(len(subset))
            axis.errorbar(
                subset[point],
                y,
                xerr=[
                    subset[point] - subset[low],
                    subset[high] - subset[point],
                ],
                fmt="o",
                label=scheme,
            )
        axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_yticks(np.arange(len(gain_df.loc[gain_df["scheme"] == "A"])))
        axis.set_yticklabels(["Ridge", "ElasticNet", "Linear"])
        axis.set_xlabel(
            "Pearson r 增益（正值 = 加入日影后相关更高）"
            if metric == "pearson_r"
            else "RMSE 增益, km/s（正值 = 加入日影后误差更低）"
        )
        axis.set_title("95% CI" if metric == "rmse" else "Pearson 95% CI")
        axis.grid(True, alpha=0.3)
        axis.legend()
    path = OUTPUT_DIR / "gain_ci.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
