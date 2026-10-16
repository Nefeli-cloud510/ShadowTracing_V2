"""Compare v3 closed-loop models against explicit reference baselines.

The comparison is deliberately restrained:

- all arms use the frozen ``dev_backtest`` partition (test_start=2021-08-26);
- only samples where every comparison feature and the target are present on the
  real calendar are kept; missing values are never imputed;
- preprocessing (StandardScaler) is fitted inside each expanding-window fold and
  on the final training segment only;
- Ridge and ElasticNet use the same finite alpha / l1 grids and the same
  calendar CV folds, so a difference cannot be blamed on unequal tuning;
- the 35-day evaluation window is called 迭代评估段 because it has already been
  inspected repeatedly and is not a sealed independent final test.
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


def _family_config(family: str) -> ElasticNetConfig:
    return ElasticNetConfig(
        model_family=family,
        alpha=0.5,
        l1_ratio=0.5,
        random_state=42,
        alpha_grid=list(ALPHA_GRID),
        l1_ratio_grid=list(L1_GRID),
        param_search_mode="auto",
        max_param_search_combinations=MAX_COMBINATIONS,
        fit_intercept=True,
        ridge_solver="svd",
    )


def _pipeline_for(family: str, search) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                make_model(
                    _family_config(family),
                    alpha_override=search.selected_alpha,
                    l1_ratio_override=search.selected_l1_ratio,
                ),
            ),
        ]
    )


def _summary_row(
    run_id: str,
    benchmark_type: str,
    family: str | None,
    n_train: int,
    n_test: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    search=None,
) -> dict:
    row: dict = {
        "run_id": run_id,
        "benchmark_type": benchmark_type,
        "model_family": family or "reference",
        "n_train": int(n_train),
        "n_test": int(n_test),
        **{f"eval_{key}": value for key, value in _metrics(y_true, y_pred).items()},
    }
    if search is not None:
        row["selected_alpha"] = _float_or_none(search.selected_alpha)
        row["selected_l1_ratio"] = _float_or_none(search.selected_l1_ratio)
        row["search_completed"] = bool(search.search_completed)
        row["boundary_hit"] = bool(search.boundary_hit)
        row["solver_converged"] = search.solver_converged
        row["grid_size"] = int(search.grid_size)
    else:
        row["selected_alpha"] = None
        row["selected_l1_ratio"] = None
        row["search_completed"] = None
        row["boundary_hit"] = None
        row["solver_converged"] = None
        row["grid_size"] = None
    return row


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _record_snapshot(split, run_id: str, y_pred: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "run_id": run_id,
            "sample_id": list(split.sample_ids_test),
            "issue_date": pd.to_datetime(split.origin_test).strftime("%Y-%m-%d"),
            "target_date": pd.to_datetime(split.time_test).strftime("%Y-%m-%d"),
            "split": "dev_backtest_eval",
            "y_true": split.y_test,
            "y_pred": y_pred,
        }
    )
    return frame


def main() -> int:
    output_dir = ROOT / "artifacts" / "v3_baseline_reference"
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
        elasticnet_config=_family_config("ridge"),
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
                "基准对照的共同样本与 E_R02_03_r1 存档不一致: "
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

    # build_sample_plan 的实际窗口是 offset 0..past_lag_days-1；
    # offset 0 对应发报日当天，列名是 `<变量>_t`，样本与 E_R02_03_r1 一致。
    def _lag_column(variable: str, lag: int) -> str:
        return f"{variable}_t" if lag == 0 else f"{variable}_t-{lag}"

    target_features = {
        _lag_column(TARGET, lag): lag for lag in range(0, PAST_LAG_DAYS)
    }
    shadow_features = {
        _lag_column(SHADOW, lag): lag for lag in range(0, PAST_LAG_DAYS)
    }

    # Reference baselines need no parameter search.
    persisted_daily = pd.DataFrame()
    rows: list[dict] = []

    train_mean_pred = np.full_like(y_test, float(np.mean(y_train)))
    rows.append(
        _summary_row(
            "train_mean",
            "训练均值",
            None,
            len(y_train),
            len(y_test),
            y_test,
            train_mean_pred,
        )
    )
    persisted_daily = pd.concat(
        [persisted_daily, _record_snapshot(split, "train_mean", train_mean_pred)],
        ignore_index=True,
    )

    persistence_col = _lag_column(TARGET, 0)
    if persistence_col not in X_test.columns:
        raise KeyError(f"持久性基准缺少特征列: {persistence_col}")
    persistence_pred = np.asarray(X_test[persistence_col], dtype=float)
    rows.append(
        _summary_row(
            "persistence",
            "持续性（Vsw@issue_date）",
            None,
            len(y_train),
            len(y_test),
            y_test,
            persistence_pred,
        )
    )
    persisted_daily = pd.concat(
        [persisted_daily, _record_snapshot(split, "persistence", persistence_pred)],
        ignore_index=True,
    )

    for family in ("ridge", "elasticnet"):
        cfg = _family_config(family)
        for benchmark_type, feature_cols, suffix in (
            ("Vsw 历史", list(target_features), "vsw_history"),
            ("Vsw 历史＋日影", list(target_features) + list(shadow_features), "vsw_history_shadow"),
        ):
            missing = [column for column in feature_cols if column not in X_train.columns]
            if missing:
                raise KeyError(f"{family} {benchmark_type} 缺少特征列: {missing}")
            search = select_best_params(
                X_train[feature_cols],
                y_train,
                cfg,
                cv_folds=split.cv_folds,
            )
            pipeline = _pipeline_for(family, search)
            pipeline.fit(X_train[feature_cols], y_train)
            y_pred = np.asarray(pipeline.predict(X_test[feature_cols]), dtype=float)
            run_id = f"{family}_{suffix}"
            rows.append(
                _summary_row(
                    run_id,
                    benchmark_type,
                    family,
                    len(y_train),
                    len(y_test),
                    y_test,
                    y_pred,
                    search=search,
                )
            )
            persisted_daily = pd.concat(
                [persisted_daily, _record_snapshot(split, run_id, y_pred)],
                ignore_index=True,
            )

    summary = pd.DataFrame(rows)
    summary_csv = output_dir / "v3_baseline_reference_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    daily_csv = output_dir / "v3_baseline_reference_daily.csv"
    persisted_daily.to_csv(daily_csv, index=False, encoding="utf-8-sig")

    manifest = {
        "source_run": "E_R02_03_r1",
        "data_version": "shadow-calendar-v1",
        "split_label": "dev_backtest",
        "phase_label": "迭代评估段",
        "test_start": str(TEST_START.date()),
        "test_end": str(
            pd.to_datetime(split.time_test).max().normalize().date()
        ),
        "sample": {
            "retained_count": int(len(retained)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_internal_folds": 2,
        },
        "time_definition": (
            "issue_date=TIME；特征日期=issue_date-滞后天数；"
            f"目标日期=issue_date+{HORIZON_DAYS} 日历天"
        ),
        "baseline_definitions": {
            "train_mean": "训练段 y 均值常数预测",
            "persistence": "持续性预测：Vsw@issue_date（与系统发报时刻口径一致）",
            "vsw_history": (
                f"Vsw 可用历史（发报日及此前 {PAST_LAG_DAYS - 1} 天，"
                f"共 {PAST_LAG_DAYS} 个日历日）的线性模型"
            ),
            "vsw_history_shadow": (
                f"Vsw 可用历史 + 日影可用历史（各 {PAST_LAG_DAYS} 个日历日）"
                "的线性模型"
            ),
        },
        "notes": [
            "共同样本计划与 E_R02_03_r1 存档一致（177 完整样本，其中 139 训练 / 35 迭代评估 / 3 边界排除）。",
            "预处理只按各折与最终训练段拟合。",
            "Ridge 与 ElasticNet 使用相同 alpha/l1 网格和相同日历 CV 折。",
            "该 35 天区间已被反复检查，故只称迭代评估段，不声称独立盲测。",
        ],
        "summary": rows,
    }
    manifest_path = output_dir / "v3_baseline_reference.json"
    manifest_path.write_text(
        json_dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"baseline reference -> {output_dir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
