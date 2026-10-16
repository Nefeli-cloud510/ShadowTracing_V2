"""Standalone migration diagnostics: legacy ElasticNet settings on v3 calendar data.

This script intentionally does not modify any existing runner/script/state. It
re-implements the old feature-lag configuration from
``models/Elasticnet.txt`` and ``models/Elasticnet_timelag.txt`` on the current
v3 full-calendar CSVs, then reports a strictly calendar-aligned dev_backtest.

Terminology
-----------
- ``issue_date``: forecast issue day.
- ``target_date``: ``issue_date + horizon_days`` calendar days.
- v3 issue-relative lags: feature date = ``issue_date - lag``.

The old notebooks named lags relative to the target row; the old xlsx sources
are no longer present, so this migration uses the lag lists as issue-relative
offsets, which is the conservative no-future-leakage choice when the old file
does not preserve the original target-row mapping.

The 35-day evaluation segment is deliberately labeled 迭代评估段 (dev_backtest)
and is NOT presented as a sealed blind final test.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.calendar_samples import (
    CalendarSample,
    FeatureSpec,
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


OUTPUT_DIR = ROOT / "artifacts" / "legacy_v3_migration"

SESSIONS_ROOT = ROOT / "runtime" / "live_session" / "sessions"
TARGET = "SW Plasma Speed, km/s"
TIME = "TIME"

TEST_START = pd.Timestamp("2021-08-26")
TEST_END = pd.Timestamp("2021-09-29")
TEST_RATIO = 0.2
N_INTERNAL_FOLDS = 2

N_BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261017
BLOCK_SIZE = 7
MIN_BLOCKS = 2

ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
L1_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
MAX_COMBINATIONS = 30

# Variable, issue-relative lag pairs from ``Elasticnet_timelag.txt``.
LEGACY_TIMELAG_MAG = [
    ("BY, nT (GSE)", 4),
    ("BY, nT (GSE)", 5),
    ("BY, nT (GSE)", 7),
    ("BY, nT (GSE)", 8),
    ("BZ, nT (GSE)", 1),
    ("BZ, nT (GSE)", 2),
    ("Scalar B, nT", 1),
    ("Scalar B, nT", 2),
]

LEGACY_TIMELAG_VARIANTS = {
    "mag": list(LEGACY_TIMELAG_MAG),
    "mag_ns": LEGACY_TIMELAG_MAG + [("p2_y中心值", 1)],
    "mag_ew": LEGACY_TIMELAG_MAG + [("p1_x中心值", 9)],
    "mag_both": LEGACY_TIMELAG_MAG + [("p2_y中心值", 1), ("p1_x中心值", 9)],
}

LEGACY_TIMELAG_PARAMS = {"alpha": 1.0, "l1_ratio": 0.9}

# Scalar B + BY from ``Elasticnet.txt``.  The old sliding-window notebook
# predicted the row after the last feature day; for a one-day-ahead forecast
# old t-1/t-2 map to issue offsets 0/-1.
LEGACY_V2_OFFSETS = [
    ("Scalar B, nT", 0),
    ("Scalar B, nT", -1),
    ("BY, nT (GSE)", 0),
    ("BY, nT (GSE)", -1),
]
LEGACY_V2_PARAMS = {"alpha": 0.3, "l1_ratio": 0.6}

VARIANT_LABELS = {
    "mag": "BY+BZ+|B| 多滞后(不带日影)",
    "mag_ns": "磁场＋p2_y中心值(南北日影)",
    "mag_ew": "磁场＋p1_x中心值(东西日影)",
    "mag_both": "磁场＋两个方向日影",
    "v2_mag": "旧 Scalar B+BY 双日窗(不带日影)",
}

plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


def _find_latest_v3_session() -> Path:
    candidates = sorted(SESSIONS_ROOT.glob("session_*"), reverse=True)
    for candidate in candidates:
        raw_dir = candidate / "data" / "raw"
        omni = raw_dir / "omni_2021_full_calendar_v3.csv"
        lhaaso = raw_dir / "lhaaso_shadow_pfss_2021_full_calendar_v3.csv"
        if omni.exists() and lhaaso.exists():
            return candidate
    raise FileNotFoundError("未找到包含 v3 全日历 CSV 的 live session")


def _specs_from_pairs(pairs: list[tuple[str, int]]) -> tuple[FeatureSpec, ...]:
    specs: list[FeatureSpec] = []
    for variable, lag in pairs:
        specs.append(FeatureSpec(variable=variable, offset_days=-int(lag)))
    return tuple(specs)


def _load_merged() -> tuple[pd.DataFrame, dict]:
    session_root = _find_latest_v3_session()
    raw_dir = session_root / "data" / "raw"
    timelag_variables = {
        variable
        for pairs in LEGACY_TIMELAG_VARIANTS.values()
        for variable, _lag in pairs
    }
    v2_variables = {variable for variable, _offset in LEGACY_V2_OFFSETS}
    all_features = sorted(timelag_variables | v2_variables)
    cfg = ExperimentConfig(
        file_config=FileConfig(
            omni_file_path=raw_dir / "omni_2021_full_calendar_v3.csv",
            lhaaso_file_path=raw_dir / "lhaaso_shadow_pfss_2021_full_calendar_v3.csv",
            time_column=TIME,
            target_column=TARGET,
            feature_columns=all_features,
            sparse_sources=["omni_2021_full_calendar_v3"],
        ),
        time_window_config=TimeWindowConfig(
            window_size=9,
            test_split_ratio=TEST_RATIO,
            use_lag_feature=True,
            max_lag_day=9,
            forecast_horizon_days=1,
            past_lag_days=9,
        ),
        elasticnet_config=ElasticNetConfig(
            model_family="elasticnet",
            alpha=1.0,
            l1_ratio=0.9,
            random_state=42,
            param_search_mode="fixed",
        ),
    )
    merged = load_and_merge_data(cfg)
    frame = merged.X.copy()
    frame[TIME] = pd.to_datetime(merged.time, errors="coerce").dt.normalize()
    frame[TARGET] = pd.to_numeric(merged.y, errors="coerce")
    audit = {
        "session_root": str(session_root),
        "coverage": [report.to_dict() for report in merged.coverage],
        "missing_counts": merged.missing_counts,
        "data_validation": (
            merged.data_validation.to_dict()
            if hasattr(merged.data_validation, "to_dict")
            else None
        ),
        "source_columns": list(frame.columns),
        "row_count": int(len(frame)),
        "date_start": pd.to_datetime(frame[TIME]).min().date().isoformat(),
        "date_end": pd.to_datetime(frame[TIME]).max().date().isoformat(),
        "duplicate_dates": int(
            pd.to_datetime(frame[TIME]).dt.normalize().duplicated().sum()
        ),
    }
    return frame, audit


def _build_records_for_specs(
    frame: pd.DataFrame,
    specs: tuple[FeatureSpec, ...],
    *,
    horizon_days: int,
) -> tuple[list[CalendarSample], dict[str, int]]:
    date_to_index = {
        pd.Timestamp(value).normalize(): index
        for index, value in enumerate(frame[TIME])
    }
    all_dates = sorted(date_to_index)
    records: list[CalendarSample] = []
    field_drop_counts: dict[str, int] = {}
    for issue_value in all_dates:
        issue = pd.Timestamp(issue_value).normalize()
        feature_dates = tuple(
            issue + pd.Timedelta(days=int(spec.offset_days)) for spec in specs
        )
        target_date = issue + pd.Timedelta(days=int(horizon_days))
        sample_id = f"{issue.date().isoformat()}__h{horizon_days}"
        common = {
            "sample_id": sample_id,
            "issue_date": issue,
            "target_date": target_date,
            "horizon_days": int(horizon_days),
            "feature_specs": specs,
            "feature_dates": feature_dates,
        }

        if any(
            pd.Timestamp(value).normalize() not in date_to_index
            for value in feature_dates
        ):
            records.append(
                CalendarSample(
                    **common,
                    sample_condition="incomplete_case",
                    drop_reason="history_gap",
                )
            )
            continue

        feature_rows = [
            date_to_index[pd.Timestamp(value).normalize()]
            for value in feature_dates
        ]
        missing = False
        for spec, row_index in zip(specs, feature_rows):
            value = frame.iloc[row_index][spec.variable]
            if pd.isna(value) or not np.isfinite(float(value)):
                field_drop_counts[spec.variable] = (
                    field_drop_counts.get(spec.variable, 0) + 1
                )
                missing = True
        if missing:
            records.append(
                CalendarSample(
                    **common,
                    sample_condition="incomplete_case",
                    drop_reason="feature_missing",
                )
            )
            continue

        if pd.Timestamp(target_date).normalize() not in date_to_index:
            records.append(
                CalendarSample(
                    **common,
                    sample_condition="incomplete_case",
                    drop_reason="target_unavailable",
                )
            )
            continue
        target_row = date_to_index[pd.Timestamp(target_date).normalize()]
        target_value = frame.iloc[target_row][TARGET]
        if pd.isna(target_value) or not np.isfinite(float(target_value)):
            field_drop_counts[TARGET] = field_drop_counts.get(TARGET, 0) + 1
            records.append(
                CalendarSample(
                    **common,
                    sample_condition="incomplete_case",
                    drop_reason="target_unavailable",
                )
            )
            continue
        records.append(CalendarSample(**common, sample_condition="complete_case"))

    return records, field_drop_counts


def _extract_feature_matrix(
    frame: pd.DataFrame,
    records: list[CalendarSample],
    specs: tuple[FeatureSpec, ...],
    *,
    horizon_days: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    date_to_index = {
        pd.Timestamp(value).normalize(): index
        for index, value in enumerate(frame[TIME])
    }
    feature_names = [spec.feature_name() for spec in specs]
    rows: list[list[float]] = []
    y_values: list[float] = []
    issue_dates: list[pd.Timestamp] = []
    target_dates: list[pd.Timestamp] = []
    sample_ids: list[str] = []
    for record in records:
        values: list[float] = []
        for spec in record.feature_specs:
            feature_date = pd.Timestamp(
                record.issue_date + pd.Timedelta(days=int(spec.offset_days))
            ).normalize()
            row_index = date_to_index[feature_date]
            values.append(float(frame.iloc[row_index][spec.variable]))
        target_index = date_to_index[pd.Timestamp(record.target_date).normalize()]
        y_values.append(float(frame.iloc[target_index][TARGET]))
        rows.append(values)
        issue_dates.append(record.issue_date)
        target_dates.append(record.target_date)
        sample_ids.append(record.sample_id)
    X = pd.DataFrame(np.asarray(rows, dtype=float), columns=feature_names)
    y = np.asarray(y_values, dtype=float)
    meta = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "issue_date": pd.Series(issue_dates, dtype="datetime64[ns]"),
            "target_date": pd.Series(target_dates, dtype="datetime64[ns]"),
            "horizon_days": int(horizon_days),
        }
    )
    return X, y, meta


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2_raw = float(r2_score(y_true, y_pred))
    r2: float | None = r2_raw if math.isfinite(r2_raw) else None
    pearson_r: float | None = None
    unavailable_reason: str | None = None
    if len(y_true) > 1:
        if float(np.std(y_true)) <= 1e-12:
            unavailable_reason = "constant_target"
        elif float(np.std(y_pred)) <= 1e-12:
            unavailable_reason = "constant_prediction"
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                raw = float(np.corrcoef(y_true, y_pred)[0, 1])
            if math.isfinite(raw):
                pearson_r = raw
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
        "true_std": float(np.std(y_true)),
    }


def _config(
    family: str,
    *,
    alpha: float,
    l1_ratio: float,
    mode: str,
) -> ElasticNetConfig:
    return ElasticNetConfig(
        model_family=family,
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=42,
        alpha_grid=list(ALPHA_GRID),
        l1_ratio_grid=list(L1_GRID),
        param_search_mode=mode,
        max_param_search_combinations=MAX_COMBINATIONS,
        fit_intercept=True,
        ridge_solver="svd",
    )


def _run_model(
    family: str,
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    cfg: ElasticNetConfig,
    cv_folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[Pipeline, object, np.ndarray, np.ndarray]:
    search = select_best_params(X_train, y_train, cfg, cv_folds=cv_folds)
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
    y_pred = np.asarray(pipeline.predict(X_test), dtype=float)
    y_pred_train = np.asarray(pipeline.predict(X_train), dtype=float)
    return pipeline, search, y_pred, y_pred_train


def _summary_row(
    run_id: str,
    group: str,
    variant: str,
    family: str,
    mode: str,
    *,
    horizon_days: int,
    n_train: int,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_train: np.ndarray | None,
    spec_label: str,
    test_start: str,
    test_end: str,
    search=None,
) -> dict:
    metrics = _metrics(y_test, y_pred)
    row: dict = {
        "run_id": run_id,
        "group": group,
        "variant": variant,
        "model_family": family,
        "mode": mode,
        "horizon_days": int(horizon_days),
        "feature_spec_label": spec_label,
        "n_train": int(n_train),
        "n_test": int(len(y_test)),
        "eval_date_start": test_start,
        "eval_date_end": test_end,
        **{f"eval_{key}": value for key, value in metrics.items()},
    }
    if y_pred_train is not None:
        train_metrics = _metrics(y_train, y_pred_train)
        row.update(
            {
                **{f"train_{key}": value for key, value in train_metrics.items()},
            }
        )
    else:
        row.update(
            {
                "train_rmse": None,
                "train_mae": None,
                "train_r2": None,
                "train_pearson_r": None,
                "train_unavailable_reason": None,
                "train_prediction_std": None,
                "train_true_std": None,
            }
        )
    if search is not None:
        row.update(
            {
                "selected_alpha": float(search.selected_alpha),
                "selected_l1_ratio": (
                    float(search.selected_l1_ratio)
                    if search.selected_l1_ratio is not None
                    else None
                ),
                "search_mode": search.search_mode,
                "search_completed": bool(search.search_completed),
                "boundary_hit": bool(search.boundary_hit),
                "solver_converged": bool(search.solver_converged),
                "grid_size": int(search.grid_size),
            }
        )
    return row


def _coefficient_row(
    run_id: str,
    group: str,
    variant: str,
    family: str,
    mode: str,
    *,
    horizon_days: int,
    pipeline: Pipeline,
    feature_cols: list[str],
    alpha: float,
    l1_ratio: float | None,
) -> dict:
    model = pipeline.named_steps["model"]
    coef = np.asarray(model.coef_, dtype=float).reshape(-1)
    return {
        "run_id": run_id,
        "group": group,
        "variant": variant,
        "model_family": family,
        "mode": mode,
        "horizon_days": int(horizon_days),
        "alpha": float(alpha),
        "l1_ratio": None if l1_ratio is None else float(l1_ratio),
        "coefficients_json": json.dumps(
            {
                column: float(coef[index])
                for index, column in enumerate(feature_cols)
            },
            ensure_ascii=False,
        ),
        "feature_count": int(len(feature_cols)),
        "intercept": float(getattr(model, "intercept_", float("nan"))),
    }


def _filter_split_test(
    split,
    *,
    horizon_days: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    origins = pd.to_datetime(split.origin_test, errors="coerce")
    keep = np.asarray(origins <= TEST_END)
    meta = pd.DataFrame(
        {
            "sample_id": np.asarray(split.sample_ids_test)[keep],
            "issue_date": origins[keep].to_numpy(dtype="datetime64[ns]"),
            "target_date": pd.to_datetime(split.time_test, errors="coerce")[
                keep
            ].to_numpy(dtype="datetime64[ns]"),
            "horizon_days": int(horizon_days),
        }
    )
    y_test = np.asarray(split.y_test, dtype=float)[keep]
    X_test = split.X_test.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    return X_test, y_test, meta


def _daily_frame(
    run_id: str,
    group: str,
    variant: str,
    family: str,
    mode: str,
    *,
    horizon_days: int,
    meta_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": run_id,
            "group": group,
            "variant": variant,
            "model_family": family,
            "mode": mode,
            "horizon_days": int(horizon_days),
            "sample_id": meta_test["sample_id"].to_numpy(),
            "issue_date": pd.to_datetime(meta_test["issue_date"]).dt.strftime(
                "%Y-%m-%d"
            ),
            "target_date": pd.to_datetime(meta_test["target_date"]).dt.strftime(
                "%Y-%m-%d"
            ),
            "y_true": np.asarray(y_test, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )


def _retained_keys(records: list[CalendarSample]) -> set[tuple[str, str, int]]:
    return {
        record.compact_key
        for record in records
        if record.drop_reason is None
    }


def _drop_summary(records: list[CalendarSample]) -> dict[str, int]:
    counts: Counter[str] = Counter(
        record.drop_reason or "retained" for record in records
    )
    return dict(sorted(counts.items()))


def _gain_interpretation(boot: dict) -> str:
    if bool(boot.get("insufficient_blocks")):
        return "有效时间块不足，未作方向性稳定判断"
    ci = boot.get("rmse_ci") or [float("nan"), float("nan")]
    if not math.isfinite(float(ci[0])) or not math.isfinite(float(ci[1])):
        return "RMSE 增益区间不可用"
    lower, upper = float(ci[0]), float(ci[1])
    if lower > 0:
        return "本样本范围内 RMSE 增益区间支持改善"
    if upper < 0:
        return "本样本范围内 RMSE 增益区间支持退化"
    return "RMSE 增益区间跨零，未检出稳定增益"


def _references_for_horizon(
    horizon_days: int,
    *,
    frame: pd.DataFrame,
    n_train: int,
    y_train: np.ndarray,
    y_test: np.ndarray,
    meta_test: pd.DataFrame,
    summary_rows: list[dict],
    daily_dfs: list[pd.DataFrame],
) -> None:
    test_start = str(pd.to_datetime(meta_test["issue_date"]).min().date())
    test_end = str(pd.to_datetime(meta_test["issue_date"]).max().date())

    train_mean = np.full_like(y_test, float(np.mean(y_train)))
    train_mean_run = f"train_mean_h{horizon_days}"
    summary_rows.append(
        _summary_row(
            train_mean_run,
            "reference",
            "train_mean",
            "none",
            "reference",
            horizon_days=horizon_days,
            n_train=int(n_train),
            y_train=y_train,
            y_test=y_test,
            y_pred=train_mean,
            y_pred_train=None,
            spec_label="训练目标均值",
            test_start=test_start,
            test_end=test_end,
        )
    )
    daily_dfs.append(
        _daily_frame(
            train_mean_run,
            "reference",
            "train_mean",
            "none",
            "reference",
            horizon_days=horizon_days,
            meta_test=meta_test,
            y_test=y_test,
            y_pred=train_mean,
        )
    )

    issue_index = {
        pd.Timestamp(value).normalize(): index
        for index, value in enumerate(frame[TIME])
    }
    persistence = np.asarray(
        [
            float(
                frame.iloc[
                    issue_index[pd.Timestamp(value).normalize()]
                ][TARGET]
            )
            for value in meta_test["issue_date"]
        ],
        dtype=float,
    )
    pers_run = f"persistence_h{horizon_days}"
    summary_rows.append(
        _summary_row(
            pers_run,
            "reference",
            "persistence",
            "none",
            "reference",
            horizon_days=horizon_days,
            n_train=int(n_train),
            y_train=y_train,
            y_test=y_test,
            y_pred=persistence,
            y_pred_train=None,
            spec_label="发报日最新可用 Vsw(持续性)",
            test_start=test_start,
            test_end=test_end,
        )
    )
    daily_dfs.append(
        _daily_frame(
            pers_run,
            "reference",
            "persistence",
            "none",
            "reference",
            horizon_days=horizon_days,
            meta_test=meta_test,
            y_test=y_test,
            y_pred=persistence,
        )
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, audit = _load_merged()
    audit["generated_at"] = datetime.now().isoformat()

    summary_rows: list[dict] = []
    daily_dfs: list[pd.DataFrame] = []
    coef_rows: list[dict] = []
    gain_rows: list[dict] = []
    sample_audit_rows: list[dict] = []
    daily_by_run: dict[str, pd.DataFrame] = {}

    for horizon_days in (1, 3):
        variant_plans: dict[str, dict] = {}
        for variant, pairs in LEGACY_TIMELAG_VARIANTS.items():
            specs = _specs_from_pairs(pairs)
            records, drops = _build_records_for_specs(
                frame, specs, horizon_days=horizon_days
            )
            variant_plans[variant] = {
                "specs": specs,
                "records": records,
                "drops": drops,
                "label": VARIANT_LABELS[variant],
            }

        common_key_sets = [
            _retained_keys(plan["records"]) for plan in variant_plans.values()
        ]
        common_keys = set.intersection(*common_key_sets)
        filtered_records = {
            variant: [
                record
                for record in plan["records"]
                if record.compact_key in common_keys
            ]
            for variant, plan in variant_plans.items()
        }

        calendar_split = make_calendar_split(
            filtered_records["mag"],
            test_ratio=TEST_RATIO,
            n_internal_folds=N_INTERNAL_FOLDS,
            test_start=TEST_START,
        )

        prepared_variants: dict[str, dict] = {}
        for variant, plan in variant_plans.items():
            records = filtered_records[variant]
            X_all, y_all, meta_all = _extract_feature_matrix(
                frame,
                records,
                plan["specs"],
                horizon_days=horizon_days,
            )
            split = split_dataset_by_calendar(
                X_all.to_numpy(dtype=float),
                y_all,
                meta_all["target_date"].to_numpy(dtype="datetime64[ns]"),
                meta_all["issue_date"].to_numpy(dtype="datetime64[ns]"),
                meta_all["sample_id"].to_list(),
                records,
                test_rate=TEST_RATIO,
                feature_names=list(X_all.columns),
                n_internal_folds=N_INTERNAL_FOLDS,
                calendar_split=calendar_split,
            )
            X_test, y_test, meta_test = _filter_split_test(
                split, horizon_days=horizon_days
            )
            prepared_variants[variant] = {
                "split": split,
                "X_test": X_test,
                "y_test": y_test,
                "meta_test": meta_test,
                "records": records,
                "specs": plan["specs"],
                "label": plan["label"],
                "drops": plan["drops"],
                "X_all": X_all,
            }

        for variant, plan in variant_plans.items():
            records = plan["records"]
            common_count = len(filtered_records[variant])
            prepared = prepared_variants[variant]
            split = prepared["split"]
            drop_counts = _drop_summary(records)
            sample_audit_rows.append(
                {
                    "horizon_days": int(horizon_days),
                    "variant": variant,
                    "feature_spec_label": plan["label"],
                    "candidate_count": int(len(records)),
                    "retained_before_common": int(len(_retained_keys(records))),
                    "retained_after_common": int(common_count),
                    "n_train": int(len(split.y_train)),
                    "n_test_full_split": int(len(split.y_test)),
                    "n_test_dev_window": int(len(prepared["y_test"])),
                    "drop_reason_counts_json": json.dumps(
                        drop_counts, ensure_ascii=False
                    ),
                    "field_drop_counts_json": json.dumps(
                        plan["drops"], ensure_ascii=False
                    ),
                }
            )

        mag_split = prepared_variants["mag"]["split"]
        mag_meta = prepared_variants["mag"]["meta_test"]
        _references_for_horizon(
            horizon_days,
            frame=frame,
            n_train=len(mag_split.y_train),
            y_train=mag_split.y_train,
            y_test=prepared_variants["mag"]["y_test"],
            meta_test=mag_meta,
            summary_rows=summary_rows,
            daily_dfs=daily_dfs,
        )

        for variant in ("mag", "mag_ns", "mag_ew", "mag_both"):
            prepared = prepared_variants[variant]
            split = prepared["split"]
            X_test = prepared["X_test"]
            y_test = prepared["y_test"]
            meta_test = prepared["meta_test"]
            for mode in ("fixed", "auto"):
                if mode == "fixed":
                    cfg = _config(
                        "elasticnet",
                        alpha=LEGACY_TIMELAG_PARAMS["alpha"],
                        l1_ratio=LEGACY_TIMELAG_PARAMS["l1_ratio"],
                        mode="fixed",
                    )
                else:
                    cfg = _config(
                        "elasticnet",
                        alpha=LEGACY_TIMELAG_PARAMS["alpha"],
                        l1_ratio=LEGACY_TIMELAG_PARAMS["l1_ratio"],
                        mode="auto",
                    )
                run_id = f"{variant}_{mode}_h{horizon_days}"
                pipeline, search, y_pred, y_pred_train = _run_model(
                    "elasticnet",
                    X_train=split.X_train,
                    y_train=split.y_train,
                    X_test=X_test,
                    cfg=cfg,
                    cv_folds=split.cv_folds,
                )
                date_start = str(pd.to_datetime(meta_test["issue_date"]).min().date())
                date_end = str(pd.to_datetime(meta_test["issue_date"]).max().date())
                summary_rows.append(
                    _summary_row(
                        run_id,
                        "model",
                        variant,
                        "elasticnet",
                        mode,
                        horizon_days=horizon_days,
                        n_train=len(split.y_train),
                        y_train=split.y_train,
                        y_test=y_test,
                        y_pred=y_pred,
                        y_pred_train=y_pred_train,
                        spec_label=prepared["label"],
                        test_start=date_start,
                        test_end=date_end,
                        search=search,
                    )
                )
                daily = _daily_frame(
                    run_id,
                    "model",
                    variant,
                    "elasticnet",
                    mode,
                    horizon_days=horizon_days,
                    meta_test=meta_test,
                    y_test=y_test,
                    y_pred=y_pred,
                )
                daily_dfs.append(daily)
                daily_by_run[run_id] = daily
                coef_rows.append(
                    _coefficient_row(
                        run_id,
                        "model",
                        variant,
                        "elasticnet",
                        mode,
                        horizon_days=horizon_days,
                        pipeline=pipeline,
                        feature_cols=list(X_test.columns),
                        alpha=search.selected_alpha,
                        l1_ratio=search.selected_l1_ratio,
                    )
                )

        # Auxiliary old Elasticnet.txt window setting, migrated to issue offsets.
        v2_specs = tuple(
            FeatureSpec(variable=variable, offset_days=int(offset))
            for variable, offset in LEGACY_V2_OFFSETS
        )
        v2_records, v2_drops = _build_records_for_specs(
            frame, v2_specs, horizon_days=horizon_days
        )
        retained_v2 = [r for r in v2_records if r.drop_reason is None]
        if len(retained_v2) > 3:
            v2_split = make_calendar_split(
                retained_v2,
                test_ratio=TEST_RATIO,
                n_internal_folds=N_INTERNAL_FOLDS,
                test_start=TEST_START,
            )
            X_v2, y_v2, meta_v2 = _extract_feature_matrix(
                frame, retained_v2, v2_specs, horizon_days=horizon_days
            )
            split_v2 = split_dataset_by_calendar(
                X_v2.to_numpy(dtype=float),
                y_v2,
                meta_v2["target_date"].to_numpy(dtype="datetime64[ns]"),
                meta_v2["issue_date"].to_numpy(dtype="datetime64[ns]"),
                meta_v2["sample_id"].to_list(),
                retained_v2,
                test_rate=TEST_RATIO,
                feature_names=list(X_v2.columns),
                n_internal_folds=N_INTERNAL_FOLDS,
                calendar_split=v2_split,
            )
            X_test_v2, y_test_v2, meta_test_v2 = _filter_split_test(
                split_v2, horizon_days=horizon_days
            )
            sample_audit_rows.append(
                {
                    "horizon_days": int(horizon_days),
                    "variant": "v2_mag",
                    "feature_spec_label": VARIANT_LABELS["v2_mag"],
                    "candidate_count": int(len(v2_records)),
                    "retained_before_common": int(len(retained_v2)),
                    "retained_after_common": int(len(retained_v2)),
                    "n_train": int(len(split_v2.y_train)),
                    "n_test_full_split": int(len(split_v2.y_test)),
                    "n_test_dev_window": int(len(y_test_v2)),
                    "drop_reason_counts_json": json.dumps(
                        _drop_summary(v2_records), ensure_ascii=False
                    ),
                    "field_drop_counts_json": json.dumps(
                        v2_drops, ensure_ascii=False
                    ),
                }
            )
            for mode in ("fixed", "auto"):
                if mode == "fixed":
                    cfg_v2 = _config(
                        "elasticnet",
                        alpha=LEGACY_V2_PARAMS["alpha"],
                        l1_ratio=LEGACY_V2_PARAMS["l1_ratio"],
                        mode="fixed",
                    )
                else:
                    cfg_v2 = _config(
                        "elasticnet",
                        alpha=LEGACY_V2_PARAMS["alpha"],
                        l1_ratio=LEGACY_V2_PARAMS["l1_ratio"],
                        mode="auto",
                    )
                run_id = f"v2_mag_{mode}_h{horizon_days}"
                pipeline, search, y_pred, y_pred_train = _run_model(
                    "elasticnet",
                    X_train=split_v2.X_train,
                    y_train=split_v2.y_train,
                    X_test=X_test_v2,
                    cfg=cfg_v2,
                    cv_folds=split_v2.cv_folds,
                )
                date_start = str(pd.to_datetime(meta_test_v2["issue_date"]).min().date())
                date_end = str(pd.to_datetime(meta_test_v2["issue_date"]).max().date())
                summary_rows.append(
                    _summary_row(
                        run_id,
                        "model",
                        "v2_mag",
                        "elasticnet",
                        mode,
                        horizon_days=horizon_days,
                        n_train=len(split_v2.y_train),
                        y_train=split_v2.y_train,
                        y_test=y_test_v2,
                        y_pred=y_pred,
                        y_pred_train=y_pred_train,
                        spec_label=VARIANT_LABELS["v2_mag"],
                        test_start=date_start,
                        test_end=date_end,
                        search=search,
                    )
                )
                daily = _daily_frame(
                    run_id,
                    "model",
                    "v2_mag",
                    "elasticnet",
                    mode,
                    horizon_days=horizon_days,
                    meta_test=meta_test_v2,
                    y_test=y_test_v2,
                    y_pred=y_pred,
                )
                daily_dfs.append(daily)
                daily_by_run[run_id] = daily
                coef_rows.append(
                    _coefficient_row(
                        run_id,
                        "model",
                        "v2_mag",
                        "elasticnet",
                        mode,
                        horizon_days=horizon_days,
                        pipeline=pipeline,
                        feature_cols=list(X_v2.columns),
                        alpha=search.selected_alpha,
                        l1_ratio=search.selected_l1_ratio,
                    )
                )

    for horizon_days in (1, 3):
        for mode in ("fixed", "auto"):
            base_run = f"mag_{mode}_h{horizon_days}"
            if base_run not in daily_by_run:
                continue
            for treatment_variant in ("mag_ns", "mag_ew", "mag_both"):
                treat_run = f"{treatment_variant}_{mode}_h{horizon_days}"
                baseline_df = daily_by_run[base_run].copy()
                treatment_df = daily_by_run[treat_run].copy()
                boot = paired_calendar_block_bootstrap(
                    baseline_df,
                    treatment_df,
                    block_size=BLOCK_SIZE,
                    n_draws=N_BOOTSTRAP_DRAWS,
                    seed=BOOTSTRAP_SEED,
                    min_blocks=MIN_BLOCKS,
                    verify_pearson=True,
                )
                gain_rows.append(
                    {
                        "horizon_days": int(horizon_days),
                        "mode": mode,
                        "baseline_variant": "mag",
                        "treatment_variant": treatment_variant,
                        "baseline_run_id": base_run,
                        "treatment_run_id": treat_run,
                        "n_paired_samples": int(boot["n_test_samples"]),
                        "point_gain_rmse": boot["point_gain_rmse"],
                        "rmse_ci_low": boot["rmse_ci"][0],
                        "rmse_ci_high": boot["rmse_ci"][1],
                        "point_gain_mae": boot["point_gain_mae"],
                        "mae_ci_low": boot["mae_ci"][0],
                        "mae_ci_high": boot["mae_ci"][1],
                        "point_gain_pearson_r": boot["point_gain_pearson_r"],
                        "pearson_r_ci_low": boot["pearson_r_ci"][0],
                        "pearson_r_ci_high": boot["pearson_r_ci"][1],
                        "n_nonempty_blocks": int(boot["n_nonempty_blocks"]),
                        "min_blocks": int(boot["min_blocks"]),
                        "insufficient_blocks": bool(
                            boot["insufficient_blocks"]
                        ),
                        "stable_rmse": bool(boot["stable_rmse"]),
                        "n_valid_pearson_draws": int(
                            boot["n_valid_pearson_draws"]
                        ),
                        "interpretation": _gain_interpretation(boot),
                    }
                )

    summary_df = (
        pd.DataFrame(summary_rows)
        .replace([np.inf, -np.inf], np.nan)
        .sort_values(
            ["horizon_days", "group", "variant", "mode"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    daily_df = (
        pd.concat(daily_dfs, ignore_index=True)
        .sort_values(["horizon_days", "run_id", "issue_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    coef_df = (
        pd.DataFrame(coef_rows)
        .replace([np.inf, -np.inf], np.nan)
        .sort_values(["horizon_days", "run_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    gain_df = (
        pd.DataFrame(gain_rows)
        .replace([np.inf, -np.inf], np.nan)
        .sort_values(["horizon_days", "mode", "treatment_variant"])
        .reset_index(drop=True)
    )
    sample_df = (
        pd.DataFrame(sample_audit_rows)
        .sort_values(["horizon_days", "variant"])
        .reset_index(drop=True)
    )

    summary_path = OUTPUT_DIR / "legacy_v3_migration_summary.csv"
    daily_path = OUTPUT_DIR / "legacy_v3_migration_daily.csv"
    coef_path = OUTPUT_DIR / "legacy_v3_migration_coefficients.csv"
    gain_path = OUTPUT_DIR / "legacy_v3_migration_gain_comparison.csv"
    sample_path = OUTPUT_DIR / "legacy_v3_migration_sample_audit.csv"
    manifest_path = OUTPUT_DIR / "legacy_v3_migration_manifest.json"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    gain_df.to_csv(gain_path, index=False, encoding="utf-8-sig")
    sample_df.to_csv(sample_path, index=False, encoding="utf-8-sig")
    coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")

    _write_plots(summary_df, gain_df, daily_df)

    manifest = {
        "schema_version": "legacy-v3-migration-v1",
        "label": "dev_backtest",
        "generated_at": audit["generated_at"],
        "scope": "探索性迁移诊断；不修改闭环状态、不批准协议、不写回假设支持度",
        "data": audit,
        "freeze": {
            "split_label": "dev_backtest",
            "test_start_issue_date": TEST_START.date().isoformat(),
            "test_end_issue_date": TEST_END.date().isoformat(),
            "test_ratio": TEST_RATIO,
            "n_internal_folds": N_INTERNAL_FOLDS,
            "horizon_days": [1, 3],
            "features": {
                "timelag_mag": [
                    [variable, lag]
                    for variable, lag in LEGACY_TIMELAG_MAG
                ],
                "timelag_mag_ns": [
                    [variable, lag]
                    for variable, lag in LEGACY_TIMELAG_VARIANTS["mag_ns"]
                ],
                "timelag_mag_ew": [
                    [variable, lag]
                    for variable, lag in LEGACY_TIMELAG_VARIANTS["mag_ew"]
                ],
                "timelag_mag_both": [
                    [variable, lag]
                    for variable, lag in LEGACY_TIMELAG_VARIANTS["mag_both"]
                ],
                "v2_scalar_b_by_offsets": list(LEGACY_V2_OFFSETS),
            },
            "fixed_params": {
                "timelag": LEGACY_TIMELAG_PARAMS,
                "v2": LEGACY_V2_PARAMS,
            },
            "selected_params": {
                "alpha_grid": ALPHA_GRID,
                "l1_ratio_grid": L1_GRID,
                "max_combinations": MAX_COMBINATIONS,
            },
            "bootstrap": {
                "block_size": BLOCK_SIZE,
                "n_draws": N_BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "min_blocks": MIN_BLOCKS,
            },
        },
        "mapping_notes": [
            "旧 Elasticnet_timelag.txt 的 lag_list 在 v3 中按发报日相对滞后解释："
            "feature_date = issue_date - lag，保证预测时点不使用未来数据。",
            "旧 Elasticnet.txt 的 window_size=2 对应旧 notebook 的 t-1/t-2；"
            "v3 迁移为发报日 offset 0/-1（即发报日与前一日历日）。",
            "所有丢弃均保留真实日历原因，不插值、不重排、不压缩缺失日期。",
        ],
        "limits": [
            "35 天评估段是迭代评估段 dev_backtest，不是未查看的独立盲测。",
            "旧 xlsx 源文件不存在，无法逐条复现旧数据；本结果只证明旧参数/特征在新 v3 CSV 上的日历对齐口径。",
            "探索性结果不写回假设树、记忆或前端闭环状态。",
        ],
        "output_files": {
            "summary": str(summary_path),
            "daily_predictions": str(daily_path),
            "coefficients": str(coef_path),
            "gain_comparison": str(gain_path),
            "sample_audit": str(sample_path),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(
        json_dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"legacy v3 migration diagnostics written -> {OUTPUT_DIR}")
    print(json_dumps(
        {
            "n_summary_rows": len(summary_df),
            "n_daily_rows": len(daily_df),
            "n_gain_comparisons": len(gain_df),
            "files": list(manifest["output_files"].values()),
        },
        indent=2,
    ))
    return 0


def _write_plots(
    summary_df: pd.DataFrame,
    gain_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> None:
    try:
        fixed_gain = gain_df[gain_df["mode"] == "fixed"].copy()
        if not fixed_gain.empty:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(12, 4.6),
                constrained_layout=True,
                sharey=True,
            )
            for ax, horizon in zip(axes, (1, 3)):
                sub = fixed_gain[fixed_gain["horizon_days"] == horizon]
                labels = {
                    "mag_ns": "N-S 日影",
                    "mag_ew": "E-W 日影",
                    "mag_both": "双方向日影",
                }
                y_pos = np.arange(len(sub))
                centers = pd.to_numeric(sub["point_gain_rmse"], errors="coerce")
                lows = pd.to_numeric(sub["rmse_ci_low"], errors="coerce")
                highs = pd.to_numeric(sub["rmse_ci_high"], errors="coerce")
                for yi, (center, low, high) in enumerate(
                    zip(centers, lows, highs)
                ):
                    if pd.isna(center) or pd.isna(low) or pd.isna(high):
                        continue
                    ax.plot([low, high], [yi, yi], color="#3b6ea5", lw=2.5)
                    ax.plot(center, yi, marker="o", color="#d97706", ms=7)
                ax.axvline(0, color="#6b7280", lw=1, ls="--", zorder=0)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(
                    [labels.get(v, str(v)) for v in sub["treatment_variant"]]
                )
                ax.set_title(f"提前 {horizon} 天，固定 alpha=1, l1=0.9")
                ax.set_xlabel("RMSE 增益, km/s（正值=加入日影改善）")
                ax.grid(alpha=0.25, axis="x")
            fig.suptitle("旧磁场多滞后设置迁移到 v3 的日影 RMSE 增益 95% CI")
            fig.savefig(
                OUTPUT_DIR / "legacy_v3_migration_gain_ci_fixed.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

        model_daily = daily_df[daily_df["group"] == "model"].copy()
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(11, 7),
            constrained_layout=True,
            sharex=False,
        )
        for ax, horizon in zip(axes, (1, 3)):
            true_sub = model_daily[model_daily["horizon_days"] == horizon]
            if true_sub.empty:
                continue
            y_true = (
                true_sub.drop_duplicates("sample_id", keep="first")
                .sort_values("issue_date")
            )
            ax.plot(
                pd.to_datetime(y_true["issue_date"]),
                y_true["y_true"],
                color="#374151",
                lw=1.3,
                marker="o",
                ms=2.5,
                label="真实 Vsw",
            )
            both_run = f"mag_both_fixed_h{horizon}"
            both = (
                model_daily[
                    (model_daily["horizon_days"] == horizon)
                    & (model_daily["run_id"] == both_run)
                ]
                .sort_values("issue_date")
            )
            if not both.empty:
                ax.plot(
                    pd.to_datetime(both["issue_date"]),
                    both["y_pred"],
                    color="#d97706",
                    lw=1.5,
                    marker="^",
                    ms=3,
                    label="固定参数+双方向日影",
                )
            ax.axvline(
                pd.Timestamp(TEST_START.date()),
                color="#6b7280",
                lw=0.8,
                ls=":",
            )
            ax.set_title(f"提前 {horizon} 天：固定旧参数迁移预测")
            ax.set_ylabel("SW Plasma Speed, km/s")
            ax.legend(frameon=False, ncol=3)
            ax.grid(alpha=0.25)
        fig.suptitle("仅展示迭代评估段（2021-08-26 至 2021-09-29 发报日）")
        fig.savefig(
            OUTPUT_DIR / "legacy_v3_migration_timeseries_fixed.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
    except Exception as exc:  # plots are auxiliary; never fail the data export
        print(f"plot export skipped: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
