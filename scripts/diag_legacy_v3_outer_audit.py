"""Read-only audit for the legacy->v3 outer time backtest artifacts.

The outer backtest itself is intentionally a supplementary reproducibility
check: the dev window (2021-08-26 .. 2021-09-29) never enters training,
parameter selection or evaluation.  This audit does not retrain anything; it
re-verifies the saved artifacts against the frozen calendar records and
documents every retained/dropped issue date in the two pre-registered early
windows.

Checks
------
1.  Planned window vs actual common-sample window, per fold and horizon,
    including the exact dropped dates and their reasons.
2.  Summary metrics recomputed from ``outer_backtest_daily.csv``.
3.  Paired-gain bootstrap recomputed with the same frozen settings.
4.  Calendar alignment: issue date + horizon == target date, finite true/pred
    values, no duplicated daily rows.
5.  Dev leakage: no training or test/selection sample may use the dev window.
6.  Search state for auto arms (completed / boundary hit / convergence).
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.json_safe import json_dumps
from scripts.diag_legacy_v3_migration import (
    _build_records_for_specs,
    _load_merged,
    _metrics,
    _retained_keys,
    _specs_from_pairs,
)
from scripts.diag_legacy_v3_outer_backtest import (
    DEV_START,
    OUT_DIR,
    OUTER_FOLDS,
    _block_bootstrap_for,
    _build_variant_plans,
    _fold_train_test,
    _gain_interpretation,
    _search_cv_folds,
)

plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False

DAILY_PATH = OUT_DIR / "outer_backtest_daily.csv"
SUMMARY_PATH = OUT_DIR / "outer_backtest_summary.csv"
GAIN_PATH = OUT_DIR / "outer_backtest_gain_comparison.csv"
MANIFEST_PATH = OUT_DIR / "outer_backtest_manifest.json"
MANIFEST_V1_PATH = OUT_DIR / "outer_backtest_manifest.v1.json"

NUMERIC_COLS = [
    "point_gain_rmse",
    "rmse_ci_low",
    "rmse_ci_high",
    "point_gain_mae",
    "mae_ci_low",
    "mae_ci_high",
    "point_gain_pearson_r",
    "pearson_r_ci_low",
    "pearson_r_ci_high",
]
INT_COLS = [
    "n_paired_samples",
    "n_nonempty_blocks",
    "min_blocks",
    "n_valid_pearson_draws",
]

INT_KEY_MAP = {
    "n_paired_samples": "n_test_samples",
    "n_nonempty_blocks": "n_nonempty_blocks",
    "min_blocks": "min_blocks",
    "n_valid_pearson_draws": "n_valid_pearson_draws",
}

BOOT_KEY_MAP = {
    "point_gain_rmse": ("point_gain_rmse", None),
    "rmse_ci_low": ("rmse_ci", 0),
    "rmse_ci_high": ("rmse_ci", 1),
    "point_gain_mae": ("point_gain_mae", None),
    "mae_ci_low": ("mae_ci", 0),
    "mae_ci_high": ("mae_ci", 1),
    "point_gain_pearson_r": ("point_gain_pearson_r", None),
    "pearson_r_ci_low": ("pearson_r_ci", 0),
    "pearson_r_ci_high": ("pearson_r_ci", 1),
}


def _close(a: Any, b: Any, *, atol: float = 1e-6, rel: float = 1e-6) -> bool:
    a_nan = a is None or (isinstance(a, float) and math.isnan(a))
    b_nan = b is None or (isinstance(b, float) and math.isnan(b))
    if a_nan or b_nan:
        return a_nan and b_nan
    try:
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=atol)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _frame_date_index(frame: pd.DataFrame) -> tuple[dict[pd.Timestamp, int], list[pd.Timestamp]]:
    date_to_index = {
        pd.Timestamp(value).normalize(): index
        for index, value in enumerate(frame["TIME"])
    }
    return date_to_index, sorted(date_to_index)


def _missing_fields_for_issue(
    frame: pd.DataFrame,
    date_to_index: dict[pd.Timestamp, int],
    issue: pd.Timestamp,
    horizon: int,
    specs_all: list[tuple[str, int]],
) -> list[str]:
    missing: list[str] = []
    for variable, offset in specs_all:
        feature_date = issue + pd.Timedelta(days=int(offset))
        if feature_date.normalize() not in date_to_index:
            missing.append(f"{variable}@{feature_date.date().isoformat()}:日历缺口")
            continue
        row_index = date_to_index[feature_date.normalize()]
        value = frame.iloc[row_index][variable]
        if pd.isna(value) or not np.isfinite(float(value)):
            missing.append(f"{variable}@{feature_date.date().isoformat()}:值为缺失/非有限")
    target_date = issue + pd.Timedelta(days=int(horizon))
    if target_date.normalize() not in date_to_index:
        missing.append(f"SW Plasma Speed, km/s@{target_date.date().isoformat()}:日历缺口")
    else:
        row_index = date_to_index[target_date.normalize()]
        value = frame.iloc[row_index]["SW Plasma Speed, km/s"]
        if pd.isna(value) or not np.isfinite(float(value)):
            missing.append(
                f"SW Plasma Speed, km/s@{target_date.date().isoformat()}:值为缺失/非有限"
            )
    return list(dict.fromkeys(missing))


def build_window_coverage(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    date_to_index, _ = _frame_date_index(frame)
    coverage_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for horizon in (1, 3):
        plans = _build_variant_plans(frame, horizon)
        planned_dates = [
            pd.Timestamp(value).normalize()
            for value in pd.date_range(
                OUTER_FOLDS[0]["eval_start"],
                OUTER_FOLDS[-1]["eval_end"],
                freq="D",
            )
        ]
        # Rebuild the same common key set as the backtest generator.
        common_keys = set.intersection(
            *[_retained_keys(plans[variant]["records"]) for variant in plans]
        )
        common_issue_dates = {
            record.issue_date.normalize()
            for record in plans["mag"]["records"]
            if record.compact_key in common_keys
        }
        specs_all: list[tuple[str, int]] = []
        for variant, plan in plans.items():
            for spec in plan["specs"]:
                specs_all.append((spec.variable, int(spec.offset_days)))

        for fold in OUTER_FOLDS:
            window_dates = [
                value
                for value in planned_dates
                if fold["eval_start"] <= value <= fold["eval_end"]
            ]
            common_dates: list[pd.Timestamp] = []
            gap_reasons: Counter[str] = Counter()
            variant_reasons: dict[str, Counter[str]] = {
                variant: Counter() for variant in plans
            }
            dropped_dates: list[str] = []
            missing_field_set: Counter[str] = Counter()

            for issue_value in window_dates:
                issue = pd.Timestamp(issue_value).normalize()
                statuses: dict[str, str] = {}
                all_retained = True
                for variant, plan in plans.items():
                    record = next(
                        (
                            record
                            for record in plan["records"]
                            if record.issue_date.normalize() == issue
                        ),
                        None,
                    )
                    if record is None or record.drop_reason is not None:
                        reason = (
                            record.drop_reason
                            if record is not None
                            else "record_absent"
                        )
                        statuses[variant] = reason or "dropped"
                        variant_reasons[variant][statuses[variant]] += 1
                        gap_reasons[statuses[variant]] += 1
                        all_retained = False
                    else:
                        statuses[variant] = "retained"
                        variant_reasons[variant]["retained"] += 1

                missing_fields = _missing_fields_for_issue(
                    frame,
                    date_to_index,
                    issue,
                    horizon,
                    specs_all,
                )
                if missing_fields:
                    for field_name in missing_fields:
                        missing_field_set[field_name] += 1
                is_common = all_retained and issue in common_issue_dates
                if is_common:
                    common_dates.append(issue)
                else:
                    dropped_dates.append(issue.date().isoformat())
                coverage_rows.append(
                    {
                        "fold_id": fold["fold_id"],
                        "horizon_days": horizon,
                        "planned_date": issue.date().isoformat(),
                        **{
                            f"{variant}_status": statuses[variant]
                            for variant in plans
                        },
                        "common_retained": bool(is_common),
                        "missing_fields_json": json.dumps(
                            missing_fields, ensure_ascii=False
                        ),
                    }
                )

            summaries.append(
                {
                    "fold_id": fold["fold_id"],
                    "horizon_days": horizon,
                    "planned_start": fold["eval_start"].date().isoformat(),
                    "planned_end": fold["eval_end"].date().isoformat(),
                    "planned_count": len(window_dates),
                    "actual_start": (
                        min(common_dates).date().isoformat()
                        if common_dates
                        else None
                    ),
                    "actual_end": (
                        max(common_dates).date().isoformat()
                        if common_dates
                        else None
                    ),
                    "actual_common_count": len(common_dates),
                    "dropped_count": len(dropped_dates),
                    "dropped_dates_json": json.dumps(
                        dropped_dates, ensure_ascii=False
                    ),
                    "reason_counts_json": json.dumps(
                        dict(gap_reasons), ensure_ascii=False
                    ),
                    "variant_reason_counts_json": json.dumps(
                        {
                            variant: dict(counts)
                            for variant, counts in variant_reasons.items()
                        },
                        ensure_ascii=False,
                    ),
                    "missing_fields_json": json.dumps(
                        dict(missing_field_set), ensure_ascii=False
                    ),
                }
            )

    coverage_df = pd.DataFrame(coverage_rows)
    return coverage_df, summaries


def recheck_summary_metrics(
    summary: pd.DataFrame,
    daily: pd.DataFrame,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked = 0
    for row_index, row in summary.iterrows():
        sub = daily[daily["run_id"] == row["run_id"]].sort_values("issue_date")
        if sub.empty:
            failures.append(
                {"run_id": row["run_id"], "check": "daily_missing"}
            )
            continue
        checked += 1
        y_true = np.asarray(sub["y_true"], dtype=float)
        y_pred = np.asarray(sub["y_pred"], dtype=float)
        metrics = _metrics(y_true, y_pred)
        for key in (
            "rmse",
            "mae",
            "r2",
            "pearson_r",
            "unavailable_reason",
            "prediction_std",
            "true_std",
        ):
            expected = row.get(f"eval_{key}")
            actual = metrics[key]
            if not _close(actual, expected):
                failures.append(
                    {
                        "run_id": row["run_id"],
                        "check": f"eval_{key}",
                        "expected": expected,
                        "actual": actual,
                    }
                )
        if int(row["n_test"]) != int(len(sub)):
            failures.append(
                {
                    "run_id": row["run_id"],
                    "check": "n_test",
                    "expected": int(row["n_test"]),
                    "actual": int(len(sub)),
                }
            )
        if sub["issue_date"].min() != row.get("eval_date_start"):
            failures.append(
                {
                    "run_id": row["run_id"],
                    "check": "eval_date_start",
                    "expected": row.get("eval_date_start"),
                    "actual": sub["issue_date"].min(),
                }
            )
        if sub["issue_date"].max() != row.get("eval_date_end"):
            failures.append(
                {
                    "run_id": row["run_id"],
                    "check": "eval_date_end",
                    "expected": row.get("eval_date_end"),
                    "actual": sub["issue_date"].max(),
                }
            )
    return {
        "rows_checked": checked,
        "passed": len(failures) == 0,
        "failures": failures[:50],
    }


def recheck_bootstrap(daily: pd.DataFrame, gain: pd.DataFrame) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked = 0
    for row_index, row in gain.iterrows():
        base = (
            daily[daily["run_id"] == row["baseline_run_id"]]
            .sort_values("issue_date")
            .reset_index(drop=True)
        )
        treat = (
            daily[daily["run_id"] == row["treatment_run_id"]]
            .sort_values("issue_date")
            .reset_index(drop=True)
        )
        if base.empty or treat.empty:
            failures.append(
                {
                    "run_id": row["baseline_run_id"],
                    "treatment_run_id": row["treatment_run_id"],
                    "check": "daily_missing",
                }
            )
            continue
        checked += 1
        boot = _block_bootstrap_for(base, treat)
        expected = row
        for key in NUMERIC_COLS:
            boot_key, boot_index = BOOT_KEY_MAP[key]
            actual = boot[boot_key]
            if boot_index is not None:
                actual = actual[boot_index]
            if not _close(actual, expected.get(key), atol=1e-7):
                failures.append(
                    {
                        "baseline_run_id": row["baseline_run_id"],
                        "treatment_run_id": row["treatment_run_id"],
                        "check": key,
                        "expected": expected.get(key),
                        "actual": actual,
                    }
                )
        for key in INT_COLS:
            actual_count = int(boot[INT_KEY_MAP[key]])
            if actual_count != int(expected.get(key, -1)):
                failures.append(
                    {
                        "baseline_run_id": row["baseline_run_id"],
                        "treatment_run_id": row["treatment_run_id"],
                        "check": key,
                        "expected": expected.get(key),
                        "actual": actual_count,
                    }
                )
        for key in (
            "insufficient_blocks",
            "stable_rmse",
        ):
            if bool(boot[key]) != bool(expected.get(key, False)):
                failures.append(
                    {
                        "baseline_run_id": row["baseline_run_id"],
                        "treatment_run_id": row["treatment_run_id"],
                        "check": key,
                        "expected": bool(expected.get(key, False)),
                        "actual": bool(boot[key]),
                    }
                )
        if _gain_interpretation(boot) != expected.get("interpretation"):
            failures.append(
                {
                    "baseline_run_id": row["baseline_run_id"],
                    "treatment_run_id": row["treatment_run_id"],
                    "check": "interpretation",
                    "expected": expected.get("interpretation"),
                    "actual": _gain_interpretation(boot),
                }
            )
    return {
        "comparisons_checked": checked,
        "passed": len(failures) == 0,
        "failures": failures[:50],
    }


def recheck_calendar_alignment(daily: pd.DataFrame) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    daily = daily.copy()
    daily["issue_dt"] = pd.to_datetime(daily["issue_date"]).dt.normalize()
    daily["target_dt"] = pd.to_datetime(daily["target_date"]).dt.normalize()
    daily["expect_target"] = daily["issue_dt"] + pd.to_timedelta(
        daily["horizon_days"], unit="D"
    )
    mismatched = daily[daily["target_dt"] != daily["expect_target"]]
    if len(mismatched):
        failures.append(
            {
                "check": "target_date_mismatch",
                "count": int(len(mismatched)),
                "sample": mismatched[
                    ["run_id", "issue_date", "target_date", "horizon_days"]
                ]
                .head(10)
                .to_dict("records"),
            }
        )
    if not np.isfinite(daily["y_true"].to_numpy(dtype=float)).all():
        failures.append({"check": "non_finite_y_true", "count": -1})
    if not np.isfinite(daily["y_pred"].to_numpy(dtype=float)).all():
        failures.append({"check": "non_finite_y_pred", "count": -1})

    dupes = daily.duplicated(subset=["run_id", "issue_date"]).sum()
    if int(dupes):
        failures.append({"check": "duplicate_daily_rows", "count": int(dupes)})
    return {"passed": len(failures) == 0, "failures": failures[:20]}


def recheck_dev_isolation(frame: pd.DataFrame) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked_folds = 0
    for horizon in (1, 3):
        plans = _build_variant_plans(frame, horizon)
        common_keys = set.intersection(
            *[_retained_keys(plans[variant]["records"]) for variant in plans]
        )
        common_records = [
            record
            for plan in plans.values()
            for record in plan["records"]
            if record.compact_key in common_keys
        ]
        for fold in OUTER_FOLDS:
            train, test = _fold_train_test(
                common_records, fold["eval_start"], fold["eval_end"]
            )
            checked_folds += 1
            train_issue_max = max(
                (record.issue_date for record in train), default=None
            )
            train_target_max = max(
                (record.target_date for record in train), default=None
            )
            if train_issue_max is not None and train_issue_max >= DEV_START:
                failures.append(
                    {
                        "fold": fold["fold_id"],
                        "horizon": horizon,
                        "check": "train_issue_in_dev",
                        "max": str(train_issue_max.date()),
                    }
                )
            if train_target_max is not None and train_target_max >= fold["eval_start"]:
                failures.append(
                    {
                        "fold": fold["fold_id"],
                        "horizon": horizon,
                        "check": "train_target_past_eval_start",
                        "max": str(train_target_max.date()),
                    }
                )
            if any(
                record.issue_date < fold["eval_start"]
                or record.issue_date > fold["eval_end"]
                for record in test
            ):
                failures.append(
                    {
                        "fold": fold["fold_id"],
                        "horizon": horizon,
                        "check": "test_outside_window",
                    }
                )
            if any(record.issue_date >= DEV_START for record in test):
                failures.append(
                    {
                        "fold": fold["fold_id"],
                        "horizon": horizon,
                        "check": "test_in_dev",
                    }
                )

            # Internal parameter-search folds must also keep labels before eval.
            cv_folds = _search_cv_folds(
                [record for record in train if record.drop_reason is None],
                2,
            )
            for train_ids, eval_ids in cv_folds:
                by_id = {record.sample_id: record for record in train}
                eval_records = [by_id[sid] for sid in eval_ids if sid in by_id]
                train_records = [by_id[sid] for sid in train_ids if sid in by_id]
                if not eval_records:
                    continue
                eval_start = eval_records[0].issue_date
                late = [
                    record.target_date
                    for record in train_records
                    if record.target_date >= eval_start
                ]
                if late:
                    failures.append(
                        {
                            "fold": fold["fold_id"],
                            "horizon": horizon,
                            "check": "internal_cv_label_leak",
                            "late_count": len(late),
                            "sample": [str(item.date()) for item in late[:5]],
                        }
                    )
    return {
        "checked_folds": checked_folds,
        "passed": len(failures) == 0,
        "failures": failures[:50],
    }


def search_state(summary: pd.DataFrame) -> dict[str, Any]:
    auto = summary[summary["mode"] == "auto"]
    fixed = summary[summary["mode"] == "fixed"]
    return {
        "auto_runs": int(len(auto)),
        "fixed_runs": int(len(fixed)),
        "search_completed": int(auto["search_completed"].fillna(False).astype(bool).sum()),
        "boundary_hit_count": int(auto["boundary_hit"].fillna(False).astype(bool).sum()),
        "solver_converged_count": int(
            auto["solver_converged"].fillna(False).astype(bool).sum()
        ),
        "grid_size_distinct": sorted(
            {
                int(value)
                for value in auto["grid_size"].dropna().tolist()
            }
        ),
    }


def write_gain_ci_plot(gain: pd.DataFrame) -> Path:
    labels = {
        "mag_ns": "南北日影",
        "mag_ew": "东西日影",
        "mag_both": "双方向日影",
    }
    marker = {"fixed": "o", "auto": "^"}
    path = OUT_DIR / "outer_backtest_gain_ci_by_fold.png"
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 6.8),
        constrained_layout=True,
        sharey=False,
    )
    for ax, horizon in zip(axes, (1, 3)):
        sub = gain[gain["horizon_days"] == horizon].copy()
        sub["label"] = (
            sub["fold_id"].str.upper()
            + " "
            + sub["mode"].map({"auto": "自动选参", "fixed": "固定参数"})
            + " "
            + sub["treatment_variant"].map(labels)
        )
        sub = sub.sort_values(["fold_id", "mode", "treatment_variant"])
        y_pos = np.arange(len(sub))
        for yi, row in enumerate(sub.itertuples(index=False)):
            low = float(row.rmse_ci_low)
            high = float(row.rmse_ci_high)
            center = float(row.point_gain_rmse)
            color = {"mag_ns": "#1f77b4", "mag_ew": "#d97706", "mag_both": "#5a4fcf"}[
                row.treatment_variant
            ]
            ax.plot(
                [low, high],
                [yi, yi],
                color=color,
                lw=2.2,
                solid_capstyle="round",
            )
            ax.plot(
                center,
                yi,
                marker=marker[row.mode],
                color=color,
                ms=7,
            )
            ax.annotate(
                f"n={int(row.n_paired_samples)}",
                xy=(high, yi),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
                color="#6b7280",
            )
        ax.axvline(0, color="#374151", lw=1.1, ls="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["label"], fontsize=8)
        ax.set_xlabel("RMSE 增益, km/s（正值 = 加入日影改善）")
        ax.set_title(f"提前 {horizon} 天：补充时间回测增益 95% CI")
        ax.grid(alpha=0.22, axis="x")
        ax.set_xlim(
            min(0.0, float(sub["rmse_ci_low"].min())) - 0.25,
            max(0.0, float(sub["rmse_ci_high"].max())) + 0.35,
        )
    fig.suptitle(
        "外层窗口 f1=07-05..07-28，f2=07-29..08-25；dev 段 08-26 起未参与训练/选参",
        fontsize=10,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    frame, source_audit = _load_merged()
    daily = _read_csv(DAILY_PATH)
    summary = _read_csv(SUMMARY_PATH)
    gain = _read_csv(GAIN_PATH)

    coverage_df, window_summaries = build_window_coverage(frame)
    coverage_path = OUT_DIR / "outer_backtest_window_coverage.csv"
    gap_summary_path = OUT_DIR / "outer_backtest_window_gap_summary.csv"
    coverage_df.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    gap_summary_df = pd.DataFrame(window_summaries).sort_values(
        ["horizon_days", "fold_id"]
    )
    gap_summary_df.to_csv(gap_summary_path, index=False, encoding="utf-8-sig")

    metric_recheck = recheck_summary_metrics(summary, daily)
    bootstrap_recheck = recheck_bootstrap(daily, gain)
    calendar_recheck = recheck_calendar_alignment(daily)
    leakage_recheck = recheck_dev_isolation(frame)
    search = search_state(summary)
    plot_path = write_gain_ci_plot(gain)
    audit_path = OUT_DIR / "outer_backtest_audit.json"

    results = {
        "schema_version": "legacy-v3-outer-backtest-audit-v1",
        "label": "outer_backtest_read_only_audit",
        "generated_at": datetime.now().isoformat(),
        "inputs": {
            "summary": str(SUMMARY_PATH),
            "daily_predictions": str(DAILY_PATH),
            "gain_comparison": str(GAIN_PATH),
            "manifest": str(MANIFEST_PATH),
        },
        "window_coverage": window_summaries,
        "metric_recheck": metric_recheck,
        "bootstrap_recheck": bootstrap_recheck,
        "calendar_recheck": calendar_recheck,
        "dev_isolation_recheck": leakage_recheck,
        "search_state": search,
        "output_files": {
            "window_coverage": str(coverage_path),
            "window_gap_summary": str(gap_summary_path),
            "audit_json": str(audit_path),
            "gain_ci_plot": str(plot_path),
        },
        "overall_passed": bool(
            metric_recheck["passed"]
            and bootstrap_recheck["passed"]
            and calendar_recheck["passed"]
            and leakage_recheck["passed"]
        ),
    }

    audit_path.write_text(json_dumps(results, indent=2) + "\n", encoding="utf-8")

    # Keep v1 manifest as the original record, then write an enriched v2 that
    # states both the pre-registered windows and the actual common dates.
    if not MANIFEST_V1_PATH.exists() and MANIFEST_PATH.exists():
        MANIFEST_V1_PATH.write_text(
            MANIFEST_PATH.read_text(encoding="utf-8"), encoding="utf-8"
        )

    v2_manifest = {
        "schema_version": "legacy-v3-outer-backtest-v2",
        "supersedes": "legacy-v3-outer-backtest-v1",
        "generated_at": results["generated_at"],
        "scope": "只读补充时间回测；不修改闭环状态、协议、前端或报告；dev 段 2021-08-26 起未进入任何训练或选参",
        "dev_segment": {
            "start": DEV_START.date().isoformat(),
            "end": "2021-09-29",
            "label": "迭代评估段，未纳入本回测训练/评价",
        },
        "outer_folds": [
            {
                "fold_id": fold["fold_id"],
                "planned_eval_start": fold["eval_start"].date().isoformat(),
                "planned_eval_end": fold["eval_end"].date().isoformat(),
                "note": fold["note"],
                "actual_common_windows": [
                    {
                        "horizon_days": int(summary_row["horizon_days"]),
                        "actual_start": summary_row["actual_start"],
                        "actual_end": summary_row["actual_end"],
                        "actual_common_count": int(
                            summary_row["actual_common_count"]
                        ),
                        "planned_count": int(summary_row["planned_count"]),
                        "dropped_count": int(summary_row["dropped_count"]),
                        "dropped_dates": json.loads(
                            summary_row["dropped_dates_json"]
                        ),
                        "reason_counts": json.loads(
                            summary_row["reason_counts_json"]
                        ),
                        "variant_reason_counts": json.loads(
                            summary_row["variant_reason_counts_json"]
                        ),
                        "missing_fields": json.loads(
                            summary_row["missing_fields_json"]
                        ),
                    }
                    for summary_row in window_summaries
                    if summary_row["fold_id"] == fold["fold_id"]
                ],
            }
            for fold in OUTER_FOLDS
        ],
        "data": source_audit,
        "grid": {
            "alpha_grid": summary["alpha_grid"].dropna().tolist()
            if "alpha_grid" in summary and summary["alpha_grid"].notna().any()
            else None,
            "l1_ratio_grid": summary["l1_ratio_grid"].dropna().tolist()
            if "l1_ratio_grid" in summary and summary["l1_ratio_grid"].notna().any()
            else None,
        },
        "bootstrap": {
            "block_size": 7,
            "n_draws": 2000,
            "seed": 20261017,
            "min_blocks": 2,
        },
        "status": {
            "dev_window_excluded_from_training": True,
            "closed_loop_state_modified": False,
            "audit_passed": results["overall_passed"],
            "scientific_claim": (
                "结果用于核验 dev 中唯一不跨零的 mag_both 改善能否在早段窗口复现；"
                "特征定义与网格仍来自 dev 探索，因此这是补充时间回测，"
                "不宣称独立盲测，也不随意外扩为最终测试。"
            ),
        },
        "output_files": {
            "window_coverage": str(coverage_path),
            "window_gap_summary": str(gap_summary_path),
            "audit": str(audit_path),
            "gain_ci_plot": str(plot_path),
        },
    }
    v2_manifest["grid"] = {
        "alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "l1_ratio_grid": [0.1, 0.3, 0.5, 0.7, 0.9],
        "max_combinations": 30,
    }
    MANIFEST_PATH.write_text(
        json_dumps(v2_manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(
        json_dumps(
            {
                "overall_passed": results["overall_passed"],
                "metric_recheck": metric_recheck["passed"],
                "bootstrap_recheck": bootstrap_recheck["passed"],
                "calendar_recheck": calendar_recheck["passed"],
                "dev_isolation_recheck": leakage_recheck["passed"],
                "window_summaries": window_summaries,
                "search_state": search,
                "output_files": results["output_files"],
                "gap_summary": str(gap_summary_path),
                "plot": str(plot_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if results["overall_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
