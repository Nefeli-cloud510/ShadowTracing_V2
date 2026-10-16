"""Pre-registered outer time backtest for the legacy->v3 migration case.

Read-only diagnostic.  It does not modify the live session, hypothesis state,
protocols, frontend or reports.  It reuses the frozen feature definitions and
grids from ``diag_legacy_v3_migration.py`` but evaluates them on held-out
early-2021 windows that were not used for model fitting during the dev
backtest (2021-08-26 .. 2021-09-29).

Design
------
- The 35-day dev window stays untouched: no training sample, no parameter
  selection, no metric comparison uses issue dates on/after 2021-08-26.
- Evaluation windows are fixed before running:
      fold1  2021-07-05 .. 2021-07-28
      fold2  2021-07-29 .. 2021-08-25
  Each fold trains on all retained issue samples strictly before its window.
- Inside each fold, auto arms hold out their own training window and run the
  same frozen 30-combination grid; fixed arms use the legacy reference
  parameters without search.
- All metrics and paired gains are recomputed from daily predictions.  Pearson
  r is always reported; undefined values are reported as unavailable instead
  of 0.

Status labels
-------------
This is a supplementary time backtest in the sense that the feature
definitions, grids, baselines and horizons were selected using the dev window.
The early 2021 folds themselves were not used to build or select any model, so
they provide a reproducibility check, not a sealed independent blind test.
"""

from __future__ import annotations

import json
import math
import sys
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

from core.calendar_samples import FeatureSpec
from core.json_safe import json_dumps
from core.robustness_diagnostics import paired_calendar_block_bootstrap
from models.elasticnet_runner import (
    ElasticNetConfig,
    make_model,
    select_best_params,
)
from scripts.diag_legacy_v3_migration import (
    ALPHA_GRID,
    BLOCK_SIZE,
    BOOTSTRAP_SEED,
    LEGACY_TIMELAG_PARAMS,
    LEGACY_TIMELAG_VARIANTS,
    LEGACY_V2_OFFSETS,
    LEGACY_V2_PARAMS,
    L1_GRID,
    MAX_COMBINATIONS,
    MIN_BLOCKS,
    N_BOOTSTRAP_DRAWS,
    N_INTERNAL_FOLDS,
    TEST_END,
    TEST_START,
    VARIANT_LABELS,
    _build_records_for_specs,
    _config,
    _extract_feature_matrix,
    _metrics,
    _retained_keys,
    _specs_from_pairs,
    _summary_row,
    make_calendar_split,
)


OUT_DIR = ROOT / "artifacts" / "legacy_v3_migration" / "outer_backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEV_START = pd.Timestamp("2021-08-26")
DEV_END = TEST_END

# Pre-registered held-out evaluation windows, all strictly before DEV_START.
OUTER_FOLDS = [
    {
        "fold_id": "f1",
        "eval_start": pd.Timestamp("2021-07-05"),
        "eval_end": pd.Timestamp("2021-07-28"),
        "note": "未参与dev选参的早段窗口",
    },
    {
        "fold_id": "f2",
        "eval_start": pd.Timestamp("2021-07-29"),
        "eval_end": pd.Timestamp("2021-08-25"),
        "note": "未参与dev选参、紧邻dev起点之前的窗口",
    },
]


def _load_frame():
    # Rebuild the exact merged frame used by the migration diagnostics.
    from scripts.diag_legacy_v3_migration import _load_merged

    frame, audit = _load_merged()
    return frame, audit


def _build_variant_plans(frame, horizon_days):
    variant_plans = {}
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
    v2_specs = tuple(
        FeatureSpec(variable=v, offset_days=int(o)) for v, o in LEGACY_V2_OFFSETS
    )
    v2_records, v2_drops = _build_records_for_specs(
        frame, v2_specs, horizon_days=horizon_days
    )
    variant_plans["v2_mag"] = {
        "specs": v2_specs,
        "records": v2_records,
        "drops": v2_drops,
        "label": VARIANT_LABELS["v2_mag"],
    }
    return variant_plans


def _fold_train_test(
    records,
    fold_start,
    fold_end,
):
    train = [
        r
        for r in records
        if r.drop_reason is None
        and r.issue_date < fold_start
        and r.target_date < fold_start
    ]
    test = [
        r
        for r in records
        if r.drop_reason is None
        and fold_start <= r.issue_date <= fold_end
    ]
    return train, test


def _search_cv_folds(train_records, n_internal_folds):
    """Expanding calendar folds internal to a *single* outer training window."""

    # Pre-registered calendar cuts, fixed before running and independent of
    # the number of retained rows.  Internal evaluations end well before the
    # outer evaluation window so that search scores never double-count the
    # held-out outer period.
    internal_specs = [
        {
            "cut": pd.Timestamp("2021-05-15"),
            "train_max_target": pd.Timestamp("2021-05-15"),
            "eval_start": pd.Timestamp("2021-05-15"),
            "eval_end": pd.Timestamp("2021-06-10"),
        },
        {
            "cut": pd.Timestamp("2021-06-10"),
            "train_max_target": pd.Timestamp("2021-06-10"),
            "eval_start": pd.Timestamp("2021-06-11"),
            "eval_end": pd.Timestamp("2021-07-03"),
        },
    ][:n_internal_folds]
    if not internal_specs:
        return []
    folds = []
    for spec in internal_specs:
        train_ids = {
            r.sample_id
            for r in train_records
            if r.issue_date < spec["cut"]
            and r.target_date < spec["train_max_target"]
        }
        eval_ids = {
            r.sample_id
            for r in train_records
            if r.issue_date >= spec["eval_start"]
            and r.issue_date <= spec["eval_end"]
            and r.target_date <= spec["eval_end"]
        }
        if not train_ids or not eval_ids:
            continue
        folds.append(
            (
                np.asarray(
                    [r.sample_id for r in train_records if r.sample_id in train_ids]
                ),
                np.asarray(
                    [r.sample_id for r in train_records if r.sample_id in eval_ids]
                ),
            )
        )
    return folds


def _run_one(
    family,
    X_train,
    y_train,
    X_test,
    cfg,
    cv_folds,
    X_test_columns,
):
    search = None
    if cfg.param_search_mode == "auto":
        search = select_best_params(X_train, y_train, cfg, cv_folds=cv_folds)
        alpha = search.selected_alpha
        l1 = search.selected_l1_ratio
    else:
        alpha = cfg.alpha
        l1 = cfg.l1_ratio
        search = None
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", make_model(cfg, alpha_override=alpha, l1_ratio_override=l1)),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = np.asarray(pipeline.predict(X_test), dtype=float)
    y_pred_train = np.asarray(pipeline.predict(X_train), dtype=float)
    return pipeline, search, y_pred, y_pred_train


def _block_bootstrap_for(
    baseline_daily,
    treatment_daily,
    block_size=BLOCK_SIZE,
    n_draws=N_BOOTSTRAP_DRAWS,
    seed=BOOTSTRAP_SEED,
    min_blocks=MIN_BLOCKS,
):
    return paired_calendar_block_bootstrap(
        baseline_daily,
        treatment_daily,
        block_size=block_size,
        n_draws=n_draws,
        seed=seed,
        min_blocks=min_blocks,
        verify_pearson=True,
    )


def _gain_row(
    fold,
    horizon,
    mode,
    base_run,
    treat_run,
    base_variant,
    treat_variant,
    boot,
):
    return {
        "fold_id": fold["fold_id"],
        "eval_start": str(fold["eval_start"].date()),
        "eval_end": str(fold["eval_end"].date()),
        "horizon_days": int(horizon),
        "mode": mode,
        "baseline_variant": base_variant,
        "treatment_variant": treat_variant,
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
        "insufficient_blocks": bool(boot["insufficient_blocks"]),
        "stable_rmse": bool(boot["stable_rmse"]),
        "n_valid_pearson_draws": int(boot["n_valid_pearson_draws"]),
        "interpretation": _gain_interpretation(boot),
    }


def _gain_interpretation(boot: dict) -> str:
    if bool(boot.get("insufficient_blocks")):
        return "有效时间块不足，未作方向性稳定判断"
    ci = boot.get("rmse_ci") or [float("nan"), float("nan")]
    if not math.isfinite(float(ci[0])) or not math.isfinite(float(ci[1])):
        return "RMSE 增益区间不可用"
    lower, upper = float(ci[0]), float(ci[1])
    if lower > 0:
        return "本回测窗口 RMSE 增益区间支持改善"
    if upper < 0:
        return "本回测窗口 RMSE 增益区间支持退化"
    return "RMSE 增益区间跨零，未检出稳定增益"


def main() -> int:
    frame, audit = _load_frame()
    audit["generated_at"] = datetime.now().isoformat()

    summary_rows: list[dict] = []
    daily_dfs: list[pd.DataFrame] = []
    coefficient_rows: list[dict] = []
    gain_rows: list[dict] = []
    fold_memo: dict[tuple[int, str], dict] = {}

    for horizon in (1, 3):
        plans = _build_variant_plans(frame, horizon)
        common_keys = set.intersection(
            *[
                _retained_keys(plans[variant]["records"])
                for variant in plans
            ]
        )
        for variant, plan in plans.items():
            common_records = [
                r for r in plan["records"] if r.compact_key in common_keys
            ]
            plan["common_records"] = common_records

        for fold in OUTER_FOLDS:
            fold_start = fold["eval_start"]
            fold_end = fold["eval_end"]
            assert fold_end < DEV_START
            fold_key = (horizon, fold["fold_id"])

            prepared = {}
            for variant, plan in plans.items():
                records = plan["common_records"]
                train_records, test_records = _fold_train_test(
                    records, fold_start, fold_end
                )
                X_all, y_all, meta_all = _extract_feature_matrix(
                    frame,
                    train_records + test_records,
                    plan["specs"],
                    horizon_days=horizon,
                )
                split_idx = len(train_records)
                X_train = X_all.iloc[:split_idx].reset_index(drop=True)
                X_test = X_all.iloc[split_idx:].reset_index(drop=True)
                y_train = np.asarray(y_all[:split_idx], dtype=float)
                y_test = np.asarray(y_all[split_idx:], dtype=float)
                meta_test = meta_all.iloc[split_idx:].reset_index(drop=True)
                prepared[variant] = {
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                    "meta_test": meta_test,
                    "train_records": train_records,
                    "test_records": test_records,
                }

            mag = prepared["mag"]
            cv_folds_mag = _search_cv_folds(
                mag["train_records"], N_INTERNAL_FOLDS
            )
            if not cv_folds_mag:
                print(
                    f"fold {fold['fold_id']} h{horizon} 无可用内部折，跳过"
                )
                continue

            # Convert sample-id folds to positional folds for each variant.
            index_by_id = {}
            for variant, data in prepared.items():
                ids = [r.sample_id for r in data["train_records"]]
                index_by_id[variant] = {sid: idx for idx, sid in enumerate(ids)}

            def _positional_folds(variant):
                idx = index_by_id[variant]
                folds = []
                for train_ids, eval_ids in cv_folds_mag:
                    t = np.asarray(
                        [idx[sid] for sid in train_ids if sid in idx], dtype=int
                    )
                    e = np.asarray(
                        [idx[sid] for sid in eval_ids if sid in idx], dtype=int
                    )
                    if len(t) and len(e):
                        folds.append((t, e))
                return folds

            daily_by_run = {}
            for variant in ("mag", "mag_ns", "mag_ew", "mag_both", "v2_mag"):
                data = prepared[variant]
                if len(data["X_test"]) == 0:
                    continue
                for mode in ("auto", "fixed"):
                    if mode == "auto":
                        cfg = _config(
                            "elasticnet",
                            alpha=LEGACY_TIMELAG_PARAMS["alpha"],
                            l1_ratio=LEGACY_TIMELAG_PARAMS["l1_ratio"],
                            mode="auto",
                        )
                    else:
                        if variant == "v2_mag":
                            cfg = _config(
                                "elasticnet",
                                alpha=LEGACY_V2_PARAMS["alpha"],
                                l1_ratio=LEGACY_V2_PARAMS["l1_ratio"],
                                mode="fixed",
                            )
                        else:
                            cfg = _config(
                                "elasticnet",
                                alpha=LEGACY_TIMELAG_PARAMS["alpha"],
                                l1_ratio=LEGACY_TIMELAG_PARAMS["l1_ratio"],
                                mode="fixed",
                            )
                    run_id = (
                        f"{variant}_{mode}_h{horizon}_"
                        f"{fold['fold_id']}"
                    )
                    _, search, y_pred, y_pred_train = _run_one(
                        "elasticnet",
                        data["X_train"],
                        data["y_train"],
                        data["X_test"],
                        cfg,
                        _positional_folds(variant),
                        list(data["X_test"].columns),
                    )
                    meta_test = data["meta_test"]
                    start = str(pd.to_datetime(meta_test["issue_date"]).min().date())
                    end = str(pd.to_datetime(meta_test["issue_date"]).max().date())
                    summary_rows.append(
                        _summary_row(
                            run_id,
                            "model",
                            variant,
                            "elasticnet",
                            mode,
                            horizon_days=horizon,
                            n_train=len(data["y_train"]),
                            y_train=data["y_train"],
                            y_test=data["y_test"],
                            y_pred=y_pred,
                            y_pred_train=y_pred_train,
                            spec_label=plans[variant]["label"],
                            test_start=start,
                            test_end=end,
                            search=search,
                        )
                    )
                    daily = pd.DataFrame(
                        {
                            "run_id": run_id,
                            "group": "model",
                            "variant": variant,
                            "model_family": "elasticnet",
                            "mode": mode,
                            "horizon_days": int(horizon),
                            "fold_id": fold["fold_id"],
                            "sample_id": meta_test["sample_id"].to_numpy(),
                            "issue_date": pd.to_datetime(
                                meta_test["issue_date"]
                            ).dt.strftime("%Y-%m-%d"),
                            "target_date": pd.to_datetime(
                                meta_test["target_date"]
                            ).dt.strftime("%Y-%m-%d"),
                            "y_true": np.asarray(data["y_test"], dtype=float),
                            "y_pred": np.asarray(y_pred, dtype=float),
                        }
                    )
                    daily_dfs.append(daily)
                    daily_by_run[
                        (variant, mode)
                    ] = daily

            if "mag" not in prepared or len(prepared["mag"]["X_test"]) == 0:
                continue
            for mode in ("auto", "fixed"):
                base = daily_by_run.get(("mag", mode))
                if base is None:
                    continue
                for treatment in ("mag_ns", "mag_ew", "mag_both"):
                    treat = daily_by_run.get((treatment, mode))
                    if treat is None:
                        continue
                    boot = _block_bootstrap_for(base, treat)
                    gain_rows.append(
                        _gain_row(
                            fold,
                            horizon,
                            mode,
                            base.loc[0, "run_id"],
                            treat.loc[0, "run_id"],
                            "mag",
                            treatment,
                            boot,
                        )
                    )

    summary_df = pd.DataFrame(summary_rows).replace([np.inf, -np.inf], np.nan)
    summary_df["fold_id"] = summary_df["run_id"].str.split("_").str[-1]
    summary_df = summary_df.sort_values(
        ["horizon_days", "fold_id", "group", "variant", "mode"],
        kind="mergesort",
    ).reset_index(drop=True)
    daily_df = (
        pd.concat(daily_dfs, ignore_index=True)
        .sort_values(
            ["horizon_days", "fold_id", "run_id", "issue_date"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    gain_df = (
        pd.DataFrame(gain_rows)
        .replace([np.inf, -np.inf], np.nan)
        .sort_values(["horizon_days", "fold_id", "mode", "treatment_variant"])
        .reset_index(drop=True)
    )

    summary_path = OUT_DIR / "outer_backtest_summary.csv"
    daily_path = OUT_DIR / "outer_backtest_daily.csv"
    gain_path = OUT_DIR / "outer_backtest_gain_comparison.csv"
    manifest_path = OUT_DIR / "outer_backtest_manifest.json"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    gain_df.to_csv(gain_path, index=False, encoding="utf-8-sig")

    # Planned windows are frozen; actual evaluation dates can be shorter
    # because a variant's lagged feature is missing.  Record both explicitly.
    actual_windows = (
        daily_df.groupby(["horizon_days", "fold_id"])["issue_date"]
        .agg(actual_start="min", actual_end="max", actual_count="nunique")
        .reset_index()
        .to_dict("records")
    )

    manifest = {
        "schema_version": "legacy-v3-outer-backtest-v2",
        "label": "supplementary_time_backtest",
        "generated_at": audit["generated_at"],
        "scope": (
            "只读补充时间回测；不修改闭环状态、协议、前端或报告；"
            "dev 段 2021-08-26 起未进入任何训练或选参"
        ),
        "dev_segment": {
            "start": DEV_START.date().isoformat(),
            "end": DEV_END.date().isoformat(),
            "label": "迭代评估段，未纳入本回测训练/评价",
        },
        "outer_folds": [
            {
                "fold_id": f["fold_id"],
                "eval_start": f["eval_start"].date().isoformat(),
                "eval_end": f["eval_end"].date().isoformat(),
                "note": f["note"],
                "actual_common_windows": [
                    {
                        "horizon_days": int(record["horizon_days"]),
                        "actual_start": record["actual_start"],
                        "actual_end": record["actual_end"],
                        "actual_common_count": int(record["actual_count"]),
                    }
                    for record in actual_windows
                    if record["fold_id"] == f["fold_id"]
                ],
            }
            for f in OUTER_FOLDS
        ],
        "data": audit,
        "grid": {
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
        "status": {
            "dev_window_excluded_from_training": True,
            "closed_loop_state_modified": False,
            "scientific_claim": (
                "结果用于核验改善在早段时段是否可重复；"
                "由于特征定义与网格仍来自 dev 探索，术语为补充时间回测，"
                "不宣称独立盲测。"
            ),
        },
    }
    manifest_path.write_text(
        json_dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"summary rows: {len(summary_df)}")
    print(f"daily rows: {len(daily_df)}")
    print(f"gain rows: {len(gain_df)}")
    print(f"output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
