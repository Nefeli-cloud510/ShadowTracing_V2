"""Standalone convergence audit for the legacy->v3 migration case.

This script reads only the frozen artifacts written by
``diag_legacy_v3_migration.py``. It does not retrain, does not expand the
hyperparameter grid, and does not touch session state, hypothesis support,
frontend files, or existing diagnostics.

Checks
------
1. 24 summary rows (20 model arms + 4 simple references) and 35 paired
   evaluation days per run.
2. Recompute RMSE/MAE/R2/Pearson from ``daily.csv`` and compare with the
   saved summary.
3. Recompute all 12 paired gains and their calendar-block bootstrap CIs.
4. Verify issue_date/target_date alignment (target = issue + horizon).
5. Verify feature dates never lie in the future at issue time.
6. Bind the final two figures to the same source run.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.json_safe import json_dumps
from core.robustness_diagnostics import paired_calendar_block_bootstrap


ART_DIR = ROOT / "artifacts" / "legacy_v3_migration"
OUT_DIR = ART_DIR / "final_case"

SUMMARY_PATH = ART_DIR / "legacy_v3_migration_summary.csv"
DAILY_PATH = ART_DIR / "legacy_v3_migration_daily.csv"
GAIN_PATH = ART_DIR / "legacy_v3_migration_gain_comparison.csv"
COEF_PATH = ART_DIR / "legacy_v3_migration_coefficients.csv"
SAMPLE_PATH = ART_DIR / "legacy_v3_migration_sample_audit.csv"
MANIFEST_PATH = ART_DIR / "legacy_v3_migration_manifest.json"

EXPECTED_SUMMARY_ROWS = 24
EXPECTED_GAIN_ROWS = 12
EXPECTED_COEF_ROWS = 20
EXPECTED_SAMPLE_ROWS = 10
EXPECTED_N_TEST = 35

BLOCK_SIZE = 7
N_DRAWS = 2000
SEED = 20261017
MIN_BLOCKS = 2

plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float | None, str | None]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return None, "insufficient_samples"
    if float(np.std(y_true)) <= 1e-12:
        return None, "constant_target"
    if float(np.std(y_pred)) <= 1e-12:
        return None, "constant_prediction"
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = float(np.corrcoef(y_true, y_pred)[0, 1])
    if math.isfinite(raw):
        return raw, None
    return None, "undefined"


def _recompute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pearson_r, reason = _pearson(y_true, y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r2": r2,
        "pearson_r": pearson_r,
        "unavailable_reason": reason,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(a: float | None, b: float | None, *, atol: float = 1e-9) -> bool:
    def _missing(value: float | None) -> bool:
        return value is None or (
            isinstance(value, float) and math.isnan(value)
        )

    if _missing(a) or _missing(b):
        return _missing(a) == _missing(b)
    a_f = float(a)
    b_f = float(b)
    return math.isfinite(a_f) and math.isfinite(b_f) and abs(a_f - b_f) <= atol


def _gain_interpretation(boot: dict) -> str:
    if bool(boot.get("insufficient_blocks")):
        return "有效时间块不足，未作方向性稳定判断"
    ci = boot.get("rmse_ci") or [float("nan"), float("nan")]
    lower, upper = float(ci[0]), float(ci[1])
    if lower > 0:
        return "本样本范围内 RMSE 增益区间支持改善"
    if upper < 0:
        return "本样本范围内 RMSE 增益区间支持退化"
    return "RMSE 增益区间跨零，未检出稳定增益"


def _figure_gain_ci(gain_df: pd.DataFrame, out_path: Path) -> None:
    fixed = gain_df[gain_df["mode"] == "fixed"]
    auto = gain_df[gain_df["mode"] == "auto"]
    labels = {
        "mag_ns": "＋南北日影",
        "mag_ew": "＋东西日影",
        "mag_both": "＋双方向日影",
    }
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.8),
        constrained_layout=True,
        sharey=True,
    )
    for ax, horizon in zip(axes, (1, 3)):
        sub_fixed = fixed[fixed["horizon_days"] == horizon].sort_values(
            "treatment_variant"
        )
        sub_auto = auto[auto["horizon_days"] == horizon].sort_values(
            "treatment_variant"
        )
        variants = list(sub_auto["treatment_variant"])
        for yi, variant in enumerate(variants):
            for sub, offset, style in (
                (sub_auto, +0.10, "main"),
                (sub_fixed, -0.10, "reference"),
            ):
                row = sub[sub["treatment_variant"] == variant]
                if row.empty:
                    continue
                center = float(row["point_gain_rmse"].iloc[0])
                low = float(row["rmse_ci_low"].iloc[0])
                high = float(row["rmse_ci_high"].iloc[0])
                if style == "main":
                    color_ci = "#1f4e79"
                    lw = 3.0
                    marker = "o"
                    color_marker = "#d97706"
                    ms = 8
                    zorder = 6
                else:
                    color_ci = "#9ca3af"
                    lw = 2.0
                    marker = "D"
                    color_marker = "#1f4e79"
                    ms = 6
                    zorder = 5
                ax.plot(
                    [low, high],
                    [yi + offset, yi + offset],
                    color=color_ci,
                    lw=lw,
                    zorder=zorder,
                )
                ax.plot(
                    center,
                    yi + offset,
                    marker=marker,
                    color=color_marker,
                    ms=ms,
                    fillstyle="none" if style == "reference" else "full",
                    zorder=zorder,
                )
        ax.axvline(0.0, color="#6b7280", lw=1.0, ls="--", zorder=2)
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels([labels.get(v, v) for v in variants])
        ax.set_title(f"提前 {horizon} 天")
        ax.set_xlabel("RMSE 增益, km/s（正值=加入日影改善）")
        ax.grid(alpha=0.25, axis="x")
        ax.text(
            0.98,
            0.02,
            "n=35, 相同迭代评估段",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#4b5563",
        )
    axes[0].legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor="#d97706",
                markeredgecolor="#1f4e79",
                label="训练内选参（正式主方案）",
            ),
            plt.Line2D(
                [],
                [],
                marker="D",
                color="w",
                markeredgecolor="#1f4e79",
                markerfacecolor="none",
                markerfacecoloralt="none",
                label="固定旧参数（参照）",
            ),
        ],
        loc="lower left",
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "日影对简洁磁场基线的 RMSE 增益与 95% CI（正值为改善）",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _figure_timeseries(
    daily_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.2),
        constrained_layout=True,
    )
    for ax, horizon in zip(axes, (1, 3)):
        sub = daily_df[daily_df["horizon_days"] == horizon].copy()
        truth = (
            sub[sub["run_id"] == f"mag_auto_h{horizon}"]
            .drop_duplicates("sample_id", keep="first")
            .sort_values("target_date")
        )
        if truth.empty:
            ax.set_title(f"提前 {horizon} 天：无可展示预测")
            continue
        x_true = pd.to_datetime(truth["target_date"])
        ax.plot(
            x_true,
            truth["y_true"],
            color="#374151",
            lw=1.4,
            marker="o",
            ms=3.2,
            label="真实 Vsw",
            zorder=4,
        )
        style_map = {
            f"mag_auto_h{horizon}": ("#2563eb", "--", "磁场基线"),
            f"mag_ns_auto_h{horizon}": ("#16a34a", "-.", "＋南北日影"),
            f"mag_both_auto_h{horizon}": ("#d97706", "-", "＋双方向日影"),
        }
        for run_id, (color, ls, label) in style_map.items():
            row = (
                sub[sub["run_id"] == run_id]
                .sort_values("target_date")
            )
            if row.empty:
                continue
            ax.plot(
                pd.to_datetime(row["target_date"]),
                row["y_pred"],
                color=color,
                ls=ls,
                lw=1.2,
                label=label,
                zorder=5,
            )
        text_lines = []
        for run_id in (
            f"mag_auto_h{horizon}",
            f"mag_ns_auto_h{horizon}",
            f"mag_both_auto_h{horizon}",
        ):
            srow = summary_df[summary_df["run_id"] == run_id]
            if srow.empty:
                continue
            srow = srow.iloc[0]
            rmse = float(srow["eval_rmse"])
            pear = srow["eval_pearson_r"]
            pear_text = f"{float(pear):.3f}" if pd.notna(pear) else "未计算"
            text_lines.append(
                f"{run_id.split('_auto')[0].split('_h')[0]} "
                f"RMSE={rmse:.2f} r={pear_text}"
            )
        text_block = " | ".join(text_lines)
        ax.set_title(
            f"提前 {horizon} 天：共同目标日期（发报日 + {horizon} 天）",
            fontsize=11,
        )
        ax.set_ylabel("SW Plasma Speed, km/s")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=4, fontsize=8.5, loc="upper left")
        ax.text(
            0.01,
            -0.19,
            text_block,
            transform=ax.transAxes,
            fontsize=8,
            color="#4b5563",
            ha="left",
        )
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%m-%d")
        )
        ax.xaxis.set_major_locator(
            mdates.DayLocator(interval=5)
        )
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(
        "训练内选参主方案：真实 Vsw 与三项 ElasticNet 预测（无预测区间时不加误差棒）",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    summary_df = pd.read_csv(SUMMARY_PATH, encoding="utf-8-sig")
    daily_df = pd.read_csv(DAILY_PATH, encoding="utf-8-sig")
    gain_df = pd.read_csv(GAIN_PATH, encoding="utf-8-sig")
    coef_df = pd.read_csv(COEF_PATH, encoding="utf-8-sig")
    sample_df = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    add_check(
        "summary row count",
        len(summary_df) == EXPECTED_SUMMARY_ROWS,
        f"expected {EXPECTED_SUMMARY_ROWS}, got {len(summary_df)} "
        f"({len(summary_df[summary_df['group'] == 'model'])} model arms + "
        f"{len(summary_df[summary_df['group'] == 'reference'])} references)",
    )
    add_check(
        "gain comparison row count",
        len(gain_df) == EXPECTED_GAIN_ROWS,
        f"expected {EXPECTED_GAIN_ROWS}, got {len(gain_df)}",
    )
    add_check(
        "coefficient row count",
        len(coef_df) == EXPECTED_COEF_ROWS,
        f"expected {EXPECTED_COEF_ROWS}, got {len(coef_df)} "
        "(references have no fitted coefficients)",
    )
    add_check(
        "sample audit row count",
        len(sample_df) == EXPECTED_SAMPLE_ROWS,
        f"expected {EXPECTED_SAMPLE_ROWS}, got {len(sample_df)}",
    )

    daily_by_run = {
        run_id: group.sort_values("issue_date").reset_index(drop=True)
        for run_id, group in daily_df.groupby("run_id", sort=False)
    }
    add_check(
        "daily run groups",
        len(daily_by_run) == EXPECTED_SUMMARY_ROWS
        and all(len(group) == EXPECTED_N_TEST for group in daily_by_run.values()),
        f"{len(daily_by_run)} run groups, each with "
        f"{sorted({len(group) for group in daily_by_run.values()})} rows",
    )

    # Recompute metrics day by day from the saved prediction file.
    max_metric_error = 0.0
    metric_mismatches: list[str] = []
    for _, srow in summary_df.iterrows():
        run_id = srow["run_id"]
        sub = daily_by_run[run_id]
        y_true = sub["y_true"].to_numpy(dtype=float)
        y_pred = sub["y_pred"].to_numpy(dtype=float)
        recomputed = _recompute_metrics(y_true, y_pred)
        compare_items = [
            ("eval_rmse", "rmse"),
            ("eval_mae", "mae"),
            ("eval_r2", "r2"),
            ("eval_pearson_r", "pearson_r"),
        ]
        for saved_col, recomp_key in compare_items:
            saved_val = srow[saved_col]
            recomp_val = recomputed[recomp_key]
            if not _close(saved_val, recomp_val):
                metric_mismatches.append(f"{run_id}:{saved_col}")
                max_metric_error = max(max_metric_error, float("inf"))
            elif saved_val is not None and pd.notna(saved_val) and pd.notna(recomp_val):
                max_metric_error = max(
                    max_metric_error, abs(float(saved_val) - float(recomp_val))
                )
    add_check(
        "metrics recomputed from daily predictions",
        not metric_mismatches,
        (
            "all 24 runs match saved RMSE/MAE/R2/Pearson, "
            f"max abs error {max_metric_error:.3e}"
            if not metric_mismatches
            else f"mismatches: {metric_mismatches}"
        ),
    )

    # Recompute all 12 paired gains and bootstrap intervals.
    gain_mismatches: list[str] = []
    max_boot_error = 0.0
    for _, grow in gain_df.iterrows():
        baseline_run = grow["baseline_run_id"]
        treat_run = grow["treatment_run_id"]
        base_df = daily_by_run[baseline_run]
        treat_df = daily_by_run[treat_run]
        boot = paired_calendar_block_bootstrap(
            base_df,
            treat_df,
            block_size=BLOCK_SIZE,
            n_draws=N_DRAWS,
            seed=SEED,
            min_blocks=MIN_BLOCKS,
            verify_pearson=True,
        )
        pairs = [
            ("point_gain_rmse", boot["point_gain_rmse"], "point_gain_rmse"),
            ("rmse_ci_low", boot["rmse_ci"][0], "rmse_ci_low"),
            ("rmse_ci_high", boot["rmse_ci"][1], "rmse_ci_high"),
            ("point_gain_mae", boot["point_gain_mae"], "point_gain_mae"),
            ("mae_ci_low", boot["mae_ci"][0], "mae_ci_low"),
            ("mae_ci_high", boot["mae_ci"][1], "mae_ci_high"),
            ("point_gain_pearson_r", boot["point_gain_pearson_r"], "point_gain_pearson_r"),
            ("pearson_r_ci_low", boot["pearson_r_ci"][0], "pearson_r_ci_low"),
            ("pearson_r_ci_high", boot["pearson_r_ci"][1], "pearson_r_ci_high"),
        ]
        for saved_col, recomputed, label in pairs:
            if not _close(grow[saved_col], recomputed):
                gain_mismatches.append(f"{baseline_run}:{label}")
            elif pd.notna(grow[saved_col]) and pd.notna(recomputed):
                max_boot_error = max(
                    max_boot_error,
                    abs(float(grow[saved_col]) - float(recomputed)),
                )
        for col in ("n_nonempty_blocks", "n_valid_pearson_draws"):
            if int(grow[col]) != int(boot[col]):
                gain_mismatches.append(f"{baseline_run}:{col}")
        for col in ("insufficient_blocks", "stable_rmse"):
            if bool(grow[col]) != bool(boot[col]):
                gain_mismatches.append(f"{baseline_run}:{col}")
    add_check(
        "12 gains and bootstrap intervals recomputed",
        not gain_mismatches,
        (
            "all saved point gains and 95% CIs reproduced, "
            f"max abs error {max_boot_error:.3e}"
            if not gain_mismatches
            else f"mismatches: {gain_mismatches}"
        ),
    )

    # Date alignment.
    date_mismatches = []
    for run_id, sub in daily_by_run.items():
        horizon = int(daily_df[daily_df["run_id"] == run_id]["horizon_days"].iloc[0])
        expected = pd.to_datetime(sub["issue_date"]) + pd.Timedelta(days=horizon)
        actual = pd.to_datetime(sub["target_date"])
        bad = int((actual.dt.normalize() != expected.dt.normalize()).sum())
        if bad:
            date_mismatches.append(run_id)
    add_check(
        "target date alignment",
        not date_mismatches,
        (
            "target_date = issue_date + horizon for every row"
            if not date_mismatches
            else f"mismatches: {date_mismatches}"
        ),
    )

    # Feature availability: all source offsets are non-positive at issue time.
    feature_rows = []
    future_usage: list[str] = []
    for variant_key, pairs in manifest["freeze"]["features"].items():
        for variable, lag in pairs:
            offset_form = variant_key == "v2_scalar_b_by_offsets"
            offset = int(lag) if offset_form else -int(lag)
            if offset > 0:
                future_usage.append(f"{variant_key}:{variable}:{lag}")
            feature_rows.append(
                {
                    "feature_variant": variant_key,
                    "variable": variable,
                    "config_value": int(lag),
                    "issue_offset_days": offset,
                    "feature_date_rule": (
                        f"issue_date + ({offset}) days (v3 direct offset)"
                        if offset_form
                        else f"issue_date - {int(lag)} days"
                    ),
                    "available_at_issue": bool(offset <= 0),
                }
            )
    add_check(
        "no future feature usage",
        not future_usage,
        (
            "all feature dates are issue date or earlier"
            if not future_usage
            else f"future offsets: {future_usage}"
        ),
    )

    # Search-state audit: auto arms completed, fixed arms are references only.
    auto_bad = summary_df[
        (summary_df["group"] == "model")
        & (summary_df["mode"] == "auto")
        & (
            (summary_df["search_completed"] != True)
            | (summary_df["grid_size"] != 30.0)
        )
    ]
    fixed_bad = summary_df[
        (summary_df["group"] == "model")
        & (summary_df["mode"] == "fixed")
        & (
            (summary_df["search_completed"] != False)
            | (summary_df["grid_size"] != 1.0)
        )
    ]
    add_check(
        "search state matches mode",
        auto_bad.empty and fixed_bad.empty,
        (
            "auto arms: search_completed=True, grid=30; "
            "fixed arms: no search, grid=1"
            if auto_bad.empty and fixed_bad.empty
            else (
                f"auto issues: {list(auto_bad['run_id'])}; "
                f"fixed issues: {list(fixed_bad['run_id'])}"
            )
        ),
    )

    # Build the final case tables and bind figures to source artifacts.
    model_df = summary_df[summary_df["group"] == "model"].copy()
    model_df["n_test"] = pd.to_numeric(model_df["n_test"], errors="coerce")
    pear_table_cols = [
        "run_id",
        "horizon_days",
        "mode",
        "variant",
        "n_train",
        "n_test",
        "eval_rmse",
        "eval_mae",
        "eval_r2",
        "eval_pearson_r",
        "eval_unavailable_reason",
        "selected_alpha",
        "selected_l1_ratio",
        "search_completed",
        "boundary_hit",
        "solver_converged",
        "grid_size",
    ]
    case_metric = model_df[pear_table_cols].copy()
    case_metric.to_csv(OUT_DIR / "final_case_metric_table.csv", index=False)

    case_gain = gain_df.copy()
    case_gain["interpretation"] = case_gain.apply(
        lambda row: _gain_interpretation(
            {
                "insufficient_blocks": bool(row["insufficient_blocks"]),
                "rmse_ci": [
                    float(row["rmse_ci_low"]),
                    float(row["rmse_ci_high"]),
                ],
            }
        ),
        axis=1,
    )
    case_gain.to_csv(OUT_DIR / "final_case_gain_table.csv", index=False)

    feature_map = pd.DataFrame(feature_rows)
    feature_map.to_csv(OUT_DIR / "final_case_feature_date_map.csv", index=False)

    gain_path = OUT_DIR / "final_gain_ci_auto_vs_fixed.png"
    timeseries_path = OUT_DIR / "final_timeseries_auto_main.png"
    _figure_gain_ci(gain_df, gain_path)
    _figure_timeseries(daily_df, summary_df, timeseries_path)
    add_check(
        "final figures generated from source run",
        gain_path.exists() and timeseries_path.exists(),
        f"gain: {gain_path.name}; timeseries: {timeseries_path.name}",
    )

    hash_map = {
        "summary": _sha256(SUMMARY_PATH),
        "daily_prediction_file": _sha256(DAILY_PATH),
        "gain_comparison": _sha256(GAIN_PATH),
        "coefficients": _sha256(COEF_PATH),
        "sample_audit": _sha256(SAMPLE_PATH),
        "source_manifest": _sha256(MANIFEST_PATH),
    }

    audit = {
        "schema_version": "legacy-v3-convergence-audit-v1",
        "generated_at": datetime.now().isoformat(),
        "scope": "只读收束审计；不修改闭环状态、协议、前端或报告",
        "checks": checks,
        "overall": "PASS" if all(c["passed"] for c in checks) else "FAIL",
        "counts": {
            "summary_rows": int(len(summary_df)),
            "model_arms": int(len(model_df)),
            "references": int(len(summary_df[summary_df["group"] == "reference"])),
            "daily_rows": int(len(daily_df)),
            "gain_rows": int(len(gain_df)),
            "coefficient_rows": int(len(coef_df)),
            "sample_audit_rows": int(len(sample_df)),
        },
        "source_input_sha256": hash_map,
        "date_definition": {
            "issue_date": "发报日（预测发布日）",
            "target_date": "issue_date + horizon_days",
            "feature_date": "issue_date - lag（v3 口径）或 issue_date + offset",
            "horizon_1_target": "2021-08-27 至 2021-09-30",
            "horizon_3_target": "2021-08-29 至 2021-10-02",
            "issue_window": "2021-08-26 至 2021-09-29",
            "n_eval_days": 35,
        },
        "selected_params_summary": (
            model_df[
                [
                    "run_id",
                    "horizon_days",
                    "mode",
                    "variant",
                    "selected_alpha",
                    "selected_l1_ratio",
                ]
            ]
            .sort_values(["horizon_days", "mode", "variant"])
            .to_dict(orient="records")
        ),
        "final_figures": {
            "gain_ci": str(gain_path.resolve()),
            "timeseries": str(timeseries_path.resolve()),
        },
        "output_files": {
            "metric_table": str(
                (OUT_DIR / "final_case_metric_table.csv").resolve()
            ),
            "gain_table": str((OUT_DIR / "final_case_gain_table.csv").resolve()),
            "feature_date_map": str(
                (OUT_DIR / "final_case_feature_date_map.csv").resolve()
            ),
            "audit": str((OUT_DIR / "convergence_case_audit.json").resolve()),
        },
        "limits": [
            "35 天评估段仍是迭代评估段 dev_backtest，不是独立盲测。",
            "只有 h3 auto mag_both 的 RMSE 增益 CI 在本样本范围内未跨零；"
            "它不能替代跨时段的独立时间回测。",
            "固定旧参数仅作为参照，不因这 35 天分数更高而事后改称正式主方案。",
        ],
    }
    (OUT_DIR / "convergence_case_audit.json").write_text(
        json_dumps(audit, indent=2),
        encoding="utf-8",
    )

    print(
        json_dumps(
            {
                "overall": audit["overall"],
                "checks": len(checks),
                "failed_checks": [
                    c["check"] for c in checks if not c["passed"]
                ],
                "counts": audit["counts"],
                "final_figures": audit["final_figures"],
                "audit_file": audit["output_files"]["audit"],
            },
            indent=2,
        )
    )
    return 0 if audit["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
