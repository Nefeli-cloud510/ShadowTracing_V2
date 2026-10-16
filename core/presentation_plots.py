"""Presentation-oriented figures for the ShadowTracing v3 demo.

These figures answer one scientific question each and are drawn from the
accepted archive artifacts of a single experiment run.  They never smooth
truth values, shift dates, connect non-adjacent calendar days, or add
prediction intervals that were not actually computed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Microsoft JhengHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

PRES_DARK = "#1f2937"
PRES_GRID = "#e5e7eb"
PRES_TRUTH = "#374151"
PRES_BLUE = "#1d4ed8"
PRES_ORANGE = "#ea580c"
PRES_MUTED = "#6b7280"

TARGET_FEATURE = "SW Plasma Speed, km/s"
SHADOW_FEATURE = "p2_y中心值"

FAMILY_LABELS = {
    "ridge": "Ridge",
    "elasticnet": "ElasticNet",
}
FAMILY_COLORS = {
    "ridge": PRES_BLUE,
    "elasticnet": PRES_ORANGE,
}

SHADOW_STATUS_COLORS = {
    "有效观测": "#2c7a7b",
    "原表缺失": "#e67e22",
    "质量不可用": "#c0392b",
}
SAMPLE_STATUS_COLORS = {
    "训练样本": "#8ea4b8",
    "迭代评估样本": "#1b6ca8",
    "测试样本": "#1b6ca8",
    "边界排除": "#d8c8a2",
    "滞后构造排除": "#c0392b",
}


def _family_label(model_family: str) -> str:
    return FAMILY_LABELS.get(model_family, model_family)


def _arm_config_path(archive_dir: Path, run_id: str) -> Path:
    return Path(archive_dir) / run_id / "config.json"


def _resolve_param_flags(
    arm: Mapping[str, Any],
    archive_dir: str | Path,
) -> tuple[str, str, str]:
    """Resolve search/boundary/solver status; old archives fall back to files."""

    search = arm.get("param_search") or {}
    sidecar = {}
    try:
        sidecar = json.loads(
            (
                Path(archive_dir) / "parameter_snapshot_note.json"
            ).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        sidecar = {}
    sidecar_entry = sidecar.get("actual_executed_search", {}).get(
        str(arm.get("run_id", "")), {}
    )

    def stringify(value: Any, default: str) -> str:
        if value is None:
            return default
        return "是" if bool(value) else "否"

    search_completed = sidecar_entry.get("search_completed", search.get("search_completed"))
    boundary_hit = sidecar_entry.get("boundary_hit", search.get("boundary_hit"))
    solver_converged = sidecar_entry.get(
        "solver_converged", search.get("solver_converged")
    )

    if search_completed is None or boundary_hit is None:
        config_path = _arm_config_path(Path(archive_dir), str(arm.get("run_id", "")))
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
        rows = _load_param_search_rows(Path(archive_dir), str(arm.get("run_id", "")))
        if not rows.empty:
            if search_completed is None:
                alpha_grid = config.get("elasticnet_config", {}).get("alpha_grid") or []
                l1_grid = config.get("elasticnet_config", {}).get("l1_ratio_grid") or []
                family = str(config.get("elasticnet_config", {}).get("model_family", "")).lower()
                l1_expected = 1 if family == "ridge" else max(int(len(l1_grid)), 1)
                expected = max(int(len(alpha_grid)), 1) * l1_expected
                search_completed = expected > 0 and len(rows) >= expected
            if boundary_hit is None:
                alphas = pd.to_numeric(rows["param_alpha"], errors="coerce").dropna()
                selected_alpha = float(arm.get("selected_alpha", float("nan")))
                boundary = bool(
                    alphas.size
                    and (
                        selected_alpha <= float(alphas.min())
                        or selected_alpha >= float(alphas.max())
                    )
                )
                l1_values = pd.to_numeric(
                    rows["param_l1_ratio"], errors="coerce"
                ).dropna()
                selected_l1 = arm.get("selected_l1_ratio")
                if selected_l1 is not None and not l1_values.empty:
                    boundary = boundary or bool(
                        float(selected_l1) <= float(l1_values.min())
                        or float(selected_l1) >= float(l1_values.max())
                    )
                boundary_hit = boundary

    if solver_converged is None:
        family = str(
            sidecar_entry.get("family", search.get("model_family", ""))
        ).lower()
        if family == "ridge":
            solver_label = "不适用（SVD 封闭解）"
        else:
            legacy_converged = search.get("converged")
            solver_label = (
                "未记录"
                if legacy_converged is None
                else stringify(legacy_converged, "未记录")
            )
    else:
        solver_label = stringify(solver_converged, "未记录")
    return (
        stringify(search_completed, "未记录"),
        stringify(boundary_hit, "未记录"),
        solver_label,
    )


def _load_param_search_rows(archive_dir: Path, run_id: str) -> pd.DataFrame:
    path = Path(archive_dir) / run_id / "param_search.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError):
        return pd.DataFrame()


def _spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=PRES_GRID, linewidth=0.8, alpha=0.85)


def _date_ticks(ax: plt.Axes, *, interval: int = 7) -> None:
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, interval)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(axis="x", linestyle=":", color=PRES_GRID, alpha=0.6)


def _common_y_limits(
    *arrays: Any,
    pad_lo: float = 0.10,
    pad_hi: float = 0.14,
) -> tuple[float, float]:
    """Compute a shared y-range that covers all finite values in arrays."""
    if not arrays:
        return 0.0, 1.0
    values = np.concatenate(
        [np.asarray(array, dtype=float).ravel() for array in arrays if np.asarray(array, dtype=float).size]
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = hi - lo
    if span <= 0:
        span = max(abs(hi) * 0.05, 1.0)
    return lo - span * pad_lo, hi + span * pad_hi


def _build_v3_test_models(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Merge archived test predictions into paired baseline/treatment frames."""
    frames = {item["run_id"]: item["frame"] for item in predictions}
    tested = {
        run_key: frame.loc[frame["split"].eq("test")].copy()
        for run_key, frame in frames.items()
        if "split" in frame.columns
    }
    missing = [
        run_key
        for run_key in (
            "ridge_baseline",
            "ridge_treatment",
            "elasticnet_baseline",
            "elasticnet_treatment",
        )
        if run_key not in tested
    ]
    if missing:
        raise ValueError(f"缺失测试段预测: {missing}")

    def merge_arm(baseline_key: str, treatment_key: str) -> pd.DataFrame:
        left = tested[baseline_key][
            ["sample_id", "issue_date", "y_true", "y_pred"]
        ].rename(columns={"y_pred": "y_base"})
        right = tested[treatment_key][
            ["sample_id", "issue_date", "y_true", "y_pred"]
        ].rename(columns={"y_pred": "y_treat"})
        combined = left.merge(right, on=["sample_id"], suffixes=("_b", "_t"))
        if (
            not np.allclose(combined["y_true_b"], combined["y_true_t"])
            or not np.allclose(
                pd.to_datetime(combined["issue_date_b"]).astype("int64"),
                pd.to_datetime(combined["issue_date_t"]).astype("int64"),
            )
        ):
            raise ValueError(f"{baseline_key}/{treatment_key} 样本未对齐")
        combined = combined.rename(
            columns={
                "y_true_b": "y_true",
                "issue_date_b": "issue_date",
            }
        )
        combined["issue_date"] = pd.to_datetime(combined["issue_date"])
        return combined.sort_values("issue_date").reset_index(drop=True)

    models = {
        "ridge": merge_arm("ridge_baseline", "ridge_treatment"),
        "elasticnet": merge_arm("elasticnet_baseline", "elasticnet_treatment"),
    }
    ridge_dates = models["ridge"]["issue_date"]
    elasticnet_dates = models["elasticnet"]["issue_date"]
    if not ridge_dates.reset_index(drop=True).equals(
        elasticnet_dates.reset_index(drop=True)
    ):
        raise ValueError("Ridge 与 ElasticNet 测试日期未对齐")
    return models


def plot_v3_gain_ci(
    metrics: Mapping[str, Any],
    output_path: str | Path,
    *,
    test_end: str | None = None,
) -> Path:
    """Horizontal RMSE-gain forest plot with the run's real bootstrap CI."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contrasts = {item["contrast_id"]: item for item in metrics["contrasts"]}
    runs = {item["run_id"]: item for item in metrics["runs"]}
    first_run = runs[min(runs, key=lambda key: list(runs).index(key))]
    horizon = int(first_run["parameters"]["forecast_horizon_days"])
    n_test = int(first_run.get("n_test", 0))
    test_start = str(metrics.get("test_start", ""))[:10]
    if test_end is None and n_test:
        test_end = (
            pd.Timestamp(test_start) + pd.Timedelta(days=n_test - 1)
        ).strftime("%Y-%m-%d")
    baseline = runs["ridge_baseline"]
    baseline_features = baseline.get("feature_columns", [])
    treatment_feature = "p2_y中心值"

    families = ["ridge", "elasticnet"]
    rows = []
    for family in families:
        contrast = contrasts[f"gain_{family}"]
        ci = contrast["bootstrap_ci"]["rmse_ci"]
        rows.append(
            {
                "family": family,
                "point": float(contrast["metrics_delta"]["rmse"]),
                "lo": float(ci[0]),
                "hi": float(ci[1]),
                "stable": bool(contrast["bootstrap_ci"]["stable"]),
            }
        )

    fig, (ax_main, ax_table) = plt.subplots(
        1,
        2,
        figsize=(12.0, 5.2),
        dpi=200,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    fig.patch.set_facecolor("white")

    ax_main.axvline(0.0, color=PRES_DARK, linewidth=1.1, zorder=1)
    y_pos = [1, 0]
    span_max = 0.0
    for row, y in zip(rows, y_pos):
        family = row["family"]
        color = FAMILY_COLORS[family]
        ax_main.plot(
            [row["lo"], row["hi"]],
            [y, y],
            color=color,
            linewidth=4,
            alpha=0.72,
            solid_capstyle="round",
            zorder=2,
        )
        ax_main.scatter(
            [row["point"]],
            [y],
            s=110,
            color=color,
            edgecolor="white",
            linewidth=1.4,
            zorder=5,
        )
        span_max = max(span_max, abs(row["lo"]), abs(row["hi"]))

    for row, y in zip(rows, y_pos):
        family = row["family"]
        label = _family_label(family)
        verdict = "CI 不含 0，支持日影增量" if row["stable"] else "CI 跨 0，证据不足"
        ax_main.text(
            row["hi"] + span_max * 0.05,
            y,
            (
                f"ΔRMSE {row['point']:+.2f} km/s\n"
                f"95% CI [{row['lo']:+.2f}, {row['hi']:+.2f}]\n"
                f"{verdict}"
            ),
            fontsize=10.5,
            color=PRES_DARK,
            va="center",
            ha="left",
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": PRES_GRID, "alpha": 0.95},
        )

    x_lo = -span_max * 0.22
    x_hi_ann = max(row["hi"] for row in rows) + span_max * 1.15
    ax_main.set_xlim(x_lo, x_hi_ann)
    ax_main.set_ylim(-0.55, 1.65)
    ax_main.set_yticks(y_pos)
    ax_main.set_yticklabels(
        [_family_label(row["family"]) for row in rows],
        fontsize=12,
    )
    ax_main.set_xlabel(
        "RMSE 降低量 (km/s)  ·  正值 = 加入日影后迭代评估段 RMSE 更低",
        fontsize=11.5,
        color=PRES_DARK,
    )
    ax_main.set_title(
        "加入日影后，迭代评估段 RMSE 改善多少？",
        fontsize=15,
        color=PRES_DARK,
        loc="left",
        pad=10,
    )
    _spines(ax_main)
    ax_main.set_axisbelow(True)

    ax_table.axis("off")
    ax_table.set_title(
        "迭代评估段绝对表现（dev_backtest）",
        fontsize=12.5,
        color=PRES_DARK,
        loc="left",
        pad=10,
    )
    table_rows = []
    for run in metrics["runs"]:
        family = _family_label(run.get("model_family", ""))
        suffix = " 基线" if run["run_id"].endswith("_baseline") else " 实验组"
        run_metrics = run["metrics"]
        table_rows.append(
            [
                f"{family}{suffix}",
                f"{float(run_metrics['rmse']):.2f}",
                f"{float(run_metrics['pearson_r']):.2f}",
                f"{float(run_metrics['r2']):.2f}",
            ]
        )
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["模型 / 臂", "RMSE", "r", "R²"],
        cellLoc="left",
        colLoc="center",
        loc="center",
        colWidths=[0.26, 0.16, 0.14, 0.14],
        bbox=[0.02, 0.26, 0.96, 0.52],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.45)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(PRES_GRID)
        if row_idx == 0:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(color=PRES_DARK, fontweight="bold")
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color=PRES_DARK)
    ax_table.text(
        0.02,
        0.10,
        "绝对 RMSE 仍明显偏高，说明当前线性模型整体捕捉能力有限；"
        "增量图只回答“加日影是否改善”。",
        fontsize=9.5,
        color=PRES_MUTED,
        transform=ax_table.transAxes,
        ha="left",
        linespacing=1.4,
    )

    caption = (
        f"运行 E_R02_03 · 迭代评估段 {test_start} 至 {test_end} · n={n_test} 个共同迭代评估目标日 · "
        f"预测提前 {horizon} 天 · 基线 = {'、'.join(baseline_features)} · "
        f"实验组 = 基线 + {treatment_feature} · 仅同模型家族内比较 · "
        f"95% CI 来自配对块 bootstrap（7 天块，4000 次重复，seed=20261008），非模拟区间。"
    )
    fig.text(
        0.02,
        0.015,
        caption,
        fontsize=9,
        color=PRES_MUTED,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_v3_test_prediction_comparison(
    predictions: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    test_start: str | None = None,
    test_end: str | None = None,
    horizon_days: int = 3,
    run_id: str = "E_R02_03",
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    """Continuous test-period line chart for both model families."""

    models = _build_v3_test_models(predictions)

    y_lo, y_hi = _common_y_limits(
        models["ridge"]["y_true"],
        models["ridge"]["y_base"],
        models["ridge"]["y_treat"],
        models["elasticnet"]["y_true"],
        models["elasticnet"]["y_base"],
        models["elasticnet"]["y_treat"],
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.0, 8.2),
        dpi=200,
        constrained_layout=True,
        sharex=True,
        height_ratios=[1, 1],
        gridspec_kw={"hspace": 0.20},
    )
    fig.patch.set_facecolor("white")

    legend_handles = [
        Line2D(
            [0], [0], color=PRES_TRUTH, linestyle="-", linewidth=1.6,
            marker="o", markersize=4,
            markerfacecolor=PRES_TRUTH, markeredgecolor=PRES_TRUTH,
        ),
        Line2D(
            [0], [0], color=PRES_BLUE, linestyle="--", linewidth=1.7,
        ),
        Line2D(
            [0], [0], color=PRES_ORANGE, linestyle="-", linewidth=1.7,
        ),
    ]
    fig.legend(
        legend_handles,
        ["真实太阳风速度", "基线预测", "加入日影后预测"],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        fontsize=11,
    )

    run_metrics = {
        item["run_id"]: item.get("metrics", {})
        for item in (metrics.get("runs", []) if metrics else [])
    }

    def arm_rmse_r(
        run_id: str, frame: pd.DataFrame, pred_col: str
    ) -> tuple[float, float]:
        source = run_metrics.get(run_id)
        if source:
            rmse = float(source.get("rmse", float("nan")))
            pearson = float(source.get("pearson_r", float("nan")))
            if np.isfinite(rmse):
                return rmse, pearson
        valid = frame[["y_true", pred_col]].dropna()
        if len(valid) == 0:
            return float("nan"), float("nan")
        rmse = float(np.sqrt(np.mean((valid["y_true"] - valid[pred_col]) ** 2)))
        if (
            len(valid) < 2
            or float(np.std(valid["y_true"])) == 0
            or float(np.std(valid[pred_col])) == 0
        ):
            return rmse, float("nan")
        pearson = float(
            np.corrcoef(valid["y_true"], valid[pred_col])[0, 1]
        )
        return rmse, pearson

    baseline_features: list[str] = []
    for run in (metrics.get("runs", []) if metrics else []):
        if run.get("run_id") == "ridge_baseline":
            baseline_features = run.get("feature_columns", [])
            break

    for index, (family, frame) in enumerate(models.items()):
        ax = axes[index]
        dates = frame["issue_date"]
        ax.plot(
            dates, frame["y_true"], color=PRES_TRUTH,
            linewidth=1.3, marker="o", markersize=3.4,
            label="真实太阳风速度", zorder=5,
        )
        ax.plot(
            dates, frame["y_base"], color=PRES_BLUE,
            linestyle="--", linewidth=1.5,
            label="基线预测", zorder=4,
        )
        ax.plot(
            dates, frame["y_treat"], color=PRES_ORANGE,
            linewidth=1.5,
            label="加入日影后预测", zorder=4,
        )
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("太阳风速度 (km/s)", fontsize=11.5, color=PRES_DARK)
        ax.set_title(
            f"{_family_label(family)}：迭代评估段预测与真实值",
            fontsize=14,
            color=PRES_DARK,
            loc="left",
            pad=6,
        )
        _spines(ax)
        _date_ticks(ax, interval=5)

        base_rmse, base_r = arm_rmse_r(
            f"{family}_baseline", frame, "y_base"
        )
        treat_rmse, treat_r = arm_rmse_r(
            f"{family}_treatment", frame, "y_treat"
        )
        metric_text = (
            f"基线 RMSE {base_rmse:.2f} · r = {base_r:+.2f}\n"
            f"日影 RMSE {treat_rmse:.2f} · r = {treat_r:+.2f}"
        )
        ax.text(
            0.03,
            0.97,
            metric_text,
            transform=ax.transAxes,
            fontsize=9.5,
            color=PRES_DARK,
            ha="left",
            va="top",
            linespacing=1.45,
            bbox={
                "boxstyle": "round,pad=0.4",
                "fc": "white",
                "ec": PRES_GRID,
                "alpha": 0.94,
            },
            zorder=8,
        )

    first_dates = models["ridge"]["issue_date"]
    test_start_text = (
        test_start
        if test_start is not None
        else first_dates.min().strftime("%Y-%m-%d")
    )
    test_end_text = (
        test_end
        if test_end is not None
        else first_dates.max().strftime("%Y-%m-%d")
    )
    start_ts = pd.Timestamp(test_start_text)
    end_ts = pd.Timestamp(test_end_text)
    for ax in axes:
        ax.set_xlim(start_ts - pd.Timedelta(days=0.5), end_ts + pd.Timedelta(days=0.5))
        ax.tick_params(axis="x", labelbottom=False)
    axes[-1].tick_params(axis="x", labelbottom=True)
    axes[-1].set_xlabel("预测发布时间 (issue date)", fontsize=11, color=PRES_DARK)

    baseline_label = "、".join(
        baseline_features or ["ss_br_mean", "BX, nT (GSE, GSM)"]
    )
    caption = (
        f"运行 {run_id} · {test_start_text} 至 {test_end_text} · "
        f"n={len(first_dates)} 个连续共同迭代评估日，无缺失 · "
        f"预测提前 {horizon_days} 天 · 基线 = {baseline_label} · "
        f"实验组 = 基线 + {SHADOW_FEATURE} · 按日期直接连线，未平滑、未插值、未改日期。"
    )
    fig.text(
        0.02,
        0.015,
        caption,
        fontsize=9,
        color=PRES_MUTED,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_v3_daily_abs_error_delta(
    predictions: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    test_start: str | None = None,
    test_end: str | None = None,
    horizon_days: int = 3,
    run_id: str = "E_R02_03",
) -> Path:
    """Daily |error| change between baseline and shadow treatment."""

    models = _build_v3_test_models(predictions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    deltas = {
        family: (
            (frame["y_true"] - frame["y_base"]).abs()
            - (frame["y_true"] - frame["y_treat"]).abs()
        ).reset_index(drop=True)
        for family, frame in models.items()
    }
    all_delta = pd.concat([deltas["ridge"], deltas["elasticnet"]])
    max_abs = float(all_delta.abs().max())
    if max_abs <= 0:
        max_abs = 1.0
    y_hi = max_abs * 1.40
    y_lo = -max_abs * 0.95

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.0, 8.2),
        dpi=200,
        constrained_layout=True,
        sharex=True,
        height_ratios=[1, 1],
        gridspec_kw={"hspace": 0.22},
    )
    fig.patch.set_facecolor("white")

    legend_handles = [
        Patch(facecolor=PRES_BLUE, alpha=0.88, label="当日误差改善（日影更小）"),
        Patch(facecolor=PRES_ORANGE, alpha=0.88, label="当日误差变差（日影更大）"),
    ]
    fig.legend(
        legend_handles,
        ["当日误差改善（日影更小）", "当日误差变差（日影更大）"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        fontsize=11,
    )

    first_dates = models["ridge"]["issue_date"]
    test_start_text = (
        test_start
        if test_start is not None
        else first_dates.min().strftime("%Y-%m-%d")
    )
    test_end_text = (
        test_end
        if test_end is not None
        else first_dates.max().strftime("%Y-%m-%d")
    )
    start_ts = pd.Timestamp(test_start_text)
    end_ts = pd.Timestamp(test_end_text)

    for index, (family, frame) in enumerate(models.items()):
        ax = axes[index]
        dates = frame["issue_date"]
        delta = deltas[family]
        colors = [
            PRES_BLUE if value >= 0 else PRES_ORANGE for value in delta.tolist()
        ]
        ax.bar(
            dates,
            delta,
            width=0.72,
            color=colors,
            alpha=0.88,
            zorder=3,
        )
        ax.axhline(0.0, color=PRES_DARK, linewidth=1.0, zorder=2)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel(
            "|基线误差| - |日影误差| (km/s)",
            fontsize=11.5,
            color=PRES_DARK,
        )
        ax.set_title(
            f"{_family_label(family)}：逐日绝对误差变化",
            fontsize=14,
            color=PRES_DARK,
            loc="left",
            pad=6,
        )
        _spines(ax)
        _date_ticks(ax, interval=5)
        improved = int((delta > 0).sum())
        worsened = int((delta < 0).sum())
        same = int((delta == 0).sum())
        ax.text(
            0.03,
            0.97,
            (
                f"改善 {improved} 天 · 变差 {worsened} 天 · 持平 {same} 天\n"
                f"日改善量合计 {delta.sum():+.1f} km/s"
            ),
            transform=ax.transAxes,
            fontsize=9.5,
            color=PRES_DARK,
            ha="left",
            va="top",
            linespacing=1.45,
            bbox={
                "boxstyle": "round,pad=0.40",
                "fc": "white",
                "ec": PRES_GRID,
                "alpha": 0.94,
            },
            zorder=8,
        )
        ax.set_xlim(start_ts - pd.Timedelta(days=0.5), end_ts + pd.Timedelta(days=0.5))
        ax.tick_params(axis="x", labelbottom=False)

    axes[-1].tick_params(axis="x", labelbottom=True)
    axes[-1].set_xlabel("预测发布时间 (issue date)", fontsize=11, color=PRES_DARK)
    caption = (
        f"运行 {run_id} · {test_start_text} 至 {test_end_text} · "
        f"n={len(first_dates)} 个连续共同迭代评估日，无缺失 · "
        f"预测提前 {horizon_days} 天 · "
        f"逐日指标 = |实际-基线| − |实际-日影|，正值表示加入日影后当日绝对误差更小 · "
        f"未平滑、未插值、未改日期。"
    )
    fig.text(
        0.02,
        0.015,
        caption,
        fontsize=9,
        color=PRES_MUTED,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def compose_v3_combined_overview(
    figures: Sequence[tuple[str, Path]],
    output_path: str | Path,
    *,
    run_id: str = "E_R02_03",
    test_start: str | None = None,
    test_end: str | None = None,
) -> Path:
    """Compose the core presentation figures into one 2x2 overview sheet."""
    if len(figures) != 4:
        raise ValueError("组合版需要 4 张图")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16.0, 10.6),
        dpi=200,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    for ax, (label, image_path) in zip(axes.flat, figures):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"组合图源不存在: {image_path}")
        ax.imshow(plt.imread(image_path), aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=12.5, color=PRES_DARK, loc="left", pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    period = (
        f"{test_start} 至 {test_end}" if test_start and test_end else "迭代评估段"
    )
    fig.suptitle(
        f"逐影 v3 图包组合版 · {run_id} · {period}",
        fontsize=15,
        color=PRES_DARK,
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.012,
        (
            "A = 迭代评估段 RMSE 增益与 95% CI；B = 逐日绝对误差变化；"
            "C = 迭代评估段预测对照；D = 完整日历经览。"
            "全部图来自同一运行，未平滑、未插值、未压缩日期轴。"
        ),
        fontsize=9,
        color=PRES_MUTED,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_v3_appendix_full_calendar_overview(
    calendar_frame: pd.DataFrame,
    predictions: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    test_start: str | None = None,
    test_end: str | None = None,
    horizon_days: int = 3,
    run_id: str = "E_R02_03",
) -> Path:
    """Appendix overview: full 196-day calendar with train/test ranges marked."""

    frames = {item["run_id"]: item["frame"] for item in predictions}
    required = (
        "ridge_baseline",
        "ridge_treatment",
        "elasticnet_baseline",
        "elasticnet_treatment",
    )
    missing = [key for key in required if key not in frames]
    if missing:
        raise ValueError(f"缺失完整日历预测: {missing}")

    test_frame = frames["ridge_baseline"].loc[
        frames["ridge_baseline"]["split"].eq("test")
    ]
    if test_start is None:
        test_start = test_frame["issue_date"].min().strftime("%Y-%m-%d")
    if test_end is None:
        test_end = test_frame["issue_date"].max().strftime("%Y-%m-%d")
    start_ts = pd.Timestamp(test_start)
    end_ts = pd.Timestamp(test_end)

    calendar = calendar_frame.sort_values("_date").reset_index(drop=True)
    truth = calendar[calendar[TARGET_FEATURE].notna()].copy()
    all_values = pd.concat(
        [
            calendar.get(TARGET_FEATURE),
            *[
                frames[run_key]["y_true"]
                for run_key in required
                if "y_true" in frames[run_key].columns
            ],
            *[
                frames[run_key]["y_pred"]
                for run_key in required
                if "y_pred" in frames[run_key].columns
            ],
        ]
    )
    y_lo, y_hi = _common_y_limits(
        all_values,
        pad_lo=0.07,
        pad_hi=0.09,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.0, 8.4),
        dpi=200,
        constrained_layout=True,
        sharex=True,
        height_ratios=[1, 1],
        gridspec_kw={"hspace": 0.20},
    )
    fig.patch.set_facecolor("white")

    legend_handles = [
        Line2D(
            [0], [0], color=PRES_TRUTH, linestyle="-", linewidth=1.2,
            marker="o", markersize=2.4,
            markerfacecolor=PRES_TRUTH, markeredgecolor=PRES_TRUTH,
        ),
        Line2D([0], [0], color=PRES_BLUE, linestyle="--", linewidth=1.2, alpha=0.55),
        Line2D([0], [0], color=PRES_ORANGE, linestyle="-", linewidth=1.1, alpha=0.55),
        Line2D([0], [0], color=PRES_BLUE, linestyle="--", linewidth=1.7),
        Line2D([0], [0], color=PRES_ORANGE, linestyle="-", linewidth=1.7),
    ]
    fig.legend(
        legend_handles,
        [
            "真实太阳风速度",
            "基线训练拟合（样本内）",
            "加入日影训练拟合（样本内）",
            "基线迭代评估预测",
            "加入日影迭代评估预测",
        ],
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        fontsize=10.5,
    )

    counts: dict[str, dict[str, int]] = {}
    for index, (family, baseline_key, treatment_key) in enumerate(
        (
            ("ridge", "ridge_baseline", "ridge_treatment"),
            ("elasticnet", "elasticnet_baseline", "elasticnet_treatment"),
        )
    ):
        ax = axes[index]
        ax.plot(
            truth["_date"],
            truth[TARGET_FEATURE],
            color=PRES_TRUTH,
            linewidth=1.1,
            marker="o",
            markersize=2.2,
            zorder=5,
        )
        train_base = frames[baseline_key].loc[
            frames[baseline_key]["split"].eq("train")
        ]
        train_treat = frames[treatment_key].loc[
            frames[treatment_key]["split"].eq("train")
        ]
        test_base = frames[baseline_key].loc[
            frames[baseline_key]["split"].eq("test")
        ]
        test_treat = frames[treatment_key].loc[
            frames[treatment_key]["split"].eq("test")
        ]
        counts[baseline_key] = {
            "train": int(train_base["split"].count()),
            "test": int(test_base["split"].count()),
            "boundary": int(
                frames[baseline_key]["split"].eq("boundary_excluded").sum()
            ),
        }
        ax.plot(
            train_base["issue_date"],
            train_base["y_pred"],
            color=PRES_BLUE,
            linestyle="--",
            linewidth=1.2,
            alpha=0.55,
            zorder=3,
        )
        ax.plot(
            train_treat["issue_date"],
            train_treat["y_pred"],
            color=PRES_ORANGE,
            linewidth=1.1,
            alpha=0.55,
            zorder=3,
        )
        ax.plot(
            test_base["issue_date"],
            test_base["y_pred"],
            color=PRES_BLUE,
            linestyle="--",
            linewidth=1.7,
            zorder=4,
        )
        ax.plot(
            test_treat["issue_date"],
            test_treat["y_pred"],
            color=PRES_ORANGE,
            linewidth=1.7,
            zorder=4,
        )
        ax.axvspan(start_ts, end_ts, color=PRES_BLUE, alpha=0.055, zorder=0)
        ax.axvline(start_ts, color=PRES_MUTED, linestyle=":", linewidth=1.0, zorder=1)
        ax.axvline(end_ts, color=PRES_MUTED, linestyle=":", linewidth=1.0, zorder=1)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("太阳风速度 (km/s)", fontsize=11.5, color=PRES_DARK)
        ax.set_title(
            f"{_family_label(family)}：完整日历概览",
            fontsize=14,
            color=PRES_DARK,
            loc="left",
            pad=6,
        )
        _spines(ax)
        _date_ticks(ax, interval=14)
        ax.tick_params(axis="x", labelbottom=False)
        ax.text(
            0.03,
            0.97,
            "阴影 = 迭代评估范围，宽线 = 迭代评估预测，浅色虚线 = 训练拟合",
            transform=ax.transAxes,
            fontsize=9,
            color=PRES_MUTED,
            ha="left",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.35",
                "fc": "white",
                "ec": PRES_GRID,
                "alpha": 0.94,
            },
        )

    axes[-1].tick_params(axis="x", labelbottom=True)
    axes[-1].set_xlabel(
        "2021 年完整日历 3 月 21 日至 10 月 2 日",
        fontsize=11,
        color=PRES_DARK,
    )
    first_counts = counts[required[0]]
    caption = (
        f"运行 {run_id} · 完整日历 {len(calendar)} 天 · 训练 {first_counts['train']} / "
        f"迭代评估 {first_counts['test']} / 边界排除 {first_counts['boundary']} · "
        f"迭代评估 {test_start} 至 {test_end} · 预测提前 {horizon_days} 天 · "
        f"训练段为重建的样本内拟合，迭代评估段为预测 · 未平滑、未插值、未压缩日期轴。"
    )
    fig.text(
        0.02,
        0.015,
        caption,
        fontsize=9,
        color=PRES_MUTED,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_v3_appendix_training_quality(
    calendar_frame: pd.DataFrame,
    predictions: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    stats: Mapping[str, Any] | None = None,
    manifest: pd.DataFrame | None = None,
) -> Path:
    """Appendix: training fit plus the full-calendar data-quality strip."""

    frames = {
        item["run_id"]: item["frame"]
        for item in predictions
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.0, 6.9), dpi=200, constrained_layout=True)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.0, 0.78],
        hspace=0.42,
        wspace=0.14,
    )

    top_lo = 300.0
    top_hi = 500.0
    for family, baseline_key, treatment_key in (
        ("ridge", "ridge_baseline", "ridge_treatment"),
        ("elasticnet", "elasticnet_baseline", "elasticnet_treatment"),
    ):
        train_base = frames[baseline_key].loc[frames[baseline_key]["split"].eq("train")]
        train_treat = frames[treatment_key].loc[frames[treatment_key]["split"].eq("train")]
        all_values = pd.concat(
            [
                train_base["y_true"],
                train_base["y_pred"],
                train_treat["y_true"],
                train_treat["y_pred"],
            ]
        )
        top_lo = min(top_lo, float(all_values.min()))
        top_hi = max(top_hi, float(all_values.max()))

    for index, (family, baseline_key, treatment_key) in enumerate(
        (
            ("ridge", "ridge_baseline", "ridge_treatment"),
            ("elasticnet", "elasticnet_baseline", "elasticnet_treatment"),
        )
    ):
        color = FAMILY_COLORS[family]
        ax = fig.add_subplot(grid[0, index])
        train_base = frames[baseline_key].loc[frames[baseline_key]["split"].eq("train")]
        train_treat = frames[treatment_key].loc[frames[treatment_key]["split"].eq("train")]
        ax.scatter(
            train_base["_date"], train_base["y_true"],
            s=18, color=PRES_TRUTH, alpha=0.85, label="真实太阳风速度", zorder=5,
        )
        ax.scatter(
            train_base["_date"], train_base["y_pred"],
            s=24, facecolor="none", edgecolor=PRES_BLUE,
            linewidths=1.0, label="基线拟合", zorder=4,
        )
        ax.scatter(
            train_treat["_date"], train_treat["y_pred"],
            s=28, marker="x", color=PRES_ORANGE,
            linewidths=1.2, label="加入日影后拟合", zorder=4,
        )
        ax.set_ylim(top_lo - 8, top_hi + 8)
        ax.set_ylabel("太阳风速度 (km/s)", fontsize=11, color=PRES_DARK)
        ax.set_title(
            f"{_family_label(family)}：训练期拟合（139 个样本）",
            fontsize=13,
            color=PRES_DARK,
            loc="left",
            pad=6,
        )
        _spines(ax)
        _date_ticks(ax, interval=12)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9, ncol=1)

    calendar = calendar_frame.sort_values("_date").reset_index(drop=True)
    if manifest is not None and not manifest.empty:
        calendar = calendar.merge(
            manifest[["issue_date", "sample_group"]],
            left_on="_date",
            right_on="issue_date",
            how="left",
            sort=False,
        ).fillna({"sample_group": "未覆盖"})
    strip = fig.add_subplot(grid[1, :])
    row_specs = [("focus_status", SHADOW_STATUS_COLORS, "日影质量")]
    if "sample_group" in calendar.columns:
        row_specs.append(("sample_group", SAMPLE_STATUS_COLORS, "评价样本"))
    legend_handles = []
    for row_idx, (status_col, colors, label) in enumerate(row_specs):
        values = calendar[status_col].astype(str)
        for status, color in colors.items():
            subset = calendar[values.eq(status)]
            if subset.empty:
                continue
            strip.bar(
                subset["_date"],
                0.8,
                bottom=row_idx * 1.0,
                width=0.85,
                color=color,
                alpha=0.95,
                linewidth=0,
            )
            legend_handles.append(
                Line2D(
                    [0], [0], marker="s", color="none", markersize=9,
                    markerfacecolor=color, markeredgecolor=color,
                    label=status,
                )
            )
    strip.set_yticks([0.5 if len(row_specs) == 1 else 0.5, 1.5] if len(row_specs) == 2 else [0.5])
    strip.set_yticklabels([spec[2] for spec in row_specs], fontsize=9.5, color=PRES_DARK)
    strip.set_ylim(-0.25, len(row_specs) + 0.05)
    strip.set_title(
        "完整日历数据质量与样本分段（附录审计）",
        fontsize=13,
        color=PRES_DARK,
        loc="left",
        pad=6,
    )
    _spines(strip)
    _date_ticks(strip, interval=12)
    strip.set_xlabel("2021 年 3 月 21 日至 10 月 2 日，按真实日历", fontsize=10, color=PRES_MUTED)
    strip.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=4 if len(legend_handles) <= 4 else 5,
        fontsize=9,
        frameon=False,
    )
    if stats:
        shadow = stats.get("shadow", {})
        sample = stats.get("sample", {})
        caption = (
            f"完整日历 {stats.get('calendar_days', 0)} 天 · 日影有效观测 {shadow.get('valid_days', 0)} 天，"
            f"原表缺失 {shadow.get('source_missing_days', 0)} 天，质量不可用 "
            f"{shadow.get('quality_unavailable_days', 0)} 天 · 共同评价样本 "
            f"{sample.get('retained_count', 0)} 个，训练 {sample.get('split_counts', {}).get('train', 0)} / "
            f"迭代评估 {sample.get('split_counts', {}).get('test', 0)}。"
        )
        fig.text(
            0.02,
            0.015,
            caption,
            fontsize=9,
            color=PRES_MUTED,
        )
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_v3_parameters_annex(
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    archive_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Render a readable run-parameter annex table."""

    archive_dir = Path(archive_dir)
    arms = diagnostics.get("arms", {})
    rows = []
    for run in metrics["runs"]:
        run_id = run["run_id"]
        arm = arms.get(run_id, {})
        search = arm.get("param_search", {})
        search_completed, boundary_hit, solver_converged = _resolve_param_flags(
            {**arm, "run_id": run_id},
            archive_dir,
        )
        rows.append(
            [
                run_id,
                _family_label(run.get("model_family", run_id)),
                f"{float(arm.get('selected_alpha', float(run['parameters'].get('alpha', 0)))):g}",
                (
                    "—"
                    if not arm.get("selected_l1_ratio")
                    else f"{float(arm['selected_l1_ratio']):g}"
                ),
                "auto" if search.get("search_mode") == "auto" else "fixed",
                search_completed,
                boundary_hit,
                solver_converged,
                f"{int(arm.get('n_train', run.get('n_train', 0)))}",
                f"{int(arm.get('n_test', run.get('n_test', 0)))}",
                "、".join(run.get("feature_columns", [])),
            ]
        )

    label = [
        "run_id",
        "模型家族",
        "alpha",
        "l1_ratio",
        "选参模式",
        "搜索完整",
        "边界命中",
        "求解器收敛",
        "n_train",
        "n_test",
        "特征列",
    ]
    csv_path = output_path.with_suffix(".csv")
    table = pd.DataFrame(rows, columns=label)
    csv_path.write_text(table.to_csv(index=False), encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12.0, 4.2), dpi=200, constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(
        "运行参数附录（E_R02_03 四臂）",
        fontsize=13,
        color=PRES_DARK,
        loc="left",
    )
    tbl = ax.table(
        cellText=rows,
        colLabels=label,
        cellLoc="left",
        colLoc="left",
        loc="center",
        bbox=[0.02, 0.02, 0.96, 0.82],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor(PRES_GRID)
        cell.set_text_props(color=PRES_DARK)
        if row_idx == 0:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
