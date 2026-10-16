"""Calendar-faithful figure builders for the v3 shadow-tracing data.

The figures in this module intentionally keep the full calendar axis (no
compression, no interpolation of missing observations) and annotate the data
quality strip below the data so missing/quality-unavailable dates stay
auditable at a glance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Microsoft JhengHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

TIME_COL = "TIME"
SHADOW_CENTER_COL = "p2_y中心值"
SHADOW_ERR_COL = "p2_y中心值误差"
TARGET_COL = "SW Plasma Speed, km/s"
SOURCE_PROBE_COL = "photo_br_mean"
QUAILITY_CHI2_COL = "_llf 最优卡方值"

SHADOW_SERIES_COLOR = "#1f6f8b"
SHADOW_SERIES_ERR_COLOR = "#6ba3b5"

TRUTH_STYLE = {"color": "#334155", "marker": "o", "size": 20, "alpha": 0.95}
BASELINE_STYLE = {"color": "#1d4ed8", "marker": "o", "size": 20, "alpha": 0.90}
TREATMENT_STYLE = {"color": "#ea580c", "marker": "o", "size": 20, "alpha": 0.90}

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

DROP_REASON_LABELS = {
    "history_gap": "历史缺口",
    "feature_missing": "历史特征缺失",
    "target_unavailable": "目标日不可用",
}


def _parse_time(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(
        series.astype(str).str.strip(),
        format="%Y%m%d",
        errors="coerce",
    )
    if values.isna().all():
        values = pd.to_datetime(series, errors="coerce")
    return pd.Series(pd.DatetimeIndex(values).normalize(), index=series.index)


def _finite(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.apply(lambda value: bool(np.isfinite(value)))


def load_calendar_frame(
    omni_path: str | Path,
    lhaaso_path: str | Path,
    *,
    time_col: str = TIME_COL,
    shadow_center_col: str = SHADOW_CENTER_COL,
    shadow_err_col: str = SHADOW_ERR_COL,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Merge v3 raw sources onto the full 2021 calendar without dropping rows."""

    omni = pd.read_csv(omni_path)
    lhaaso = pd.read_csv(lhaaso_path)
    omni["_date"] = _parse_time(omni[time_col])
    lhaaso["_date"] = _parse_time(lhaaso[time_col])
    omni = omni.dropna(subset=["_date"]).drop_duplicates("_date", keep="first")
    lhaaso = lhaaso.dropna(subset=["_date"]).drop_duplicates("_date", keep="first")

    start = min(omni["_date"].min(), lhaaso["_date"].min())
    end = max(omni["_date"].max(), lhaaso["_date"].max())
    calendar = pd.DataFrame({"_date": pd.date_range(start, end, freq="D")})

    omni_cols = [
        column
        for column in omni.columns
        if column != time_col and column != "_date"
    ]
    lha_cols = [
        column
        for column in lhaaso.columns
        if column != time_col and column != "_date"
    ]
    frame = calendar.merge(
        omni[["_date", *omni_cols]],
        on="_date",
        how="left",
    )
    frame = frame.merge(
        lhaaso[["_date", *lha_cols]],
        on="_date",
        how="left",
        suffixes=("_omni", "_lhaaso"),
    )

    source_probe = (
        pd.to_numeric(frame.get(SOURCE_PROBE_COL), errors="coerce")
        if SOURCE_PROBE_COL in frame.columns
        else pd.Series(np.nan, index=frame.index)
    )
    center_ok = _finite(frame[shadow_center_col]) if shadow_center_col in frame.columns else pd.Series(False, index=frame.index)
    err_ok = _finite(frame[shadow_err_col]) if shadow_err_col in frame.columns else pd.Series(False, index=frame.index)
    center_present = frame[shadow_center_col].notna() if shadow_center_col in frame.columns else pd.Series(False, index=frame.index)
    err_present = frame[shadow_err_col].notna() if shadow_err_col in frame.columns else pd.Series(False, index=frame.index)
    quality_chi2_ok = (
        _finite(frame[QUAILITY_CHI2_COL])
        if QUAILITY_CHI2_COL in frame.columns
        else pd.Series(True, index=frame.index)
    )
    lha_row_present = frame["_date"].isin(set(lhaaso["_date"]))
    focus_status = pd.Series("原表缺失", index=frame.index, dtype=object)
    focus_status.loc[lha_row_present & center_ok & err_ok] = "有效观测"
    focus_status.loc[
        lha_row_present & (center_present | err_present) & ~(center_ok & err_ok)
    ] = "质量不可用"
    focus_status.loc[
        lha_row_present & center_ok & err_ok & ~quality_chi2_ok
    ] = "质量不可用"

    target_values = pd.to_numeric(frame[target_col], errors="coerce")
    target_status = pd.Series("原表缺失", index=frame.index, dtype=object)
    target_status.loc[target_values.notna() & _finite(frame[target_col])] = "有效观测"
    target_status.loc[target_values.notna() & ~_finite(frame[target_col])] = "质量不可用"

    frame["focus_status"] = focus_status
    frame["target_status"] = target_status
    frame["focus_valid"] = frame["focus_status"] == "有效观测"
    frame["target_valid"] = frame["target_status"] == "有效观测"
    return frame


def load_sample_manifest(archive_dir: str | Path) -> pd.DataFrame:
    manifest_path = Path(archive_dir) / "sample_manifest.csv"
    if not manifest_path.exists():
        return pd.DataFrame(columns=["issue_date", "split", "drop_reason"])
    manifest = pd.read_csv(manifest_path)
    manifest["issue_date"] = _parse_time(manifest["issue_date"])
    if "drop_reason" not in manifest.columns:
        manifest["drop_reason"] = None
    manifest["sample_group"] = "未覆盖"
    manifest.loc[manifest["drop_reason"].notna(), "sample_group"] = "滞后构造排除"
    manifest.loc[manifest["drop_reason"].isna() & (manifest["split"] == "train"), "sample_group"] = "训练样本"
    manifest.loc[manifest["drop_reason"].isna() & (manifest["split"] == "test"), "sample_group"] = "迭代评估样本"
    manifest.loc[manifest["drop_reason"].isna() & (manifest["split"] == "boundary_excluded"), "sample_group"] = "边界排除"
    return manifest


def compute_quality_stats(
    calendar_frame: pd.DataFrame,
    manifest: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return the audit counts shown on the calendar-faithful figures."""

    shadow = calendar_frame["focus_status"].value_counts().to_dict()
    target = calendar_frame["target_status"].value_counts().to_dict()
    stats: dict[str, Any] = {
        "calendar_days": int(len(calendar_frame)),
        "date_start": calendar_frame["_date"].min().date().isoformat(),
        "date_end": calendar_frame["_date"].max().date().isoformat(),
        "shadow": {
            "valid_days": int(shadow.get("有效观测", 0)),
            "source_missing_days": int(shadow.get("原表缺失", 0)),
            "quality_unavailable_days": int(shadow.get("质量不可用", 0)),
        },
        "target": {
            "valid_days": int(target.get("有效观测", 0)),
            "source_missing_days": int(target.get("原表缺失", 0)),
            "quality_unavailable_days": int(target.get("质量不可用", 0)),
        },
        "sample": {
            "candidate_count": 0,
            "retained_count": 0,
            "drop_reasons": {},
            "split_counts": {},
        },
    }
    if manifest is None or manifest.empty:
        return stats
    drop_counts = manifest["drop_reason"].value_counts(dropna=False).to_dict()
    drop_counts.pop(np.nan, None)
    stats["sample"]["candidate_count"] = int(len(manifest))
    stats["sample"]["retained_count"] = int(manifest["drop_reason"].isna().sum())
    stats["sample"]["drop_reasons"] = {
        str(DROP_REASON_LABELS.get(key, key)): int(value)
        for key, value in drop_counts.items()
    }
    raw_missing_dates = set(
        calendar_frame.loc[
            calendar_frame["focus_status"] == "原表缺失",
            "_date",
        ].dt.normalize()
    )
    feature_dropped = manifest[
        manifest["drop_reason"].eq("feature_missing")
    ]
    source_missing_issue_days = int(
        feature_dropped["issue_date"].dt.normalize().isin(raw_missing_dates).sum()
    )
    stats["sample"]["drop_detail"] = {
        "source_missing_issue_days": source_missing_issue_days,
        "lag_induced_issue_days": int(len(feature_dropped) - source_missing_issue_days),
        "history_gap_days": int(
            manifest["drop_reason"].eq("history_gap").sum()
        ),
        "target_unavailable_days": int(
            manifest["drop_reason"].eq("target_unavailable").sum()
        ),
    }
    stats["sample"]["split_counts"] = {
        str(key): int(value)
        for key, value in manifest["split"].value_counts().items()
    }
    return stats


def _draw_status_strip(
    ax: plt.Axes,
    calendar_frame: pd.DataFrame,
    *,
    status_col: str,
    colors: Mapping[str, str],
    y_bottom: float,
    height: float,
    label: str,
    add_legend: bool = True,
) -> None:
    values = calendar_frame[status_col].astype(str)
    for status, color in colors.items():
        subset = calendar_frame[values == status]
        if subset.empty:
            continue
        ax.bar(
            subset["_date"],
            height,
            bottom=y_bottom,
            width=0.85,
            color=color,
            alpha=0.95,
            linewidth=0,
        )
    ax.text(
        calendar_frame["_date"].iloc[0] + pd.Timedelta(days=1),
        y_bottom + height * 1.5,
        label,
        fontsize=8,
        color="#2d3748",
        va="bottom",
    )
    if add_legend:
        ax.scatter([], [], marker="s", s=26, c="#64748b", linewidths=0, label=label)


def _force_full_calendar_axis(
    ax: plt.Axes,
    dates: pd.Series,
    *,
    full_range: bool = False,
) -> None:
    if full_range and not dates.empty:
        ax.set_xlim(dates.iloc[0], dates.iloc[-1])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="x", linestyle=":", color="#cbd5e1", alpha=0.7)


def _stats_footer(stats: Mapping[str, Any]) -> str:
    shadow = stats.get("shadow", {})
    sample = stats.get("sample", {})
    reasons = sample.get("drop_reasons", {})
    reason_text = " / ".join(
        f"{label} {count}" for label, count in reasons.items()
    )
    detail = sample.get("drop_detail", {})
    detail_text = " / ".join(
        f"{label} {count}"
        for label, count in {
            "源表缺失日直接不可用": detail.get("source_missing_issue_days"),
            "滞后构造新增不可用": detail.get("lag_induced_issue_days"),
            "日历首段历史缺口": detail.get("history_gap_days"),
            "目标日无标签": detail.get("target_unavailable_days"),
        }.items()
        if count
    )
    split = sample.get("split_counts", {})
    split_text = " / ".join(f"{label} {count}" for label, count in split.items())
    return (
        f"完整日历 {stats.get('calendar_days', '')} 天（{stats.get('date_start', '')} 至 "
        f"{stats.get('date_end', '')}）；日影有效观测 {shadow.get('valid_days', 0)} 天，"
        f"原表缺失 {shadow.get('source_missing_days', 0)} 天，质量不可用 "
        f"{shadow.get('quality_unavailable_days', 0)} 天；共同评价样本 "
        f"{sample.get('retained_count', 0)} 个，滞后构造排除 "
        f"{sum(reasons.values())} 个（{reason_text or '无'}；{detail_text or '无明细'}）；"
        f"样本分段：{split_text or '未加载'}。"
    )


def plot_shadow_observation_series(
    calendar_frame: pd.DataFrame,
    output_path: str | Path,
    *,
    stats: Mapping[str, Any] | None = None,
    representative_window: tuple[str, str] = ("2021-07-01", "2021-09-15"),
    title: str = "宇宙线日影南北偏移观测序列（完整日历）",
) -> Path:
    """Full-calendar shadow series with error bars plus a quality marker strip."""

    stats = stats or compute_quality_stats(calendar_frame)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = calendar_frame[calendar_frame["focus_status"] == "有效观测"].copy()

    center = pd.to_numeric(valid[SHADOW_CENTER_COL], errors="coerce")
    err = pd.to_numeric(valid[SHADOW_ERR_COL], errors="coerce")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12.5, 9),
        dpi=180,
        sharex=False,
    )
    fig.subplots_adjust(hspace=0.42, top=0.91, bottom=0.13, left=0.08, right=0.97)
    panes = [
        ("全时段概览", (calendar_frame["_date"].iloc[0], calendar_frame["_date"].iloc[-1])),
        (
            f"代表时段放大（{representative_window[0]} 至 {representative_window[1]}）",
            (pd.Timestamp(representative_window[0]), pd.Timestamp(representative_window[1])),
        ),
    ]

    for ax, (pane_title, x_limits) in zip(axes, panes):
        mask = (valid["_date"] >= x_limits[0]) & (valid["_date"] <= x_limits[1])
        sub = valid[mask]
        sub_center = pd.to_numeric(sub[SHADOW_CENTER_COL], errors="coerce")
        sub_err = pd.to_numeric(sub[SHADOW_ERR_COL], errors="coerce")

        ax.errorbar(
            sub["_date"],
            sub_center,
            yerr=sub_err,
            fmt="o",
            ms=4.2,
            elinewidth=1.0,
            capsize=0,
            color=SHADOW_SERIES_COLOR,
            ecolor=SHADOW_SERIES_ERR_COLOR,
            alpha=0.92,
            zorder=3,
            label="日影南北偏移（有效观测，误差棒为拟合中心值误差）",
        )
        ax.axhline(0.0, color="#94a3b8", lw=1.0, linestyle="--", alpha=0.8, zorder=1)

        all_center = pd.to_numeric(
            calendar_frame.loc[
                (calendar_frame["_date"] >= x_limits[0])
                & (calendar_frame["_date"] <= x_limits[1]),
                SHADOW_CENTER_COL,
            ],
            errors="coerce",
        )
        clean_center = all_center.dropna()
        span_max = float(
            max(
                abs(float(clean_center.min())) if not clean_center.empty else 0.0,
                abs(float(clean_center.max())) if not clean_center.empty else 0.0,
                0.1,
            )
        )
        data_bottom = -span_max * 1.18
        data_top = span_max * 1.22
        width = data_top - data_bottom
        strip_y = data_bottom + width * 0.10
        strip_h = max(width * 0.055, 0.03)

        ax.set_title(
            f"{title} · {pane_title}",
            fontsize=14,
            loc="left",
            pad=10,
        )
        ax.set_xlim(x_limits[0], x_limits[1])
        ax.set_ylim(data_bottom, data_top)
        _draw_status_strip(
            ax,
            calendar_frame[
                (calendar_frame["_date"] >= x_limits[0])
                & (calendar_frame["_date"] <= x_limits[1])
            ],
            status_col="focus_status",
            colors=SHADOW_STATUS_COLORS,
            y_bottom=strip_y,
            height=strip_h,
            label="质量标记带：有效 / 原表缺失 / 质量不可用",
        )
        _force_full_calendar_axis(
            ax,
            calendar_frame["_date"],
            full_range=pane_title.startswith("全时段"),
        )
        ax.set_ylabel("南北偏移中心值 (deg)")
        ax.grid(axis="y", alpha=0.25, color="#e5e7eb")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        if pane_title.startswith("代表时段"):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    fig.suptitle(
        "按完整日历绘制的日影观测：有效日期为散点 + 测量误差棒，不补入不合格观测",
        fontsize=16,
        y=0.975,
    )
    fig.text(0.01, 0.005, _stats_footer(stats), fontsize=8.5, color="#334155")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_segments_by_gap(
    ax: plt.Axes,
    dates: pd.Series,
    values: pd.Series,
    *,
    color: str,
    label: str,
    linewidth: float = 1.2,
    marker: str = ".",
    markersize: float = 3.0,
    alpha: float = 0.85,
) -> None:
    frame = pd.DataFrame({"_date": dates, "_value": values}).dropna().sort_values("_date")
    if frame.empty:
        return
    run_dates: list[pd.Timestamp] = []
    run_values: list[float] = []

    def flush() -> None:
        if len(run_dates) > 1:
            ax.plot(
                run_dates,
                run_values,
                color=color,
                linewidth=linewidth,
                marker=marker,
                markersize=markersize,
                alpha=alpha,
                label=label,
            )
        else:
            ax.scatter(run_dates, run_values, s=markersize + 9, color=color, alpha=alpha, label=label)
        run_dates.clear()
        run_values.clear()

    rows = frame.reset_index(drop=True)
    for idx, row in rows.iterrows():
        if run_dates and (row["_date"] - run_dates[-1]).days > 1:
            flush()
        run_dates.append(row["_date"])
        run_values.append(float(row["_value"]))
    flush()


def plot_prediction_comparison(
    calendar_frame: pd.DataFrame,
    predictions: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    stats: Mapping[str, Any] | None = None,
    manifest: pd.DataFrame | None = None,
    test_start: str | pd.Timestamp | None = None,
    target_col: str = TARGET_COL,
    representative_window: tuple[str, str] = ("2021-08-10", "2021-10-02"),
) -> Path:
    """Prediction vs truth on the full calendar; line segments break at gaps."""

    stats = stats or compute_quality_stats(calendar_frame, manifest)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12.5, 9),
        dpi=180,
        sharex=False,
    )
    fig.subplots_adjust(hspace=0.42, top=0.91, bottom=0.13, left=0.08, right=0.97)
    panes = [
        ("全时段概览", (calendar_frame["_date"].iloc[0], calendar_frame["_date"].iloc[-1])),
        (
            f"代表时段放大（{representative_window[0]} 至 {representative_window[1]}）",
            (pd.Timestamp(representative_window[0]), pd.Timestamp(representative_window[1])),
        ),
    ]

    for ax, (pane_title, x_limits) in zip(axes, panes):
        sub_frame = calendar_frame[
            (calendar_frame["_date"] >= x_limits[0])
            & (calendar_frame["_date"] <= x_limits[1])
        ].copy()
        y_true = pd.to_numeric(sub_frame[target_col], errors="coerce")
        _plot_segments_by_gap(
            ax,
            sub_frame["_date"],
            y_true,
            color="#1e293b",
            label="真实太阳风速度（完整日历归档值）",
            linewidth=1.5,
            marker="o",
            markersize=2.8,
            alpha=0.8,
        )

        for run in predictions:
            run_frame = run["frame"]
            run_frame = run_frame[
                (run_frame["_date"] >= x_limits[0])
                & (run_frame["_date"] <= x_limits[1])
            ]
            _plot_segments_by_gap(
                ax,
                run_frame["_date"],
                pd.to_numeric(run_frame["y_pred"], errors="coerce"),
                color=run["color"],
                label=run["label"],
                linewidth=1.1,
                marker=".",
                markersize=5.0,
                alpha=0.9,
            )

        if test_start is not None:
            ax.axvline(pd.Timestamp(test_start), color="#c0392b", linestyle=":", lw=1.4, zorder=0)
            ax.text(
                pd.Timestamp(test_start),
                ax.get_ylim()[1],
                "迭代评估段起点",
                fontsize=8,
                color="#c0392b",
                ha="right",
                va="top",
            )

        y_values = pd.concat(
            [
                pd.to_numeric(sub_frame[target_col], errors="coerce"),
                *[
                    pd.to_numeric(run["frame"]["y_pred"], errors="coerce")
                    for run in predictions
                    if not run["frame"].empty
                ],
            ],
            ignore_index=True,
        ).dropna()
        y_min = float(np.nanmin(y_values)) if not y_values.empty else 0.0
        y_max = float(np.nanmax(y_values)) if not y_values.empty else 1.0
        data_bottom = y_min - (y_max - y_min) * 0.18
        data_top = y_max + (y_max - y_min) * 0.10
        width = data_top - data_bottom
        strip_y = data_bottom + width * 0.02
        strip_h = max(width * 0.035, 1.2)

        ax.set_ylim(data_bottom, data_top)
        _draw_status_strip(
            ax,
            sub_frame,
            status_col="focus_status",
            colors=SHADOW_STATUS_COLORS,
            y_bottom=strip_y,
            height=strip_h,
            label="质量标记带：有效 / 原表缺失 / 质量不可用",
        )
        if manifest is not None and not manifest.empty:
            _draw_status_strip(
                ax,
                sub_frame.merge(
                    manifest[["issue_date", "sample_group"]],
                    left_on="_date",
                    right_on="issue_date",
                    how="left",
                ).fillna({"sample_group": "未覆盖"}),
                status_col="sample_group",
                colors=SAMPLE_STATUS_COLORS,
                y_bottom=strip_y - strip_h * 2.1,
                height=strip_h,
                label="共同评价样本：训练 / 迭代评估 / 边界排除 / 滞后构造排除",
            )

        ax.set_title(f"预测 vs 真实 · {pane_title}", fontsize=14, loc="left", pad=10)
        ax.set_xlim(x_limits[0], x_limits[1])
        _force_full_calendar_axis(
            ax,
            calendar_frame["_date"],
            full_range=pane_title.startswith("全时段"),
        )
        ax.set_ylabel("太阳风速度 (km/s)")
        ax.grid(axis="y", alpha=0.25, color="#e5e7eb")
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92, ncol=2)
        if pane_title.startswith("代表时段"):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    fig.suptitle(
        "预测与真实值同日期比较：缺失日期不连线、日期轴不压缩",
        fontsize=16,
        y=0.975,
    )
    fig.text(0.01, 0.005, _stats_footer(stats), fontsize=8.5, color="#334155")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _finite_metrics(
    frame: pd.DataFrame | None,
    *,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "pearson_r": np.nan}
    use = pd.DataFrame(
        {
            "y_true": pd.to_numeric(frame[y_true_col], errors="coerce"),
            "y_pred": pd.to_numeric(frame[y_pred_col], errors="coerce"),
        }
    ).dropna()
    n = len(use)
    if n == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "pearson_r": np.nan}
    y = use["y_true"].to_numpy(dtype=float)
    z = use["y_pred"].to_numpy(dtype=float)
    diff = y - z
    pearson = (
        float(np.corrcoef(y, z)[0, 1])
        if n >= 2 and np.std(y) > 0 and np.std(z) > 0
        else np.nan
    )
    return {
        "n": n,
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mae": float(np.mean(np.abs(diff))),
        "pearson_r": pearson,
    }


def _metric_text(metric: Mapping[str, Any] | None) -> str:
    if not metric or metric.get("n", 0) <= 0:
        return "未计算"
    r_value = metric.get("pearson_r", np.nan)
    r_text = f"r {r_value:.2f}" if np.isfinite(r_value) else "r 未计算"
    return f"RMSE {metric['rmse']:.1f} · {r_text} · n={metric['n']}"


def _split_run_groups(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    groups: dict[str, dict[str, Mapping[str, Any]]] = {
        "Ridge": {},
        "ElasticNet": {},
    }
    for run in predictions:
        run_id = str(run.get("run_id", "")).lower()
        label = str(run.get("label", ""))
        if run_id.startswith("ridge"):
            model = "Ridge"
        elif run_id.startswith("elasticnet"):
            model = "ElasticNet"
        else:
            model = str(run.get("model") or "未识别模型")
        arm = "baseline" if ("baseline" in run_id or "对照组" in label) else "treatment"
        groups.setdefault(model, {})[arm] = run
    return groups


def _scatter_valid(
    ax: plt.Axes,
    dates: pd.Series,
    values: pd.Series,
    *,
    color: str,
    marker: str = "o",
    size: float = 20.0,
    alpha: float = 0.9,
    label: str | None = None,
    zorder: int = 3,
    facecolors: str | None = None,
    edgecolors: str | None = None,
    linewidths: float | None = None,
) -> pd.DataFrame | None:
    frame = pd.DataFrame(
        {
            "_date": pd.to_datetime(pd.Series(dates), errors="coerce"),
            "_value": pd.to_numeric(pd.Series(values), errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return None
    kwargs = {
        "marker": marker,
        "s": size,
        "alpha": alpha,
        "label": label,
        "zorder": zorder,
    }
    if facecolors is None:
        kwargs["color"] = color
    else:
        kwargs["facecolors"] = facecolors
        kwargs["edgecolors"] = edgecolors if edgecolors is not None else color
        kwargs["linewidths"] = linewidths if linewidths is not None else 1.0
    ax.scatter(frame["_date"], frame["_value"], **kwargs)
    return frame


def _style_date_axis(
    ax: plt.Axes,
    *,
    daily_ticks: bool = False,
    tick_interval: int = 5,
) -> None:
    if daily_ticks:
        ax.xaxis.set_major_locator(
            mdates.DayLocator(interval=max(1, int(tick_interval)))
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="x", linestyle=":", color="#cbd5e1", alpha=0.65)


def _frame_split(
    frame: pd.DataFrame | None,
    split: str,
) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    if "split" not in frame.columns:
        return frame
    subset = frame.loc[frame["split"].eq(split)]
    return subset if not subset.empty else None


def _train_test_metric_text(
    train_metric: Mapping[str, Any] | None,
    test_metric: Mapping[str, Any] | None,
) -> str:
    parts: list[str] = []
    for metric in (train_metric, test_metric):
        if not metric or metric.get("n", 0) <= 0:
            continue
        if not np.isfinite(float(metric.get("rmse", np.nan))):
            parts.append("未计算")
            continue
        text = f"RMSE {float(metric['rmse']):.1f}"
        if np.isfinite(float(metric.get("pearson_r", np.nan))):
            text += f" · r {float(metric['pearson_r']):.2f}"
        parts.append(text)
    if not parts:
        return "未计算"
    return " / ".join(parts)


def plot_prediction_scatter_quality(
    calendar_frame: pd.DataFrame,
    predictions: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    stats: Mapping[str, Any] | None = None,
    manifest: pd.DataFrame | None = None,
    target_col: str = TARGET_COL,
    evaluation_window: tuple[str, str] | None = None,
) -> Path:
    """Date-axis scatter comparison covering the full train+test calendar.

    When ``evaluation_window`` is omitted the plot shows every available true
    value and prediction on the real issue-date axis. Predictions are never
    connected with lines and never imputed; quality and sample statuses are
    merged into one narrow strip below the two model rows.
    """

    stats = stats or compute_quality_stats(calendar_frame, manifest)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    calendar = calendar_frame.sort_values("_date").reset_index(drop=True)
    if evaluation_window is None:
        plot_start = calendar["_date"].min()
        plot_end = calendar["_date"].max()
    else:
        plot_start = pd.Timestamp(evaluation_window[0])
        plot_end = pd.Timestamp(evaluation_window[1])
    plot_mask = (calendar["_date"] >= plot_start) & (calendar["_date"] <= plot_end)
    plot_calendar = calendar[plot_mask].copy()

    test_start: pd.Timestamp | None = None
    test_end: pd.Timestamp | None = None
    if manifest is not None and not manifest.empty:
        test_issues = manifest.loc[
            manifest["sample_group"].eq("迭代评估样本"),
            "issue_date",
        ].dropna()
        if not test_issues.empty:
            test_start = pd.Timestamp(test_issues.min())
            test_end = pd.Timestamp(test_issues.max())

    y_series: list[pd.Series] = [
        pd.to_numeric(plot_calendar[target_col], errors="coerce"),
    ]
    for run in predictions:
        run_frame = run["frame"]
        y_series.append(
            pd.to_numeric(
                run_frame.loc[
                    (run_frame["_date"] >= plot_start)
                    & (run_frame["_date"] <= plot_end),
                    "y_pred",
                ],
                errors="coerce",
            )
        )
    y_values = pd.concat(y_series, ignore_index=True).dropna()
    if y_values.empty:
        y_min, y_max = 300.0, 600.0
    else:
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        span = y_max - y_min
        y_min -= span * 0.10
        y_max += span * 0.08

    groups = _split_run_groups(predictions)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#334155",
            markeredgecolor="#334155",
            markeredgewidth=0.8,
            markersize=6.5,
            label="真实太阳风速度",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#1d4ed8",
            markeredgecolor="#1d4ed8",
            markeredgewidth=0.8,
            markersize=6.5,
            label="基线预测",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#ea580c",
            markeredgecolor="#ea580c",
            markeredgewidth=0.8,
            markersize=6.5,
            label="加入日影后预测",
        ),
    ]
    if test_start is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#1b6ca8",
                linestyle="--",
                marker="",
                linewidth=1.2,
                label="迭代评估段",
            )
        )

    # 保持真实日期与完全日历间隔，仅缩小横向绘图尺寸以压缩日期间绘制间距。
    fig = plt.figure(figsize=(8.5, 6.75), dpi=180, constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[1.0, 1.0, 0.62],
        hspace=0.06,
    )
    ax_ridge = fig.add_subplot(grid[0, 0])
    ax_en = fig.add_subplot(grid[1, 0], sharex=ax_ridge)
    ax_strip = fig.add_subplot(grid[2, 0], sharex=ax_ridge)

    tick_interval = max(7, (plot_end - plot_start).days // 14)
    for model_name, ax_model in (("Ridge", ax_ridge), ("ElasticNet", ax_en)):
        if test_start is not None and test_end is not None:
            ax_model.axvspan(
                test_start,
                test_end,
                color="#1b6ca8",
                alpha=0.06,
                linewidth=0,
                zorder=0,
            )
            ax_model.axvline(
                test_start,
                color="#1b6ca8",
                linestyle="--",
                linewidth=0.9,
                zorder=0,
            )
        _scatter_valid(
            ax_model,
            plot_calendar["_date"],
            pd.to_numeric(plot_calendar[target_col], errors="coerce"),
            color=TRUTH_STYLE["color"],
            marker=TRUTH_STYLE["marker"],
            size=TRUTH_STYLE["size"],
            alpha=TRUTH_STYLE["alpha"],
            label="真实太阳风速度",
            zorder=4,
        )
        baseline_run = groups.get(model_name, {}).get("baseline")
        treatment_run = groups.get(model_name, {}).get("treatment")
        baseline_frame = None
        treatment_frame = None
        if baseline_run is not None:
            baseline_frame = baseline_run["frame"].loc[
                (baseline_run["frame"]["_date"] >= plot_start)
                & (baseline_run["frame"]["_date"] <= plot_end)
            ].copy()
            _scatter_valid(
                ax_model,
                baseline_frame["_date"],
                pd.to_numeric(baseline_frame["y_pred"], errors="coerce"),
                color=BASELINE_STYLE["color"],
                marker=BASELINE_STYLE["marker"],
                size=BASELINE_STYLE["size"],
                alpha=BASELINE_STYLE["alpha"],
                label="基线预测",
                zorder=1,
            )
        if treatment_run is not None:
            treatment_frame = treatment_run["frame"].loc[
                (treatment_run["frame"]["_date"] >= plot_start)
                & (treatment_run["frame"]["_date"] <= plot_end)
            ].copy()
            _scatter_valid(
                ax_model,
                treatment_frame["_date"],
                pd.to_numeric(treatment_frame["y_pred"], errors="coerce"),
                color=TREATMENT_STYLE["color"],
                marker=TREATMENT_STYLE["marker"],
                size=TREATMENT_STYLE["size"],
                alpha=TREATMENT_STYLE["alpha"],
                label="加入日影后预测",
                zorder=1,
            )
        baseline_train_metric = _finite_metrics(
            _frame_split(baseline_frame, "train")
        )
        baseline_test_metric = _finite_metrics(
            _frame_split(baseline_frame, "test")
        )
        treatment_train_metric = _finite_metrics(
            _frame_split(treatment_frame, "train")
        )
        treatment_test_metric = _finite_metrics(
            _frame_split(treatment_frame, "test")
        )
        baseline_text = _train_test_metric_text(
            baseline_train_metric,
            baseline_test_metric,
        )
        treatment_text = _train_test_metric_text(
            treatment_train_metric,
            treatment_test_metric,
        )
        ax_model.set_title(
            f"{model_name} · 基线[{baseline_text}]  ->  "
            f"加日影[{treatment_text}]",
            fontsize=9,
            loc="left",
            pad=8,
        )
        ax_model.set_xlim(
            plot_start - pd.Timedelta(days=0.5),
            plot_end + pd.Timedelta(days=0.5),
        )
        ax_model.set_ylim(y_min, y_max)
        ax_model.set_ylabel("太阳风速度 (km/s)")
        ax_model.grid(axis="y", alpha=0.25, color="#e5e7eb")
        ax_model.tick_params(axis="x", labelbottom=False)
        ax_model.legend(
            handles=legend_handles,
            loc="upper right",
            ncol=2 if test_start is not None else 1,
            fontsize=7.5,
            framealpha=0.85,
            handletextpad=0.3,
            columnspacing=0.8,
        )
        _style_date_axis(
            ax_model,
            daily_ticks=True,
            tick_interval=tick_interval,
        )
        ax_model.margins(x=0)

    strip_calendar = plot_calendar.copy()
    if manifest is not None and not manifest.empty:
        strip_calendar = strip_calendar.merge(
            manifest[["issue_date", "sample_group"]],
            left_on="_date",
            right_on="issue_date",
            how="left",
            sort=False,
        ).fillna({"sample_group": "未覆盖"})
    _draw_status_strip(
        ax_strip,
        strip_calendar,
        status_col="focus_status",
        colors=SHADOW_STATUS_COLORS,
        y_bottom=0.50,
        height=0.24,
        label="日影质量：有效 / 原表缺失 / 质量不可用",
        add_legend=False,
    )
    _draw_status_strip(
        ax_strip,
        strip_calendar,
        status_col="sample_group",
        colors=SAMPLE_STATUS_COLORS,
        y_bottom=0.08,
        height=0.24,
        label="样本：训练 / 迭代评估 / 边界排除 / 滞后构造排除",
        add_legend=False,
    )
    ax_strip.set_xlim(
        plot_start - pd.Timedelta(days=0.5),
        plot_end + pd.Timedelta(days=0.5),
    )
    ax_strip.set_ylim(0.0, 1.0)
    ax_strip.set_yticks([])
    _style_date_axis(
        ax_strip,
        daily_ticks=True,
        tick_interval=tick_interval,
    )
    for spine_name in ("left", "right", "top"):
        ax_strip.spines[spine_name].set_visible(False)
    ax_strip.set_ylabel("")
    ax_strip.margins(x=0)

    fig.suptitle(
        f"预测 vs 真实散点对照 · 全时段 {plot_start:%Y-%m-%d} 至 {plot_end:%Y-%m-%d}"
        "（点估计直接散点，不插值、不连线、不绘未计算的误差棒）",
        fontsize=13,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path

