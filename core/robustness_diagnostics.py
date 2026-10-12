from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.unified_schema import (
    BootstrapReport,
    RobustnessOverall,
    RobustnessReport,
    TimeSliceReport,
)


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        return float("nan")
    return float(np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / total))


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        return float("nan")
    return float(np.sum(w * np.abs(y_true - y_pred)) / total)


def weighted_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        return float("nan")
    mean_x = float(np.sum(w * y_true) / total)
    mean_y = float(np.sum(w * y_pred) / total)
    cov_xy = float(np.sum(w * (y_true - mean_x) * (y_pred - mean_y)) / total)
    var_x = float(np.sum(w * (y_true - mean_x) ** 2) / total)
    var_y = float(np.sum(w * (y_pred - mean_y) ** 2) / total)
    if var_x <= 0 or var_y <= 0:
        return float("nan")
    return float(cov_xy / np.sqrt(var_x * var_y))


def _block_ids(issue_dates: pd.Series, block_size: int) -> np.ndarray:
    dates = pd.to_datetime(issue_dates).dt.normalize()
    years = dates.dt.year
    anchors = pd.to_datetime(
        pd.Series(years).apply(lambda year: f"{int(year)}-01-01")
    )
    return ((dates - anchors).dt.days // int(block_size)).to_numpy(dtype=int)


def paired_calendar_block_bootstrap(
    baseline: pd.DataFrame,
    treatment: pd.DataFrame,
    *,
    block_size: int = 7,
    n_draws: int = 10000,
    seed: int = 20261008,
    ci_level: float = 95.0,
) -> dict[str, Any]:
    """Paired calendar-block bootstrap for the common-sample test set.

    Blocks are anchored to the first day of each calendar year (verified
    reference scripts use fixed January-1 anchors). Each draw resamples whole
    issue-date blocks with replacement; metrics are recomputed with block
    counts as weights, never by deduplicating repeated rows.
    """

    if len(baseline) != len(treatment):
        raise ValueError("配对 bootstrap 要求两组测试样本一一对应")
    if list(baseline["sample_id"]) != list(treatment["sample_id"]):
        raise ValueError("配对 bootstrap 要求两组 sample_id 顺序完全一致")
    if not np.allclose(baseline["y_true"].to_numpy(), treatment["y_true"].to_numpy()):
        raise ValueError("配对 bootstrap 要求两组 y_true 完全一致")
    if n_draws < 100:
        raise ValueError("n_draws 至少为 100")

    y_true = baseline["y_true"].to_numpy(dtype=float)
    yp_base = baseline["y_pred"].to_numpy(dtype=float)
    yp_treat = treatment["y_pred"].to_numpy(dtype=float)
    block_ids = _block_ids(baseline["issue_date"], block_size)
    nonempty = np.unique(block_ids)
    n_total_blocks = int(block_ids.max()) + 1

    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(nonempty), size=(n_draws, len(nonempty)))
    chosen_blocks = nonempty[chosen]
    counts = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=n_total_blocks),
        axis=1,
        arr=chosen_blocks,
    )
    weights = counts[:, block_ids].astype(float)
    row_sums = weights.sum(axis=1, keepdims=True)

    base_rmse = np.sqrt(np.sum(weights * (y_true - yp_base) ** 2, axis=1) / row_sums[:, 0])
    treat_rmse = np.sqrt(np.sum(weights * (y_true - yp_treat) ** 2, axis=1) / row_sums[:, 0])
    base_mae = np.sum(weights * np.abs(y_true - yp_base), axis=1) / row_sums[:, 0]
    treat_mae = np.sum(weights * np.abs(y_true - yp_treat), axis=1) / row_sums[:, 0]

    w_total = row_sums[:, 0]
    base_mean = np.sum(weights * y_true, axis=1) / w_total
    treat_mean = np.sum(weights * y_true, axis=1) / w_total
    base_r_num = np.sum(weights * (y_true - base_mean[:, None]) * (yp_base - base_mean[:, None]), axis=1)
    treat_r_num = np.sum(weights * (y_true - treat_mean[:, None]) * (yp_treat - treat_mean[:, None]), axis=1)
    base_var_x = np.sum(weights * (y_true - base_mean[:, None]) ** 2, axis=1) / w_total
    base_var_y = np.sum(weights * (yp_base - base_mean[:, None]) ** 2, axis=1) / w_total
    treat_var_x = np.sum(weights * (y_true - treat_mean[:, None]) ** 2, axis=1) / w_total
    treat_var_y = np.sum(weights * (yp_treat - treat_mean[:, None]) ** 2, axis=1) / w_total
    with np.errstate(invalid="ignore", divide="ignore"):
        base_r = base_r_num / w_total / np.sqrt(base_var_x * base_var_y)
        treat_r = treat_r_num / w_total / np.sqrt(treat_var_x * treat_var_y)
    base_r = np.where(np.isfinite(base_r), base_r, np.nan)
    treat_r = np.where(np.isfinite(treat_r), treat_r, np.nan)

    gain_rmse = base_rmse - treat_rmse
    gain_mae = base_mae - treat_mae
    gain_r = treat_r - base_r

    def _ci(values: np.ndarray) -> list[float]:
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return [float("nan"), float("nan")]
        lo = (100.0 - ci_level) / 2.0
        hi = 100.0 - lo
        lower, upper = np.nanpercentile(finite, [lo, hi])
        return [float(lower), float(upper)]

    return {
        "method": "paired_calendar_block_bootstrap",
        "status": "completed",
        "n_iterations": int(n_draws),
        "block_size": int(block_size),
        "seed": int(seed),
        "ci_level": float(ci_level),
        "n_test_samples": int(len(baseline)),
        "n_nonempty_blocks": int(len(nonempty)),
        "point_gain_rmse": weighted_rmse(y_true, yp_base, np.ones_like(y_true))
        - weighted_rmse(y_true, yp_treat, np.ones_like(y_true)),
        "point_gain_mae": weighted_mae(y_true, yp_base, np.ones_like(y_true))
        - weighted_mae(y_true, yp_treat, np.ones_like(y_true)),
        "point_gain_pearson_r": weighted_pearson(y_true, yp_treat, np.ones_like(y_true))
        - weighted_pearson(y_true, yp_base, np.ones_like(y_true)),
        "rmse_ci": _ci(gain_rmse),
        "mae_ci": _ci(gain_mae),
        "pearson_r_ci": _ci(gain_r),
        "stable_rmse": bool(
            np.isfinite(gain_rmse).all()
            and (float(np.nanpercentile(gain_rmse, 100 - (100.0 - ci_level) / 2.0)) < 0
                 or float(np.nanpercentile(gain_rmse, (100.0 - ci_level) / 2.0)) > 0)
        ),
    }


def empty_bootstrap_summary(reason: str) -> dict[str, Any]:
    return {
        "method": "paired_calendar_block_bootstrap",
        "status": "not_run",
        "n_iterations": 0,
        "block_size": None,
        "seed": None,
        "reason": reason,
    }


def summary_to_robustness_report(summary: dict[str, Any] | None) -> RobustnessReport:
    if summary is None or summary.get("status") != "completed":
        reason = (summary or {}).get("reason", "未配置配对日历块 bootstrap")
        bootstrap = BootstrapReport(
            stable=False,
            n_iterations=0,
            status="not_run",
        )
        return RobustnessReport(
            bootstrap=bootstrap,
            time_slice=TimeSliceReport(stable=False),
            overall=RobustnessOverall(
                stable=False,
                concern=f"稳健性检验未运行：{reason}",
            ),
            recommendation="在样本按共同日历对齐后补充配对 bootstrap 再作方向性结论。",
        )

    stable_rmse = bool(summary.get("stable_rmse"))
    rmse_ci = summary.get("rmse_ci")
    if rmse_ci is not None:
        ci_text = f"RMSE 增益 95% CI=[{rmse_ci[0]:+.3f}, {rmse_ci[1]:+.3f}]"
    else:
        ci_text = "RMSE 增益 CI 不可用"
    concern = (
        None
        if stable_rmse
        else f"配对 bootstrap 的 {ci_text} 跨越零，主增益方向尚不稳定。"
    )
    recommendation = (
        f"配对块 bootstrap（{summary.get('n_iterations', 0)} 次、7 天块）"
        f"{'支持主增益方向' if stable_rmse else '不支持稳定增益方向'}，"
        "建议直接以该 CI 辅助假设支持度更新。"
    )
    return RobustnessReport(
        bootstrap=BootstrapReport(
            pearson_r_ci=summary.get("pearson_r_ci"),
            rmse_ci=summary.get("rmse_ci"),
            mae_ci=summary.get("mae_ci"),
            stable=stable_rmse,
            n_iterations=int(summary.get("n_iterations") or 0),
            status="completed",
            block_size=int(summary["block_size"]) if summary.get("block_size") is not None else None,
            seed=int(summary["seed"]) if summary.get("seed") is not None else None,
            point_gain_rmse=summary.get("point_gain_rmse"),
            point_gain_mae=summary.get("point_gain_mae"),
            point_gain_pearson_r=summary.get("point_gain_pearson_r"),
        ),
        time_slice=TimeSliceReport(stable=stable_rmse),
        overall=RobustnessOverall(stable=stable_rmse, concern=concern),
        recommendation=recommendation,
    )
