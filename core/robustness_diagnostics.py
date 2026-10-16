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
    """Year-anchored calendar block ids, compacted to contiguous integers.

    Blocks restart at January 1 of each year; the same day-of-year in
    different years therefore never shares an id and cannot be merged by
    the bootstrap bincount.
    """

    dates = pd.to_datetime(issue_dates).dt.normalize()
    years = dates.dt.year
    min_year = int(years.min())
    anchors = pd.to_datetime(
        pd.Series([f"{int(year)}-01-01" for year in years])
    )
    day_offsets = (dates - anchors).dt.days // int(block_size)
    raw = (years - min_year) * 366 + day_offsets
    raw = raw.to_numpy(dtype=np.int64)
    unique = np.unique(raw)
    return np.searchsorted(unique, raw).astype(dtype=int)


def paired_calendar_block_bootstrap(
    baseline: pd.DataFrame,
    treatment: pd.DataFrame,
    *,
    block_size: int = 7,
    n_draws: int = 10000,
    seed: int = 20261008,
    ci_level: float = 95.0,
    verify_pearson: bool = False,
    min_blocks: int = 2,
) -> dict[str, Any]:
    """Paired calendar-block bootstrap for the common-sample test set.

    Blocks are anchored to the first day of each calendar year (verified
    reference scripts use fixed January-1 anchors). Each draw resamples whole
    issue-date blocks with replacement; metrics are recomputed with block
    counts as weights, never by deduplicating repeated rows. When the number
    of non-empty calendar blocks is below ``min_blocks``, no directional
    stability conclusion is made.
    """

    if len(baseline) != len(treatment):
        raise ValueError("配对 bootstrap 要求两组测试样本一一对应")
    if list(baseline["sample_id"]) != list(treatment["sample_id"]):
        raise ValueError("配对 bootstrap 要求两组 sample_id 顺序完全一致")
    if not np.allclose(baseline["y_true"].to_numpy(), treatment["y_true"].to_numpy()):
        raise ValueError("配对 bootstrap 要求两组 y_true 完全一致")
    if n_draws < 100:
        raise ValueError("n_draws 至少为 100")
    if min_blocks < 1:
        raise ValueError("min_blocks 至少为 1")

    y_true = baseline["y_true"].to_numpy(dtype=float)
    yp_base = baseline["y_pred"].to_numpy(dtype=float)
    yp_treat = treatment["y_pred"].to_numpy(dtype=float)
    block_ids = _block_ids(baseline["issue_date"], block_size)
    nonempty = np.unique(block_ids)
    n_total_blocks = int(block_ids.max()) + 1
    n_nonempty_blocks = int(len(nonempty))
    insufficient_blocks = n_nonempty_blocks < int(min_blocks)

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
    true_mean = np.sum(weights * y_true, axis=1) / w_total
    base_mean = np.sum(weights * yp_base, axis=1) / w_total
    treat_mean = np.sum(weights * yp_treat, axis=1) / w_total
    base_r_num = np.sum(
        weights * (y_true - true_mean[:, None]) * (yp_base - base_mean[:, None]),
        axis=1,
    )
    treat_r_num = np.sum(
        weights * (y_true - true_mean[:, None]) * (yp_treat - treat_mean[:, None]),
        axis=1,
    )
    base_var_x = np.sum(weights * (y_true - true_mean[:, None]) ** 2, axis=1) / w_total
    base_var_y = np.sum(weights * (yp_base - base_mean[:, None]) ** 2, axis=1) / w_total
    treat_var_x = np.sum(weights * (y_true - true_mean[:, None]) ** 2, axis=1) / w_total
    treat_var_y = np.sum(weights * (yp_treat - treat_mean[:, None]) ** 2, axis=1) / w_total
    with np.errstate(invalid="ignore", divide="ignore"):
        base_r = base_r_num / w_total / np.sqrt(base_var_x * base_var_y)
        treat_r = treat_r_num / w_total / np.sqrt(treat_var_x * treat_var_y)
    base_r = np.where(np.isfinite(base_r), base_r, np.nan)
    treat_r = np.where(np.isfinite(treat_r), treat_r, np.nan)
    n_valid_pearson_draws = int(
        np.count_nonzero(np.isfinite(base_r) & np.isfinite(treat_r))
    )

    if verify_pearson:
        probe = min(25, n_draws)
        for draw_index in range(probe):
            row_weights = weights[draw_index]
            expected_base = weighted_pearson(y_true, yp_base, row_weights)
            expected_treat = weighted_pearson(y_true, yp_treat, row_weights)
            if not np.isclose(
                base_r[draw_index], expected_base, atol=1e-9, equal_nan=True
            ):
                raise AssertionError(
                    f"第 {draw_index} 次 baseline 向量化 Pearson 与加权实现不一致："
                    f"{base_r[draw_index]} vs {expected_base}"
                )
            if not np.isclose(
                treat_r[draw_index], expected_treat, atol=1e-9, equal_nan=True
            ):
                raise AssertionError(
                    f"第 {draw_index} 次 treatment 向量化 Pearson 与加权实现不一致："
                    f"{treat_r[draw_index]} vs {expected_treat}"
                )

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

    stable_rmse = bool(
        np.isfinite(gain_rmse).all()
        and (float(np.nanpercentile(gain_rmse, 100 - (100.0 - ci_level) / 2.0)) < 0
             or float(np.nanpercentile(gain_rmse, (100.0 - ci_level) / 2.0)) > 0)
    )
    return {
        "method": "paired_calendar_block_bootstrap",
        "status": "completed",
        "n_iterations": int(n_draws),
        "block_size": int(block_size),
        "min_blocks": int(min_blocks),
        "seed": int(seed),
        "ci_level": float(ci_level),
        "n_test_samples": int(len(baseline)),
        "n_nonempty_blocks": n_nonempty_blocks,
        "insufficient_blocks": insufficient_blocks,
        "n_valid_pearson_draws": n_valid_pearson_draws,
        "point_gain_rmse": weighted_rmse(y_true, yp_base, np.ones_like(y_true))
        - weighted_rmse(y_true, yp_treat, np.ones_like(y_true)),
        "point_gain_mae": weighted_mae(y_true, yp_base, np.ones_like(y_true))
        - weighted_mae(y_true, yp_treat, np.ones_like(y_true)),
        "point_gain_pearson_r": weighted_pearson(y_true, yp_treat, np.ones_like(y_true))
        - weighted_pearson(y_true, yp_base, np.ones_like(y_true)),
        "rmse_ci": _ci(gain_rmse),
        "mae_ci": _ci(gain_mae),
        "pearson_r_ci": _ci(gain_r),
        "stable_rmse": stable_rmse and not insufficient_blocks,
    }


def empty_bootstrap_summary(reason: str) -> dict[str, Any]:
    return {
        "method": "paired_calendar_block_bootstrap",
        "status": "not_run",
        "n_iterations": 0,
        "block_size": None,
        "min_blocks": None,
        "n_nonempty_blocks": None,
        "insufficient_blocks": False,
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
            min_blocks=(
                int(summary["min_blocks"])
                if summary is not None and summary.get("min_blocks") is not None
                else None
            ),
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
    insufficient_blocks = bool(summary.get("insufficient_blocks"))
    n_nonempty_blocks = summary.get("n_nonempty_blocks")
    min_blocks = summary.get("min_blocks")
    rmse_ci = summary.get("rmse_ci")
    point_gain_rmse = summary.get("point_gain_rmse")
    identical_predictions = bool(
        isinstance(rmse_ci, (list, tuple))
        and len(rmse_ci) >= 2
        and float(rmse_ci[0]) == 0.0
        and float(rmse_ci[1]) == 0.0
        and (point_gain_rmse is None or abs(float(point_gain_rmse)) < 1e-9)
    )
    if rmse_ci is not None:
        ci_text = f"RMSE 增益 95% CI=[{rmse_ci[0]:+.3f}, {rmse_ci[1]:+.3f}]"
    else:
        ci_text = "RMSE 增益 CI 不可用"
    if identical_predictions:
        concern = (
            "两组预测完全一致：RMSE 增益点估计与 bootstrap CI 均为 [0,0]，"
            "未检出任何预测差异；该结果不能直接证明日影没有物理作用，"
            "需先检查特征是否真正进入模型、系数是否被压缩为零，"
            "或对照组与实验组是否意外使用了相同信息。"
        )
    elif insufficient_blocks:
        concern = (
            f"配对 bootstrap 有效日历块数为 {n_nonempty_blocks}，"
            f"低于预登记最小块数 {min_blocks}，无法作出方向性稳定判断。"
        )
    else:
        concern = (
            None
            if stable_rmse
            else f"配对 bootstrap 的 {ci_text} 跨越零，主增益方向尚不稳定。"
        )
    block_text = f"{summary.get('block_size', 7)} 天块"
    if identical_predictions:
        recommendation = (
            f"配对块 bootstrap（{summary.get('n_iterations', 0)} 次、{block_text}）"
            "确认了两组预测完全一致；下一步应核对实际运行的对照/实验特征、"
            "模型系数与预测文件，确认日影变量是否真正参与建模后再作方向性实验。"
        )
    elif insufficient_blocks:
        recommendation = (
            f"配对块 bootstrap（{summary.get('n_iterations', 0)} 次、{block_text}）"
            f"仅有 {n_nonempty_blocks} 个非空日历块，低于 min_blocks={min_blocks}，"
            "不能据此作方向性稳定结论；需扩大共同测试日历范围或降低最小块门槛预登记后重跑。"
        )
    else:
        recommendation = (
            f"配对块 bootstrap（{summary.get('n_iterations', 0)} 次、{block_text}）"
            f"{'支持主增益方向' if stable_rmse else '不支持稳定增益方向'}；"
            "未运行分时段检验，不据此作分时段稳定性结论。"
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
            n_nonempty_blocks=(
                int(n_nonempty_blocks) if n_nonempty_blocks is not None else None
            ),
            min_blocks=int(min_blocks) if min_blocks is not None else None,
            insufficient_blocks=insufficient_blocks,
            seed=int(summary["seed"]) if summary.get("seed") is not None else None,
            point_gain_rmse=summary.get("point_gain_rmse"),
            point_gain_mae=summary.get("point_gain_mae"),
            point_gain_pearson_r=summary.get("point_gain_pearson_r"),
        ),
        time_slice=TimeSliceReport(stable=False),
        overall=RobustnessOverall(stable=stable_rmse, concern=concern),
        recommendation=recommendation,
    )
