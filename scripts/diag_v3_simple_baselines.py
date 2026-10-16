"""Simple-baseline comparison for the frozen v3 dev_backtest segment.

This diagnostic reuses the already-fitted fixed-parameter arms from
``v3_fixed_param_sensitivity`` and adds two zero-parameter scientific
references:

* train-target mean forecast;
* latest-available solar wind speed persistence (value known at issue date).

It does not tune any hyper-parameter against the 35-day segment and does not
touch closed-loop state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Microsoft JhengHei",
]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.json_safe import json_dumps
from core.robustness_diagnostics import paired_calendar_block_bootstrap


SESSION_ROOT = (
    ROOT
    / "runtime"
    / "live_session"
    / "sessions"
    / "session_20261015_132955_705440"
)
OMNI_CSV = SESSION_ROOT / "data" / "raw" / "omni_2021_full_calendar_v3.csv"
ARCHIVE_PREDICTIONS_CSV = (
    SESSION_ROOT
    / "results"
    / "round_02"
    / "archive"
    / "E_R02_03_r1"
    / "predictions.csv"
)
SENSITIVITY_DIR = ROOT / "artifacts" / "v3_fixed_param_sensitivity"
DAILY_CSV = SENSITIVITY_DIR / "fixed_param_daily.csv"
SUMMARY_CSV = SENSITIVITY_DIR / "fixed_param_summary.csv"
OUTPUT_DIR = ROOT / "artifacts" / "v3_simple_baseline"

TARGET = "SW Plasma Speed, km/s"
N_DRAWS = 4000
SEED = 20261008
MIN_BLOCKS = 2


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
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
    }


def _load_test_order() -> tuple[pd.DataFrame, pd.Series]:
    daily = pd.read_csv(DAILY_CSV, dtype={"sample_id": str})
    canonical_run = next(
        run_id
        for run_id in daily["run_id"].unique()
        if run_id.endswith("vsw_history")
    )
    canonical = (
        daily.loc[daily["run_id"] == canonical_run]
        .sort_values("issue_date")
        .reset_index(drop=True)
    )
    return daily, canonical


def _load_archive_predictions() -> dict[str, pd.DataFrame]:
    archive = pd.read_csv(
        ARCHIVE_PREDICTIONS_CSV, dtype={"sample_id": str}
    )
    frames: dict[str, pd.DataFrame] = {}
    for arm in sorted(archive["arm"].unique()):
        frames[str(arm)] = (
            archive.loc[archive["arm"] == arm]
            .sort_values("issue_date")
            .reset_index(drop=True)
        )
    return frames


def _persistence_values(
    canonical: pd.DataFrame,
) -> tuple[list[float], int]:
    omni = pd.read_csv(OMNI_CSV)
    omni["date"] = pd.to_datetime(omni["TIME"], format="%Y%m%d")
    omni["vsw"] = pd.to_numeric(omni[TARGET], errors="coerce")
    omni = omni.dropna(subset=["vsw"]).sort_values("date").reset_index(drop=True)
    valid_dates = omni["date"].to_numpy()
    valid_values = omni["vsw"].to_numpy(dtype=float)

    values: list[float] = []
    fallback_count = 0
    for issue_text in canonical["issue_date"]:
        issue = pd.Timestamp(issue_text)
        available = np.flatnonzero(valid_dates <= np.datetime64(issue))
        if len(available) == 0:
            raise RuntimeError(f"发报日 {issue_text} 之前没有任何有效 Vsw 记录")
        index = int(available[-1])
        values.append(float(valid_values[index]))
        if pd.Timestamp(valid_dates[index]) != issue:
            fallback_count += 1
    return values, fallback_count


def main() -> int:
    daily, canonical = _load_test_order()
    archive_predictions = _load_archive_predictions()
    summary = pd.read_csv(SUMMARY_CSV, dtype={"run_id": str})
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train_mean_row = summary.loc[
        (summary["benchmark_type"] == "vsw_history")
        & (summary["family"] == "ridge")
    ].iloc[0]
    train_mean = float(train_mean_row["intercept"])
    y_true = canonical["y_true"].to_numpy(dtype=float)
    n = len(canonical)
    if n != 35:
        raise RuntimeError(f"预期 35 天，实际 {n}")

    persistence, fallback_count = _persistence_values(canonical)
    ref_order = {
        "train_mean": np.full(n, train_mean),
        "persistence_latest_vsw": np.asarray(persistence, dtype=float),
    }

    baseline_rows: list[dict] = []
    for name, values in ref_order.items():
        baseline_rows.append(
            {
                "baseline_name": name,
                "description": (
                    "训练目标均值预测"
                    if name == "train_mean"
                    else "发报日最新可用风速持续预测"
                ),
                "source": "no_model",
                **{f"eval_{key}": value for key, value in _metrics(y_true, values).items()},
            }
        )

    model_rows: list[dict] = []
    for _, row in summary.iterrows():
        if row["benchmark_type"] not in {"vsw_history", "vsw_history_shadow"}:
            continue
        run_id = str(row["run_id"])
        run_daily = (
            daily.loc[daily["run_id"] == run_id]
            .sort_values("issue_date")
            .reset_index(drop=True)
        )
        if list(run_daily["sample_id"]) != list(canonical["sample_id"]):
            raise RuntimeError(f"{run_id} 与基准样本顺序不一致")
        preds = run_daily["y_pred"].to_numpy(dtype=float)
        model_rows.append(
            {
                "family": str(row["family"]),
                "fixed_label": str(row["fixed_label"]),
                "benchmark_type": str(row["benchmark_type"]),
                "description": (
                    "风速历史基线"
                    if row["benchmark_type"] == "vsw_history"
                    else "风速历史基线＋日影"
                ),
                "alpha": float(row["alpha"]),
                "l1_ratio": (
                    float(row["l1_ratio"])
                    if str(row["family"]) == "elasticnet"
                    else None
                ),
                "shadow_abs_share": float(row["coef_shadow_abs_share"]),
                "source": "fixed_sensitivity",
                **{
                    f"eval_{key}": value
                    for key, value in _metrics(y_true, preds).items()
                },
            },
        )

    archive_arm_spec = [
        ("ridge_baseline", "ridge", "pfss_bx_baseline"),
        ("ridge_treatment", "ridge", "pfss_bx_shadow"),
        ("elasticnet_baseline", "elasticnet", "pfss_bx_baseline"),
        ("elasticnet_treatment", "elasticnet", "pfss_bx_shadow"),
    ]
    for arm, family, benchmark_type in archive_arm_spec:
        run_daily = archive_predictions[arm]
        if list(run_daily["sample_id"]) != list(canonical["sample_id"]):
            raise RuntimeError(f"archive {arm} 与基准样本顺序不一致")
        preds = run_daily["y_pred"].to_numpy(dtype=float)
        model_rows.append(
            {
                "family": family,
                "fixed_label": "E_R02_03_r1",
                "benchmark_type": benchmark_type,
                "description": (
                    "当前v3存档（PFSS＋BX基线）"
                    if benchmark_type == "pfss_bx_baseline"
                    else "当前v3存档（PFSS＋BX＋日影）"
                ),
                "archive_arm": arm,
                "source": "archive",
                "alpha": None,
                "l1_ratio": None,
                "shadow_abs_share": None,
                **{
                    f"eval_{key}": value
                    for key, value in _metrics(y_true, preds).items()
                },
            }
        )

    metrics_rows = []
    for row in baseline_rows:
        metrics_rows.append(row)
    for row in model_rows:
        metrics_rows.append(
            {
                "baseline_name": (
                    f"{row['description']} | {row['family']} | {row['fixed_label']}"
                    if row["benchmark_type"] == "vsw_history_shadow"
                    else f"{row['description']} | {row['family']} | {row['fixed_label']}"
                ),
                "description": row["description"],
                "source": f"{row['family']}::{row['fixed_label']}::{row['benchmark_type']}",
                "family": row["family"],
                "fixed_label": row["fixed_label"],
                "benchmark_type": row["benchmark_type"],
                "source_kind": row["source"],
                **{
                    key: value
                    for key, value in row.items()
                    if key.startswith("eval_")
                },
            }
        )

    gain_rows: list[dict] = []
    for row in model_rows:
        if row["source"] == "archive":
            run_daily = archive_predictions[row["archive_arm"]].copy()
        else:
            run_id = next(
                str(item["run_id"])
                for item in summary.to_dict("records")
                if item["family"] == row["family"]
                and item["fixed_label"] == row["fixed_label"]
                and item["benchmark_type"] == row["benchmark_type"]
            )
            run_daily = (
                daily.loc[daily["run_id"] == run_id]
                .sort_values("issue_date")
                .reset_index(drop=True)
            )
        treatment = run_daily[
            ["sample_id", "issue_date", "y_true", "y_pred"]
        ].copy()
        treatment["issue_date"] = pd.to_datetime(treatment["issue_date"])
        for ref_name, ref_values in ref_order.items():
            baseline = canonical[
                ["sample_id", "issue_date", "y_true"]
            ].copy()
            baseline["issue_date"] = pd.to_datetime(baseline["issue_date"])
            baseline["y_pred"] = ref_values
            bootstrap = paired_calendar_block_bootstrap(
                baseline,
                treatment,
                block_size=7,
                n_draws=N_DRAWS,
                seed=SEED,
                min_blocks=MIN_BLOCKS,
                verify_pearson=False,
            )
            gain_rows.append(
                {
                    "family": row["family"],
                    "fixed_label": row["fixed_label"],
                    "benchmark_type": row["benchmark_type"],
                    "source_kind": row["source"],
                    "reference": ref_name,
                    "model_rmse": row["eval_rmse"],
                    "reference_rmse": float(
                        np.sqrt(mean_squared_error(y_true, ref_values))
                    ),
                    "rmse_gain": float(
                        np.sqrt(mean_squared_error(y_true, ref_values))
                        - row["eval_rmse"]
                    ),
                    "rmse_ci_low": float(bootstrap["rmse_ci"][0]),
                    "rmse_ci_high": float(bootstrap["rmse_ci"][1]),
                    "stable_rmse": bool(bootstrap["stable_rmse"]),
                    "n_nonempty_blocks": int(bootstrap["n_nonempty_blocks"]),
                    "hint": (
                        "positive gain means the model improves RMSE over the reference"
                    ),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    gain_df = pd.DataFrame(gain_rows)
    metrics_csv = output_dir / "simple_baseline_metrics.csv"
    gain_csv = output_dir / "simple_baseline_gain_ci.csv"
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    gain_df.to_csv(gain_csv, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    order = metrics_df["baseline_name"].tolist()
    y_positions = np.arange(len(order))
    colors = []
    for row in metrics_df.itertuples():
        if row.source == "no_model":
            colors.append("#777777")
        elif row.source_kind == "archive":
            colors.append(
                "#2ca02c"
                if row.benchmark_type == "pfss_bx_baseline"
                else "#9467bd"
            )
        elif row.benchmark_type == "vsw_history":
            colors.append("#1f77b4")
        else:
            colors.append("#d97b29")
    ax.barh(
        y_positions,
        metrics_df["eval_rmse"].to_numpy(dtype=float),
        color=colors,
        alpha=0.92,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(metrics_df["baseline_name"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE, km/s (35-day dev_backtest)")
    ax.set_title("Simple baselines vs wind-history model arms, same 35-day dates")
    for index, value in enumerate(metrics_df["eval_rmse"]):
        ax.text(
            value + 0.15,
            index,
            f"{value:.2f}",
            va="center",
            fontsize=8,
        )
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    plot_path = output_dir / "simple_baseline_rmse.png"
    fig.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "source_run": "E_R02_03_r1",
        "data_version": "shadow-calendar-v1",
        "split_label": "dev_backtest",
        "phase_label": "迭代评估段",
        "test_start": str(canonical["issue_date"].iloc[0]),
        "test_end": str(canonical["issue_date"].iloc[-1]),
        "n_test": n,
        "time_definition": (
            "features and persistence use issue-date information; target date = "
            "issue date + 3 calendar days"
        ),
        "method": (
            "simple zero-parameter references plus already-fitted fixed-parameter "
            "arms; no hyper-parameter tuning against the dev segment; no closed-loop "
            "state update"
        ),
        "persistence_fallback_count": fallback_count,
        "expected_caveat": (
            "35-day dev evidence only, not a sealed final test"
        ),
        "summary_files": [
            metrics_csv.name,
            gain_csv.name,
            plot_path.name,
        ],
        "updated_at": "2026-10-16T18:00:00+08:00",
    }
    (output_dir / "manifest.json").write_text(
        json_dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"simple baselines -> {output_dir}")
    print("\n=== eval metrics, same 35 issue dates ===")
    print(
        metrics_df[
            [
                "baseline_name",
                "eval_rmse",
                "eval_mae",
                "eval_r2",
                "eval_pearson_r",
                "eval_pred_std",
            ]
        ].to_string(index=False)
    )
    print("\n=== RMSE gain over simple references (positive = model better) ===")
    print(
        gain_df[
            [
                "family",
                "fixed_label",
                "benchmark_type",
                "reference",
                "reference_rmse",
                "model_rmse",
                "rmse_gain",
                "rmse_ci_low",
                "rmse_ci_high",
                "stable_rmse",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
