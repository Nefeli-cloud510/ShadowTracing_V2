"""Pearson-focused companion diagnostics for the v3 fixed-parameter runs.

This script only consumes the already-generated fixed-parameter daily
forecasts; it does not retrain models and does not touch closed-loop state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.json_safe import json_dumps
from core.robustness_diagnostics import paired_calendar_block_bootstrap


OUTPUT_DIR = ROOT / "artifacts" / "v3_fixed_param_sensitivity"
DAILY_CSV = OUTPUT_DIR / "fixed_param_daily.csv"
COMPARISON_CSV = OUTPUT_DIR / "fixed_param_gain_comparison.csv"
SUMMARY_CSV = OUTPUT_DIR / "fixed_param_summary.csv"

N_DRAWS = 4000
SEED = 20261008
MIN_BLOCKS = 2


def _unavailable_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float | None, str | None]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) <= 1:
        return None, "insufficient_samples"
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = float(np.corrcoef(y_true, y_pred)[0, 1])
    if np.isfinite(raw):
        return raw, None
    if float(np.std(y_true)) <= 1e-12:
        return None, "constant_target"
    if float(np.std(y_pred)) <= 1e-12:
        return None, "constant_prediction"
    return None, "undefined"


def _slice_metrics(frame: pd.DataFrame) -> dict:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_base = frame["y_base"].to_numpy(dtype=float)
    y_shadow = frame["y_shadow"].to_numpy(dtype=float)
    pearson_base, base_reason = _unavailable_pearson(y_true, y_base)
    pearson_shadow, shadow_reason = _unavailable_pearson(y_true, y_shadow)
    base_rmse = float(np.sqrt(np.mean((y_true - y_base) ** 2)))
    shadow_rmse = float(np.sqrt(np.mean((y_true - y_shadow) ** 2)))
    return {
        "n": int(len(frame)),
        "start_date": str(frame["issue_date"].min()),
        "end_date": str(frame["issue_date"].max()),
        "rmse_base": base_rmse,
        "rmse_shadow": shadow_rmse,
        "rmse_gain": base_rmse - shadow_rmse,
        "pearson_base": pearson_base,
        "pearson_base_reason": base_reason,
        "pearson_shadow": pearson_shadow,
        "pearson_shadow_reason": shadow_reason,
        "pearson_gain": (
            pearson_shadow - pearson_base
            if pearson_base is not None and pearson_shadow is not None
            else None
        ),
    }


def main() -> int:
    daily = pd.read_csv(DAILY_CSV, dtype={"sample_id": str})
    summary = pd.read_csv(SUMMARY_CSV, dtype={"run_id": str})
    comparison = pd.read_csv(COMPARISON_CSV, dtype={"family": str})
    daily["issue_date"] = pd.to_datetime(daily["issue_date"]).dt.strftime(
        "%Y-%m-%d"
    )

    pearson_rows: list[dict] = []
    slice_rows: list[dict] = []
    for _, comparison_row in comparison.iterrows():
        family = str(comparison_row["family"])
        fixed_label = str(comparison_row["fixed_label"])
        alpha = float(comparison_row["alpha"])
        l1_ratio = (
            float(comparison_row["l1_ratio"])
            if family == "elasticnet"
            else None
        )
        matching = summary.loc[
            (summary["family"] == family)
            & (np.isclose(summary["alpha"], alpha))
            & (
                summary["l1_ratio"].isna()
                if l1_ratio is None
                else np.isclose(summary["l1_ratio"], l1_ratio)
            )
        ]
        if matching.empty:
            raise RuntimeError(f"{family} {fixed_label} 在 summary 中缺失")
        base_id = next(
            row.run_id
            for row in matching.itertuples()
            if row.benchmark_type == "vsw_history"
        )
        shadow_id = next(
            row.run_id
            for row in matching.itertuples()
            if row.benchmark_type == "vsw_history_shadow"
        )

        base = daily.loc[daily["run_id"] == base_id].copy()
        shadow = daily.loc[daily["run_id"] == shadow_id].copy()
        paired = pd.merge(
            base[["sample_id", "issue_date", "target_date", "y_true", "y_pred"]],
            shadow[["sample_id", "y_pred"]],
            on="sample_id",
            suffixes=("", "_shadow"),
        )
        paired = paired.rename(
            columns={"y_pred": "y_base", "y_pred_shadow": "y_shadow"}
        )
        paired = paired.sort_values("issue_date").reset_index(drop=True)
        if len(paired) == 0:
            raise RuntimeError(f"{base_id}/{shadow_id} 无配对样本")
        paired["issue_date"] = pd.to_datetime(paired["issue_date"])
        x_true = paired["y_true"].to_numpy(dtype=float)

        bootstrap = paired_calendar_block_bootstrap(
            paired.rename(columns={"y_base": "y_pred"}),
            paired.rename(columns={"y_shadow": "y_pred"}),
            block_size=7,
            n_draws=N_DRAWS,
            seed=SEED,
            min_blocks=MIN_BLOCKS,
            verify_pearson=True,
        )
        pearson_r_base, base_reason = _unavailable_pearson(
            x_true, paired["y_base"].to_numpy(dtype=float)
        )
        pearson_r_shadow, shadow_reason = _unavailable_pearson(
            x_true, paired["y_shadow"].to_numpy(dtype=float)
        )
        point_gain = float(bootstrap["point_gain_pearson_r"])
        ci_low, ci_high = (float(item) for item in bootstrap["pearson_r_ci"])
        valid_draws = int(bootstrap["n_valid_pearson_draws"])
        stable_pearson = bool(
            (not bootstrap["insufficient_blocks"])
            and valid_draws == N_DRAWS
            and np.isfinite(ci_low)
            and np.isfinite(ci_high)
            and (ci_low > 0 or ci_high < 0)
        )
        pearson_rows.append(
            {
                "family": family,
                "fixed_label": fixed_label,
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "n_test": int(len(paired)),
                "pearson_base": pearson_r_base,
                "pearson_base_reason": base_reason,
                "pearson_shadow": pearson_r_shadow,
                "pearson_shadow_reason": shadow_reason,
                "pearson_gain": point_gain,
                "pearson_ci_low": ci_low,
                "pearson_ci_high": ci_high,
                "n_valid_pearson_draws": valid_draws,
                "n_nonempty_blocks": int(bootstrap["n_nonempty_blocks"]),
                "stable_pearson": stable_pearson,
            }
        )

        n = len(paired)
        boundaries = (0, 12, n - 12, n)
        for index, (start, end) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        ):
            part = paired.iloc[start:end]
            slice_rows.append(
                {
                    "family": family,
                    "fixed_label": fixed_label,
                    "slice": ["early", "middle", "late"][index],
                    **_slice_metrics(part),
                }
            )

    pearson_df = pd.DataFrame(pearson_rows)
    slice_df = pd.DataFrame(slice_rows)
    pearson_csv = OUTPUT_DIR / "fixed_param_pearson.csv"
    slice_csv = OUTPUT_DIR / "fixed_param_pearson_time_slices.csv"
    pearson_df.to_csv(pearson_csv, index=False, encoding="utf-8-sig")
    slice_df.to_csv(slice_csv, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 4.4),
        constrained_layout=True,
        sharey=True,
    )
    y_positions = np.arange(len(pearson_df))
    pearson_r = pearson_df["pearson_gain"].to_numpy(dtype=float)
    ci_low = pearson_df["pearson_ci_low"].to_numpy(dtype=float)
    ci_high = pearson_df["pearson_ci_high"].to_numpy(dtype=float)
    colors = [
        "#1f77b4" if family == "ridge" else "#d97b29"
        for family in pearson_df["family"]
    ]
    labels = [
        label.replace("alpha=", "a=").replace("l1=", "l1=")
        for label in pearson_df["fixed_label"]
    ]

    ax = axes[0]
    for index, point_y in enumerate(y_positions):
        ax.errorbar(
            [pearson_r[index]],
            [point_y],
            xerr=[
                [pearson_r[index] - ci_low[index]],
                [ci_high[index] - pearson_r[index]],
            ],
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            capsize=3,
            markersize=6,
        )
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Pearson r gain (shadow - baseline)")
    ax.set_title("Pearson r gain, 95% block bootstrap CI")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)

    ax2 = axes[1]
    pivot = slice_df.pivot_table(
        index=["family", "fixed_label"],
        columns="slice",
        values="pearson_gain",
    ).reindex(
        [(family, label) for family, label in zip(pearson_df["family"], pearson_df["fixed_label"])]
    )
    width = 0.22
    positions = np.arange(len(pivot))
    for offset, column in zip((-width, 0, width), pivot.columns):
        ax2.barh(
            positions + offset,
            pivot[column].to_numpy(dtype=float),
            height=width,
            label=column,
        )
    ax2.axvline(0, color="#444444", linewidth=1)
    ax2.set_yticks(positions)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Pearson r gain (shadow - baseline)")
    ax2.set_title("Gain by early / middle / late slice")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(axis="x", color="#dddddd", linewidth=0.7)

    plot_path = OUTPUT_DIR / "fixed_param_pearson_gain_ci.png"
    fig.savefig(plot_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest = json_loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary_files"] = sorted(
        set(manifest.get("summary_files", []))
        | {
            "fixed_param_pearson.csv",
            "fixed_param_pearson_time_slices.csv",
            "fixed_param_pearson_gain_ci.png",
        }
    )
    manifest["updated_at"] = "2026-10-16T16:00:00+08:00"
    manifest["audit_note"] = (
        "fixed_param_pearson.csv uses the same paired 7-day block bootstrap "
        f"(n_draws={N_DRAWS}, seed={SEED}, min_blocks={MIN_BLOCKS}); "
        "Pearson gain is shadow - baseline inside the 35-day dev_backtest segment only."
    )
    manifest_path.write_text(json_dumps(manifest, indent=2), encoding="utf-8")

    print(f"Pearson diagnostics -> {OUTPUT_DIR}")
    print("\n=== Pearson gain (positive = shadow has higher linear correlation) ===")
    print(
        pearson_df[
            [
                "family",
                "fixed_label",
                "pearson_base",
                "pearson_shadow",
                "pearson_gain",
                "pearson_ci_low",
                "pearson_ci_high",
                "stable_pearson",
            ]
        ].to_string(index=False)
    )
    print("\n=== Pearson gain by time slice ===")
    print(
        slice_df[
            [
                "family",
                "fixed_label",
                "slice",
                "n",
                "rmse_gain",
                "pearson_base",
                "pearson_shadow",
                "pearson_gain",
            ]
        ].to_string(index=False)
    )
    return 0


def json_loads(text: str):
    import json

    return json.loads(text)


if __name__ == "__main__":
    raise SystemExit(main())
