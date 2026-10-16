"""Render presentation figures from the accepted E_R02_03 archive run.

The main prediction chart in v3_quality_figures stays as a diagnostic;
this script produces the question-driven slides: RMSE-gain CI,
dev_backtest-period prediction comparison, and appendix figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import presentation_plots, quality_plots
from scripts.render_v3_quality_figures import (
    reconstruct_v3_prediction_frames,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session",
        default=(
            ROOT
            / "runtime"
            / "live_session"
            / "sessions"
            / "session_20261015_132955_705440"
        ),
        help="live session root",
    )
    parser.add_argument(
        "--round",
        default="round_02",
        help="round directory",
    )
    parser.add_argument(
        "--experiment",
        default="E_R02_03",
        help="experiment id",
    )
    parser.add_argument(
        "--output",
        default=ROOT / "artifacts" / "v3_presentation_figures",
        help="output directory",
    )
    args = parser.parse_args()

    session = Path(args.session)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = (
        session
        / "results"
        / args.round
        / "archive"
        / f"{args.experiment}_r1"
    )
    predictions_dir = (
        session
        / "results"
        / args.round
        / args.experiment
        / "predictions"
    )
    omni_path = (
        session / "data" / "raw" / "omni_2021_full_calendar_v3.csv"
    )
    lhaaso_path = (
        session / "data" / "raw" / "lhaaso_shadow_pfss_2021_full_calendar_v3.csv"
    )
    if not omni_path.exists() or not lhaaso_path.exists():
        raise FileNotFoundError(f"v3 raw data not found under {omni_path.parent}")
    if not archive_dir.exists():
        raise FileNotFoundError(f"archive not found: {archive_dir}")

    calendar = quality_plots.load_calendar_frame(omni_path, lhaaso_path)
    manifest = quality_plots.load_sample_manifest(archive_dir)
    stats = quality_plots.compute_quality_stats(calendar, manifest)
    predictions = reconstruct_v3_prediction_frames(
        experiment_id=args.experiment,
        archive_dir=archive_dir,
        predictions_dir=predictions_dir,
        output_dir=output_dir,
    )
    metrics = json.loads(
        (archive_dir / "metrics.json").read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        (archive_dir / "diagnostics.json").read_text(encoding="utf-8")
    )

    gain_path = presentation_plots.plot_v3_gain_ci(
        metrics,
        output_dir / "v3_gain_ci.png",
    )
    comparison_path = presentation_plots.plot_v3_test_prediction_comparison(
        predictions,
        output_dir / "v3_test_prediction_comparison.png",
        test_start=metrics.get("test_start"),
        test_end=None,
        horizon_days=int(
            metrics["runs"][0]["parameters"]["forecast_horizon_days"]
        ),
        run_id=f"{args.experiment}_r1",
        metrics=metrics,
    )
    daily_delta_path = presentation_plots.plot_v3_daily_abs_error_delta(
        predictions,
        output_dir / "v3_daily_abs_error_delta.png",
        test_start=metrics.get("test_start"),
        test_end=None,
        horizon_days=int(
            metrics["runs"][0]["parameters"]["forecast_horizon_days"]
        ),
        run_id=f"{args.experiment}_r1",
    )
    full_overview_path = presentation_plots.plot_v3_appendix_full_calendar_overview(
        calendar,
        predictions,
        output_dir / "v3_appendix_full_calendar_overview.png",
        test_start=metrics.get("test_start"),
        test_end=None,
        horizon_days=int(
            metrics["runs"][0]["parameters"]["forecast_horizon_days"]
        ),
        run_id=f"{args.experiment}_r1",
    )
    appendix_path = presentation_plots.plot_v3_appendix_training_quality(
        calendar,
        predictions,
        output_dir / "v3_appendix_training_quality.png",
        stats=stats,
        manifest=manifest,
    )
    params_path = presentation_plots.plot_v3_parameters_annex(
        metrics,
        diagnostics,
        archive_dir,
        output_dir / "v3_appendix_parameters.png",
    )
    print(f"gain CI figure -> {gain_path}")
    print(f"dev_backtest comparison figure -> {comparison_path}")
    print(f"daily abs error delta figure -> {daily_delta_path}")
    print(f"full calendar overview -> {full_overview_path}")
    print(f"appendix figure -> {appendix_path}")
    print(f"parameters annex -> {params_path} / {params_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
