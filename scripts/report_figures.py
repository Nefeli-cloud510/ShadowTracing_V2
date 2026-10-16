"""Render the compact question-driven chart set used by generated reports.

The charts are drawn from the archived metrics and the per-arm test
prediction CSVs of an already accepted experiment.  Nothing is refitted and
no evaluation state is changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import presentation_plots


REQUIRED_RUNS = (
    "ridge_baseline",
    "ridge_treatment",
    "elasticnet_baseline",
    "elasticnet_treatment",
)

CORE_CHART_NAMES = (
    "v3_gain_ci.png",
    "v3_test_prediction_comparison.png",
    "v3_daily_abs_error_delta.png",
)


def _load_test_prediction_frames(
    experiment_id: str,
    predictions_dir: Path,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for run_id in REQUIRED_RUNS:
        path = predictions_dir / f"{experiment_id}_{run_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing prediction csv: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["split"] = "test"
        frame["issue_date"] = pd.to_datetime(
            frame["issue_date"], errors="coerce"
        ).dt.normalize()
        frames.append({"run_id": run_id, "frame": frame})
    return frames


def _resolve_archive(
    round_root: Path,
    experiment_id: str | None,
) -> tuple[Path, dict[str, Any], str]:
    archive_root = round_root / "archive"
    candidates: list[tuple[Path, dict[str, Any]]] = []
    if archive_root.is_dir():
        for archive_dir in sorted(archive_root.iterdir()):
            metrics_path = archive_dir / "metrics.json"
            if not metrics_path.is_file():
                continue
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(metrics, dict):
                continue
            if metrics.get("experiment_type") != "model_sensitivity":
                continue
            run_ids = [str(run.get("run_id", "")) for run in metrics.get("runs", [])]
            if not set(REQUIRED_RUNS).issubset(set(run_ids)):
                continue
            candidates.append((archive_dir, metrics))

    if experiment_id:
        for archive_dir, metrics in candidates:
            if str(metrics.get("experiment_id", "")) == experiment_id:
                return archive_dir, metrics, str(metrics["experiment_id"])
        raise FileNotFoundError(
            f"no model-sensitivity archive for experiment_id={experiment_id}"
        )
    if not candidates:
        raise FileNotFoundError("no completed model-sensitivity archive found")
    archive_dir, metrics = max(
        candidates,
        key=lambda item: str(item[0].parent / item[0].name),
    )
    return archive_dir, metrics, str(metrics["experiment_id"])


def render_round_core_charts(
    *,
    session_root: str | Path,
    round_number: int,
    experiment_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    session_root = Path(session_root)
    round_root = session_root / "results" / f"round_{round_number:02d}"
    archive_dir, metrics, resolved_experiment_id = _resolve_archive(
        round_root, experiment_id
    )
    predictions_dir = (
        round_root / resolved_experiment_id / "predictions"
    )
    frames = _load_test_prediction_frames(
        resolved_experiment_id, predictions_dir
    )
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else session_root / "exports" / "figures" / f"round_{round_number:02d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{resolved_experiment_id}_r1"
    horizon = int(
        metrics["runs"][0]["parameters"]["forecast_horizon_days"]
    )
    test_start = metrics.get("test_start")

    paths = {
        "v3_gain_ci.png": presentation_plots.plot_v3_gain_ci(
            metrics,
            output_dir / "v3_gain_ci.png",
        ),
        "v3_test_prediction_comparison.png": (
            presentation_plots.plot_v3_test_prediction_comparison(
                frames,
                output_dir / "v3_test_prediction_comparison.png",
                test_start=test_start,
                horizon_days=horizon,
                run_id=run_id,
                metrics=metrics,
            )
        ),
        "v3_daily_abs_error_delta.png": (
            presentation_plots.plot_v3_daily_abs_error_delta(
                frames,
                output_dir / "v3_daily_abs_error_delta.png",
                test_start=test_start,
                horizon_days=horizon,
                run_id=run_id,
            )
        ),
    }
    manifest = {
        "schema_version": "report-charts/v1",
        "round_id": round_number,
        "experiment_id": resolved_experiment_id,
        "run_id": run_id,
        "archive_dir": str(archive_dir.relative_to(session_root)),
        "prediction_source": f"results/round_{round_number:02d}/{resolved_experiment_id}/predictions",
        "split_label": metrics.get("split_label", "dev_backtest"),
        "test_start": test_start,
        "forecast_horizon_days": horizon,
        "charts": sorted(str(path.relative_to(output_dir)) for path in paths.values()),
    }
    manifest_path = output_dir / "report_chart_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "charts": {name: str(path) for name, path in paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session",
        default=ROOT / "runtime" / "live_session" / "current",
        help="live session root",
    )
    parser.add_argument(
        "--round",
        type=int,
        required=True,
        help="round number, e.g. 2",
    )
    parser.add_argument("--experiment", help="experiment id, e.g. E_R02_03")
    parser.add_argument("--output", help="output directory")
    args = parser.parse_args()

    result = render_round_core_charts(
        session_root=args.session,
        round_number=args.round,
        experiment_id=args.experiment,
        output_dir=args.output,
    )
    for name, path in result["charts"].items():
        print(f"{name} -> {path}")
    print(f"manifest -> {result['manifest']}")


if __name__ == "__main__":
    main()
