import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import report_exporter
from scripts import report_figures


REQUIRED_RUNS = report_figures.REQUIRED_RUNS


def _metrics_fixture() -> dict:
    runs = []
    for run_id in REQUIRED_RUNS:
        family = run_id.split("_", 1)[0]
        suffix = " 基线" if run_id.endswith("_baseline") else " 实验组"
        runs.append(
            {
                "run_id": run_id,
                "name": f"{family}{suffix}",
                "model_family": family,
                "feature_columns": ["ss_br_mean"],
                "n_train": 90,
                "n_test": 10,
                "parameters": {
                    "forecast_horizon_days": 3,
                },
                "metrics": {
                    "rmse": 50.0,
                    "mae": 40.0,
                    "pearson_r": 0.1,
                    "r2": -0.2,
                },
            }
        )
    contrasts = []
    for family in ("ridge", "elasticnet"):
        starter = 0.8 if family == "ridge" else 1.2
        contrasts.append(
            {
                "contrast_id": f"gain_{family}",
                "kind": "incremental_gain",
                "run_ids": [f"{family}_baseline", f"{family}_treatment"],
                "metrics_delta": {"rmse": starter},
                "bootstrap_ci": {
                    "rmse_ci": [starter - 0.5, starter + 0.5],
                    "stable": False,
                },
            }
        )
    return {
        "run_id": "E_R01_01_r1",
        "experiment_id": "E_R01_01",
        "round_id": 1,
        "status": "completed",
        "experiment_type": "model_sensitivity",
        "split_label": "dev_backtest",
        "test_start": "2021-09-01",
        "runs": runs,
        "contrasts": contrasts,
    }


def _write_fixture_session(root: Path) -> None:
    metrics = _metrics_fixture()
    archive_dir = (
        root
        / "results"
        / "round_01"
        / "archive"
        / "E_R01_01_r1"
    )
    archive_dir.mkdir(parents=True)
    (archive_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    predictions_dir = root / "results" / "round_01" / "E_R01_01" / "predictions"
    predictions_dir.mkdir(parents=True)
    dates = [
        ("2021-09-01", "2021-09-04"),
        ("2021-09-02", "2021-09-05"),
        ("2021-09-03", "2021-09-06"),
    ]
    rows = [
        f"{issue}__h3,{issue},{target},{float(400 + idx)},{float(395 + idx)}"
        for idx, (issue, target) in enumerate(dates)
    ]
    csv_text = "sample_id,issue_date,target_date,y_true,y_pred\n" + "\n".join(rows) + "\n"
    for run_id in REQUIRED_RUNS:
        (predictions_dir / f"E_R01_01_{run_id}.csv").write_text(
            csv_text, encoding="utf-8"
        )


class ReportFiguresTest(unittest.TestCase):
    def test_render_round_core_charts_writes_manifest_and_three_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture_session(root)

            result = report_figures.render_round_core_charts(
                session_root=root,
                round_number=1,
            )

            manifest_path = Path(result["manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["experiment_id"], "E_R01_01")
            for name in report_figures.CORE_CHART_NAMES:
                self.assertIn(name, result["charts"])
                path = Path(result["charts"][name])
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 1000)

            self.assertEqual(manifest["schema_version"], "report-charts/v1")
            self.assertEqual(manifest["round_id"], 1)
            self.assertEqual(manifest["experiment_id"], "E_R01_01")
            self.assertEqual(manifest["forecast_horizon_days"], 3)
            self.assertEqual(
                manifest["prediction_source"],
                "results/round_01/E_R01_01/predictions",
            )
            self.assertEqual(manifest["run_id"], "E_R01_01_r1")
            self.assertEqual(manifest["execution_run_id"], "E_R01_01_r1")
            self.assertEqual(
                set(manifest["arms"]),
                set(REQUIRED_RUNS),
            )

    def test_render_prefers_latest_run_id_when_archive_has_two_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture_session(root)
            archive_dir = (
                root
                / "results"
                / "round_01"
                / "archive"
                / "E_R01_01_r1"
            )
            rerun_dir = root / "results" / "round_01" / "archive" / "E_R01_01_r2"
            rerun_dir.mkdir(parents=True)
            metrics = json.loads(
                (archive_dir / "metrics.json").read_text(encoding="utf-8")
            )
            metrics["run_id"] = "E_R01_01_r2"
            (rerun_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            (root / "results" / "round_01" / "latest_run_id.txt").write_text(
                "E_R01_01_r2",
                encoding="utf-8",
            )

            result = report_figures.render_round_core_charts(
                session_root=root,
                round_number=1,
                experiment_id="E_R01_01",
            )

            manifest = json.loads(
                Path(result["manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["run_id"], "E_R01_01_r2")
            self.assertEqual(manifest["execution_run_id"], "E_R01_01_r2")

    def test_presentation_figures_html_redraws_when_run_id_drifts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture_session(root)

            with patch.object(report_exporter, "LIVE_ROOT", root):
                report_exporter._presentation_figures_html(1)

            figure_dir = root / "exports" / "figures" / "round_01"
            manifest_path = figure_dir / "report_chart_manifest.json"
            before_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            before_png = (figure_dir / "v3_test_prediction_comparison.png").read_bytes()
            self.assertEqual(before_manifest["run_id"], "E_R01_01_r1")

            archive_metrics = (
                root
                / "results"
                / "round_01"
                / "archive"
                / "E_R01_01_r1"
                / "metrics.json"
            )
            metrics = json.loads(archive_metrics.read_text(encoding="utf-8"))
            metrics["run_id"] = "E_R01_01_r2"
            archive_metrics.write_text(
                json.dumps(metrics, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )

            with patch.object(report_exporter, "LIVE_ROOT", root):
                report_exporter._presentation_figures_html(1)

            after_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            after_png = (figure_dir / "v3_test_prediction_comparison.png").read_bytes()
            self.assertEqual(after_manifest["run_id"], "E_R01_01_r2")
            self.assertNotEqual(
                hashlib.sha256(before_png).hexdigest(),
                hashlib.sha256(after_png).hexdigest(),
            )

    def test_presentation_figures_html_embeds_existing_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            figure_dir = root / "exports" / "figures" / "round_01"
            figure_dir.mkdir(parents=True)
            for name in report_figures.CORE_CHART_NAMES:
                (figure_dir / name).write_bytes(b"\x89PNG\r\n\x1a\nfakepngdata")

            with patch.object(report_exporter, "LIVE_ROOT", root):
                html = report_exporter._presentation_figures_html(
                    1, include=("v3_gain_ci.png",)
                )

            self.assertIn("<figure class=\"figure\">", html)
            self.assertIn("data:image/png;base64,", html)
            self.assertNotIn("figure-grid--compact", html)

    def test_presentation_figures_html_compact_grid_and_missing_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            figure_dir = root / "exports" / "figures" / "round_01"
            figure_dir.mkdir(parents=True)
            (figure_dir / "v3_gain_ci.png").write_bytes(b"png")

            with patch.object(report_exporter, "LIVE_ROOT", root):
                compact = report_exporter._presentation_figures_html(
                    1, include=("v3_gain_ci.png",), compact=True
                )
                missing = report_exporter._presentation_figures_html(
                    2, include=("v3_gain_ci.png",)
                )

            self.assertIn("figure-grid figure-grid--compact", compact)
            self.assertEqual(missing, "")


if __name__ == "__main__":
    unittest.main()
