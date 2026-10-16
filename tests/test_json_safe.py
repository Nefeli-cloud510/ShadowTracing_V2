import json
import tempfile
import unittest
from pathlib import Path

from core.json_safe import json_dumps, to_json_safe
from core.result_unified import export_result_bundle
from core.unified_schema import (
    ExecutionSummary,
    ExperimentDesign,
    ExperimentProtocol,
    ExperimentResult,
    ExperimentRun,
    RunMetrics,
)


class JsonSafeTest(unittest.TestCase):
    def test_non_finite_scalars_become_null(self) -> None:
        payload = {
            "ok": 1.5,
            "missing": None,
            "nan": float("nan"),
            "inf": float("inf"),
            "negative_inf": float("-inf"),
            "nested": {"pearson_r": float("nan")},
        }

        safe = to_json_safe(payload)
        raw = json.dumps(json.loads(json_dumps(payload)))

        self.assertEqual(safe["nan"], None)
        self.assertEqual(safe["inf"], None)
        self.assertEqual(safe["negative_inf"], None)
        self.assertEqual(safe["nested"]["pearson_r"], None)
        self.assertEqual(raw.count("NaN"), 0)

    def test_export_result_bundle_is_parseable_with_nan_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = ExperimentProtocol(
                experiment_id="E_JSON_SAFE",
                round_id=1,
                task_id="T",
                target="Vsw",
                features=ExperimentDesign(
                    target="Vsw",
                    control=["By"],
                    treatment=["By", "DeltaDec"],
                ),
            )
            result = ExperimentResult(
                experiment_id="E_JSON_SAFE",
                round_id=1,
                status="completed",
                execution=ExecutionSummary(status="completed"),
                runs=[
                    ExperimentRun(
                        run_id="baseline",
                        name="baseline",
                        metrics=RunMetrics(
                            rmse=1.25,
                            pearson_r=float("nan"),
                            unavailable_reasons={
                                "pearson_r": "constant_prediction"
                            },
                        ),
                    ),
                    ExperimentRun(
                        run_id="treatment",
                        name="treatment",
                        metrics=RunMetrics(rmse=1.10, pearson_r=0.92),
                    ),
                ],
            )

            export_result_bundle(
                result,
                protocol,
                round_dir=root / "results" / "round_01" / "E_JSON_SAFE",
                latest_protocol_path=root / "latest_protocol.json",
                latest_metrics_path=root / "latest_metrics.json",
            )

            for path in [
                root / "results" / "round_01" / "E_JSON_SAFE" / "result_unified.json",
                root / "latest_metrics.json",
            ]:
                payload = json.loads(path.read_text(encoding="utf-8"))
                run_entry = payload["runs"][0]
                baseline_metrics = (
                    run_entry["metrics"] if "metrics" in run_entry else run_entry
                )
                self.assertEqual(baseline_metrics["rmse"], 1.25)
                self.assertIsNone(baseline_metrics["pearson_r"])
                self.assertEqual(
                    baseline_metrics["unavailable_reasons"]["pearson_r"],
                    "constant_prediction",
                )


if __name__ == "__main__":
    unittest.main()
