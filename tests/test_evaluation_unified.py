import unittest

from core.evaluation_unified import (
    build_minimal_scientific_evaluation,
    build_performance_metrics,
)
from core.unified_schema import (
    BootstrapReport,
    ExecutionSummary,
    ExperimentDesign,
    ExperimentProtocol,
    ExperimentResult,
    HypothesisPrediction,
    MetricDelta,
    RobustnessOverall,
    RobustnessReport,
    TimeSliceReport,
)


def _metric_protocol(
    metric: str | None,
    *,
    expected_range: list[float] | None = None,
) -> ExperimentProtocol:
    return ExperimentProtocol(
        experiment_id="E_METRIC_01",
        round_id=1,
        task_id="ST_METRIC",
        target="Vsw",
        features=ExperimentDesign(
            target="Vsw",
            control=["By", "Bz"],
            treatment=["By", "Bz", "DeltaDec"],
            design_focus="DeltaDec",
        ),
        tested_hypotheses=["H1"],
        hypothesis_predictions={
            "H1": HypothesisPrediction(
                expected_effect="positive",
                expected_range=expected_range or [0.05, 0.07],
                metric=metric,
            )
        },
    )


def _metric_result(
    *,
    rmse_delta: float | None,
    pearson_delta: float | None,
    rmse_ci: list[float] | None,
    pearson_ci: list[float] | None,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="E_METRIC_01",
        round_id=1,
        status="completed",
        comparison=MetricDelta(rmse=rmse_delta, pearson_r=pearson_delta),
        execution=ExecutionSummary(status="completed"),
        robustness=RobustnessReport(
            bootstrap=BootstrapReport(
                rmse_ci=rmse_ci,
                pearson_r_ci=pearson_ci,
                stable=False,
            ),
            time_slice=TimeSliceReport(splits=[], stable=False),
            overall=RobustnessOverall(stable=False),
        ),
    )


class MetricSpecificEvaluationTest(unittest.TestCase):
    def test_pearson_prediction_uses_pearson_delta_and_ci(self) -> None:
        evaluation = build_minimal_scientific_evaluation(
            _metric_result(
                rmse_delta=1.2,
                pearson_delta=0.06,
                rmse_ci=[1.0, 1.5],
                pearson_ci=[0.02, 0.10],
            ),
            protocol=_metric_protocol("pearson_r"),
        )
        assessment = evaluation.hypothesis_assessments[0]

        self.assertEqual(assessment.metric, "pearson_r")
        self.assertEqual(assessment.observed_metric_delta, 0.06)
        self.assertEqual(assessment.metric_ci, [0.02, 0.10])
        self.assertTrue(assessment.magnitude_matched)
        self.assertIn("Pearson r", assessment.reason)
        self.assertNotIn("RMSE 指标", assessment.reason)

    def test_pearson_alias_maps_to_pearson_delta(self) -> None:
        evaluation = build_minimal_scientific_evaluation(
            _metric_result(
                rmse_delta=1.2,
                pearson_delta=0.06,
                rmse_ci=[1.0, 1.5],
                pearson_ci=[0.02, 0.10],
            ),
            protocol=_metric_protocol("Pearson"),
        )
        assessment = evaluation.hypothesis_assessments[0]

        self.assertEqual(assessment.metric, "pearson_r")
        self.assertEqual(assessment.observed_metric_delta, 0.06)
        self.assertTrue(assessment.magnitude_matched)

    def test_metricless_prediction_never_borrows_rmse_delta(self) -> None:
        evaluation = build_minimal_scientific_evaluation(
            _metric_result(
                rmse_delta=0.06,
                pearson_delta=0.02,
                rmse_ci=[0.02, 0.10],
                pearson_ci=[0.0, 0.04],
            ),
            protocol=_metric_protocol(None),
        )
        assessment = evaluation.hypothesis_assessments[0]

        self.assertEqual(assessment.metric, "unknown")
        self.assertIsNone(assessment.observed_metric_delta)
        self.assertIsNone(assessment.metric_ci)
        self.assertEqual(assessment.magnitude_matched, "partial")
        self.assertIn("对应指标", assessment.reason)
        self.assertNotIn("RMSE 指标", assessment.reason)

    def test_rmse_prediction_keeps_rmse_delta(self) -> None:
        evaluation = build_minimal_scientific_evaluation(
            _metric_result(
                rmse_delta=1.2,
                pearson_delta=0.06,
                rmse_ci=[1.0, 1.5],
                pearson_ci=[0.02, 0.10],
            ),
            protocol=_metric_protocol("RMSE", expected_range=[0.5, 1.5]),
        )
        assessment = evaluation.hypothesis_assessments[0]

        self.assertEqual(assessment.metric, "rmse")
        self.assertEqual(assessment.observed_metric_delta, 1.2)
        self.assertEqual(assessment.metric_ci, [1.0, 1.5])
        self.assertTrue(assessment.magnitude_matched)
        self.assertIn("RMSE", assessment.reason)

    def test_mixed_metric_ranges_skip_kl_audit(self) -> None:
        protocol = ExperimentProtocol(
            experiment_id="E_MIXED_AUDIT",
            round_id=1,
            task_id="ST_MIXED",
            target="Vsw",
            features=ExperimentDesign(
                target="Vsw",
                control=["By"],
                treatment=["By", "DeltaDec"],
            ),
            tested_hypotheses=["H1", "H2"],
            hypothesis_predictions={
                "H1": HypothesisPrediction(
                    expected_effect="positive",
                    expected_range=[0.05, 0.07],
                    metric="pearson_r",
                ),
                "H2": HypothesisPrediction(
                    expected_effect="positive",
                    expected_range=[0.10, 0.20],
                    metric="skill",
                ),
            },
        )
        result = _metric_result(
            rmse_delta=1.2,
            pearson_delta=0.06,
            rmse_ci=[1.0, 1.5],
            pearson_ci=[0.02, 0.10],
        )
        result.comparison.skill = 0.15

        metrics = build_performance_metrics(
            result,
            protocol=protocol,
            prior_supports={"H1": 0.6, "H2": 0.4},
        )
        self.assertIsNone(metrics.information_gain_kl)


if __name__ == "__main__":
    unittest.main()
