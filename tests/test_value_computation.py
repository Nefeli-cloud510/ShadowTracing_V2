import unittest

from core.decision_unified import (
    _build_hypothesis_predictions,
    _formal_expected_performance_gain,
    _formal_information_gain,
    _prediction_overlap,
)
from core.evaluation_unified import (
    build_performance_metrics,
    evaluate_experiment,
)
from core.unified_schema import (
    CandidateExperiment,
    ExecutionSummary,
    ExperimentDesign,
    ExperimentProtocol,
    ExperimentResult,
    ExperimentRun,
    HypothesisNode,
    HypothesisPrediction,
    MetricDelta,
    PredictionRecord,
    RunMetrics,
    UncertaintyRecord,
)


def _hypothesis_node(
    hypothesis_id: str,
    *,
    support_score: float,
    direction: str,
    expected_range: list[float],
) -> HypothesisNode:
    return HypothesisNode(
        hypothesis_id=hypothesis_id,
        display_hypothesis_id=hypothesis_id.replace("H", "H"),
        statement=f"{hypothesis_id} 的竞争性科学假设",
        level=1,
        status="active",
        support_score=support_score,
        activation_condition="始终激活",
        predictions=[
            PredictionRecord(
                experiment_id=f"{hypothesis_id}_llm_pred",
                metric="Skill",
                expected_direction=direction,
                expected_range=expected_range,
            )
        ],
    )


def _uncertainty_record(*, related_hypotheses: list[str]) -> UncertaintyRecord:
    return UncertaintyRecord(
        uncertainty_id="U_IG_01",
        question="日影南北偏移是否带来独立预测信息？",
        description="区分独立信息与路径变量伴随相关。",
        related_hypotheses=related_hypotheses,
        status="active",
        priority="high",
        created_at_round=1,
        created_by="test",
    )


def _candidate(*, predictions: dict[str, HypothesisPrediction]) -> CandidateExperiment:
    return CandidateExperiment(
        experiment_id="E_IG_01",
        type="distinguishing",
        purpose="检验日影南北偏移的增量预测信息",
        tested_hypotheses=list(predictions.keys()),
        design=ExperimentDesign(
            target="Vsw",
            control=["By"],
            treatment=["By", "DeltaDec"],
            lags={"By": [1, 3], "DeltaDec": [1, 3]},
            design_focus="DeltaDec",
        ),
        hypothesis_predictions=predictions,
        requires_human_review=True,
    )


class ValueComputationTest(unittest.TestCase):
    def test_llm_expected_ranges_enter_information_gain(self) -> None:
        nodes = {
            "H1": _hypothesis_node("H1", support_score=0.6, direction="positive", expected_range=[0.05, 0.06]),
            "H2": _hypothesis_node("H2", support_score=0.6, direction="negative", expected_range=[-0.06, -0.05]),
        }
        record = _uncertainty_record(related_hypotheses=["H1", "H2"])
        predictions = _build_hypothesis_predictions(
            record=record,
            node_index=nodes,
            design_focus="DeltaDec",
        )
        self.assertEqual(predictions["H1"].expected_range, [0.05, 0.06])
        self.assertEqual(predictions["H1"].metric, "Skill")
        self.assertEqual(predictions["H2"].expected_effect, "negative")

        ig, rationale = _formal_information_gain(
            _candidate(predictions=predictions),
            current_round=1,
        )
        self.assertGreater(ig, 0.70)
        self.assertIn("LLM 输出", rationale)
        self.assertIn("Skill", rationale)

    def test_overlapping_llm_ranges_give_low_ig(self) -> None:
        nodes = {
            "H1": _hypothesis_node("H1", support_score=0.6, direction="positive", expected_range=[0.0, 0.1]),
            "H2": _hypothesis_node("H2", support_score=0.6, direction="positive", expected_range=[0.0, 0.1]),
        }
        record = _uncertainty_record(related_hypotheses=["H1", "H2"])
        predictions = _build_hypothesis_predictions(
            record=record,
            node_index=nodes,
            design_focus="DeltaDec",
        )
        ig, _ = _formal_information_gain(
            _candidate(predictions=predictions),
            current_round=1,
        )
        self.assertLess(ig, 0.05)

    def test_post_experiment_kl_audit_is_written_to_metrics(self) -> None:
        protocol = ExperimentProtocol(
            experiment_id="E_AUDIT_01",
            round_id=2,
            task_id="ST_AUDIT",
            target="Vsw",
            features=ExperimentDesign(
                target="Vsw",
                control=["By"],
                treatment=["By", "DeltaDec"],
                lags={"By": [1, 3], "DeltaDec": [1, 3]},
                design_focus="DeltaDec",
            ),
            tested_hypotheses=["H1", "H2"],
            hypothesis_predictions={
                "H1": HypothesisPrediction(
                    expected_effect="positive",
                    expected_range=[0.03, 0.07],
                    metric="Skill",
                ),
                "H2": HypothesisPrediction(
                    expected_effect="near_zero",
                    expected_range=[-0.01, 0.01],
                    metric="Skill",
                ),
            },
        )
        result = ExperimentResult(
            experiment_id="E_AUDIT_01",
            round_id=2,
            status="completed",
            runs=[
                ExperimentRun(
                    run_id="baseline",
                    name="baseline",
                    metrics=RunMetrics(rmse=10.0, skill=0.20),
                ),
                ExperimentRun(
                    run_id="treatment",
                    name="treatment",
                    metrics=RunMetrics(rmse=9.5, skill=0.25),
                ),
            ],
            comparison=MetricDelta(skill=0.05, pearson_r=0.02, rmse=0.5),
            execution=ExecutionSummary(status="completed"),
        )
        metrics = build_performance_metrics(
            result,
            protocol=protocol,
            prior_supports={"H1": 0.6, "H2": 0.4},
        )
        self.assertIsNotNone(metrics.information_gain_kl)
        self.assertGreater(metrics.information_gain_kl, 0.0)
        self.assertGreater(metrics.ig_posterior_probs["H1"], metrics.ig_prior_probs["H1"])

    def test_mixed_statistic_metrics_are_not_compared(self) -> None:
        predictions = {
            "H_shadow_incremental_gain": HypothesisPrediction(
                expected_effect="positive",
                expected_range=[0.02, 0.12],
                metric="Pearson_r_delta",
            ),
            "H_window_stability": HypothesisPrediction(
                expected_effect="positive",
                expected_range=[0.75, 1.0],
                metric="stability_rate",
            ),
        }
        ig, rationale = _formal_information_gain(
            _candidate(predictions=predictions),
            current_round=3,
        )
        self.assertEqual(ig, 0.5)
        self.assertIn("统计量/语义不一致", rationale)
        self.assertIsNone(
            _prediction_overlap(
                [0.02, 0.12],
                [0.75, 1.0],
                left_metric="Pearson_r_delta",
                right_metric="stability_rate",
            )
        )

    def test_pearson_absolute_and_pearson_delta_are_incomparable(self) -> None:
        predictions = {
            "H_shadow_incremental_gain": HypothesisPrediction(
                expected_effect="positive",
                expected_range=[0.02, 0.12],
                metric="Pearson_r_absolute",
            ),
            "H_window_stability": HypothesisPrediction(
                expected_effect="positive",
                expected_range=[0.75, 1.0],
                metric="Pearson_r_delta",
            ),
        }
        ig, rationale = _formal_information_gain(
            _candidate(predictions=predictions),
            current_round=3,
        )
        self.assertEqual(ig, 0.5)
        self.assertIn("统计量/语义不一致", rationale)

    def test_expected_pg_is_zero_in_first_round(self) -> None:
        predictions = {
            "H1": HypothesisPrediction(expected_effect="positive", expected_range=[0.05, 0.06]),
            "H2": HypothesisPrediction(expected_effect="negative", expected_range=[-0.06, -0.05]),
        }
        pg, rationale = _formal_expected_performance_gain(
            candidate=_candidate(predictions=predictions),
            experiment_memory=None,
            current_round=1,
        )
        self.assertEqual(pg, 0.0)
        self.assertIn("置为 0", rationale)

    def test_evaluate_experiment_accepts_prior_supports(self) -> None:
        protocol = ExperimentProtocol(
            experiment_id="E_AUDIT_02",
            round_id=2,
            task_id="ST_AUDIT",
            target="Vsw",
            features=ExperimentDesign(
                target="Vsw",
                control=["By"],
                treatment=["By", "DeltaDec"],
                lags={"By": [1, 3], "DeltaDec": [1, 3]},
                design_focus="DeltaDec",
            ),
            tested_hypotheses=["H1", "H2"],
            hypothesis_predictions={
                "H1": HypothesisPrediction(
                    expected_effect="positive",
                    expected_range=[0.03, 0.07],
                    metric="Skill",
                ),
                "H2": HypothesisPrediction(
                    expected_effect="near_zero",
                    expected_range=[-0.01, 0.01],
                    metric="Skill",
                ),
            },
        )
        result = ExperimentResult(
            experiment_id="E_AUDIT_02",
            round_id=2,
            status="completed",
            runs=[],
            comparison=MetricDelta(skill=0.05),
            execution=ExecutionSummary(status="completed"),
        )
        evaluation = evaluate_experiment(
            result,
            protocol=protocol,
            prior_supports={"H1": 0.6, "H2": 0.4},
        )
        self.assertIsNotNone(evaluation.metrics.information_gain_kl)


if __name__ == "__main__":
    unittest.main()
