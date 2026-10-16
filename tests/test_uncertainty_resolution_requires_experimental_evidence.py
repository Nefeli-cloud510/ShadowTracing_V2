import tempfile
import unittest
from pathlib import Path

from core.evaluation_unified import _build_disagreement_updates, build_minimal_scientific_evaluation
from core.state_repository import UnifiedStateRepository
from core.state_updater import UnifiedStateUpdater
from core.support_update_rules import (
    EVIDENCE_IDENTICAL_PREDICTIONS,
    EVIDENCE_INCONCLUSIVE,
    EVIDENCE_STABLE_IMPROVEMENT,
)
from core.unified_schema import (
    ExecutionSummary,
    DisagreementUpdate,
    ExperimentDesign,
    ExperimentProtocol,
    ExperimentResult,
    EvaluationSpec,
    HypothesisAssessment,
    HypothesisNode,
    ExperimentRun,
    MetricDelta,
    ResearchQuestion,
    RobustnessOverall,
    RobustnessReport,
    ScientificConstraints,
    ScientificTask,
    ScientificTaskPayload,
    UncertaintyRecord,
    UncertaintyDisagreement,
    VariableBinding,
)


def _minimal_protocol() -> ExperimentProtocol:
    return ExperimentProtocol(
        experiment_id="E_STATE_GATE",
        round_id=1,
        task_id="ST_GATE_STATE",
        target="Vsw",
        features=ExperimentDesign(
            target="Vsw",
            control=["By"],
            treatment=["By", "DeltaDec"],
        ),
        tested_hypotheses=["H1", "H2"],
        target_uncertainties=["U1"],
        disagreement_context={
            "U1": {
                "H1": UncertaintyDisagreement(
                    hypothesis_id="H1",
                    expected="positive",
                    status="active",
                    support_score=0.8,
                ),
                "H2": UncertaintyDisagreement(
                    hypothesis_id="H2",
                    expected="negative",
                    status="active",
                    support_score=0.4,
                ),
            }
        },
    )


def _minimal_result() -> ExperimentResult:
    return ExperimentResult(
        experiment_id="E_STATE_GATE",
        round_id=1,
        status="completed",
        runs=[
            ExperimentRun(run_id="baseline", name="baseline"),
            ExperimentRun(run_id="treatment", name="treatment"),
        ],
        comparison=MetricDelta(rmse=0.0, pearson_r=0.0),
        execution=ExecutionSummary(status="completed", duration_seconds=1.0),
        robustness=RobustnessReport(overall=RobustnessOverall(stable=False)),
    )


def _protocol(*, target_uncertainties: list[str] | None = None) -> ExperimentProtocol:
    return ExperimentProtocol(
        experiment_id="E_EVIDENCE_GATE",
        round_id=1,
        task_id="ST_GATE",
        target="Vsw",
        features=ExperimentDesign(
            target="Vsw",
            control=["By"],
            treatment=["By", "DeltaDec"],
        ),
        tested_hypotheses=["H1", "H2"],
        target_uncertainties=target_uncertainties or ["U1"],
        disagreement_context={
            "U1": {
                "H1": UncertaintyDisagreement(
                    hypothesis_id="H1",
                    expected="positive",
                    status="active",
                    support_score=0.8,
                ),
                "H2": UncertaintyDisagreement(
                    hypothesis_id="H2",
                    expected="negative",
                    status="active",
                    support_score=0.4,
                ),
            }
        },
    )


def _assessment(
    hypothesis_id: str,
    *,
    support_after: float,
    evidence_classification: str | None,
    updated: bool,
) -> HypothesisAssessment:
    return HypothesisAssessment(
        hypothesis_id=hypothesis_id,
        support_before=support_after,
        support_after=support_after,
        status="partially_supported",
        direction_matched="partial",
        magnitude_matched="partial",
        reason="test assessment",
        evidence_classification=evidence_classification,
        updated=updated,
    )


def _stable_assessments() -> list[HypothesisAssessment]:
    return [
        _assessment(
            "H1",
            support_after=0.85,
            evidence_classification=EVIDENCE_STABLE_IMPROVEMENT,
            updated=True,
        ),
        _assessment(
            "H2",
            support_after=0.35,
            evidence_classification=EVIDENCE_STABLE_IMPROVEMENT,
            updated=True,
        ),
    ]


class UncertaintyResolutionRequiresExperimentalEvidenceTest(unittest.TestCase):
    def test_no_experiment_assessment_never_resolves_from_score_gap(self) -> None:
        updates = _build_disagreement_updates(_protocol(), assessments=[])

        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].resolution_status, "unresolved")
        self.assertEqual(
            updates[0].evidence_basis,
            "missing_valid_experiment_evidence",
        )
        self.assertIn("不能仅凭既有分数差距关闭", updates[0].summary)

    def test_valid_experiment_evidence_can_resolve_same_uncertainty(self) -> None:
        updates = _build_disagreement_updates(
            _protocol(),
            assessments=_stable_assessments(),
        )

        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].resolution_status, "resolved")
        self.assertEqual(updates[0].evidence_basis, "stable_experiment_evidence")

    def test_valid_experiment_evidence_can_partially_resolve_when_narrowed(self) -> None:
        assessments = [
            _assessment(
                "H1",
                support_after=0.55,
                evidence_classification=EVIDENCE_STABLE_IMPROVEMENT,
                updated=True,
            ),
            _assessment(
                "H2",
                support_after=0.4,
                evidence_classification=EVIDENCE_STABLE_IMPROVEMENT,
                updated=False,
            ),
        ]
        updates = _build_disagreement_updates(_protocol(), assessments=assessments)

        self.assertEqual(updates[0].resolution_status, "partially_resolved")
        self.assertEqual(updates[0].evidence_basis, "stable_experiment_evidence")

    def test_inconclusive_experiment_does_not_resolve(self) -> None:
        assessments = [
            _assessment(
                "H1",
                support_after=0.8,
                evidence_classification=EVIDENCE_INCONCLUSIVE,
                updated=False,
            ),
            _assessment(
                "H2",
                support_after=0.4,
                evidence_classification=EVIDENCE_INCONCLUSIVE,
                updated=False,
            ),
        ]
        updates = _build_disagreement_updates(_protocol(), assessments=assessments)

        self.assertEqual(updates[0].resolution_status, "unresolved")
        self.assertEqual(
            updates[0].evidence_basis,
            "missing_valid_experiment_evidence",
        )

    def test_identical_predictions_do_not_close_uncertainty(self) -> None:
        assessments = [
            _assessment(
                "H1",
                support_after=0.8,
                evidence_classification=EVIDENCE_IDENTICAL_PREDICTIONS,
                updated=False,
            ),
            _assessment(
                "H2",
                support_after=0.4,
                evidence_classification=EVIDENCE_IDENTICAL_PREDICTIONS,
                updated=False,
            ),
        ]
        updates = _build_disagreement_updates(_protocol(), assessments=assessments)

        self.assertEqual(updates[0].resolution_status, "unresolved")

    def test_untargeted_hypothesis_result_does_not_resolve_uncertainty(self) -> None:
        assessments = [
            _assessment(
                "H3",
                support_after=0.9,
                evidence_classification=EVIDENCE_STABLE_IMPROVEMENT,
                updated=True,
            )
        ]
        updates = _build_disagreement_updates(_protocol(), assessments=assessments)

        self.assertEqual(updates[0].resolution_status, "unresolved")

    def test_public_evaluation_without_tested_hypotheses_does_not_resolve(self) -> None:
        protocol = _protocol()
        protocol.tested_hypotheses = []
        result = ExperimentResult(
            experiment_id="E_EVIDENCE_GATE",
            round_id=1,
            status="failed",
            comparison=MetricDelta(),
            execution=ExecutionSummary(status="failed"),
            robustness=RobustnessReport(overall=RobustnessOverall(stable=False)),
        )

        evaluation = build_minimal_scientific_evaluation(
            result,
            protocol=protocol,
            prior_supports={"H1": 0.8, "H2": 0.4},
        )
        self.assertEqual(len(evaluation.disagreement_updates), 1)
        self.assertEqual(
            evaluation.disagreement_updates[0].resolution_status,
            "unresolved",
        )


class UncertaintyStateWritebackRequiresExperimentalEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.repository = UnifiedStateRepository(self.project_root)
        task = ScientificTask(
            task_id="ST_GATE_STATE",
            payload=ScientificTaskPayload(
                research_question=ResearchQuestion(
                    text="DeltaDec 是否改善 Vsw 预测",
                    target="Vsw",
                    question_type="forecasting",
                    variables=VariableBinding(
                        x="DeltaDec",
                        y="Vsw",
                        m_candidates=["By"],
                    ),
                ),
                evaluation=EvaluationSpec(
                    primary_metric="RMSE",
                    secondary_metrics=["Pearson_r"],
                ),
                constraints=ScientificConstraints(),
            ),
        )
        initial_hypotheses = [
            HypothesisNode(
                hypothesis_id="H1",
                statement="DeltaDec 带来独立增量",
                level=1,
                status="active",
                support_score=0.8,
                activation_condition="always",
            ),
            HypothesisNode(
                hypothesis_id="H2",
                statement="DeltaDec 不带来独立增量",
                level=1,
                status="active",
                support_score=0.4,
                activation_condition="always",
            ),
        ]
        initial_uncertainties = [
            UncertaintyRecord(
                uncertainty_id="U1",
                question="DeltaDec 是否具有增量价值",
                description="H1 与 H2 的分歧",
                related_hypotheses=["H1", "H2"],
                status="identified",
                priority="high",
                created_at_round=0,
                created_by="system",
                disagreement={
                    "H1": UncertaintyDisagreement(
                        hypothesis_id="H1",
                        expected="positive",
                        status="active",
                        support_score=0.8,
                    ),
                    "H2": UncertaintyDisagreement(
                        hypothesis_id="H2",
                        expected="negative",
                        status="active",
                        support_score=0.4,
                    ),
                },
            )
        ]
        self.repository.initialize_state_skeleton(
            task,
            initial_hypotheses=initial_hypotheses,
            initial_uncertainties=initial_uncertainties,
            overwrite=True,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _apply_with_externally_forced_resolved(
        self,
        update: DisagreementUpdate,
    ) -> UncertaintyRecord:
        from core.evaluation_unified import evaluate_experiment

        result = _minimal_result()
        protocol = _minimal_protocol()
        evaluation = evaluate_experiment(
            result,
            protocol=protocol,
            prior_supports={"H1": 0.8, "H2": 0.4},
        )
        evaluation.scientific.disagreement_updates = [
            DisagreementUpdate(
                uncertainty_id=update.uncertainty_id,
                compared_hypotheses=update.compared_hypotheses,
                support_span_before=update.support_span_before,
                support_span_after=update.support_span_after,
                narrowed=update.narrowed,
                resolution_status=update.resolution_status,
                evidence_basis=update.evidence_basis,
                leading_hypothesis_id=update.leading_hypothesis_id,
                unresolved_hypotheses=update.unresolved_hypotheses,
                summary=update.summary,
            )
        ]
        updater = UnifiedStateUpdater(self.repository)
        updater.apply_evaluation(
            protocol=protocol,
            result=result,
            evaluation=evaluation,
            protocol_path="config/latest_protocol.json",
            result_path="results/round_01/result_unified.json",
            evaluation_path="results/round_01/evaluation_unified.json",
        )
        uncertainties = self.repository.load_uncertainties()
        return uncertainties.record_index()["U1"]

    def test_missing_valid_evidence_overrides_external_resolved(self) -> None:
        record = self._apply_with_externally_forced_resolved(
            DisagreementUpdate(
                uncertainty_id="U1",
                compared_hypotheses=["H1", "H2"],
                support_span_before=0.4,
                support_span_after=0.4,
                narrowed=False,
                resolution_status="resolved",
                evidence_basis="missing_valid_experiment_evidence",
                leading_hypothesis_id="H1",
                summary="外部路径声称已解决",
            )
        )

        self.assertEqual(record.resolution_status, "unresolved")
        self.assertEqual(record.status, "active")
        self.assertIsNone(record.resolved_at_round)
        self.assertIn("缺少针对该不确定性的有效实验判定", record.notes)

    def test_legacy_update_without_basis_defaults_to_missing_evidence(self) -> None:
        record = self._apply_with_externally_forced_resolved(
            DisagreementUpdate(
                uncertainty_id="U1",
                compared_hypotheses=["H1", "H2"],
                support_span_before=0.4,
                support_span_after=0.4,
                narrowed=False,
                resolution_status="resolved",
                leading_hypothesis_id="H1",
                summary="旧版协议未提供证据来源",
            )
        )

        self.assertEqual(
            record.resolution_status,
            "unresolved",
        )


if __name__ == "__main__":
    unittest.main()
