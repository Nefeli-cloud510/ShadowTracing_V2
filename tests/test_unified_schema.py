import unittest
from datetime import datetime

from core.unified_schema import (
    CandidateExperiment,
    CandidateExperimentSet,
    DataDictionary,
    DecisionEntry,
    DecisionLog,
    EvaluationResult,
    EvaluationSpec,
    EvidenceSummary,
    ExecutionSummary,
    ExperimentDesign,
    ExperimentProtocol,
    ExperimentResult,
    ExperimentRun,
    FieldDescriptor,
    HypothesisAssessment,
    HypothesisNode,
    HypothesisTreeState,
    LatestTreeUpdate,
    MetricDelta,
    PerformanceMetrics,
    PrioritizedUncertainty,
    ProcessState,
    ProcessStep,
    ResearchQuestion,
    RobustnessOverall,
    RobustnessReport,
    ScientificConstraints,
    ScientificEvaluation,
    ScientificTask,
    ScientificTaskPayload,
    TreeSummary,
    UncertaintyPriorityQueue,
    VariableBinding,
    VisualizationArtifact,
)


class UnifiedSchemaSmokeTest(unittest.TestCase):
    def test_scientific_task_and_dictionary(self) -> None:
        task = ScientificTask(
            task_id="ST_001",
            payload=ScientificTaskPayload(
                research_question=ResearchQuestion(
                    text="DeltaDec 是否能改善 Vsw 预测？",
                    target="Vsw",
                    question_type="forecasting",
                    variables=VariableBinding(
                        x="DeltaDec",
                        y="Vsw",
                        m_candidates=["By", "Bz"],
                    ),
                ),
                evaluation=EvaluationSpec(
                    primary_metric="RMSE",
                    secondary_metrics=["MAE", "Pearson_r", "Skill"],
                ),
                constraints=ScientificConstraints(
                    no_future_information=True,
                    validation_feedback_allowed=True,
                    final_test_blind=True,
                ),
            ),
        )
        dictionary = DataDictionary(
            dictionary_id="DICT_001",
            generated_at=datetime.now(),
            version="1.0",
            dataset_name="demo_dataset",
            total_samples=196,
            time_column="TIME",
            time_format="YYYYMMDD",
            fields=[
                FieldDescriptor(
                    field_name="TIME",
                    data_type="int",
                    physical_meaning="observation day",
                    is_target=False,
                ),
                FieldDescriptor(
                    field_name="Vsw",
                    data_type="float",
                    physical_meaning="solar wind speed",
                    is_target=True,
                ),
            ],
            target_candidates=["Vsw"],
            feature_candidates=["By", "Bz", "DeltaDec"],
        )

        self.assertEqual(task.message_type, "scientific_task")
        self.assertIn("Vsw", dictionary.target_candidates)
        self.assertEqual(dictionary.get_available_fields(), ["TIME", "Vsw"])

    def test_tree_candidate_and_protocol(self) -> None:
        tree = HypothesisTreeState(
            tree_id="HT_001",
            task_id="ST_001",
            current_round=1,
            root_question="DeltaDec 是否能改善 Vsw 预测？",
            nodes=[
                HypothesisNode(
                    hypothesis_id="H1",
                    statement="DeltaDec 有独立增益",
                    level=1,
                    status="active",
                    support_score=0.6,
                    activation_condition="always",
                ),
                HypothesisNode(
                    hypothesis_id="H2",
                    statement="DeltaDec 主要通过 By 起作用",
                    level=1,
                    status="pending",
                    support_score=0.4,
                    activation_condition="always",
                ),
            ],
            active_hypotheses=["H1"],
            pending_hypotheses=["H2"],
            latest_update=LatestTreeUpdate(
                round=1,
                event="initialized",
                description="initial tree created",
            ),
            tree_summary=TreeSummary(
                total_nodes=2,
                active_count=1,
                pruned_count=0,
                pending_count=1,
            ),
        )
        candidate = CandidateExperiment(
            experiment_id="E01",
            type="distinguishing",
            purpose="distinguish H1 and H2",
            tested_hypotheses=["H1", "H2"],
            design=ExperimentDesign(
                target="Vsw",
                control=["OMNI", "OMNI+By"],
                treatment=["OMNI+By+DeltaDec"],
                lags={"DeltaDec": [2, 3, 4]},
            ),
        )
        candidate_set = CandidateExperimentSet(
            round=1,
            generated_at=datetime.now(),
            candidates=[candidate],
        )
        protocol = ExperimentProtocol(
            experiment_id="E01",
            round_id=1,
            task_id="ST_001",
            target="Vsw",
            features=candidate.design,
            tested_hypotheses=candidate.tested_hypotheses,
            source_candidate_id="E01",
        )

        self.assertEqual(tree.node_index()["H1"].support_score, 0.6)
        self.assertEqual(candidate_set.candidates[0].experiment_id, "E01")
        self.assertEqual(protocol.features.treatment, ["OMNI+By+DeltaDec"])

    def test_result_and_evaluation_payload(self) -> None:
        result = ExperimentResult(
            experiment_id="E01",
            round_id=1,
            status="completed",
            runs=[
                ExperimentRun(run_id="baseline", name="baseline"),
                ExperimentRun(run_id="treatment", name="treatment"),
            ],
            comparison=MetricDelta(rmse=3.4, pearson_r=0.1, skill=0.016),
            visualizations=[
                VisualizationArtifact(name="prediction_vs_truth", path="plots/pvt.png"),
            ],
            execution=ExecutionSummary(status="completed", duration_seconds=10.0),
        )
        evaluation = EvaluationResult(
            metrics=PerformanceMetrics(
                experiment_id="E01",
                round_id=1,
                baseline_rmse=46.2,
                treatment_rmse=42.8,
                delta=MetricDelta(rmse=3.4, pearson_r=0.1, skill=0.016),
            ),
            robustness=RobustnessReport(
                overall=RobustnessOverall(stable=True, concern="acceptable"),
                recommendation="continue",
            ),
            scientific=ScientificEvaluation(
                experiment_id="E01",
                round_id=1,
                hypothesis_assessments=[
                    HypothesisAssessment(
                        hypothesis_id="H1",
                        support_before=0.5,
                        support_after=0.6,
                        status="supported",
                        direction_matched=True,
                        magnitude_matched="partial",
                        reason="direction matches but effect size is smaller than expected",
                    )
                ],
                evidence_summary=EvidenceSummary(
                    remaining_uncertainties=[{"id": "U01", "question": "what is the mediator?"}]
                ),
            ),
            visualizations=result.visualizations,
        )
        payload = evaluation.to_interpreter_payload()

        self.assertEqual(result.execution.status, "completed")
        self.assertTrue(payload["robustness"]["overall"]["stable"])
        self.assertEqual(
            evaluation.robustness.overall.status,
            "not_tested",
        )
        self.assertEqual(payload["remaining_uncertainties"][0]["id"], "U01")

    def test_process_and_decision_log(self) -> None:
        process = ProcessState(
            task_id="ST_001",
            current_round=1,
            current_stage="planning",
            started_at=datetime.now(),
            steps={
                "step_1": ProcessStep(name="identify uncertainty", status="completed"),
                "step_2": ProcessStep(name="generate candidates", status="in_progress"),
            },
        )
        queue = UncertaintyPriorityQueue(
            last_updated=datetime.now(),
            current_round=1,
            queue=[
                PrioritizedUncertainty(
                    uncertainty_id="U01",
                    question="Is DeltaDec independent of By?",
                    priority_score=0.85,
                    status="active",
                    estimated_resolution_round=2,
                )
            ],
        )
        log = DecisionLog(
            task_id="ST_001",
            decisions=[
                DecisionEntry(
                    timestamp=datetime.now(),
                    round_id=1,
                    decision_type="experiment_approved",
                    made_by="human_pi",
                    summary="approve E01",
                )
            ],
        )

        self.assertEqual(process.steps["step_2"].status, "in_progress")
        self.assertEqual(queue.top_uncertainties()[0].uncertainty_id, "U01")
        self.assertEqual(log.decisions[0].decision_type, "experiment_approved")


if __name__ == "__main__":
    unittest.main()
