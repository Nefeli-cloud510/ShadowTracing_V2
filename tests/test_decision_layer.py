import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from core.decision_unified import (
    CandidateExperimentGenerator,
    DecisionLayerService,
    UncertaintyPrioritizer,
    UtilityScorer,
    _apply_model_sensitivity_design,
    _build_design_for_strategy,
    _candidate_traceability,
    _candidate_experiment_type,
    _candidate_type,
    _lag_stability_values,
    _uncertainty_design_strategy,
    recompute_candidate_plan_metadata,
)
from core.state_repository import UnifiedStateRepository
from core.unified_schema import (
    CandidateExperiment,
    CandidateExperimentSet,
    DataDictionary,
    EvaluationSpec,
    ExperimentDesign,
    ExperimentMemoryEntry,
    ExperimentMemoryState,
    ExperimentRun,
    FieldDescriptor,
    HypothesisGenerationRationale,
    HypothesisNode,
    HypothesisSourceSignal,
    LatestTreeUpdate,
    MetricDelta,
    PerformanceMetrics,
    ResearchQuestion,
    ScientificConstraints,
    ScientificTask,
    ScientificTaskPayload,
    UncertaintyRecord,
    VariableBinding,
)


class StubCandidateExperimentDesigner:
    def design(self, **kwargs) -> bool:
        return False


class StubCandidateExperimentWriter:
    """Offline writer that fills candidate prose without calling an LLM."""

    def write_all(
        self,
        candidates,
        *,
        task=None,
        semantic_service=None,
        node_index=None,
        round_id=None,
        uncertainty_index=None,
        max_workers=4,
    ) -> None:
        for candidate in candidates:
            target = (
                candidate.design.display_target
                or candidate.design.target
                or "目标变量"
            )
            candidate.purpose = (
                f"验证 {target} 预测中焦点变量的独立贡献（stub writer）"
            )
            candidate.distinguishing_insight = (
                "比较对照组（不含待验证焦点变量）与实验组（加入待验证焦点变量）"
                "的实证差异，判断增量贡献是否成立。"
            )
            candidate.value_analysis = "\n".join(
                [
                    "信息增益分析：当前 IG 表示候选实验对竞争假设的区分能力。",
                    "性能增益分析：PG 置 0 表示本场景未启用历史性能信息。",
                    "风险分析：风险分由支持度先验与执行条件加权，整体处于可接受水平。",
                    "成本分析：成本分来自令牌预算与人工审核估算，本轮整体较低。",
                    "综合价值判断：综合 IG、PG、风险与成本，本实验值得进入审批。",
                ]
            )


def _freeze_tree(repository: UnifiedStateRepository) -> None:
    tree = repository.load_hypothesis_tree()
    tree.latest_update = LatestTreeUpdate(
        round=max(int(tree.current_round or 0), 1),
        event="hypothesis_tree_confirmed",
        description="test fixture freezes the hypothesis tree",
    )
    repository.save_hypothesis_tree(tree)


class CandidateTypeSemanticsTest(unittest.TestCase):
    def _record(self, question: str, description: str = "semantic regression fixture") -> UncertaintyRecord:
        return UncertaintyRecord(
            uncertainty_id="U_SEM",
            question=question,
            description=description,
            status="active",
            priority="medium",
            created_at_round=1,
            created_by="tester",
        )

    def test_nonlinear_uncertainty_stays_incremental(self) -> None:
        record = self._record("DeltaDec 与 Vsw 之间是否是非线性关系？")
        self.assertEqual(_candidate_experiment_type(record), "incremental_prediction")

    def test_model_family_question_still_maps_to_sensitivity(self) -> None:
        record = self._record("Ridge 与 ElasticNet 的家族内增益是否一致？")
        self.assertEqual(_candidate_experiment_type(record), "model_sensitivity")

    def test_model_sensitivity_design_marks_linear_capability_boundary(self) -> None:
        design = ExperimentDesign(target="Vsw", control=["By"], treatment=["By", "DeltaDec"])
        _apply_model_sensitivity_design(design)
        self.assertTrue(
            any(
                note.startswith("capability_boundary:model_sensitivity")
                for note in design.notes
            )
        )


class TemporalStabilitySemanticsTest(unittest.TestCase):
    def _external_record(self) -> UncertaintyRecord:
        return UncertaintyRecord(
            uncertainty_id="U_EXT_L3_01",
            question="外部补充回测中，dev 段 mag -> mag_both（提前 3 天、训练内自动选参）"
            "的唯一不跨零 RMSE 改善是否具有跨时段稳定性？",
            description="需要在 dev/f1/f2 三个窗口逐窗口报告增益与 CI；跨窗口未复现时不得写成已稳定支持。",
            mining_sources=["external_diagnostic:legacy_v3_outer_backtest"],
            status="active",
            priority="high",
            created_at_round=3,
            created_by="external_diagnostic_import",
        )

    def test_cross_window_uncertainty_maps_to_cross_window_candidate(self) -> None:
        record = self._external_record()
        self.assertEqual(_candidate_type(record), "cross_window_stability")
        self.assertEqual(
            _uncertainty_design_strategy(
                record,
                f"{record.notes or ''} {record.question} {record.description}",
                has_non_primary_focus=True,
            ),
            "temporal_stability_probe",
        )
        self.assertEqual(_lag_stability_values(record), [])

    def test_temporal_stability_probe_inherits_both_shadow_axes_and_frozen_lags(self) -> None:
        record = self._external_record()
        design, _ = _build_design_for_strategy(
            strategy="temporal_stability_probe",
            focus="p2_y中心值",
            primary_x="p2_y中心值",
            target="SW Plasma Speed, km/s",
            feature_pool=[
                "BY, nT (GSE)",
                "BZ, nT (GSE)",
                "Scalar B, nT",
                "p2_y中心值",
                "p1_x中心值",
            ],
        )
        self.assertIn("p2_y中心值", design.treatment)
        self.assertIn("p1_x中心值", design.treatment)
        for shadow in ("p2_y中心值", "p1_x中心值"):
            self.assertNotIn(shadow, design.control)
        self.assertEqual(design.lags["p2_y中心值"], [1])
        self.assertEqual(design.lags["p1_x中心值"], [9])
        self.assertEqual(design.lags["BY, nT (GSE)"], [4, 5, 7, 8])
        self.assertEqual(design.lags["BZ, nT (GSE)"], [1, 2])
        self.assertEqual(design.lags["Scalar B, nT"], [1, 2])
        self.assertNotIn([1, 3, 5], list(design.lags.values()))
        self.assertEqual(design.probe_axis, "cross_window_stability")

    def test_external_traceability_does_not_cite_round_two_or_freeze_time_split(self) -> None:
        record = self._external_record()
        design, _ = _build_design_for_strategy(
            strategy="temporal_stability_probe",
            focus="p2_y中心值",
            primary_x="p2_y中心值",
            target="SW Plasma Speed, km/s",
            feature_pool=[
                "BY, nT (GSE)",
                "BZ, nT (GSE)",
                "Scalar B, nT",
                "p2_y中心值",
                "p1_x中心值",
            ],
        )
        trace = _candidate_traceability(
            record=record,
            design=design,
            experiment_type="incremental_prediction",
            experiment_memory=None,
            next_round=3,
        )
        self.assertIn("未获得跨窗口一致支持", trace["responded_prior_finding"])
        self.assertNotIn("E_R02_01", trace["responded_prior_finding"])
        self.assertNotIn("训练/评价时间划分", trace["held_fixed_factors"])
        self.assertEqual(trace["next_round_kind"], "reanalysis")
        self.assertIn(
            "评价窗口按冻结回测逐窗口分别报告（dev/f1/f2）",
            trace["held_fixed_factors"],
        )
        self.assertTrue(
            any(
                "未获得跨窗口一致支持" in value
                for value in trace["expected_outcome_branches"].values()
            )
        )
        self.assertIn(
            "external_diagnostic:legacy_v3_outer_backtest",
            record.mining_sources,
        )


class DecisionLayerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.repository = UnifiedStateRepository(self.project_root)

        self.task = ScientificTask(
            task_id="ST_200",
            payload=ScientificTaskPayload(
                research_question=ResearchQuestion(
                    text="DeltaDec 是否能改善 Vsw 预测",
                    target="Vsw",
                    question_type="forecasting",
                    variables=VariableBinding(
                        x="DeltaDec",
                        y="Vsw",
                        m_candidates=["By", "Bz"],
                    ),
                ),
                evaluation=EvaluationSpec(primary_metric="RMSE", secondary_metrics=["Pearson_r"]),
                constraints=ScientificConstraints(),
            ),
        )
        self.dictionary = DataDictionary(
            dictionary_id="DICT_200",
            generated_at=datetime.now(),
            version="1.0",
            dataset_name="demo",
            total_samples=100,
            time_column="TIME",
            time_format="YYYYMMDD",
            fields=[
                FieldDescriptor(field_name="TIME", data_type="int", physical_meaning="time"),
                FieldDescriptor(field_name="Vsw", data_type="float", physical_meaning="target", is_target=True),
                FieldDescriptor(field_name="By", data_type="float", physical_meaning="IMF By"),
                FieldDescriptor(field_name="Bz", data_type="float", physical_meaning="IMF Bz"),
                FieldDescriptor(field_name="DeltaDec", data_type="float", physical_meaning="LHAASO DeltaDec"),
            ],
            target_candidates=["Vsw"],
            feature_candidates=["By", "Bz", "DeltaDec"],
        )
        self.repository.initialize_state_skeleton(
            self.task,
            initial_hypotheses=[
                HypothesisNode(
                    hypothesis_id="H1",
                    statement="DeltaDec 提供独立增量信息",
                    level=1,
                    status="active",
                    support_score=0.65,
                    activation_condition="always",
                ),
                HypothesisNode(
                    hypothesis_id="H2",
                    statement="DeltaDec 主要通过 By 起作用",
                    level=1,
                    status="active",
                    support_score=0.35,
                    activation_condition="always",
                ),
            ],
            initial_uncertainties=[
                UncertaintyRecord(
                    uncertainty_id="U01",
                    question="DeltaDec 是否具有独立增量信息？",
                    description="需要区分独立增量与中介解释。",
                    related_hypotheses=["H1", "H2"],
                    status="active",
                    priority="high",
                    created_at_round=1,
                    created_by="tester",
                ),
                UncertaintyRecord(
                    uncertainty_id="U02",
                    question="DeltaDec 的增益是否跨时间段稳定？",
                    description="需要检验滞后与稳定性。",
                    related_hypotheses=["H1"],
                    status="active",
                    priority="medium",
                    created_at_round=1,
                    created_by="tester",
                ),
            ],
            overwrite=True,
        )
        _freeze_tree(self.repository)
        uncertainties = self.repository.load_uncertainties()
        uncertainties.current_round = 1
        self.repository.save_uncertainties(uncertainties)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prioritize_generate_and_score(self) -> None:
        tree = self.repository.load_hypothesis_tree()
        uncertainties = self.repository.load_uncertainties()

        prioritized = UncertaintyPrioritizer().prioritize(uncertainties, tree)
        self.assertEqual(prioritized.queue[0].uncertainty_id, "U01")
        self.assertGreater(prioritized.queue[0].priority_score, prioritized.queue[1].priority_score)
        self.assertIn("H1", uncertainties.record_index()["U01"].disagreement)
        self.assertAlmostEqual(uncertainties.record_index()["U01"].disagreement["H1"].support_score, 0.65)

        candidates = CandidateExperimentGenerator().generate(
            task=self.task,
            data_dictionary=self.dictionary,
            hypothesis_tree=tree,
            uncertainty_state=uncertainties,
        )
        scored = UtilityScorer().score(candidates)

        self.assertTrue(scored.candidates)
        self.assertEqual(scored.candidates[0].design.target, "Vsw")
        self.assertIn("DeltaDec", scored.candidates[0].design.treatment)
        self.assertIsNotNone(scored.candidates[0].utility_score)
        review_candidates = [candidate for candidate in scored.candidates if candidate.requires_human_review]
        self.assertTrue(any("待区分假设：" in note for note in review_candidates[0].design.notes))

    def test_service_saves_candidate_plan(self) -> None:
        service = DecisionLayerService(
            self.repository,
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )
        plan = service.build_candidate_plan(task=self.task, data_dictionary=self.dictionary)

        self.assertTrue(plan.candidates)
        self.assertTrue((self.project_root / "state" / "candidate_experiments.json").exists())
        self.assertIsNotNone(plan.top_candidate())
        self.assertGreaterEqual(plan.top_candidate().utility_score, plan.candidates[-1].utility_score)

    def test_generated_hypothesis_rationale_flows_into_candidate_design(self) -> None:
        repository = UnifiedStateRepository(self.project_root / "generated")
        repository.initialize_state_skeleton(
            self.task,
            initial_hypotheses=[
                HypothesisNode(
                    hypothesis_id="H10",
                    statement="DeltaDec 提供独立于 By 的增量信息",
                    level=1,
                    status="active",
                    support_score=0.6,
                    activation_condition="always",
                    generation_rationale=HypothesisGenerationRationale(
                        trigger="independent_increment_found",
                        summary="DeltaDec 与太阳风速度预测残差相关",
                        derived_features=["DeltaDec"],
                        source_signals=[
                            HypothesisSourceSignal(
                                signal_type="residual_correlation",
                                source_id="demo",
                                excerpt="DeltaDec 与预测残差相关",
                            )
                        ],
                    ),
                )
            ],
            initial_uncertainties=[
                UncertaintyRecord(
                    uncertainty_id="U10",
                    question="DeltaDec 的贡献是否独立于 By？",
                    description="需要区分独立增量与中介解释。",
                    related_hypotheses=["H10"],
                    status="active",
                    priority="high",
                    created_at_round=0,
                    created_by="tester",
                )
            ],
            data_dictionary=self.dictionary,
            overwrite=True,
        )
        service = DecisionLayerService(
            repository,
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )
        plan = service.build_candidate_plan(task=self.task, data_dictionary=self.dictionary)

        review_candidates = [candidate for candidate in plan.candidates if candidate.requires_human_review]
        self.assertTrue(review_candidates)
        self.assertTrue(any("假设触发来源：" in note for note in review_candidates[0].design.notes))
        self.assertIn("实证差异", review_candidates[0].distinguishing_insight)

    def test_service_injects_human_feedback_into_next_round_candidates(self) -> None:
        service = DecisionLayerService(
            self.repository,
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )
        plan = service.build_candidate_plan(
            task=self.task,
            data_dictionary=self.dictionary,
            planning_feedback="下一轮先测试 By 单变量路径",
        )

        focused = [
            candidate for candidate in plan.candidates if "human_feedback:下一轮先测试 By 单变量路径" in candidate.design.notes
        ]
        self.assertTrue(focused)
        review_candidates = [candidate for candidate in focused if candidate.requires_human_review]
        self.assertTrue(review_candidates)
        self.assertIn("DeltaDec", review_candidates[0].design.treatment)
        self.assertIn("By", review_candidates[0].design.control)
        self.assertIn("human_feedback_focus_single_feature:By", review_candidates[0].design.notes)
        self.assertIn("human_feedback", plan.note)

    def test_uncertainty_specific_focus_produces_distinct_candidate_designs(self) -> None:
        repository = UnifiedStateRepository(self.project_root / "focused")
        repository.initialize_state_skeleton(
            self.task,
            initial_hypotheses=[
                HypothesisNode(
                    hypothesis_id="H_independent",
                    statement="DeltaDec 提供独立增量信息",
                    level=1,
                    status="active",
                    support_score=0.65,
                    activation_condition="always",
                    generation_rationale=HypothesisGenerationRationale(
                        trigger="task_bootstrap",
                        summary="主变量效果",
                        derived_features=["DeltaDec", "Vsw"],
                    ),
                ),
                HypothesisNode(
                    hypothesis_id="H_via_by",
                    statement="DeltaDec 主要通过 By 起作用",
                    level=1,
                    status="active",
                    support_score=0.45,
                    activation_condition="always",
                    generation_rationale=HypothesisGenerationRationale(
                        trigger="mediator_bootstrap",
                        summary="By 中介路径",
                        derived_features=["DeltaDec", "By", "Vsw"],
                    ),
                ),
                HypothesisNode(
                    hypothesis_id="H_via_bz",
                    statement="DeltaDec 主要通过 Bz 起作用",
                    level=1,
                    status="active",
                    support_score=0.35,
                    activation_condition="always",
                    generation_rationale=HypothesisGenerationRationale(
                        trigger="mediator_bootstrap",
                        summary="Bz 中介路径",
                        derived_features=["DeltaDec", "Bz", "Vsw"],
                    ),
                ),
            ],
            initial_uncertainties=[
                UncertaintyRecord(
                    uncertainty_id="U_BY",
                    question="DeltaDec 是否独立于 By 提供增量信息？",
                    description="需要区分独立增量与 By 中介解释。",
                    related_hypotheses=["H_via_by", "H_independent"],
                    status="active",
                    priority="high",
                    created_at_round=1,
                    created_by="tester",
                ),
                UncertaintyRecord(
                    uncertainty_id="U_IND",
                    question="DeltaDec 是否具有独立增量信息？",
                    description="需要检验主变量的独立预测价值。",
                    related_hypotheses=["H_independent"],
                    status="active",
                    priority="medium",
                    created_at_round=1,
                    created_by="tester",
                ),
                UncertaintyRecord(
                    uncertainty_id="U_BZ",
                    question="DeltaDec 是否独立于 Bz 提供增量信息？",
                    description="需要区分独立增量与 Bz 中介解释。",
                    related_hypotheses=["H_via_bz", "H_independent"],
                    status="active",
                    priority="medium",
                    created_at_round=1,
                    created_by="tester",
                ),
            ],
            overwrite=True,
        )
        service = DecisionLayerService(
            repository,
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )
        plan = service.build_candidate_plan(task=self.task, data_dictionary=self.dictionary)
        designs = [candidate.design for candidate in plan.candidates]
        treatment_signatures = {tuple(design.treatment) for design in designs}
        self.assertGreaterEqual(len(treatment_signatures), 2)
        by_candidate = next(
            candidate for candidate in plan.candidates if candidate.design.control == ["By"]
        )
        bz_candidate = next(
            candidate for candidate in plan.candidates if candidate.design.control == ["Bz"]
        )
        self.assertNotEqual(by_candidate.design.treatment, bz_candidate.design.treatment)
        self.assertNotEqual(by_candidate.design.control, bz_candidate.design.control)

    def test_uncertainty_types_produce_structurally_distinct_candidate_designs(self) -> None:
        repository = UnifiedStateRepository(self.project_root / "structural")
        repository.initialize_state_skeleton(
            self.task,
            initial_hypotheses=[
                HypothesisNode(
                    hypothesis_id="H_independent",
                    statement="DeltaDec 提供独立增量信息",
                    level=1,
                    status="active",
                    support_score=0.65,
                    activation_condition="always",
                    generation_rationale=HypothesisGenerationRationale(
                        trigger="task_bootstrap",
                        summary="主变量效果",
                        derived_features=["DeltaDec", "Vsw"],
                    ),
                ),
                HypothesisNode(
                    hypothesis_id="H_null_by",
                    statement="By 是主要竞争解释",
                    level=1,
                    status="active",
                    support_score=0.42,
                    activation_condition="always",
                    generation_rationale=HypothesisGenerationRationale(
                        trigger="competition_bootstrap",
                        summary="By 竞争路径",
                        derived_features=["DeltaDec", "By", "Vsw"],
                    ),
                ),
                HypothesisNode(
                    hypothesis_id="H_branch_bz",
                    statement="Bz 条件下存在路径变化",
                    level=1,
                    status="active",
                    support_score=0.36,
                    activation_condition="always",
                    generation_rationale=HypothesisGenerationRationale(
                        trigger="branch_bootstrap",
                        summary="Bz 条件路径",
                        derived_features=["DeltaDec", "Bz", "Vsw"],
                    ),
                ),
            ],
            initial_uncertainties=[
                UncertaintyRecord(
                    uncertainty_id="U_IND",
                    question="DeltaDec 是否具有独立增量信息？",
                    description="需要检验主变量的独立预测价值。",
                    related_hypotheses=["H_independent"],
                    status="active",
                    priority="high",
                    created_at_round=2,
                    created_by="tester",
                ),
                UncertaintyRecord(
                    uncertainty_id="U_COMP",
                    question="DeltaDec 是否超越 By 的竞争解释提供增量信息？",
                    description="需要区分独立增量与竞争解释。",
                    related_hypotheses=["H_null_by", "H_independent"],
                    status="active",
                    priority="high",
                    created_at_round=2,
                    created_by="tester",
                    notes="mining_sources:from_hypothesis_conflict;mining_features:DeltaDec,SW Plasma Speed, km/s,By;question_type:independent_gain",
                ),
                UncertaintyRecord(
                    uncertainty_id="U_COND",
                    question="DeltaDec 与 Bz 的条件路径是否受未建模条件约束？",
                    description="需要区分受 Bz 条件约束的路径。",
                    related_hypotheses=["H_branch_bz", "H_independent"],
                    status="active",
                    priority="medium",
                    created_at_round=2,
                    created_by="tester",
                    notes="mining_sources:from_missing_evidence;mining_features:DeltaDec,SW Plasma Speed, km/s,Bz;question_type:conditioned_path",
                ),
            ],
            overwrite=True,
        )
        service = DecisionLayerService(
            repository,
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )
        plan = service.build_candidate_plan(task=self.task, data_dictionary=self.dictionary)

        designs = [candidate.design for candidate in plan.candidates]
        self.assertEqual(len({tuple(design.control) for design in designs}), 3)
        self.assertEqual(len({tuple(design.treatment) for design in designs}), 3)
        self.assertFalse(
            any(
                "generic distinguishing experiment" in note
                for design in designs
                for note in design.notes
            )
        )

        by_candidate = next(
            candidate for candidate in plan.candidates if candidate.design.control == ["By"]
        )
        bz_candidate = next(
            candidate for candidate in plan.candidates if candidate.design.control == ["Bz"]
        )
        self.assertEqual(by_candidate.design.control, ["By"])
        self.assertEqual(by_candidate.design.treatment, ["By", "DeltaDec"])
        self.assertEqual(bz_candidate.design.control, ["Bz"])
        self.assertEqual(bz_candidate.design.treatment, ["Bz", "DeltaDec"])
        ig_values = {
            candidate.estimated_information_gain.value
            for candidate in plan.candidates
            if candidate.estimated_information_gain
        }
        utility_values = {
            candidate.utility_score
            for candidate in plan.candidates
            if candidate.utility_score is not None
        }
        self.assertGreaterEqual(len(ig_values), 2)
        self.assertGreaterEqual(len(utility_values), 2)

    def test_partially_resolved_uncertainty_turns_into_validation_followup(self) -> None:
        tree = self.repository.load_hypothesis_tree()
        uncertainties = self.repository.load_uncertainties()
        uncertainties.record_index()["U01"].resolution_status = "partially_resolved"
        uncertainties.record_index()["U01"].priority = "medium"
        self.repository.save_uncertainties(uncertainties)

        prioritized = UncertaintyPrioritizer().prioritize(uncertainties, tree)
        service = DecisionLayerService(
            self.repository,
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )
        plan = service.build_candidate_plan(task=self.task, data_dictionary=self.dictionary)

        candidate = next(
            item
            for item in plan.candidates
            if item.requires_human_review and "U01" in item.related_uncertainties
        )
        self.assertEqual(candidate.type, "validation_followup")
        self.assertIn("validation_followup_for_partially_resolved_disagreement", candidate.design.notes)
        self.assertLess(prioritized.queue[0].priority_score, 0.85)

    def test_expected_performance_gain_uses_history_based_rmse_formula(self) -> None:
        experiment_memory = self.repository.load_experiment_memory()
        experiment_memory.entries.append(
            ExperimentMemoryEntry(
                experiment_id="E_HIST_01",
                round_id=1,
                status="completed",
                tested_hypotheses=["H1"],
                metrics_snapshot=PerformanceMetrics(
                    experiment_id="E_HIST_01",
                    round_id=1,
                    baseline_rmse=12.0,
                    treatment_rmse=11.0,
                    delta=MetricDelta(rmse=1.0, pearson_r=0.04),
                ),
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        )
        self.repository.save_experiment_memory(experiment_memory)

        tree = self.repository.load_hypothesis_tree()
        uncertainties = self.repository.load_uncertainties()
        UncertaintyPrioritizer().prioritize(uncertainties, tree)
        candidates = CandidateExperimentGenerator().generate(
            task=self.task,
            data_dictionary=self.dictionary,
            hypothesis_tree=tree,
            uncertainty_state=uncertainties,
            experiment_memory=experiment_memory,
        )

        formula_candidates = [
            candidate
            for candidate in candidates.candidates
            if candidate.estimated_performance_gain
            and candidate.estimated_performance_gain.rationale
            and "PG_expected(E)" in candidate.estimated_performance_gain.rationale
        ]
        self.assertTrue(formula_candidates)
        positive_gain_candidates = [
            candidate for candidate in formula_candidates if candidate.estimated_performance_gain.value > 0.0
        ]
        self.assertTrue(positive_gain_candidates)
        self.assertIn("PG_expected(E)", positive_gain_candidates[0].estimated_performance_gain.rationale)

    def test_candidate_traceability_fields_are_pre_registered(self) -> None:
        experiment_memory = self.repository.load_experiment_memory()
        experiment_memory.entries.append(
            ExperimentMemoryEntry(
                experiment_id="E_HIST_01",
                round_id=1,
                status="completed",
                tested_hypotheses=["H1"],
                key_findings=["日影增量点估计为正但 CI 跨零"],
                metrics_snapshot=PerformanceMetrics(
                    experiment_id="E_HIST_01",
                    round_id=1,
                    runs=[
                        ExperimentRun(
                            run_id="E_HIST_01_control",
                            name="control",
                            model_family="elasticnet",
                            feature_columns=["By"],
                            parameters={
                                "forecast_horizon_days": 3,
                                "past_lag_days": 3,
                                "window_size": 14,
                            },
                        ),
                        ExperimentRun(
                            run_id="E_HIST_01_treatment",
                            name="treatment",
                            model_family="elasticnet",
                            feature_columns=["By", "DeltaDec"],
                            parameters={
                                "forecast_horizon_days": 3,
                                "past_lag_days": 3,
                                "window_size": 14,
                            },
                        ),
                    ],
                ),
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        )
        self.repository.save_experiment_memory(experiment_memory)

        tree = self.repository.load_hypothesis_tree()
        uncertainties = self.repository.load_uncertainties()
        uncertainties.current_round = 1
        UncertaintyPrioritizer().prioritize(uncertainties, tree)
        candidates = CandidateExperimentGenerator().generate(
            task=self.task,
            data_dictionary=self.dictionary,
            hypothesis_tree=tree,
            uncertainty_state=uncertainties,
            experiment_memory=experiment_memory,
            target_round=2,
        )
        candidate = candidates.candidates[0]

        self.assertIn("E_HIST_01", candidate.responded_prior_finding or "")
        self.assertIn("CI 跨零", candidate.responded_prior_finding or "")
        self.assertTrue(candidate.changed_factors)
        self.assertTrue(candidate.held_fixed_factors)
        self.assertTrue(
            not any(re.match(r"^历史长度（3 天 → \d+ 天）$", item) for item in candidate.changed_factors),
            candidate.changed_factors,
        )
        self.assertTrue(
            not any(re.match(r"^时间窗口（14 天 → \d+ 天）$", item) for item in candidate.changed_factors),
            candidate.changed_factors,
        )
        self.assertIn("预测超前 3 天", candidate.held_fixed_factors)
        self.assertIn("历史长度 3 天", candidate.held_fixed_factors)
        self.assertIn("时间窗口 14 天", candidate.held_fixed_factors)
        self.assertTrue(
            any(item.startswith("window_inherited_from_previous_round:") for item in candidate.design.notes),
            candidate.design.notes,
        )
        self.assertIn("CI 全部落在正区间", candidate.expected_outcome_branches)
        branch_values = list(candidate.expected_outcome_branches.values())
        self.assertTrue(all(isinstance(item, str) and item for item in branch_values))
        self.assertEqual(candidate.next_round_kind, "new_hypothesis_test")

    def test_traceability_without_meaningful_design_change_marks_recompute(self) -> None:
        experiment_memory = ExperimentMemoryState(
            task_id="ST_TEST",
            current_round=1,
            last_updated=datetime.now(),
            entries=[],
        )
        experiment_memory.entries.append(
            ExperimentMemoryEntry(
                experiment_id="E_HIST_01",
                round_id=1,
                status="completed",
                tested_hypotheses=["H1"],
                metrics_snapshot=PerformanceMetrics(
                    experiment_id="E_HIST_01",
                    round_id=1,
                    runs=[
                        ExperimentRun(
                            run_id="E_HIST_01_treatment",
                            name="treatment",
                            model_family="elasticnet",
                            feature_columns=["By", "DeltaDec"],
                            parameters={
                                "forecast_horizon_days": 3,
                                "past_lag_days": 3,
                                "window_size": 14,
                            },
                        )
                    ],
                ),
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        )
        design = ExperimentDesign(
            target="Vsw",
            control=["By"],
            treatment=["By", "DeltaDec"],
            forecast_horizon_days=3,
            past_lag_days=3,
            window_size=14,
        )
        record = UncertaintyRecord(
            uncertainty_id="U_LOCAL_01",
            question="加入日影后，在相同条件下重跑能否复现逐日预测？",
            description="复算 evaluator 兼容性验收",
            status="active",
            priority="medium",
            created_at_round=1,
            created_by="tester",
        )
        trace = _candidate_traceability(
            record=record,
            design=design,
            experiment_type="incremental_prediction",
            experiment_memory=experiment_memory,
            next_round=2,
        )

        self.assertEqual(trace["next_round_kind"], "recompute")
        self.assertEqual(
            trace["changed_factors"],
            ["在本轮固定条件下验证/复测与上一轮相关的不确定性"],
        )

    def test_window_inheritance_does_not_rotate_with_history_count(self) -> None:
        experiment_memory = self.repository.load_experiment_memory()
        for entry_id, horizon, lag, window in (
            ("E_HIST_01", 3, 3, 14),
            ("E_HIST_02", 5, 7, 21),
        ):
            experiment_memory.entries.append(
                ExperimentMemoryEntry(
                    experiment_id=entry_id,
                    round_id=1 if entry_id == "E_HIST_01" else 2,
                    status="completed",
                    tested_hypotheses=["H1"],
                    metrics_snapshot=PerformanceMetrics(
                        experiment_id=entry_id,
                        round_id=1 if entry_id == "E_HIST_01" else 2,
                        runs=[
                            ExperimentRun(
                                run_id=f"{entry_id}_control",
                                name="control",
                                model_family="elasticnet",
                                feature_columns=["By"],
                                parameters={
                                    "forecast_horizon_days": horizon,
                                    "past_lag_days": lag,
                                    "window_size": window,
                                },
                            ),
                            ExperimentRun(
                                run_id=f"{entry_id}_treatment",
                                name="treatment",
                                model_family="elasticnet",
                                feature_columns=["By", "DeltaDec"],
                                parameters={
                                    "forecast_horizon_days": horizon,
                                    "past_lag_days": lag,
                                    "window_size": window,
                                },
                            ),
                        ],
                    ),
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
            )
        self.repository.save_experiment_memory(experiment_memory)

        tree = self.repository.load_hypothesis_tree()
        uncertainties = self.repository.load_uncertainties()
        uncertainties.current_round = 2
        UncertaintyPrioritizer().prioritize(uncertainties, tree)
        candidates = CandidateExperimentGenerator().generate(
            task=self.task,
            data_dictionary=self.dictionary,
            hypothesis_tree=tree,
            uncertainty_state=uncertainties,
            experiment_memory=experiment_memory,
            target_round=3,
        )
        candidate = candidates.candidates[0]
        self.assertEqual(candidate.design.forecast_horizon_days, 5)
        self.assertEqual(candidate.design.past_lag_days, 7)
        self.assertEqual(candidate.design.window_size, 21)
        self.assertTrue(
            any(
                item.startswith("window_inherited_from_previous_round:")
                and "forecast_horizon=5" in item
                and "past_lag=7" in item
                and "window=21" in item
                for item in candidate.design.notes
            ),
            candidate.design.notes,
        )

    def test_recompute_metadata_after_design_change_is_idempotent(self) -> None:
        experiment_memory = self.repository.load_experiment_memory()
        experiment_memory.entries.append(
            ExperimentMemoryEntry(
                experiment_id="E_HIST_01",
                round_id=1,
                status="completed",
                tested_hypotheses=["H1"],
                metrics_snapshot=PerformanceMetrics(
                    experiment_id="E_HIST_01",
                    round_id=1,
                    runs=[
                        ExperimentRun(
                            run_id="E_HIST_01_control",
                            name="control",
                            model_family="elasticnet",
                            feature_columns=["By"],
                            parameters={
                                "forecast_horizon_days": 3,
                                "past_lag_days": 3,
                                "window_size": 14,
                            },
                        ),
                        ExperimentRun(
                            run_id="E_HIST_01_treatment",
                            name="treatment",
                            model_family="elasticnet",
                            feature_columns=["By", "DeltaDec"],
                            parameters={
                                "forecast_horizon_days": 3,
                                "past_lag_days": 3,
                                "window_size": 14,
                            },
                        ),
                    ],
                ),
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        )
        self.repository.save_experiment_memory(experiment_memory)

        tree = self.repository.load_hypothesis_tree()
        uncertainties = self.repository.load_uncertainties()
        uncertainties.current_round = 1
        UncertaintyPrioritizer().prioritize(uncertainties, tree)
        candidates = CandidateExperimentGenerator().generate(
            task=self.task,
            data_dictionary=self.dictionary,
            hypothesis_tree=tree,
            uncertainty_state=uncertainties,
            experiment_memory=experiment_memory,
            target_round=2,
        )
        candidate = candidates.candidates[0]
        candidate.design.forecast_horizon_days = 5
        candidate.design.past_lag_days = 5
        candidate.design.window_size = 20

        recomputed = recompute_candidate_plan_metadata(
            candidates,
            task=self.task,
            data_dictionary=self.dictionary,
            hypothesis_tree=tree,
            uncertainty_state=uncertainties,
            experiment_memory=experiment_memory,
            current_round=2,
        )
        updated = recomputed.candidates[0]
        self.assertIn(
            "预测超前天数（3 天 → 5 天）",
            updated.changed_factors,
        )
        self.assertIn(
            "历史长度（3 天 → 5 天）",
            updated.changed_factors,
        )
        self.assertIn(
            "时间窗口（14 天 → 20 天）",
            updated.changed_factors,
        )
        self.assertNotIn(
            "预测超前 3 天",
            updated.held_fixed_factors,
        )
        self.assertEqual(updated.next_round_kind, "new_hypothesis_test")
        self.assertIsNotNone(updated.utility_score)

        applied_again = recompute_candidate_plan_metadata(
            recomputed,
            task=self.task,
            data_dictionary=self.dictionary,
            hypothesis_tree=tree,
            uncertainty_state=uncertainties,
            experiment_memory=experiment_memory,
            current_round=2,
        )
        self.assertEqual(
            applied_again.candidates[0].utility_score,
            updated.utility_score,
        )
        self.assertEqual(
            applied_again.candidates[0].changed_factors,
            updated.changed_factors,
        )
        self.assertEqual(
            applied_again.candidates[0].next_round_kind,
            updated.next_round_kind,
        )

    def test_utility_scorer_demotes_candidates_duplicating_executed_history(self) -> None:
        experiment_memory = self.repository.load_experiment_memory()
        experiment_memory.entries.append(
            ExperimentMemoryEntry(
                experiment_id="E_HIST_01",
                round_id=1,
                status="completed",
                tested_hypotheses=["H1"],
                metrics_snapshot=PerformanceMetrics(
                    experiment_id="E_HIST_01",
                    round_id=1,
                    baseline_rmse=12.0,
                    treatment_rmse=11.0,
                    delta=MetricDelta(rmse=1.0, pearson_r=0.04),
                ),
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        )
        self.repository.save_experiment_memory(experiment_memory)

        candidate = CandidateExperiment(
            experiment_id="E_R02_01",
            type="distinguishing",
            purpose="resolve U01: test focus",
            tested_hypotheses=["H1"],
            design=ExperimentDesign(
                target="Vsw",
                control=["DeltaDec"],
                treatment=["DeltaDec", "By"],
                design_focus="By",
            ),
        )
        candidate_set = CandidateExperimentSet(
            round=2,
            generated_at=datetime.now(),
            candidates=[candidate],
        )
        scored = UtilityScorer().score(candidate_set, experiment_memory=experiment_memory)
        penalized = [
            candidate
            for candidate in scored.candidates
            if any(
                note.startswith("selection_penalty:executed_history_similarity")
                for note in candidate.design.notes
            )
        ]
        self.assertTrue(penalized)
        self.assertTrue(
            all(
                any("E_HIST_01" in note for note in candidate.design.notes)
                for candidate in penalized
            )
        )
        self.assertTrue(all(candidate.utility_score is not None for candidate in scored.candidates))


if __name__ == "__main__":
    unittest.main()
