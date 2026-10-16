from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.calendar_samples import is_blind_sealed
from core.central_controller_llm import CentralControllerLLM
from core.decision_unified import DIFFERENTIATING_EXPERIMENT_ERROR
from core.decision_unified import DecisionLayerService
from core.display_text_cleaner import clean_llm_text, deep_clean_text
from core.evaluation_unified import export_evaluation_result
from core.experiment_tools.dispatcher import ExperimentDispatcher
from core.experiment_tools.evaluators import evaluate_experiment_result
from core.experiment_tools.registry import get_tool
from core.experiment_planner_llm import (
    CandidateExperimentDesignerLLM,
    CandidateExperimentWriterLLM,
    ExperimentPlannerLLM,
)
from core.harness_unified import UnifiedExperimentHarness
from core.hypothesis_identity import infer_related_hypothesis_ids
from core.hypothesis_linkage import (
    build_predictions_for_nodes,
    related_hypotheses_for_uncertainty,
)
from core.hypothesis_proposer_llm import HypothesisProposerLLM
from core.hypothesis_generation import (
    HypothesisGenerationService,
    compute_targeted_interrogation_scope,
    refresh_tree_metadata,
)
from core.hypothesis_state_machine import ACTIVE_LIKE
from core.json_safe import json_dumps
from core.llm_gateway import LLMGateway
from core.multi_source_uncertainty_miner import (
    MiningContext,
    MultiSourceUncertaintyMiner,
)
from core.parallel_reasoning_orchestrator import ParallelReasoningOrchestrator
from core.planner_output_applier import PlannerOutputApplier
from core.planner_unified import PlannerOutputBuilder, ReasoningPlannerInputBuilder
from core.protocol_bridge import ElasticNetDataSourceConfig
from core.rag_service import RAGContextBundle, RAGService
from core.round_orchestrator import RoundOrchestrator
from core.runtime_config import get_llm_model_for_role
from core.scientific_interpreter_llm import InterpreterRoundAnalysis, ScientificInterpreterLLM
from core.scientific_questioner_llm import ProposedUncertainty, ScientificQuestionerLLM
from core.state_repository import UnifiedStateRepository
from core.state_updater import UnifiedStateUpdater
from core.support_update_rules import (
    EVIDENCE_STABLE_DEGRADATION,
    EVIDENCE_STABLE_IMPROVEMENT,
    compute_support_update_from_reasoning,
)
from core.tuning_narrative_llm import TuningNarrativeService
from core.unified_schema import (
    CandidateExperiment,
    DataDictionary,
    DataDictionarySummary,
    DecisionEntry,
    DecisionLog,
    EvaluationResult,
    ExperimentProtocol,
    ExperimentResult,
    ExperimentStep,
    HypothesisQuestioningRecord,
    HypothesisSnapshot,
    HumanFeedbackEntry,
    LatestTreeUpdate,
    PlannerEvaluationSummary,
    PlannerDisagreementUpdateItem,
    PlannerReasoningTraceItem,
    PlannerUncertaintyItem,
    ProcessPhase,
    ProcessState,
    ProcessStep,
    ProtocolRefinementSuggestion,
    ProtocolConstraints,
    RejectedCandidate,
    ReasoningPlannerInput,
    ReasoningPlannerOutput,
    ScientificTask,
    StopHistoryEntry,
    SupportHistoryEntry,
    UncertaintyRecord,
)
from core.variable_semantic_service import VariableSemanticService


class CandidateProtocolMapper:
    """Map selected candidate experiments into executable ExperimentProtocol objects."""

    def to_protocol(
        self,
        *,
        task: ScientificTask,
        candidate: CandidateExperiment,
        round_id: int,
        model_name: str = "ElasticNet",
        model_parameters: dict[str, Any] | None = None,
        protocol_refinements: list[ProtocolRefinementSuggestion] | None = None,
        tree=None,
        uncertainty_records=None,
    ) -> ExperimentProtocol:
        effective_experiment_type = str(
            (candidate.experiment_type or candidate.type or "incremental_prediction").strip()
        )
        tool = get_tool(effective_experiment_type)
        if tool is None:
            raise ValueError(
                f"实验类型 {effective_experiment_type!r} 未注册，"
                "禁止静默降级为普通增量实验，无法生成执行协议。"
            )
        if not tool.available:
            refusal = "；".join(tool.refusal_reasons) or "执行器尚未接通"
            raise ValueError(
                f"实验类型 {effective_experiment_type!r} 当前不可选（{refusal}），"
                "不允许改标签后按增量实验执行。"
            )
        is_baseline_experiment = effective_experiment_type == "baseline_benchmark"
        control_features = [item for item in candidate.design.control if item]
        treatment_features = [item for item in candidate.design.treatment if item]
        if is_baseline_experiment:
            if not treatment_features:
                raise ValueError("对照组实验至少需要一组对照组变量，无法生成执行协议。")
            control_features = []
        else:
            if not control_features or not treatment_features:
                raise ValueError("区分性实验缺少对照组或实验组变量，无法生成执行协议。")
            if set(control_features) == set(treatment_features):
                raise ValueError(DIFFERENTIATING_EXPERIMENT_ERROR)
        candidate.design.control = control_features
        candidate.design.treatment = treatment_features
        design = candidate.design
        max_lag_day = max((max(values) for values in design.lags.values()), default=0)
        forecast_horizon_days = design.forecast_horizon_days or max_lag_day or 3
        past_lag_days = design.past_lag_days or min(max_lag_day, 3) or 1
        window_size = design.window_size or 14
        parameters = {
            "window_size": window_size,
            "alpha": 0.3,
            "l1_ratio": 0.5,
            "use_lag_feature": bool(candidate.design.lags),
            "max_lag_day": max_lag_day,
            "forecast_horizon_days": forecast_horizon_days,
            "past_lag_days": past_lag_days,
            "random_state": 42,
            "test_split_ratio": 0.2,
        }
        if effective_experiment_type == "model_sensitivity":
            normalized_families: list[str] = []
            for family in (design.model_families or ["ridge", "elasticnet"]):
                normalized = str(family).strip().lower()
                if normalized == "enet":
                    normalized = "elasticnet"
                if normalized in {"ridge", "elasticnet"} and normalized not in normalized_families:
                    normalized_families.append(normalized)
            if len(normalized_families) < 2:
                normalized_families = ["ridge", "elasticnet"]
            parameters["model_families"] = normalized_families
        parameters["feature_groups"] = {
            "mode": "baseline_single_arm" if is_baseline_experiment else "comparative",
            "baseline": list(treatment_features if is_baseline_experiment else control_features),
            "treatment": list(treatment_features),
        }
        parameters["candidate_design_snapshot"] = {
            "candidate_id": candidate.experiment_id,
            "experiment_type": effective_experiment_type,
            "target": candidate.design.target,
            "control": list(control_features),
            "treatment": list(treatment_features),
            "model_families": parameters.get("model_families", []),
            "lags": {key: list(values) for key, values in candidate.design.lags.items()},
            "probe_axis": design.probe_axis,
            "forecast_horizon_days": forecast_horizon_days,
            "past_lag_days": past_lag_days,
            "window_size": window_size,
            "control_lag_days": design.control_lag_days,
            "treatment_lag_days": design.treatment_lag_days,
            "control_forecast_horizon_days": design.control_forecast_horizon_days,
            "treatment_forecast_horizon_days": design.treatment_forecast_horizon_days,
        }
        if model_parameters:
            parameters.update(model_parameters)
        refinement_notes: list[str] = []
        refinement_steps: list[ExperimentStep] = []
        if protocol_refinements:
            for refinement in protocol_refinements:
                parameters.update(refinement.suggested_model_parameters)
                refinement_notes.append(f"planner_refinement:{refinement.refinement_type}")
                if refinement.rationale:
                    refinement_notes.append(f"planner_rationale:{refinement.rationale}")
                if refinement.suggested_model_parameters:
                    refinement_notes.append(
                        "planner_model_parameters:"
                        + json.dumps(
                            refinement.suggested_model_parameters,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                refinement_notes.extend(refinement.protocol_notes)
                refinement_steps.extend(step.model_copy(deep=True) for step in refinement.suggested_steps)

        if model_parameters:
            # 用户在审批时显式给出的模型参数优先级最高，覆盖 LLM refinement 的自动建议。
            parameters.update(model_parameters)
            refinement_notes.append(
                "planner_model_parameters:"
                + json.dumps(model_parameters, ensure_ascii=False, separators=(",", ":"))
            )
            refinement_notes.append("planner_rationale:PI 在审批环节显式覆写模型参数")

        # 时间窗必须使用最终生效的模型参数；仅当候选设计显式给出不同的双臂窗口
        # （例如超前窗口轴）时才保留双臂差异，避免调参结果被旧默认值覆盖。
        control_lag = int(parameters.get("past_lag_days", past_lag_days))
        treatment_lag = int(parameters.get("past_lag_days", past_lag_days))
        if (
            design.control_lag_days is not None
            and design.treatment_lag_days is not None
            and design.control_lag_days != design.treatment_lag_days
        ):
            control_lag = design.control_lag_days
            treatment_lag = design.treatment_lag_days
        final_horizon = int(parameters.get("forecast_horizon_days", forecast_horizon_days))
        final_window = int(parameters.get("window_size", window_size))
        parameters["arm_time_windows"] = {
            "baseline": {
                "window_size": final_window,
                "forecast_horizon_days": final_horizon,
                "past_lag_days": control_lag,
            },
            "treatment": {
                "window_size": final_window,
                "forecast_horizon_days": final_horizon,
                "past_lag_days": treatment_lag,
            },
        }
        forecast_horizon_days = final_horizon
        past_lag_days = control_lag
        window_size = final_window

        # 审批参数可能与候选设计默认值不同（例如 PI 把 5/7/21 覆写为
        # 3/3/14）。协议校验要求所有时间源一致，因此必须把最终值同步回
        # design 快照，不能只写 model.parameters / arm_time_windows。
        final_design = design.model_copy(
            update={
                "forecast_horizon_days": final_horizon,
                "past_lag_days": control_lag,
                "window_size": final_window,
                "control_lag_days": control_lag,
                "treatment_lag_days": treatment_lag,
                "control_forecast_horizon_days": final_horizon,
                "treatment_forecast_horizon_days": final_horizon,
            }
        )
        design = final_design
        parameters["candidate_design_snapshot"].update(
            {
                "forecast_horizon_days": final_horizon,
                "past_lag_days": control_lag,
                "window_size": final_window,
                "control_lag_days": control_lag,
                "treatment_lag_days": treatment_lag,
                "control_forecast_horizon_days": final_horizon,
                "treatment_forecast_horizon_days": final_horizon,
            }
        )

        base_steps = [ExperimentStep(step=1, action="data_quality_check")]
        if is_baseline_experiment:
            base_steps.append(
                ExperimentStep(step=2, action="run_model", parameters={"features": "treatment", "label": "baseline"})
            )
        elif effective_experiment_type == "model_sensitivity":
            run_specs = design.runs_spec or []
            if run_specs:
                for step_index, run_spec in enumerate(run_specs, start=2):
                    base_steps.append(
                        ExperimentStep(
                            step=step_index,
                            action="run_model",
                            parameters={
                                "run_id": run_spec.run_id,
                                "features": list(run_spec.feature_columns),
                                "model_family": run_spec.model_family,
                            },
                        )
                    )
                base_steps.append(
                    ExperimentStep(
                        step=len(run_specs) + 2,
                        action="comparison",
                        parameters={
                            "contrasts_spec": [item.contrast_id for item in design.contrasts_spec],
                        },
                    )
                )
            else:
                for family in parameters["model_families"]:
                    base_steps.extend(
                        [
                            ExperimentStep(
                                step=2,
                                action="run_model",
                                parameters={"features": "control", "label": f"{family}_baseline", "model_family": family},
                            ),
                            ExperimentStep(
                                step=3,
                                action="run_model",
                                parameters={"features": "treatment", "label": f"{family}_treatment", "model_family": family},
                            ),
                        ]
                    )
                base_steps.append(
                    ExperimentStep(
                        step=4,
                        action="comparison",
                        parameters={"model_families": parameters["model_families"]},
                    )
                )
        else:
            base_steps.extend(
                [
                    ExperimentStep(step=2, action="run_model", parameters={"features": "control", "label": "baseline"}),
                    ExperimentStep(step=3, action="run_model", parameters={"features": "treatment", "label": "treatment"}),
                    ExperimentStep(
                        step=4,
                        action="comparison",
                        parameters={"baseline_label": "baseline", "treatment_label": "treatment"},
                    ),
                ]
            )
        disagreement_steps = _build_disagreement_steps(candidate)
        steps = _renumber_steps(refinement_steps + disagreement_steps + base_steps)
        protocol_notes = [
            f"mapped_from_candidate:{candidate.experiment_id}",
            candidate.distinguishing_insight or "no distinguishing insight provided",
        ]
        if is_baseline_experiment:
            protocol_notes.append("experiment_mode:baseline_single_arm")
        protocol_notes.extend(
            _disagreement_notes(
                candidate,
                tree=tree,
                uncertainty_records=uncertainty_records,
            )
        )

        return ExperimentProtocol(
            experiment_id=candidate.experiment_id,
            round_id=round_id,
            task_id=task.task_id,
            target=design.target,
            experiment_type=effective_experiment_type,
            scientific_objective=candidate.scientific_question or candidate.purpose,
            forecast_horizon_days=forecast_horizon_days,
            features=design,
            runs_spec=list(design.runs_spec or []),
            contrasts_spec=list(design.contrasts_spec or []),
            diagnostics_spec=list(design.diagnostics_spec or []),
            alignment=design.alignment,
            model={
                "name": model_name,
                "parameters": parameters,
            },
            tested_hypotheses=candidate.tested_hypotheses,
            target_uncertainties=candidate.related_uncertainties,
            disagreement_context=candidate.disagreement_context,
            hypothesis_source_context=candidate.hypothesis_source_context,
            hypothesis_predictions=candidate.hypothesis_predictions,
            budget=task.payload.constraints.resource_budget,
            steps=steps,
            constraints=ProtocolConstraints(
                no_future_information=task.payload.constraints.no_future_information,
                target_immutable=True,
            ),
            source_candidate_id=candidate.experiment_id,
            requires_human_review=candidate.requires_human_review,
            responded_prior_finding=candidate.responded_prior_finding,
            target_uncertainty_focus=candidate.target_uncertainty_focus,
            changed_factors=list(candidate.changed_factors or []),
            held_fixed_factors=list(candidate.held_fixed_factors or []),
            expected_outcome_branches=dict(candidate.expected_outcome_branches or {}),
            exploratory=bool(candidate.exploratory),
            notes=deep_clean_text(
                protocol_notes + refinement_notes,
                tree=tree,
                uncertainty_records=uncertainty_records,
            ),
        )


class HumanControlService:
    """Persist human-in-the-loop approvals, pauses, and protocol generation."""

    def __init__(
        self,
        repository: UnifiedStateRepository,
        *,
        project_root: Path | None = None,
        mapper: CandidateProtocolMapper | None = None,
        hypothesis_generator: HypothesisGenerationService | None = None,
        run_shap: bool = False,
        planner_mode: str = "programmatic",
        planner_output_builder: PlannerOutputBuilder | None = None,
        scientific_interpreter: ScientificInterpreterLLM | None = None,
        rag_service: RAGService | None = None,
        hypothesis_proposer: HypothesisProposerLLM | None = None,
        scientific_questioner: ScientificQuestionerLLM | None = None,
        experiment_planner: ExperimentPlannerLLM | None = None,
        experiment_designer: CandidateExperimentDesignerLLM | None = None,
        experiment_writer: CandidateExperimentWriterLLM | None = None,
        parallel_reasoning_orchestrator: ParallelReasoningOrchestrator | None = None,
    ) -> None:
        self.repository = repository
        self.project_root = project_root or repository.project_root
        self.config_dir = self.project_root / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.protocol_path = self.config_dir / "latest_protocol.json"
        self.mapper = mapper or CandidateProtocolMapper()
        self.hypothesis_generator = hypothesis_generator or HypothesisGenerationService()
        self.run_shap = run_shap
        self.planner_mode = planner_mode
        if planner_output_builder is not None:
            self.planner_output_builder = planner_output_builder
        elif planner_mode == "llm":
            self.planner_output_builder = PlannerOutputBuilder(
                mode="llm",
                central_controller=CentralControllerLLM(
                    gateway=LLMGateway(model=get_llm_model_for_role("central_controller"))
                ),
            )
        else:
            self.planner_output_builder = PlannerOutputBuilder(mode=planner_mode)
        self.scientific_interpreter = scientific_interpreter or ScientificInterpreterLLM(
            gateway=LLMGateway(model=get_llm_model_for_role("scientific_interpreter"))
        )
        self.rag_service = rag_service or RAGService(project_root=self.project_root)
        self.hypothesis_proposer = hypothesis_proposer or HypothesisProposerLLM(
            gateway=LLMGateway(model=get_llm_model_for_role("hypothesis_proposer"))
        )
        self.scientific_questioner = scientific_questioner or ScientificQuestionerLLM(
            gateway=LLMGateway(
                model=get_llm_model_for_role("scientific_questioner"),
                allow_fallback=False,
            )
        )
        self.experiment_planner = experiment_planner or ExperimentPlannerLLM(
            gateway=LLMGateway(model=get_llm_model_for_role("experiment_planner"))
        )
        self.experiment_designer = experiment_designer or CandidateExperimentDesignerLLM()
        self.experiment_writer = experiment_writer or CandidateExperimentWriterLLM()
        self.parallel_reasoning_orchestrator = parallel_reasoning_orchestrator or ParallelReasoningOrchestrator(
            hypothesis_proposer=self.hypothesis_proposer,
            scientific_questioner=self.scientific_questioner,
        )

    def request_experiment_selection_review(self) -> dict[str, Any]:
        candidate_set = self.repository.load_candidate_experiments()
        top_candidate = candidate_set.top_candidate()
        process_state = self.repository.load_process_state()
        process_state = self._mark_selection_waiting(process_state)
        self.repository.save_process_state(process_state)
        decision_log = self.repository.load_decision_log()

        if top_candidate is None:
            decision_log.decisions.append(
                DecisionEntry(
                    decision_id=self._next_decision_id(decision_log),
                    timestamp=datetime.now(),
                    round_id=candidate_set.round,
                    phase="experiment_planning",
                    step="experiment_selection",
                    decision_type="experiment_selection_requested",
                    made_by="system",
                    summary="系统尝试发起候选实验审批，但当前没有可审批候选。",
                    details={"status": "no_candidate"},
                )
            )
            self.repository.save_decision_log(decision_log)
            return {
                "status": "no_candidate",
                "message": "当前没有可审批的候选实验。",
            }

        decision_log.decisions.append(
            DecisionEntry(
                decision_id=self._next_decision_id(decision_log),
                timestamp=datetime.now(),
                round_id=candidate_set.round,
                phase="experiment_planning",
                step="experiment_selection",
                decision_type="experiment_selection_requested",
                made_by="system",
                summary=f"系统请求 human_pi 审批候选实验 {top_candidate.experiment_id}",
                details={
                    "candidate_id": top_candidate.experiment_id,
                    "utility_score": top_candidate.utility_score,
                },
            )
        )
        self.repository.save_decision_log(decision_log)
        return {
            "status": "awaiting_human_review",
            "recommended_candidate": top_candidate.model_dump(mode="json", exclude_none=True),
            "current_phase": process_state.current_phase,
            "current_step": process_state.current_step,
            "approval_options": ["批准并继续", "修改参数", "拒绝并重选", "暂停实验"],
        }

    def prepare_initial_review_context(
        self,
        *,
        data_dictionary: DataDictionary,
    ) -> ReasoningPlannerInput:
        """Persist first-round planner input so the UI can display questioner traces before approval."""
        uncertainties = self.repository.load_uncertainties()
        planner_input_builder = ReasoningPlannerInputBuilder(self.repository)
        planner_input = planner_input_builder.build(
            data_dictionary=data_dictionary,
            source_round_id=0,
            human_feedback=None,
        )
        if self.planner_mode == "llm":
            planner_input, uncertainties = self._augment_next_round_context_with_llm_services(
                planner_input=planner_input,
                uncertainties=uncertainties,
            )
            self.repository.save_uncertainties(uncertainties)
        else:
            planner_input, uncertainties = self._mine_programmatic_uncertainties(
                planner_input=planner_input,
                uncertainties=uncertainties,
            )
            self.repository.save_uncertainties(uncertainties)
        self.repository.save_planner_input(planner_input)
        DecisionLayerService(
            self.repository,
            experiment_designer=self.experiment_designer,
            experiment_writer=self.experiment_writer,
        ).build_candidate_plan(
            task=self.repository.load_task(),
            data_dictionary=data_dictionary,
            planner_input=planner_input,
            target_round=planner_input.next_round_id,
        )
        self._log_planner_input_prepared(planner_input)
        return planner_input

    def prepare_initial_hypothesis_review_context(
        self,
        *,
        data_dictionary: DataDictionary,
    ) -> ReasoningPlannerInput:
        """Phase 1 first-round bootstrap: H/C proposals and frozen hypothesis tree, then stop.

        Uncertainty mining and candidate generation are intentionally deferred until
        ``confirm_hypothesis_tree`` is called, so the downstream work always consumes
        the exact tree the human PI has reviewed.
        """
        uncertainties = self.repository.load_uncertainties()
        planner_input_builder = ReasoningPlannerInputBuilder(self.repository)
        planner_input = planner_input_builder.build(
            data_dictionary=data_dictionary,
            source_round_id=0,
            human_feedback=None,
        )
        if self.planner_mode == "llm":
            planner_input = self._augment_hypothesis_context_with_llm_services(
                planner_input=planner_input,
                uncertainties=uncertainties,
            )

        task = self.repository.load_task()
        generation_result = self.hypothesis_generator.build_tree(
            task=task,
            data_dictionary=data_dictionary,
            uncertainties=uncertainties,
            planner_input=planner_input,
            existing_tree=self.repository.load_hypothesis_tree(),
            current_round=planner_input.next_round_id,
        )
        uncertainties.records = list(generation_result.updated_uncertainties)
        uncertainties.current_round = max(uncertainties.current_round, planner_input.next_round_id)
        self.repository.save_uncertainties(uncertainties)

        tree = generation_result.tree
        if tree.latest_update is None or tree.latest_update.round < planner_input.next_round_id:
            tree.latest_update = LatestTreeUpdate(
                round=planner_input.next_round_id,
                event="hypothesis_review_ready",
                description="首轮竞争假设与科学质询已完成，等待人工确认后进入不确定性识别。",
            )
        self.repository.save_hypothesis_tree(tree)
        if generation_result.generated_node_ids:
            self._log_hypothesis_generation(
                round_id=0,
                next_round_id=planner_input.next_round_id,
                generation_result=generation_result,
                mode=generation_result.mode,
            )

        self.repository.save_planner_input(planner_input)
        self._log_planner_input_prepared(planner_input)
        process_state = self.repository.load_process_state()
        process_state.current_round = max(process_state.current_round, planner_input.next_round_id)
        self.repository.save_process_state(self._mark_hypothesis_confirmation_waiting(process_state))
        return planner_input

    def confirm_hypothesis_tree(
        self,
        *,
        human_notes: str | None = None,
        nodes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Freeze the reviewed hypothesis tree and stop for scientific questioning."""
        repository = self.repository
        process_state = repository.load_process_state()
        if process_state.current_stage != "awaiting_hypothesis_confirmation":
            raise ValueError(
                "当前不在假设树待确认状态；请先让系统生成假设树后再提交确认。"
            )

        decision_log = repository.load_decision_log()
        decision_log.decisions.append(
            DecisionEntry(
                decision_id=self._next_decision_id(decision_log),
                timestamp=datetime.now(),
                round_id=process_state.current_round,
                phase=process_state.current_phase or "hypothesis_generation",
                    step="hypothesis_review",
                    decision_type="hypothesis_tree_confirmed",
                    made_by="human_pi",
                summary="人工确认当前假设树，冻结后先进入科学质询再进入不确定性识别。",
                details={"human_notes": human_notes},
            )
        )
        repository.save_decision_log(decision_log)

        if process_state.current_phase == "next_round":
            return RoundOrchestrator(self).freeze_next_round_hypothesis_tree(
                human_notes=human_notes,
                nodes=nodes,
            )

        task = repository.load_task()
        data_dictionary = repository.load_data_dictionary()
        planner_input = repository.load_planner_input()
        uncertainties = repository.load_uncertainties()
        if nodes:
            validated_nodes = [
                node
                for raw_node in nodes
                if (node := self._coerce_hypothesis_node(raw_node)) is not None
            ]
            if not validated_nodes:
                raise ValueError("假设树修改载荷无法通过校验，请刷新页面后重试。")
            tree = self.hypothesis_generator.rebuild_tree(
                task=task,
                nodes=validated_nodes,
                current_round=planner_input.next_round_id,
                description="假设树经人工 PI 修改后冻结",
                data_dictionary=data_dictionary,
            )
            repository.save_hypothesis_tree(tree)
        else:
            tree = repository.load_hypothesis_tree()
            tree.latest_update = LatestTreeUpdate(
                round=max(int(tree.current_round or 0), planner_input.next_round_id),
                event="hypothesis_tree_confirmed",
                description="人工确认当前假设树，冻结后先进入科学质询。",
            )
            repository.save_hypothesis_tree(tree)

        repository.save_pre_questioning_tree(tree)
        repository.save_planner_input(planner_input)
        process_state = repository.load_process_state()
        process_state.current_stage = "awaiting_scientific_questioning"
        process_state.current_step = "scientific_questioning"
        process_state.progress_percentage = 40
        process_state.phases["hypothesis_generation"] = ProcessPhase(
            status="in_progress",
            started_at=process_state.phases.get("hypothesis_generation", ProcessPhase(status="in_progress")).started_at or datetime.now(),
            steps={
                "hypothesis_generation": ProcessStep(
                    name="假设生成",
                    status="completed",
                    requires_user_approval=False,
                ),
                "hypothesis_review": ProcessStep(
                    name="假设树确认",
                    status="completed",
                    requires_user_approval=True,
                    notes="假设树已冻结，等待开始科学质询。",
                ),
                "scientific_questioning": ProcessStep(
                    name="科学质询",
                    status="in_progress",
                    requires_user_approval=True,
                    notes="LLM 将逐条质询假设并更新支持度与状态。",
                ),
            },
        )
        repository.save_process_state(process_state)
        return {
            "planning_status": "hypothesis_tree_frozen",
            "status": "awaiting_scientific_questioning",
            "message": "假设树已冻结，等待开始科学质询。",
        }

    def run_hypothesis_scientific_questioning(self) -> dict[str, Any]:
        """Run the LLM scientific questioner and apply per-hypothesis support updates."""
        repository = self.repository
        process_state = repository.load_process_state()
        if process_state.current_stage != "awaiting_scientific_questioning":
            raise ValueError("当前不在科学质询待开始状态；请先确认假设树。")
        if process_state.current_phase == "next_round":
            return RoundOrchestrator(self).run_next_round_scientific_questioning()
        return self._run_first_round_scientific_questioning()

    def rerun_hypothesis_scientific_questioning(self) -> dict[str, Any]:
        """Re-run LLM scientific questioning after it has already completed once."""
        repository = self.repository
        process_state = repository.load_process_state()
        if process_state.current_stage not in {
            "awaiting_scientific_questioning",
            "awaiting_uncertainty_identification",
        }:
            raise ValueError("当前阶段不支持重新生成科学质询；请先完成假设树确认。")
        if process_state.current_phase == "next_round":
            return RoundOrchestrator(self).run_next_round_scientific_questioning()
        return self._run_first_round_scientific_questioning()

    def undo_scientific_questioning(self) -> dict[str, Any]:
        """Restore the pre-questioning tree after the newest advisory pass."""
        repository = self.repository
        process_state = repository.load_process_state()
        if process_state.current_stage != "awaiting_uncertainty_identification":
            raise ValueError("当前阶段不是科学质询完成后，无法撤销本轮质询结果。")
        pre_tree = repository.load_pre_questioning_tree()
        if pre_tree is None:
            raise ValueError("未找到质询前的假设树快照，无法撤销本轮质询。")
        planner_input = repository.load_planner_input()
        round_id = max(
            int(pre_tree.current_round or 0),
            int(process_state.current_round or 0),
            int(planner_input.next_round_id or 0),
        )
        repository.save_hypothesis_tree(pre_tree)
        trace_prefix = f"SQ{round_id:02d}"
        planner_input.recent_reasoning_traces = [
            trace
            for trace in planner_input.recent_reasoning_traces
            if not (trace.stage == "scientific_questioner" and trace.trace_id.startswith(trace_prefix))
        ]
        planner_input.planner_guidance = [
            note
            for note in planner_input.planner_guidance
            if not note.startswith("supplemental_llm_hypothesis:")
        ]
        repository.save_planner_input(planner_input)
        process_state.current_stage = "awaiting_scientific_questioning"
        process_state.current_step = "scientific_questioning"
        process_state.progress_percentage = 40
        phase = process_state.phases.get("hypothesis_generation")
        if phase is not None:
            step = phase.steps.get("scientific_questioning")
            if step is not None:
                step.status = "in_progress"
                step.notes = "人工 PI 撤销本轮科学质询，等待重新开始质询。"
        repository.save_process_state(process_state)
        decision_log = repository.load_decision_log()
        decision_log.decisions.append(
            DecisionEntry(
                decision_id=self._next_decision_id(decision_log),
                timestamp=datetime.now(),
                round_id=round_id,
                phase="hypothesis_generation",
                step="scientific_questioning",
                decision_type="scientific_questioning_undone",
                made_by="human_pi",
                summary="人工 PI 撤销本轮科学质询，假设树已恢复至质询前快照，等待重新质询。",
                details={
                    "restored_tree_path": repository.relativize(repository.paths.hypothesis_tree),
                    "pre_questioning_snapshot": repository.pre_questioning_tree_path().name,
                    "cleared_traces": "scientific_questioner",
                    "cleared_trace_prefix": trace_prefix,
                    "cleared_guidance": "supplemental_llm_hypothesis",
                },
            )
        )
        repository.save_decision_log(decision_log)
        return {
            "planning_status": "scientific_questioning_undone",
            "status": "awaiting_scientific_questioning",
            "message": "已撤销本轮科学质询，假设树恢复至确认时点，可以重新开始科学质询。",
        }

    def _run_first_round_scientific_questioning(self) -> dict[str, Any]:
        repository = self.repository
        task = repository.load_task()
        data_dictionary = repository.load_data_dictionary()
        planner_input = repository.load_planner_input()
        tree = repository.load_hypothesis_tree()
        process_state = repository.load_process_state()
        current_round = int(planner_input.next_round_id)
        # 重新生成质询时按质询前快照计算作用域，保持与首次运行一致。
        scope_tree = tree
        if process_state.current_stage == "awaiting_uncertainty_identification":
            pre_questioning_tree = repository.load_pre_questioning_tree()
            if pre_questioning_tree is not None:
                scope_tree = pre_questioning_tree
        targeted_ids = compute_targeted_interrogation_scope(
            scope_tree,
            current_round=current_round,
        )
        rag_context = self.rag_service.build_context_bundle(planner_input)
        questioner_output = self.scientific_questioner.challenge_hypothesis_tree(
            planner_input=planner_input,
            rag_context=rag_context,
            tree=scope_tree,
            targeted_hypothesis_ids=sorted(targeted_ids),
        )
        questioner_output = self._align_questioner_output_to_dictionary(
            planner_input=planner_input,
            questioner_output=questioner_output,
            tree=tree,
            uncertainty_records=repository.load_uncertainties().records,
        )
        self._apply_hypothesis_updates_to_tree(
            tree=tree,
            updates=questioner_output.hypothesis_updates,
            round_id=current_round,
            targeted_hypothesis_ids=targeted_ids,
        )
        min_active = int(task.payload.constraints.min_active_hypotheses or 3)
        self._propagate_tree_state_for_round(
            tree=tree,
            round_id=current_round,
            min_active_hypotheses=min_active,
        )
        refresh_tree_metadata(tree)
        self._sync_questioning_record_statuses(
            tree=tree,
            round_id=current_round,
        )
        supplemental = self._supplement_active_hypotheses_if_needed(
            tree=tree,
            task=task,
            data_dictionary=data_dictionary,
            planner_input=planner_input,
            rag_context=rag_context,
            min_active_hypotheses=min_active,
        )
        tree.latest_update = LatestTreeUpdate(
            round=current_round,
            event="scientific_questioning_completed",
            description=(
                (
                    f"科学质询完成，本轮定向质询 {len(targeted_ids)} 条假设，"
                    "已按 LLM 结论更新支持度与状态；继承节点保留上一轮质询痕迹"
                )
                + (
                    f"；活跃假设不足下限，已由真实 LLM 补充生成 {supplemental.display_hypothesis_id}。"
                    if supplemental is not None
                    else "。"
                )
            ),
        )
        repository.save_hypothesis_tree(tree)
        planner_input.planner_guidance.extend(
            note
            for note in questioner_output.guidance_notes
            if note not in planner_input.planner_guidance
        )
        planner_input.recent_reasoning_traces.extend(
            self._notes_to_traces(
                notes=questioner_output.challenge_points,
                round_id=planner_input.source_round_id,
                stage="scientific_questioner",
                prefix="SQ",
                referenced_variables=[
                    feature
                    for item in questioner_output.proposed_uncertainties
                    for feature in item.features
                ],
            )
        )
        planner_input.recent_reasoning_traces = planner_input.recent_reasoning_traces[-10:]
        repository.save_planner_input(planner_input)
        self._log_scientific_questioning_completed(
            round_id=current_round,
            tree=tree,
            updates=questioner_output.hypothesis_updates,
            supplemental_hypothesis=supplemental,
        )
        process_state = repository.load_process_state()
        phase = process_state.phases.get("hypothesis_generation")
        if phase is not None:
            step = phase.steps.get("scientific_questioning")
            if step is None:
                step = ProcessStep(
                    name="科学质询",
                    status="completed",
                    requires_user_approval=True,
                )
                phase.steps["scientific_questioning"] = step
            step.status = "completed"
            step.notes = "LLM 完成科学质询，支持度与状态已回写。"
        process_state.current_stage = "awaiting_uncertainty_identification"
        process_state.current_step = "step_1"
        process_state.progress_percentage = 55
        repository.save_process_state(process_state)
        return {
            "planning_status": "scientific_questioning_completed",
            "status": "awaiting_uncertainty_identification",
            "message": "科学质询完成，假设树支持度与状态已更新，等待进入不确定性识别。",
            "hypothesis_updates": [
                {
                    "hypothesis_id": update.hypothesis_id,
                    "impact_direction": update.impact_direction,
                    "rationale": update.rationale,
                }
                for update in questioner_output.hypothesis_updates
            ],
        }

    def start_uncertainty_identification(self) -> dict[str, Any]:
        """Generate uncertainties, candidate experiments and enter approval review."""
        repository = self.repository
        process_state = repository.load_process_state()
        if process_state.current_stage != "awaiting_uncertainty_identification":
            raise ValueError("当前不在等待不确定性识别状态；请先完成科学质询。")
        if process_state.current_phase == "next_round":
            return RoundOrchestrator(self).continue_next_round_planning()
        return self._continue_first_round_planning_after_questioning()

    def _continue_first_round_planning_after_questioning(self) -> dict[str, Any]:
        repository = self.repository
        task = repository.load_task()
        data_dictionary = repository.load_data_dictionary()
        planner_input = repository.load_planner_input()
        uncertainties = repository.load_uncertainties()
        if self.planner_mode == "llm":
            planner_input, uncertainties = self._augment_uncertainty_context_with_llm_services(
                planner_input=planner_input,
                uncertainties=uncertainties,
            )
        else:
            planner_input, uncertainties = self._mine_programmatic_uncertainties(
                planner_input=planner_input,
                uncertainties=uncertainties,
            )
        repository.save_uncertainties(uncertainties)
        repository.save_planner_input(planner_input)
        candidate_plan = DecisionLayerService(
            repository,
            experiment_designer=self.experiment_designer,
            experiment_writer=self.experiment_writer,
        ).build_candidate_plan(
            task=task,
            data_dictionary=data_dictionary,
            planner_input=planner_input,
            target_round=planner_input.next_round_id,
        )
        review = self.request_experiment_selection_review()
        return {
            "planning_status": "candidate_plan_rebuilt",
            "status": review["status"],
            "message": review.get("message", "候选实验已生成，等待人工审批。"),
            "candidate_plan": candidate_plan,
            "next_review": review,
        }

    def rebuild_candidate_plan(
        self,
        *,
        task: ScientificTask,
        data_dictionary: DataDictionary,
        planner_input: ReasoningPlannerInput,
        target_round: int | None,
    ) -> dict[str, Any]:
        """Rebuild the candidate plan with the configured LLM designer/writer."""
        candidate_plan = DecisionLayerService(
            self.repository,
            experiment_designer=self.experiment_designer,
            experiment_writer=self.experiment_writer,
        ).build_candidate_plan(
            task=task,
            data_dictionary=data_dictionary,
            planner_input=planner_input,
            target_round=target_round,
        )
        review = self.request_experiment_selection_review()
        return {
            "planning_status": "candidate_plan_rebuilt",
            "status": review["status"],
            "message": review.get("message", "候选实验已重新生成，等待人工审批。"),
            "candidate_plan": candidate_plan,
            "next_review": review,
        }

    def _apply_hypothesis_updates_to_tree(
        self,
        *,
        tree,
        updates,
        round_id: int,
        targeted_hypothesis_ids: set[str] | list[str] | None = None,
    ) -> None:
        from core.scientific_questioner_llm import HypothesisQuestioningUpdate

        try:
            uncertainty_records = self.repository.load_uncertainties().records
        except Exception:
            uncertainty_records = []

        def clean_questioning_text(value: str | None) -> str:
            return clean_llm_text(
                value or "",
                tree=tree,
                uncertainty_records=uncertainty_records,
            ).strip()

        targeted = (
            set(targeted_hypothesis_ids)
            if targeted_hypothesis_ids is not None
            else None
        )
        updates_by_id: dict[str, HypothesisQuestioningUpdate] = {}
        for update in updates:
            if not update.hypothesis_id:
                continue
            if targeted is not None and update.hypothesis_id not in targeted:
                continue
            updates_by_id[update.hypothesis_id] = update
        node_index = tree.node_index()
        for node in tree.nodes:
            if targeted is not None and node.hypothesis_id not in targeted:
                continue
            update = updates_by_id.get(node.hypothesis_id)
            if update is None:
                update = HypothesisQuestioningUpdate(
                    hypothesis_id=node.hypothesis_id,
                    impact_direction="clarifies",
                    impact_strength=0.0,
                    confidence=0.5,
                    rationale="科学质询未给出明确方向，本轮仅保留观察记录。",
                )
            node.questioning_records = [
                record
                for record in node.questioning_records
                if record.round != round_id
            ]
            parent_support = None
            if node.parent_id:
                parent_node = node_index.get(node.parent_id)
                if parent_node is not None:
                    parent_support = parent_node.support_score
            outcome = compute_support_update_from_reasoning(
                current_support=node.support_score,
                impact_direction=update.impact_direction,
                impact_strength=update.impact_strength,
                confidence=update.confidence,
                falsification_basis=update.falsification_basis,
                previous_status=node.status,
                support_history=node.support_history,
                current_round=round_id,
                draft_to_pending=True,
                parent_support=parent_support,
                evidence_source="advisory",
            )
            node.support_score = outcome.support_after
            node.status = outcome.status
            node.updated_at_round = round_id
            if not any(
                item.round == round_id and item.event == "scientific_questioning"
                for item in node.support_history
            ):
                node.support_history.append(
                    SupportHistoryEntry(
                        round=round_id,
                        score=outcome.support_after,
                        event="scientific_questioning",
                        evidence_source="advisory",
                    )
                )
            node.questioning_records.append(
                HypothesisQuestioningRecord(
                    round=round_id,
                    impact_direction=update.impact_direction,
                    impact_strength=update.impact_strength,
                    confidence=update.confidence,
                    rationale=(
                        clean_questioning_text(update.rationale)
                        or "科学质询未给出明确理由。"
                    ),
                    support_before=outcome.support_before,
                    support_after=outcome.support_after,
                    status=outcome.status,
                    source_type="advisory",
                    falsification_basis=clean_questioning_text(update.falsification_basis),
                )
            )
            if node.status == "pruned":
                node.pruned_at_round = node.pruned_at_round or round_id
                node.prune_reason = f"科学质询：{(clean_questioning_text(update.rationale) or '证据不足，假设被剪枝。')[:180]}"

    def _propagate_tree_state_for_round(
        self,
        *,
        tree,
        round_id: int,
        min_active_hypotheses: int,
    ) -> None:
        from core.state_updater import _propagate_tree_state

        _propagate_tree_state(
            tree=tree,
            round_id=round_id,
            min_active_hypotheses=min_active_hypotheses,
        )

    @staticmethod
    def _sync_questioning_record_statuses(*, tree, round_id: int) -> None:
        """让科学质询输出与节点最终状态使用同一个状态字段。"""
        for node in tree.nodes:
            for record in node.questioning_records:
                if record.round == round_id:
                    record.status = node.status

    def _supplement_active_hypotheses_if_needed(
        self,
        *,
        tree,
        task,
        data_dictionary,
        planner_input,
        rag_context,
        min_active_hypotheses: int,
    ):
        active_count = len(tree.active_hypotheses)
        if active_count >= min_active_hypotheses:
            return None
        proposal = self.hypothesis_proposer.propose_supplemental(
            planner_input=planner_input,
            rag_context=rag_context,
            tree=tree,
        )
        node = self.hypothesis_generator.append_supplemental_node(
            task=task,
            data_dictionary=data_dictionary,
            tree=tree,
            proposal=proposal,
            current_round=planner_input.next_round_id,
            min_active_hypotheses=min_active_hypotheses,
        )
        decision_log = self.repository.load_decision_log()
        decision_log.decisions.append(
            DecisionEntry(
                decision_id=self._next_decision_id(decision_log),
                timestamp=datetime.now(),
                round_id=planner_input.next_round_id,
                phase="hypothesis_generation",
                step="scientific_questioning",
                decision_type="supplemental_hypothesis_added",
                made_by="hypothesis_proposer_llm",
                summary=(
                    f"科学质询后活跃假设仅 {active_count} 条，低于下限 {min_active_hypotheses} 条，"
                    f"真实 LLM 补充生成 {node.display_hypothesis_id}。"
                ),
                details={
                    "hypothesis_id": node.hypothesis_id,
                    "display_hypothesis_id": node.display_hypothesis_id,
                    "support_score": node.support_score,
                    "status": node.status,
                    "source": proposal.get("source"),
                    "model": proposal.get("model"),
                    "generated_at": proposal.get("generated_at"),
                },
            )
        )
        self.repository.save_decision_log(decision_log)
        note = (
            f"supplemental_llm_hypothesis:{node.display_hypothesis_id} "
            f"质询后活跃假设仅 {active_count} 条，低于下限 {min_active_hypotheses} 条，"
            "已由真实 LLM 补充生成。"
        )
        if note not in planner_input.planner_guidance:
            planner_input.planner_guidance.append(note)
        return node

    def _log_scientific_questioning_completed(
        self,
        *,
        round_id: int,
        tree,
        updates,
        supplemental_hypothesis=None,
    ) -> None:
        decision_log = self.repository.load_decision_log()
        node_audit = []
        for node in tree.nodes:
            for record in node.questioning_records:
                if record.round != round_id:
                    continue
                node_audit.append(
                    {
                        "hypothesis_id": node.hypothesis_id,
                        "display_hypothesis_id": node.display_hypothesis_id,
                        "impact_direction": record.impact_direction,
                        "impact_strength": record.impact_strength,
                        "confidence": record.confidence,
                        "support_before": record.support_before,
                        "support_after": record.support_after,
                        "status": record.status,
                        "evidence_source": record.source_type,
                        "falsification_basis": record.falsification_basis or "",
                    }
                )
        decision_log.decisions.append(
            DecisionEntry(
                decision_id=self._next_decision_id(decision_log),
                timestamp=datetime.now(),
                round_id=round_id,
                phase="hypothesis_generation",
                step="scientific_questioning",
                decision_type="scientific_questioning_completed",
                made_by="scientific_questioner_llm",
                summary=(
                    "LLM 以顾问身份（advisory）完成逐假设科学质询；"
                    "弱化必须有可证否依据，且不支持越级剪枝或直接收敛。"
                ),
                details={
                    "updated_count": len(updates),
                    "active_like": sum(1 for node in tree.nodes if node.status in ACTIVE_LIKE),
                    "node_audit": node_audit,
                    "supplemental_hypothesis": (
                        {
                            "hypothesis_id": supplemental_hypothesis.hypothesis_id,
                            "display_hypothesis_id": supplemental_hypothesis.display_hypothesis_id,
                        }
                        if supplemental_hypothesis is not None
                        else None
                    ),
                },
            )
        )
        self.repository.save_decision_log(decision_log)

    def _coerce_hypothesis_node(self, raw_node: dict[str, Any]):
        from core.unified_schema import HypothesisNode

        try:
            return HypothesisNode.model_validate(raw_node)
        except Exception:
            return None

    def _repair_candidate_hypothesis_links(self, candidate: CandidateExperiment) -> None:
        if candidate.tested_hypotheses and candidate.hypothesis_predictions:
            return
        tree = self.repository.load_hypothesis_tree()
        node_index = tree.node_index()
        if not candidate.tested_hypotheses:
            inferences: list[str] = []
            uncertainty_state = self.repository.load_uncertainties()
            record_index = uncertainty_state.record_index()
            for uncertainty_id in candidate.related_uncertainties:
                record = record_index.get(uncertainty_id)
                if record is not None:
                    inferences.extend(record.related_hypotheses)
                    if not record.related_hypotheses:
                        inferences.extend(
                            related_hypotheses_for_uncertainty(
                                tree=tree,
                                question=record.question or candidate.scientific_question or "",
                                description=record.description or "",
                                existing=[],
                            )
                        )
            if not inferences:
                inferences = infer_related_hypothesis_ids(
                    tree,
                    candidate.scientific_question or candidate.purpose or "",
                    candidate.distinguishing_insight or "",
                    [],
                )
            candidate.tested_hypotheses = [
                hypothesis_id
                for hypothesis_id in dict.fromkeys(inferences)
                if hypothesis_id in node_index
            ] or [hypothesis_id for hypothesis_id in tree.active_hypotheses if hypothesis_id in node_index]
        if candidate.tested_hypotheses and not candidate.hypothesis_predictions:
            candidate.hypothesis_predictions = build_predictions_for_nodes(
                tree,
                candidate.tested_hypotheses,
            )

    def _repair_protocol_hypothesis_links(self, protocol: ExperimentProtocol) -> None:
        if protocol.tested_hypotheses and protocol.hypothesis_predictions:
            return
        candidate = protocol.source_candidate_id or ""
        candidate_set = self.repository.load_candidate_experiments()
        candidate_model = next(
            (
                item
                for item in candidate_set.candidates
                if item.experiment_id == candidate
            ),
            None,
        )
        tree = self.repository.load_hypothesis_tree()
        if candidate_model is not None:
            self._repair_candidate_hypothesis_links(candidate_model)
            protocol.tested_hypotheses = list(candidate_model.tested_hypotheses)
            protocol.hypothesis_predictions = candidate_model.hypothesis_predictions
        if not protocol.tested_hypotheses:
            protocol.tested_hypotheses = [
                hypothesis_id
                for hypothesis_id in tree.active_hypotheses
                if hypothesis_id in tree.node_index()
            ]
        if protocol.tested_hypotheses and not protocol.hypothesis_predictions:
            protocol.hypothesis_predictions = build_predictions_for_nodes(
                tree,
                protocol.tested_hypotheses,
            )

    def approve_candidate(
        self,
        *,
        candidate_id: str | None = None,
        human_notes: str | None = None,
        auto_continue: bool | None = None,
        model_parameters: dict[str, Any] | None = None,
        retry_failed: bool = False,
    ) -> ExperimentProtocol:
        task = self.repository.load_task()
        candidate_set = self.repository.load_candidate_experiments()
        candidate = self._select_candidate(
            candidate_set,
            candidate_id,
            allow_failed_retry=retry_failed,
        )
        self._repair_candidate_hypothesis_links(candidate)
        try:
            tree_for_display = self.repository.load_hypothesis_tree()
        except Exception:
            tree_for_display = None
        try:
            uncertainty_records = self.repository.load_uncertainties().records
        except Exception:
            uncertainty_records = []
        protocol_refinements = self._build_protocol_refinements(
            task=task,
            candidate=candidate,
            source_round=candidate_set.round,
        )
        protocol = self.mapper.to_protocol(
            task=task,
            candidate=candidate,
            round_id=candidate_set.round,
            model_parameters=model_parameters,
            protocol_refinements=protocol_refinements,
            tree=tree_for_display,
            uncertainty_records=uncertainty_records,
        )
        self._repair_protocol_hypothesis_links(protocol)
        plan_summary = self._build_execution_plan_summary(
            task=task,
            candidate=candidate,
            protocol=protocol,
            protocol_refinements=protocol_refinements,
        )
        protocol.notes.append(
            "execution_plan_summary:"
            + clean_llm_text(
                plan_summary,
                tree=tree_for_display,
                uncertainty_records=uncertainty_records,
            )
        )
        tuning_entries = [
            {
                "refinement_type": item.refinement_type,
                "rationale": item.rationale,
                "model_parameters": item.suggested_model_parameters,
                "protocol_notes": item.protocol_notes,
            }
            for item in (protocol_refinements or [])
        ]
        if model_parameters:
            tuning_entries.append(
                {
                    "refinement_type": "human_pi_override",
                    "rationale": "PI 在审批环节显式覆写模型参数",
                    "model_parameters": model_parameters,
                    "protocol_notes": [],
                }
            )
        tuning_narrative = TuningNarrativeService().polish(
            candidate_id=candidate.experiment_id,
            scientific_objective=protocol.scientific_objective,
            tuning_entries=tuning_entries,
            plan_summary=plan_summary,
            history_feedback=self._build_history_context(
                candidate,
                source_round=candidate_set.round,
            ),
        )["narrative"]
        self._save_protocol(protocol)

        process_state = self.repository.load_process_state()
        process_state = self._mark_protocol_generated(process_state, auto_continue=auto_continue)
        self.repository.save_process_state(process_state)

        decision_log = self.repository.load_decision_log()
        decision_log.decisions.append(
            DecisionEntry(
                decision_id=self._next_decision_id(decision_log),
                timestamp=datetime.now(),
                round_id=protocol.round_id,
                phase="experiment_planning",
                step="experiment_selection",
                decision_type="experiment_approved",
                made_by="human_pi",
                summary=f"用户批准实验 {candidate.experiment_id}",
                details={
                    "candidate_id": candidate.experiment_id,
                    "utility_score": candidate.utility_score,
                    "notes": human_notes,
                    "auto_continue": auto_continue,
                    "plan_summary": plan_summary,
                },
            )
        )
        decision_log.decisions.append(
            DecisionEntry(
                decision_id=self._next_decision_id(decision_log),
                timestamp=datetime.now(),
                round_id=protocol.round_id,
                phase="experiment_planning",
                step="protocol_generation",
                decision_type="protocol_generated",
                made_by="system",
                summary=f"已为 {candidate.experiment_id} 生成执行协议",
                details={
                    "protocol_path": self.repository.relativize(self.protocol_path),
                    "candidate_id": candidate.experiment_id,
                    "scientific_objective": protocol.scientific_objective,
                    "refinement_types": [item.refinement_type for item in protocol_refinements or []],
                    "plan_summary": plan_summary,
                    "model_parameters": protocol.model.parameters,
                    "tuning_entries": tuning_entries,
                    "tuning_narrative": tuning_narrative,
                    "protocol_notes": protocol.notes,
                },
            )
        )
        self.repository.save_decision_log(decision_log)
        return protocol

    def reject_candidate(
        self,
        *,
        candidate_id: str | None = None,
        reason: str,
    ) -> dict[str, Any]:
        candidate_set = self.repository.load_candidate_experiments()
        candidate = self._select_candidate(candidate_set, candidate_id)
        candidate_set.candidates = [
            item for item in candidate_set.candidates if item.experiment_id != candidate.experiment_id
        ]
        candidate_set.rejected_candidates.append(
            RejectedCandidate(
                experiment_id=candidate.experiment_id,
                reason=reason,
                similar_to=candidate.experiment_id,
            )
        )
        candidate_set.current_count = len(candidate_set.candidates)
        if candidate_set.current_count < candidate_set.minimum_required:
            candidate_set.note = "候选实验被人工拒绝后数量不足，建议重新生成或补充候选实验。"
        self.repository.save_candidate_experiments(candidate_set)

        process_state = self.repository.load_process_state()
        process_state = self._mark_reselection_waiting(process_state)
        self.repository.save_process_state(process_state)

        decision_log = self.repository.load_decision_log()
        decision_log.decisions.append(
            DecisionEntry(
                decision_id=self._next_decision_id(decision_log),
                timestamp=datetime.now(),
                round_id=candidate_set.round,
                phase="experiment_planning",
                step="experiment_selection",
                decision_type="experiment_rejected",
                made_by="human_pi",
                summary=f"用户拒绝实验 {candidate.experiment_id}",
                details={"reason": reason},
            )
        )
        self.repository.save_decision_log(decision_log)
        return {
            "status": "rejected",
            "candidate_id": candidate.experiment_id,
            "reason": reason,
            "remaining_candidates": candidate_set.current_count,
        }

    def approve_and_execute_candidate(
        self,
        *,
        candidate_id: str | None = None,
        human_notes: str | None = None,
        auto_continue: bool | None = None,
        model_parameters: dict[str, Any] | None = None,
        remaining_uncertainties: list[dict[str, Any]] | None = None,
        retry_failed: bool = False,
    ) -> dict[str, Any]:
        protocol = self.approve_candidate(
            candidate_id=candidate_id,
            human_notes=human_notes,
            auto_continue=auto_continue,
            model_parameters=model_parameters,
            retry_failed=retry_failed,
        )
        process_state = self.repository.load_process_state()
        process_state.current_round = max(process_state.current_round, protocol.round_id)
        process_state.current_phase = "experiment_execution"
        process_state.current_step = "dispatch_execution"
        process_state.current_stage = "running_experiment"
        process_state.progress_percentage = 62
        process_state.phases["experiment_execution"] = ProcessPhase(
            status="in_progress",
            started_at=datetime.now(),
            steps={
                "dispatch_execution": ProcessStep(name="下发实验任务", status="completed"),
                "core_execution": ProcessStep(name="执行实验与训练", status="in_progress"),
                "result_collection": ProcessStep(name="收集结果产物", status="pending"),
            },
            notes="approved experiment is being executed by the harness",
        )
        self.repository.save_process_state(process_state)

        task = self.repository.load_task()
        harness = UnifiedExperimentHarness(
            data_source=self._build_data_source_config(task, protocol=protocol),
            project_root=self.project_root,
            run_shap=self.run_shap,
        )
        try:
            result = ExperimentDispatcher(harness).execute(protocol)
        except Exception as exc:
            return self._record_execution_failure(protocol=protocol, error=exc)
        if result.status != "completed" or result.validity_status in {"invalid", "pending_scientific_review"}:
            message = result.warnings[0] if result.warnings else (
                result.execution.message or f"实验未完成：{result.validity_status}"
            )
            return self._record_execution_failure(
                protocol=protocol,
                error=RuntimeError(message),
                result=result,
            )
        return self._run_result_analysis(
            protocol=protocol,
            result=result,
            remaining_uncertainties=remaining_uncertainties,
        )

    def _record_execution_failure(
        self,
        *,
        protocol: ExperimentProtocol,
        error: Exception,
        result: ExperimentResult | None = None,
    ) -> dict[str, Any]:
        updater = UnifiedStateUpdater(self.repository)
        updater.apply_execution_failure(
            protocol=protocol,
            error=error,
            protocol_path=self.repository.relativize(self.protocol_path),
        )
        process_state = self.repository.load_process_state()
        process_state.current_phase = "experiment_execution"
        process_state.current_step = "execution_failed"
        process_state.current_stage = "failed"
        process_state.phases["experiment_execution"] = ProcessPhase(
            status="failed",
            completed_at=datetime.now(),
            steps={
                "dispatch_execution": ProcessStep(name="下发实验任务", status="completed"),
                "core_execution": ProcessStep(name="执行实验与训练", status="failed"),
                "result_collection": ProcessStep(name="收集结果产物", status="skipped"),
            },
            notes=f"execution failed: {error}",
        )
        process_state.progress_percentage = max(process_state.progress_percentage or 0, 62)
        self.repository.save_process_state(process_state)
        self.request_round_review(
            round_id=protocol.round_id,
            experiment_id=protocol.experiment_id,
            evaluation_summary={
                "experiment_id": protocol.experiment_id,
                "round_id": protocol.round_id,
                "stable": False,
                "key_findings": [f"实验失败：{error}"],
            },
            hypothesis_assessments=[],
            disagreement_updates=[],
            reasoning_traces=[],
            failure_summary={
                "message": str(error),
                "error_type": error.__class__.__name__,
                "can_continue": True,
            },
        )
        return {
            "protocol": protocol,
            "result": result,
            "evaluation": None,
            "review": {
                "status": "awaiting_round_decision",
                "round_id": protocol.round_id,
                "experiment_id": protocol.experiment_id,
            },
            "failure": str(error),
        }

    def retry_failed_experiment(
        self,
        *,
        candidate_id: str | None = None,
        human_notes: str | None = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Re-execute one failed experiment from the round-decision stage."""
        process_state = self.repository.load_process_state()
        if process_state.current_stage != "awaiting_round_decision":
            raise ValueError("当前未处于整轮反馈阶段，无法重试失败实验。")
        candidate_set = self.repository.load_candidate_experiments()
        target_candidate = (
            candidate_set.top_candidate()
            if candidate_id is None
            else next(
                (item for item in candidate_set.candidates if item.experiment_id == candidate_id),
                None,
            )
        )
        if target_candidate is None:
            raise KeyError(f"找不到可重试的候选实验: {candidate_id}")
        failure_history = self.repository.load_failure_history()
        recoverable = any(
            item.experiment_id == target_candidate.experiment_id and item.can_continue
            for item in failure_history.records
        )
        if not recoverable:
            raise ValueError(
                f"候选实验 {target_candidate.experiment_id} 没有可重试的失败记录，"
                "不能重复执行已成功或已终止的实验。"
            )
        return self.approve_and_execute_candidate(
            candidate_id=target_candidate.experiment_id,
            human_notes=human_notes,
            auto_continue=False,
            model_parameters=model_parameters,
            retry_failed=True,
        )

    def _run_result_analysis(
        self,
        *,
        protocol: ExperimentProtocol,
        result: ExperimentResult,
        remaining_uncertainties: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        process_state = self.repository.load_process_state()
        execution_phase = process_state.phases.get("experiment_execution")
        result_analysis_phase = process_state.phases.get("result_analysis")
        process_state.current_phase = "result_analysis"
        process_state.current_step = "evaluation"
        process_state.current_stage = "evaluating_results"
        process_state.progress_percentage = 72
        process_state.phases["experiment_execution"] = ProcessPhase(
            status="completed",
            completed_at=(
                execution_phase.completed_at
                if execution_phase is not None and execution_phase.completed_at is not None
                else datetime.now()
            ),
            steps={
                "dispatch_execution": ProcessStep(name="下发实验任务", status="completed"),
                "core_execution": ProcessStep(name="执行实验与训练", status="completed"),
                "result_collection": ProcessStep(name="收集结果产物", status="completed"),
            },
        )
        process_state.phases["result_analysis"] = ProcessPhase(
            status="in_progress",
            started_at=(
                result_analysis_phase.started_at
                if result_analysis_phase is not None and result_analysis_phase.started_at is not None
                else datetime.now()
            ),
            steps={
                "evaluation": ProcessStep(name="计算指标与稳健性分析", status="in_progress"),
                "scientific_interpretation": ProcessStep(name="生成科学解释", status="pending"),
            },
        )
        self.repository.save_process_state(process_state)

        unresolved = remaining_uncertainties or self._remaining_uncertainties_for_review(protocol)
        task = self.repository.load_task()
        main_question = task.payload.research_question.text or (
            "宇宙线日影南北偏移是否可提前改善太阳风速度预测？"
        )
        tree = self.repository.load_hypothesis_tree()
        prior_supports = {
            node.hypothesis_id: node.support_score
            for node in tree.nodes
            if node.hypothesis_id in protocol.tested_hypotheses
        }
        evaluation = evaluate_experiment_result(
            result,
            protocol=protocol,
            remaining_uncertainties=unresolved,
            main_question=main_question,
            prior_supports=prior_supports,
        )
        pending_review = (
            protocol.experiment_type != "incremental_prediction"
            or result.validity_status == "pending_scientific_review"
        )
        interpreter_analysis = None
        if not pending_review:
            interpreter_analysis = self._build_interpreter_analysis(
                protocol=protocol,
                evaluation=evaluation,
            )
        self._apply_interpreter_analysis_to_conclusion(
            evaluation,
            interpreter_analysis,
            protocol=protocol,
        )
        round_dir = self.project_root / "results" / f"round_{protocol.round_id:02d}"
        evaluation_path = round_dir / "evaluation_unified.json"
        export_evaluation_result(evaluation, evaluation_path)

        updater = UnifiedStateUpdater(self.repository)
        updater.apply_evaluation(
            protocol=protocol,
            result=result,
            evaluation=evaluation,
            interpretation_enhancements=(
                interpreter_analysis.enhancements if interpreter_analysis is not None else []
            ),
            protocol_path=self.repository.relativize(self.protocol_path),
            result_path=f"results/round_{protocol.round_id:02d}/result_unified.json",
            evaluation_path=self.repository.relativize(evaluation_path),
        )
        self._log_scientific_interpreter_applied(
            round_id=protocol.round_id,
            experiment_id=protocol.experiment_id,
            interpretation_enhancements=(
                interpreter_analysis.enhancements if interpreter_analysis is not None else []
            ),
        )

        process_state = self.repository.load_process_state()
        analysis_phase = process_state.phases.get("result_analysis")
        if analysis_phase is not None:
            analysis_phase.status = "completed"
            analysis_phase.completed_at = datetime.now()
            evaluation_step = analysis_phase.steps.get("evaluation")
            if evaluation_step is not None:
                evaluation_step.status = "completed"
            interpretation_step = analysis_phase.steps.get("scientific_interpretation")
            if interpretation_step is not None:
                interpretation_step.status = "completed"
            self.repository.save_process_state(process_state)

        review_payload = self.request_round_review(
            round_id=protocol.round_id,
            experiment_id=protocol.experiment_id,
            evaluation_summary=self._build_evaluation_summary(evaluation),
            hypothesis_assessments=[
                item.model_dump(mode="json", exclude_none=True)
                for item in evaluation.scientific.hypothesis_assessments
            ],
            disagreement_updates=[
                item.model_dump(mode="json", exclude_none=True)
                for item in evaluation.scientific.disagreement_updates
            ],
            reasoning_traces=self._recent_reasoning_traces_for_experiment(protocol.experiment_id),
        )
        return {
            "protocol": protocol,
            "result": result,
            "evaluation": evaluation,
            "review": review_payload,
        }

    def resume_evaluation_from_persisted_run(
        self,
        *,
        round_id: int | None = None,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        """Resume result analysis from already persisted protocol/result artifacts."""
        process_state = self.repository.load_process_state()
        if process_state.current_stage == "awaiting_round_decision":
            return {"status": "already_awaits_review", "round_id": process_state.current_round}
        if process_state.current_phase not in {"result_analysis", "experiment_execution"}:
            raise ValueError("当前状态不需要恢复结果分析，不能调用恢复接口。")

        target_round = round_id or process_state.current_round
        round_dir = self.project_root / "results" / f"round_{target_round:02d}"
        protocol_path = round_dir / "protocol.json"
        result_path = round_dir / "result_unified.json"
        evaluation_path = round_dir / "evaluation_unified.json"
        if not protocol_path.exists() or not result_path.exists():
            raise ValueError("未找到已完成的实验协议与结果，无法恢复分析；请不要重复批准实验。")

        protocol = ExperimentProtocol.model_validate(json.loads(protocol_path.read_text(encoding="utf-8")))
        result = ExperimentResult.model_validate(json.loads(result_path.read_text(encoding="utf-8")))
        if experiment_id is not None and protocol.experiment_id != experiment_id:
            raise ValueError(
                f"指定实验 {experiment_id} 与第 {target_round} 轮已落盘协议 {protocol.experiment_id} 不一致。"
            )
        if result.status != "completed":
            raise ValueError("已落盘实验结果不是 completed 状态，不能恢复结果分析。")
        if evaluation_path.exists():
            evaluation = EvaluationResult.model_validate(json.loads(evaluation_path.read_text(encoding="utf-8")))
            review_payload = self.request_round_review(
                round_id=protocol.round_id,
                experiment_id=protocol.experiment_id,
                evaluation_summary=self._build_evaluation_summary(evaluation),
                hypothesis_assessments=[
                    item.model_dump(mode="json", exclude_none=True)
                    for item in evaluation.scientific.hypothesis_assessments
                ],
                disagreement_updates=[
                    item.model_dump(mode="json", exclude_none=True)
                    for item in evaluation.scientific.disagreement_updates
                ],
                reasoning_traces=self._recent_reasoning_traces_for_experiment(protocol.experiment_id),
            )
            return {
                "status": "resumed_from_existing_evaluation",
                "protocol": protocol,
                "result": result,
                "evaluation": evaluation,
                "review": review_payload,
            }

        return self._run_result_analysis(protocol=protocol, result=result)

    def request_stop(self, *, phase: str, reason: str, pause: bool = True) -> None:
        process_state = self.repository.load_process_state()
        process_state.user_requests.stop_requested = True
        process_state.user_requests.stop_at_phase = phase
        process_state.user_requests.pause_reason = reason
        process_state.current_stage = "paused" if pause else "stopped"
        process_state.current_phase = phase
        process_state.current_step = "manual_control"
        self.repository.save_process_state(process_state)

        decision_log = self.repository.load_decision_log()
        decision_log.decisions.append(
            DecisionEntry(
                decision_id=self._next_decision_id(decision_log),
                timestamp=datetime.now(),
                round_id=process_state.current_round,
                phase=phase,
                step="manual_control",
                decision_type="pause_requested" if pause else "stop_requested",
                made_by="human_pi",
                summary="用户请求暂停实验" if pause else "用户请求停止实验",
                details={"reason": reason},
            )
        )
        decision_log.stop_history.append(
            StopHistoryEntry(
                stop_id=f"S{len(decision_log.stop_history) + 1:03d}",
                timestamp=datetime.now(),
                type="pause" if pause else "stop",
                phase=phase,
                reason=reason,
            )
        )
        self.repository.save_decision_log(decision_log)

    def request_round_review(
        self,
        *,
        round_id: int,
        experiment_id: str | None = None,
        evaluation_summary: dict[str, Any] | None = None,
        hypothesis_assessments: list[dict[str, Any]] | None = None,
        disagreement_updates: list[dict[str, Any]] | None = None,
        reasoning_traces: list[dict[str, Any]] | None = None,
        failure_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        process_state = self.repository.load_process_state()
        process_state.current_round = max(process_state.current_round, round_id)
        process_state.current_phase = "decision_making"
        process_state.current_step = "round_review"
        process_state.current_stage = "awaiting_round_decision"
        process_state.progress_percentage = 85
        process_state.phases["decision_making"] = ProcessPhase(
            status="in_progress",
            steps={
                "round_review": ProcessStep(
                    name="round_review",
                    status="pending",
                    requires_user_approval=True,
                    notes="等待 human_pi 决定是否进入下一轮。",
                )
            },
        )
        self.repository.save_process_state(process_state)

        closure_snapshot = self._build_round_closure_snapshot(round_id)

        decision_log = self.repository.load_decision_log()
        decision_log.decisions.append(
            DecisionEntry(
                decision_id=self._next_decision_id(decision_log),
                timestamp=datetime.now(),
                round_id=round_id,
                phase="decision_making",
                step="round_review",
                decision_type="round_review_requested",
                made_by="system",
                summary=f"系统请求 human_pi 对 round {round_id} 作出下一步决策",
                details={
                    "experiment_id": experiment_id,
                    "evaluation_summary": evaluation_summary or {},
                    "hypothesis_assessments": hypothesis_assessments or [],
                    "disagreement_updates": disagreement_updates or [],
                    "reasoning_traces": reasoning_traces or [],
                    "failure_summary": failure_summary or {},
                    "closure_checklist": closure_snapshot["closure_checklist"],
                    "gating_ready": closure_snapshot["gating_ready"],
                },
            )
        )
        self.repository.save_decision_log(decision_log)
        return {
            "status": "awaiting_round_decision",
            "round_id": round_id,
            "experiment_id": experiment_id,
            "options": ["进入下一轮", "进入下一轮并调整方向", "停止实验"],
            "evaluation_summary": evaluation_summary or {},
            "hypothesis_assessments": hypothesis_assessments or [],
            "disagreement_updates": disagreement_updates or [],
            "reasoning_traces": reasoning_traces or [],
            "failure_summary": failure_summary or {},
            "closure_checklist": closure_snapshot["closure_checklist"],
            "gating_ready": closure_snapshot["gating_ready"],
        }

    def record_round_decision(
        self,
        *,
        round_id: int,
        decision: str,
        summary: str | None = None,
        human_feedback: str | None = None,
        auto_continue: bool | None = None,
        data_dictionary: DataDictionary | None = None,
    ) -> dict[str, Any]:
        if (
            decision != "stop"
            and is_blind_sealed(self.project_root)
        ):
            raise ValueError(
                "final_test_blind 已封存：最终测试段的评估结果不得驱动下一轮设计，"
                "请使用 stop 结束工作流并仅把盲测结果作为交付报告。"
            )

        decision_type_map = {
            "continue": "continue_next_round",
            "adjust": "round_adjusted",
            "stop": "terminate_workflow",
        }
        if decision not in decision_type_map:
            raise ValueError("decision must be one of: continue, adjust, stop")

        process_state = self.repository.load_process_state()
        if process_state.current_stage != "awaiting_round_decision":
            raise ValueError("当前轮次尚未进入可提交整轮反馈的状态。")
        closure_snapshot = self._build_round_closure_snapshot(round_id)
        if not closure_snapshot["gating_ready"]:
            missing = next(
                (item.get("label") for item in closure_snapshot["closure_checklist"] if not item.get("completed")),
                "闭环阶段校验",
            )
            raise ValueError(f"本轮闭环尚未完成，仍缺少：{missing}")
        if auto_continue is not None:
            process_state.user_settings.auto_continue = auto_continue
        process_state.current_round = max(process_state.current_round, round_id)
        process_state.current_phase = "next_round" if decision != "stop" else "decision_making"
        process_state.current_step = "handoff" if decision != "stop" else "workflow_terminated"
        process_state.current_stage = "next_round_ready" if decision != "stop" else "workflow_terminated"
        process_state.progress_percentage = 100 if decision == "stop" else 90
        process_state.phases["decision_making"] = ProcessPhase(
            status="completed",
            completed_at=datetime.now(),
            steps={
                "round_review": ProcessStep(
                    name="round_review",
                    status="completed",
                    requires_user_approval=True,
                )
            },
        )
        process_state.phases["next_round"] = ProcessPhase(
            status="pending" if decision != "stop" else "skipped",
            notes="waiting for next-round planning" if decision != "stop" else "workflow terminated by human_pi",
        )
        if decision == "stop":
            process_state.user_requests.stop_requested = True
            process_state.user_requests.stop_at_phase = "decision_making"
            process_state.user_requests.pause_reason = (
                human_feedback or summary or "人工 PI 选择停止实验"
            )
        self.repository.save_process_state(process_state)

        decision_log = self.repository.load_decision_log()
        review_context = self._latest_round_review_context(decision_log, round_id)
        decision_log.decisions.append(
            DecisionEntry(
                decision_id=self._next_decision_id(decision_log),
                timestamp=datetime.now(),
                round_id=round_id,
                phase="decision_making",
                step="round_review",
                decision_type=decision_type_map[decision],
                made_by="human_pi",
                summary=summary or self._default_round_summary(round_id, decision),
                details={
                    "human_feedback": human_feedback,
                    "auto_continue": process_state.user_settings.auto_continue,
                },
            )
        )
        if human_feedback:
            decision_log.human_feedback.append(
                HumanFeedbackEntry(
                    feedback_id=f"F{len(decision_log.human_feedback) + 1:03d}",
                    timestamp=datetime.now(),
                    phase="decision_making",
                    round_number=round_id,
                    feedback_type="改进建议" if decision == "adjust" else "轮次备注",
                    content=human_feedback,
                    implemented_in_round=round_id + 1 if decision != "stop" else None,
                    status="recorded",
                    related_experiment_id=review_context.get("experiment_id"),
                    linked_reasoning_trace_ids=review_context.get("reasoning_trace_ids", []),
                    context_summary=review_context.get("context_summary"),
                )
            )
        if decision == "stop":
            decision_log.stop_history.append(
                StopHistoryEntry(
                    stop_id=f"S{len(decision_log.stop_history) + 1:03d}",
                    timestamp=datetime.now(),
                    type="stop",
                    phase="decision_making",
                    reason=human_feedback or summary or "人工 PI 选择停止实验",
                )
            )
        self.repository.save_decision_log(decision_log)
        self._persist_round_decision_history(
            round_id=round_id,
            decision=decision,
            human_feedback=human_feedback,
            summary=summary or self._default_round_summary(round_id, decision),
        )

        payload: dict[str, Any] = {
            "status": "recorded",
            "decision_type": decision_type_map[decision],
            "round_id": round_id,
        }
        if decision != "stop":
            payload.update(
                RoundOrchestrator(self).bootstrap_round(
                    round_id=round_id,
                    human_feedback=human_feedback if decision == "adjust" else None,
                    data_dictionary=data_dictionary,
                    stop_at_hypothesis_confirmation=True,
                )
            )
        return payload

    def _build_next_phase(
        self,
        process_state: ProcessState,
        round_id: int,
        human_feedback: str | None,
    ) -> ProcessPhase:
        return ProcessPhase(
            status="in_progress",
            started_at=datetime.now(),
            notes="planner input prepared for next-round reasoning/planning",
            steps={
                "planner_input_prepared": ProcessStep(
                    name="planner_input_prepared",
                    status="completed",
                    requires_user_approval=False,
                    notes=human_feedback or "reuse current planning policy",
                ),
                "candidate_generation": ProcessStep(
                    name="candidate_generation",
                    status="in_progress",
                    requires_user_approval=False,
                    notes=f"{self.planner_mode} candidate generation is consuming planner_input",
                ),
            },
        )

    def _save_protocol(self, protocol: ExperimentProtocol) -> None:
        self.protocol_path.write_text(
            json_dumps(protocol.model_dump(mode="json", exclude_none=True), indent=2),
            encoding="utf-8",
        )

    def _build_execution_plan_summary(
        self,
        *,
        task: ScientificTask,
        candidate: CandidateExperiment,
        protocol: ExperimentProtocol,
        protocol_refinements: list[ProtocolRefinementSuggestion] | None,
    ) -> str:
        semantic = self._semantic_service()
        refinement_labels = [item.refinement_type for item in protocol_refinements or []]
        control = "、".join(protocol.features.control) or "无"
        treatment = "、".join(protocol.features.treatment) or "无"
        target = protocol.target or task.payload.research_question.target or "目标变量"
        control_display = semantic.display_text(control)
        treatment_display = semantic.display_text(treatment)
        target_display = semantic.to_display(target)
        objective_display = semantic.display_text(protocol.scientific_objective or candidate.purpose)
        effective_experiment_type = str(candidate.experiment_type or candidate.type or protocol.experiment_type)
        is_baseline_experiment = (
            "experiment_mode:baseline_single_arm" in (protocol.notes or [])
            or effective_experiment_type == "baseline_benchmark"
        )
        analyzer = "四组运行（Ridge / ElasticNet × 对照组 / 实验组）"
        if effective_experiment_type == "model_sensitivity":
            execution_steps = "数据质量检查→Ridge 对照组/实验组→ElasticNet 对照组/实验组→家族内增益对比"
        elif is_baseline_experiment:
            execution_steps = "数据质量检查→单组对照训练→对照结果评估"
        else:
            execution_steps = "数据质量检查→对照实验自动执行→实验对照执行→结果对比"
        return (
            f"目标：{objective_display}；"
            f"预测目标：{target_display}；"
            f"实验类型：{analyzer if effective_experiment_type == 'model_sensitivity' else effective_experiment_type}；"
            f"对照组变量：{treatment_display if is_baseline_experiment else control_display}；"
            f"实验组变量：{'不设置实验组' if is_baseline_experiment else treatment_display}；"
            f"执行步骤：{execution_steps}；"
            f"变量输入来源：candidate.design 已完整写入 protocol.features 与 model.parameters.feature_groups(JSON)；"
            f"关联假设：{semantic.display_text('、'.join(protocol.tested_hypotheses[:3])) or '本轮以不确定性验证为主'}；"
            f"目标不确定性：{semantic.display_text('、'.join(protocol.target_uncertainties[:3])) or '待从本轮评价中补充'}；"
            f"规划增强：{ '、'.join(refinement_labels) if refinement_labels else '无额外 LLM 协议增强'}。"
        )

    def _build_round_closure_snapshot(self, round_id: int) -> dict[str, Any]:
        try:
            round_history = self.repository.load_round_history()
        except Exception:
            return {"gating_ready": False, "closure_checklist": []}

        entry = next((item for item in round_history.entries if item.round_id == round_id), None)
        if entry is None:
            return {"gating_ready": False, "closure_checklist": []}
        return {
            "gating_ready": bool(entry.gating_ready),
            "closure_checklist": [
                item.model_dump(mode="json", exclude_none=True) if hasattr(item, "model_dump") else item
                for item in entry.closure_checklist
            ],
        }

    def _persist_round_decision_history(
        self,
        *,
        round_id: int,
        decision: str,
        human_feedback: str | None,
        summary: str,
    ) -> None:
        try:
            history = self.repository.load_round_history()
        except Exception:
            return
        entry = next((item for item in history.entries if item.round_id == round_id), None)
        if entry is None:
            return
        entry.next_round_decision = decision
        entry.human_feedback = human_feedback
        entry.decision_summary = summary
        entry.updated_at = datetime.now()
        history.current_round = max(history.current_round, round_id)
        history.last_updated = entry.updated_at
        self.repository.save_round_history(history)

    def _build_data_source_config(
        self,
        task: ScientificTask,
        protocol: ExperimentProtocol | None = None,
    ) -> ElasticNetDataSourceConfig:
        data_sources = task.payload.data_sources
        omni = data_sources.get("omni")
        lhaaso = data_sources.get("lhaaso")
        if not omni or not lhaaso:
            raise KeyError("task.payload.data_sources 必须同时包含 omni 和 lhaaso 配置")

        data_dictionary = self._load_data_dictionary_or_default()
        time_column = omni.get("time_column") or lhaaso.get("time_column") or "TIME"
        target_column = omni.get("target_column") or task.payload.research_question.target
        omni_resolved = self._resolve_project_path(omni["path"]).resolve()
        lhaaso_resolved = self._resolve_project_path(lhaaso["path"]).resolve()
        extra_sources: dict[str, dict[str, str]] = {}
        for source_id, details in data_sources.items():
            if source_id in {"omni", "lhaaso"} or not details or not details.get("path"):
                continue
            source_path = self._resolve_project_path(details["path"])
            if source_path.resolve() in {omni_resolved, lhaaso_resolved}:
                # 同一物理文件在 task.json 中可能同时存在别名与规范名；
                # 别名不视为额外稀疏源，避免触发与 PFSS 同级的缺失日剔除。
                continue
            extra_sources[source_id] = details
        extra_paths = [self._resolve_project_path(details["path"]) for details in extra_sources.values()]
        column_aliases = {}
        for field in data_dictionary.fields:
            column_aliases[field.field_name] = field.field_name
            if field.physical_meaning and field.physical_meaning not in {"feature", "target", "time", "弃用字段"}:
                column_aliases[field.physical_meaning] = field.field_name

        # 只有当前协议真正使用了稀疏源特征时，才对该源启用缺失记录审计。
        sparse_sources: list[str] = []
        if protocol is not None:
            used_features = {
                str(item).strip()
                for item in (*protocol.features.control, *protocol.features.treatment)
                if str(item).strip()
            }
            extra_source_stems: dict[str, list[str]] = {}
            for source_id, details in extra_sources.items():
                extra_source_stems.setdefault(Path(details["path"]).name, []).append(Path(details["path"]).stem)
            for field in data_dictionary.fields:
                if field.field_name not in used_features:
                    continue
                for chunk in str(field.user_notes or "").split(";"):
                    if not chunk.startswith("sources="):
                        continue
                    for file_name in chunk.split("=", 1)[1].split(","):
                        for source_stem in extra_source_stems.get(file_name, []):
                            if source_stem not in sparse_sources:
                                sparse_sources.append(source_stem)
        return ElasticNetDataSourceConfig(
            omni_file_path=self._resolve_project_path(omni["path"]),
            lhaaso_file_path=self._resolve_project_path(lhaaso["path"]),
            extra_file_paths=extra_paths,
            time_column=time_column,
            target_column=target_column,
            column_aliases=column_aliases,
            sparse_sources=sparse_sources,
        )

    def _load_data_dictionary_or_default(self) -> DataDictionary:
        try:
            return self.repository.load_data_dictionary()
        except FileNotFoundError:
            return DataDictionary(
                dictionary_id="auto_resolved",
                generated_at=datetime.now(),
                version="1.0",
                dataset_name="project",
                total_samples=0,
                time_column="TIME",
                time_format="YYYYMMDD",
            )

    def _resolve_project_path(self, path_value: str) -> Path:
        candidate = Path(path_value)
        return candidate if candidate.is_absolute() else self.project_root / candidate

    def _remaining_uncertainties_for_review(self, protocol: ExperimentProtocol) -> list[dict[str, Any]]:
        state = self.repository.load_uncertainties()
        unresolved: list[dict[str, Any]] = []
        for record in state.records:
            if record.status in {"resolved", "deprecated"}:
                continue
            unresolved.append(
                {
                    "uncertainty_id": record.uncertainty_id,
                    "question": record.question,
                    "priority": record.priority,
                }
            )
            if len(unresolved) >= 3:
                break
        return unresolved

    @staticmethod
    def _build_evaluation_summary(evaluation: EvaluationResult) -> dict[str, Any]:
        metrics = evaluation.metrics
        return {
            "experiment_id": metrics.experiment_id,
            "round_id": metrics.round_id,
            "baseline_rmse": metrics.baseline_rmse,
            "treatment_rmse": metrics.treatment_rmse,
            "baseline_pearson_r": metrics.baseline_pearson_r,
            "treatment_pearson_r": metrics.treatment_pearson_r,
            "delta_pearson_r": metrics.delta.pearson_r,
            "delta_rmse": metrics.delta.rmse,
            "pg_actual_signed": metrics.pg_actual_signed,
            "pg_actual_clipped": metrics.pg_actual_clipped,
            "information_gain_kl": metrics.information_gain_kl,
            "ig_posterior_probs": metrics.ig_posterior_probs,
            "remaining_uncertainties": evaluation.scientific.evidence_summary.remaining_uncertainties,
            "key_findings": [
                item.claim for item in evaluation.scientific.evidence_summary.new_evidence_for
            ][:3],
            "robustness_recommendation": evaluation.robustness.recommendation,
            "stable": evaluation.robustness.overall.stable,
        }

    def _resume_next_round_planning(
        self,
        *,
        round_id: int,
        human_feedback: str | None,
        data_dictionary: DataDictionary | None,
    ) -> dict[str, Any]:
        if is_blind_sealed(self.project_root):
            raise ValueError(
                "final_test_blind 已封存：最终测试结果不得进入下一轮假设扩展、"
                "不确定性挖掘或候选实验规划。"
            )
        if data_dictionary is None:
            return {
                "planning_status": "awaiting_data_dictionary",
                "message": "需要提供 DataDictionary 后才能自动生成下一轮候选实验。",
            }

        task = self.repository.load_task()
        uncertainties = self.repository.load_uncertainties()
        uncertainties.current_round = max(uncertainties.current_round, round_id)
        self.repository.save_uncertainties(uncertainties)

        planner_input_builder = ReasoningPlannerInputBuilder(self.repository)
        planner_input = planner_input_builder.build(
            data_dictionary=data_dictionary,
            source_round_id=round_id,
            human_feedback=human_feedback,
        )
        if self.planner_mode == "llm":
            planner_input, uncertainties = self._augment_next_round_context_with_llm_services(
                planner_input=planner_input,
                uncertainties=uncertainties,
            )
        generation_result = self.hypothesis_generator.build_tree(
            task=task,
            data_dictionary=data_dictionary,
            uncertainties=uncertainties,
            planner_input=planner_input,
            existing_tree=self.repository.load_hypothesis_tree(),
            current_round=round_id + 1,
        )
        uncertainties.records = list(generation_result.updated_uncertainties)
        self.repository.save_uncertainties(uncertainties)
        if generation_result.generated_node_ids:
            self.repository.save_hypothesis_tree(generation_result.tree)
            self._log_hypothesis_generation(
                round_id=round_id,
                next_round_id=round_id + 1,
                generation_result=generation_result,
                mode=generation_result.mode,
            )
            planner_input = planner_input_builder.build(
                data_dictionary=data_dictionary,
                source_round_id=round_id,
                human_feedback=human_feedback,
            )
            if self.planner_mode == "llm":
                planner_input, uncertainties = self._augment_next_round_context_with_llm_services(
                    planner_input=planner_input,
                    uncertainties=uncertainties,
                )
                self.repository.save_uncertainties(uncertainties)
        self.repository.save_planner_input(planner_input)
        self._log_planner_input_prepared(planner_input)

        process_state = self.repository.load_process_state()
        process_state.current_round = max(process_state.current_round, round_id)
        process_state.current_phase = "next_round"
        process_state.current_step = "planner_input_prepared"
        process_state.current_stage = "planner_input_ready"
        process_state.progress_percentage = 92
        process_state.phases["next_round"] = self._build_next_phase(process_state, round_id, human_feedback)
        self.repository.save_process_state(process_state)

        plan = DecisionLayerService(
            self.repository,
            experiment_designer=self.experiment_designer,
            experiment_writer=self.experiment_writer,
        ).build_candidate_plan(
            task=task,
            data_dictionary=data_dictionary,
            planner_input=planner_input,
            target_round=planner_input.next_round_id,
        )
        planner_output = self.planner_output_builder.build(
            planner_input=planner_input,
            candidate_plan=plan,
        )
        self.repository.save_planner_output(planner_output)
        self._log_planner_output_generated(planner_output)
        applied_output = PlannerOutputApplier(self.repository).apply(planner_output)
        review = self.request_experiment_selection_review()
        process_state = self.repository.load_process_state()
        process_state.current_round = max(process_state.current_round, round_id + 1)
        process_state.current_phase = "experiment_planning"
        process_state.current_step = "step_4"
        process_state.current_stage = "awaiting_human_approval"
        process_state.progress_percentage = 45
        self.repository.save_process_state(process_state)
        return {
            "planning_status": "candidate_plan_rebuilt",
            "hypothesis_generation": {
                "mode": generation_result.mode,
                "generated_node_ids": list(generation_result.generated_node_ids),
            },
            "planner_input": planner_input,
            "planner_output": planner_output,
            "applied_output": applied_output,
            "candidate_plan": plan,
            "next_review": review,
        }

    def _select_candidate(
        self,
        candidate_set,
        candidate_id: str | None,
        *,
        allow_failed_retry: bool = False,
    ) -> CandidateExperiment:
        process_state = self.repository.load_process_state()
        if process_state.current_stage == "awaiting_round_decision" and not allow_failed_retry:
            raise ValueError(
                "当前轮次已进入整轮反馈阶段，不能继续审批旧轮候选实验；请先提交轮次决策。"
            )
        expected_round = max(process_state.current_round or 0, candidate_set.round or 0, 1)

        if candidate_id is None:
            top_candidate = candidate_set.top_candidate()
            if top_candidate is None:
                raise ValueError("当前没有可审批的候选实验。")
            if not top_candidate.experiment_id.startswith(f"E_R{expected_round:02d}_"):
                raise ValueError(
                    f"候选实验 {top_candidate.experiment_id} 不属于当前规划轮次 E_R{expected_round:02d}_*，"
                    "请先重新生成当前轮候选实验。"
                )
            return top_candidate

        for candidate in candidate_set.candidates:
            if candidate.experiment_id == candidate_id:
                if not candidate.experiment_id.startswith(f"E_R{expected_round:02d}_"):
                    raise ValueError(
                        f"候选实验 {candidate.experiment_id} 不属于当前规划轮次 E_R{expected_round:02d}_*，"
                        "禁止审批旧轮候选。"
                    )
                return candidate
        raise KeyError(f"找不到候选实验: {candidate_id}")

    def _mark_selection_waiting(self, process_state: ProcessState) -> ProcessState:
        now = datetime.now()
        process_state.current_phase = "experiment_planning"
        process_state.current_step = "step_4"
        process_state.current_stage = "awaiting_human_approval"
        process_state.progress_percentage = 45
        hypothesis_phase = process_state.phases.get(
            "hypothesis_generation",
            ProcessPhase(status="completed", completed_at=now),
        )
        hypothesis_phase.status = "completed"
        hypothesis_phase.completed_at = hypothesis_phase.completed_at or now
        hypothesis_steps = hypothesis_phase.steps or {}
        review_step = hypothesis_steps.get(
            "hypothesis_review",
            ProcessStep(name="假设树确认", status="completed", requires_user_approval=True),
        )
        review_step.status = "completed"
        review_step.notes = "假设树已由人工确认并冻结。"
        hypothesis_steps["hypothesis_generation"] = ProcessStep(
            name="假设生成",
            status="completed",
            requires_user_approval=False,
        )
        hypothesis_steps["hypothesis_review"] = review_step
        hypothesis_phase.steps = hypothesis_steps
        process_state.phases["hypothesis_generation"] = hypothesis_phase
        planning = process_state.phases.get(
            "experiment_planning",
            ProcessPhase(status="in_progress", started_at=now),
        )
        planning.status = "in_progress"
        planning.started_at = planning.started_at or now
        planning.steps = {
            "step_1": ProcessStep(name="识别关键不确定性", status="completed"),
            "step_2": ProcessStep(name="生成候选实验", status="completed"),
            "step_3": ProcessStep(name="计算综合价值", status="completed"),
            "step_4": ProcessStep(name="选择E*", status="pending", requires_user_approval=True),
            "step_5": ProcessStep(name="生成实验协议", status="pending"),
        }
        process_state.phases["experiment_planning"] = planning
        return process_state

    def _mark_hypothesis_confirmation_waiting(self, process_state: ProcessState) -> ProcessState:
        now = datetime.now()
        process_state.current_phase = "hypothesis_generation"
        process_state.current_step = "hypothesis_review"
        process_state.current_stage = "awaiting_hypothesis_confirmation"
        process_state.progress_percentage = 30
        hypothesis_phase = process_state.phases.get(
            "hypothesis_generation",
            ProcessPhase(status="in_progress", started_at=now),
        )
        hypothesis_phase.status = "in_progress"
        hypothesis_phase.started_at = hypothesis_phase.started_at or now
        hypothesis_phase.steps = {
            "hypothesis_generation": ProcessStep(
                name="假设生成",
                status="completed",
                requires_user_approval=False,
            ),
            "hypothesis_review": ProcessStep(
                name="假设树确认",
                status="in_progress",
                requires_user_approval=True,
                notes="H/C 生成完成，等待人工确认后进入不确定性识别。",
            ),
        }
        process_state.phases["hypothesis_generation"] = hypothesis_phase
        return process_state

    def _mark_reselection_waiting(self, process_state: ProcessState) -> ProcessState:
        process_state = self._mark_selection_waiting(process_state)
        process_state.current_stage = "candidate_rejected"
        planning = process_state.phases["experiment_planning"]
        planning.steps["step_4"].notes = "上一候选已被 human_pi 拒绝，等待重新选择。"
        process_state.phases["experiment_planning"] = planning
        return process_state

    def _mark_protocol_generated(self, process_state: ProcessState, auto_continue: bool | None) -> ProcessState:
        now = datetime.now()
        process_state.current_phase = "experiment_planning"
        process_state.current_step = "step_5"
        process_state.current_stage = "protocol_ready"
        process_state.progress_percentage = 55
        if auto_continue is not None:
            process_state.user_settings.auto_continue = auto_continue
        planning = process_state.phases.get(
            "experiment_planning",
            ProcessPhase(status="in_progress", started_at=now),
        )
        planning.status = "completed" if process_state.user_settings.auto_continue else "in_progress"
        planning.steps = {
            "step_1": ProcessStep(name="识别关键不确定性", status="completed"),
            "step_2": ProcessStep(name="生成候选实验", status="completed"),
            "step_3": ProcessStep(name="计算综合价值", status="completed"),
            "step_4": ProcessStep(name="选择E*", status="completed", requires_user_approval=True),
            "step_5": ProcessStep(name="生成实验协议", status="completed"),
        }
        planning.completed_at = now if planning.status == "completed" else None
        process_state.phases["experiment_planning"] = planning
        process_state.phases["experiment_execution"] = ProcessPhase(
            status="pending",
            steps={},
            notes="waiting for harness execution",
        )
        return process_state

    @staticmethod
    def _next_decision_id(log: DecisionLog) -> str:
        return f"D{len(log.decisions) + 1:03d}"

    @staticmethod
    def _default_round_summary(round_id: int, decision: str) -> str:
        mapping = {
            "continue": f"用户批准 round {round_id} 进入下一轮",
            "adjust": f"用户批准 round {round_id} 进入下一轮，但要求调整后续方向",
            "stop": f"用户在 round {round_id} 结束后停止工作流",
        }
        return mapping[decision]

    def _persist_llm_gateway_stats(self) -> None:
        snapshot = {
            "captured_at": datetime.now().isoformat(),
            **LLMGateway.statistics_summary(),
        }
        stats_path = self.config_dir / "llm_gateway_stats.json"
        stats_path.write_text(
            json_dumps(snapshot, indent=2),
            encoding="utf-8",
        )

    def _log_planner_input_prepared(self, planner_input: ReasoningPlannerInput) -> None:
        self._persist_llm_gateway_stats()
        decision_log = self.repository.load_decision_log()
        decision_log.decisions.append(
            DecisionEntry(
                decision_id=self._next_decision_id(decision_log),
                timestamp=datetime.now(),
                round_id=planner_input.source_round_id,
                phase="next_round",
                step="planner_input_prepared",
                decision_type="planner_input_prepared",
                made_by="system",
                summary=f"已为 round {planner_input.next_round_id} 生成统一 planner input",
                details={
                    "planner_input_path": self.repository.relativize(self.repository.paths.planner_input),
                    "source_experiment_id": planner_input.source_experiment_id,
                    "human_feedback": planner_input.human_feedback,
                    "planner_guidance": planner_input.planner_guidance[:8],
                    "llm_gateway_stats": LLMGateway.statistics_summary(),
                },
            )
        )
        self.repository.save_decision_log(decision_log)

    def _log_planner_output_generated(self, planner_output: ReasoningPlannerOutput) -> None:
        self._persist_llm_gateway_stats()
        decision_log = self.repository.load_decision_log()
        decision_log.decisions.append(
            DecisionEntry(
                decision_id=self._next_decision_id(decision_log),
                timestamp=datetime.now(),
                round_id=planner_output.source_round_id,
                phase="next_round",
                step="planner_output_generated",
                decision_type="planner_output_generated",
                made_by="system",
                summary=f"已为 round {planner_output.next_round_id} 生成统一 planner output",
                details={
                    "planner_output_path": self.repository.relativize(self.repository.paths.planner_output),
                    "planner_mode": planner_output.planner_mode,
                    "candidate_supplements": len(planner_output.candidate_supplements),
                    "protocol_refinements": len(planner_output.protocol_refinements),
                    "llm_gateway_stats": LLMGateway.statistics_summary(),
                },
            )
        )
        self.repository.save_decision_log(decision_log)

    def _log_scientific_interpreter_applied(
        self,
        *,
        round_id: int,
        experiment_id: str,
        interpretation_enhancements: list,
    ) -> None:
        self._persist_llm_gateway_stats()
        if not interpretation_enhancements:
            return
        decision_log = self.repository.load_decision_log()
        decision_log.decisions.append(
            DecisionEntry(
                decision_id=self._next_decision_id(decision_log),
                timestamp=datetime.now(),
                round_id=round_id,
                phase="result_analysis",
                step="scientific_interpreter",
                decision_type="scientific_interpreter_applied",
                made_by="system",
                summary=f"科学解释者已为 {experiment_id} 生成解释增强",
                details={
                    "experiment_id": experiment_id,
                    "enhancement_count": len(interpretation_enhancements),
                    "enhancement_ids": [item.enhancement_id for item in interpretation_enhancements],
                    "llm_gateway_stats": LLMGateway.statistics_summary(),
                },
            )
        )
        self.repository.save_decision_log(decision_log)

    def _build_interpreter_analysis(
        self,
        *,
        protocol: ExperimentProtocol,
        evaluation: EvaluationResult,
    ) -> InterpreterRoundAnalysis | None:
        if self.planner_mode != "llm":
            return None
        planner_input = self._build_interpreter_input(protocol=protocol, evaluation=evaluation)
        candidate_plan = self.repository.load_candidate_experiments()
        return self.scientific_interpreter.build_round_analysis(
            planner_input=planner_input,
            candidate_plan=candidate_plan,
            evaluation=evaluation,
        )

    @staticmethod
    def _apply_interpreter_analysis_to_conclusion(
        evaluation: EvaluationResult,
        analysis: InterpreterRoundAnalysis | None,
        *,
        protocol: ExperimentProtocol | None = None,
    ) -> None:
        if analysis is None or evaluation.scientific.three_layer_conclusion is None:
            return
        conclusion = evaluation.scientific.three_layer_conclusion
        assessment_index = {
            item.hypothesis_id: item
            for item in evaluation.scientific.hypothesis_assessments
        }
        tested_hypotheses = list(protocol.tested_hypotheses) if protocol is not None else []
        decisive = any(
            assessment_index.get(hypothesis_id) is not None
            and assessment_index[hypothesis_id].evidence_classification
            in {
                EVIDENCE_STABLE_IMPROVEMENT,
                EVIDENCE_STABLE_DEGRADATION,
            }
            for hypothesis_id in tested_hypotheses
        )
        if decisive and analysis.data_layer is not None:
            conclusion.data_layer = analysis.data_layer
            conclusion.data_layer.interpretation_source = "llm_interpreter"
        if decisive and analysis.conclusion_text:
            conclusion.scientific_layer.answer = analysis.conclusion_text
        if decisive and analysis.path_answer:
            conclusion.scientific_layer.path_answer = analysis.path_answer
        if (
            decisive
            and (analysis.conclusion_text or analysis.path_answer)
        ):
            conclusion.scientific_layer.interpretation_source = "llm_interpreter"
        if analysis.hypothesis_layer_summary:
            conclusion.hypothesis_layer_summary = analysis.hypothesis_layer_summary
        if analysis.overall_summary:
            conclusion.overall_summary = analysis.overall_summary
        if analysis.next_round_suggestion is not None:
            conclusion.next_round_suggestion = analysis.next_round_suggestion

    def _build_interpreter_input(
        self,
        *,
        protocol: ExperimentProtocol,
        evaluation: EvaluationResult,
    ) -> ReasoningPlannerInput:
        task = self.repository.load_task()
        tree = self.repository.load_hypothesis_tree()
        uncertainties = self.repository.load_uncertainties()
        time_column = next(
            (
                config.get("time_column")
                for config in task.payload.data_sources.values()
                if isinstance(config, dict) and config.get("time_column")
            ),
            "TIME",
        )
        return ReasoningPlannerInput(
            task_id=task.task_id,
            source_round_id=protocol.round_id,
            next_round_id=protocol.round_id,
            source_experiment_id=protocol.experiment_id,
            scientific_question=task.payload.research_question.text,
            target=protocol.target,
            evaluation_summary=self._build_runtime_evaluation_summary(protocol, evaluation),
            unresolved_uncertainties=[
                PlannerUncertaintyItem(
                    uncertainty_id=record.uncertainty_id,
                    question=record.question,
                    priority=record.priority,
                    priority_score=None,
                    status=record.status,
                    resolution_status=record.resolution_status,
                    related_hypotheses=record.related_hypotheses,
                )
                for record in uncertainties.records
                if record.status not in {"resolved", "deprecated"}
            ][:5],
            active_hypotheses=[
                HypothesisSnapshot(
                    hypothesis_id=node.hypothesis_id,
                    statement=node.statement,
                    status=node.status,
                    support_score=node.support_score,
                )
                for node in sorted(
                    [
                        item
                        for item in tree.nodes
                        if item.status in {"active", "observing", "converged"}
                    ],
                    key=lambda item: item.support_score,
                    reverse=True,
                )[:5]
            ],
            recent_hypothesis_assessments=evaluation.scientific.hypothesis_assessments,
            recent_disagreement_updates=[
                PlannerDisagreementUpdateItem(
                    uncertainty_id=item.uncertainty_id,
                    compared_hypotheses=item.compared_hypotheses,
                    narrowed=item.narrowed,
                    resolution_status=item.resolution_status,
                    leading_hypothesis_id=item.leading_hypothesis_id,
                    unresolved_hypotheses=item.unresolved_hypotheses,
                    support_span_before=item.support_span_before,
                    support_span_after=item.support_span_after,
                    summary=item.summary,
                )
                for item in evaluation.scientific.disagreement_updates
            ],
            recent_reasoning_traces=[],
            recent_human_feedback=[],
            data_dictionary_summary=self._runtime_dictionary_summary(
                task=task,
                time_column=time_column,
                protocol=protocol,
            ),
            planning_constraints={
                "no_future_information": task.payload.constraints.no_future_information,
                "validation_feedback_allowed": task.payload.constraints.validation_feedback_allowed,
                "final_test_blind": task.payload.constraints.final_test_blind,
            },
            planner_guidance=[
                f"robustness:{evaluation.robustness.recommendation}",
                *[
                    item.get("question") or item.get("description")
                    for item in evaluation.scientific.evidence_summary.remaining_uncertainties[:2]
                    if isinstance(item, dict)
                ],
            ],
        )

    @staticmethod
    def _build_runtime_evaluation_summary(
        protocol: ExperimentProtocol,
        evaluation: EvaluationResult,
    ) -> PlannerEvaluationSummary:
        return PlannerEvaluationSummary(
            experiment_id=protocol.experiment_id,
            round_id=protocol.round_id,
            baseline_rmse=evaluation.metrics.baseline_rmse,
            treatment_rmse=evaluation.metrics.treatment_rmse,
            baseline_pearson_r=evaluation.metrics.baseline_pearson_r,
            treatment_pearson_r=evaluation.metrics.treatment_pearson_r,
            delta_pearson_r=evaluation.metrics.delta.pearson_r,
            delta_rmse=evaluation.metrics.delta.rmse,
            pg_actual_signed=evaluation.metrics.pg_actual_signed,
            pg_actual_clipped=evaluation.metrics.pg_actual_clipped,
            key_findings=(
                [item.claim for item in evaluation.scientific.evidence_summary.new_evidence_for]
                + [item.claim for item in evaluation.scientific.evidence_summary.new_evidence_against]
            )[:3],
            robustness_recommendation=evaluation.robustness.recommendation,
            stable=evaluation.robustness.overall.stable,
        )

    def _runtime_dictionary_summary(
        self,
        *,
        task: ScientificTask,
        time_column: str,
        protocol: ExperimentProtocol,
    ) -> DataDictionarySummary:
        base_summary = DataDictionarySummary(
            dictionary_id="runtime_inferred",
            dataset_name=task.task_id,
            time_column=time_column,
            target_candidates=[protocol.target],
            feature_candidates=list(dict.fromkeys(protocol.features.control + protocol.features.treatment)),
        )
        try:
            data_dictionary = self.repository.load_data_dictionary()
            if not data_dictionary:
                return base_summary
            semantic = VariableSemanticService.from_data_dictionary(data_dictionary)
            return semantic.build_display_summary(base_summary)
        except Exception:
            return base_summary

    def _augment_next_round_context_with_llm_services(
        self,
        *,
        planner_input: ReasoningPlannerInput,
        uncertainties,
    ) -> tuple[ReasoningPlannerInput, object]:
        """Combined phase for legacy callers: H/C proposals plus uncertainty generation."""
        planner_input = self._augment_hypothesis_context_with_llm_services(
            planner_input=planner_input,
            uncertainties=uncertainties,
        )
        planner_input, uncertainties = self._augment_uncertainty_context_with_llm_services(
            planner_input=planner_input,
            uncertainties=uncertainties,
        )
        return planner_input, uncertainties

    def _augment_hypothesis_context_with_llm_services(
        self,
        *,
        planner_input: ReasoningPlannerInput,
        uncertainties,
    ) -> ReasoningPlannerInput:
        """Phase 1 LLM pass: only generate H/C proposals, then stop for human confirmation."""
        miner = MultiSourceUncertaintyMiner(self.repository)
        mining_context = miner._build_context(
            planner_input=planner_input,
            uncertainties=uncertainties,
            round_id=planner_input.next_round_id,
        )

        mining_result = miner.mine(context=mining_context)
        mined_candidates = [candidate.to_prompt_dict() for candidate in mining_result.candidates]
        rag_context = self.rag_service.build_context_bundle(planner_input)
        planner_input.planner_guidance.extend(
            note for note in rag_context.guidance_notes() if note not in planner_input.planner_guidance
        )
        planner_input.recent_reasoning_traces.extend(
            self._rag_context_to_traces(rag_context, planner_input.source_round_id)
        )

        proposer_output = self.hypothesis_proposer.propose(
            planner_input=planner_input,
            rag_context=rag_context,
        )
        semantic = VariableSemanticService.from_summary(planner_input.data_dictionary_summary)
        focus_features_raw = semantic.raw_list(proposer_output.focus_features)
        explicit_focus_notes = [
            f"proposer_focus:{feature}"
            for feature in focus_features_raw[:6]
            if feature in planner_input.data_dictionary_summary.feature_candidates
        ]
        display_proposals: list[dict[str, object]] = []
        for item in proposer_output.proposed_hypotheses:
            proposal = item.model_dump(mode="json", exclude_none=True)
            for field_name in ("statement", "falsification", "expected_effect"):
                if proposal.get(field_name):
                    proposal[field_name] = semantic.display_text(str(proposal[field_name]))
            proposal["source"] = proposer_output.source
            proposal["model"] = proposer_output.model
            proposal["generated_at"] = (
                proposer_output.generated_at.isoformat()
                if proposer_output.generated_at is not None
                else None
            )
            display_proposals.append(proposal)
        planner_input.llm_hypothesis_proposals = display_proposals
        if proposer_output.generated_at is not None:
            planner_input.generated_at = proposer_output.generated_at
        proposer_summaries_display = [
            semantic.display_text(summary_text)
            for summary_text in proposer_output.proposal_summaries
            if summary_text
        ]
        planner_input.planner_guidance.extend(
            note
            for note in [*explicit_focus_notes, *proposer_output.guidance_notes]
            if note not in planner_input.planner_guidance
        )
        planner_input.recent_reasoning_traces.extend(
            self._notes_to_traces(
                notes=proposer_summaries_display,
                round_id=planner_input.source_round_id,
                stage="hypothesis_proposer",
                prefix="HP",
                referenced_variables=focus_features_raw,
            )
        )
        planner_input.recent_reasoning_traces = planner_input.recent_reasoning_traces[-10:]
        return planner_input

    def _augment_uncertainty_context_with_llm_services(
        self,
        *,
        planner_input: ReasoningPlannerInput,
        uncertainties,
    ) -> tuple[ReasoningPlannerInput, object]:
        """Phase 2 LLM pass: consume the frozen tree, then generate uncertainties and traces."""
        hypothesis_tree = self.repository.load_hypothesis_tree()
        miner = MultiSourceUncertaintyMiner(self.repository)
        mining_context = miner._build_context(
            planner_input=planner_input,
            uncertainties=uncertainties,
            round_id=planner_input.next_round_id,
        )
        mining_result = miner.mine(context=mining_context)
        mined_candidates = [candidate.to_prompt_dict() for candidate in mining_result.candidates]
        rag_context = self.rag_service.build_context_bundle(planner_input)
        questioner_output = self.scientific_questioner.question(
            planner_input=planner_input,
            rag_context=rag_context,
            mined_candidates=mined_candidates,
        )

        questioner_output = self._align_questioner_output_to_dictionary(
            planner_input=planner_input,
            questioner_output=questioner_output,
            tree=hypothesis_tree,
            uncertainty_records=uncertainties.records,
        )
        planner_input.planner_guidance.extend(
            note for note in questioner_output.guidance_notes if note not in planner_input.planner_guidance
        )
        planner_input.recent_reasoning_traces.extend(
            self._notes_to_traces(
                notes=questioner_output.challenge_points,
                round_id=planner_input.source_round_id,
                stage="scientific_questioner",
                prefix="SQ",
                referenced_variables=[
                    feature
                    for item in questioner_output.proposed_uncertainties
                    for feature in item.features
                ],
            )
        )
        uncertainties = self._merge_questioner_uncertainties(
            uncertainties=uncertainties,
            planner_input=planner_input,
            proposed=questioner_output.proposed_uncertainties,
            hypothesis_tree=hypothesis_tree,
        )
        planner_input.planner_guidance.append(
            "multi_source_mining:"
            f"hypothesis_conflict={mining_result.source_detail_counts.get('hypothesis_conflict', 0)},"
            f"residual_pattern={mining_result.source_detail_counts.get('residual_pattern', 0)},"
            f"support_shift={mining_result.source_detail_counts.get('support_shift', 0)},"
            f"failure_attribution={mining_result.source_detail_counts.get('failure_attribution', 0)},"
            f"missing_evidence={mining_result.source_detail_counts.get('missing_evidence', 0)}"
        )
        planner_input.recent_reasoning_traces = planner_input.recent_reasoning_traces[-10:]
        return planner_input, uncertainties

    def _mine_programmatic_uncertainties(
        self,
        *,
        planner_input: ReasoningPlannerInput,
        uncertainties,
    ) -> tuple[ReasoningPlannerInput, object]:
        raise RuntimeError(
            "planner_mode=programmatic 已禁用程序补位：科学不确定性必须由真实 LLM 或人工 PI 生成，"
            "不允许使用写死的模板记录。请切换为 llm 模式后重试。"
        )

    @staticmethod
    def _align_questioner_output_to_dictionary(
        *,
        planner_input: ReasoningPlannerInput,
        questioner_output,
        tree=None,
        uncertainty_records=None,
    ):
        """Map LLM uncertainty text to display names without rewriting its wording."""
        semantic = VariableSemanticService.from_summary(planner_input.data_dictionary_summary)

        def clean_prose(value: str) -> str:
            return clean_llm_text(
                value or "",
                tree=tree,
                uncertainty_records=uncertainty_records,
            ).strip()

        aligned_uncertainties = [
            item.model_copy(
                update={
                    "question": clean_prose(semantic.display_text(item.question or "")),
                    "description": (
                        clean_prose(semantic.display_text(item.description or ""))
                        or f"仍需验证：{item.question or '当前科学不确定性问题。'}"
                    ),
                    "features": semantic.raw_list(item.features),
                    "related_hypotheses": [
                        value for value in item.related_hypotheses if str(value or "").strip()
                    ],
                }
            )
            for item in questioner_output.proposed_uncertainties[:10]
        ]
        questioner_output.challenge_points = [
            clean_prose(semantic.display_text(item or ""))
            for item in questioner_output.challenge_points
            if str(item or "").strip()
        ]
        questioner_output.guidance_notes = [
            clean_prose(semantic.display_text(item or ""))
            for item in questioner_output.guidance_notes
            if str(item or "").strip()
        ]
        questioner_output.hypothesis_updates = [
            item.model_copy(
                update={
                    "rationale": clean_prose(item.rationale) or "科学质询未给出明确理由。",
                    "falsification_basis": clean_prose(item.falsification_basis),
                }
            )
            for item in questioner_output.hypothesis_updates
        ]
        questioner_output.proposed_uncertainties = aligned_uncertainties
        return questioner_output

    @staticmethod
    def _rag_context_to_traces(rag_context: RAGContextBundle, round_id: int) -> list[PlannerReasoningTraceItem]:
        traces: list[PlannerReasoningTraceItem] = []
        for index, item in enumerate(rag_context.project_evidence[:2], start=1):
            traces.append(
                PlannerReasoningTraceItem(
                    trace_id=f"RAGP{index:03d}",
                    stage="rag_project",
                    summary=item.excerpt,
                )
            )
        for index, item in enumerate(rag_context.literature_evidence[:2], start=1):
            traces.append(
                PlannerReasoningTraceItem(
                    trace_id=f"RAGL{index:03d}",
                    stage="rag_literature",
                    summary=item.excerpt,
                )
            )
        return traces

    @staticmethod
    def _notes_to_traces(
        *,
        notes: list[str],
        round_id: int,
        stage: str,
        prefix: str,
        referenced_variables: list[str] | None = None,
    ) -> list[PlannerReasoningTraceItem]:
        traces: list[PlannerReasoningTraceItem] = []
        variable_refs = list(dict.fromkeys(referenced_variables or []))
        for index, note in enumerate(notes[:3], start=1):
            traces.append(
                PlannerReasoningTraceItem(
                    trace_id=f"{prefix}{round_id:02d}{index:02d}",
                    stage=stage,
                    summary=note,
                    referenced_variables=variable_refs,
                )
            )
        return traces

    @staticmethod
    def _merge_questioner_uncertainties(
        *,
        uncertainties,
        planner_input: ReasoningPlannerInput,
        proposed: list,
        hypothesis_tree=None,
    ):
        existing_questions = {
            "".join(str(record.question or "").lower().split())
            for record in uncertainties.records
        }
        existing_ids = {record.uncertainty_id for record in uncertainties.records}
        semantic = VariableSemanticService.from_summary(planner_input.data_dictionary_summary)
        next_index = len(uncertainties.records) + 1
        for item in proposed:
            display_question = semantic.display_text(item.question or "")
            display_description = semantic.display_text(item.description or "")
            normalized_question = "".join(str(display_question).lower().split())
            if not normalized_question or normalized_question in existing_questions:
                continue
            uncertainty_id = item.uncertainty_id or f"U_LLM_R{planner_input.next_round_id:02d}_{next_index:02d}"
            while uncertainty_id in existing_ids:
                next_index += 1
                uncertainty_id = f"U_LLM_R{planner_input.next_round_id:02d}_{next_index:02d}"
            related_hypotheses = list(item.related_hypotheses)
            if not related_hypotheses:
                related_hypotheses = related_hypotheses_for_uncertainty(
                    tree=hypothesis_tree,
                    question=display_question or item.question or "",
                    description=display_description or item.description or "",
                    existing=list(item.related_hypotheses),
                )
                if not related_hypotheses:
                    related_hypotheses = [
                        entry.hypothesis_id
                        for entry in planner_input.active_hypotheses
                        if entry.hypothesis_id
                        and (
                            (entry.statement and f"假设“{entry.statement}”" in f"{item.question} {item.description}")
                            or (entry.statement and entry.statement in f"{item.question} {item.description}")
                        )
                    ]
            # Keep the true mapping from the LLM when it is present; the helper
            # only fills missing links so later protocols stay linked.
            notes = ["generated_from_scientific_questioner_llm"]
            if item.mining_sources:
                notes.append(f"mining_sources:{','.join(item.mining_sources)}")
            if item.features:
                notes.append(f"mining_features:{','.join(item.features)}")
            uncertainties.records.append(
                UncertaintyRecord(
                    uncertainty_id=uncertainty_id,
                    question=display_question,
                    description=display_description,
                    related_hypotheses=related_hypotheses,
                    mining_sources=list(item.mining_sources),
                    status="active",
                    priority=item.priority if item.priority in {"low", "medium", "high"} else "medium",
                    created_at_round=planner_input.next_round_id,
                    created_by="scientific_questioner_llm",
                    notes=";".join(notes),
                )
            )
            existing_ids.add(uncertainty_id)
            existing_questions.add(normalized_question)
            next_index += 1
        return uncertainties

    def _log_hypothesis_generation(
        self,
        *,
        round_id: int,
        next_round_id: int,
        generation_result,
        mode: str,
    ) -> None:
        decision_log = self.repository.load_decision_log()
        node_index = generation_result.tree.node_index()
        generated_node_ids = list(generation_result.generated_node_ids)
        decision_log.decisions.append(
            DecisionEntry(
                decision_id=self._next_decision_id(decision_log),
                timestamp=datetime.now(),
                round_id=round_id,
                phase="hypothesis_generation",
                step="llm_hypothesis_generation",
                decision_type="hypothesis_generated",
                made_by="system",
                summary=f"真实 LLM 已为 round {next_round_id} 生成 H1..H5 竞争假设树",
                details={
                    "mode": mode,
                    "source": "real_llm",
                    "model": generation_result.model,
                    "generated_at": (
                        generation_result.tree.generated_at.isoformat()
                        if generation_result.tree.generated_at is not None
                        else None
                    ),
                    "generated_node_ids": generated_node_ids,
                    "generated_hypotheses": [
                        {
                            "hypothesis_id": hypothesis_id,
                            "statement": node_index[hypothesis_id].statement,
                            "generation_rationale": (
                                node_index[hypothesis_id].generation_rationale.model_dump(mode="json", exclude_none=True)
                                if node_index[hypothesis_id].generation_rationale
                                else None
                            ),
                        }
                        for hypothesis_id in generated_node_ids
                    ],
                    "hypothesis_tree_path": self.repository.relativize(self.repository.paths.hypothesis_tree),
                },
            )
        )
        self.repository.save_decision_log(decision_log)

    def _latest_round_review_context(self, decision_log: DecisionLog, round_id: int) -> dict[str, Any]:
        for entry in reversed(decision_log.decisions):
            if entry.round_id == round_id and entry.decision_type == "round_review_requested":
                details = entry.details
                traces = details.get("reasoning_traces", [])
                assessments = details.get("hypothesis_assessments", [])
                context_bits: list[str] = []
                if traces:
                    context_bits.append(f"linked_traces={','.join(item.get('trace_id', '') for item in traces[:3] if item.get('trace_id'))}")
                if assessments:
                    top = assessments[0]
                    try:
                        tree = self.repository.load_hypothesis_tree()
                    except Exception:
                        tree = None
                    try:
                        uncertainty_records = self.repository.load_uncertainties().records
                    except Exception:
                        uncertainty_records = []
                    assessment_text = clean_llm_text(
                        f"assessment:{top.get('hypothesis_id')}->{top.get('status')}",
                        tree=tree,
                        uncertainty_records=uncertainty_records,
                    )
                    context_bits.append(assessment_text)
                return {
                    "experiment_id": details.get("experiment_id"),
                    "reasoning_trace_ids": [item.get("trace_id") for item in traces if item.get("trace_id")],
                    "context_summary": "; ".join(context_bits) or None,
                }
        return {}

    def _recent_reasoning_traces_for_experiment(self, experiment_id: str) -> list[dict[str, Any]]:
        experiment_memory = self.repository.load_experiment_memory()
        entry = experiment_memory.entry_index().get(experiment_id)
        if entry is None:
            return []
        return [
            trace.model_dump(mode="json", exclude_none=True)
            for trace in entry.reasoning_traces[-5:]
        ]

    def _relevant_protocol_refinements(
        self,
        candidate_id: str,
    ) -> list[ProtocolRefinementSuggestion] | None:
        if not self.repository.paths.planner_output.exists():
            return None
        planner_output = self.repository.load_planner_output()
        refinements = [
            item
            for item in planner_output.protocol_refinements
            if item.target_candidate_id in {None, candidate_id}
        ]
        return refinements or None

    def _build_protocol_refinements(
        self,
        *,
        task: ScientificTask,
        candidate: CandidateExperiment,
        source_round: int | None = None,
    ) -> list[ProtocolRefinementSuggestion] | None:
        refinements = list(self._relevant_protocol_refinements(candidate.experiment_id) or [])
        if self.planner_mode != "llm":
            return refinements or None
        planner_refinement = self.experiment_planner.refine(
            task=task,
            candidate=candidate,
            existing_refinements=refinements,
            semantic_service=self._semantic_service(),
            history_context=self._build_history_context(candidate, source_round=source_round),
        )
        refinements.append(planner_refinement)
        return refinements

    def _build_history_context(
        self,
        candidate: CandidateExperiment,
        *,
        source_round: int | None = None,
    ) -> str | None:
        try:
            history = self.repository.load_round_history()
        except Exception:
            return None
        try:
            tree = self.repository.load_hypothesis_tree()
        except Exception:
            tree = None
        try:
            uncertainty_records = self.repository.load_uncertainties().records
        except Exception:
            uncertainty_records = []
        if source_round is None:
            return None
        current_round = source_round
        entries = [entry for entry in history.entries if entry.round_id < current_round]
        if not entries:
            return None
        latest = max(entries, key=lambda entry: entry.round_id)
        metrics = latest.metrics_snapshot
        parts = [
            f"source_round={latest.round_id}",
            f"experiment={latest.source_experiment_id or latest.approved_candidate_id or '--'}",
        ]
        if metrics is not None:
            parts.extend(
                [
                    f"baseline_pearson_r={metrics.baseline_pearson_r}",
                    f"treatment_pearson_r={metrics.treatment_pearson_r}",
                    f"delta_pearson_r={(metrics.delta.pearson_r if metrics.delta else None)}",
                    f"delta_rmse={(metrics.delta.rmse if metrics.delta else None)}",
                ]
            )
        if latest.failure_reason:
            parts.append(
                "failure_reason="
                + clean_llm_text(
                    latest.failure_reason,
                    tree=tree,
                    uncertainty_records=uncertainty_records,
                )
            )
        if latest.scientific_findings:
            parts.append(
                "findings="
                + " | ".join(
                    clean_llm_text(
                        item,
                        tree=tree,
                        uncertainty_records=uncertainty_records,
                    )
                    for item in latest.scientific_findings[:4]
                )
            )
        if (
            latest.three_layer_conclusion is not None
            and latest.three_layer_conclusion.next_round_suggestion is not None
        ):
            parts.append(
                "next_round_suggestion="
                + json.dumps(
                    deep_clean_text(
                        latest.three_layer_conclusion.next_round_suggestion.model_dump(
                            mode="json", exclude_none=True
                        ),
                        tree=tree,
                        uncertainty_records=uncertainty_records,
                    ),
                    ensure_ascii=False,
                )
            )
        return "; ".join(parts)

    def _semantic_service(self) -> VariableSemanticService:
        try:
            return VariableSemanticService.from_data_dictionary(self.repository.load_data_dictionary())
        except Exception:
            return VariableSemanticService()


def _renumber_steps(steps: list[ExperimentStep]) -> list[ExperimentStep]:
    normalized: list[ExperimentStep] = []
    for index, step in enumerate(steps, start=1):
        normalized.append(
            ExperimentStep(
                step=index,
                action=step.action,
                parameters=step.parameters,
            )
        )
    return normalized


def _build_disagreement_steps(candidate: CandidateExperiment) -> list[ExperimentStep]:
    steps: list[ExperimentStep] = []
    for uncertainty_id in candidate.related_uncertainties:
        disagreement = candidate.disagreement_context.get(uncertainty_id, {})
        if not disagreement:
            continue
        compared_hypotheses = list(disagreement.keys())
        source_triggers = sorted(
            {
                item.rationale_trigger
                for item in disagreement.values()
                if hasattr(item, "rationale_trigger") and item.rationale_trigger
            }
        )
        steps.append(
            ExperimentStep(
                step=1,
                action="resolve_disagreement",
                parameters={
                    "uncertainty_id": uncertainty_id,
                    "question": candidate.scientific_question,
                    "compared_hypotheses": compared_hypotheses,
                    "source_triggers": source_triggers,
                },
            )
        )
    return steps


def _disagreement_notes(
    candidate: CandidateExperiment,
    *,
    tree=None,
    uncertainty_records=None,
) -> list[str]:
    notes: list[str] = []
    for uncertainty_id in candidate.related_uncertainties:
        disagreement = candidate.disagreement_context.get(uncertainty_id, {})
        if not disagreement:
            continue
        record = None
        for item in uncertainty_records or []:
            if getattr(item, "uncertainty_id", None) == uncertainty_id:
                record = item
                break
        question = getattr(record, "question", None) if record is not None else None
        notes.append(f"目标不确定性：{question or '相关科学不确定性'}")
        labels: list[str] = []
        for hypothesis_id in disagreement.keys():
            node = tree.node_index().get(hypothesis_id) if tree is not None else None
            labels.append(
                getattr(node, "display_hypothesis_id", None)
                or (f"H{node.level}" if node is not None and getattr(node, "level", None) else "")
                or "相关假设"
            )
        notes.append(f"待区分假设：{','.join(labels) or '相关假设'}")
        trigger_set = sorted(
            {
                item.rationale_trigger
                for item in disagreement.values()
                if hasattr(item, "rationale_trigger") and item.rationale_trigger
            }
        )
        if trigger_set:
            notes.append(f"分歧触发来源：{','.join(trigger_set)}")
    return notes
