from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.display_text_cleaner import clean_llm_text
from core.json_safe import json_dumps
from core.state_repository import UnifiedStateRepository
from core.support_update_rules import (
    EVIDENCE_INCONCLUSIVE,
    EVIDENCE_IDENTICAL_PREDICTIONS,
    EVIDENCE_NO_STABLE_GAIN,
    EVIDENCE_STABLE_DEGRADATION,
    EVIDENCE_STABLE_IMPROVEMENT,
    compute_support_update_from_evidence,
    priority_after_action,
)
from core.hypothesis_linkage import build_predictions_for_nodes
from core.hypothesis_state_machine import (
    ACTIVATION_THRESHOLD,
    ACTIVE,
    CONVERGED,
    OBSERVING,
    PENDING,
    PRUNED,
    resolve_status,
)
from core.unified_schema import (
    ClosureChecklistItem,
    CritiqueRecord,
    EvidenceItem,
    EvaluationResult,
    ExperimentMemoryEntry,
    ExperimentMemoryState,
    ExperimentProtocol,
    ExperimentResult,
    FailureHistoryState,
    FailureRecord,
    HypothesisTreeState,
    InterpretationEnhancement,
    LatestTreeUpdate,
    MAX_ACTIVE_UNCERTAINTIES,
    MetricsTimelineEntry,
    MetricsTimelineState,
    ProcessPhase,
    ProcessState,
    PrioritizedUncertainty,
    ProcessStep,
    ReasoningTraceEntry,
    RoundHistoryEntry,
    RoundHistoryState,
    SupportHistoryEntry,
    TreeSummary,
    UncertaintyHistoryEntry,
    UncertaintyPriorityQueue,
    UncertaintyRecord,
    UncertaintyState,
)
from core.variable_semantic_service import VariableSemanticService


def translate_three_layer_conclusion(
    repository: UnifiedStateRepository,
    evaluation: EvaluationResult,
) -> None:
    """Replace raw column names with physical display names in the conclusion."""
    conclusion = evaluation.scientific.three_layer_conclusion
    if conclusion is None:
        return
    try:
        data_dictionary = repository.load_data_dictionary()
        semantic = (
            VariableSemanticService.from_data_dictionary(data_dictionary)
            if data_dictionary
            else VariableSemanticService()
        )
    except Exception:
        semantic = VariableSemanticService()

    try:
        tree = repository.load_hypothesis_tree()
    except Exception:
        tree = None

    try:
        uncertainty_records = repository.load_uncertainties().records
    except Exception:
        uncertainty_records = []

    tree = tree or HypothesisTreeState(
        tree_id="empty",
        task_id="empty",
        root_question="",
        nodes=[],
    )

    def _clean(text: str | None) -> str:
        return clean_llm_text(
            semantic.display_text(text),
            tree=tree,
            uncertainty_records=uncertainty_records,
        )

    conclusion.experiment_layer.design_summary = _clean(
        conclusion.experiment_layer.design_summary
    )
    scientific = conclusion.scientific_layer
    scientific.main_question = _clean(scientific.main_question)
    scientific.answer = _clean(scientific.answer)
    scientific.path_question = _clean(scientific.path_question)
    scientific.path_answer = _clean(scientific.path_answer)
    scientific.evidence_text = _clean(scientific.evidence_text)
    if conclusion.data_layer is not None:
        data = conclusion.data_layer
        data.rmse_attribution = _clean(data.rmse_attribution)
        data.pearson_attribution = _clean(data.pearson_attribution)
        data.skill_delta_meaning = _clean(data.skill_delta_meaning)
        data.anomalies = [
            _clean(item)
            for item in data.anomalies
        ]
        data.next_focus = _clean(data.next_focus)
        data.comparison_analysis = _clean(data.comparison_analysis)
        for chart in data.chart_analyses:
            chart.description = _clean(chart.description)
            chart.key_observations = [
                _clean(item) for item in chart.key_observations
            ]
            chart.anomaly_or_insight = _clean(chart.anomaly_or_insight)
    conclusion.hypothesis_layer_summary = _clean(conclusion.hypothesis_layer_summary)
    conclusion.overall_summary = _clean(conclusion.overall_summary)
    if conclusion.next_round_suggestion is not None:
        suggestion = conclusion.next_round_suggestion
        suggestion.evidence_summary = _clean(suggestion.evidence_summary)
        suggestion.remaining_uncertainty_analysis = _clean(
            suggestion.remaining_uncertainty_analysis
        )
        suggestion.pi_decision_advice = _clean(suggestion.pi_decision_advice)
        suggestion.experiment_design_advice = _clean(
            suggestion.experiment_design_advice
        )
        suggestion.hypothesis_space_advice = _clean(
            suggestion.hypothesis_space_advice
        )
        suggestion.notes = [_clean(item) for item in suggestion.notes]
    if conclusion.tracking_layer is not None:
        conclusion.tracking_layer.audit_items = [
            _clean(item) for item in conclusion.tracking_layer.audit_items
        ]
    from core.hypothesis_identity import remap_hypothesis_id, resolve_hypothesis_display

    for row in conclusion.hypothesis_layer:
        resolved = resolve_hypothesis_display(tree, row.hypothesis_id, row.statement, semantic)
        if resolved is None:
            continue
        display_label, canonical_statement = resolved
        row.hypothesis_id = remap_hypothesis_id(tree, row.hypothesis_id, row.statement)
        row.display_hypothesis_id = display_label
        row.statement = clean_llm_text(
            canonical_statement,
            tree=tree,
            uncertainty_records=uncertainty_records,
        )
        original_prefix = row.conclusion.split(" ")[0].strip("：:：")
        if original_prefix and row.conclusion.startswith(original_prefix):
            row.conclusion = row.conclusion.replace(original_prefix, row.display_hypothesis_id, 1)
        row.conclusion = clean_llm_text(
            row.conclusion,
            tree=tree,
            uncertainty_records=uncertainty_records,
        )


class UnifiedStateUpdater:
    """Apply execution and evaluation outputs back into the persistent knowledge state."""

    def __init__(self, repository: UnifiedStateRepository) -> None:
        self.repository = repository

    def _display_service(self) -> VariableSemanticService:
        try:
            data_dictionary = self.repository.load_data_dictionary()
            if data_dictionary:
                return VariableSemanticService.from_data_dictionary(data_dictionary)
        except Exception:
            pass
        return VariableSemanticService()

    def _translate_three_layer_conclusion(self, evaluation: EvaluationResult) -> None:
        translate_three_layer_conclusion(self.repository, evaluation)

    def apply_evaluation(
        self,
        *,
        protocol: ExperimentProtocol,
        result: ExperimentResult,
        evaluation: EvaluationResult,
        interpretation_enhancements: list[InterpretationEnhancement] | None = None,
        interpreter_source: str = "scientific_interpreter_llm",
        protocol_path: str | None = "config/latest_protocol.json",
        result_path: str | None = None,
        evaluation_path: str | None = None,
    ) -> dict[str, object]:
        evidence_version = _resolve_evidence_version(protocol, result)
        if self.repository.has_state_update_commit(
            protocol.experiment_id,
            evidence_version,
            update_type="evaluation",
        ):
            return self._load_current_state_snapshot()
        tree = self.repository.load_hypothesis_tree()
        uncertainties = self.repository.load_uncertainties()
        experiment_memory = self.repository.load_experiment_memory()
        round_history = self.repository.load_round_history()
        metrics_timeline = self.repository.load_metrics_timeline()
        process_state = self.repository.load_process_state()
        protocol = self._ensure_protocol_core_outputs(
            protocol=protocol,
            evaluation=evaluation,
            tree=tree,
            uncertainties=uncertainties,
        )
        self._write_exploratory_protocol_outputs(
            protocol=protocol,
            evaluation=evaluation,
            evidence_version=evidence_version,
        )

        tree = self._update_hypothesis_tree(
            tree,
            protocol,
            evaluation,
            result=result,
            evidence_version=evidence_version,
        )
        uncertainties = self._update_uncertainties(
            uncertainties,
            protocol,
            evaluation,
            result=result,
            evidence_version=evidence_version,
        )
        experiment_memory = self._update_experiment_memory(
            experiment_memory,
            protocol=protocol,
            result=result,
            evaluation=evaluation,
            protocol_path=protocol_path,
            result_path=result_path,
            evaluation_path=evaluation_path,
        )
        if interpretation_enhancements:
            tree, uncertainties, experiment_memory = self._apply_interpretation_enhancements(
                tree=tree,
                uncertainties=uncertainties,
                experiment_memory=experiment_memory,
                protocol=protocol,
                result=result,
                evaluation=evaluation,
                interpretation_enhancements=interpretation_enhancements,
                interpreter_source=interpreter_source,
            )
        if evaluation and evaluation.scientific.three_layer_conclusion is not None:
            self._translate_three_layer_conclusion(evaluation)
        round_history = self._update_round_history(
            round_history,
            protocol=protocol,
            evaluation=evaluation,
            experiment_memory=experiment_memory,
            failure_reason=None,
        )
        metrics_timeline = self._update_metrics_timeline(
            metrics_timeline,
            protocol=protocol,
            evaluation=evaluation,
            experiment_memory=experiment_memory,
            execution_status=result.status,
        )
        process_state = self._update_process_state(process_state, protocol.round_id)

        self.repository.save_hypothesis_tree(tree)
        self.repository.save_uncertainties(uncertainties)
        self.repository.save_experiment_memory(experiment_memory)
        self.repository.save_round_history(round_history)
        self.repository.save_metrics_timeline(metrics_timeline)
        self.repository.save_process_state(process_state)
        self.repository.mark_state_update_committed(
            protocol.experiment_id,
            evidence_version,
            update_type="evaluation",
            extra={"round_id": protocol.round_id, "experiment_id": protocol.experiment_id},
        )

        return {
            "hypothesis_tree": tree,
            "uncertainties": uncertainties,
            "experiment_memory": experiment_memory,
            "process_state": process_state,
        }

    def apply_execution_failure(
        self,
        *,
        protocol: ExperimentProtocol,
        error: Exception,
        protocol_path: str | None = "config/latest_protocol.json",
    ) -> dict[str, object]:
        evidence_version = _resolve_evidence_version(protocol)
        if self.repository.has_state_update_commit(
            protocol.experiment_id,
            evidence_version,
            update_type="failure",
        ):
            return {
                "experiment_memory": self.repository.load_experiment_memory(),
                "round_history": self.repository.load_round_history(),
                "metrics_timeline": self.repository.load_metrics_timeline(),
                "failure_history": self.repository.load_failure_history(),
            }
        now = datetime.now()
        experiment_memory = self.repository.load_experiment_memory()
        round_history = self.repository.load_round_history()
        metrics_timeline = self.repository.load_metrics_timeline()
        failure_history = self.repository.load_failure_history()
        process_state = self.repository.load_process_state()
        protocol = self._ensure_protocol_core_outputs(
            protocol=protocol,
            evaluation=None,
            tree=self.repository.load_hypothesis_tree(),
            uncertainties=self.repository.load_uncertainties(),
        )
        self._write_exploratory_protocol_outputs(
            protocol=protocol,
            evaluation=None,
            evidence_version=evidence_version,
        )

        failure_message = str(error).strip() or repr(error)
        entry = experiment_memory.entry_index().get(protocol.experiment_id)
        if entry is None:
            entry = ExperimentMemoryEntry(
                experiment_id=protocol.experiment_id,
                round_id=protocol.round_id,
                status="failed",
                tested_hypotheses=protocol.tested_hypotheses,
                target_uncertainties=protocol.target_uncertainties,
                protocol_path=self.repository.relativize(protocol_path),
                result_path=None,
                evaluation_path=None,
                metrics_snapshot=None,
                key_findings=[f"实验失败：{failure_message}"],
                visualizations=[],
                created_at=now,
                updated_at=now,
            )
            experiment_memory.entries.append(entry)
        else:
            entry.round_id = protocol.round_id
            entry.status = "failed"
            entry.tested_hypotheses = protocol.tested_hypotheses
            entry.target_uncertainties = protocol.target_uncertainties
            entry.protocol_path = self.repository.relativize(protocol_path)
            entry.result_path = None
            entry.evaluation_path = None
            entry.metrics_snapshot = None
            entry.key_findings = [f"实验失败：{failure_message}"]
            entry.visualizations = []
            entry.updated_at = now

        failure_history.records.append(
            FailureRecord(
                failure_id=f"FAIL_R{protocol.round_id:02d}_{len(failure_history.records) + 1:03d}",
                round_id=protocol.round_id,
                phase=process_state.current_phase or "experiment_execution",
                step=process_state.current_step or "execution_failed",
                experiment_id=protocol.experiment_id,
                error_type=error.__class__.__name__,
                message=failure_message,
                traceback_excerpt=repr(error),
                can_continue=True,
            )
        )
        failure_history.current_round = max(failure_history.current_round, protocol.round_id)
        failure_history.last_updated = now

        round_history = self._update_round_history(
            round_history,
            protocol=protocol,
            evaluation=None,
            experiment_memory=experiment_memory,
            failure_reason=failure_message,
        )
        metrics_timeline = self._update_metrics_timeline(
            metrics_timeline,
            protocol=protocol,
            evaluation=None,
            experiment_memory=experiment_memory,
            execution_status="failed",
        )

        self.repository.save_experiment_memory(experiment_memory)
        self.repository.save_round_history(round_history)
        self.repository.save_metrics_timeline(metrics_timeline)
        self.repository.save_failure_history(failure_history)
        self.repository.mark_state_update_committed(
            protocol.experiment_id,
            evidence_version,
            update_type="failure",
            extra={"round_id": protocol.round_id, "experiment_id": protocol.experiment_id},
        )
        return {
            "experiment_memory": experiment_memory,
            "round_history": round_history,
            "metrics_timeline": metrics_timeline,
            "failure_history": failure_history,
        }

    def _ensure_protocol_core_outputs(
        self,
        *,
        protocol: ExperimentProtocol,
        evaluation: EvaluationResult | None,
        tree: HypothesisTreeState,
        uncertainties: UncertaintyState,
    ) -> ExperimentProtocol:
        effective = protocol.model_copy(deep=True)
        tested_hypotheses = _unique_preserve_order(effective.tested_hypotheses)
        if not tested_hypotheses:
            tested_hypotheses = _infer_tested_hypotheses(
                protocol=protocol,
                evaluation=evaluation,
                tree=tree,
                uncertainties=uncertainties,
            )
            if tested_hypotheses:
                effective.notes.append("auto_filled_tested_hypotheses")
        target_uncertainties = _unique_preserve_order(effective.target_uncertainties)
        if not target_uncertainties:
            target_uncertainties = _infer_target_uncertainties(
                protocol=protocol,
                evaluation=evaluation,
                uncertainties=uncertainties,
            )
            if target_uncertainties:
                effective.notes.append("auto_filled_target_uncertainties")
        effective.tested_hypotheses = tested_hypotheses
        effective.target_uncertainties = target_uncertainties
        if tested_hypotheses and not effective.hypothesis_predictions:
            predicted = build_predictions_for_nodes(tree, tested_hypotheses)
            if predicted:
                effective.hypothesis_predictions.update(predicted)
                effective.notes.append("auto_filled_hypothesis_predictions")
        return effective

    def _write_exploratory_protocol_outputs(
        self,
        *,
        protocol: ExperimentProtocol,
        evidence_version: str,
        evaluation: EvaluationResult | None,
    ) -> None:
        """Persist inferred protocol links without mutating the approved protocol file."""
        outputs: dict[str, object] = {}
        if protocol.notes:
            auto_filled = [
                note for note in protocol.notes if note.startswith("auto_filled_")
            ]
            if auto_filled:
                outputs["auto_filled_notes"] = auto_filled
        if protocol.hypothesis_predictions:
            outputs["hypothesis_predictions"] = {
                key: (
                    value.model_dump(mode="json", exclude_none=True)
                    if hasattr(value, "model_dump")
                    else value
                )
                for key, value in protocol.hypothesis_predictions.items()
            }
        if not outputs:
            return
        payload = {
            "experiment_id": protocol.experiment_id,
            "round_id": protocol.round_id,
            "tested_hypotheses": list(protocol.tested_hypotheses),
            "target_uncertainties": list(protocol.target_uncertainties),
            "exploratory": True,
            "evidence_version": evidence_version,
            "evidence_basis": (
                [evaluation.scientific.experiment_id]
                if evaluation is not None
                else [protocol.experiment_id]
            ),
            **outputs,
        }
        text = json_dumps(payload, indent=2)
        config_path = (
            self.repository.project_root
            / "config"
            / "latest_protocol.core_outputs.json"
        )
        round_path = (
            self.repository.project_root
            / "results"
            / f"round_{protocol.round_id:02d}"
            / "protocol.core_outputs.json"
        )
        for path in (config_path, round_path):
            UnifiedStateRepository._write_atomic_text(path, text)

    def _load_current_state_snapshot(self) -> dict[str, object]:
        return {
            "hypothesis_tree": self.repository.load_hypothesis_tree(),
            "uncertainties": self.repository.load_uncertainties(),
            "experiment_memory": self.repository.load_experiment_memory(),
            "process_state": self.repository.load_process_state(),
        }

    def _apply_interpretation_enhancements(
        self,
        *,
        tree: HypothesisTreeState,
        uncertainties: UncertaintyState,
        experiment_memory: ExperimentMemoryState,
        protocol: ExperimentProtocol,
        result: ExperimentResult | None = None,
        evaluation: EvaluationResult,
        interpretation_enhancements: list[InterpretationEnhancement],
        interpreter_source: str,
    ) -> tuple[HypothesisTreeState, UncertaintyState, ExperimentMemoryState]:
        node_index = tree.node_index()
        assessment_index = {
            item.hypothesis_id: item
            for item in evaluation.scientific.hypothesis_assessments
        }
        uncertainty_index = uncertainties.record_index()
        memory_entry = experiment_memory.entry_index().get(protocol.experiment_id)
        now = datetime.now()
        fallback_version = _resolve_evidence_version(protocol, result)

        for enhancement in interpretation_enhancements:
            evidence_version = enhancement.evidence_version or fallback_version
            run_id = enhancement.run_id or (result.run_id if result is not None else None)
            contrast_id = enhancement.contrast_id
            enhancement_key = enhancement.evidence_key or _record_key(
                protocol.experiment_id,
                f"enhancement:{enhancement.enhancement_id}",
                run_id=run_id,
                contrast_id=contrast_id,
            )
            cleaned_interpretation = clean_llm_text(
                enhancement.interpretation,
                tree=tree,
                uncertainty_records=uncertainties.records,
            )
            cleaned_state_note = clean_llm_text(
                enhancement.suggested_state_note,
                tree=tree,
                uncertainty_records=uncertainties.records,
            )
            numeric_support_delta = 0.0
            if enhancement.target_hypothesis_id:
                node = node_index.get(enhancement.target_hypothesis_id)
                if node is not None:
                    measurement_assessment = assessment_index.get(
                        enhancement.target_hypothesis_id
                    )
                    if (
                        measurement_assessment is not None
                        and measurement_assessment.evidence_classification
                        in {
                            EVIDENCE_STABLE_IMPROVEMENT,
                            EVIDENCE_STABLE_DEGRADATION,
                        }
                    ):
                        numeric_support_delta = (
                            node.support_score - measurement_assessment.support_before
                        )
                    _replace_or_append(
                        node.evidence_items,
                        f"{enhancement_key}:evidence",
                        EvidenceItem(
                            id=enhancement.enhancement_id,
                            type="expert_judgment",
                            description=cleaned_interpretation,
                            source=interpreter_source,
                            added_at_round=protocol.round_id,
                            support_weight=0.0,
                            experiment_id=protocol.experiment_id,
                            run_id=run_id,
                            contrast_id=contrast_id,
                            evidence_version=evidence_version,
                            evidence_key=f"{enhancement_key}:evidence",
                        ),
                    )
                    _replace_or_append(
                        node.critiques,
                        f"{enhancement_key}:critique",
                        CritiqueRecord(
                            id=f"{enhancement.enhancement_id}_critique",
                            content=cleaned_interpretation,
                            from_role="scientific_interpreter",
                            added_at_round=protocol.round_id,
                            experiment_id=protocol.experiment_id,
                            run_id=run_id,
                            contrast_id=contrast_id,
                            evidence_version=evidence_version,
                            evidence_key=f"{enhancement_key}:critique",
                        ),
                    )
                    _append_unique_text(node.alternative_explanations, cleaned_interpretation)
                    if enhancement.suggested_state_note:
                        _append_unique_text(node.alternative_explanations, cleaned_state_note)

            total_priority_delta = 0.0
            for uncertainty_id in enhancement.related_uncertainties:
                record = uncertainty_index.get(uncertainty_id)
                if record is None:
                    continue
                history_key = f"{enhancement_key}:uncertainty:{uncertainty_id}"
                existing_history = next(
                    (
                        item
                        for item in record.history
                        if getattr(item, "evidence_key", None) == history_key
                    ),
                    None,
                )
                if existing_history is not None and existing_history.evidence_version == evidence_version:
                    continue
                before_priority = record.priority
                record.priority = priority_after_action(
                    record.priority,
                    enhancement.uncertainty_priority_action,
                )
                record.notes = (
                    f"{record.notes} | {cleaned_interpretation}"
                    if record.notes and cleaned_interpretation not in record.notes
                    else (record.notes or cleaned_interpretation)
                )
                _replace_or_append(
                    record.history,
                    history_key,
                    UncertaintyHistoryEntry(
                        round=protocol.round_id,
                        event="scientific_interpreter",
                        source=interpreter_source,
                        description=cleaned_interpretation,
                        experiment_id=protocol.experiment_id,
                        run_id=run_id,
                        contrast_id=contrast_id,
                        evidence_version=evidence_version,
                        evidence_key=history_key,
                    ),
                )
                total_priority_delta += _priority_to_score(record.priority) - _priority_to_score(before_priority)

            if memory_entry is not None:
                trace_key = f"{enhancement_key}:trace"
                _replace_or_append(
                    memory_entry.reasoning_traces,
                    trace_key,
                    ReasoningTraceEntry(
                        trace_id=enhancement.enhancement_id,
                        round_id=protocol.round_id,
                        source=interpreter_source,
                        stage="scientific_interpreter",
                        summary=cleaned_interpretation,
                        related_hypotheses=[enhancement.target_hypothesis_id] if enhancement.target_hypothesis_id else [],
                        related_uncertainties=enhancement.related_uncertainties,
                        support_delta=numeric_support_delta,
                        priority_delta=round(total_priority_delta, 4) if enhancement.related_uncertainties else None,
                        experiment_id=protocol.experiment_id,
                        run_id=run_id,
                        contrast_id=contrast_id,
                        evidence_version=evidence_version,
                        evidence_key=trace_key,
                    ),
                )
                memory_entry.updated_at = now
                _append_unique_text(memory_entry.key_findings, cleaned_interpretation)

        _propagate_tree_state(
            tree=tree,
            round_id=protocol.round_id,
            min_active_hypotheses=_minimum_active_hypotheses(protocol),
        )
        tree.active_hypotheses = [
            node.hypothesis_id for node in tree.nodes if node.status in {"active", "converged"}
        ]
        tree.pruned_hypotheses = [node.hypothesis_id for node in tree.nodes if node.status == "pruned"]
        tree.pending_hypotheses = [
            node.hypothesis_id for node in tree.nodes if node.status in {"draft", "pending"}
        ]
        tree.tree_summary = _rebuild_tree_summary(tree)
        tree.latest_update = LatestTreeUpdate(
            round=protocol.round_id,
            event="scientific_interpreter_applied",
            description=f"scientific interpreter applied {len(interpretation_enhancements)} interpretation enhancements",
        )
        uncertainties.last_updated = now
        uncertainties.priority_queue = UncertaintyPriorityQueue(
            last_updated=now,
            current_round=protocol.round_id,
            queue=sorted(
                [
                    PrioritizedUncertainty(
                        uncertainty_id=record.uncertainty_id,
                        question=record.question,
                        priority_score=_priority_to_score(record.priority),
                        status=record.status,
                        estimated_resolution_round=record.created_at_round + 1,
                    )
                    for record in uncertainties.records
                    if record.status not in {"resolved", "deprecated"}
                ],
                key=lambda item: item.priority_score,
                reverse=True,
            )[:MAX_ACTIVE_UNCERTAINTIES],
            resolved=[record.uncertainty_id for record in uncertainties.records if record.status == "resolved"],
            deprecated=[record.uncertainty_id for record in uncertainties.records if record.status == "deprecated"],
        )
        return tree, uncertainties, experiment_memory

    def _update_hypothesis_tree(
        self,
        tree: HypothesisTreeState,
        protocol: ExperimentProtocol,
        evaluation: EvaluationResult,
        *,
        result: ExperimentResult,
        evidence_version: str,
    ) -> HypothesisTreeState:
        node_index = tree.node_index()
        assessment_index = {
            item.hypothesis_id: item for item in evaluation.scientific.hypothesis_assessments
        }
        updated_nodes = 0
        skipped_hypotheses: list[str] = []
        signal = _classify_support_signal(evaluation)
        pending_review = protocol.experiment_type != "incremental_prediction"
        base_run_id = result.run_id
        base_contrast_id = None

        for hypothesis_id in protocol.tested_hypotheses:
            node = node_index.get(hypothesis_id)
            if node is None:
                skipped_hypotheses.append(hypothesis_id)
                continue

            assessment = assessment_index.get(hypothesis_id)
            run_id = getattr(assessment, "run_id", None) or base_run_id
            contrast_id = getattr(assessment, "contrast_id", None) or base_contrast_id
            node_key = _record_key(
                protocol.experiment_id,
                f"hypothesis:{hypothesis_id}",
                run_id=run_id,
                contrast_id=contrast_id,
            )
            support_key = f"{node_key}:support"
            evidence_key = f"{node_key}:evidence"
            existing_support = next(
                (
                    item
                    for item in node.support_history
                    if getattr(item, "evidence_key", None) == support_key
                ),
                None,
            )
            if existing_support is not None and existing_support.evidence_version == evidence_version:
                continue

            evidence_identity = _scientific_evidence_identity(
                protocol=protocol,
                result=result,
                evaluation=evaluation,
                hypothesis_id=hypothesis_id,
            )
            if existing_support is not None:
                if existing_support.evidence_identity == evidence_identity:
                    # 同一科学证据只刷新审计版本，不重复累计支持度。
                    existing_support.evidence_version = evidence_version
                    existing_evidence = next(
                        (
                            item
                            for item in node.evidence_items
                            if getattr(item, "evidence_key", None) == evidence_key
                        ),
                        None,
                    )
                    if existing_evidence is not None:
                        existing_evidence.evidence_version = evidence_version
                    continue
                old_delta = existing_support.support_delta
                if old_delta is None and existing_support.support_before is not None:
                    old_delta = round(
                        existing_support.score - existing_support.support_before,
                        4,
                    )
                before = (
                    round(node.support_score - old_delta, 4)
                    if old_delta is not None
                    else node.support_score
                )
            else:
                before = node.support_score
            if pending_review:
                after = before
                reason = "该实验类型结果待科学评价，仅记录证据，不自动调整支持度。"
            else:
                if assessment is not None:
                    after = _support_after_from_assessment(
                        before=before,
                        assessment=assessment,
                        prediction=protocol.hypothesis_predictions.get(hypothesis_id),
                        observed_delta=_observed_delta_from_evaluation(evaluation),
                    )
                    reason = assessment.reason
                else:
                    after = _apply_support_shift(before, signal)
                    reason = (
                        "缺少结构化支持度判定；"
                        "仅 bootstrap CI 稳定的方向可做小幅保守更新。"
                    )
            support_delta = round(after - before, 4)

            node.support_score = after
            if not pending_review:
                node.status = resolve_status(
                    previous_status=node.status,
                    support_after=after,
                    support_history=node.support_history,
                    current_round=protocol.round_id,
                    parent_support=(
                        node_index.get(node.parent_id).support_score
                        if node.parent_id and node.parent_id in node_index
                        else None
                    ),
                    evidence_classification=(
                        getattr(assessment, "evidence_classification", None)
                        if assessment is not None
                        else None
                    ),
                    direction_matched=(
                        getattr(assessment, "direction_matched", None)
                        if assessment is not None
                        else None
                    ),
                )
            node.updated_at_round = protocol.round_id
            _replace_or_append(
                node.support_history,
                support_key,
                SupportHistoryEntry(
                    round=protocol.round_id,
                    score=after,
                    support_before=before,
                    support_delta=support_delta,
                    evidence_source="experimental",
                    event=(
                        f"pending_scientific_review:{protocol.experiment_id}"
                        if pending_review
                        else f"evaluation_writeback:{protocol.experiment_id}"
                    ),
                    experiment_id=protocol.experiment_id,
                    run_id=run_id,
                    contrast_id=contrast_id,
                    evidence_version=evidence_version,
                    evidence_key=support_key,
                    evidence_identity=evidence_identity,
                    evidence_classification=(
                        getattr(assessment, "evidence_classification", None)
                        if assessment is not None
                        else None
                    ),
                    direction_matched=(
                        getattr(assessment, "direction_matched", None)
                        if assessment is not None
                        else None
                    ),
                ),
            )
            _replace_or_append(
                node.evidence_items,
                evidence_key,
                EvidenceItem(
                    id=f"{protocol.experiment_id}_{hypothesis_id}",
                    type="experimental_result",
                    description=reason,
                    source=protocol.experiment_id,
                    added_at_round=protocol.round_id,
                    support_weight=(
                        0.0
                        if pending_review
                        else min(max(abs(after - before), 0.05), 1.0)
                    ),
                    experiment_id=protocol.experiment_id,
                    run_id=run_id,
                    contrast_id=contrast_id,
                    evidence_version=evidence_version,
                    evidence_key=evidence_key,
                ),
            )
            updated_nodes += 1

        min_active = _minimum_active_hypotheses(protocol)
        _propagate_tree_state(
            tree=tree,
            round_id=protocol.round_id,
            min_active_hypotheses=min_active,
        )
        tree.current_round = max(tree.current_round, protocol.round_id)
        tree.active_hypotheses = [
            node.hypothesis_id for node in tree.nodes if node.status in {"active", "converged"}
        ]
        tree.pruned_hypotheses = [node.hypothesis_id for node in tree.nodes if node.status == "pruned"]
        tree.pending_hypotheses = [
            node.hypothesis_id for node in tree.nodes if node.status in {"draft", "pending"}
        ]
        tree.tree_summary = _rebuild_tree_summary(tree)
        description = f"updated {updated_nodes} hypotheses from {protocol.experiment_id}"
        if skipped_hypotheses:
            description += f"; skipped missing ids {skipped_hypotheses}"
        tree.latest_update = LatestTreeUpdate(
            round=protocol.round_id,
            event="evaluation_writeback",
            description=description,
        )
        return tree

    def _update_uncertainties(
        self,
        state: UncertaintyState,
        protocol: ExperimentProtocol,
        evaluation: EvaluationResult,
        *,
        result: ExperimentResult,
        evidence_version: str,
    ) -> UncertaintyState:
        now = datetime.now()
        record_index = state.record_index()
        remaining = evaluation.scientific.evidence_summary.remaining_uncertainties
        disagreement_index = {
            item.uncertainty_id: item for item in evaluation.scientific.disagreement_updates
        }
        run_id = result.run_id

        for uncertainty_id in protocol.target_uncertainties:
            record = record_index.get(uncertainty_id)
            update = disagreement_index.get(uncertainty_id)
            if record is None or update is None:
                continue
            history_key = _record_key(
                protocol.experiment_id,
                f"uncertainty:{uncertainty_id}",
                run_id=run_id,
                contrast_id=None,
            )
            existing_history = next(
                (
                    item
                    for item in record.history
                    if getattr(item, "evidence_key", None) == history_key
                ),
                None,
            )
            if existing_history is not None and existing_history.evidence_version == evidence_version:
                continue
            for hypothesis_id, disagreement in record.disagreement.items():
                if hypothesis_id in update.compared_hypotheses:
                    assessment = next(
                        (item for item in evaluation.scientific.hypothesis_assessments if item.hypothesis_id == hypothesis_id),
                        None,
                    )
                    if assessment is not None and hasattr(disagreement, "support_score"):
                        disagreement.support_score = assessment.support_after
                        disagreement.status = assessment.status
            record.status = "resolved" if update.resolution_status == "resolved" else "active"
            record.resolution_status = update.resolution_status
            record.resolved_at_round = protocol.round_id if update.resolution_status == "resolved" else None
            record.resolved_by = protocol.experiment_id if update.resolution_status == "resolved" else None
            record.resolution = update.summary if update.resolution_status == "resolved" else None
            record.resolving_experiment = protocol.experiment_id
            record.notes = update.summary
            _replace_or_append(
                record.history,
                history_key,
                UncertaintyHistoryEntry(
                    round=protocol.round_id,
                    event="disagreement_evaluation",
                    source=protocol.experiment_id,
                    description=update.summary,
                    experiment_id=protocol.experiment_id,
                    run_id=run_id,
                    contrast_id=None,
                    evidence_version=evidence_version,
                    evidence_key=history_key,
                ),
            )
            if update.resolution_status == "resolved":
                continue
            if update.resolution_status == "partially_resolved":
                record.priority = "medium" if record.priority == "high" else record.priority

        for offset, item in enumerate(remaining, start=1):
            normalized = _normalize_uncertainty_item(item, protocol.round_id, offset)
            record = record_index.get(normalized["uncertainty_id"])
            remaining_key = (
                f"{_record_key(protocol.experiment_id, 'remaining-uncertainty', run_id=run_id)}"
                f":{normalized['uncertainty_id']}"
            )
            existing_history = None
            if record is not None:
                existing_history = next(
                    (
                        history
                        for history in record.history
                        if getattr(history, "evidence_key", None) == remaining_key
                    ),
                    None,
                )
            if existing_history is not None and existing_history.evidence_version == evidence_version:
                continue
            if record is None:
                record = UncertaintyRecord(
                    uncertainty_id=normalized["uncertainty_id"],
                    question=normalized["question"],
                    description=normalized["description"],
                    related_hypotheses=protocol.tested_hypotheses,
                    status="active",
                    priority=normalized["priority"],
                    created_at_round=protocol.round_id,
                    created_by="state_updater",
                    notes="auto-created from evaluation remaining_uncertainties",
                )
                state.records.append(record)
                record_index[record.uncertainty_id] = record

            record.question = normalized["question"]
            record.description = normalized["description"]
            record.priority = normalized["priority"]
            record.status = "active"
            record.resolving_experiment = protocol.experiment_id
            _replace_or_append(
                record.history,
                remaining_key,
                UncertaintyHistoryEntry(
                    round=protocol.round_id,
                    event="evaluation_writeback",
                    source=protocol.experiment_id,
                    description="remaining uncertainty retained after experiment evaluation",
                    experiment_id=protocol.experiment_id,
                    run_id=run_id,
                    contrast_id=None,
                    evidence_version=evidence_version,
                    evidence_key=remaining_key,
                ),
            )

        queue_items = []
        for record in state.records:
            if record.status in {"resolved", "deprecated"}:
                continue
            queue_items.append(
                PrioritizedUncertainty(
                    uncertainty_id=record.uncertainty_id,
                    question=record.question,
                    priority_score=_priority_to_score(record.priority),
                    status=record.status,
                    estimated_resolution_round=record.created_at_round + 1,
                )
            )

        state.current_round = max(state.current_round, protocol.round_id)
        state.last_updated = now
        state.priority_queue = UncertaintyPriorityQueue(
            last_updated=now,
            current_round=protocol.round_id,
            queue=sorted(queue_items, key=lambda item: item.priority_score, reverse=True)[
                :MAX_ACTIVE_UNCERTAINTIES
            ],
            resolved=[record.uncertainty_id for record in state.records if record.status == "resolved"],
            deprecated=[record.uncertainty_id for record in state.records if record.status == "deprecated"],
        )
        return state

    def _update_experiment_memory(
        self,
        memory: ExperimentMemoryState,
        *,
        protocol: ExperimentProtocol,
        result: ExperimentResult,
        evaluation: EvaluationResult,
        protocol_path: str | None,
        result_path: str | None,
        evaluation_path: str | None,
    ):
        now = datetime.now()
        entry_index = memory.entry_index()
        semantic = self._display_service()
        key_findings = [
            semantic.display_text(item.claim)
            for item in evaluation.scientific.evidence_summary.new_evidence_for
        ] + [
            semantic.display_text(item.claim)
            for item in evaluation.scientific.evidence_summary.new_evidence_against
        ]
        key_findings.append(
            semantic.display_text(
                _build_vsw_performance_summary(
                    target=protocol.target,
                    metrics=evaluation.metrics,
                    evaluation=evaluation,
                )
            )
        )

        entry = entry_index.get(protocol.experiment_id)
        if entry is None:
            entry = ExperimentMemoryEntry(
                experiment_id=protocol.experiment_id,
                round_id=protocol.round_id,
                experiment_type=protocol.experiment_type,
                model_families=list(dict.fromkeys(protocol.features.model_families or [])),
                status=result.status,
                tested_hypotheses=protocol.tested_hypotheses,
                target_uncertainties=protocol.target_uncertainties,
                protocol_path=self.repository.relativize(protocol_path),
                result_path=self.repository.relativize(result_path),
                evaluation_path=self.repository.relativize(evaluation_path),
                metrics_snapshot=evaluation.metrics,
                key_findings=key_findings,
                visualizations=[item.path for item in evaluation.visualizations],
                data_coverage=result.data_coverage,
                created_at=now,
                updated_at=now,
            )
            memory.entries.append(entry)
        else:
            entry.round_id = protocol.round_id
            entry.status = result.status
            entry.experiment_type = protocol.experiment_type
            entry.model_families = list(dict.fromkeys(protocol.features.model_families or entry.model_families))
            entry.tested_hypotheses = protocol.tested_hypotheses
            entry.target_uncertainties = protocol.target_uncertainties
            entry.protocol_path = self.repository.relativize(protocol_path)
            entry.result_path = self.repository.relativize(result_path)
            entry.evaluation_path = self.repository.relativize(evaluation_path)
            entry.metrics_snapshot = evaluation.metrics
            entry.key_findings = key_findings
            entry.visualizations = [item.path for item in evaluation.visualizations]
            entry.data_coverage = result.data_coverage
            entry.updated_at = now

        memory.current_round = max(memory.current_round, protocol.round_id)
        memory.last_updated = now
        return memory

    def _update_round_history(
        self,
        state: RoundHistoryState,
        *,
        protocol: ExperimentProtocol,
        evaluation: EvaluationResult | None,
        experiment_memory: ExperimentMemoryState,
        failure_reason: str | None,
    ) -> RoundHistoryState:
        now = datetime.now()
        entry = next((item for item in state.entries if item.round_id == protocol.round_id), None)
        if entry is None:
            entry = RoundHistoryEntry(round_id=protocol.round_id, created_at=now, updated_at=now)
            state.entries.append(entry)

        memory_entry = experiment_memory.entry_index().get(protocol.experiment_id)
        findings = list(memory_entry.key_findings[:5]) if memory_entry is not None else []
        tested_hypotheses = _unique_preserve_order(
            list(protocol.tested_hypotheses) + (list(memory_entry.tested_hypotheses) if memory_entry is not None else [])
        )
        target_uncertainties = _unique_preserve_order(
            list(protocol.target_uncertainties) + (list(memory_entry.target_uncertainties) if memory_entry is not None else [])
        )
        closure = [
            _closure_item("scientific_question", "科学问题已定义", True, protocol.scientific_objective or protocol.target),
            _closure_item(
                "hypothesis_generation",
                "竞争假设已生成",
                bool(tested_hypotheses),
                ",".join(tested_hypotheses[:3]) or "缺少 tested_hypotheses（协议与实验记忆均未写入）",
            ),
            _closure_item(
                "uncertainty_generation",
                "科学不确定性已生成",
                bool(target_uncertainties),
                ",".join(target_uncertainties[:3]) or "缺少 target_uncertainties（协议与实验记忆均未写入）",
            ),
            _closure_item("candidate_approval", "候选实验已审批", True, protocol.source_candidate_id or protocol.experiment_id),
            _closure_item(
                "experiment_execution",
                "实验执行已完成或已失败归因",
                failure_reason is not None or (memory_entry is not None and memory_entry.status in {"completed", "failed"}),
                failure_reason or (memory_entry.status if memory_entry is not None else "缺少实验记忆"),
            ),
            _closure_item(
                "result_feedback",
                "结果解释与反馈已写回",
                failure_reason is not None or bool(findings) or evaluation is not None,
                failure_reason or "已写入评价摘要",
            ),
        ]
        semantic = self._display_service()
        try:
            tree = self.repository.load_hypothesis_tree()
        except Exception:
            tree = None
        try:
            uncertainty_records = self.repository.load_uncertainties().records
        except Exception:
            uncertainty_records = []

        def _clean(text: str | None) -> str:
            return clean_llm_text(
                semantic.display_text(text or ""),
                tree=tree,
                uncertainty_records=uncertainty_records,
            )

        closure = [
            _closure_item(
                item.item_id,
                item.label,
                item.completed,
                _clean(item.detail),
            )
            for item in closure
        ]

        entry.status = "failed" if failure_reason else "completed"
        entry.source_experiment_id = protocol.experiment_id
        entry.approved_candidate_id = protocol.source_candidate_id or protocol.experiment_id
        entry.gating_ready = all(item.completed or not item.required for item in closure)
        entry.closure_checklist = closure
        entry.tested_hypotheses = list(tested_hypotheses[:8])
        entry.target_uncertainties = list(target_uncertainties[:8])
        entry.unresolved_uncertainties = list(target_uncertainties[:5])
        entry.highlighted_hypotheses = list(tested_hypotheses[:5])
        entry.scientific_findings = [
            _clean(str(item))
            for item in (findings or ([failure_reason] if failure_reason else []))
        ]
        entry.three_layer_conclusion = (
            evaluation.scientific.three_layer_conclusion.model_copy(deep=True)
            if evaluation is not None and evaluation.scientific.three_layer_conclusion is not None
            else None
        )
        entry.failure_reason = _clean(failure_reason) if failure_reason else None
        entry.metrics_snapshot = evaluation.metrics if evaluation is not None else None
        entry.updated_at = now

        state.current_round = max(state.current_round, protocol.round_id)
        state.last_updated = now
        state.entries.sort(key=lambda item: item.round_id)
        return state

    def _update_metrics_timeline(
        self,
        state: MetricsTimelineState,
        *,
        protocol: ExperimentProtocol,
        evaluation: EvaluationResult | None,
        experiment_memory: ExperimentMemoryState,
        execution_status: str,
    ) -> MetricsTimelineState:
        now = datetime.now()
        state.entries = [item for item in state.entries if item.round_id != protocol.round_id]
        support_scores = []
        for entry in experiment_memory.entries:
            if entry.round_id != protocol.round_id:
                continue
            if entry.metrics_snapshot is None:
                continue
            support_scores.append(entry.metrics_snapshot.delta.pearson_r or 0.0)

        unresolved_count = len(protocol.target_uncertainties)
        metrics = evaluation.metrics if evaluation is not None else None
        state.entries.append(
            MetricsTimelineEntry(
                round_id=protocol.round_id,
                experiment_id=protocol.experiment_id,
                execution_status=execution_status,
                baseline_rmse=metrics.baseline_rmse if metrics else None,
                treatment_rmse=metrics.treatment_rmse if metrics else None,
                baseline_pearson_r=metrics.baseline_pearson_r if metrics else None,
                treatment_pearson_r=metrics.treatment_pearson_r if metrics else None,
                delta_rmse=metrics.delta.rmse if metrics else None,
                delta_pearson_r=metrics.delta.pearson_r if metrics else None,
                stable=evaluation.robustness.overall.stable if evaluation is not None else False,
                support_mean=round(sum(support_scores) / len(support_scores), 4) if support_scores else None,
                support_max=round(max(support_scores), 4) if support_scores else None,
                unresolved_uncertainty_count=unresolved_count,
                recorded_at=now,
            )
        )
        state.current_round = max(state.current_round, protocol.round_id)
        state.last_updated = now
        state.entries.sort(key=lambda item: item.round_id)
        return state

    def _update_process_state(self, process_state: ProcessState, round_id: int) -> ProcessState:
        process_state.current_round = max(process_state.current_round, round_id)
        process_state.current_stage = "knowledge_memory_updated"
        process_state.current_phase = "decision_making"
        process_state.current_step = "round_review"
        process_state.progress_percentage = 75
        process_state.steps["execution"] = ProcessStep(name="execution", status="completed")
        process_state.steps["evaluation"] = ProcessStep(name="evaluation", status="completed")
        process_state.steps["state_writeback"] = ProcessStep(name="state_writeback", status="completed")
        process_state.phases["experiment_execution"] = ProcessPhase(status="completed")
        process_state.phases["result_analysis"] = ProcessPhase(status="completed")
        process_state.phases["decision_making"] = ProcessPhase(
            status="pending",
            steps={
                "round_review": ProcessStep(
                    name="round_review",
                    status="pending",
                    requires_user_approval=True,
                )
            },
        )
        return process_state


def _normalize_uncertainty_item(item: dict[str, Any], round_id: int, offset: int) -> dict[str, Any]:
    question = item.get("question") or item.get("description") or f"round_{round_id}_uncertainty_{offset}"
    return {
        "uncertainty_id": item.get("uncertainty_id") or item.get("id") or f"U_R{round_id:02d}_{offset:02d}",
        "question": question,
        "description": item.get("description") or question,
        "priority": item.get("priority", "medium"),
    }


def _closure_item(item_id: str, label: str, completed: bool, detail: str | None) -> ClosureChecklistItem:
    return ClosureChecklistItem(
        item_id=item_id,
        label=label,
        completed=completed,
        required=True,
        detail=detail,
    )


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _content_digest(*models: object) -> str:
    hasher = hashlib.sha256()
    for model in models:
        if model is None:
            continue
        if hasattr(model, "model_dump"):
            payload = model.model_dump(mode="json", exclude_none=True)
        else:
            payload = model
        hasher.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        )
        hasher.update(b"|")
    return hasher.hexdigest()[:16]


def _resolve_evidence_version(
    protocol: ExperimentProtocol,
    result: ExperimentResult | None = None,
) -> str:
    # 协议版本是协议内容的一部分，已经包含在内容摘要中；这里只用内容摘要定位版本，
    # 避免默认 protocol_version="1.0" 把所有修正都当成同一版证据。
    if result is not None and result.run_id:
        return f"{result.run_id}:{_content_digest(protocol, result)}"
    return _content_digest(protocol, result)


def _scientific_evidence_identity(
    *,
    protocol: ExperimentProtocol,
    result: ExperimentResult | None,
    evaluation: EvaluationResult,
    hypothesis_id: str,
) -> str:
    """Per-hypothesis identity of the numeric evidence, excluding audit text."""
    assessment = next(
        (
            item
            for item in evaluation.scientific.hypothesis_assessments
            if item.hypothesis_id == hypothesis_id
        ),
        None,
    )
    prediction = protocol.hypothesis_predictions.get(hypothesis_id)
    robustness = result.robustness if result is not None else None
    payload: dict[str, Any] = {
        "hypothesis_id": hypothesis_id,
        "experiment_type": protocol.experiment_type,
        "target": protocol.target,
        "features": protocol.features.model_dump(mode="json", exclude_none=True),
        "forecast_horizon_days": protocol.forecast_horizon_days,
        "prediction": (
            prediction.model_dump(mode="json", exclude_none=True)
            if prediction is not None
            else None
        ),
        "execution_status": None if result is None else result.status,
        "validity_status": None if result is None else result.validity_status,
        "comparison": (
            result.comparison.model_dump(mode="json", exclude_none=True)
            if result is not None
            else None
        ),
        "robustness": (
            {
                "bootstrap": (
                    robustness.bootstrap.model_dump(mode="json", exclude_none=True)
                    if robustness.bootstrap is not None
                    else None
                ),
                "time_slice": (
                    robustness.time_slice.model_dump(mode="json", exclude_none=True)
                    if robustness.time_slice is not None
                    else None
                ),
                "overall_stable": (
                    robustness.overall.stable if robustness.overall is not None else None
                ),
            }
            if robustness is not None
            else None
        ),
        "assessment": (
            {
                "metric": assessment.metric,
                "observed_metric_delta": assessment.observed_metric_delta,
                "metric_ci": assessment.metric_ci,
                "evidence_classification": assessment.evidence_classification,
                "updated": assessment.updated,
            }
            if assessment is not None
            else None
        ),
    }
    return _content_digest(payload)


def _record_key(
    experiment_id: str,
    slot: str,
    run_id: str | None = None,
    contrast_id: str | None = None,
) -> str:
    return f"{experiment_id}:{run_id or '-'}:{contrast_id or '-'}:{slot}"


def _replace_or_append(records: list, evidence_key: str, new_record) -> bool:
    """Append a record or replace its previous evidence version, in place.

    Returns True when the record was written; False when an unchanged version
    of the same evidence was already present.
    """
    if not evidence_key:
        records.append(new_record)
        return True
    existing_index = next(
        (
            index
            for index, record in enumerate(records)
            if getattr(record, "evidence_key", None) == evidence_key
        ),
        None,
    )
    if existing_index is None:
        records.append(new_record)
        return True
    previous = records[existing_index]
    if getattr(previous, "evidence_version", None) == getattr(new_record, "evidence_version", None):
        return False
    records[existing_index] = new_record
    return True


def _append_unique_text(items: list[str], text: str | None) -> None:
    normalized = (text or "").strip()
    if normalized and normalized not in items:
        items.append(normalized)


def _infer_tested_hypotheses(
    *,
    protocol: ExperimentProtocol,
    evaluation: EvaluationResult | None,
    tree: HypothesisTreeState,
    uncertainties: UncertaintyState,
) -> list[str]:
    inferred = []
    uncertainty_index = uncertainties.record_index()
    for uncertainty_id in protocol.target_uncertainties:
        record = uncertainty_index.get(uncertainty_id)
        if record is not None:
            inferred.extend(
                hypothesis_id
                for hypothesis_id in record.related_hypotheses
                if hypothesis_id
            )
    if evaluation is not None:
        inferred.extend(item.hypothesis_id for item in evaluation.scientific.hypothesis_assessments if item.hypothesis_id)
        inferred.extend(
            item.leading_hypothesis_id
            for item in evaluation.scientific.disagreement_updates
            if getattr(item, "leading_hypothesis_id", None)
        )
    if protocol.source_candidate_id:
        inferred.extend(
            node.hypothesis_id
            for node in tree.nodes
            if protocol.source_candidate_id in [e.source for e in node.evidence_items]
        )
    if not inferred:
        inferred.extend(tree.active_hypotheses[:3])
    return _unique_preserve_order(inferred)


def _infer_target_uncertainties(
    *,
    protocol: ExperimentProtocol,
    evaluation: EvaluationResult | None,
    uncertainties: UncertaintyState,
) -> list[str]:
    inferred = []
    if evaluation is not None:
        inferred.extend(item.uncertainty_id for item in evaluation.scientific.disagreement_updates if item.uncertainty_id)
        inferred.extend(
            _normalize_uncertainty_item(item, protocol.round_id, index).get("uncertainty_id")
            for index, item in enumerate(evaluation.scientific.evidence_summary.remaining_uncertainties, start=1)
        )
    if not inferred:
        inferred.extend(item.uncertainty_id for item in uncertainties.records if item.status not in {"resolved", "deprecated"})
    return _unique_preserve_order(inferred)


def _build_vsw_performance_summary(
    *,
    target: str,
    metrics: PerformanceMetrics,
    evaluation: EvaluationResult | None = None,
) -> str:
    target_label = target or "目标变量"
    if evaluation is not None and evaluation.metrics.experiment_type == "model_sensitivity":
        conclusion = evaluation.scientific.three_layer_conclusion
        families = (
            conclusion.experiment_layer.model_sensitivity_families
            if conclusion is not None and conclusion.experiment_layer is not None
            else []
        )
        if families:
            parts = []
            for item in families:
                point = item.point_gain_rmse
                point_text = "--" if point is None else f"{point:.4f}"
                ci = item.rmse_ci
                ci_text = (
                    f"CI=[{ci[0]:.4f}, {ci[1]:.4f}]"
                    if ci and len(ci) >= 2
                    else "CI 不可用"
                )
                parts.append(f"{item.family} ΔRMSE={point_text}（{ci_text}）")
            return (
                f"{target_label} 预测效果专项分析：模型敏感性实验共 {len(metrics.runs)} 组运行、"
                f"{len(metrics.contrasts)} 个家族内对比；" + "；".join(parts)
                + "。CI 跨零时按保守规则判定为未检出稳定增益，不自动改变假设支持度。"
            )
    baseline_r = metrics.baseline_pearson_r
    treatment_r = metrics.treatment_pearson_r
    delta_r = metrics.delta.pearson_r
    delta_rmse = metrics.delta.rmse
    trend = "提升" if (delta_r or 0.0) > 0 else "下降" if (delta_r or 0.0) < 0 else "持平"
    return (
        f"{target_label} 预测效果专项分析：Pearson r 从 {baseline_r if baseline_r is not None else '--'} "
        f"变化到 {treatment_r if treatment_r is not None else '--'}，整体{trend}；"
        f"ΔPearson r={delta_r if delta_r is not None else '--'}，"
        f"ΔRMSE={delta_rmse if delta_rmse is not None else '--'}。"
    )


def _priority_to_score(priority: str) -> float:
    mapping = {"low": 0.3, "medium": 0.6, "high": 0.9}
    return mapping.get(priority, 0.6)


def _classify_support_signal(evaluation: EvaluationResult) -> str:
    delta = evaluation.metrics.delta
    rmse_delta = delta.rmse
    pearson_delta = delta.pearson_r
    if rmse_delta is None:
        return "inconclusive"
    bootstrap = evaluation.robustness.bootstrap if evaluation.robustness is not None else None
    ci = (
        bootstrap.rmse_ci
        if bootstrap is not None
        and bootstrap.rmse_ci
        and len(bootstrap.rmse_ci) >= 2
        else None
    )
    if ci:
        lower, upper = ci
        if lower > 0 and (pearson_delta is None or pearson_delta >= 0):
            return "stable_improvement"
        if upper < 0 and (pearson_delta is None or pearson_delta <= 0):
            return "stable_degradation"
        return "inconclusive"
    if pearson_delta is not None and (
        (rmse_delta > 0 and pearson_delta < 0)
        or (rmse_delta < 0 and pearson_delta > 0)
    ):
        return "inconclusive"
    return "mixed"


def _apply_support_shift(current_score: float, signal: str) -> float:
    # 无结构化判定时只允许 bootstrap CI 支撑的稳定方向做小幅度移动。
    shift = {
        "stable_improvement": 0.03,
        "stable_degradation": -0.03,
        "mixed": 0.0,
        "inconclusive": 0.0,
    }.get(signal, 0.0)
    return round(min(max(current_score + shift, 0.0), 1.0), 4)


def _rebuild_tree_summary(tree: HypothesisTreeState) -> TreeSummary:
    return TreeSummary(
        total_nodes=len(tree.nodes),
        active_count=sum(1 for node in tree.nodes if node.status in {"active", "converged"}),
        pruned_count=sum(1 for node in tree.nodes if node.status == "pruned"),
        pending_count=sum(1 for node in tree.nodes if node.status in {"draft", "pending"}),
    )


def _minimum_active_hypotheses(protocol: ExperimentProtocol) -> int:
    if protocol.budget and protocol.budget.max_candidates:
        return max(1, protocol.budget.max_candidates)
    return 3


def _propagate_tree_state(
    *,
    tree: HypothesisTreeState,
    round_id: int,
    min_active_hypotheses: int,
) -> None:
    node_index = tree.node_index()

    for node in tree.nodes:
        if node.status != "pruned" or not node.children_ids:
            continue
        for child_id in node.children_ids:
            child = node_index.get(child_id)
            if child is None or child.status == "pruned":
                continue
            child.status = "pruned"
            child.pruned_at_round = round_id
            child.prune_reason = "父假设剪枝，继承剪枝"
            child.updated_at_round = round_id

    changed = True
    while changed:
        changed = False

        for node in tree.nodes:
            if node.parent_id is None:
                continue
            parent = node_index.get(node.parent_id)
            if parent is None:
                continue
            if parent.status == "pruned" and node.status != "pruned":
                node.status = "pruned"
                node.pruned_at_round = round_id
                node.prune_reason = "父假设剪枝，继承剪枝"
                node.updated_at_round = round_id
                changed = True
                continue
            if node.status == PENDING and parent.support_score >= ACTIVATION_THRESHOLD:
                # 父假设达标只解锁参与资格，子节点仍按自身支持度定状态。
                node.status = (
                    ACTIVE if node.support_score >= ACTIVATION_THRESHOLD else OBSERVING
                )
                node.activated_at_round = node.activated_at_round or round_id
                node.updated_at_round = round_id
                if not any(item.round == round_id and item.event == "parent_support_activation" for item in node.support_history):
                    node.support_history.append(
                        SupportHistoryEntry(
                            round=round_id,
                            score=node.support_score,
                            event="parent_support_activation",
                        )
                    )
                changed = True


def _support_after_from_assessment(
    *,
    before: float,
    assessment,
    prediction,
    observed_delta: float,
) -> float:
    classification = getattr(assessment, "evidence_classification", None) or EVIDENCE_NO_STABLE_GAIN
    if classification in {
        EVIDENCE_INCONCLUSIVE,
        EVIDENCE_NO_STABLE_GAIN,
        EVIDENCE_IDENTICAL_PREDICTIONS,
    }:
        return round(min(max(before, 0.05), 0.95), 4)
    assessment_metric = _normalize_assessment_metric(
        getattr(assessment, "metric", None)
    )
    prediction_metric = (
        _normalize_assessment_metric(getattr(prediction, "metric", None))
        if prediction is not None
        else None
    )
    metric_delta = getattr(assessment, "observed_metric_delta", None)
    expected_range = (
        getattr(prediction, "expected_range", None)
        if prediction is not None
        else None
    )
    if assessment_metric is None or prediction_metric != assessment_metric:
        # metric 缺失或指标不一致时，不拿 RMSE 增量匹配预期区间。
        expected_range = None
        metric_delta = None
    elif metric_delta is None and assessment_metric == "rmse":
        # 兼容旧评估记录：RMSE 主证据缺失时沿用全局 RMSE 增量。
        metric_delta = observed_delta
    elif metric_delta is None:
        expected_range = None
    outcome = compute_support_update_from_evidence(
        current_support=before,
        evidence_classification=classification,
        expected_effect=getattr(prediction, "expected_effect", None)
        if prediction is not None
        else None,
        expected_range=expected_range,
        observed_delta=metric_delta,
    )
    return outcome.support_after


def _normalize_assessment_metric(metric: str | None) -> str | None:
    if metric is None:
        return None
    normalized = str(metric).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "rmse": "rmse",
        "root_mean_squared_error": "rmse",
        "mae": "mae",
        "mean_absolute_error": "mae",
        "pearson_r": "pearson_r",
        "pearson": "pearson_r",
        "r": "pearson_r",
        "r2": "r2",
        "r_squared": "r2",
        "skill": "skill",
        "sst": "skill",
    }
    return aliases.get(normalized)


def _observed_delta_from_evaluation(evaluation: EvaluationResult) -> float:
    # 统一以 RMSE 增量为主指标，避免报告主指标与支持度更新脱节。
    if evaluation.metrics.delta.rmse is not None:
        return evaluation.metrics.delta.rmse
    if evaluation.metrics.delta.skill is not None:
        return evaluation.metrics.delta.skill
    if evaluation.metrics.delta.pearson_r is not None:
        return evaluation.metrics.delta.pearson_r
    return 0.0
