from __future__ import annotations

from datetime import datetime

from core.state_repository import UnifiedStateRepository
from core.hypothesis_state_machine import ACTIVATION_THRESHOLD, previous_round_score
from core.support_update_rules import (
    compute_support_update_from_reasoning,
    priority_after_action,
    priority_to_score,
)
from core.decision_unified import recompute_candidate_plan_metadata
from core.unified_schema import (
    CritiqueRecord,
    DecisionEntry,
    EvidenceItem,
    ExperimentMemoryEntry,
    LatestTreeUpdate,
    MAX_ACTIVE_UNCERTAINTIES,
    PrioritizedUncertainty,
    ReasoningPlannerOutput,
    ReasoningTraceEntry,
    SupportHistoryEntry,
    TreeSummary,
    UncertaintyHistoryEntry,
    UncertaintyPriorityQueue,
)


class PlannerOutputApplier:
    """Apply planner_output back into candidate/state/log artifacts."""

    def __init__(self, repository: UnifiedStateRepository) -> None:
        self.repository = repository

    def apply(self, planner_output: ReasoningPlannerOutput) -> dict[str, object]:
        task = self.repository.load_task()
        data_dictionary = self.repository.load_data_dictionary()
        candidate_set = self.repository.load_candidate_experiments()
        tree = self.repository.load_hypothesis_tree()
        uncertainties = self.repository.load_uncertainties()
        experiment_memory = self.repository.load_experiment_memory()
        decision_log = self.repository.load_decision_log()

        applied_candidates = self._apply_candidate_supplements(candidate_set, planner_output)
        if applied_candidates:
            candidate_set = recompute_candidate_plan_metadata(
                candidate_set,
                task=task,
                data_dictionary=data_dictionary,
                hypothesis_tree=tree,
                uncertainty_state=uncertainties,
                experiment_memory=experiment_memory,
                current_round=planner_output.next_round_id,
            )
        hypothesis_updates, uncertainty_updates, trace_updates = self._apply_interpretation_enhancements(
            tree,
            uncertainties,
            experiment_memory,
            planner_output,
            min_active_hypotheses=task.payload.constraints.min_active_hypotheses or 3,
        )

        decision_log.decisions.append(
            DecisionEntry(
                decision_id=f"D{len(decision_log.decisions) + 1:03d}",
                timestamp=datetime.now(),
                round_id=planner_output.source_round_id,
                phase="next_round",
                step="planner_output_applied",
                decision_type="planner_output_applied",
                made_by="system",
                summary=f"已将 planner_output 应用到候选实验与知识记忆层（round {planner_output.next_round_id}）",
                details={
                    "candidate_updates": applied_candidates,
                    "hypothesis_updates": hypothesis_updates,
                    "uncertainty_updates": uncertainty_updates,
                    "trace_updates": trace_updates,
                },
            )
        )

        self.repository.save_candidate_experiments(candidate_set)
        self.repository.save_hypothesis_tree(tree)
        self.repository.save_uncertainties(uncertainties)
        self.repository.save_experiment_memory(experiment_memory)
        self.repository.save_decision_log(decision_log)
        return {
            "candidate_experiments": candidate_set,
            "hypothesis_tree": tree,
            "uncertainties": uncertainties,
            "experiment_memory": experiment_memory,
            "decision_log": decision_log,
            "applied_candidates": applied_candidates,
            "hypothesis_updates": hypothesis_updates,
            "uncertainty_updates": uncertainty_updates,
            "trace_updates": trace_updates,
        }

    def _apply_candidate_supplements(self, candidate_set, planner_output: ReasoningPlannerOutput) -> int:
        existing = {candidate.experiment_id: candidate for candidate in candidate_set.candidates}
        applied = 0
        for supplement in planner_output.candidate_supplements:
            proposal = supplement.candidate.model_copy(deep=True)
            note = f"planner_candidate_supplement:{supplement.proposal_id}"
            rationale_note = f"planner_rationale:{supplement.rationale}"
            if proposal.experiment_id in existing:
                candidate = existing[proposal.experiment_id]
                llm_refined = any(
                    note.startswith("llm_controller_") for note in proposal.design.notes
                )
                if llm_refined:
                    candidate.design = proposal.design.model_copy(deep=True)
                for proposal_note in proposal.design.notes:
                    if proposal_note not in candidate.design.notes:
                        candidate.design.notes.append(proposal_note)
                if note not in candidate.design.notes:
                    candidate.design.notes.append(note)
                if rationale_note not in candidate.design.notes:
                    candidate.design.notes.append(rationale_note)
                if proposal.novelty:
                    candidate.novelty = proposal.novelty
                applied += 1
                continue

            if note not in proposal.design.notes:
                proposal.design.notes.append(note)
            if rationale_note not in proposal.design.notes:
                proposal.design.notes.append(rationale_note)
            candidate_set.candidates.append(proposal)
            existing[proposal.experiment_id] = proposal
            applied += 1

        candidate_set.current_count = len(candidate_set.candidates)
        return applied

    def _apply_interpretation_enhancements(
        self,
        tree,
        uncertainties,
        experiment_memory,
        planner_output: ReasoningPlannerOutput,
        min_active_hypotheses: int,
    ) -> tuple[int, int, int]:
        node_index = tree.node_index()
        uncertainty_index = uncertainties.record_index()
        hypothesis_updates = 0
        uncertainty_updates = 0
        trace_updates = 0
        trace_entry = _get_or_create_memory_entry(experiment_memory, planner_output)

        for enhancement in planner_output.interpretation_enhancements:
            target_node = (
                node_index.get(enhancement.target_hypothesis_id)
                if enhancement.target_hypothesis_id and enhancement.target_hypothesis_id in node_index
                else None
            )
            support_outcome = compute_support_update_from_reasoning(
                current_support=target_node.support_score if target_node is not None else 0.5,
                impact_direction=enhancement.impact_direction,
                impact_strength=enhancement.impact_strength,
                confidence=enhancement.confidence,
                previous_status=target_node.status if target_node is not None else None,
            support_history=target_node.support_history if target_node is not None else [],
            current_round=planner_output.next_round_id,
            parent_support=(
                node_index.get(target_node.parent_id).support_score
                if target_node is not None
                and target_node.parent_id
                and target_node.parent_id in node_index
                else None
            ),
        )
            if enhancement.target_hypothesis_id:
                node = node_index.get(enhancement.target_hypothesis_id)
                if node is not None:
                    node.support_score = support_outcome.support_after
                    node.status = support_outcome.status
                    node.evidence_items.append(
                        EvidenceItem(
                            id=enhancement.enhancement_id,
                            type="expert_judgment",
                            description=enhancement.interpretation,
                            source="planner_output",
                            added_at_round=planner_output.next_round_id,
                            support_weight=min(max(abs(support_outcome.support_delta), 0.03), 0.2),
                        )
                    )
                    node.critiques.append(
                        CritiqueRecord(
                            id=f"{enhancement.enhancement_id}_critique",
                            content=enhancement.interpretation,
                            from_role="scientific_interpreter",
                            added_at_round=planner_output.next_round_id,
                        )
                    )
                    if enhancement.suggested_state_note:
                        node.alternative_explanations.append(enhancement.suggested_state_note)
                    node.support_history.append(
                        SupportHistoryEntry(
                            round=planner_output.next_round_id,
                            score=support_outcome.support_after,
                            event=f"planner_interpretation:{enhancement.enhancement_id}",
                            evidence_source="advisory",
                        )
                    )
                    node.updated_at_round = planner_output.next_round_id
                    hypothesis_updates += 1

            total_priority_delta = 0.0
            for uncertainty_id in enhancement.related_uncertainties:
                record = uncertainty_index.get(uncertainty_id)
                if record is None:
                    continue
                previous_priority = record.priority
                record.priority = priority_after_action(
                    record.priority,
                    enhancement.uncertainty_priority_action,
                )
                record.history.append(
                    UncertaintyHistoryEntry(
                        round=planner_output.next_round_id,
                        event="planner_interpretation",
                        source="planner_output",
                        description=enhancement.interpretation,
                    )
                )
                addition = enhancement.interpretation
                if record.notes:
                    if addition not in record.notes:
                        record.notes = f"{record.notes} | {addition}"
                else:
                    record.notes = addition
                total_priority_delta += priority_to_score(record.priority) - priority_to_score(previous_priority)
                uncertainty_updates += 1

            if trace_entry is not None:
                trace_entry.reasoning_traces.append(
                    ReasoningTraceEntry(
                        trace_id=enhancement.enhancement_id,
                        round_id=planner_output.next_round_id,
                        source="planner_output",
                        stage="interpretation_enhancement",
                        summary=enhancement.interpretation,
                        related_hypotheses=[enhancement.target_hypothesis_id] if enhancement.target_hypothesis_id else [],
                        related_uncertainties=enhancement.related_uncertainties,
                        support_delta=support_outcome.support_delta if enhancement.target_hypothesis_id else None,
                        priority_delta=round(total_priority_delta, 4) if enhancement.related_uncertainties else None,
                    )
                )
                trace_entry.updated_at = datetime.now()
                trace_updates += 1

        _ensure_min_active_hypotheses(
            tree.nodes,
            min_required=min_active_hypotheses,
            current_round=planner_output.next_round_id,
        )
        tree.current_round = max(tree.current_round, planner_output.next_round_id)
        tree.active_hypotheses = [
            node.hypothesis_id for node in tree.nodes if node.status in {"active", "converged"}
        ]
        tree.pruned_hypotheses = [node.hypothesis_id for node in tree.nodes if node.status == "pruned"]
        tree.pending_hypotheses = [
            node.hypothesis_id for node in tree.nodes if node.status in {"draft", "pending"}
        ]
        tree.tree_summary = TreeSummary(
            total_nodes=len(tree.nodes),
            active_count=sum(1 for node in tree.nodes if node.status in {"active", "converged"}),
            pruned_count=sum(1 for node in tree.nodes if node.status == "pruned"),
            pending_count=sum(1 for node in tree.nodes if node.status in {"draft", "pending"}),
        )
        tree.latest_update = LatestTreeUpdate(
            round=planner_output.next_round_id,
            event="planner_output_applied",
            description="planner interpretation enhancements written back into knowledge memory",
        )
        uncertainties.current_round = max(uncertainties.current_round, planner_output.next_round_id)
        uncertainties.last_updated = datetime.now()
        uncertainties.priority_queue = _rebuild_priority_queue(uncertainties)
        return hypothesis_updates, uncertainty_updates, trace_updates


def _rebuild_priority_queue(uncertainties) -> UncertaintyPriorityQueue:
    queue = [
        PrioritizedUncertainty(
            uncertainty_id=record.uncertainty_id,
            question=record.question,
            priority_score=priority_to_score(record.priority),
            status=record.status,
            estimated_resolution_round=record.created_at_round + 1,
        )
        for record in uncertainties.records
        if record.status not in {"resolved", "deprecated"}
    ]
    queue.sort(key=lambda item: item.priority_score, reverse=True)
    seen_ids: set[str] = set()
    bounded: list[PrioritizedUncertainty] = []
    for item in queue:
        if item.uncertainty_id in seen_ids:
            continue
        seen_ids.add(item.uncertainty_id)
        bounded.append(item)
        if len(bounded) >= MAX_ACTIVE_UNCERTAINTIES:
            break
    queue = bounded
    return UncertaintyPriorityQueue(
        last_updated=datetime.now(),
        current_round=uncertainties.current_round,
        queue=queue,
        resolved=[record.uncertainty_id for record in uncertainties.records if record.status == "resolved"],
        deprecated=[record.uncertainty_id for record in uncertainties.records if record.status == "deprecated"],
    )


def _ensure_min_active_hypotheses(
    nodes,
    *,
    min_required: int,
    current_round: int,
) -> None:
    active_count = sum(1 for node in nodes if node.status in {"active", "converged"})
    if active_count >= min_required:
        return

    node_index = {node.hypothesis_id: node for node in nodes}
    candidates = sorted(
        [
            node
            for node in nodes
            if _may_activate_by_rule(
                node,
                node_index=node_index,
                current_round=current_round,
            )
        ],
        key=lambda item: item.support_score,
        reverse=True,
    )
    for node in candidates:
        node.status = "active"
        node.activated_at_round = node.activated_at_round or current_round
        node.updated_at_round = current_round
        if not any(item.round == current_round and item.event == "min_active_fallback" for item in node.support_history):
            node.support_history.append(
                SupportHistoryEntry(
                    round=current_round,
                    score=node.support_score,
                    event="min_active_fallback",
                )
            )
        active_count += 1
        if active_count >= min_required:
            break


def _may_activate_by_rule(node, *, node_index, current_round: int) -> bool:
    if node.status == "pending":
        parent = node_index.get(node.parent_id) if node.parent_id else None
        return (
            parent is not None
            and parent.support_score >= ACTIVATION_THRESHOLD
            and node.support_score >= ACTIVATION_THRESHOLD
        )
    if node.status == "observing":
        previous = previous_round_score(
            node.support_history,
            current_round=current_round,
            current_score=node.support_score,
        )
        return (
            node.support_score >= ACTIVATION_THRESHOLD
            and (previous is None or node.support_score > previous)
        )
    return False


def _get_or_create_memory_entry(experiment_memory, planner_output: ReasoningPlannerOutput):
    if not planner_output.source_experiment_id:
        return None
    entry = experiment_memory.entry_index().get(planner_output.source_experiment_id)
    now = datetime.now()
    if entry is None:
        entry = ExperimentMemoryEntry(
            experiment_id=planner_output.source_experiment_id,
            round_id=planner_output.source_round_id,
            status="completed",
            created_at=now,
            updated_at=now,
            key_findings=["created from planner_output trace writeback"],
        )
        experiment_memory.entries.append(entry)
    experiment_memory.current_round = max(experiment_memory.current_round, planner_output.next_round_id)
    experiment_memory.last_updated = now
    return entry
