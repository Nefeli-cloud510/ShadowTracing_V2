from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.decision_unified import DecisionLayerService
from core.display_text_cleaner import clean_llm_text
from core.json_safe import json_dumps
from core.multi_source_uncertainty_miner import MIN_UNCERTAINTIES
from core.planner_output_applier import PlannerOutputApplier
from core.planner_unified import ReasoningPlannerInputBuilder
from core.state_updater import UnifiedStateUpdater
from core.unified_schema import (
    DataDictionary,
    EvaluationResult,
    ExperimentProtocol,
    HypothesisNode,
    LatestTreeUpdate,
    ProcessPhase,
    ProcessStep,
    RoundHistoryEntry,
)

if TYPE_CHECKING:
    from core.control_unified import HumanControlService


class RoundOrchestrator:
    """Archive the finished round, bootstrap the next round, then validate the iteration."""

    def __init__(self, control: Any) -> None:
        self.control = control
        self.repository = control.repository
        self.project_root = control.project_root

    def archive_round(self, round_id: int) -> dict[str, Any]:
        repo = self.repository
        snapshot_path = repo.round_snapshot_path(round_id)
        closure = self._round_closure_from_history(round_id)
        snapshot = {
            "schema_version": "1.0",
            "snapshot_id": f"round_{round_id:02d}",
            "round_id": round_id,
            "archived_at": datetime.now().isoformat(),
            "provenance": {
                "reason": "round_archive",
                "state_dir": repo.relativize(repo.state_dir),
            },
            "closure": closure,
            "protocol": self._read_json(self.project_root / "config" / "latest_protocol.json"),
            "evaluation_unified": self._read_json(
                self.project_root / "results" / f"round_{round_id:02d}" / "evaluation_unified.json"
            ),
            "result_unified": self._read_json(
                self.project_root / "results" / f"round_{round_id:02d}" / "result_unified.json"
            ),
            "artifacts": {
                "task": self._dump_model(repo.load_task),
                "data_dictionary": self._dump_model(repo.load_data_dictionary),
                "hypothesis_tree": self._dump_model(repo.load_hypothesis_tree),
                "uncertainties": self._dump_model(repo.load_uncertainties),
                "experiment_memory": self._dump_model(repo.load_experiment_memory),
                "round_history": self._dump_model(repo.load_round_history),
                "metrics_timeline": self._dump_model(repo.load_metrics_timeline),
                "failure_history": self._dump_model(repo.load_failure_history),
                "decision_log": self._dump_model(repo.load_decision_log),
                "candidate_experiments": self._dump_model(repo.load_candidate_experiments),
                "planner_input": self._dump_model(repo.load_planner_input),
                "planner_output": self._dump_model(repo.load_planner_output),
            },
        }
        repo.save_round_snapshot(round_id, snapshot)
        self._mark_snapshot_on_entry(round_id, snapshot_path)
        return snapshot

    def bootstrap_round(
        self,
        *,
        round_id: int,
        human_feedback: str | None,
        data_dictionary: DataDictionary | None,
        stop_at_hypothesis_confirmation: bool = False,
    ) -> dict[str, Any]:
        if data_dictionary is None:
            return {
                "planning_status": "awaiting_data_dictionary",
                "message": "需要提供 DataDictionary 后才能自动生成下一轮候选实验。",
            }

        snapshot = self.archive_round(round_id)
        if stop_at_hypothesis_confirmation:
            result = self._prepare_next_round_hypotheses(
                round_id=round_id,
                human_feedback=human_feedback,
                data_dictionary=data_dictionary,
                snapshot=snapshot,
            )
            return result
        result = self._build_next_round_plan(
            round_id=round_id,
            human_feedback=human_feedback,
            data_dictionary=data_dictionary,
            snapshot=snapshot,
        )
        return result

    def continue_next_round_planning(
        self,
        *,
        human_notes: str | None = None,
        nodes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Phase 3 of next-round bootstrap after scientific questioning completes."""
        repo = self.repository
        control = self.control
        process_state = repo.load_process_state()
        if process_state.current_stage != "awaiting_uncertainty_identification":
            raise ValueError("当前不在等待不确定性识别状态，无法继续下一轮规划。")

        previous_round_id = max(int(process_state.current_round or 1) - 1, 0)
        data_dictionary = repo.load_data_dictionary()
        task = repo.load_task()
        planner_input = repo.load_planner_input()
        uncertainties = repo.load_uncertainties()

        if control.planner_mode == "llm":
            planner_input, uncertainties = control._augment_uncertainty_context_with_llm_services(
                planner_input=planner_input,
                uncertainties=uncertainties,
            )
        else:
            planner_input, uncertainties = control._mine_programmatic_uncertainties(
                planner_input=planner_input,
                uncertainties=uncertainties,
            )
        repo.save_uncertainties(uncertainties)
        repo.save_planner_input(planner_input)

        plan = DecisionLayerService(
            repo,
            experiment_designer=control.experiment_designer,
            experiment_writer=control.experiment_writer,
        ).build_candidate_plan(
            task=task,
            data_dictionary=data_dictionary,
            planner_input=planner_input,
            target_round=planner_input.next_round_id,
        )
        uncertainties = repo.load_uncertainties()
        uncertainties.current_round = max(uncertainties.current_round, planner_input.next_round_id)
        repo.save_uncertainties(uncertainties)
        planner_output = control.planner_output_builder.build(
            planner_input=planner_input,
            candidate_plan=plan,
        )
        repo.save_planner_output(planner_output)
        control._log_planner_output_generated(planner_output)
        applied_output = PlannerOutputApplier(repo).apply(planner_output)
        review = control.request_experiment_selection_review()

        process_state = repo.load_process_state()
        process_state.current_round = max(
            process_state.current_round,
            planner_input.next_round_id,
        )
        process_state.current_phase = "experiment_planning"
        process_state.current_step = "step_4"
        process_state.current_stage = "awaiting_human_approval"
        process_state.progress_percentage = 45
        repo.save_process_state(process_state)

        candidate_set = repo.load_candidate_experiments()
        new_tree = repo.load_hypothesis_tree()
        new_uncertainties = repo.load_uncertainties()
        snapshot = self._load_snapshot_safely(previous_round_id)
        validations = self._validate_iteration(
            previous_round_id=previous_round_id,
            snapshot=snapshot,
            planner_input=planner_input,
            process_state=process_state,
            candidate_set=candidate_set,
            tree=new_tree,
            uncertainties=new_uncertainties,
        )
        input_sources = self._build_input_sources(snapshot=snapshot, human_feedback=planner_input.human_feedback)
        self._ensure_next_round_history(
            previous_round_id=previous_round_id,
            source_experiment_id=planner_input.source_experiment_id,
            validations=validations,
            input_sources=input_sources,
        )

        return {
            "planning_status": "candidate_plan_rebuilt",
            "status": review.get("status", "awaiting_human_review"),
            "message": review.get("message", "下一轮候选实验已生成，等待人工审批。"),
            "planner_input": planner_input,
            "planner_output": planner_output,
            "applied_output": applied_output,
            "candidate_plan": candidate_set,
            "next_review": review,
            "round_bootstrap": {
                "source_round_id": previous_round_id,
                "next_round_id": planner_input.next_round_id,
                "snapshot_path": repo.relativize(repo.round_snapshot_path(previous_round_id)),
                "validations": validations,
                "iteration_effective": all(item.get("passed") for item in validations),
            },
            "iteration_input_sources": input_sources,
        }

    def freeze_next_round_hypothesis_tree(
        self,
        *,
        human_notes: str | None = None,
        nodes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Freeze the reviewed next-round tree and wait for scientific questioning."""
        repo = self.repository
        control = self.control
        process_state = repo.load_process_state()
        if process_state.current_stage != "awaiting_hypothesis_confirmation":
            raise ValueError("当前不在下一轮假设树待确认状态，无法冻结假设树。")

        data_dictionary = repo.load_data_dictionary()
        task = repo.load_task()
        planner_input = repo.load_planner_input()
        if nodes:
            validated_nodes: list[HypothesisNode] = []
            for raw_node in nodes:
                try:
                    validated_nodes.append(HypothesisNode.model_validate(raw_node))
                except Exception:
                    continue
            if not validated_nodes:
                raise ValueError("下一轮假设树修改载荷无法通过校验，请刷新页面后重试。")
            tree = control.hypothesis_generator.rebuild_tree(
                task=task,
                nodes=validated_nodes,
                current_round=int(process_state.current_round or 1),
                description="下一轮假设树经人工 PI 修改后冻结",
                data_dictionary=data_dictionary,
            )
            repo.save_hypothesis_tree(tree)
            repo.save_pre_questioning_tree(tree)
        else:
            tree = repo.load_hypothesis_tree()
            tree.latest_update = LatestTreeUpdate(
                round=max(int(tree.current_round or 0), int(process_state.current_round or 1)),
                event="hypothesis_tree_confirmed",
                description="人工确认下一轮假设树，冻结后先进入科学质询。",
            )
            repo.save_hypothesis_tree(tree)
            repo.save_pre_questioning_tree(tree)
        if human_notes:
            note = f"human_tree_notes:{human_notes}"
            if note not in planner_input.planner_guidance:
                planner_input.planner_guidance.append(note)
        repo.save_planner_input(planner_input)
        process_state = repo.load_process_state()
        process_state.current_stage = "awaiting_scientific_questioning"
        process_state.current_step = "scientific_questioning"
        process_state.progress_percentage = 62
        repo.save_process_state(process_state)
        return {
            "planning_status": "hypothesis_tree_frozen",
            "status": "awaiting_scientific_questioning",
            "message": "下一轮假设树已冻结，等待开始科学质询。",
        }

    def run_next_round_scientific_questioning(self) -> dict[str, Any]:
        """Run LLM scientific questioning against the frozen next-round tree."""
        return self.control._run_first_round_scientific_questioning()

    def _prepare_next_round_hypotheses(
        self,
        *,
        round_id: int,
        human_feedback: str | None,
        data_dictionary: DataDictionary,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Phase 1 of next-round bootstrap: planner input + H/C + frozen tree, then stop."""
        control = self.control
        repo = self.repository
        task = repo.load_task()
        uncertainties = repo.load_uncertainties()
        uncertainties.current_round = max(uncertainties.current_round, round_id)

        planner_input_builder = ReasoningPlannerInputBuilder(repo)
        planner_input = planner_input_builder.build(
            data_dictionary=data_dictionary,
            source_round_id=round_id,
            human_feedback=human_feedback,
        )
        if control.planner_mode == "llm":
            planner_input = control._augment_hypothesis_context_with_llm_services(
                planner_input=planner_input,
                uncertainties=uncertainties,
            )

        generation_result = control.hypothesis_generator.build_tree(
            task=task,
            data_dictionary=data_dictionary,
            uncertainties=uncertainties,
            planner_input=planner_input,
            existing_tree=repo.load_hypothesis_tree(),
            current_round=round_id + 1,
        )
        uncertainties.records = list(generation_result.updated_uncertainties)
        uncertainties.current_round = max(uncertainties.current_round, round_id + 1)
        repo.save_uncertainties(uncertainties)

        tree = generation_result.tree
        if tree.latest_update is None or tree.latest_update.round < round_id + 1:
            tree.latest_update = LatestTreeUpdate(
                round=round_id + 1,
                event="hypothesis_review_ready",
                description="下一轮竞争假设已基于上一轮实验结果与反馈生成，等待人工确认。",
            )
        repo.save_hypothesis_tree(tree)
        if generation_result.generated_node_ids:
            control._log_hypothesis_generation(
                round_id=round_id,
                next_round_id=round_id + 1,
                generation_result=generation_result,
                mode=generation_result.mode,
            )

        repo.save_planner_input(planner_input)
        control._log_planner_input_prepared(planner_input)

        process_state = repo.load_process_state()
        process_state.current_round = max(process_state.current_round, round_id + 1)
        process_state.current_phase = "next_round"
        process_state.current_step = "hypothesis_review"
        process_state.current_stage = "awaiting_hypothesis_confirmation"
        process_state.progress_percentage = 60
        process_state.phases["next_round"] = ProcessPhase(
            status="in_progress",
            started_at=datetime.now(),
            notes="已生成下一轮假设树，等待人工确认。",
            steps={
                "planner_input_prepared": ProcessStep(
                    name="planner_input_prepared",
                    status="completed",
                    requires_user_approval=False,
                    notes=human_feedback or "reuse current planning policy",
                ),
                "hypothesis_review": ProcessStep(
                    name="假设树确认",
                    status="in_progress",
                    requires_user_approval=True,
                    notes="H/C 生成完成，确认后进入不确定性识别。",
                ),
            },
        )
        repo.save_process_state(process_state)

        return {
            "planning_status": "awaiting_hypothesis_confirmation",
            "status": "awaiting_hypothesis_confirmation",
            "message": "下一轮假设树已生成，等待人工确认后进入不确定性识别。",
            "hypothesis_generation": {
                "mode": generation_result.mode,
                "generated_node_ids": list(generation_result.generated_node_ids),
            },
            "planner_input": planner_input,
            "round_bootstrap": {
                "source_round_id": round_id,
                "next_round_id": round_id + 1,
                "snapshot_path": repo.relativize(repo.round_snapshot_path(round_id)),
                "iteration_effective": False,
            },
        }

    def _load_snapshot_safely(self, round_id: int) -> dict[str, Any]:
        try:
            return self.repository.load_round_snapshot(round_id)
        except Exception:
            return {}

    def repair_round_closure(self, round_id: int) -> dict[str, Any]:
        """Self-heal a round whose protocol/memory lost core hypothesis or uncertainty links."""
        repo = self.repository
        protocol_path = self.project_root / "config" / "latest_protocol.json"
        evaluation_path = self.project_root / "results" / f"round_{round_id:02d}" / "evaluation_unified.json"
        protocol = ExperimentProtocol.model_validate(json.loads(protocol_path.read_text(encoding="utf-8")))
        evaluation = (
            EvaluationResult.model_validate(json.loads(evaluation_path.read_text(encoding="utf-8")))
            if evaluation_path.exists()
            else None
        )
        tree = repo.load_hypothesis_tree()
        uncertainties = repo.load_uncertainties()
        memory = repo.load_experiment_memory()
        history = repo.load_round_history()

        updater = UnifiedStateUpdater(repo)
        protocol = updater._ensure_protocol_core_outputs(
            protocol=protocol,
            evaluation=evaluation,
            tree=tree,
            uncertainties=uncertainties,
        )
        protocol_payload = json_dumps(
            protocol.model_dump(mode="json", exclude_none=True),
            indent=2,
        )
        protocol_path.write_text(protocol_payload, encoding="utf-8")
        result_protocol = self.project_root / "results" / f"round_{round_id:02d}" / "protocol.json"
        if result_protocol.exists():
            result_protocol.write_text(protocol_payload, encoding="utf-8")

        memory_entry = memory.entry_index().get(protocol.experiment_id)
        if memory_entry is not None:
            memory_entry.tested_hypotheses = list(protocol.tested_hypotheses)
            memory_entry.target_uncertainties = list(protocol.target_uncertainties)
            memory_entry.updated_at = datetime.now()
            repo.save_experiment_memory(memory)

        history = updater._update_round_history(
            history,
            protocol=protocol,
            evaluation=evaluation,
            experiment_memory=memory,
            failure_reason=None,
        )
        repo.save_round_history(history)
        snapshot = self.archive_round(round_id)
        return {
            "round_id": round_id,
            "gating_ready": next(
                (entry.gating_ready for entry in history.entries if entry.round_id == round_id),
                False,
            ),
            "tested_hypotheses": list(protocol.tested_hypotheses),
            "target_uncertainties": list(protocol.target_uncertainties),
            "snapshot_path": repo.relativize(repo.round_snapshot_path(round_id)),
            "snapshot_archived": bool(snapshot),
        }

    def _build_next_round_plan(
        self,
        *,
        round_id: int,
        human_feedback: str | None,
        data_dictionary: DataDictionary,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        control = self.control
        repo = self.repository
        task = repo.load_task()
        uncertainties = repo.load_uncertainties()
        uncertainties.current_round = max(uncertainties.current_round, round_id)

        planner_input_builder = ReasoningPlannerInputBuilder(repo)
        planner_input = planner_input_builder.build(
            data_dictionary=data_dictionary,
            source_round_id=round_id,
            human_feedback=human_feedback,
        )
        if control.planner_mode == "llm":
            planner_input, uncertainties = control._augment_next_round_context_with_llm_services(
                planner_input=planner_input,
                uncertainties=uncertainties,
            )
        else:
            planner_input, uncertainties = control._mine_programmatic_uncertainties(
                planner_input=planner_input,
                uncertainties=uncertainties,
            )
        generation_result = control.hypothesis_generator.build_tree(
            task=task,
            data_dictionary=data_dictionary,
            uncertainties=uncertainties,
            planner_input=planner_input,
            existing_tree=repo.load_hypothesis_tree(),
            current_round=round_id + 1,
        )
        uncertainties.records = list(generation_result.updated_uncertainties)
        uncertainties.current_round = round_id
        repo.save_uncertainties(uncertainties)

        tree = generation_result.tree
        if tree.latest_update is None or tree.latest_update.round < round_id + 1:
            tree.latest_update = LatestTreeUpdate(
                round=round_id + 1,
                event="iteration_bootstrap",
                description="本轮假设树已带入上一轮实验结果与反馈，作为下一轮规划输入。",
            )
        repo.save_hypothesis_tree(tree)

        if generation_result.generated_node_ids:
            control._log_hypothesis_generation(
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
            if control.planner_mode == "llm":
                planner_input, uncertainties = control._augment_next_round_context_with_llm_services(
                    planner_input=planner_input,
                    uncertainties=uncertainties,
                )
                repo.save_uncertainties(uncertainties)

        repo.save_planner_input(planner_input)
        control._log_planner_input_prepared(planner_input)

        process_state = repo.load_process_state()
        process_state.current_round = max(process_state.current_round, round_id)
        process_state.current_phase = "next_round"
        process_state.current_step = "planner_input_prepared"
        process_state.current_stage = "planner_input_ready"
        process_state.progress_percentage = 92
        process_state.phases["next_round"] = self.control._build_next_phase(process_state, round_id, human_feedback)
        repo.save_process_state(process_state)

        plan = DecisionLayerService(
            repo,
            experiment_designer=control.experiment_designer,
            experiment_writer=control.experiment_writer,
        ).build_candidate_plan(
            task=task,
            data_dictionary=data_dictionary,
            planner_input=planner_input,
            target_round=planner_input.next_round_id,
        )
        uncertainties = repo.load_uncertainties()
        uncertainties.current_round = max(uncertainties.current_round, round_id + 1)
        repo.save_uncertainties(uncertainties)
        planner_output = control.planner_output_builder.build(
            planner_input=planner_input,
            candidate_plan=plan,
        )
        repo.save_planner_output(planner_output)
        control._log_planner_output_generated(planner_output)
        applied_output = PlannerOutputApplier(repo).apply(planner_output)
        review = control.request_experiment_selection_review()

        process_state = repo.load_process_state()
        process_state.current_round = max(process_state.current_round, round_id + 1)
        process_state.current_phase = "experiment_planning"
        process_state.current_step = "step_4"
        process_state.current_stage = "awaiting_human_approval"
        process_state.progress_percentage = 45
        repo.save_process_state(process_state)

        candidate_set = repo.load_candidate_experiments()
        new_tree = repo.load_hypothesis_tree()
        new_uncertainties = repo.load_uncertainties()
        validations = self._validate_iteration(
            previous_round_id=round_id,
            snapshot=snapshot,
            planner_input=planner_input,
            process_state=process_state,
            candidate_set=candidate_set,
            tree=new_tree,
            uncertainties=new_uncertainties,
        )
        input_sources = self._build_input_sources(snapshot=snapshot, human_feedback=human_feedback)
        self._ensure_next_round_history(
            previous_round_id=round_id,
            source_experiment_id=planner_input.source_experiment_id,
            validations=validations,
            input_sources=input_sources,
        )

        return {
            "planning_status": "candidate_plan_rebuilt",
            "hypothesis_generation": {
                "mode": generation_result.mode,
                "generated_node_ids": list(generation_result.generated_node_ids),
            },
            "planner_input": planner_input,
            "planner_output": planner_output,
            "applied_output": applied_output,
            "candidate_plan": candidate_set,
            "next_review": review,
            "round_bootstrap": {
                "source_round_id": round_id,
                "next_round_id": round_id + 1,
                "snapshot_path": repo.relativize(repo.round_snapshot_path(round_id)),
                "validations": validations,
                "iteration_effective": all(item.get("passed") for item in validations),
            },
            "iteration_input_sources": input_sources,
        }

    def _round_closure_from_history(self, round_id: int) -> dict[str, Any]:
        try:
            history = self.repository.load_round_history()
        except Exception:
            return {}
        entry = next((item for item in history.entries if item.round_id == round_id), None)
        if entry is None:
            return {}
        return {
            "gating_ready": bool(entry.gating_ready),
            "tested_hypotheses": list(entry.tested_hypotheses),
            "target_uncertainties": list(entry.target_uncertainties),
            "unresolved_uncertainties": list(entry.unresolved_uncertainties),
            "highlighted_hypotheses": list(entry.highlighted_hypotheses),
            "scientific_findings": list(entry.scientific_findings),
            "failure_reason": entry.failure_reason,
            "metrics_snapshot": (
                entry.metrics_snapshot.model_dump(mode="json", exclude_none=True)
                if entry.metrics_snapshot is not None
                else None
            ),
        }

    def _mark_snapshot_on_entry(self, round_id: int, snapshot_path: Path) -> None:
        history = self.repository.load_round_history()
        entry = next((item for item in history.entries if item.round_id == round_id), None)
        if entry is None:
            return
        entry.round_snapshot_path = self.repository.relativize(snapshot_path)
        entry.updated_at = datetime.now()
        history.last_updated = entry.updated_at
        self.repository.save_round_history(history)

    def _ensure_next_round_history(
        self,
        *,
        previous_round_id: int,
        source_experiment_id: str | None,
        validations: list[dict[str, Any]],
        input_sources: list[str],
    ) -> None:
        round_id = previous_round_id + 1
        history = self.repository.load_round_history()
        entry = next((item for item in history.entries if item.round_id == round_id), None)
        now = datetime.now()
        if entry is None:
            entry = RoundHistoryEntry(
                round_id=round_id,
                status="in_progress",
                created_at=now,
                updated_at=now,
            )
            history.entries.append(entry)
        entry.source_experiment_id = source_experiment_id
        entry.iterative_validations = validations
        entry.iteration_input_sources = input_sources
        entry.updated_at = now
        history.current_round = max(history.current_round, round_id)
        history.last_updated = now
        self.repository.save_round_history(history)

    def _validate_iteration(
        self,
        *,
        previous_round_id: int,
        snapshot: dict[str, Any],
        planner_input: Any,
        process_state: Any,
        candidate_set: Any,
        tree: Any,
        uncertainties: Any,
    ) -> list[dict[str, Any]]:
        old_tree = (snapshot.get("artifacts") or {}).get("hypothesis_tree") or {}
        old_uncertainties = (snapshot.get("artifacts") or {}).get("uncertainties") or {}
        expected_round = previous_round_id + 1
        validations: list[dict[str, Any]] = []

        round_incremented = (
            process_state.current_round >= expected_round
            and (candidate_set.round or 0) >= expected_round
            and planner_input.next_round_id == expected_round
        )
        validations.append(
            {
                "item_id": "round_incremented",
                "label": "轮次编号已递增",
                "passed": bool(round_incremented),
                "expected": expected_round,
                "actual": {
                    "process_state": process_state.current_round,
                    "candidate_round": candidate_set.round,
                    "planner_next_round": planner_input.next_round_id,
                },
                "detail": "上一轮归档后已进入新一轮轮次上下文。",
            }
        )

        tree_refreshed = self._tree_fingerprint(old_tree) != self._tree_fingerprint(
            tree.model_dump(mode="json", exclude_none=True)
        )
        validations.append(
            {
                "item_id": "hypothesis_tree_refreshed",
                "label": "假设空间已基于上一轮反馈更新",
                "passed": bool(tree_refreshed),
                "detail": (
                    "新树与快照树存在节点、支持度或更新标记差异。"
                    if tree_refreshed
                    else "新树与上一轮快照保持一致，未检测到结构或支持度变化。"
                ),
            }
        )

        uncertainty_refreshed = self._uncertainty_fingerprint(old_uncertainties) != self._uncertainty_fingerprint(
            uncertainties.model_dump(mode="json", exclude_none=True)
        )
        validations.append(
            {
                "item_id": "uncertainty_queue_refreshed",
                "label": "不确定性队列已重建并进入新一轮",
                "passed": bool(uncertainty_refreshed),
                "detail": (
                    "新一轮队列已基于旧轮目标不确定性重建。"
                    if uncertainty_refreshed
                    else "新一轮队列与上一轮完全一致，需要补充生成入口。"
                ),
            }
        )
        active_count = sum(
            1
            for record in uncertainties.records
            if record.status not in {"resolved", "deprecated"}
        )
        validations.append(
            {
                "item_id": "uncertainty_batch_sufficient",
                "label": "新一轮不确定性批次数量充足",
                "passed": bool(active_count >= MIN_UNCERTAINTIES),
                "expected": MIN_UNCERTAINTIES,
                "actual": active_count,
                "detail": (
                    f"新一轮保留 {active_count} 条有效不确定性，可支撑多条独立候选区分实验。"
                    if active_count >= MIN_UNCERTAINTIES
                    else f"有效不确定性仅 {active_count} 条，低于最小数量 {MIN_UNCERTAINTIES}，需要扩展生成。"
                ),
            }
        )
        return validations

    def _build_input_sources(self, *, snapshot: dict[str, Any], human_feedback: str | None) -> list[str]:
        closure = snapshot.get("closure") or {}
        raw_round_id = snapshot.get("round_id", 0) or 0
        sources = [f"round_archive:R{int(raw_round_id):02d}"]
        sources.extend(f"tested_hypothesis:{item}" for item in closure.get("tested_hypotheses", [])[:8])
        sources.extend(f"target_uncertainty:{item}" for item in closure.get("target_uncertainties", [])[:8])
        sources.extend(f"scientific_finding:{item}" for item in closure.get("scientific_findings", [])[:5])
        if closure.get("failure_reason"):
            sources.append(f"failure_reason:{closure['failure_reason']}")
        metrics = closure.get("metrics_snapshot") or {}
        if metrics:
            sources.append(
                "metric_delta:{},{}".format(
                    metrics.get("delta", {}).get("rmse", "--"),
                    metrics.get("delta", {}).get("pearson_r", "--"),
                )
            )
        if human_feedback:
            sources.append(f"human_feedback:{human_feedback}")

        memory = ((snapshot.get("artifacts") or {}).get("experiment_memory") or {}).get("entries", [])
        for entry in memory:
            for trace in entry.get("reasoning_traces", [])[:5]:
                sources.append(f"reasoning_trace:{trace.get('stage')}:{str(trace.get('summary', ''))[:80]}")
        try:
            tree = self.repository.load_hypothesis_tree()
        except Exception:
            tree = None
        try:
            uncertainty_records = self.repository.load_uncertainties().records
        except Exception:
            uncertainty_records = []
        sources = [
            clean_llm_text(item, tree=tree, uncertainty_records=uncertainty_records)
            for item in sources
        ]
        return sources

    @staticmethod
    def _tree_fingerprint(tree: dict[str, Any] | None) -> tuple[Any, ...]:
        if not tree:
            return ()
        nodes = tuple(
            (
                str(node.get("hypothesis_id")),
                node.get("support_score"),
                str(node.get("status")),
                node.get("updated_at_round"),
            )
            for node in (tree.get("nodes") or [])
        )
        latest = tree.get("latest_update") or {}
        return (nodes, latest.get("round"), latest.get("event"))

    @staticmethod
    def _uncertainty_fingerprint(state: dict[str, Any] | None) -> tuple[Any, ...]:
        if not state:
            return ()
        records = tuple(
            (
                str(record.get("uncertainty_id")),
                str(record.get("question")),
                str(record.get("priority")),
                str(record.get("status")),
            )
            for record in (state.get("records") or [])
        )
        queue = state.get("priority_queue") or {}
        queue_items = tuple(
            (str(item.get("uncertainty_id")), item.get("priority_score"))
            for item in queue.get("queue") or []
        )
        return (records, queue.get("current_round"), queue_items)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    @staticmethod
    def _dump_model(loader: Any) -> dict[str, Any] | None:
        try:
            model = loader()
            if hasattr(model, "model_dump"):
                return model.model_dump(mode="json", exclude_none=True)
            return model
        except Exception:
            return None
