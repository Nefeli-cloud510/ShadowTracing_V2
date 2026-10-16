from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from pydantic import BaseModel

from core.central_controller_llm import CentralControllerLLM, CentralControllerLLMResponse
from core.control_unified import HumanControlService
from core.experiment_planner_llm import (
    CandidateExperimentDesignerLLM,
    CandidateExperimentWriterLLM,
    ExperimentPlannerLLM,
)
from core.hypothesis_proposer_llm import (
    HypothesisProposerLLM,
    HypothesisProposerResponse,
    _display_dictionary_block,
)
from core.llm_gateway import LLMGateway
from core.planner_unified import PlannerOutputBuilder
from core.rag_service import RAGContextBundle, RAGEvidenceSnippet, RAGService
from core.runtime_config import (
    get_bailian_knowledge_agent_id,
    get_bailian_workspace_id,
    get_dasyscope_api_key,
    get_default_llm_model,
    get_knowledge_search_endpoint,
    load_project_env,
)
from core.scientific_interpreter_llm import ScientificInterpreterLLM, ScientificInterpreterLLMResponse
from core.scientific_questioner_llm import ScientificQuestionerLLM, ScientificQuestionerResponse
from core.state_repository import UnifiedStateRepository
from core.unified_schema import (
    DataDictionary,
    EvaluationSpec,
    FieldDescriptor,
    ResearchQuestion,
    ScientificConstraints,
    ScientificTask,
    ScientificTaskPayload,
    UncertaintyRecord,
    VariableBinding,
)
from core.variable_semantic_service import VariableSemanticService


@dataclass
class StageRecord:
    stage: str
    started_at: str
    finished_at: str | None
    duration_seconds: float | None
    status: str
    note: str | None = None


class ObservedLLMGateway(LLMGateway):
    def __init__(self, *, role_name: str, **kwargs: Any) -> None:
        super().__init__(statistics_observer=self._observe_statistics, **kwargs)
        self.role_name = role_name
        self.call_history: list[dict[str, Any]] = []

    def _observe_statistics(self, entry: dict[str, Any]) -> None:
        self.call_history.append(
            {
                **entry,
                "timestamp": datetime.now().isoformat(),
                "role": entry.get("role") or self.role_name,
                "client_available": self.client is not None,
                "used_fallback": bool(entry.get("used_fallback", False)),
                "error": None if entry.get("ok") else entry.get("error", "llm_call_failed"),
            }
        )

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        fallback_factory: Callable[[], BaseModel | dict[str, Any]] | None = None,
        payload_fixer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        temperature: float | None = None,
        image_paths: list[str] | None = None,
        role: str | None = None,
        use_cache: bool = True,
        deadline_seconds: float | None = None,
    ) -> BaseModel:
        effective_role = role or self.role_name

        def combined_fixer(payload: dict[str, Any]) -> dict[str, Any]:
            payload = _normalize_llm_payload(response_model, payload)
            if payload_fixer is not None:
                payload = payload_fixer(payload)
            return payload

        return super().generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            fallback_factory=fallback_factory,
            payload_fixer=combined_fixer,
            temperature=temperature,
            image_paths=image_paths,
            role=effective_role,
            use_cache=use_cache,
            deadline_seconds=deadline_seconds,
        )

    def telemetry(self) -> dict[str, Any]:
        history = list(self.call_history)
        total = len(history)
        fallbacks = sum(1 for item in history if item["used_fallback"])
        remote_success = sum(
            1 for item in history if item.get("ok") and not item.get("used_fallback")
        )
        return {
            "role": self.role_name,
            "model": self.model,
            "client_available": self.client is not None,
            "calls": total,
            "remote_success_calls": remote_success,
            "fallback_calls": fallbacks,
            "cache_hits": sum(1 for item in history if item.get("cache_hit")),
            "retry_count": sum(int(item.get("retries") or 0) for item in history),
            "input_tokens": sum(int(item.get("input_tokens") or 0) for item in history),
            "output_tokens": sum(int(item.get("output_tokens") or 0) for item in history),
            "elapsed_seconds": round(
                sum(float(item.get("elapsed_seconds") or 0.0) for item in history),
                3,
            ),
            "history": history,
        }


def _normalize_llm_payload(response_model: type[BaseModel], payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    model_name = response_model.__name__

    if model_name == "CentralControllerLLMResponse":
        value = normalized.get("impact_strength")
        if isinstance(value, str):
            normalized["impact_strength"] = {
                "low": 0.3,
                "medium": 0.55,
                "strong": 0.8,
                "high": 0.8,
            }.get(value.lower(), 0.55)
        notes = normalized.get("protocol_notes")
        if isinstance(notes, str):
            normalized["protocol_notes"] = [notes]
        for field in ("candidate_focus_ids", "related_uncertainty_ids", "feature_focus"):
            if isinstance(normalized.get(field), str):
                normalized[field] = [normalized[field]]
        return normalized

    if model_name == "ScientificInterpreterLLMResponse":
        if "interpretation" not in normalized:
            normalized["interpretation"] = (
                normalized.get("summary")
                or normalized.get("conclusion")
                or normalized.get("analysis")
                or normalized.get("suggested_state_note")
                or "需要保守解释当前结果。"
            )
        for field in ("related_uncertainty_ids",):
            if isinstance(normalized.get(field), str):
                normalized[field] = [normalized[field]]
        suggestion = normalized.get("next_round_suggestion")
        if isinstance(suggestion, str):
            suggestion_text = suggestion.strip()
            try:
                parsed_suggestion = json.loads(suggestion_text)
            except Exception:
                parsed_suggestion = None
            normalized["next_round_suggestion"] = (
                parsed_suggestion
                if isinstance(parsed_suggestion, dict)
                else {"recommendation": suggestion_text}
            )
        chart_analyses = normalized.get("chart_analyses")
        if isinstance(chart_analyses, dict):
            normalized["chart_analyses"] = [chart_analyses]
        return normalized

    if model_name == "ScientificQuestionerResponse":
        return ScientificQuestionerLLM._fix_payload(normalized)

    if model_name == "HypothesisProposerResponse":
        proposed = normalized.get("proposed_hypotheses")
        if isinstance(proposed, list):
            repaired = []
            for index, item in enumerate(proposed, start=1):
                if isinstance(item, dict):
                    fixed = dict(item)
                    fixed["statement"] = (
                        fixed.get("statement")
                        or fixed.get("question")
                        or fixed.get("summary")
                        or f"围绕下一轮候选方向提出待验证假设 {index}"
                    )
                    fixed["display_hypothesis_id"] = (
                        fixed.get("display_hypothesis_id")
                        or fixed.get("hypothesis_id")
                        or f"H{index}"
                    )
                    predictions = fixed.get("predictions")
                    if isinstance(predictions, list):
                        for prediction in predictions:
                            if not isinstance(prediction, dict):
                                continue
                            raw_range = prediction.get("expected_range")
                            parsed_range = _parse_range_pair(raw_range)
                            if parsed_range is not None:
                                prediction["expected_range"] = parsed_range
                            elif isinstance(raw_range, str):
                                prediction["expected_range"] = None
                                observable = str(
                                    prediction.get("expected_observable") or ""
                                ).strip()
                                if observable:
                                    prediction["expected_observable"] = (
                                        f"{observable}；{raw_range}"
                                    )
                                else:
                                    prediction["expected_observable"] = raw_range
                    repaired.append(fixed)
                else:
                    repaired.append(
                        {
                            "display_hypothesis_id": f"H{index}",
                            "statement": str(item),
                            "falsification": "",
                            "expected_effect": "",
                        }
                    )
            normalized["proposed_hypotheses"] = repaired
        return normalized

    if model_name == "ExperimentPlannerLLMResponse":
        extra_steps = normalized.get("extra_steps")
        if isinstance(extra_steps, list):
            normalized["extra_steps"] = [
                item
                if isinstance(item, dict)
                else {"action": "planner_note_step", "parameters": {"note": str(item)}}
                for item in extra_steps
            ]
        for field in ("suggested_feature_focus", "protocol_notes"):
            if isinstance(normalized.get(field), str):
                normalized[field] = [normalized[field]]
        return normalized

    return normalized


def _parse_range_pair(value: Any) -> list[float] | None:
    """Best-effort parse of an LLM-provided numeric [low, high] range."""
    if isinstance(value, list):
        try:
            numbers = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
        return [min(numbers[0], numbers[-1]), max(numbers[0], numbers[-1])] if numbers else None
    if not isinstance(value, str):
        return None
    numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
    if len(numbers) < 2:
        return None
    try:
        first, last = float(numbers[0]), float(numbers[-1])
    except ValueError:
        return None
    return [min(first, last), max(first, last)]


class ObservedRAGService(RAGService):
    def __init__(self, *, project_root: Path) -> None:
        super().__init__(project_root=project_root)
        self.remote_search_history: list[dict[str, Any]] = []

    def _search_remote_knowledge_base(self, query: str, *, max_results: int) -> list[dict[str, object]]:
        event: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "max_results": max_results,
            "endpoint": self.bailian_knowledge_endpoint,
            "agent_id": self.bailian_knowledge_agent_id,
            "attempted": True,
            "success": False,
            "result_count": 0,
            "error": None,
        }
        payload = {
            "agent_id": self.bailian_knowledge_agent_id,
            "query": query,
            "images": [],
        }
        if self.bailian_knowledge_agent_version:
            payload["agent_version"] = self.bailian_knowledge_agent_version
        request = urllib.request.Request(
            self.bailian_knowledge_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.bailian_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                body = response.read().decode("utf-8")
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                event["error"] = "invalid_response_type"
                self.remote_search_history.append(event)
                return []
            if parsed.get("success") is False:
                event["error"] = parsed.get("message") or "remote_success_false"
                self.remote_search_history.append(event)
                return []
            data = parsed.get("data")
            if isinstance(data, dict) and isinstance(data.get("nodes"), list):
                normalized = [
                    self._normalize_remote_node(item)
                    for item in data["nodes"][:max_results]
                    if isinstance(item, dict)
                ]
                event["success"] = True
                event["result_count"] = len(normalized)
                self.remote_search_history.append(event)
                return normalized
            event["success"] = True
            self.remote_search_history.append(event)
            return []
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            event["error"] = repr(exc)
            self.remote_search_history.append(event)
            return []

    def telemetry(self) -> dict[str, Any]:
        attempts = len(self.remote_search_history)
        successes = sum(1 for item in self.remote_search_history if item["success"])
        return {
            "configured": bool(
                self.bailian_api_key and self.bailian_knowledge_agent_id and self.bailian_knowledge_endpoint
            ),
            "attempts": attempts,
            "successful_attempts": successes,
            "history": self.remote_search_history,
        }


class CompactCentralControllerLLM(CentralControllerLLM):
    def _build_user_prompt(
        self,
        *,
        planner_input,
        candidate_plan,
        baseline_output,
    ) -> str:
        top_candidates = [
            {
                "experiment_id": candidate.experiment_id,
                "type": candidate.type,
                "question": candidate.scientific_question,
                "hypotheses": candidate.tested_hypotheses[:3],
                "uncertainties": candidate.related_uncertainties[:2],
                "utility_score": candidate.utility_score,
            }
            for candidate in candidate_plan.candidates[:3]
        ]
        payload = {
            "question": planner_input.scientific_question,
            "target": planner_input.target,
            "source_round_id": planner_input.source_round_id,
            "next_round_id": planner_input.next_round_id,
            "human_feedback": planner_input.human_feedback,
            "evaluation": {
                "experiment_id": planner_input.evaluation_summary.experiment_id,
                "delta_pearson_r": planner_input.evaluation_summary.delta_pearson_r,
                "delta_rmse": planner_input.evaluation_summary.delta_rmse,
                "stable": planner_input.evaluation_summary.stable,
                "key_findings": planner_input.evaluation_summary.key_findings[:2],
            },
            "active_hypotheses": [
                {
                    "hypothesis_id": item.hypothesis_id,
                    "statement": item.statement,
                    "support_score": item.support_score,
                }
                for item in planner_input.active_hypotheses[:3]
            ],
            "unresolved_uncertainties": [
                {
                    "uncertainty_id": item.uncertainty_id,
                    "question": item.question,
                    "priority": item.priority,
                }
                for item in planner_input.unresolved_uncertainties[:3]
            ],
            "planner_guidance": planner_input.planner_guidance[:6],
            "top_candidates": top_candidates,
            "baseline_summary": baseline_output.summary,
        }
        return (
            "请基于以下紧凑上下文，为下一轮输出结构化裁决。"
            "只能使用已有 candidate_id、hypothesis_id、uncertainty_id。\n"
            f"context={payload}\n"
            "请输出 JSON，包括：summary、candidate_focus_ids、candidate_rationale、"
            "target_hypothesis_id、interpretation、related_uncertainty_ids、impact_direction、"
            "impact_strength、confidence、uncertainty_priority_action、feature_focus、"
            "protocol_notes、suggested_model_parameters。"
        )


class CompactHypothesisProposerLLM(HypothesisProposerLLM):
    def propose(
        self,
        *,
        planner_input,
        rag_context,
    ) -> HypothesisProposerResponse:
        """Delegate to the canonical real-LLM proposer.

        The base implementation owns the strict H1..H5 validation, provenance
        fields (source/model/generated_at) and the no-fallback policy.
        """
        return super().propose(
            planner_input=planner_input,
            rag_context=rag_context,
        )

class CompactScientificQuestionerLLM(ScientificQuestionerLLM):
    def question(
        self,
        *,
        planner_input,
        rag_context,
        mined_candidates: list[dict[str, object]] | None = None,
    ) -> ScientificQuestionerResponse:
        semantic = VariableSemanticService.from_summary(planner_input.data_dictionary_summary)
        compact = {
            "question": semantic.display_text(planner_input.scientific_question),
            "display_target": semantic.to_display(planner_input.target),
            "display_dictionary": _display_dictionary_block(planner_input.data_dictionary_summary),
            "human_feedback": planner_input.human_feedback,
            "evaluation": {
                "delta_pearson_r": planner_input.evaluation_summary.delta_pearson_r,
                "stable": planner_input.evaluation_summary.stable,
                "key_findings": planner_input.evaluation_summary.key_findings[:2],
            },
            "active_hypotheses": [
                {
                    "hypothesis_id": item.hypothesis_id,
                    "statement": semantic.display_text(item.statement),
                }
                for item in planner_input.active_hypotheses[:3]
            ],
            "unresolved_uncertainties": [
                {
                    "uncertainty_id": item.uncertainty_id,
                    "question": semantic.display_text(item.question),
                }
                for item in planner_input.unresolved_uncertainties[:3]
            ],
            "recent_disagreement_updates": [
                {
                    "uncertainty_id": item.uncertainty_id,
                    "resolution_status": item.resolution_status,
                    "summary": item.summary,
                }
                for item in planner_input.recent_disagreement_updates[:2]
            ],
            "rag_notes": rag_context.guidance_notes()[:4],
            "mined_candidates": (mined_candidates or [])[:8],
        }
        return self.gateway.generate_structured(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=(
                f"context={compact}\n"
                "科学书写规则：proposed_uncertainties 必须写成自然、专业、物理语义清晰的科学问题，"
                "明确自变量、因变量与物理路径关系，并结合假设分歧、残差模式、支持度变化和失败归因；"
                "禁止输出“关于 A 与 B 的解释是否仍受约束”这类固定填空句式。\n"
                "名称规则：问题文本、description、features 一律使用 display_dictionary 中的展示名，"
                "例如“宇宙线日影南北偏移”“太阳风速度”“行星际磁场Y分量”，"
                "禁止输出原始表头或“SW Plasma Speed, km/s”这类英文单位标签。\n"
                "related_hypotheses 只能填 active_hypotheses 中确实相关的假设编号，没有明确对应时留空；"
                "请结合 mined_candidates 的假设冲突、残差、支持度变化与失败归因线索归纳去重，"
                "输出 6-10 条有效 uncertainty，并尽量保留 mining_sources 来源标签。"
            ),
            response_model=ScientificQuestionerResponse,
            fallback_factory=lambda: self._fallback(planner_input, rag_context),
            role="科学质询者llm",
        )


class CompactScientificInterpreterLLM(ScientificInterpreterLLM):
    def build_interpretations(
        self,
        *,
        planner_input,
        candidate_plan,
    ):
        baseline_output = self.baseline_builder.build(
            planner_input=planner_input,
            candidate_plan=candidate_plan,
        )
        baseline = baseline_output.interpretation_enhancements
        compact = {
            "experiment_id": planner_input.source_experiment_id,
            "round_id": planner_input.source_round_id,
            "question": planner_input.scientific_question,
            "target": planner_input.target,
            "evaluation": {
                "delta_pearson_r": planner_input.evaluation_summary.delta_pearson_r,
                "delta_rmse": planner_input.evaluation_summary.delta_rmse,
                "stable": planner_input.evaluation_summary.stable,
                "key_findings": planner_input.evaluation_summary.key_findings[:2],
            },
            "active_hypotheses": [
                {
                    "hypothesis_id": item.hypothesis_id,
                    "statement": item.statement,
                    "support_score": item.support_score,
                }
                for item in planner_input.active_hypotheses[:3]
            ],
            "unresolved_uncertainties": [
                {
                    "uncertainty_id": item.uncertainty_id,
                    "question": item.question,
                    "priority": item.priority,
                }
                for item in planner_input.unresolved_uncertainties[:3]
            ],
        }
        response = self.gateway.generate_structured(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=f"context={compact}\n请输出一条结构化解释增强。",
            response_model=ScientificInterpreterLLMResponse,
            fallback_factory=lambda: self._build_fallback_response(planner_input, baseline),
            role="科学解释者llm",
        )
        llm_enhancement = __import__(
            "core.unified_schema", fromlist=["InterpretationEnhancement"]
        ).InterpretationEnhancement(
            enhancement_id="LIE_INT_001",
            target_hypothesis_id=(
                response.target_hypothesis_id
                if response.target_hypothesis_id in {item.hypothesis_id for item in planner_input.active_hypotheses}
                else (planner_input.active_hypotheses[0].hypothesis_id if planner_input.active_hypotheses else None)
            ),
            interpretation=response.interpretation,
            evidence_basis=planner_input.evaluation_summary.key_findings[:2],
            related_uncertainties=[
                item
                for item in response.related_uncertainty_ids
                if item in {entry.uncertainty_id for entry in planner_input.unresolved_uncertainties}
            ]
            or [item.uncertainty_id for item in planner_input.unresolved_uncertainties[:2]],
            suggested_state_note=response.suggested_state_note,
            impact_direction=response.impact_direction if response.impact_direction in {"supports", "weakens", "clarifies"} else "clarifies",
            impact_strength=response.impact_strength,
            confidence=response.confidence,
            uncertainty_priority_action=(
                response.uncertainty_priority_action
                if response.uncertainty_priority_action in {"increase", "decrease", "maintain"}
                else "maintain"
            ),
        )
        return baseline[:1] + [llm_enhancement]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real-API two-round Shadow Tracing demo.")
    parser.add_argument(
        "--model",
        default=get_default_llm_model(),
        help="Model used by the demo-time LLM roles. Does not modify the main project default.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "artifacts" / "real_api_runs"),
        help="Directory where timestamped demo outputs will be created.",
    )
    parser.add_argument(
        "--allow-llm-fallback",
        action="store_true",
        help="Do not fail the run when any LLM role falls back to the local deterministic builder.",
    )
    parser.add_argument(
        "--allow-kb-skip",
        action="store_true",
        help="Do not fail the run when the remote Bailian knowledge search is not attempted successfully.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_project_env(REPO_ROOT)
    started_at = datetime.now()
    output_root = Path(args.output_root)
    run_root = output_root / started_at.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    stage_log: list[StageRecord] = []
    summary_path = run_root / "run_summary.json"
    audit_path = run_root / "audit_report.json"
    telemetry_path = run_root / "telemetry.json"
    failure_path = run_root / "failure_report.json"

    print(f"[info] demo run root: {run_root}")
    print(f"[info] project default model: {get_default_llm_model()}")
    print(f"[info] demo runtime model: {args.model}")

    try:
        repo = UnifiedStateRepository(run_root)
        task, dictionary = initialize_demo_workspace(run_root, repo)
        gateway_bundle = build_real_llm_control(repo, run_root, model=args.model)
        control: HumanControlService = gateway_bundle["control"]
        role_gateways: dict[str, ObservedLLMGateway] = gateway_bundle["gateways"]
        rag_service: ObservedRAGService = gateway_bundle["rag_service"]

        payload: dict[str, Any] = {}

        payload["initial_review"] = run_stage(
            stage_log,
            "initial_review",
            lambda: control.request_experiment_selection_review(),
            note="请求第一轮 PI 审批入口",
        )
        print(f"[stage] initial_review -> {payload['initial_review']['status']}")

        payload["round1"] = run_stage(
            stage_log,
            "round1_execute",
            lambda: control.approve_and_execute_candidate(
                human_notes="real api round1 批准执行",
                auto_continue=False,
            ),
            note="真实 API + 本地执行闭环 round1",
        )
        print_round_summary(payload["round1"], label="round1")

        payload["after_round1"] = run_stage(
            stage_log,
            "round1_review_decision",
            lambda: control.record_round_decision(
                round_id=payload["round1"]["protocol"].round_id,
                decision="adjust",
                human_feedback="下一轮继续关注 By 与 Np 条件路径，并保持验证型方案",
                data_dictionary=dictionary,
            ),
            note="保留 PI 调整路径，进入 round2 规划",
        )
        print(f"[stage] after_round1 -> {payload['after_round1']['planning_status']}")

        payload["round2"] = run_stage(
            stage_log,
            "round2_execute",
            lambda: control.approve_and_execute_candidate(
                human_notes="real api round2 批准执行",
                auto_continue=False,
            ),
            note="真实 API + 本地执行闭环 round2",
        )
        print_round_summary(payload["round2"], label="round2")

        payload["after_round2"] = run_stage(
            stage_log,
            "round2_review_decision",
            lambda: control.record_round_decision(
                round_id=payload["round2"]["protocol"].round_id,
                decision="continue",
                data_dictionary=dictionary,
            ),
            note="保留 PI 继续路径，生成下一轮规划留档",
        )
        print(f"[stage] after_round2 -> {payload['after_round2']['planning_status']}")

        audit_report = run_stage(
            stage_log,
            "audit_checks",
            lambda: build_audit_report(repo, payload),
            note="按 AUDIT_CHECKLIST.md 做状态留档检查",
        )
        telemetry = {
            "llm_roles": {name: gateway.telemetry() for name, gateway in role_gateways.items()},
            "rag": rag_service.telemetry(),
        }

        validation = validate_real_api_requirements(
            telemetry=telemetry,
            allow_llm_fallback=args.allow_llm_fallback,
            allow_kb_skip=args.allow_kb_skip,
        )

        summary = build_run_summary(
            repo=repo,
            task=task,
            model=args.model,
            payload=payload,
            audit_report=audit_report,
            telemetry=telemetry,
            validation=validation,
            stage_log=stage_log,
            started_at=started_at,
            finished_at=datetime.now(),
        )

        audit_path.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8")
        telemetry_path.write_text(json.dumps(telemetry, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print("[done] round2 completed and artifacts archived")
        print(f"[done] summary: {summary_path}")
        print(f"[done] audit:   {audit_path}")
        print(f"[done] telem:   {telemetry_path}")
        print(json.dumps(summary["final_status"], ensure_ascii=False, indent=2))

        return 0 if validation["passed"] else 2
    except Exception as exc:  # pragma: no cover - exercised only during manual/live runs
        failure_report = {
            "timestamp": datetime.now().isoformat(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "stage_log": [asdict(item) for item in stage_log],
        }
        failure_path.write_text(json.dumps(failure_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[error] demo failed, details written to {failure_path}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


def initialize_demo_workspace(
    run_root: Path,
    repo: UnifiedStateRepository,
) -> tuple[ScientificTask, DataDictionary]:
    create_demo_data(run_root)
    create_demo_literature_stub(run_root)
    task = ScientificTask(
        task_id=f"ST_REAL_API_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        payload=ScientificTaskPayload(
            research_question=ResearchQuestion(
                text="DeltaDec 是否提供可重复验证的 Vsw 预测增量信息",
                target="Vsw",
                question_type="forecasting",
                variables=VariableBinding(x="DeltaDec", y="Vsw", m_candidates=["By", "Bz", "Np"]),
                keywords=["DeltaDec", "Vsw", "By", "Np"],
                clarified=True,
            ),
            evaluation=EvaluationSpec(
                primary_metric="Pearson_r",
                secondary_metrics=["RMSE", "MAE"],
                visual_analysis=["prediction_vs_truth", "scatter_plot"],
            ),
            constraints=ScientificConstraints(
                no_future_information=True,
                validation_feedback_allowed=True,
                final_test_blind=True,
                max_rounds=10,
                max_experiments_per_round=3,
                resource_budget={
                    "token_budget": 100000,
                    "max_time_seconds_per_round": 300,
                    "max_candidates": 3,
                },
            ),
            data_sources={
                "omni": {"path": "data/raw/omni.csv", "time_column": "TIME", "target_column": "Vsw"},
                "lhaaso": {"path": "data/raw/lhaaso.csv", "time_column": "TIME", "target_column": "DeltaDec"},
            },
        ),
    )
    dictionary = DataDictionary(
        dictionary_id="DICT_REAL_API_DEMO",
        generated_at=datetime.now(),
        version="1.0",
        dataset_name="real_api_demo",
        total_samples=48,
        time_column="TIME",
        time_format="YYYYMMDD",
        fields=[
            FieldDescriptor(field_name="TIME", data_type="int", physical_meaning="time"),
            FieldDescriptor(field_name="Vsw", data_type="float", physical_meaning="target", is_target=True),
            FieldDescriptor(field_name="By", data_type="float", physical_meaning="IMF By"),
            FieldDescriptor(field_name="Bz", data_type="float", physical_meaning="IMF Bz"),
            FieldDescriptor(field_name="Np", data_type="float", physical_meaning="Solar wind density"),
            FieldDescriptor(field_name="DeltaDec", data_type="float", physical_meaning="LHAASO DeltaDec"),
        ],
        target_candidates=["Vsw"],
        feature_candidates=["By", "Bz", "Np", "DeltaDec"],
    )
    repo.initialize_state_skeleton(
        task,
        initial_uncertainties=[
            UncertaintyRecord(
                uncertainty_id="U01",
                question="DeltaDec 的贡献是否独立于 By？",
                description="需要区分独立增量与中介解释。",
                related_hypotheses=[],
                status="active",
                priority="high",
                created_at_round=0,
                created_by="real_api_demo",
            ),
            UncertaintyRecord(
                uncertainty_id="U02",
                question="DeltaDec 的增益是否跨时间段稳定？",
                description="需要检验滞后稳定性。",
                related_hypotheses=[],
                status="active",
                priority="medium",
                created_at_round=0,
                created_by="real_api_demo",
            ),
        ],
        data_dictionary=dictionary,
        overwrite=True,
    )
    from core.decision_unified import DecisionLayerService

    DecisionLayerService(repo).build_candidate_plan(task=task, data_dictionary=dictionary)
    (run_root / "state" / "demo_dictionary.json").write_text(
        json.dumps(dictionary.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return task, dictionary


def create_demo_data(run_root: Path) -> None:
    raw_dir = run_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    time_values = pd.date_range("2021-01-01", periods=48, freq="D")
    omni = pd.DataFrame(
        {
            "TIME": time_values.strftime("%Y%m%d"),
            "By": [(-1) ** i * (i % 5 + 1) for i in range(48)],
            "Bz": [(i % 7) - 3 for i in range(48)],
            "Np": [5.0 + (i % 4) * 0.5 for i in range(48)],
            "Vsw": [360 + i * 1.7 for i in range(48)],
        }
    )
    lhaaso = pd.DataFrame(
        {
            "TIME": time_values.strftime("%Y%m%d"),
            "DeltaDec": [0.08 * ((i % 6) - 3) + (0.02 if i % 9 == 0 else 0.0) for i in range(48)],
        }
    )
    omni.to_csv(raw_dir / "omni.csv", index=False)
    lhaaso.to_csv(raw_dir / "lhaaso.csv", index=False)


def create_demo_literature_stub(run_root: Path) -> None:
    pdf_text_dir = run_root / "outputs" / "pdf_texts"
    pdf_text_dir.mkdir(parents=True, exist_ok=True)
    (pdf_text_dir / "demo_background.txt").write_text(
        (
            "LHAASO DeltaDec may provide early information about heliospheric structure. "
            "A key uncertainty is whether the predictive gain on Vsw is independent of By "
            "or mostly mediated by IMF-related variables."
        ),
        encoding="utf-8",
    )


def build_real_llm_control(
    repo: UnifiedStateRepository,
    run_root: Path,
    *,
    model: str,
) -> dict[str, Any]:
    gateway_kwargs = {"model": model, "allow_fallback": False}
    controller_gateway = ObservedLLMGateway(role_name="central_controller", **gateway_kwargs)
    interpreter_gateway = ObservedLLMGateway(role_name="scientific_interpreter", **gateway_kwargs)
    proposer_gateway = ObservedLLMGateway(role_name="hypothesis_proposer", **gateway_kwargs)
    questioner_gateway = ObservedLLMGateway(role_name="scientific_questioner", **gateway_kwargs)
    planner_gateway = ObservedLLMGateway(role_name="experiment_planner", **gateway_kwargs)
    designer_gateway = ObservedLLMGateway(role_name="experiment_designer", **gateway_kwargs)
    writer_gateway = ObservedLLMGateway(role_name="experiment_writer", **gateway_kwargs)
    rag_service = ObservedRAGService(project_root=run_root)

    control = HumanControlService(
        repo,
        project_root=run_root,
        run_shap=False,
        planner_mode="llm",
        planner_output_builder=PlannerOutputBuilder(
            mode="llm",
            central_controller=CompactCentralControllerLLM(gateway=controller_gateway),
        ),
        scientific_interpreter=CompactScientificInterpreterLLM(gateway=interpreter_gateway),
        rag_service=rag_service,
        hypothesis_proposer=CompactHypothesisProposerLLM(gateway=proposer_gateway),
        scientific_questioner=CompactScientificQuestionerLLM(gateway=questioner_gateway),
        experiment_planner=ExperimentPlannerLLM(gateway=planner_gateway),
        experiment_designer=CandidateExperimentDesignerLLM(gateway=designer_gateway),
        experiment_writer=CandidateExperimentWriterLLM(gateway=writer_gateway),
    )
    return {
        "control": control,
        "rag_service": rag_service,
        "gateways": {
            "central_controller": controller_gateway,
            "scientific_interpreter": interpreter_gateway,
            "hypothesis_proposer": proposer_gateway,
            "scientific_questioner": questioner_gateway,
            "experiment_planner": planner_gateway,
            "experiment_designer": designer_gateway,
            "experiment_writer": writer_gateway,
        },
    }


def run_stage(
    stage_log: list[StageRecord],
    stage_name: str,
    func: Callable[[], Any],
    *,
    note: str | None = None,
) -> Any:
    started_at = datetime.now()
    started_perf = perf_counter()
    record = StageRecord(
        stage=stage_name,
        started_at=started_at.isoformat(),
        finished_at=None,
        duration_seconds=None,
        status="running",
        note=note,
    )
    stage_log.append(record)
    print(f"[stage:start] {stage_name}" + (f" | {note}" if note else ""))
    try:
        result = func()
        record.finished_at = datetime.now().isoformat()
        record.duration_seconds = round(perf_counter() - started_perf, 3)
        record.status = "completed"
        print(f"[stage:done] {stage_name} ({record.duration_seconds:.3f}s)")
        return result
    except Exception:
        record.finished_at = datetime.now().isoformat()
        record.duration_seconds = round(perf_counter() - started_perf, 3)
        record.status = "failed"
        print(f"[stage:fail] {stage_name} ({record.duration_seconds:.3f}s)")
        raise


def print_round_summary(payload: dict[str, Any], *, label: str) -> None:
    protocol = payload["protocol"]
    evaluation = payload["evaluation"]
    print(
        f"[stage] {label} -> experiment={protocol.experiment_id}, "
        f"round={protocol.round_id}, "
        f"delta_pearson_r={evaluation.metrics.delta.pearson_r:.4f}, "
        f"stable={evaluation.robustness.overall.stable}"
    )


def build_audit_report(repo: UnifiedStateRepository, payload: dict[str, Any]) -> dict[str, Any]:
    tree = repo.load_hypothesis_tree()
    uncertainties = repo.load_uncertainties()
    experiment_memory = repo.load_experiment_memory()
    decision_log = repo.load_decision_log()
    planner_input = repo.load_planner_input()
    planner_output = repo.load_planner_output()

    round1_protocol = payload["round1"]["protocol"]
    round2_protocol = payload["round2"]["protocol"]
    node_index = tree.node_index()
    record_index = uncertainties.record_index()
    decision_types = [entry.decision_type for entry in decision_log.decisions]
    entry_index = experiment_memory.entry_index()

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    required_paths = {
        "hypothesis_tree": repo.paths.hypothesis_tree,
        "uncertainties": repo.paths.uncertainties,
        "experiment_memory": repo.paths.experiment_memory,
        "decision_log": repo.paths.decision_log,
        "planner_input": repo.paths.planner_input,
        "planner_output": repo.paths.planner_output,
    }
    for name, path in required_paths.items():
        check(f"file_exists:{name}", path.exists(), str(path))

    check("tree_task_matches", bool(tree.task_id), f"task_id={tree.task_id}")
    check("tree_nodes_non_empty", bool(tree.nodes), f"node_count={len(tree.nodes)}")
    check(
        "tree_active_ids_valid",
        set(tree.active_hypotheses).issubset(set(node_index)),
        f"active_count={len(tree.active_hypotheses)}",
    )
    check(
        "tree_latest_update_applied",
        tree.latest_update.event == "planner_output_applied",
        f"latest_event={tree.latest_update.event}",
    )
    check("tree_has_active_hypothesis", bool(tree.active_hypotheses), f"active={tree.active_hypotheses[:5]}")

    check(
        "uncertainty_task_matches",
        uncertainties.task_id == tree.task_id,
        f"uncertainty_task={uncertainties.task_id}",
    )
    check(
        "uncertainty_links_valid",
        all(h_id in node_index for record in uncertainties.records for h_id in record.related_hypotheses),
        f"records={len(uncertainties.records)}",
    )
    check(
        "uncertainty_history_present",
        any(record.history for record in uncertainties.records),
        "at least one uncertainty record should have history",
    )
    check(
        "uncertainty_queue_valid",
        all(item.uncertainty_id in record_index for item in uncertainties.priority_queue.queue),
        f"queue_len={len(uncertainties.priority_queue.queue)}",
    )

    check(
        "experiment_memory_round1_present",
        round1_protocol.experiment_id in entry_index,
        f"round1={round1_protocol.experiment_id}",
    )
    check(
        "experiment_memory_round2_present",
        round2_protocol.experiment_id in entry_index,
        f"round2={round2_protocol.experiment_id}",
    )
    check(
        "experiment_memory_has_reasoning_traces",
        bool(entry_index.get(round1_protocol.experiment_id).reasoning_traces)
        and bool(entry_index.get(round2_protocol.experiment_id).reasoning_traces),
        "both executed experiments should have reasoning traces",
    )

    check(
        "decision_round_review_requested_twice",
        decision_types.count("round_review_requested") >= 2,
        f"count={decision_types.count('round_review_requested')}",
    )
    check(
        "decision_planner_input_prepared_twice",
        decision_types.count("planner_input_prepared") >= 2,
        f"count={decision_types.count('planner_input_prepared')}",
    )
    check(
        "decision_planner_output_generated_twice",
        decision_types.count("planner_output_generated") >= 2,
        f"count={decision_types.count('planner_output_generated')}",
    )
    check(
        "decision_planner_output_applied_twice",
        decision_types.count("planner_output_applied") >= 2,
        f"count={decision_types.count('planner_output_applied')}",
    )
    check(
        "decision_paths_include_adjust_and_continue",
        "round_adjusted" in decision_types and "continue_next_round" in decision_types,
        f"types={sorted(set(decision_types))}",
    )

    check(
        "planner_input_round_alignment",
        planner_input.source_round_id == round2_protocol.round_id
        and planner_input.next_round_id == round2_protocol.round_id + 1,
        f"source={planner_input.source_round_id}, next={planner_input.next_round_id}",
    )
    check(
        "planner_input_has_disagreement_updates",
        bool(planner_input.recent_disagreement_updates),
        f"count={len(planner_input.recent_disagreement_updates)}",
    )
    check(
        "planner_input_has_unresolved_uncertainties",
        bool(planner_input.unresolved_uncertainties),
        f"count={len(planner_input.unresolved_uncertainties)}",
    )

    candidate_set = repo.load_candidate_experiments()
    candidate_ids = {candidate.experiment_id for candidate in candidate_set.candidates}
    supplemented_ids = {item.candidate.experiment_id for item in planner_output.candidate_supplements}
    check(
        "planner_output_round_alignment",
        planner_output.source_round_id == planner_input.source_round_id
        and planner_output.next_round_id == planner_input.next_round_id,
        f"source={planner_output.source_round_id}, next={planner_output.next_round_id}",
    )
    check(
        "planner_output_sections_present",
        bool(planner_output.candidate_supplements)
        and bool(planner_output.interpretation_enhancements)
        and bool(planner_output.protocol_refinements),
        (
            f"supplements={len(planner_output.candidate_supplements)}, "
            f"interpretations={len(planner_output.interpretation_enhancements)}, "
            f"refinements={len(planner_output.protocol_refinements)}"
        ),
    )
    check(
        "planner_output_candidates_valid",
        supplemented_ids.issubset(candidate_ids),
        f"supplemented={sorted(supplemented_ids)}",
    )

    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed,
        "checks": checks,
    }


def validate_real_api_requirements(
    *,
    telemetry: dict[str, Any],
    allow_llm_fallback: bool,
    allow_kb_skip: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    llm_roles = telemetry["llm_roles"]
    for role_name, info in llm_roles.items():
        check(
            f"llm_client_available:{role_name}",
            info["client_available"],
            f"calls={info['calls']}, fallback_calls={info['fallback_calls']}",
        )
        check(
            f"llm_called:{role_name}",
            info["calls"] > 0,
            f"remote_success_calls={info['remote_success_calls']}",
        )
        check(
            f"llm_no_fallback:{role_name}",
            allow_llm_fallback or info["fallback_calls"] == 0,
            f"fallback_calls={info['fallback_calls']}",
        )

    rag = telemetry["rag"]
    check("rag_configured", rag["configured"] or allow_kb_skip, f"attempts={rag['attempts']}")
    check("rag_remote_attempted", allow_kb_skip or rag["attempts"] > 0, f"attempts={rag['attempts']}")
    check(
        "rag_remote_success",
        allow_kb_skip or rag["successful_attempts"] > 0,
        f"successful_attempts={rag['successful_attempts']}",
    )

    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }


def build_run_summary(
    *,
    repo: UnifiedStateRepository,
    task: ScientificTask,
    model: str,
    payload: dict[str, Any],
    audit_report: dict[str, Any],
    telemetry: dict[str, Any],
    validation: dict[str, Any],
    stage_log: list[StageRecord],
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    decision_log = repo.load_decision_log()
    decision_types = [entry.decision_type for entry in decision_log.decisions]
    state_files = {
        name: repo.relativize(path)
        for name, path in {
            "hypothesis_tree": repo.paths.hypothesis_tree,
            "uncertainties": repo.paths.uncertainties,
            "experiment_memory": repo.paths.experiment_memory,
            "decision_log": repo.paths.decision_log,
            "planner_input": repo.paths.planner_input,
            "planner_output": repo.paths.planner_output,
        }.items()
    }
    return {
        "task_id": task.task_id,
        "planner_mode": "llm",
        "model": model,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "rounds": {
            "round1": {
                "round_id": payload["round1"]["protocol"].round_id,
                "experiment_id": payload["round1"]["protocol"].experiment_id,
                "delta_pearson_r": payload["round1"]["evaluation"].metrics.delta.pearson_r,
                "stable": payload["round1"]["evaluation"].robustness.overall.stable,
            },
            "round2": {
                "round_id": payload["round2"]["protocol"].round_id,
                "experiment_id": payload["round2"]["protocol"].experiment_id,
                "delta_pearson_r": payload["round2"]["evaluation"].metrics.delta.pearson_r,
                "stable": payload["round2"]["evaluation"].robustness.overall.stable,
            },
        },
        "telemetry": {
            "llm_roles": {
                name: {
                    "calls": info["calls"],
                    "remote_success_calls": info["remote_success_calls"],
                    "fallback_calls": info["fallback_calls"],
                }
                for name, info in telemetry["llm_roles"].items()
            },
            "rag": {
                "configured": telemetry["rag"]["configured"],
                "attempts": telemetry["rag"]["attempts"],
                "successful_attempts": telemetry["rag"]["successful_attempts"],
            },
        },
        "final_status": {
            "real_llm_verified": all(
                item["fallback_calls"] == 0 and item["remote_success_calls"] > 0
                for item in telemetry["llm_roles"].values()
            ),
            "real_knowledge_base_hit": telemetry["rag"]["successful_attempts"] > 0,
            "scientific_interpreter_writeback": decision_types.count("scientific_interpreter_applied"),
            "round2_completed": payload["round2"]["review"]["status"] == "awaiting_round_decision",
            "state_files_written": all((repo.project_root / relative_path).exists() for relative_path in state_files.values()),
            "audit_passed": audit_report["passed"],
            "validation_passed": validation["passed"],
        },
        "validation": validation,
        "audit_report": audit_report,
        "stage_log": [asdict(item) for item in stage_log],
        "state_files": state_files,
    }


if __name__ == "__main__":
    sys.exit(main())
