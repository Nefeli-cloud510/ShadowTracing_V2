from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.llm_gateway import LLMGateway
from core.prompt_rules import ELASTIC_NET_EXECUTION_RULE
from core.rag_service import RAGContextBundle
from core.unified_schema import ReasoningPlannerInput
from core.variable_semantic_service import VariableSemanticService


class ProposedUncertainty(BaseModel):
    uncertainty_id: str | None = None
    question: str = Field(min_length=1)
    description: str = Field(min_length=1)
    priority: str = Field(default="medium", min_length=1)
    mining_sources: list[str] = Field(default_factory=list)
    related_hypotheses: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)


class HypothesisQuestioningUpdate(BaseModel):
    hypothesis_id: str = Field(min_length=1)
    impact_direction: Literal["supports", "weakens", "clarifies"] = "clarifies"
    impact_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = Field(default="", min_length=0)
    falsification_basis: str = Field(default="", min_length=0)


class ScientificQuestionerResponse(BaseModel):
    challenge_points: list[str] = Field(default_factory=list)
    guidance_notes: list[str] = Field(default_factory=list)
    proposed_uncertainties: list[ProposedUncertainty] = Field(default_factory=list)
    hypothesis_updates: list[HypothesisQuestioningUpdate] = Field(default_factory=list)


class ScientificQuestionerLLM:
    """LLM-backed scientific questioner that turns evidence gaps into explicit uncertainties."""

    SYSTEM_PROMPT = """你是“逐影 Shadow Tracing”的科学质询者。

你的任务是识别当前闭环中的证据缺口、替代解释和仍需验证的问题，并输出结构化 JSON。
必须充分利用全部假设树节点、竞争假设之间的分歧、证据缺口、参数敏感性、模型残差和失败归因，尽量批量产出 6-10 条具有物理意义的不确定性。
所有变量名称必须严格来自 planner_input.data_dictionary_summary 中已有的字段，不允许发明数据表里不存在的新变量。
proposed_uncertainties 的 question 和 description 必须写成自然、专业、可直接展示的物理语言，
表述成真正的科学问题（例如“控制行星际磁场Y分量后，宇宙线日影南北偏移对太阳风速度的预测增益是否仍然存在？”），
禁止使用“关于 A 与 B 的解释是否仍受未建模条件约束”这类固定填空句式，也不要机械复述字段标签。
任何叙述文本不得出现 H_*、U_*、status=、narrowed=、before_span= 等内部机器标识；提及假设用 H1..H5 或自然语言名称，提及不确定性用问题原文，内部 ID 只允许出现在指定 id 字段。
proposed_uncertainties 可附带 mining_sources、related_hypotheses、features 字段用于来源追溯；请优先使用系统给出的 mined_candidates 线索，并在归纳时保留原始来源标签。
不要直接修改状态文件，只给出 challenge_points、guidance_notes、proposed_uncertainties。""" + "\n\n" + ELASTIC_NET_EXECUTION_RULE

    def __init__(self, *, gateway: LLMGateway | None = None) -> None:
        self.gateway = gateway or LLMGateway()

    @staticmethod
    def _fix_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Repair common JSON null/type quirks so a real LLM answer survives validation."""
        normalized = dict(payload)
        for field in ("challenge_points", "guidance_notes"):
            values = normalized.get(field)
            if isinstance(values, list):
                normalized[field] = [_text_or_json(item) for item in values]

        proposed = normalized.get("proposed_uncertainties")
        if isinstance(proposed, list):
            repaired = []
            for item in proposed:
                if not isinstance(item, dict):
                    repaired.append(
                        {
                            "question": _nonempty_text(item),
                            "description": _nonempty_text(item),
                            "priority": "medium",
                        }
                    )
                    continue
                fixed = dict(item)
                fixed["question"] = _nonempty_text(
                    fixed.get("question") or fixed.get("text") or fixed.get("summary")
                )
                if not fixed["question"]:
                    continue
                fixed["description"] = _nonempty_text(
                    fixed.get("description")
                    or fixed.get("details")
                    or fixed.get("summary")
                    or fixed.get("question")
                )
                fixed["priority"] = _nonempty_text(fixed.get("priority")) or "medium"
                for list_field in ("mining_sources", "related_hypotheses", "features"):
                    raw_list = fixed.get(list_field)
                    if isinstance(raw_list, list):
                        fixed[list_field] = [str(item) for item in raw_list if item is not None]
                repaired.append(fixed)
            normalized["proposed_uncertainties"] = repaired

        updates = normalized.get("hypothesis_updates")
        if isinstance(updates, list):
            repaired = []
            for item in updates:
                if not isinstance(item, dict):
                    continue
                fixed = dict(item)
                direction = _nonempty_text(fixed.get("impact_direction")).lower()
                if direction not in {"supports", "weakens", "clarifies"}:
                    direction = "clarifies"
                falsification_basis = _nonempty_text(fixed.get("falsification_basis"))
                if direction == "weakens" and not falsification_basis:
                    direction = "clarifies"
                fixed["impact_direction"] = direction
                fixed["falsification_basis"] = falsification_basis
                fixed["rationale"] = _nonempty_text(fixed.get("rationale"))
                fixed["hypothesis_id"] = _nonempty_text(fixed.get("hypothesis_id"))
                if not fixed["hypothesis_id"]:
                    continue
                for numeric_field, default in (("impact_strength", 0.5), ("confidence", 0.5)):
                    raw_value = fixed.get(numeric_field)
                    try:
                        numeric = float(raw_value) if raw_value is not None else default
                    except (TypeError, ValueError):
                        numeric = default
                    fixed[numeric_field] = max(0.0, min(1.0, numeric))
                repaired.append(fixed)
            normalized["hypothesis_updates"] = repaired
        return normalized

    def question(
        self,
        *,
        planner_input: ReasoningPlannerInput,
        rag_context: RAGContextBundle,
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
                for item in planner_input.active_hypotheses[:5]
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
                "related_hypotheses 应尽量填写 active_hypotheses 中确实相关的假设编号，"
                "不确定时填最相关的一个或多个，不要整组留空；"
                "请结合 mined_candidates 的假设冲突、残差、支持度变化与失败归因线索归纳去重，"
                "输出 6-10 条有效 uncertainty，并尽量保留 mining_sources 来源标签。"
            ),
            response_model=ScientificQuestionerResponse,
            payload_fixer=self._fix_payload,
            fallback_factory=lambda: self._fallback(planner_input, rag_context, mined_candidates),
            role="科学质询者llm",
        )

    def challenge_hypothesis_tree(
        self,
        *,
        planner_input: ReasoningPlannerInput,
        rag_context: RAGContextBundle,
        tree,
        mined_candidates: list[dict[str, object]] | None = None,
        targeted_hypothesis_ids: list[str] | None = None,
    ) -> ScientificQuestionerResponse:
        """Question the frozen hypothesis tree within the targeted scope.

        Historically questioned stable nodes are inherited without a new pass:
        they appear only as retained context, while hypothesis_updates are
        requested solely for targeted_hypothesis_ids.
        """
        semantic = VariableSemanticService.from_summary(planner_input.data_dictionary_summary)
        target_set = (
            set(targeted_hypothesis_ids)
            if targeted_hypothesis_ids is not None
            else {node.hypothesis_id for node in tree.nodes}
        )

        def _node_snapshot(node) -> dict[str, object]:
            latest = node.questioning_records[-1] if node.questioning_records else None
            snapshot: dict[str, object] = {
                "hypothesis_id": node.hypothesis_id,
                "display_hypothesis_id": node.display_hypothesis_id or f"H{node.level}",
                "statement": semantic.display_text(node.display_statement or node.statement),
                "status": node.status,
                "support_score": node.support_score,
                "level": node.level,
                "parent_id": node.parent_id,
            }
            if latest is not None:
                snapshot["latest_questioning"] = {
                    "round": latest.round,
                    "impact_direction": latest.impact_direction,
                    "support_after": latest.support_after,
                    "rationale": latest.rationale,
                    "falsification_basis": latest.falsification_basis or "",
                }
            return snapshot

        hypothesis_nodes = [
            _node_snapshot(node) for node in tree.nodes if node.hypothesis_id in target_set
        ]
        retained_nodes = [
            _node_snapshot(node) for node in tree.nodes if node.hypothesis_id not in target_set
        ]
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
            "targeted_hypotheses": hypothesis_nodes,
            "retained_hypotheses": retained_nodes,
            "rag_notes": rag_context.guidance_notes()[:4],
            "mined_candidates": (mined_candidates or [])[:8],
        }
        return self.gateway.generate_structured(
            system_prompt=self.QUESTIONING_SYSTEM_PROMPT,
            user_prompt=(
                f"context={compact}\n"
                "逐条输出 hypothesis_updates：只对 targeted_hypotheses 中每一个 hypothesis_id"
                "给出一项更新；retained_hypotheses 是已继承的稳定节点，本轮不重新质询，"
                "禁止为 retained_hypotheses 输出 hypothesis_updates。"
                "impact_direction 只能是 supports / weakens / clarifies，"
                "impact_strength 与 confidence 均为 0-1 小数，rationale 用自然物理语言说明判断依据；"
                "选择 weakens 时必须填写 falsification_basis，给出可证否的具体依据"
                "（证据缺口、违反物理规律或与真实实验结果矛盾），只表达担忧而没有依据时只能选 clarifies。\n"
                "继承说明：targeted_hypotheses 与 retained_hypotheses 里的 latest_questioning"
                "是历史质询结果，供你在已有判断基础上给出增量意见；未重新质询的节点保持原状。\n"
                "科学书写规则：challenge_points 与 proposed_uncertainties 必须写成自然、专业、物理语义清晰的文本，"
                "明确自变量、因变量与物理路径关系；禁止固定填空句式，禁止机械复述字段标签。\n"
                "名称规则：文本与 features 一律使用 display_dictionary 中的展示名，禁止输出原始表头或英文单位标签。"
            ),
            response_model=ScientificQuestionerResponse,
            payload_fixer=self._fix_payload,
            fallback_factory=lambda: self._fallback(
                planner_input,
                rag_context,
                mined_candidates,
                hypothesis_nodes=hypothesis_nodes,
            ),
            role="科学质询者llm",
        )

    @staticmethod
    def _fallback(
        planner_input: ReasoningPlannerInput,
        rag_context: RAGContextBundle,
        mined_candidates: list[dict[str, object]] | None = None,
        hypothesis_nodes: list[dict[str, object]] | None = None,
    ) -> ScientificQuestionerResponse:
        uncertainties: list[ProposedUncertainty] = []
        challenges: list[str] = []
        notes: list[str] = []
        seen_questions: set[str] = set()

        def _add(item: ProposedUncertainty) -> None:
            normalized = "".join((item.question or "").lower().split())
            if not normalized or normalized in seen_questions:
                return
            seen_questions.add(normalized)
            uncertainties.append(item)

        summary = planner_input.data_dictionary_summary
        semantic = VariableSemanticService.from_summary(summary)
        focus_fields = list(summary.display_feature_candidates or summary.feature_candidates)[:8]
        target = planner_input.target
        display_target = semantic.to_display(target)
        hypothesis_snapshots = planner_input.active_hypotheses[:8]
        primary = focus_fields[0] if focus_fields else display_target
        by_feature = next(
            (
                feature
                for feature in focus_fields
                if feature != primary
                and any(token in feature for token in ("行星际磁场Y", "行星际磁场", "BY", "By"))
            ),
            focus_fields[1] if len(focus_fields) > 1 else display_target,
        )

        disagreement_updates = planner_input.recent_disagreement_updates[:6]
        reasoning_traces = planner_input.recent_reasoning_traces[-6:]
        if planner_input.evaluation_summary.stable is False:
            challenges.append(f"上一轮结果稳定性不足，需要继续检验 {primary} 对 {display_target} 的作用是否只在局部时间窗成立。")
            _add(
                ProposedUncertainty(
                    question=f"{primary} 对 {display_target} 的预测增益是否只在局部时间窗口成立？",
                    description=f"需要围绕 {primary} 与 {display_target} 的关系增加时间片或稳定性验证，避免把局部提升误判为普遍规律。",
                    priority="high",
                )
            )
        for update in disagreement_updates:
            challenges.append(f"{update.uncertainty_id} 对应的问题仍未解决，需要继续设计区分性验证。")
            _add(
                ProposedUncertainty(
                    question=(
                        f"上一轮实验中，{primary} 对 {display_target} 的预测增益在改变评价窗口或"
                        f"控制 {by_feature} 后是否仍然存在，还是由参数敏感性或残差结构变化主导？"
                    ),
                    description=(
                        f"{update.uncertainty_id} 仍未关闭，需要继续围绕 {primary}、{by_feature} 与 "
                        f"{display_target} 的关系做窗口敏感性、残差来源与竞争解释验证。"
                    ),
                    priority="high",
                )
            )
        for index, trace in enumerate(reasoning_traces, start=1):
            if not trace.summary:
                continue
            focus = focus_fields[(index - 1) % len(focus_fields)] if focus_fields else display_target
            _add(
                ProposedUncertainty(
                    question=(
                        f"控制 {focus} 之后，{primary} 对 {display_target} 的预测残差是否仍然集中在"
                        f"特定时段，提示日志中未覆盖的系统性偏差？"
                    ),
                    description=(
                        f"推理日志提示：{trace.summary[:80]}。需要确认 {primary} 是否只是代理变量，"
                        f"残留误差是否仍与 {focus} 或时间滞后结构相关。"
                    ),
                    priority="medium",
                )
            )
        for candidate in mined_candidates or []:
            question = str(candidate.get("question") or "")
            description = str(candidate.get("description") or "")
            if not question:
                continue
            _add(
                ProposedUncertainty(
                    question=question,
                    description=description,
                    priority=str(candidate.get("priority") or "medium"),
                    mining_sources=[
                        str(item)
                        for item in (candidate.get("source_labels") or [])
                    ],
                    related_hypotheses=[
                        str(item)
                        for item in (candidate.get("related_hypotheses") or [])
                    ],
                    features=[
                        str(item)
                        for item in (candidate.get("features") or [])
                    ],
                )
            )
        if rag_context.project_evidence:
            notes.append(f"questioner_project:{rag_context.project_evidence[0].excerpt}")
        if rag_context.literature_evidence:
            notes.append(f"questioner_literature:{rag_context.literature_evidence[0].excerpt}")
        natural_followups = [
            (
                f"{primary} 对 {display_target} 的预测作用是否只在特定太阳活动时段成立，"
                "而不是全时段的普遍规律？",
                f"需要结合时间片与太阳活动状态检验 {primary} 的作用是否稳定。",
            ),
            (
                f"控制 {by_feature} 之后，{primary} 对 {display_target} 的额外预测能力是否依然存在？",
                f"需要构造以 {by_feature} 为对照变量的实验，区分独立增量与伴随路径。",
            ),
            (
                f"{primary} 的预测增益是否能够提前 {display_target} 的变化，还是仅与同期观测相关？",
                f"需要对比不同超前窗口下 {primary} 与 {display_target} 的预测关系。",
            ),
        ]
        for question_text, description_text in natural_followups:
            if len(uncertainties) >= 4:
                break
            _add(
                ProposedUncertainty(
                    question=question_text,
                    description=description_text,
                    priority="medium",
                    related_hypotheses=[
                        snapshot.hypothesis_id for snapshot in hypothesis_snapshots[:1] if snapshot.hypothesis_id
                    ],
                )
            )
        if not uncertainties:
            _add(
                ProposedUncertainty(
                    question=f"{primary} 能否为 {display_target} 预测提供超越现有对照组特征的增量信息？",
                    description=f"需要围绕现有字段 {primary} 与 {display_target} 补充一个竞争性不确定性，避免闭环只沿单一路径收敛。",
                    priority="medium",
                )
            )
        hypothesis_updates = [
            HypothesisQuestioningUpdate(
                hypothesis_id=str(node.get("hypothesis_id") or ""),
                impact_direction=(
                    "weakens"
                    if str(node.get("status") or "") == "pruned"
                    else "supports" if float(node.get("support_score") or 0.0) >= 0.70 else "clarifies"
                ),
                impact_strength=(
                    0.45
                    if str(node.get("status") or "") == "pruned"
                    else 0.30 if float(node.get("support_score") or 0.0) >= 0.70 else 0.22
                ),
                confidence=0.65,
                rationale=(
                    f"科学质询应对“{node.get('statement') or node.get('hypothesis_id') or ''}”给出明确检验口径；"
                    "当前支持度与状态需要在进入不确定性识别前再确认一次。"
                ),
                falsification_basis=(
                    "该假设此前已因证据不足被剪枝，本轮不改变其状态。"
                    if str(node.get("status") or "") == "pruned"
                    else ""
                ),
            )
            for node in hypothesis_nodes or []
            if str(node.get("hypothesis_id") or "")
        ]
        return ScientificQuestionerResponse(
            challenge_points=(challenges or ["当前证据仍需通过竞争解释和稳健性验证进一步约束。"])[:10],
            guidance_notes=notes,
            proposed_uncertainties=uncertainties[:10],
            hypothesis_updates=hypothesis_updates,
        )

    SYSTEM_PROMPT = """你是“逐影 Shadow Tracing”的科学质询顾问。

你的任务是对冻结后的假设树开展定向科学质询：作为顾问提供支持、削弱或澄清意见，
同时识别证据缺口、替代解释和仍需验证的问题。你的质询意见是 advisory，不是最终裁决：
单次弱化最多把活跃节点降至待观察，不能直接剪枝；只有真实弹性回归实验结果或人工 PI 覆盖
才是强证据，能驱动剪枝与收敛。
必须严格为 prompt 中 targeted_hypotheses 里的每一个 hypothesis_id 输出一条 hypothesis_updates，
retained_hypotheses 只是继承上下文，禁止为其输出更新；
impact_direction 只能是 supports / weakens / clarifies，
impact_strength 与 confidence 均为 0-1 小数，rationale 必须写明物理判断依据。
选择 weakens 时必须同时填写 falsification_basis，给出可证否的具体依据
（证据缺口、已违反的物理规律、或与真实实验结果矛盾）；只有“表达担忧”而无具体依据时，
最多只能选 clarifies 并禁止产生负向分数，不允许为了削弱而削弱。
所有变量名称必须严格来自 planner_input.data_dictionary_summary 中已有的展示名，
不允许发明数据表里不存在的新变量，也不要机械复述字段标签。
challenge_points 和 proposed_uncertainties 要写成自然、专业、可直接展示的物理语言。
不要直接修改状态文件，只给出 challenge_points、guidance_notes、proposed_uncertainties、hypothesis_updates。""" + "\n\n" + ELASTIC_NET_EXECUTION_RULE

    QUESTIONING_SYSTEM_PROMPT = SYSTEM_PROMPT


def _allowed_display_fields(planner_input: ReasoningPlannerInput) -> list[str]:
    summary = planner_input.data_dictionary_summary
    display_features = list(summary.display_feature_candidates or summary.feature_candidates)
    display_targets = list(summary.display_target_candidates or summary.target_candidates)
    display_time = summary.display_time_column or summary.time_column
    return [*display_features, *display_targets, display_time]


def _nonempty_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return ""
    return str(value).strip()


def _text_or_json(value: Any) -> str:
    text = _nonempty_text(value)
    if text:
        return text
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return ""


def _display_dictionary_block(summary) -> str:
    display_features = list(summary.display_feature_candidates or summary.feature_candidates)
    display_targets = list(summary.display_target_candidates or summary.target_candidates)
    display_time = summary.display_time_column or summary.time_column
    return (
        "{"
        f'"dataset_name": "{summary.dataset_name}", '
        f'"display_time_column": "{display_time}", '
        f'"display_target_candidates": {display_targets}, '
        f'"display_feature_candidates": {display_features}, '
        f'"raw_display_map": {dict(summary.raw_display_map or {})}'
        "}"
    )
