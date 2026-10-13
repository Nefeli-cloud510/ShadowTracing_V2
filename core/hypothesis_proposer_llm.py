from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from core.llm_gateway import LLMGateway
from core.prompt_rules import ELASTIC_NET_EXECUTION_RULE
from core.rag_service import RAGContextBundle
from core.unified_schema import HypothesisTreeState, ReasoningPlannerInput
from core.variable_semantic_service import VariableSemanticService


class LlmPrediction(BaseModel):
    """A falsifiable prediction written by the LLM for one hypothesis."""

    experiment_condition: str = Field(default="")
    expected_observable: str = Field(default="")
    expected_direction: str = Field(default="unknown")
    expected_range: list[float] | None = None


class LlmEvidenceItem(BaseModel):
    """Evidence or evidence gap supplied by the LLM."""

    description: str = Field(min_length=1)
    evidence_type: str = Field(
        default="physical_reasoning",
        description="literature / observational_data / physical_law / physical_reasoning / evidence_gap",
    )
    source: str = Field(default="")


class ProposedHypothesis(BaseModel):
    """One of the five LLM-written hypotheses in H1..H5 order."""

    display_hypothesis_id: str = Field(default="")
    statement: str = Field(min_length=1)
    level: int = Field(default=1, ge=1)
    parent_id: str | None = None
    activation_condition: str = Field(default="")
    initial_support: float | None = Field(default=None, ge=0.0, le=1.0)
    falsification: str = Field(default="")
    expected_effect: str = Field(default="")
    falsification_conditions: list[str] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    predictions: list[LlmPrediction] = Field(default_factory=list)
    evidence_items: list[LlmEvidenceItem] = Field(default_factory=list)


class HypothesisProposerResponse(BaseModel):
    focus_features: list[str] = Field(default_factory=list)
    guidance_notes: list[str] = Field(default_factory=list)
    proposal_summaries: list[str] = Field(default_factory=list)
    proposed_hypotheses: list[ProposedHypothesis] = Field(default_factory=list)
    source: str = Field(default="real_llm")
    model: str | None = None
    generated_at: datetime | None = None


class HypothesisProposerLLM:
    """Real-LLM-only hypothesis proposer.

    There is intentionally no local fallback here: if the LLM call fails or
    returns fewer than five proposals, the workflow fails fast and shows the
    error instead of silently producing hardcoded science.
    """

    SYSTEM_PROMPT = """你是“逐影 Shadow Tracing”科研实验闭环系统的假设提出者。

你的任务是围绕给定的科学问题，结合数据变量库、RAG 证据与上一轮实验反馈，生成且只生成 5 条竞争性科学假设（H1..H5）。假设必须遵循两层结构：
- 第 1 层：H1 回答“宇宙线日影南北偏移等主特征是否对太阳风速度预测带来增量信息”；H2 是主要的竞争解释，例如“观测增益是否主要来自路径变量的伴随相关”；
- 第 2 层：H3、H4、H5 在 H1 初步成立的前提下展开，分别覆盖“超越路径变量的独立信息”“超前预测窗口与路径感应窗口是否一致”“结论对窗口与评价期的稳健性”。

每个假设必须包含 md 设计方案要求的完整结构化字段：
statement、level、parent_id、activation_condition、predictions（含 experiment_condition、expected_observable、expected_direction、expected_range）、falsification_conditions、alternative_explanations、evidence_items（含 description、evidence_type、source）。
initial_support 不要填写数值：首轮由系统程序依据 evidence_items 的证据类型自动加权计算；第二轮及以后由系统直接继承上一轮轮末支持度、状态与支持度历史，你在 initial_support 里填写的任何数值都会被忽略。

科学书写要求：
- statement 必须是自然、专业、可直接展示的科学问句，体现自变量、因变量与物理路径的真实含义；
- 禁止机械复述字段标签，禁止使用“关于 A 与 B 的解释是否仍受约束”这类模板句式；
- 所有对外文本必须使用 display_dictionary 中的展示名（例如“宇宙线日影南北偏移”“太阳风速度”），禁止输出 raw_display_map 左侧的原始 CSV 表头；
- 区分已有证据与推测：evidence_type 只能取 literature / observational_data / physical_law / physical_reasoning / evidence_gap，证据不足就诚实标注；
- 预测必须可检验、可量化（Pearson_r、Skill、RMSE、MAE 等），并给出合理的 expected_range；
- expected_range 必须是 JSON 数字数组，形如 [0.02, 0.08]；禁止把解释文字、百分比描述或单位写进该字段；
- 结合上一轮残差、支持度变化、失败归因与剩余不确定性改写，避免与上一轮文本逐字重复；
- 初始状态与支持度由系统程序管理：第二轮及以后你在继承树（已给出 status/support_score）上保留/修改/新增假设内容，只需输出科学文本，不要填写最终状态，也不要重置或重算支持度。

只输出结构化 JSON，不要修改任何文件，不要输出 JSON 之外的文字。""" + "\n\n" + ELASTIC_NET_EXECUTION_RULE

    SUPPLEMENT_SYSTEM_PROMPT = """你是“逐影 Shadow Tracing”科研实验闭环系统的假设补充生成器。

当前场景：科学质询完成后，竞争假设树中的活跃假设数量低于系统下限，需要你补充生成且只生成 1 条新的竞争性科学假设，用于恢复活跃竞争。

要求：
- 新假设必须与树中现有假设互补，不得逐字重复，也不得只是简单改写现有假设；
- 必须面向当前科学问题，结合现有假设、未解决不确定性与最高支持度假设，提出一个有真实区分度的替代路径或可证伪预测；
- statement 必须是自然、专业、可直接展示的科学问句，明确自变量、因变量与物理路径；
- 所有对外文本必须使用 display_dictionary 中的展示名，禁止输出原始 CSV 表头；
- predictions 必须可检验、可量化，expected_range 必须是 JSON 数字数组；
- evidence_items 诚实区分已有证据与推测；
- 只输出一个结构化 JSON 对象，不要输出 JSON 之外的文字。""" + "\n\n" + ELASTIC_NET_EXECUTION_RULE

    def __init__(self, *, gateway: LLMGateway | None = None) -> None:
        self.gateway = gateway or LLMGateway()

    def propose(
        self,
        *,
        planner_input: ReasoningPlannerInput,
        rag_context: RAGContextBundle,
    ) -> HypothesisProposerResponse:
        if self.gateway.client is None or not self.gateway.api_key:
            raise RuntimeError(
                "真实 LLM 不可用：未配置 DASHSCOPE_API_KEY/BAILIAN_API_KEY，"
                "当前已禁用本地兜底，无法生成科学假设。"
            )
        semantic = VariableSemanticService.from_summary(planner_input.data_dictionary_summary)
        compact = {
            "question": semantic.display_text(planner_input.scientific_question),
            "round": planner_input.next_round_id,
            "display_target": semantic.to_display(planner_input.target),
            "display_dictionary": _display_dictionary_block(planner_input.data_dictionary_summary),
            "human_feedback": planner_input.human_feedback,
            "evaluation": {
                "delta_pearson_r": planner_input.evaluation_summary.delta_pearson_r,
                "stable": planner_input.evaluation_summary.stable,
            },
            "active_hypotheses": [
                {
                    "hypothesis_id": item.hypothesis_id,
                    "statement": semantic.display_text(item.statement),
                    "status": item.status,
                    "support_score": item.support_score,
                }
                for item in planner_input.active_hypotheses
            ],
            "unresolved_uncertainties": [
                {
                    "uncertainty_id": item.uncertainty_id,
                    "question": semantic.display_text(item.question),
                }
                for item in planner_input.unresolved_uncertainties[:3]
            ],
            "planner_guidance": planner_input.planner_guidance[:5],
            "rag_notes": rag_context.guidance_notes()[:4],
        }
        response = self.gateway.generate_structured(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=(
                f"context={compact}\n"
                "输出规则：请严格按照上述系统提示输出 JSON；"
                f"当前轮次为 {planner_input.next_round_id}；若该轮次 > 1，"
                "active_hypotheses 就是上一轮轮末继承树，必须保留其支持度与状态，只改写科学文本；"
                "proposed_hypotheses 恰好 5 条，display_hypothesis_id 依次为 H1..H5；"
                "H1、H2 为第 1 层根节点（parent_id=null），"
                "H3、H4、H5 为第 2 层子节点（parent_id 为 H1）；"
                "每个假设都填写 statement、predictions、falsification_conditions、"
                "alternative_explanations、evidence_items；字段名保持英文，正文用中文或规范物理量名。"
            ),
            response_model=HypothesisProposerResponse,
            payload_fixer=_prune_malformed_evidence_items,
            temperature=0.35,
            role="假设提出者llm",
        )
        response.source = "real_llm"
        response.model = self.gateway.model
        response.generated_at = datetime.now(timezone.utc)
        self._validate_five_proposals(response)
        return response

    def propose_supplemental(
        self,
        *,
        planner_input: ReasoningPlannerInput,
        rag_context: RAGContextBundle,
        tree: HypothesisTreeState,
    ) -> dict[str, Any]:
        """Ask the real LLM for one supplemental hypothesis after questioning."""
        if self.gateway.client is None or not self.gateway.api_key:
            raise RuntimeError(
                "真实 LLM 不可用：未配置 DASHSCOPE_API_KEY/BAILIAN_API_KEY，"
                "当前已禁用本地兜底，无法补充生成科学假设。"
            )
        semantic = VariableSemanticService.from_summary(planner_input.data_dictionary_summary)
        next_display_id = _next_display_hypothesis_id(tree.nodes)
        existing_hypotheses = [
            {
                "display_hypothesis_id": node.display_hypothesis_id or f"H{node.level}",
                "statement": semantic.display_text(node.statement),
                "status": node.status,
                "support_score": node.support_score,
            }
            for node in tree.nodes
        ]
        compact = {
            "question": semantic.display_text(planner_input.scientific_question),
            "display_target": semantic.to_display(planner_input.target),
            "display_dictionary": _display_dictionary_block(planner_input.data_dictionary_summary),
            "existing_hypotheses": existing_hypotheses,
            "unresolved_uncertainties": [
                {
                    "uncertainty_id": item.uncertainty_id,
                    "question": semantic.display_text(item.question),
                }
                for item in planner_input.unresolved_uncertainties[:3]
            ],
            "rag_notes": rag_context.guidance_notes()[:4],
        }
        proposal = self.gateway.generate_structured(
            system_prompt=self.SUPPLEMENT_SYSTEM_PROMPT,
            user_prompt=(
                f"context={compact}\n"
                "输出规则：请输出且只输出 1 条补充假设；"
                f"display_hypothesis_id 必须是 {next_display_id}；"
                "level=1、parent_id=null，作为新的竞争根假设；"
                "必须填写 statement、predictions、falsification_conditions、"
                "alternative_explanations、evidence_items；字段名保持英文，正文用中文或规范物理量名；"
                "initial_support 不要填写数值。"
            ),
            response_model=ProposedHypothesis,
            payload_fixer=_prune_malformed_evidence_items,
            temperature=0.45,
            role="假设提出者llm",
        )
        proposal.display_hypothesis_id = next_display_id
        proposal.level = 1
        proposal.parent_id = None
        proposal.initial_support = None
        if not (proposal.statement or "").strip():
            raise ValueError("真实 LLM 补充假设缺少 statement，请重试。")
        if (proposal.display_hypothesis_id or "").strip() != next_display_id:
            raise ValueError(
                f"真实 LLM 补充假设编号 {proposal.display_hypothesis_id!r} 与要求 {next_display_id} 不一致。"
            )
        data = proposal.model_dump(mode="json", exclude_none=True)
        data.update(
            {
                "source": "real_llm",
                "model": self.gateway.model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return data

    @staticmethod
    def _validate_five_proposals(response: HypothesisProposerResponse) -> None:
        proposals = response.proposed_hypotheses
        if len(proposals) != 5:
            raise ValueError(
                f"真实 LLM 返回了 {len(proposals)} 条假设，未达到 5 条；"
                "请检查模型输出后重试。"
            )
        seen: set[str] = set()
        for proposal in proposals:
            display_id = (proposal.display_hypothesis_id or "").strip()
            if not display_id.startswith("H") or display_id in seen:
                raise ValueError(
                    f"真实 LLM 返回的假设编号 {display_id!r} 无效或重复，必须为互不重复的 H1..H5。"
                )
            seen.add(display_id)
        expected = {"H1", "H2", "H3", "H4", "H5"}
        if seen != expected:
            raise ValueError(
                f"真实 LLM 返回的假设编号 {sorted(seen)} 与 H1..H5 不一致。"
            )


def _prune_malformed_evidence_items(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop evidence entries missing a readable description before pydantic validation."""

    def clean(items: Any) -> list[Any]:
        if not isinstance(items, list):
            return []
        cleaned: list[Any] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if not str(item.get("description") or "").strip():
                continue
            cleaned.append(item)
        return cleaned

    cleaned = dict(payload)
    raw = cleaned.get("evidence_items")
    if isinstance(raw, list):
        cleaned["evidence_items"] = clean(raw)
    proposals = cleaned.get("proposed_hypotheses")
    if isinstance(proposals, list):
        cleaned["proposed_hypotheses"] = [
            _prune_malformed_evidence_items(item) if isinstance(item, dict) else item
            for item in proposals
        ]
    return cleaned


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


def _next_display_hypothesis_id(nodes) -> str:
    used = {
        str(node.display_hypothesis_id or "").strip()
        for node in nodes
        if str(node.display_hypothesis_id or "").strip()
    }
    max_number = 0
    for label in used:
        match = re.fullmatch(r"H(\d+)", label)
        if match:
            max_number = max(max_number, int(match.group(1)))
    candidate = max_number + 1
    while f"H{candidate}" in used:
        candidate += 1
    return f"H{candidate}"


def display_feature_list(planner_input: ReasoningPlannerInput) -> list[str]:
    summary = planner_input.data_dictionary_summary
    return list(summary.display_feature_candidates or summary.feature_candidates)


def display_target(planner_input: ReasoningPlannerInput) -> str:
    summary = planner_input.data_dictionary_summary
    targets = list(summary.display_target_candidates or summary.target_candidates)
    return targets[0] if targets else planner_input.target
