from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.unified_schema import (
    DataDictionary,
    EvidenceItem,
    EvidenceType,
    HypothesisGenerationRationale,
    HypothesisNode,
    HypothesisTreeState,
    LatestTreeUpdate,
    PredictionRecord,
    ReasoningPlannerInput,
    ScientificTask,
    SupportHistoryEntry,
    TreeSummary,
    UncertaintyRecord,
    UncertaintyState,
)
from core.variable_semantic_service import VariableSemanticService


CANONICAL_ID_BY_DISPLAY = {
    "H1": "H_shadow_incremental_gain",
    "H2": "H_by_mediated_path",
    "H3": "H_by_beyond_effect",
    "H4": "H_lead_time_window",
    "H5": "H_window_stability",
}
DISPLAY_BY_CANONICAL = {value: key for key, value in CANONICAL_ID_BY_DISPLAY.items()}
PROGRAMMATIC_BASE_SUPPORT = {
    "H1": 0.50,
    "H2": 0.45,
    "H3": 0.42,
    "H4": 0.40,
    "H5": 0.38,
}
EVIDENCE_ADJUSTMENTS = {
    "literature": 0.25,
    "observational_data": 0.20,
    "physical_law": 0.15,
    "physical_prior": 0.15,
    "physical_reasoning": 0.10,
    "expert_judgment": 0.00,
    "evidence_gap": -0.10,
}
DEFAULT_TWO_LAYER_PARENT = {
    "H1": None,
    "H2": None,
    "H3": CANONICAL_ID_BY_DISPLAY["H1"],
    "H4": CANONICAL_ID_BY_DISPLAY["H1"],
    "H5": CANONICAL_ID_BY_DISPLAY["H1"],
}
DIVERGENCE_GAP_THRESHOLD = 0.10
TARGETED_STATUSES = frozenset({"draft", "observing", "pending"})


@dataclass(frozen=True)
class HypothesisGenerationResult:
    tree: HypothesisTreeState
    generated_node_ids: tuple[str, ...]
    updated_uncertainties: tuple[UncertaintyRecord, ...]
    mode: str
    model: str | None = None
    source: str = "real_llm"


def refresh_tree_metadata(tree: HypothesisTreeState) -> None:
    """Recompute tree-level active/pruned/pending lists and summary from node statuses."""
    tree.active_hypotheses = [
        node.hypothesis_id
        for node in tree.nodes
        if node.status in {"active", "converged"}
    ]
    tree.pruned_hypotheses = [
        node.hypothesis_id for node in tree.nodes if node.status == "pruned"
    ]
    tree.pending_hypotheses = [
        node.hypothesis_id
        for node in tree.nodes
        if node.status in {"draft", "pending"}
    ]
    tree.tree_summary = TreeSummary(
        total_nodes=len(tree.nodes),
        active_count=len(tree.active_hypotheses),
        pruned_count=len(tree.pruned_hypotheses),
        pending_count=len(tree.pending_hypotheses),
    )


def compute_targeted_interrogation_scope(
    tree: HypothesisTreeState | None,
    *,
    current_round: int,
) -> set[str]:
    """Return hypothesis ids that need re-questioning in the current round.

    Inherited, already-questioned stable nodes are deliberately excluded so
    their questioning traces stay untouched; the LLM only re-examines:
    - nodes created in this round or never questioned before,
    - weak nodes still in draft/observing/pending,
    - the closest-split level-1 competing hypotheses from the inherited tree.
    """
    if tree is None or not tree.nodes:
        return set()
    targeted: set[str] = set()
    root_candidates: list[HypothesisNode] = []
    for node in tree.nodes:
        if node.status == "pruned":
            continue
        if node.created_at_round >= current_round or not node.questioning_records:
            targeted.add(node.hypothesis_id)
        if node.status in TARGETED_STATUSES:
            targeted.add(node.hypothesis_id)
        if node.level <= 1:
            root_candidates.append(node)
    if len(root_candidates) >= 2:
        root_candidates.sort(key=lambda node: node.support_score)
        min_gap = min(
            round(next_node.support_score - node.support_score, 4)
            for node, next_node in zip(root_candidates, root_candidates[1:])
        )
        for node, next_node in zip(root_candidates, root_candidates[1:]):
            gap = round(next_node.support_score - node.support_score, 4)
            if gap <= min_gap + 1e-9:
                targeted.update({node.hypothesis_id, next_node.hypothesis_id})
    return targeted


class HypothesisGenerationService:
    """Build and rebuild hypothesis trees exclusively from real LLM proposals."""

    def append_supplemental_node(
        self,
        *,
        task: ScientificTask,
        data_dictionary: DataDictionary,
        tree: HypothesisTreeState,
        proposal: dict[str, Any],
        current_round: int,
        min_active_hypotheses: int,
    ) -> HypothesisNode:
        """Append one real-LLM supplemental hypothesis and refresh tree metadata."""
        display_id = str(proposal.get("display_hypothesis_id") or "").strip()
        if not display_id:
            raise ValueError("真实 LLM 补充假设缺少 display_hypothesis_id。")
        if any(node.display_hypothesis_id == display_id for node in tree.nodes):
            raise ValueError(f"补充假设编号 {display_id} 与现有节点重复。")
        canonical_id = f"H_supplemental_{display_id}"
        if any(node.hypothesis_id == canonical_id for node in tree.nodes):
            raise ValueError(f"补充假设 canonical id {canonical_id} 重复。")

        semantic = VariableSemanticService.from_data_dictionary(data_dictionary)
        statement = semantic.display_text(str(proposal.get("statement") or "").strip())
        if not statement:
            raise ValueError("真实 LLM 补充假设缺少 statement。")
        generated_at = _extract_generated_at([proposal])
        node = HypothesisNode(
            hypothesis_id=canonical_id,
            display_hypothesis_id=display_id,
            statement=statement,
            level=max(1, int(proposal.get("level") or 1)),
            parent_id=None,
            children_ids=[],
            status="active",
            support_score=0.40,
            support_history=[
                SupportHistoryEntry(
                    round=current_round,
                    score=0.40,
                    event="llm_supplemental_generated",
                    evidence_source="advisory",
                )
            ],
            activation_condition=(
                str(proposal.get("activation_condition") or "").strip()
                or f"始终激活：科学质询后补充的竞争根假设 {display_id}。"
            ),
            activated_at_round=current_round,
            evidence_items=_coerce_evidence_items(
                proposal=proposal,
                canonical_id=canonical_id,
                current_round=current_round,
                semantic=semantic,
            ),
            predictions=_coerce_predictions(
                proposal=proposal,
                canonical_id=canonical_id,
            ),
            falsification_conditions=_coerce_falsifications(proposal, semantic),
            alternative_explanations=[
                semantic.display_text(str(item))
                for item in (proposal.get("alternative_explanations") or [])
                if str(item).strip()
            ],
            created_at_round=current_round,
            updated_at_round=current_round,
            generation_rationale=_build_llm_rationale(
                display_id=display_id,
                canonical_id=canonical_id,
                initial_support=0.40,
                model=str(proposal.get("model") or ""),
                source=str(proposal.get("source") or "real_llm"),
                generated_at=generated_at,
            ),
        )
        _apply_display_names([node], data_dictionary)
        tree.nodes.append(node)
        _wire_children(tree.nodes)
        if generated_at is not None:
            tree.generated_at = generated_at
        refresh_tree_metadata(tree)
        tree.latest_update = LatestTreeUpdate(
            round=current_round,
            event="supplemental_hypothesis_added",
            description=(
                f"科学质询后活跃假设不足 {min_active_hypotheses} 条，"
                f"已由真实 LLM 补充生成 {display_id}。"
            ),
        )
        return node

    def rebuild_tree(
        self,
        *,
        task: ScientificTask,
        nodes: list[HypothesisNode],
        current_round: int,
        event: str = "hypothesis_tree_confirmed",
        description: str = "hypothesis tree frozen by human_pi after H/C review",
        data_dictionary: DataDictionary | None = None,
    ) -> HypothesisTreeState:
        """Rebuild a validated tree after human-pi confirmation edits."""
        tree = _build_tree_state(
            task=task,
            nodes=nodes,
            current_round=current_round,
            event=event,
            description=description,
        )
        if data_dictionary is not None:
            _apply_display_names(tree.nodes, data_dictionary)
        return tree

    def build_tree(
        self,
        *,
        task: ScientificTask,
        data_dictionary: DataDictionary,
        uncertainties: list[UncertaintyRecord] | UncertaintyState | None = None,
        planner_input: ReasoningPlannerInput | None = None,
        existing_tree: HypothesisTreeState | None = None,
        current_round: int | None = None,
    ) -> HypothesisGenerationResult:
        """Build the current-round tree from real LLM proposals.

        Round 1 starts from LLM drafts: H1/H2 are active and H3/H4/H5 are gray
        draft nodes at 0.4. From round 2 onward, the tree inherits the previous
        round-end support scores, statuses, and support history while the LLM
        rewrites the scientific text on top of that inherited tree.
        """
        current_round = (
            current_round
            if current_round is not None
            else (planner_input.next_round_id if planner_input else (existing_tree.current_round if existing_tree else 0))
        )
        inheriting = current_round > 1 and existing_tree is not None and bool(existing_tree.nodes)
        uncertainty_records = _normalize_uncertainties(uncertainties)
        proposals = list(planner_input.llm_hypothesis_proposals) if planner_input else []
        if not proposals:
            raise ValueError(
                "本轮没有真实 LLM 假设提案，已禁止程序化/本地兜底生成；"
                "请等待 LLM 完成 H1..H5 假设生成后重试。"
            )
        nodes, generated_node_ids, generated_at = _nodes_from_llm_proposals(
            task=task,
            data_dictionary=data_dictionary,
            proposals=proposals,
            current_round=current_round,
            existing_tree=existing_tree,
        )
        tree = _build_tree_state(
            task=task,
            nodes=nodes,
            current_round=current_round,
            event="llm_hypothesis_generated",
            description=(
                (
                    f"真实 LLM 于 {generated_at.isoformat() if generated_at else '未知时间'} 生成 5 条假设，"
                    "在上一轮轮末树上继承支持度、状态与支持度历史后生成当前轮假设树。"
                )
                if inheriting
                else (
                    "真实 LLM 于 "
                    f"{generated_at.isoformat() if generated_at else '未知时间'} 生成 5 条假设，"
                    "并重建当前轮假设树。"
                )
            ),
            generated_at=generated_at,
        )
        _apply_display_names(tree.nodes, data_dictionary)
        updated_uncertainties = tuple(self._sync_uncertainty_links(tree.nodes, uncertainty_records))
        mode = "llm_initial" if current_round <= 1 else "llm_next_round"
        model = next(
            (str(item.get("model") or "").strip() for item in proposals if item.get("model")),
            None,
        )
        return HypothesisGenerationResult(
            tree=tree,
            generated_node_ids=generated_node_ids,
            updated_uncertainties=updated_uncertainties,
            mode=mode,
            model=model,
            source="real_llm",
        )

    def _sync_uncertainty_links(
        self,
        nodes: list[HypothesisNode],
        uncertainties: list[UncertaintyRecord],
    ) -> list[UncertaintyRecord]:
        if not uncertainties:
            return []
        updated = [item.model_copy(deep=True) for item in uncertainties]
        for record in updated:
            linked: list[str] = list(record.related_hypotheses)
            text = f"{record.question} {record.description}".lower()
            for node in nodes:
                rationale = node.generation_rationale
                if rationale and record.uncertainty_id in rationale.linked_uncertainties:
                    linked.append(node.hypothesis_id)
                    continue
                node_text = (node.display_statement or node.statement or "").lower()
                if node_text and node_text.rstrip("？?。") in text:
                    linked.append(node.hypothesis_id)
            record.related_hypotheses = _unique_preserve_order(linked)
        return updated


def _find_inherited_node(
    existing_tree: HypothesisTreeState | None,
    *,
    display_id: str,
    canonical_id: str,
) -> HypothesisNode | None:
    """Locate the previous round-end node by display or canonical id."""
    if existing_tree is None or not existing_tree.nodes:
        return None
    for node in existing_tree.nodes:
        if node.display_hypothesis_id == display_id or node.hypothesis_id == canonical_id:
            return node
    return None


def _merge_evidence_items(
    *,
    existing: list[EvidenceItem],
    new: list[EvidenceItem],
) -> list[EvidenceItem]:
    """Keep accumulated evidence and append only non-duplicate LLM evidence."""
    merged = [item.model_copy(deep=True) for item in existing]
    seen = {item.id for item in merged}
    for item in new:
        if item.id in seen:
            continue
        merged.append(item)
        seen.add(item.id)
    return merged


def _merge_texts(existing: list[str], new: list[str]) -> list[str]:
    """Merge text lists while preserving order and dropping exact duplicates."""
    merged = list(existing)
    for text in new:
        if text and text not in merged:
            merged.append(text)
    return merged


def _nodes_from_llm_proposals(
    *,
    task: ScientificTask,
    data_dictionary: DataDictionary,
    proposals: list[dict],
    current_round: int,
    existing_tree: HypothesisTreeState | None = None,
) -> tuple[list[HypothesisNode], tuple[str, ...], datetime | None]:
    """Map the five LLM H1..H5 proposals into the current-round tree."""
    if len(proposals) != 5:
        raise ValueError(
            f"真实 LLM 返回 {len(proposals)} 条假设，必须恰好 5 条（H1..H5）。"
        )
    semantic = VariableSemanticService.from_data_dictionary(data_dictionary)
    generated_at = _extract_generated_at(proposals)
    inherit_existing = existing_tree is not None and current_round > 1 and bool(existing_tree.nodes)
    nodes: list[HypothesisNode] = []
    node_by_proposal = {str(item.get("display_hypothesis_id") or "").strip(): item for item in proposals}
    for display_id in ("H1", "H2", "H3", "H4", "H5"):
        proposal = node_by_proposal.get(display_id)
        if proposal is None:
            raise ValueError(f"真实 LLM 假设提案缺少 {display_id}。")
        canonical_id = CANONICAL_ID_BY_DISPLAY[display_id]
        statement = semantic.display_text(str(proposal.get("statement") or "").strip())
        if not statement:
            raise ValueError(f"真实 LLM 假设提案 {display_id} 缺少 statement。")
        inherited_node = (
            _find_inherited_node(
                existing_tree,
                display_id=display_id,
                canonical_id=canonical_id,
            )
            if inherit_existing
            else None
        )
        new_evidence_items = _coerce_evidence_items(
            proposal=proposal,
            canonical_id=canonical_id,
            current_round=current_round,
            semantic=semantic,
        )
        new_alternative_explanations = [
            semantic.display_text(str(item))
            for item in (proposal.get("alternative_explanations") or [])
            if str(item).strip()
        ]
        if inherited_node is not None:
            # 第二轮起继承上一轮轮末的支持度、状态、支持度历史，并保留历史科学
            # 质询记录与批判痕迹；新一轮只对定向作用域内的节点补充质询。
            initial_support = round(inherited_node.support_score, 3)
            status = str(inherited_node.status)
            support_history = [
                entry.model_copy(deep=True) for entry in inherited_node.support_history
            ]
            support_history.append(
                SupportHistoryEntry(
                    round=current_round,
                    score=initial_support,
                    event="round_continuation",
                    evidence_source="advisory",
                )
            )
            questioning_records = [
                record.model_copy(deep=True)
                for record in inherited_node.questioning_records
            ]
            evidence_items = _merge_evidence_items(
                existing=inherited_node.evidence_items,
                new=new_evidence_items,
            )
            evidence_against = [
                item.model_copy(deep=True) for item in inherited_node.evidence_against
            ]
            critiques = [
                record.model_copy(deep=True) for record in inherited_node.critiques
            ]
            alternative_explanations = _merge_texts(
                inherited_node.alternative_explanations,
                new_alternative_explanations,
            )
            created_at_round = inherited_node.created_at_round
            activated_at_round = inherited_node.activated_at_round
            pruned_at_round = inherited_node.pruned_at_round
            prune_reason = inherited_node.prune_reason
        else:
            # 生成阶段：一级父假设直接进入活跃池参与竞争；子女假设先以草稿身份展示，
            # 支持度统一记为 0.4，待科学质询完成后再由状态机转出活跃/待定/观察等状态。
            initial_support = (
                _programmatic_initial_support(display_id=display_id, proposal=proposal)
                if display_id in {"H1", "H2"}
                else 0.4
            )
            status = "active" if display_id in {"H1", "H2"} else "draft"
            support_history = [
                SupportHistoryEntry(
                    round=current_round,
                    score=round(initial_support, 3),
                    event="llm_hypothesis_generated",
                    evidence_source="advisory",
                )
            ]
            questioning_records = []
            evidence_items = new_evidence_items
            evidence_against = []
            critiques = []
            alternative_explanations = new_alternative_explanations
            created_at_round = current_round
            activated_at_round = current_round if status == "active" else None
            pruned_at_round = None
            prune_reason = None
        activation_condition = _coerce_activation_condition(proposal, display_id)
        if inherited_node is not None and not str(proposal.get("activation_condition") or "").strip():
            activation_condition = inherited_node.activation_condition
        node = HypothesisNode(
            hypothesis_id=canonical_id,
            display_hypothesis_id=display_id,
            statement=statement,
            level=2 if display_id in {"H3", "H4", "H5"} else 1,
            parent_id=DEFAULT_TWO_LAYER_PARENT[display_id],
            children_ids=[],
            status=status,
            support_score=round(initial_support, 3),
            support_history=support_history,
            activation_condition=activation_condition,
            activated_at_round=activated_at_round,
            evidence_items=evidence_items,
            evidence_against=evidence_against,
            critiques=critiques,
            questioning_records=questioning_records,
            predictions=_coerce_predictions(
                proposal=proposal,
                canonical_id=canonical_id,
            ),
            falsification_conditions=_coerce_falsifications(proposal, semantic),
            alternative_explanations=alternative_explanations,
            created_at_round=created_at_round,
            updated_at_round=current_round,
            pruned_at_round=pruned_at_round,
            prune_reason=prune_reason,
            generation_rationale=_build_llm_rationale(
                display_id=display_id,
                canonical_id=canonical_id,
                initial_support=initial_support,
                model=str(proposal.get("model") or ""),
                source=str(proposal.get("source") or "real_llm"),
                generated_at=generated_at,
                inherited=inherited_node is not None,
            ),
        )
        nodes.append(node)
    _wire_children(nodes)
    return nodes, tuple(node.hypothesis_id for node in nodes), generated_at


def _wire_children(nodes: list[HypothesisNode]) -> None:
    parent_index: dict[str | None, list[HypothesisNode]] = {}
    for node in nodes:
        parent_index.setdefault(node.parent_id, []).append(node)
    for node in nodes:
        node.children_ids = [child.hypothesis_id for child in parent_index.get(node.hypothesis_id, [])]


def _extract_generated_at(proposals: list[dict]) -> datetime | None:
    for proposal in proposals:
        raw = proposal.get("generated_at")
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
    return datetime.now()


def _programmatic_initial_support(display_id: str, proposal: dict) -> float:
    """Compute initial support from evidence type weights; LLM numbers are ignored."""
    score = PROGRAMMATIC_BASE_SUPPORT.get(display_id, 0.40)
    for raw in proposal.get("evidence_items") or []:
        if not isinstance(raw, dict):
            continue
        evidence_type = str(raw.get("evidence_type") or "physical_reasoning").strip().lower()
        evidence_type = evidence_type.replace(" ", "_")
        score += EVIDENCE_ADJUSTMENTS.get(evidence_type, 0.00)
    return max(0.05, min(0.95, round(score, 3)))


def _coerce_activation_condition(proposal: dict, display_id: str) -> str:
    raw = str(proposal.get("activation_condition") or "").strip()
    if raw:
        return raw
    if display_id in {"H1", "H2"}:
        return "始终激活：当前轮竞争性根假设。"
    return "当父假设获得初步支持（支持度>=0.40）后由系统激活。"


def _coerce_evidence_items(
    *,
    proposal: dict,
    canonical_id: str,
    current_round: int,
    semantic: VariableSemanticService,
) -> list[EvidenceItem]:
    items = proposal.get("evidence_items") or []
    result: list[EvidenceItem] = []
    for index, raw in enumerate(items, start=1):
        if not isinstance(raw, dict):
            continue
        description = semantic.display_text(str(raw.get("description") or "").strip())
        if not description:
            continue
        result.append(
            EvidenceItem(
                id=f"{canonical_id}_ev{index}",
                type=_coerce_evidence_type(str(raw.get("evidence_type") or "physical_reasoning")),
                description=description,
                source=str(raw.get("source") or "LLM提案"),
                added_at_round=current_round,
                support_weight=_evidence_weight(str(raw.get("evidence_type") or "physical_reasoning")),
            )
        )
    return result


def _coerce_evidence_type(value: str) -> EvidenceType:
    normalized = value.strip().replace(" ", "_").lower()
    mapping = {
        "literature": "literature",
        "observational_data": "observational_data",
        "experimental_result": "experimental_result",
        "physical_law": "physical_prior",
        "physical_reasoning": "physical_prior",
        "physical_prior": "physical_prior",
        "expert_judgment": "expert_judgment",
        "evidence_gap": "expert_judgment",
    }
    return mapping.get(normalized, "physical_prior")


def _evidence_weight(value: str) -> float:
    normalized = value.strip().replace(" ", "_").lower()
    weights = {
        "literature": 0.8,
        "observational_data": 0.7,
        "experimental_result": 0.7,
        "physical_law": 0.9,
        "physical_reasoning": 0.5,
        "physical_prior": 0.5,
        "expert_judgment": 0.5,
    }
    return weights.get(normalized, 0.5)


def _coerce_predictions(
    *,
    proposal: dict,
    canonical_id: str,
) -> list[PredictionRecord]:
    items = proposal.get("predictions") or []
    result: list[PredictionRecord] = []
    for index, raw in enumerate(items, start=1):
        if not isinstance(raw, dict):
            continue
        expected_range = raw.get("expected_range")
        if isinstance(expected_range, list) and len(expected_range) == 2:
            try:
                expected_range = [float(expected_range[0]), float(expected_range[1])]
            except (TypeError, ValueError):
                expected_range = None
        else:
            expected_range = None
        result.append(
            PredictionRecord(
                experiment_id=f"{canonical_id}_llm_pred{index}",
                metric=_coerce_metric(str(raw.get("expected_observable") or "")),
                expected_direction=_coerce_direction(str(raw.get("expected_direction") or "positive")),
                expected_range=expected_range,
            )
        )
    return result


def _coerce_metric(raw: str) -> str:
    lowered = raw.lower()
    if "pearson" in lowered or "correlation" in lowered:
        return "Pearson_r"
    if "skill" in lowered:
        return "Skill"
    if "rmse" in lowered:
        return "RMSE"
    if "mae" in lowered:
        return "MAE"
    if "r2" in lowered or "r²" in lowered or "r^2" in lowered:
        return "R2"
    return "Pearson_r"


def _coerce_direction(raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized in {"positiva", "positive", "正", "正向"}:
        return "positive"
    if normalized in {"negative", "负", "负向"}:
        return "negative"
    if normalized in {"near_zero", "zero", "接近零", "near zero"}:
        return "near_zero"
    if normalized in {"nonlinear", "非线性"}:
        return "nonlinear"
    return "unknown"


def _coerce_falsifications(
    proposal: dict,
    semantic: VariableSemanticService,
) -> list[str]:
    conditions = [
        semantic.display_text(str(item).strip())
        for item in (proposal.get("falsification_conditions") or [])
        if str(item).strip()
    ]
    single = str(proposal.get("falsification") or "").strip()
    if single and semantic.display_text(single) not in conditions:
        conditions.append(semantic.display_text(single))
    return conditions


def _build_llm_rationale(
    *,
    display_id: str,
    canonical_id: str,
    initial_support: float,
    model: str,
    source: str,
    generated_at: datetime | None,
    inherited: bool = False,
) -> HypothesisGenerationRationale:
    timestamp = generated_at.isoformat() if generated_at else "未知时间"
    return HypothesisGenerationRationale(
        trigger=f"llm_hypothesis_{display_id}",
        summary=(
            f"假设 {display_id} 继承上一轮轮末状态，当前支持度 {initial_support}，"
            "本轮科学质询后按证据调整。"
            if inherited
            else (
                f"假设 {display_id} 为 LLM 起草，当前支持度 {initial_support}，"
                "待科学质询后按证据调整。"
            )
        ),
        linked_uncertainties=[],
        derived_features=[],
        source_signals=[],
        confidence=round(initial_support, 3),
    )


def _normalize_uncertainties(
    uncertainties: list[UncertaintyRecord] | UncertaintyState | None,
) -> list[UncertaintyRecord]:
    if uncertainties is None:
        return []
    if isinstance(uncertainties, UncertaintyState):
        return [item.model_copy(deep=True) for item in uncertainties.records]
    return [item.model_copy(deep=True) for item in uncertainties]


def _apply_display_names(nodes: list[HypothesisNode], data_dictionary: DataDictionary) -> None:
    """Convert user-facing hypothesis text to display names while keeping raw feature links."""
    semantic = VariableSemanticService.from_data_dictionary(data_dictionary)
    for node in nodes:
        node.display_statement = semantic.display_text(node.statement)
        node.statement = node.display_statement
        node.display_hypothesis_id = (
            f"H{node.level}"
            if not node.display_hypothesis_id
            else node.display_hypothesis_id
        )
        node.falsification_conditions = [
            semantic.display_text(text)
            for text in node.falsification_conditions
        ]
        node.alternative_explanations = [
            semantic.display_text(text)
            for text in node.alternative_explanations
        ]
        if node.generation_rationale is not None:
            node.generation_rationale.summary = semantic.display_text(
                node.generation_rationale.summary
            )


def _ensure_min_active_hypotheses(
    nodes: list[HypothesisNode],
    *,
    min_required: int,
    current_round: int,
) -> None:
    active_count = sum(1 for node in nodes if node.status in {"active", "converged"})
    if active_count >= min_required:
        return
    candidates = sorted(
        [node for node in nodes if node.status in {"pending", "observing"}],
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


def _build_tree_state(
    *,
    task: ScientificTask,
    nodes: list[HypothesisNode],
    current_round: int,
    event: str,
    description: str,
    generated_at: datetime | None = None,
) -> HypothesisTreeState:
    return HypothesisTreeState(
        tree_id=f"{task.task_id}_tree",
        task_id=task.task_id,
        current_round=current_round,
        root_question=task.payload.research_question.text,
        generated_at=generated_at,
        nodes=nodes,
        active_hypotheses=[
            node.hypothesis_id
            for node in nodes
            if node.status in {"active", "converged"}
        ],
        pruned_hypotheses=[
            node.hypothesis_id for node in nodes if node.status == "pruned"
        ],
        pending_hypotheses=[
            node.hypothesis_id
            for node in nodes
            if node.status in {"draft", "pending"}
        ],
        latest_update=LatestTreeUpdate(
            round=current_round,
            event=event,
            description=description,
        ),
        tree_summary=TreeSummary(
            total_nodes=len(nodes),
            active_count=sum(
                1 for node in nodes if node.status in {"active", "converged"}
            ),
            pruned_count=sum(1 for node in nodes if node.status == "pruned"),
            pending_count=sum(
                1 for node in nodes if node.status in {"draft", "pending"}
            ),
        ),
    )


def _unique_preserve_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))
