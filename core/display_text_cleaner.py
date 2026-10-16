"""Clean internal tokens out of user-facing narrative text.

The persistence layer intentionally keeps structured identifiers such as
``H_shadow_incremental_gain`` and ``U_LLM_R01_03`` for programmatic mapping.
Before those values are shown to the PI or embedded into an LLM prompt as
natural language, they should be converted into display labels and concise
Chinese descriptions.  This module owns that conversion so every caller uses
the same wording.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from core.hypothesis_identity import remap_hypothesis_id
from core.json_safe import json_dumps
from core.unified_schema import HypothesisTreeState, UncertaintyRecord, UncertaintyState

DISPLAY_UNCERTAINTY_FALLBACK = "相关科学不确定性"

STATUS_LABELS = {
    "active": "活跃",
    "observing": "待观察",
    "draft": "草稿",
    "converged": "收敛",
    "pruned": "已剪枝",
    "pending": "待定",
    "supported": "获得支持",
    "partially_supported": "部分获得支持",
    "weakened": "已削弱",
    "resolved": "已解决",
    "partially_resolved": "部分解决",
    "unresolved": "未解决",
    "clarifies": "仅澄清",
}

RESOLUTION_LABELS = {
    "resolved": "已解决",
    "partially_resolved": "部分解决",
    "unresolved": "未解决",
}

_NOTE_PREFIX_LABELS = (
    ("generated_from:", "来源："),
    ("uncertainty_baseline_feature:", "对照特征："),
    ("probe_lagged_stability_across_windows:", "滞后窗口稳健性探测："),
    ("probe_residual_source_with_focus:", "残差来源探测："),
    ("history_evolved_window_preset:", "窗口预设："),
    ("window_inherited_from_previous_round:", "窗口继承上一轮配置："),
    ("explicit_window_design:", "窗口按实验设计要求指定："),
    ("hypothesis_triggers:", "假设触发来源："),
    ("hypothesis_source_summary:", "假设来源摘要："),
    ("disagreement_hypotheses:", "待区分假设："),
    ("disagreement_triggers:", "区分触发依据："),
    ("design_focus:", "设计焦点："),
    ("llm_candidate_design:", "实验设计来源："),
    ("llm_candidate_design_rationale:", "实验设计理由："),
    ("llm_candidate_prose:", "候选实验表述来源："),
    ("human_feedback:", "人工反馈："),
    ("execution_plan_summary:", "执行方案摘要："),
    ("supplemental_llm_hypothesis:", "LLM 补充假设："),
)

_HYPOTHESIS_RE = re.compile(r"(?<![A-Za-z0-9_])H_[A-Za-z0-9_]+")
_UNCERTAINTY_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:U|U_LLM)[A-Za-z0-9_]*_R\d{2}_\d{2}"
)
_SPAN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])before_span=([0-9.]+)\s*,\s*after_span=([0-9.]+)"
    r"(?:\s*,\s*leading=([^,\s]+))?"
    r"(?:\s*,\s*status=([^\s,]+))?"
)
_LEADING_PATTERN = re.compile(r"(?<![A-Za-z0-9_])leading\s*=\s*([^\s,]+)")
_LEADING_PHRASE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])leading\s+hypothesis(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_SPAN_NAKED_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:before_span|after_span)\s*=\s*([0-9.]+)"
)
_STATUS_PATTERN = re.compile(r"(?<![A-Za-z0-9_])status\s*=\s*([A-Za-z_]+)")
_DIRECTION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:direction_matched|magnitude_matched)"
    r"\s*=\s*(True|False|partial)",
    re.IGNORECASE,
)
_METRIC_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])metric\s*=\s*([A-Za-z0-9_/|]+)"
)
_FORMULA_HYPOTHESIS_RE = re.compile(
    r"(?<![A-Za-z0-9_])H_([ijk])(?![A-Za-z0-9_])"
)
_EFFECT_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(positive|negative|neutral)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_NAKED_STATUS_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(partially_supported|partially_resolved|supported|weakened|resolved|unresolved)"
    r"(?![A-Za-z0-9_])"
)
_NAKED_NARROWED_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])narrowed\s*=\s*(True|False)",
    re.IGNORECASE,
)
_LLM_HYPOTHESIS_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_])llm_hypothesis_(H\d+)(?![A-Za-z0-9_])"
)
_METRIC_LABELS = {
    "pearson_r": "皮尔逊相关系数",
    "pearsonr": "皮尔逊相关系数",
    "skill": "预报技巧评分",
    "rmse": "均方根误差",
    "mae": "平均绝对误差",
    "r2": "决定系数",
}


def _hypothesis_display_label(
    tree: HypothesisTreeState | None,
    hypothesis_id: str,
) -> str | None:
    if tree is None:
        return None
    node = tree.node_index().get(hypothesis_id)
    if node is not None:
        return node.display_hypothesis_id or None
    canonical_id = remap_hypothesis_id(tree, hypothesis_id)
    if canonical_id != hypothesis_id:
        canonical = tree.node_index().get(canonical_id)
        if canonical is not None:
            return canonical.display_hypothesis_id or None
    return None


def _uncertainty_question(
    uncertainty_records: list[UncertaintyRecord | Any],
    uncertainty_id: str,
) -> str | None:
    for record in uncertainty_records or []:
        if getattr(record, "uncertainty_id", None) == uncertainty_id:
            question = getattr(record, "question", "") or ""
            return question.strip() or None
    return None


def _replace_span_tokens(text: str) -> str:
    """Rewrite disagreement machine summaries into natural Chinese."""

    def replace_span(match: re.Match[str]) -> str:
        before = match.group(1)
        after = match.group(2)
        leading = match.group(3)
        status = match.group(4)
        parts = [f"支持度跨度由 {before} 变化至 {after}"]
        if leading:
            parts.append(f"前沿假设为 {leading}")
        if status:
            parts.append(f"状态为 {STATUS_LABELS.get(status, status)}")
        return "，".join(parts)

    text = _SPAN_PATTERN.sub(replace_span, text)

    def replace_leading(match: re.Match[str]) -> str:
        return f"前沿假设为 {match.group(1)}"

    text = _LEADING_PATTERN.sub(replace_leading, text)
    text = _LEADING_PHRASE_PATTERN.sub("前沿假设", text)

    def replace_span_value(match: re.Match[str]) -> str:
        name = "before" if match.group(0).startswith("before") else "after"
        return f"{name}跨度 {match.group(1)}"

    text = _SPAN_NAKED_PATTERN.sub(replace_span_value, text)

    def replace_status(match: re.Match[str]) -> str:
        return STATUS_LABELS.get(match.group(1), match.group(1))

    text = _STATUS_PATTERN.sub(replace_status, text)

    def replace_direction(match: re.Match[str]) -> str:
        name = "方向" if match.group(0).lower().startswith("direction") else "幅度"
        value = match.group(1).lower()
        label = "一致" if value == "true" else ("部分一致" if value == "partial" else "不一致")
        return f"{name}匹配={label}"

    text = _DIRECTION_PATTERN.sub(replace_direction, text)

    def replace_metric(match: re.Match[str]) -> str:
        names = []
        for raw_name in re.split(r"[/|]", match.group(1)):
            names.append(_METRIC_LABELS.get(raw_name.strip().lower(), raw_name.strip()))
        return f"指标={'/'.join(names)}"

    text = _METRIC_PATTERN.sub(replace_metric, text)
    text = _FORMULA_HYPOTHESIS_RE.sub(
        lambda match: {"i": "假设甲", "j": "假设乙", "k": "假设丙"}[match.group(1)],
        text,
    )
    text = re.sub(r"(?<![A-Za-z0-9_])IG_pair(?![A-Za-z0-9_])", "两两区分度", text)
    text = re.sub(r"(?<![A-Za-z0-9_])IG\(E\)(?![A-Za-z0-9_])", "实验整体区分度", text)
    text = re.sub(r"(?<![A-Za-z0-9_])avg\((?![A-Za-z0-9_])", "平均(", text)
    text = _EFFECT_VALUE_RE.sub(
        lambda match: {
            "positive": "正向增强",
            "negative": "负向削弱",
            "neutral": "中性",
        }[match.group(1).lower()],
        text,
    )

    def replace_narrowed(match: re.Match[str]) -> str:
        return "支持度跨度已收窄" if match.group(1).lower() == "true" else "支持度跨度未收窄"

    text = _NAKED_NARROWED_PATTERN.sub(replace_narrowed, text)

    def replace_naked_status(match: re.Match[str]) -> str:
        return STATUS_LABELS.get(match.group(1), match.group(1))

    text = _NAKED_STATUS_PATTERN.sub(replace_naked_status, text)
    text = re.sub(
        r"(?<![A-Za-z0-9_])narrowed(?![A-Za-z0-9_])",
        "支持度跨度收窄",
        text,
    )
    return text


def _replace_note_prefixes(text: str) -> str:
    for prefix, label in _NOTE_PREFIX_LABELS:
        text = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(prefix)}",
            label,
            text,
        )
    return text


def clean_llm_text(
    text: str | None,
    *,
    tree: HypothesisTreeState | None = None,
    uncertainty_records: list[UncertaintyRecord] | list[Any] | None = None,
) -> str:
    """Replace internal ids and machine tokens inside one narrative string."""
    if not text:
        return text or ""
    cleaned = _replace_span_tokens(text)
    cleaned = _LLM_HYPOTHESIS_PREFIX_RE.sub(r"\1（LLM 起草）", cleaned)
    cleaned = _replace_note_prefixes(cleaned)

    hypothesis_labels: list[tuple[str, str]] = []
    for match in _HYPOTHESIS_RE.finditer(cleaned):
        hypothesis_id = match.group(0)
        label = _hypothesis_display_label(tree, hypothesis_id)
        # 没有树上下文时仍要把机器代号换成可读占位，绝不向 PI 泄漏内部 ID。
        hypothesis_labels.append((hypothesis_id, label or "相关假设"))

    uncertainty_labels: list[tuple[str, str]] = []
    for match in _UNCERTAINTY_RE.finditer(cleaned):
        uncertainty_id = match.group(0)
        question = _uncertainty_question(uncertainty_records, uncertainty_id)
        label = question or DISPLAY_UNCERTAINTY_FALLBACK
        uncertainty_labels.append((uncertainty_id, label))

    all_replacements = sorted(
        hypothesis_labels + uncertainty_labels,
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for old, replacement in all_replacements:
        cleaned = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])",
            replacement.replace("\\", "\\\\"),
            cleaned,
        )
    return cleaned


def deep_clean_text(value: Any, *, tree: HypothesisTreeState | None = None, uncertainty_records=None) -> Any:
    """Recursively clean only string values inside arbitrary JSON-like data."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, str):
        return clean_llm_text(value, tree=tree, uncertainty_records=uncertainty_records)
    if isinstance(value, list):
        return [
            deep_clean_text(item, tree=tree, uncertainty_records=uncertainty_records)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: deep_clean_text(item, tree=tree, uncertainty_records=uncertainty_records)
            for key, item in value.items()
        }
    return value


_STRUCTURED_ID_KEYS = frozenset(
    {
        "id",
        "item_id",
        "trace_id",
        "run_id",
        "failure_id",
        "decision_id",
        "feedback_id",
        "stop_id",
        "experiment_id",
        "source_experiment_id",
        "approved_candidate_id",
        "source_candidate_id",
        "hypothesis_id",
        "leading_hypothesis_id",
        "parent_id",
        "uncertainty_id",
        "tested_hypotheses",
        "target_uncertainties",
        "unresolved_uncertainties",
        "highlighted_hypotheses",
        "related_hypotheses",
        "related_uncertainties",
        "linked_uncertainties",
        "linked_reasoning_trace_ids",
        "active_hypotheses",
        "pruned_hypotheses",
        "pending_hypotheses",
        "children_ids",
    }
)


def deep_clean_preserving_ids(
    value: Any,
    *,
    tree: HypothesisTreeState | None = None,
    uncertainty_records=None,
) -> Any:
    """Clean narrative strings while leaving machine identity keys untouched."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, str):
        return clean_llm_text(value, tree=tree, uncertainty_records=uncertainty_records)
    if isinstance(value, list):
        return [
            deep_clean_preserving_ids(
                item,
                tree=tree,
                uncertainty_records=uncertainty_records,
            )
            for item in value
        ]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in _STRUCTURED_ID_KEYS:
                cleaned[key] = item
            else:
                cleaned[key] = deep_clean_preserving_ids(
                    item,
                    tree=tree,
                    uncertainty_records=uncertainty_records,
                )
        return cleaned
    return value


def _clean_display_note(
    note: str,
    *,
    tree: HypothesisTreeState | None,
    uncertainty_records,
) -> str:
    if not note:
        return note or ""
    cleaned = clean_llm_text(note, tree=tree, uncertainty_records=uncertainty_records)
    cleaned = re.sub(
        r"(?<![A-Za-z0-9_])(H\d+)\s*,\s*(H\d+)(?![A-Za-z0-9_])",
        r"\1、\2",
        cleaned,
    )
    if cleaned == "compare constrained baseline vs baseline+primary treatment":
        cleaned = "比较约束基线与加入焦点变量后的实验设置"
    return cleaned


def _write_model_json(path, model) -> None:
    payload = model.model_dump(mode="json", exclude_none=True)
    path.write_text(json_dumps(payload, indent=2), encoding="utf-8")


def _clean_saved_protocol(
    protocol,
    *,
    tree: HypothesisTreeState | None,
    uncertainty_records,
) -> None:
    """Clean user-facing narrative fields on a stored ExperimentProtocol."""
    from core.unified_schema import ExperimentProtocol

    protocol.scientific_objective = clean_llm_text(
        protocol.scientific_objective, tree=tree, uncertainty_records=uncertainty_records
    )
    protocol.features.notes = [
        _clean_display_note(item, tree=tree, uncertainty_records=uncertainty_records)
        for item in protocol.features.notes
    ]
    protocol.disagreement_context = deep_clean_preserving_ids(
        protocol.disagreement_context,
        tree=tree,
        uncertainty_records=uncertainty_records,
    )
    protocol.hypothesis_source_context = deep_clean_preserving_ids(
        protocol.hypothesis_source_context,
        tree=tree,
        uncertainty_records=uncertainty_records,
    )
    for step in protocol.steps:
        step.action = clean_llm_text(
            step.action, tree=tree, uncertainty_records=uncertainty_records
        )
        step.parameters = deep_clean_text(
            step.parameters,
            tree=tree,
            uncertainty_records=uncertainty_records,
        )
    protocol.notes = [
        _clean_display_note(item, tree=tree, uncertainty_records=uncertainty_records)
        for item in protocol.notes
    ]


def _clean_round_result_files(
    root,
    *,
    tree: HypothesisTreeState | None,
    uncertainty_records,
) -> None:
    """Clean user-facing narratives inside saved per-round report artifacts."""
    from core.unified_schema import EvaluationResult, ExperimentProtocol, ExperimentResult

    results_root = Path(root) / "results"
    if not results_root.exists():
        return

    protocol_paths: list[Path] = []
    for round_dir in sorted(results_root.glob("round_*")):
        protocol_path = round_dir / "protocol.json"
        if protocol_path.exists():
            protocol_paths.append(protocol_path)

        evaluation_path = round_dir / "evaluation_unified.json"
        if evaluation_path.exists():
            evaluation = EvaluationResult.model_validate(json.loads(evaluation_path.read_text(encoding="utf-8")))
            if evaluation.robustness.overall is not None:
                evaluation.robustness.overall.concern = clean_llm_text(
                    evaluation.robustness.overall.concern,
                    tree=tree,
                    uncertainty_records=uncertainty_records,
                )
            evaluation.robustness.recommendation = clean_llm_text(
                evaluation.robustness.recommendation,
                tree=tree,
                uncertainty_records=uncertainty_records,
            )
            scientific = evaluation.scientific
            for assessment in scientific.hypothesis_assessments:
                assessment.reason = clean_llm_text(
                    assessment.reason,
                    tree=tree,
                    uncertainty_records=uncertainty_records,
                )
            for update in scientific.disagreement_updates:
                update.summary = clean_llm_text(
                    update.summary,
                    tree=tree,
                    uncertainty_records=uncertainty_records,
                )
            evidence = scientific.evidence_summary
            for claim in [*evidence.new_evidence_for, *evidence.new_evidence_against]:
                claim.claim = clean_llm_text(
                    claim.claim,
                    tree=tree,
                    uncertainty_records=uncertainty_records,
                )
            evidence.remaining_uncertainties = deep_clean_preserving_ids(
                evidence.remaining_uncertainties,
                tree=tree,
                uncertainty_records=uncertainty_records,
            )

            conclusion = scientific.three_layer_conclusion
            if conclusion is not None:
                conclusion.experiment_layer.design_summary = clean_llm_text(
                    conclusion.experiment_layer.design_summary,
                    tree=tree,
                    uncertainty_records=uncertainty_records,
                )
                for row in conclusion.hypothesis_layer:
                    label = _hypothesis_display_label(tree, row.hypothesis_id)
                    row.statement = label or clean_llm_text(
                        row.statement,
                        tree=tree,
                        uncertainty_records=uncertainty_records,
                    )
                    row.conclusion = clean_llm_text(
                        row.conclusion,
                        tree=tree,
                        uncertainty_records=uncertainty_records,
                    )
                sci = conclusion.scientific_layer
                sci.main_question = clean_llm_text(
                    sci.main_question, tree=tree, uncertainty_records=uncertainty_records
                )
                sci.answer = clean_llm_text(
                    sci.answer, tree=tree, uncertainty_records=uncertainty_records
                )
                sci.path_question = clean_llm_text(
                    sci.path_question, tree=tree, uncertainty_records=uncertainty_records
                )
                sci.path_answer = clean_llm_text(
                    sci.path_answer, tree=tree, uncertainty_records=uncertainty_records
                )
                sci.evidence_text = clean_llm_text(
                    sci.evidence_text, tree=tree, uncertainty_records=uncertainty_records
                )
                if conclusion.data_layer is not None:
                    data = conclusion.data_layer
                    data.rmse_attribution = clean_llm_text(
                        data.rmse_attribution, tree=tree, uncertainty_records=uncertainty_records
                    )
                    data.pearson_attribution = clean_llm_text(
                        data.pearson_attribution, tree=tree, uncertainty_records=uncertainty_records
                    )
                    data.skill_delta_meaning = clean_llm_text(
                        data.skill_delta_meaning, tree=tree, uncertainty_records=uncertainty_records
                    )
                    data.anomalies = [
                        clean_llm_text(item, tree=tree, uncertainty_records=uncertainty_records)
                        for item in data.anomalies
                    ]
                    data.next_focus = clean_llm_text(
                        data.next_focus, tree=tree, uncertainty_records=uncertainty_records
                    )
                    data.comparison_analysis = clean_llm_text(
                        data.comparison_analysis, tree=tree, uncertainty_records=uncertainty_records
                    )
                    for chart in data.chart_analyses:
                        chart.description = clean_llm_text(
                            chart.description, tree=tree, uncertainty_records=uncertainty_records
                        )
                        chart.key_observations = [
                            clean_llm_text(item, tree=tree, uncertainty_records=uncertainty_records)
                            for item in chart.key_observations
                        ]
                        chart.anomaly_or_insight = clean_llm_text(
                            chart.anomaly_or_insight,
                            tree=tree,
                            uncertainty_records=uncertainty_records,
                        )
                if conclusion.tracking_layer is not None:
                    conclusion.tracking_layer.audit_items = [
                        clean_llm_text(item, tree=tree, uncertainty_records=uncertainty_records)
                        for item in conclusion.tracking_layer.audit_items
                    ]
                conclusion.hypothesis_layer_summary = clean_llm_text(
                    conclusion.hypothesis_layer_summary,
                    tree=tree,
                    uncertainty_records=uncertainty_records,
                )
                conclusion.overall_summary = clean_llm_text(
                    conclusion.overall_summary,
                    tree=tree,
                    uncertainty_records=uncertainty_records,
                )
                if conclusion.next_round_suggestion is not None:
                    suggestion = conclusion.next_round_suggestion
                    suggestion.evidence_summary = clean_llm_text(
                        suggestion.evidence_summary,
                        tree=tree,
                        uncertainty_records=uncertainty_records,
                    )
                    suggestion.remaining_uncertainty_analysis = clean_llm_text(
                        suggestion.remaining_uncertainty_analysis,
                        tree=tree,
                        uncertainty_records=uncertainty_records,
                    )
                    suggestion.pi_decision_advice = clean_llm_text(
                        suggestion.pi_decision_advice,
                        tree=tree,
                        uncertainty_records=uncertainty_records,
                    )
                    suggestion.experiment_design_advice = clean_llm_text(
                        suggestion.experiment_design_advice,
                        tree=tree,
                        uncertainty_records=uncertainty_records,
                    )
                    suggestion.hypothesis_space_advice = clean_llm_text(
                        suggestion.hypothesis_space_advice,
                        tree=tree,
                        uncertainty_records=uncertainty_records,
                    )
                    suggestion.notes = [
                        clean_llm_text(item, tree=tree, uncertainty_records=uncertainty_records)
                        for item in suggestion.notes
                    ]
            _write_model_json(evaluation_path, evaluation)

        result_path = round_dir / "result_unified.json"
        if result_path.exists():
            result = ExperimentResult.model_validate(json.loads(result_path.read_text(encoding="utf-8")))
            result.execution.message = clean_llm_text(
                result.execution.message,
                tree=tree,
                uncertainty_records=uncertainty_records,
            )
            for coverage in result.data_coverage:
                coverage.note = clean_llm_text(
                    coverage.note,
                    tree=tree,
                    uncertainty_records=uncertainty_records,
                )
            _write_model_json(result_path, result)

    latest_protocol_path = Path(root) / "config" / "latest_protocol.json"
    if latest_protocol_path.exists():
        protocol_paths.append(latest_protocol_path)
    for protocol_path in protocol_paths:
        protocol = ExperimentProtocol.model_validate(
            json.loads(protocol_path.read_text(encoding="utf-8"))
        )
        _clean_saved_protocol(
            protocol,
            tree=tree,
            uncertainty_records=uncertainty_records,
        )
        _write_model_json(protocol_path, protocol)


def _clean_tree_narratives(
    tree: HypothesisTreeState,
    *,
    tree_for_labels: HypothesisTreeState,
    uncertainty_records,
) -> None:
    """Clean user-facing narrative text on a hypothesis tree in place."""
    for node in tree.nodes:
        node.statement = clean_llm_text(
            node.statement, tree=tree_for_labels, uncertainty_records=uncertainty_records
        )
        node.display_statement = clean_llm_text(
            node.display_statement,
            tree=tree_for_labels,
            uncertainty_records=uncertainty_records,
        )
        node.activation_condition = clean_llm_text(
            node.activation_condition,
            tree=tree_for_labels,
            uncertainty_records=uncertainty_records,
        )
        if node.generation_rationale is not None:
            node.generation_rationale.summary = clean_llm_text(
                node.generation_rationale.summary,
                tree=tree_for_labels,
                uncertainty_records=uncertainty_records,
            )
            for signal in node.generation_rationale.source_signals:
                signal.excerpt = clean_llm_text(
                    signal.excerpt,
                    tree=tree_for_labels,
                    uncertainty_records=uncertainty_records,
                )
        for evidence in [*node.evidence_items, *node.evidence_against]:
            evidence.description = clean_llm_text(
                evidence.description,
                tree=tree_for_labels,
                uncertainty_records=uncertainty_records,
            )
        for critique in node.critiques:
            critique.content = clean_llm_text(
                critique.content,
                tree=tree_for_labels,
                uncertainty_records=uncertainty_records,
            )
        for record in node.questioning_records:
            record.rationale = clean_llm_text(
                record.rationale,
                tree=tree_for_labels,
                uncertainty_records=uncertainty_records,
            )
            record.falsification_basis = clean_llm_text(
                record.falsification_basis,
                tree=tree_for_labels,
                uncertainty_records=uncertainty_records,
            )
        node.falsification_conditions = [
            clean_llm_text(
                item,
                tree=tree_for_labels,
                uncertainty_records=uncertainty_records,
            )
            for item in node.falsification_conditions
        ]
        node.alternative_explanations = [
            clean_llm_text(
                item,
                tree=tree_for_labels,
                uncertainty_records=uncertainty_records,
            )
            for item in node.alternative_explanations
        ]
        node.prune_reason = clean_llm_text(
            node.prune_reason,
            tree=tree_for_labels,
            uncertainty_records=uncertainty_records,
        )


def clean_repository_text(root) -> None:
    """Clean narrative fields of live-state repositories in place.

    ``root`` is the repository root (usually ``runtime/live_session/current``).
    Structured identifiers in machine-only fields are preserved; only
    user-facing narrative strings are rewritten.
    """
    from core.state_repository import UnifiedStateRepository

    repository = UnifiedStateRepository(root)
    tree = repository.load_hypothesis_tree()
    uncertainties = repository.load_uncertainties()
    records = uncertainties.records

    _clean_tree_narratives(tree, tree_for_labels=tree, uncertainty_records=records)
    repository.save_hypothesis_tree(tree)

    pre_tree = repository.load_pre_questioning_tree()
    if pre_tree is not None:
        _clean_tree_narratives(pre_tree, tree_for_labels=tree, uncertainty_records=records)
        repository.save_pre_questioning_tree(pre_tree)

    for record in uncertainties.records:
        record.description = clean_llm_text(
            record.description, tree=tree, uncertainty_records=records
        )
        record.notes = clean_llm_text(record.notes, tree=tree, uncertainty_records=records)
        record.resolution = clean_llm_text(
            record.resolution, tree=tree, uncertainty_records=records
        )
        for history in record.history:
            history.description = clean_llm_text(
                history.description, tree=tree, uncertainty_records=records
            )
        record.alternative_resolutions = [
            clean_llm_text(item, tree=tree, uncertainty_records=records)
            for item in record.alternative_resolutions
        ]
        record.disagreement = deep_clean_preserving_ids(
            record.disagreement,
            tree=tree,
            uncertainty_records=records,
        )
        for history in record.history:
            history.event = clean_llm_text(
                history.event, tree=tree, uncertainty_records=records
            )
    repository.save_uncertainties(uncertainties)

    history = repository.load_round_history()
    for entry in history.entries:
        entry.decision_summary = clean_llm_text(
            entry.decision_summary, tree=tree, uncertainty_records=records
        )
        entry.human_feedback = clean_llm_text(
            entry.human_feedback, tree=tree, uncertainty_records=records
        )
        entry.failure_reason = clean_llm_text(
            entry.failure_reason, tree=tree, uncertainty_records=records
        )
        entry.scientific_findings = [
            clean_llm_text(item, tree=tree, uncertainty_records=records)
            for item in entry.scientific_findings
        ]
        entry.iteration_input_sources = [
            clean_llm_text(item, tree=tree, uncertainty_records=records)
            for item in entry.iteration_input_sources
        ]
        for checklist in entry.closure_checklist:
            checklist.detail = clean_llm_text(
                checklist.detail, tree=tree, uncertainty_records=records
            )
        entry.iterative_validations = deep_clean_preserving_ids(
            entry.iterative_validations,
            tree=tree,
            uncertainty_records=records,
        )
        if entry.three_layer_conclusion is not None:
            conclusion = entry.three_layer_conclusion
            conclusion.experiment_layer.design_summary = clean_llm_text(
                conclusion.experiment_layer.design_summary,
                tree=tree,
                uncertainty_records=records,
            )
            for row in conclusion.hypothesis_layer:
                row.statement = clean_llm_text(
                    row.statement, tree=tree, uncertainty_records=records
                )
                row.conclusion = clean_llm_text(
                    row.conclusion, tree=tree, uncertainty_records=records
                )
            scientific = conclusion.scientific_layer
            scientific.main_question = clean_llm_text(
                scientific.main_question, tree=tree, uncertainty_records=records
            )
            scientific.answer = clean_llm_text(
                scientific.answer, tree=tree, uncertainty_records=records
            )
            scientific.path_question = clean_llm_text(
                scientific.path_question, tree=tree, uncertainty_records=records
            )
            scientific.path_answer = clean_llm_text(
                scientific.path_answer, tree=tree, uncertainty_records=records
            )
            scientific.evidence_text = clean_llm_text(
                scientific.evidence_text, tree=tree, uncertainty_records=records
            )
            if conclusion.data_layer is not None:
                data = conclusion.data_layer
                data.rmse_attribution = clean_llm_text(
                    data.rmse_attribution, tree=tree, uncertainty_records=records
                )
                data.pearson_attribution = clean_llm_text(
                    data.pearson_attribution, tree=tree, uncertainty_records=records
                )
                data.skill_delta_meaning = clean_llm_text(
                    data.skill_delta_meaning, tree=tree, uncertainty_records=records
                )
                data.anomalies = [
                    clean_llm_text(item, tree=tree, uncertainty_records=records)
                    for item in data.anomalies
                ]
                data.next_focus = clean_llm_text(
                    data.next_focus, tree=tree, uncertainty_records=records
                )
                data.comparison_analysis = clean_llm_text(
                    data.comparison_analysis, tree=tree, uncertainty_records=records
                )
                for chart in data.chart_analyses:
                    chart.description = clean_llm_text(
                        chart.description, tree=tree, uncertainty_records=records
                    )
                    chart.key_observations = [
                        clean_llm_text(item, tree=tree, uncertainty_records=records)
                        for item in chart.key_observations
                    ]
                    chart.anomaly_or_insight = clean_llm_text(
                        chart.anomaly_or_insight, tree=tree, uncertainty_records=records
                    )
            conclusion.hypothesis_layer_summary = clean_llm_text(
                conclusion.hypothesis_layer_summary, tree=tree, uncertainty_records=records
            )
            conclusion.overall_summary = clean_llm_text(
                conclusion.overall_summary, tree=tree, uncertainty_records=records
            )
            if conclusion.next_round_suggestion is not None:
                suggestion = conclusion.next_round_suggestion
                suggestion.evidence_summary = clean_llm_text(
                    suggestion.evidence_summary, tree=tree, uncertainty_records=records
                )
                suggestion.remaining_uncertainty_analysis = clean_llm_text(
                    suggestion.remaining_uncertainty_analysis,
                    tree=tree,
                    uncertainty_records=records,
                )
                suggestion.pi_decision_advice = clean_llm_text(
                    suggestion.pi_decision_advice, tree=tree, uncertainty_records=records
                )
                suggestion.experiment_design_advice = clean_llm_text(
                    suggestion.experiment_design_advice,
                    tree=tree,
                    uncertainty_records=records,
                )
                suggestion.hypothesis_space_advice = clean_llm_text(
                    suggestion.hypothesis_space_advice,
                    tree=tree,
                    uncertainty_records=records,
                )
                suggestion.notes = [
                    clean_llm_text(item, tree=tree, uncertainty_records=records)
                    for item in suggestion.notes
                ]
            if conclusion.tracking_layer is not None:
                conclusion.tracking_layer.audit_items = [
                    clean_llm_text(item, tree=tree, uncertainty_records=records)
                    for item in conclusion.tracking_layer.audit_items
                ]
    repository.save_round_history(history)

    try:
        memory = repository.load_experiment_memory()
        for memory_entry in memory.entries:
            memory_entry.key_findings = [
                clean_llm_text(item, tree=tree, uncertainty_records=records)
                for item in memory_entry.key_findings
            ]
            for trace in memory_entry.reasoning_traces:
                trace.summary = clean_llm_text(
                    trace.summary, tree=tree, uncertainty_records=records
                )
            for coverage in memory_entry.data_coverage:
                coverage.note = clean_llm_text(
                    coverage.note, tree=tree, uncertainty_records=records
                )
        repository.save_experiment_memory(memory)
    except Exception:
        pass

    try:
        decision_log = repository.load_decision_log()
        for decision in decision_log.decisions:
            decision.summary = clean_llm_text(
                decision.summary, tree=tree, uncertainty_records=records
            )
            decision.details = deep_clean_preserving_ids(
                decision.details,
                tree=tree,
                uncertainty_records=records,
            )
        for feedback in decision_log.human_feedback:
            feedback.content = clean_llm_text(
                feedback.content, tree=tree, uncertainty_records=records
            )
            feedback.context_summary = clean_llm_text(
                feedback.context_summary, tree=tree, uncertainty_records=records
            )
        for stop_entry in decision_log.stop_history:
            stop_entry.reason = clean_llm_text(
                stop_entry.reason, tree=tree, uncertainty_records=records
            )
        repository.save_decision_log(decision_log)
    except Exception:
        pass

    try:
        failure_history = repository.load_failure_history()
        for failure in failure_history.records:
            failure.message = clean_llm_text(
                failure.message, tree=tree, uncertainty_records=records
            )
        repository.save_failure_history(failure_history)
    except Exception:
        pass

    try:
        planner_input = repository.load_planner_input()
    except Exception:
        planner_input = None
    if planner_input is not None:
        planner_input.human_feedback = clean_llm_text(
            planner_input.human_feedback, tree=tree, uncertainty_records=records
        )
        planner_input.evaluation_summary.key_findings = [
            clean_llm_text(item, tree=tree, uncertainty_records=records)
            for item in planner_input.evaluation_summary.key_findings
        ]
        planner_input.evaluation_summary.robustness_recommendation = clean_llm_text(
            planner_input.evaluation_summary.robustness_recommendation,
            tree=tree,
            uncertainty_records=records,
        )
        for item in planner_input.unresolved_uncertainties:
            item.question = clean_llm_text(
                item.question, tree=tree, uncertainty_records=records
            )
        for item in planner_input.recent_hypothesis_assessments:
            item.reason = clean_llm_text(item.reason, tree=tree, uncertainty_records=records)
        for item in planner_input.recent_disagreement_updates:
            item.summary = clean_llm_text(item.summary, tree=tree, uncertainty_records=records)
        for item in planner_input.recent_reasoning_traces:
            item.summary = clean_llm_text(item.summary, tree=tree, uncertainty_records=records)
        for item in planner_input.recent_human_feedback:
            item.content = clean_llm_text(item.content, tree=tree, uncertainty_records=records)
            item.context_summary = clean_llm_text(
                item.context_summary, tree=tree, uncertainty_records=records
            )
        planner_input.planner_guidance = [
            clean_llm_text(item, tree=tree, uncertainty_records=records)
            for item in planner_input.planner_guidance
        ]
        repository.save_planner_input(planner_input)

    try:
        candidates = repository.load_candidate_experiments()
    except Exception:
        candidates = None
    if candidates is not None:
        for candidate in candidates.candidates:
            candidate.scientific_question = clean_llm_text(
                candidate.scientific_question, tree=tree, uncertainty_records=records
            )
            candidate.purpose = clean_llm_text(
                candidate.purpose, tree=tree, uncertainty_records=records
            )
            candidate.distinguishing_insight = clean_llm_text(
                candidate.distinguishing_insight, tree=tree, uncertainty_records=records
            )
            candidate.value_analysis = clean_llm_text(
                candidate.value_analysis, tree=tree, uncertainty_records=records
            )
            candidate.novelty = clean_llm_text(
                candidate.novelty, tree=tree, uncertainty_records=records
            )
            candidate.design.notes = [
                _clean_display_note(item, tree=tree, uncertainty_records=records)
                for item in candidate.design.notes
            ]
            for estimate in [
                candidate.estimated_information_gain,
                candidate.estimated_performance_gain,
                candidate.estimated_risk,
                candidate.estimated_cost,
            ]:
                if estimate is not None and estimate.rationale:
                    estimate.rationale = clean_llm_text(
                        estimate.rationale,
                        tree=tree,
                        uncertainty_records=records,
                    )
            candidate.hypothesis_source_context = deep_clean_text(
                candidate.hypothesis_source_context, tree=tree, uncertainty_records=records
            ) or {}
            candidate.disagreement_context = deep_clean_text(
                candidate.disagreement_context, tree=tree, uncertainty_records=records
            ) or {}
        candidates.note = _clean_display_note(
            candidates.note, tree=tree, uncertainty_records=records
        )
        for rejected in candidates.rejected_candidates:
            rejected.reason = _clean_display_note(
                rejected.reason, tree=tree, uncertainty_records=records
            )
            rejected.similar_to = _clean_display_note(
                rejected.similar_to, tree=tree, uncertainty_records=records
            )
        repository.save_candidate_experiments(candidates)

    _clean_round_result_files(
        root,
        tree=tree,
        uncertainty_records=records,
    )
