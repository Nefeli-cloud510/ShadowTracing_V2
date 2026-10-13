from __future__ import annotations

from dataclasses import dataclass

from core.hypothesis_state_machine import EvidenceSource, OBSERVING_LOW, resolve_status


@dataclass(frozen=True)
class SupportUpdateOutcome:
    support_before: float
    support_after: float
    direction_matched: bool | str
    magnitude_matched: bool | str
    status: str
    support_delta: float
    updated: bool = True
    evidence_classification: str | None = None


# 证据分类。数值结论一律以 RMSE 增量为主指标：
# gain_rmse = RMSE_baseline - RMSE_treatment，正值表示实验组更优。
EVIDENCE_NOT_EXECUTED = "not_executed"
EVIDENCE_INVALID = "invalid"
EVIDENCE_NO_STABLE_GAIN = "no_stable_gain"
EVIDENCE_STABLE_IMPROVEMENT = "stable_improvement"
EVIDENCE_STABLE_DEGRADATION = "stable_degradation"
EVIDENCE_INCONCLUSIVE = "inconclusive"

EVIDENCE_CLASSIFICATION_LABELS = {
    EVIDENCE_NOT_EXECUTED: "实验未执行，无可用证据",
    EVIDENCE_INVALID: "实验执行无效，仅作审计记录",
    EVIDENCE_NO_STABLE_GAIN: "未检出稳定增益，不更新支持度",
    EVIDENCE_STABLE_IMPROVEMENT: "检测到稳定 RMSE 增益",
    EVIDENCE_STABLE_DEGRADATION: "检测到稳定 RMSE 退化",
    EVIDENCE_INCONCLUSIVE: "RMSE 增益 CI 跨零，证据不足以确定方向",
}


@dataclass(frozen=True)
class EvidenceClassification:
    classification: str
    labels: list[str]
    updated: bool = False
    direction_matched: bool | str = "partial"
    magnitude_matched: bool | str = "partial"


def observed_delta_from_comparison(
    *,
    skill_delta: float | None,
    pearson_delta: float | None,
    rmse_delta: float | None,
    primary_metric: str = "rmse",
) -> float:
    # 所有 delta 统一为“正值 = 实验组增益”。
    metric_deltas = {
        "rmse": rmse_delta,
        "pearson_r": pearson_delta,
        "skill": skill_delta,
    }
    metric = (primary_metric or "rmse").lower()
    if metric_deltas.get(metric) is not None:
        return metric_deltas[metric]
    for value in metric_deltas.values():
        if value is not None:
            return value
    return 0.0


def classify_rmse_gain_evidence(
    *,
    rmse_delta: float | None,
    rmse_ci: list[float] | None = None,
    execution_status: str = "completed",
    validity_status: str = "valid",
    allowed_conclusions: list[str] | None = None,
    warnings: list[str] | None = None,
) -> EvidenceClassification:
    """把一次对照实验的主指标证据归入可审计分类。

    CI 跨零时记录为 inconclusive，不自动当作反证；只有稳定改善/稳定退化
    才允许保守更新支持度。``allowed_conclusions`` 中的特定结论只用于附加
    标签（如 ElasticNet 筛零、无预测增量、磁场信息冗余），不改变分类本身。
    """

    if str(validity_status) == "invalid":
        return EvidenceClassification(
            classification=EVIDENCE_INVALID,
            labels=_conclusion_labels(allowed_conclusions)
            + (["实验校验失败"] if warnings else []),
        )
    if str(execution_status) != "completed":
        return EvidenceClassification(
            classification=EVIDENCE_NOT_EXECUTED,
            labels=["实验未完成，暂缓支持度判定"],
        )
    if rmse_delta is None or not _finite(rmse_delta):
        return EvidenceClassification(
            classification=EVIDENCE_NO_STABLE_GAIN,
            labels=["缺少成对 RMSE 增量，无法进行支持度判定"],
        )

    labels = _conclusion_labels(allowed_conclusions)
    ci = _finite_ci(rmse_ci)
    if ci is not None:
        lower, upper = ci
        if lower > 0.0:
            return EvidenceClassification(
                classification=EVIDENCE_STABLE_IMPROVEMENT,
                labels=["bootstrap RMSE 增益 95% CI 整体为正"]
                + labels,
                updated=True,
                direction_matched=True,
            )
        if upper < 0.0:
            return EvidenceClassification(
                classification=EVIDENCE_STABLE_DEGRADATION,
                labels=["bootstrap RMSE 增益 95% CI 整体为负"]
                + labels,
                updated=True,
                direction_matched=False,
            )
        return EvidenceClassification(
            classification=EVIDENCE_INCONCLUSIVE,
            labels=["bootstrap RMSE 增益 95% CI 跨越零，证据不足以确定方向"]
            + labels,
        )

    labels.append("缺少 bootstrap CI，点估计无法确认稳定性")
    if abs(rmse_delta) < 1e-6:
        labels.append("观测 RMSE 增量接近零")
    return EvidenceClassification(
        classification=EVIDENCE_NO_STABLE_GAIN,
        labels=labels,
    )


def compute_support_update_from_evidence(
    *,
    current_support: float,
    evidence_classification: str,
    evidence_labels: list[str] | None = None,
    expected_effect: str | None = None,
    expected_range: list[float] | None = None,
    observed_delta: float | None = None,
) -> SupportUpdateOutcome:
    """按证据分类做保守、有界的支持度更新。

    只有稳定改善/稳定退化会产生小幅度移动；CI 跨零、未检出稳定增益、
    未执行或无效证据一律保持支持度不变。该结果不声称是后验概率。
    """

    effect = str(expected_effect or "").lower()
    no_update_classifications = {
        EVIDENCE_NOT_EXECUTED,
        EVIDENCE_INVALID,
        EVIDENCE_NO_STABLE_GAIN,
        EVIDENCE_INCONCLUSIVE,
    }
    magnitude_matched = (
        match_magnitude(observed_delta or 0.0, expected_range)
        if expected_range
        else "partial"
    )
    if evidence_classification in no_update_classifications:
        return SupportUpdateOutcome(
            support_before=current_support,
            support_after=current_support,
            direction_matched="partial",
            magnitude_matched="partial",
            status="partially_supported",
            support_delta=0.0,
            updated=False,
            evidence_classification=evidence_classification,
        )

    if evidence_classification == EVIDENCE_STABLE_IMPROVEMENT:
        if effect == "positive":
            direction_matched, delta = True, 0.05
        elif effect == "negative":
            direction_matched, delta = False, -0.05
        elif effect == "near_zero":
            direction_matched, delta = False, -0.03
        else:
            direction_matched, delta = True, 0.03
    elif evidence_classification == EVIDENCE_STABLE_DEGRADATION:
        if effect == "positive":
            direction_matched, delta = False, -0.05
        elif effect == "negative":
            direction_matched, delta = True, 0.05
        elif effect == "near_zero":
            direction_matched, delta = False, -0.03
        else:
            direction_matched, delta = False, -0.03
    else:
        direction_matched, delta = "partial", 0.0

    support_after = round(min(max(current_support + delta, 0.05), 0.95), 4)
    return SupportUpdateOutcome(
        support_before=current_support,
        support_after=support_after,
        direction_matched=direction_matched,
        magnitude_matched=magnitude_matched,
        status=_assessment_status(direction_matched, magnitude_matched),
        support_delta=round(support_after - current_support, 4),
        updated=True,
        evidence_classification=evidence_classification,
    )


def _finite(value: float) -> bool:
    try:
        return float(value) == float(value) and abs(float(value)) != float("inf")
    except (TypeError, ValueError):
        return False


def _finite_ci(rmse_ci: list[float] | None) -> list[float] | None:
    if not rmse_ci or len(rmse_ci) < 2:
        return None
    try:
        lower, upper = float(rmse_ci[0]), float(rmse_ci[1])
    except (TypeError, ValueError):
        return None
    if not (_finite(lower) and _finite(upper)):
        return None
    return [lower, upper]


def _conclusion_labels(allowed_conclusions: list[str] | None) -> list[str]:
    if not allowed_conclusions:
        return []
    text = " ".join(str(item).lower() for item in allowed_conclusions)
    labels: list[str] = []
    if "elasticnet" in text and any(token in text for token in ("zero", "筛零", "shrunk")):
        labels.append("ElasticNet 焦点系数被筛为零")
    if any(token in text for token in ("无预测增量", "no_incremental_gain", "no_incremental_value")):
        labels.append("未检出焦点变量预测增量")
    if any(token in text for token in ("磁场信息冗余", "redundant", "redundancy")):
        labels.append("磁场信息冗余，增量信息已被对照变量覆盖")
    return labels


def match_direction(
    observed_delta: float,
    expected_effect: str,
) -> bool | str:
    effect = expected_effect.lower()
    if effect == "positive":
        if observed_delta > 0.005:
            return True
        if observed_delta >= 0.0:
            return "partial"
        return False
    if effect == "negative":
        if observed_delta < -0.005:
            return True
        if observed_delta <= 0.0:
            return "partial"
        return False
    if effect == "near_zero":
        absolute = abs(observed_delta)
        if absolute <= 0.01:
            return True
        if absolute <= 0.02:
            return "partial"
        return False
    return "partial"


def match_magnitude(
    observed_delta: float,
    expected_range: list[float] | None,
) -> bool | str:
    if not expected_range:
        return "partial"
    lower, upper = expected_range
    if lower <= observed_delta <= upper:
        return True
    if (observed_delta < lower and observed_delta >= min(lower, 0.0)) or (
        observed_delta > upper and observed_delta <= max(upper, 0.0)
    ):
        return "partial"
    return False


def compute_support_update_from_prediction(
    *,
    current_support: float,
    expected_effect: str,
    expected_range: list[float] | None,
    observed_delta: float,
) -> SupportUpdateOutcome:
    direction_matched = match_direction(observed_delta, expected_effect)
    magnitude_matched = match_magnitude(observed_delta, expected_range)
    direction_factor = {True: 1.15, "partial": 1.0, False: 0.8}[direction_matched]
    magnitude_factor = {True: 1.05, "partial": 0.95, False: 0.85}[magnitude_matched]
    precision_factor = _precision_factor(expected_range, observed_delta)
    updated = current_support * direction_factor * magnitude_factor * precision_factor
    support_after = round(min(max(updated, 0.05), 0.95), 4)
    return SupportUpdateOutcome(
        support_before=current_support,
        support_after=support_after,
        direction_matched=direction_matched,
        magnitude_matched=magnitude_matched,
        status=_assessment_status(direction_matched, magnitude_matched),
        support_delta=round(support_after - current_support, 4),
    )


def compute_support_update_from_reasoning(
    *,
    current_support: float,
    impact_direction: str,
    impact_strength: float,
    confidence: float,
    falsification_basis: str | None = None,
    previous_status: str | None = None,
    support_history: list | None = None,
    current_round: int | None = None,
    parent_support: float | None = None,
    draft_to_pending: bool = False,
    evidence_source: EvidenceSource = "advisory",
) -> SupportUpdateOutcome:
    advisory = evidence_source == "advisory"
    effective_direction = impact_direction
    effective_strength = impact_strength
    if (
        advisory
        and impact_direction == "weakens"
        and not (falsification_basis or "").strip()
    ):
        # 只有“表达担忧”没有可证否依据时，最多算澄清，不允许产生负分。
        effective_direction = "clarifies"
        effective_strength = 0.0
    direction_scale = {
        "supports": 1.0,
        "weakens": -0.4,
        "clarifies": 0.35,
    }.get(effective_direction, 0.0)
    support_delta = round(direction_scale * effective_strength * confidence * 0.18, 4)
    support_after = current_support + support_delta
    # 顾问意见第一次把支持度打到 20% 以下时，先垫到观察下限，避免越级剪枝。
    if advisory and support_after < OBSERVING_LOW:
        support_after = OBSERVING_LOW
    support_after = round(min(max(support_after, 0.0), 1.0), 4)
    direction_matched = {
        "supports": True,
        "weakens": False,
        "clarifies": "partial",
    }.get(effective_direction, "partial")
    magnitude_matched = True if effective_strength >= 0.6 else "partial"
    status = resolve_status(
        previous_status=previous_status or "active",
        support_after=support_after,
        support_history=support_history or [],
        current_round=current_round,
        parent_support=parent_support,
        draft_to_pending=draft_to_pending,
        evidence_source=evidence_source,
    )
    return SupportUpdateOutcome(
        support_before=current_support,
        support_after=support_after,
        direction_matched=direction_matched,
        magnitude_matched=magnitude_matched,
        status=status,
        support_delta=support_delta,
    )


def priority_after_action(priority: str, action: str) -> str:
    ordered = ["low", "medium", "high"]
    if priority not in ordered:
        return priority
    index = ordered.index(priority)
    if action == "increase":
        return ordered[min(index + 1, len(ordered) - 1)]
    if action == "decrease":
        return ordered[max(index - 1, 0)]
    return priority


def priority_to_score(priority: str) -> float:
    return {"low": 0.3, "medium": 0.6, "high": 0.9}.get(priority, 0.6)


def signal_from_metric_deltas(rmse_delta: float | None, pearson_delta: float | None) -> str:
    rmse_better = rmse_delta is not None and rmse_delta > 0
    rmse_worse = rmse_delta is not None and rmse_delta < 0
    pearson_better = pearson_delta is not None and pearson_delta > 0
    pearson_worse = pearson_delta is not None and pearson_delta < 0
    if rmse_better and pearson_better:
        return "supports"
    if rmse_worse and pearson_worse:
        return "weakens"
    return "mixed"


def signal_strength(rmse_delta: float | None, pearson_delta: float | None) -> str:
    rmse_gap = abs(rmse_delta or 0.0)
    pearson_gap = abs(pearson_delta or 0.0)
    if rmse_gap >= 1.0 or pearson_gap >= 0.1:
        return "high"
    if rmse_gap >= 0.1 or pearson_gap >= 0.02:
        return "medium"
    return "low"


def _precision_factor(expected_range: list[float] | None, observed_delta: float) -> float:
    if not expected_range:
        return 1.0
    lower, upper = expected_range
    hit = lower <= observed_delta <= upper
    width = abs(upper - lower)
    if hit and width < 0.03:
        return 1.05
    if hit and width > 0.10:
        return 0.98
    return 0.95


def _assessment_status(direction_matched: bool | str, magnitude_matched: bool | str) -> str:
    if direction_matched is True and magnitude_matched is True:
        return "supported"
    if direction_matched is False:
        return "weakened"
    return "partially_supported"
