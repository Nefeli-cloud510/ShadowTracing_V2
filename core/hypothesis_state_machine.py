from __future__ import annotations

from typing import Any, Literal

from core.unified_schema import SupportHistoryEntry


EvidenceSource = Literal["advisory", "experimental", "human"]

ACTIVATION_THRESHOLD = 0.40
OBSERVING_LOW = 0.20
OBSERVING_HIGH = 0.40
CONVERGENCE_THRESHOLD = 0.70
WEAK_ROUNDS_TO_PRUNE = 3
STRONG_ROUNDS_TO_CONVERGE = 2

EXPERIMENTAL_EVENT_PREFIXES = ("evaluation_writeback", "experiment", "experimental_result")
DECISIVE_EVIDENCE_CLASSIFICATIONS = frozenset(
    {"stable_improvement", "stable_degradation"}
)

PRUNED = "pruned"
ACTIVE = "active"
OBSERVING = "observing"
CONVERGED = "converged"
PENDING = "pending"
DRAFT = "draft"

ACTIVE_LIKE = frozenset({ACTIVE, CONVERGED})


def _stringified(value: object | None) -> str | None:
    return None if value is None else str(value)


def _is_experimental_event(event: str | None) -> bool:
    normalized = _stringified(event)
    if not normalized:
        return False
    normalized = normalized.lower()
    if normalized in EXPERIMENTAL_EVENT_PREFIXES:
        return True
    return normalized.startswith(
        tuple(f"{prefix}:" for prefix in EXPERIMENTAL_EVENT_PREFIXES)
    )


def _direction_matched(value: bool | str | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _is_decisive_convergence_evidence(
    *,
    evidence_classification: str | None,
    direction_matched: bool | str | None,
) -> bool:
    classification = _stringified(evidence_classification)
    if not classification or classification.lower() not in DECISIVE_EVIDENCE_CLASSIFICATIONS:
        return False
    if direction_matched is None:
        # 旧记录缺少方向标记时，只把稳定改善算作方向一致；稳定退化必须显式
        # 说明假设预期的是负效应，否则不能计入收敛证据。
        return classification.lower() == "stable_improvement"
    return _direction_matched(direction_matched)


def recent_scores(
    support_history: list[SupportHistoryEntry] | list[dict[str, Any]] | None,
    *,
    current_round: int | None = None,
    current_score: float | None = None,
    limit: int = WEAK_ROUNDS_TO_PRUNE,
) -> list[float]:
    """Return the most recent per-round support scores, current round first."""
    if current_round is not None and current_score is not None:
        latest = {int(current_round): current_score}
    else:
        latest = {}
    for entry in support_history or []:
        if isinstance(entry, SupportHistoryEntry):
            round_id = entry.round
            score = entry.score
        else:
            round_id = entry.get("round")
            score = entry.get("score")
        if round_id is None or score is None:
            continue
        latest[int(round_id)] = float(score)
    ordered = [latest[round_id] for round_id in sorted(latest)]
    return ordered[-limit:]


def recent_experimental_convergence_scores(
    support_history: list[SupportHistoryEntry] | list[dict[str, Any]] | None,
    *,
    current_round: int | None = None,
    current_score: float | None = None,
    current_classification: str | None = None,
    current_direction: bool | str | None = None,
    current_evidence_source: EvidenceSource = "advisory",
    limit: int = STRONG_ROUNDS_TO_CONVERGE,
) -> list[float]:
    """Return the most recent per-round scores backed by decisive experiments."""
    latest: dict[int, float] = {}
    if (
        current_round is not None
        and current_score is not None
        and current_evidence_source != "advisory"
        and _is_decisive_convergence_evidence(
            evidence_classification=current_classification,
            direction_matched=current_direction,
        )
    ):
        latest[int(current_round)] = float(current_score)
    for entry in support_history or []:
        if isinstance(entry, SupportHistoryEntry):
            round_id = entry.round
            event = entry.event
            score = entry.score
            classification = entry.evidence_classification
            direction = entry.direction_matched
            source = entry.evidence_source
        else:
            round_id = entry.get("round")
            event = entry.get("event")
            score = entry.get("score")
            classification = entry.get("evidence_classification")
            direction = entry.get("direction_matched")
            source = entry.get("evidence_source")
        if round_id is None or score is None:
            continue
        if not _is_experimental_event(event):
            continue
        if source is not None and str(source).lower() == "advisory":
            continue
        if not _is_decisive_convergence_evidence(
            evidence_classification=classification,
            direction_matched=direction,
        ):
            continue
        latest[int(round_id)] = float(score)
    ordered = [latest[round_id] for round_id in sorted(latest)]
    return ordered[-limit:]


def previous_round_score(
    support_history: list[SupportHistoryEntry] | list[dict[str, Any]] | None,
    *,
    current_round: int | None,
    current_score: float | None,
) -> float | None:
    """Return the most recent round support strictly before the current one."""
    scores = recent_scores(
        support_history,
        current_round=current_round,
        current_score=current_score,
        limit=2,
    )
    return scores[-2] if len(scores) >= 2 else None


def resolve_status(
    *,
    previous_status: str,
    support_after: float,
    support_history: list[SupportHistoryEntry] | list[dict[str, Any]] | None = None,
    current_round: int | None = None,
    parent_support: float | None = None,
    draft_to_pending: bool = False,
    evidence_source: EvidenceSource = "experimental",
    evidence_classification: str | None = None,
    direction_matched: bool | str | None = None,
) -> str:
    """Apply the canonical hypothesis state machine.

    LLM 质询属于 ``advisory`` 意见，只能温和调整状态；真实实验回写与人工覆盖
    属于强证据，才允许直接驱动剪枝或收敛。
    """
    if previous_status == PRUNED:
        return PRUNED

    advisory_first_save = evidence_source == "advisory" and previous_status != PRUNED
    # 支持度 < 20% 直接剪枝；首次顾问质询弱化最多降到待观察，不能越级剪枝。
    if support_after < OBSERVING_LOW and not advisory_first_save:
        return PRUNED

    if previous_status == DRAFT:
        return PENDING if draft_to_pending else DRAFT

    if previous_status == PENDING:
        if parent_support is None or parent_support < ACTIVATION_THRESHOLD:
            return PENDING
        # 父假设达标只是一个准入门槛，最终状态仍由子假设自身支持度决定。
        return ACTIVE if support_after >= ACTIVATION_THRESHOLD else OBSERVING

    if previous_status == ACTIVE:
        if evidence_source != "advisory":
            experimental_scores = recent_experimental_convergence_scores(
                support_history,
                current_round=current_round,
                current_score=support_after,
                current_classification=evidence_classification,
                current_direction=direction_matched,
                current_evidence_source=evidence_source,
                limit=STRONG_ROUNDS_TO_CONVERGE,
            )
            if (
                len(experimental_scores) >= STRONG_ROUNDS_TO_CONVERGE
                and all(
                    score >= CONVERGENCE_THRESHOLD
                    for score in experimental_scores[-STRONG_ROUNDS_TO_CONVERGE:]
                )
            ):
                return CONVERGED
        return ACTIVE if support_after >= ACTIVATION_THRESHOLD else OBSERVING

    if previous_status == OBSERVING:
        previous = previous_round_score(
            support_history,
            current_round=current_round,
            current_score=support_after,
        )
        if support_after >= ACTIVATION_THRESHOLD and (
            previous is None or support_after > previous
        ):
            return ACTIVE
        scores = recent_scores(
            support_history,
            current_round=current_round,
            current_score=support_after,
            limit=WEAK_ROUNDS_TO_PRUNE,
        )
        if evidence_source != "advisory":
            if len(scores) >= WEAK_ROUNDS_TO_PRUNE and all(
                score < OBSERVING_HIGH for score in scores[-WEAK_ROUNDS_TO_PRUNE:]
            ):
                return PRUNED
        return OBSERVING

    if previous_status == CONVERGED:
        if support_after < ACTIVATION_THRESHOLD:
            return OBSERVING
        return ACTIVE if support_after < CONVERGENCE_THRESHOLD else CONVERGED

    # 兼容旧评估口径遗留的 supported / weakened / partially_supported。
    return ACTIVE if support_after >= ACTIVATION_THRESHOLD else OBSERVING
