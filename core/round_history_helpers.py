"""Shared helpers for round-to-round evidence linkage.

These helpers reconcile planned candidate prose with the actually compiled
experimental configuration, so the state updater and the standalone repair
script write the same cross-round audit fields.
"""

from __future__ import annotations

from typing import Any

from core.unified_schema import ExperimentProtocol, RoundHistoryEntry


def _models_from_protocol(protocol: ExperimentProtocol | None) -> list[str]:
    if protocol is None:
        return []
    families = list(dict.fromkeys(protocol.features.model_families or []))
    if not families:
        for run in protocol.runs_spec or []:
            family = str(run.model_family or "").strip().lower()
            if family:
                families.append(family)
    if not families:
        family = str(protocol.model.name or "").strip().lower()
        if family:
            families.append(family)
    return list(dict.fromkeys(families))


def actual_config_snapshot(protocol: ExperimentProtocol | None) -> dict[str, Any]:
    """Return the compiled parameter snapshot used to audit a round."""

    if protocol is None:
        return {}
    features = protocol.features
    return {
        "target": protocol.target,
        "experiment_type": protocol.experiment_type,
        "forecast_horizon_days": (
            protocol.forecast_horizon_days
            if protocol.forecast_horizon_days is not None
            else features.forecast_horizon_days
        ),
        "past_lag_days": features.past_lag_days,
        "window_size": features.window_size,
        "control_lag_days": features.control_lag_days,
        "treatment_lag_days": features.treatment_lag_days,
        "control_forecast_horizon_days": features.control_forecast_horizon_days,
        "treatment_forecast_horizon_days": features.treatment_forecast_horizon_days,
        "control_feature_count": len(features.control or []),
        "treatment_feature_count": len(features.treatment or []),
        "design_focus": features.design_focus,
        "model_families": _models_from_protocol(protocol),
        "run_count": len(protocol.runs_spec or []),
        "contrast_count": len(protocol.contrasts_spec or []),
    }


def actual_changed_factor_labels(protocol: ExperimentProtocol | None) -> list[str]:
    """Turn the executed protocol into a compact human-readable factor list."""

    if protocol is None:
        return []
    snapshot = actual_config_snapshot(protocol)
    labels: list[str] = []
    horizon = snapshot.get("forecast_horizon_days")
    if horizon is not None:
        labels.append(f"预测超前天数 = {int(horizon)} 天")
    past_lag = snapshot.get("past_lag_days")
    if past_lag is not None:
        labels.append(f"历史滞后天数 = {int(past_lag)} 天")
    window = snapshot.get("window_size")
    if window is not None:
        labels.append(f"时间窗口 = {int(window)} 天")
    families = snapshot.get("model_families") or []
    if families:
        labels.append(f"模型家族 = {'/'.join(families)}")
    labels.append(f"对照组特征数 = {snapshot.get('control_feature_count', 0)}")
    labels.append(f"实验组特征数 = {snapshot.get('treatment_feature_count', 0)}")
    design_focus = snapshot.get("design_focus")
    if design_focus:
        labels.append(f"焦点变量 = {design_focus}")
    return labels


def protocol_delta_notes(
    protocol: ExperimentProtocol | None,
    actual_labels: list[str],
) -> list[str]:
    """Record known mismatches between planned prose and compiled execution."""

    if protocol is None:
        return []
    notes: list[str] = []
    planned = list(protocol.changed_factors or [])
    if protocol.exploratory:
        notes.append("协议标记为探索性实验：同时改变多个条件，不做单因素归因。")
    if not planned and not protocol.notes:
        return notes
    snapshot = actual_config_snapshot(protocol)
    horizon = snapshot.get("forecast_horizon_days")
    past_lag = snapshot.get("past_lag_days")
    window = snapshot.get("window_size")
    if planned and horizon is not None and any("5 天" in text for text in planned) and int(horizon) != 5:
        notes.append(
            f"计划文本提到预测超前改为 5 天，实际编译/执行仍为 {int(horizon)} 天；以编译后协议为准。"
        )
    if planned and past_lag is not None and any("7 天" in text for text in planned) and int(past_lag) != 7:
        notes.append(
            f"计划文本提到历史长度改为 7 天，实际编译/执行仍为 {int(past_lag)} 天；以编译后协议为准。"
        )
    if planned and window is not None and any("21 天" in text for text in planned) and int(window) != 21:
        notes.append(
            f"计划文本提到时间窗口改为 21 天，实际编译/执行仍为 {int(window)} 天；以编译后协议为准。"
        )
    markers = [
        note
        for note in (protocol.features.notes or [])
        if isinstance(note, str)
        and ("llm_controller" in note or note.startswith("repair:") or note.startswith("llm_candidate"))
    ]
    if markers:
        notes.append(
            "计划文案经控制器重写/修复（features.notes 标记："
            f"{'；'.join(markers[:3])}），实验内容以编译后 runs_spec 和参数字段为准。"
        )
    return notes


def selection_rationale(protocol: ExperimentProtocol | None) -> str | None:
    if protocol is None:
        return None
    parts: list[str] = []
    if protocol.responded_prior_finding:
        parts.append(f"回应上一轮发现：{protocol.responded_prior_finding}")
    if protocol.target_uncertainty_focus:
        parts.append(f"目标不确定性：{protocol.target_uncertainty_focus}")
    candidate = protocol.candidate_id or protocol.source_candidate_id
    if candidate:
        parts.append(f"候选实验：{candidate}")
    return "；".join(parts) or None


def populate_cross_round_fields(
    entry: RoundHistoryEntry,
    protocol: ExperimentProtocol | None,
    previous_entry: RoundHistoryEntry | None,
) -> RoundHistoryEntry:
    """Fill round-to-round linkage fields without touching user feedback."""

    entry.source_round_id = previous_entry.round_id if previous_entry is not None else None
    entry.source_findings = (
        list(previous_entry.scientific_findings or [])
        if previous_entry is not None
        else []
    )
    entry.source_delta_snapshot = (
        previous_entry.metrics_snapshot.model_dump(mode="json", exclude_none=True)
        if previous_entry is not None and previous_entry.metrics_snapshot is not None
        else None
    )
    entry.target_uncertainty_ids = list(
        entry.target_uncertainties
        or (list(protocol.target_uncertainties) if protocol else [])
        or []
    )
    if entry.selection_rationale is None:
        entry.selection_rationale = selection_rationale(protocol)
    if protocol is not None:
        planned = list(protocol.changed_factors or [])
        if planned:
            entry.planned_changed_factors = planned
        actual_labels = actual_changed_factor_labels(protocol)
        if actual_labels:
            entry.actual_changed_factors = actual_labels
        delta_notes = protocol_delta_notes(protocol, actual_labels)
        if delta_notes:
            merged = list(dict.fromkeys(list(entry.protocol_delta_notes or []) + delta_notes))
            entry.protocol_delta_notes = merged
        entry.candidate_id = (
            protocol.candidate_id
            or protocol.source_candidate_id
            or entry.approved_candidate_id
        )
        entry.executed_experiment_id = protocol.experiment_id
        entry.actual_config_snapshot = actual_config_snapshot(protocol)
    return entry
