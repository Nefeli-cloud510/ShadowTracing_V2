from __future__ import annotations

import json
import math
from pathlib import Path

from core.support_update_rules import (
    compute_support_update_from_prediction,
    observed_delta_from_comparison,
    signal_from_metric_deltas,
    signal_strength,
)
from core.robustness_diagnostics import summary_to_robustness_report
from core.hypothesis_identity import CANONICAL_HYPOTHESIS_IDS
from core.unified_schema import (
    BootstrapReport,
    ConclusionDataLayer,
    ConclusionExperimentLayer,
    ConclusionHypothesisRow,
    ConclusionScientificLayer,
    ConclusionTrackingLayer,
    DisagreementUpdate,
    EvidenceClaim,
    EvaluationResult,
    HypothesisAssessment,
    HypothesisPrediction,
    MetricDelta,
    ExperimentProtocol,
    ExperimentResult,
    PerformanceMetrics,
    RobustnessOverall,
    RobustnessReport,
    ScientificEvaluation,
    ThreeLayerConclusion,
    TimeSliceReport,
)

EPSILON = 1e-8
_CONCLUSION_THRESHOLD = 0.01
_DECISIVE_THRESHOLD = 0.02


def build_performance_metrics(
    result: ExperimentResult,
    *,
    protocol: ExperimentProtocol | None = None,
    prior_supports: dict[str, float] | None = None,
) -> PerformanceMetrics:
    baseline = _get_run(result, "baseline")
    treatment = _get_run(result, "treatment")
    baseline_rmse = baseline.metrics.rmse if baseline else None
    treatment_rmse = treatment.metrics.rmse if treatment else None
    pg_actual_signed = _compute_pg_actual_signed(
        baseline_rmse=baseline_rmse,
        treatment_rmse=treatment_rmse,
    )
    ig_audit = _posterior_information_gain_audit(
        protocol=protocol,
        result=result,
        prior_supports=prior_supports,
    )

    return PerformanceMetrics(
        experiment_id=result.experiment_id,
        round_id=result.round_id,
        baseline_rmse=baseline_rmse,
        baseline_mae=baseline.metrics.mae if baseline else None,
        baseline_pearson_r=baseline.metrics.pearson_r if baseline else None,
        baseline_skill=baseline.metrics.skill if baseline else None,
        treatment_rmse=treatment_rmse,
        treatment_mae=treatment.metrics.mae if treatment else None,
        treatment_pearson_r=treatment.metrics.pearson_r if treatment else None,
        treatment_skill=treatment.metrics.skill if treatment else None,
        pg_actual_signed=pg_actual_signed,
        pg_actual_clipped=min(max(pg_actual_signed, 0.0), 1.0),
        delta=result.comparison,
        information_gain_kl=ig_audit["kl"],
        ig_prior_probs=ig_audit["prior"],
        ig_posterior_probs=ig_audit["posterior"],
    )


def build_minimal_scientific_evaluation(
    result: ExperimentResult,
    *,
    protocol: ExperimentProtocol | None = None,
    remaining_uncertainties: list[dict] | None = None,
    main_question: str | None = None,
    prior_supports: dict[str, float] | None = None,
) -> ScientificEvaluation:
    evidence_for: list[EvidenceClaim] = []
    evidence_against: list[EvidenceClaim] = []
    assessments: list[HypothesisAssessment] = []
    disagreement_updates: list[DisagreementUpdate] = []

    rmse_delta = result.comparison.rmse
    pearson_delta = result.comparison.pearson_r
    improvement_signal = signal_from_metric_deltas(rmse_delta, pearson_delta)
    observed_delta = observed_delta_from_comparison(
        skill_delta=result.comparison.skill,
        pearson_delta=pearson_delta,
        rmse_delta=rmse_delta,
    )

    if improvement_signal == "supports":
        evidence_for.append(
            EvidenceClaim(
                claim="实验组相比对照组带来更好的预测表现，说明新增实验条件可能包含有效信息。",
                strength=signal_strength(rmse_delta, pearson_delta),
                source="experiment_result.comparison",
            )
        )
    elif improvement_signal == "weakens":
        evidence_against.append(
            EvidenceClaim(
                claim="实验组未优于对照组，当前新增实验条件的增量信息尚不充分。",
                strength=signal_strength(rmse_delta, pearson_delta),
                source="experiment_result.comparison",
            )
        )
    else:
        evidence_for.append(
            EvidenceClaim(
                claim="本轮实验结果呈混合信号，需要结合更多实验或稳健性分析继续判断。",
                strength="medium",
                source="experiment_result.comparison",
            )
        )

    if protocol and protocol.tested_hypotheses:
        for hypothesis_id in protocol.tested_hypotheses:
            prediction = protocol.hypothesis_predictions.get(hypothesis_id)
            if prediction is not None:
                outcome = compute_support_update_from_prediction(
                    current_support=(prior_supports or {}).get(hypothesis_id, 0.5),
                    expected_effect=prediction.expected_effect,
                    expected_range=prediction.expected_range,
                    observed_delta=observed_delta,
                )
                assessments.append(
                    HypothesisAssessment(
                        hypothesis_id=hypothesis_id,
                        support_before=outcome.support_before,
                        support_after=outcome.support_after,
                        status=outcome.status,
                        direction_matched=outcome.direction_matched,
                        magnitude_matched=outcome.magnitude_matched,
                        reason=_assessment_reason(
                            hypothesis_id,
                            prediction,
                            observed_delta,
                            outcome.direction_matched,
                            outcome.magnitude_matched,
                        ),
                    )
                )
            target_list = evidence_for if improvement_signal == "supports" else evidence_against
            display_label = _display_hypothesis_label(hypothesis_id)
            target_list.append(
                EvidenceClaim(
                    claim=f"{display_label} 在本轮被间接检验，但仍缺少与方向/量级预测严格对齐的证据。",
                    strength="medium",
                    source="protocol.tested_hypotheses",
                )
            )

    if protocol and protocol.target_uncertainties:
        disagreement_updates = _build_disagreement_updates(protocol, assessments)

    unresolved_from_disagreement = [
        {
            "uncertainty_id": item.uncertainty_id,
            "question": _uncertainty_question_from_protocol(protocol, item.uncertainty_id) if protocol else item.uncertainty_id,
            "description": item.summary,
            "priority": "high" if item.resolution_status == "unresolved" else "medium",
        }
        for item in disagreement_updates
        if item.resolution_status != "resolved"
    ]
    merged_remaining = list(remaining_uncertainties or []) + unresolved_from_disagreement

    three_layer = _build_three_layer_conclusion(
        result=result,
        protocol=protocol,
        observed_delta=observed_delta,
        assessments=assessments,
        main_question=main_question,
    )

    return ScientificEvaluation(
        experiment_id=result.experiment_id,
        round_id=result.round_id,
        hypothesis_assessments=assessments,
        disagreement_updates=disagreement_updates,
        evidence_summary={
            "new_evidence_for": evidence_for,
            "new_evidence_against": evidence_against,
            "remaining_uncertainties": merged_remaining,
        },
        three_layer_conclusion=three_layer,
    )


def build_minimal_robustness_report(result: ExperimentResult) -> RobustnessReport:
    complete = result.status == "completed" and len(result.runs) >= 2
    rmse_delta = result.comparison.rmse
    pearson_delta = result.comparison.pearson_r
    mixed_signal = (
        rmse_delta is not None
        and pearson_delta is not None
        and ((rmse_delta > 0 and pearson_delta < 0) or (rmse_delta < 0 and pearson_delta > 0))
    )
    stable = complete and not mixed_signal
    recommendation = (
        "可以进入下一轮假设解释，但建议补充 bootstrap/time-slice 稳健性检验。"
        if stable
        else "建议先补充更多稳健性检验，再据此更新假设支持度。"
    )

    return RobustnessReport(
        bootstrap=BootstrapReport(stable=stable, n_iterations=0),
        time_slice=TimeSliceReport(stable=stable),
        overall=RobustnessOverall(
            stable=stable,
            concern=None if stable else "当前仅完成单次主实验比较，稳健性证据不足。",
        ),
        recommendation=recommendation,
    )


def evaluate_experiment(
    result: ExperimentResult,
    *,
    protocol: ExperimentProtocol | None = None,
    remaining_uncertainties: list[dict] | None = None,
    main_question: str | None = None,
    prior_supports: dict[str, float] | None = None,
) -> EvaluationResult:
    robustness = (
        result.robustness
        if result.robustness is not None
        else summary_to_robustness_report(None)
    )
    return EvaluationResult(
        metrics=build_performance_metrics(
            result,
            protocol=protocol,
            prior_supports=prior_supports,
        ),
        robustness=robustness,
        scientific=build_minimal_scientific_evaluation(
            result,
            protocol=protocol,
            remaining_uncertainties=remaining_uncertainties,
            main_question=main_question,
            prior_supports=prior_supports,
        ),
        visualizations=result.visualizations,
    )


def export_evaluation_result(evaluation: EvaluationResult, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(evaluation.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def recover_three_layer_conclusion(
    evaluation: EvaluationResult,
    protocol: ExperimentProtocol,
    *,
    result: ExperimentResult | None = None,
    main_question: str | None = None,
) -> ThreeLayerConclusion | None:
    """Rebuild a missing three-layer conclusion for legacy persisted rounds.

    Older backend versions wrote ``evaluation_unified.json`` before the
    conclusion existed or serialized it as null.  This routine derives only the
    missing conclusion from the already persisted protocol, result and metrics,
    keeping historical hypothesis assessments intact.
    """
    if evaluation.scientific.three_layer_conclusion is not None:
        return evaluation.scientific.three_layer_conclusion
    if protocol is None or result is None or evaluation.metrics is None:
        return None

    metrics_delta = evaluation.metrics.delta or MetricDelta()
    if result.comparison is None:
        result.comparison = MetricDelta(
            rmse=metrics_delta.rmse,
            mae=metrics_delta.mae,
            pearson_r=metrics_delta.pearson_r,
            skill=metrics_delta.skill,
            r2=metrics_delta.r2,
        )
    observed_delta = observed_delta_from_comparison(
        skill_delta=metrics_delta.skill,
        pearson_delta=metrics_delta.pearson_r,
        rmse_delta=metrics_delta.rmse,
    )
    return _build_three_layer_conclusion(
        result=result,
        protocol=protocol,
        observed_delta=observed_delta,
        assessments=list(evaluation.scientific.hypothesis_assessments),
        main_question=main_question,
    )


def _hypothesis_layer_conclusion(
    hypothesis_id: str,
    direction_matched: bool | str | None,
    magnitude_matched: bool | str | None,
    predicted_direction: str | None,
    actual_delta: float | None,
) -> str:
    display_label = _display_hypothesis_label(hypothesis_id)
    if direction_matched is False:
        return f"{display_label} 观测方向与预期不符，证据不支持该约定"
    if magnitude_matched is False:
        return f"{display_label} 方向正确，幅度不足预期"
    if direction_matched is True:
        return f"{display_label} 与预期一致，获得支持"
    if predicted_direction == "near_zero" and actual_delta is not None and abs(actual_delta) < _CONCLUSION_THRESHOLD:
        return f"{display_label} 观测接近零效应，与预期一致"
    if actual_delta is not None and abs(actual_delta) < _CONCLUSION_THRESHOLD:
        return f"{display_label} 观测接近零增益，证据不足以支持"
    return f"{display_label} 证据方向待后续实验继续收敛"


def _build_three_layer_conclusion(
    *,
    result: ExperimentResult,
    protocol: ExperimentProtocol | None,
    observed_delta: float,
    assessments: list[HypothesisAssessment],
    main_question: str | None,
) -> ThreeLayerConclusion | None:
    """Build the experiment -> hypothesis -> scientific-question conclusion."""
    if protocol is None:
        return None
    design = protocol.features
    baseline = _get_run(result, "baseline")
    treatment = _get_run(result, "treatment")
    baseline_rmse = baseline.metrics.rmse if baseline is not None else None
    treatment_rmse = treatment.metrics.rmse if treatment is not None else None
    baseline_pearson_r = baseline.metrics.pearson_r if baseline is not None else None
    treatment_pearson_r = treatment.metrics.pearson_r if treatment is not None else None
    skill_delta = result.comparison.skill

    horizon = design.treatment_forecast_horizon_days or design.forecast_horizon_days or protocol.forecast_horizon_days or 0
    control_lag = design.control_lag_days or design.past_lag_days
    treatment_lag = design.treatment_lag_days or design.past_lag_days
    control_text = ", ".join(design.control) or "校准对照组"
    treatment_text = ", ".join(design.treatment) or "实验组"
    if control_lag != treatment_lag:
        design_summary = (
            f"对照组 [{control_text}]（滞后 {control_lag} 天）→ "
            f"实验组 [{treatment_text}]（滞后 {treatment_lag} 天），预测超前 {horizon} 天"
        )
    else:
        lag_text = f"，滞后 {control_lag} 天" if control_lag else ""
        design_summary = (
            f"对照组 [{control_text}] → 实验组 [{treatment_text}]{lag_text}，预测超前 {horizon} 天"
        )

    assessment_index = {item.hypothesis_id: item for item in assessments}
    hypothesis_rows: list[ConclusionHypothesisRow] = []
    for hypothesis_id in protocol.tested_hypotheses:
        prediction = protocol.hypothesis_predictions.get(hypothesis_id)
        assessment = assessment_index.get(hypothesis_id)
        predicted_direction = prediction.expected_effect if prediction is not None else None
        predicted_range = prediction.expected_range if prediction is not None else None
        direction_matched = assessment.direction_matched if assessment is not None else None
        magnitude_matched = assessment.magnitude_matched if assessment is not None else None
        hypothesis_rows.append(
            ConclusionHypothesisRow(
                hypothesis_id=hypothesis_id,
                statement=_display_hypothesis_label(hypothesis_id),
                predicted_direction=predicted_direction,
                predicted_range=predicted_range,
                actual_delta=observed_delta,
                direction_matched=direction_matched,
                magnitude_matched=magnitude_matched,
                conclusion=_hypothesis_layer_conclusion(
                    hypothesis_id,
                    direction_matched,
                    magnitude_matched,
                    predicted_direction,
                    observed_delta,
                ),
                support_after=assessment.support_after if assessment is not None else None,
            )
        )

    core_question = main_question or protocol.scientific_objective or "宇宙线日影南北偏移是否可提前改善太阳风速度预测？"
    if observed_delta >= _CONCLUSION_THRESHOLD:
        answer = (
            "能。观测到可比较的正向预测技能增量，日影南北偏移为太阳风速度预测提供超越对照组的前置信息。"
        )
    elif observed_delta <= -_CONCLUSION_THRESHOLD:
        answer = "暂不能。当前实验组未优于对照组，前置探针假设在本轮未获得支持。"
    else:
        answer = "尚不能明确判断。观测增量接近零，需要结合路径轴与窗口轴证据继续收敛。"

    path_question = "该增量信息通过什么物理路径传递？"
    if design.probe_axis == "physical_path":
        if observed_delta <= _CONCLUSION_THRESHOLD:
            path_answer = "主要通过与 IMF By 共享的物理路径传递；控制 By 后加入日影无额外增益。"
        else:
            path_answer = "控制 By 后仍观察到增量，日影存在超越 By 的独立预测信息。"
    elif design.probe_axis == "lead_time":
        path_answer = "本轮确认超前窗口有效性，路径归属仍需物理路径轴控制实验继续收敛。"
    else:
        path_answer = "本轮先确认前置增量是否存在，中介路径由物理路径轴继续检验。"

    if baseline_pearson_r is not None and treatment_pearson_r is not None:
        evidence_text = (
            f"ΔPearson r={treatment_pearson_r - baseline_pearson_r:+.4f}，"
            f"Pearson r {baseline_pearson_r:.4f} → {treatment_pearson_r:.4f}"
        )
    elif baseline_pearson_r is not None or treatment_pearson_r is not None:
        evidence_text = (
            f"Pearson r 未成对取得（对照组={baseline_pearson_r}、实验组={treatment_pearson_r}）"
        )
    else:
        evidence_text = "Pearson r 未取得"
    if baseline_rmse is not None and treatment_rmse is not None:
        evidence_text += f"，RMSE {baseline_rmse:.3f} → {treatment_rmse:.3f}"
    evidence_text += f"，预测超前 {horizon} 天"

    return ThreeLayerConclusion(
        experiment_layer=ConclusionExperimentLayer(
            experiment_id=result.experiment_id,
            design_summary=design_summary,
            probe_axis=design.probe_axis,
            forecast_horizon_days=horizon,
            baseline_rmse=baseline_rmse,
            treatment_rmse=treatment_rmse,
            baseline_pearson_r=baseline_pearson_r,
            treatment_pearson_r=treatment_pearson_r,
            skill_delta=skill_delta,
            decisive=abs(observed_delta) >= _DECISIVE_THRESHOLD,
        ),
        hypothesis_layer=hypothesis_rows,
        scientific_layer=ConclusionScientificLayer(
            main_question=core_question,
            answer=answer,
            path_question=path_question,
            path_answer=path_answer,
            evidence_text=evidence_text,
        ),
        data_layer=_build_default_data_layer(
            baseline_rmse=baseline_rmse,
            treatment_rmse=treatment_rmse,
            baseline_pearson_r=baseline_pearson_r,
            treatment_pearson_r=treatment_pearson_r,
            skill_delta=skill_delta,
            observed_delta=observed_delta,
        ),
        tracking_layer=ConclusionTrackingLayer(
            audit_items=[
                "指标由程序计算，不依赖 LLM 数值判断。",
                "假设支持度由程序按方向/幅度匹配规则更新。",
                "状态转换遵循 active/observing/converged/pruned 状态机。",
            ],
            sources=[
                "core.harness_unified.UnifiedExperimentHarness",
                "core.evaluation_unified.evaluate_experiment",
                "core.scientific_interpreter_llm.ScientificInterpreterLLM",
            ],
            snapshot_refs=[
                f"experiment:{result.experiment_id}",
                f"protocol:{protocol.experiment_id}",
            ],
        ),
    )


def _build_default_data_layer(
    *,
    baseline_rmse: float | None,
    treatment_rmse: float | None,
    baseline_pearson_r: float | None,
    treatment_pearson_r: float | None,
    skill_delta: float | None,
    observed_delta: float,
) -> ConclusionDataLayer:
    if baseline_rmse is not None and treatment_rmse is not None:
        rmse_text = (
            f"RMSE 由 {baseline_rmse:.4f} 变化至 {treatment_rmse:.4f}，"
            f"ΔRMSE={baseline_rmse - treatment_rmse:+.4f}。"
            "该变化反映整体误差水平，需结合数据覆盖与缺失对齐判断是否源自特征信息。"
        )
    else:
        rmse_text = "本轮未取得成对 RMSE，无法做增量归因。"
    if baseline_pearson_r is not None and treatment_pearson_r is not None:
        pearson_text = (
            f"Pearson r 由 {baseline_pearson_r:.4f} 变化至 {treatment_pearson_r:.4f}，"
            f"ΔPearson r={treatment_pearson_r - baseline_pearson_r:+.4f}。"
            "该变化反映趋势捕捉能力，趋势增强但误差未降时需警惕方向性偏差。"
        )
    else:
        pearson_text = "本轮未取得成对 Pearson r，无法做趋势归因。"
    if skill_delta is not None:
        skill_text = f"ΔSkill={skill_delta:+.4f}，表示实验组相对对照组的预测技能增量；"
    else:
        skill_text = "本轮未取得成对 Skill 增量。"
    skill_text += "技能增量须与 RMSE、Pearson r 方向一致才构成稳定证据。"
    anomalies: list[str] = []
    if (
        baseline_rmse is not None
        and treatment_rmse is not None
        and baseline_pearson_r is not None
        and treatment_pearson_r is not None
        and (baseline_rmse - treatment_rmse) > 0
        and (treatment_pearson_r - baseline_pearson_r) < 0
    ):
        anomalies.append("RMSE 改善但 Pearson r 下降，属于误差与趋势方向不一致的异常信号。")
    elif observed_delta >= 0.01 and skill_delta is not None and skill_delta < 0:
        anomalies.append("综合增量方向与 Skill 增量方向不一致。")
    if not anomalies:
        anomalies.append("未发现明显方向冲突，仍须结合时间片稳健性继续确认。")
    return ConclusionDataLayer(
        rmse_attribution=rmse_text,
        pearson_attribution=pearson_text,
        skill_delta_meaning=skill_text,
        anomalies=anomalies,
        next_focus=(
            "下一轮优先检查数据覆盖、缺失对齐与时间窗口稳定性，"
            "再结合路径控制实验收敛假设支持度。"
        ),
    )


def _get_run(result: ExperimentResult, run_name: str):
    for run in result.runs:
        if run.name == run_name:
            return run
    return None


def _assessment_reason(
    hypothesis_id: str,
    prediction: HypothesisPrediction | None,
    observed_delta: float,
    direction_matched: bool | str,
    magnitude_matched: bool | str,
) -> str:
    display_label = _display_hypothesis_label(hypothesis_id)
    if prediction is None:
        return f"{display_label} 缺少结构化预测范围，当前仅依据实验增量信号做最小判断。"
    return (
        f"{display_label} 预期效应为 {prediction.expected_effect}，"
        f"观测到的增量为 {observed_delta:.4f}；"
        f"方向匹配情况为{_matched_label(direction_matched)}，"
        f"幅度匹配情况为{_matched_label(magnitude_matched)}。"
    )


def _display_hypothesis_label(hypothesis_id: str) -> str:
    if hypothesis_id in CANONICAL_HYPOTHESIS_IDS:
        return f"H{CANONICAL_HYPOTHESIS_IDS.index(hypothesis_id) + 1}"
    if str(hypothesis_id).startswith("H_supplemental_"):
        return "补充假设"
    return hypothesis_id


def _matched_label(value: bool | str | None) -> str:
    if value is True:
        return "一致"
    if value is False:
        return "不一致"
    if value == "partial":
        return "部分一致"
    return "未知"


def _build_disagreement_updates(
    protocol: ExperimentProtocol,
    assessments: list[HypothesisAssessment],
) -> list[DisagreementUpdate]:
    assessment_index = {item.hypothesis_id: item for item in assessments}
    updates: list[DisagreementUpdate] = []
    for uncertainty_id in protocol.target_uncertainties:
        disagreement = protocol.disagreement_context.get(uncertainty_id, {})
        if not disagreement:
            continue
        before_scores = {
            hypothesis_id: (item.support_score if hasattr(item, "support_score") and item.support_score is not None else 0.5)
            for hypothesis_id, item in disagreement.items()
        }
        after_scores = {
            hypothesis_id: assessment_index[hypothesis_id].support_after
            if hypothesis_id in assessment_index
            else before_scores[hypothesis_id]
            for hypothesis_id in disagreement
        }
        sorted_after = sorted(after_scores.items(), key=lambda item: item[1], reverse=True)
        before_span = round(max(before_scores.values()) - min(before_scores.values()), 4) if before_scores else 0.0
        after_span = round(max(after_scores.values()) - min(after_scores.values()), 4) if after_scores else 0.0
        top_gap = round(sorted_after[0][1] - sorted_after[1][1], 4) if len(sorted_after) >= 2 else sorted_after[0][1]
        narrowed = after_span <= max(before_span - 0.03, before_span * 0.75)
        if top_gap >= 0.18 and sorted_after[0][1] >= 0.58:
            resolution_status = "resolved"
        elif narrowed or top_gap >= 0.10:
            resolution_status = "partially_resolved"
        else:
            resolution_status = "unresolved"
        unresolved_hypotheses = [
            hypothesis_id
            for hypothesis_id, score in after_scores.items()
            if sorted_after and sorted_after[0][1] - score < 0.12
        ]
        question = _uncertainty_question_from_protocol(protocol, uncertainty_id)
        summary_question = (
            f"“{question}”"
            if question and question != uncertainty_id
            else "对应科学不确定性"
        )
        leading_hypothesis = sorted_after[0][0] if sorted_after else None
        leading_text = (
            _display_hypothesis_label(leading_hypothesis)
            if leading_hypothesis
            else "无"
        )
        status_text = {
            "resolved": "已解决",
            "partially_resolved": "部分解决",
            "unresolved": "未解决",
        }[resolution_status]
        narrowed_text = "已收窄" if narrowed else "尚未收窄"
        summary = (
            f"{summary_question} 实验后支持度跨度由 {before_span:.3f} 变化至 {after_span:.3f}，"
            f"前沿假设为 {leading_text}，区分状态为{status_text}，跨度{narrowed_text}。"
        )
        updates.append(
            DisagreementUpdate(
                uncertainty_id=uncertainty_id,
                compared_hypotheses=list(disagreement.keys()),
                support_span_before=before_span,
                support_span_after=after_span,
                narrowed=narrowed,
                resolution_status=resolution_status,
                leading_hypothesis_id=sorted_after[0][0] if sorted_after else None,
                unresolved_hypotheses=unresolved_hypotheses,
                summary=summary,
            )
        )
    return updates


def _uncertainty_question_from_protocol(protocol: ExperimentProtocol, uncertainty_id: str) -> str:
    for step in protocol.steps:
        if step.action == "resolve_disagreement" and step.parameters.get("uncertainty_id") == uncertainty_id:
            return step.parameters.get("question") or uncertainty_id
    return uncertainty_id


def _compute_pg_actual_signed(
    *,
    baseline_rmse: float | None,
    treatment_rmse: float | None,
) -> float:
    if baseline_rmse is None or treatment_rmse is None:
        return 0.0
    return (baseline_rmse - treatment_rmse) / max(baseline_rmse, EPSILON)


def _posterior_information_gain_audit(
    *,
    protocol: ExperimentProtocol | None,
    result: ExperimentResult | None,
    prior_supports: dict[str, float] | None,
) -> dict[str, object]:
    """KL(P_posterior || P_prior) audit after an experiment has real results."""
    empty: dict[str, object] = {"kl": None, "prior": None, "posterior": None}
    if protocol is None or result is None or not protocol.hypothesis_predictions:
        return dict(empty)
    observed_delta = observed_delta_from_comparison(
        skill_delta=result.comparison.skill,
        pearson_delta=result.comparison.pearson_r,
        rmse_delta=result.comparison.rmse,
    )
    ranged = [
        (hypothesis_id, prediction)
        for hypothesis_id, prediction in protocol.hypothesis_predictions.items()
        if hypothesis_id in protocol.tested_hypotheses
        and prediction.expected_range is not None
        and len(prediction.expected_range) == 2
    ]
    if len(ranged) < 2:
        return dict(empty)
    if prior_supports:
        raw_priors = [prior_supports.get(hypothesis_id, 0.5) for hypothesis_id, _ in ranged]
    else:
        raw_priors = [1.0 / len(ranged)] * len(ranged)
    prior_total = sum(max(value, EPSILON) for value in raw_priors)
    priors = [max(value, EPSILON) / prior_total for value in raw_priors]

    likelihoods: list[float] = []
    for _, prediction in ranged:
        lower, upper = prediction.expected_range  # type: ignore[misc]
        mean = (lower + upper) / 2.0
        sigma = max(abs(upper - lower) / 4.0, EPSILON)
        likelihoods.append(max(_normal_density_value(observed_delta, mean, sigma), EPSILON))

    unnormalized = [
        likelihood * prior
        for likelihood, prior in zip(likelihoods, priors)
    ]
    posterior_total = sum(unnormalized)
    posteriors = [value / posterior_total for value in unnormalized]
    kl = sum(
        posterior * math.log(posterior / max(prior, EPSILON))
        for posterior, prior in zip(posteriors, priors)
    )
    return {
        "kl": max(kl, 0.0),
        "prior": dict(zip((hypothesis_id for hypothesis_id, _ in ranged), priors)),
        "posterior": dict(zip((hypothesis_id for hypothesis_id, _ in ranged), posteriors)),
    }


def _normal_density_value(x: float, mean: float, sigma: float) -> float:
    z = (x - mean) / max(sigma, EPSILON)
    return math.exp(-0.5 * z * z) / (max(sigma, EPSILON) * math.sqrt(2.0 * math.pi))
