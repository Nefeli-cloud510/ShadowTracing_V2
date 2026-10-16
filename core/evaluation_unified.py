from __future__ import annotations

import json
import math
from pathlib import Path

from core.support_update_rules import (
    EVIDENCE_CLASSIFICATION_LABELS,
    EVIDENCE_INCONCLUSIVE,
    EVIDENCE_IDENTICAL_PREDICTIONS,
    EVIDENCE_INVALID,
    EVIDENCE_NO_STABLE_GAIN,
    EVIDENCE_NOT_EXECUTED,
    EVIDENCE_STABLE_DEGRADATION,
    EVIDENCE_STABLE_IMPROVEMENT,
    classify_rmse_gain_evidence,
    compute_support_update_from_evidence,
    observed_delta_from_comparison,
)
from core.robustness_diagnostics import summary_to_robustness_report
from core.hypothesis_identity import CANONICAL_HYPOTHESIS_IDS
from core.json_safe import json_dumps
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
        experiment_type=result.experiment_type,
        runs=list(result.runs),
        contrasts=list(result.contrasts),
        diagnostics=list(result.diagnostics),
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
    rmse_ci = _rmse_ci_from_result(result)
    evidence = classify_rmse_gain_evidence(
        rmse_delta=rmse_delta,
        rmse_ci=rmse_ci,
        execution_status=str(result.status or "not_run"),
        validity_status=str(result.validity_status or "valid"),
        allowed_conclusions=result.allowed_conclusions,
        warnings=result.warnings,
    )
    classification = evidence.classification
    observed_delta = observed_delta_from_comparison(
        skill_delta=result.comparison.skill,
        pearson_delta=pearson_delta,
        rmse_delta=rmse_delta,
        primary_metric="rmse",
    )

    if classification == EVIDENCE_STABLE_IMPROVEMENT:
        evidence_for.append(
            EvidenceClaim(
                claim=(
                    "bootstrap RMSE 增益 95% CI 整体为正，"
                    "实验组相比对照组存在稳定误差改善。"
                ),
                strength="high" if rmse_ci else "medium",
                source="experiment_result.robustness.bootstrap.rmse_ci",
            )
        )
    elif classification == EVIDENCE_STABLE_DEGRADATION:
        evidence_against.append(
            EvidenceClaim(
                claim=(
                    "bootstrap RMSE 增益 95% CI 整体为负，"
                    "实验组相对对照组的误差出现稳定退化。"
                ),
                strength="high" if rmse_ci else "medium",
                source="experiment_result.robustness.bootstrap.rmse_ci",
            )
        )
    elif classification in {EVIDENCE_INVALID, EVIDENCE_NOT_EXECUTED}:
        evidence_against.append(
            EvidenceClaim(
                claim="本轮实验无效或未完成，仅保留审计证据，不做支持度判定。",
                strength="high",
                source="experiment_result.validity_status",
            )
        )
    elif classification == EVIDENCE_IDENTICAL_PREDICTIONS:
        evidence_for.append(
            EvidenceClaim(
                claim=(
                    "两组预测完全一致（RMSE 增益 CI=[0,0]），"
                    "未检出任何预测差异；该结果不能直接证明日影无物理作用，"
                    "需先核查特征是否真正进入模型或系数是否被压缩为零。"
                ),
                strength="medium",
                source="experiment_result.robustness.bootstrap.rmse_ci",
            )
        )
    else:
        evidence_for.append(
            EvidenceClaim(
                claim=(
                    "RMSE 增益 CI 跨零或缺少 bootstrap 稳定性证据，"
                    "当前不足以确定支持度方向。"
                ),
                strength="medium",
                source="experiment_result.comparison",
            )
        )

    if protocol and protocol.tested_hypotheses:
        for hypothesis_id in protocol.tested_hypotheses:
            prediction = protocol.hypothesis_predictions.get(hypothesis_id)
            if prediction is not None:
                metric = _normalize_metric(prediction.metric)
                metric_delta = (
                    _metric_delta_from_result(result, metric)
                    if metric is not None
                    else None
                )
                metric_ci = (
                    _metric_ci_from_result(result, metric)
                    if metric is not None
                    else None
                )
                # 预期区间只允许与同指标的实际增量比较；metric 缺失或取不到
                # 对应数值时不得拿 RMSE 套用，避免量纲混用。
                comparable_range = (
                    prediction.expected_range
                    if metric is not None and metric_delta is not None
                    else None
                )
                outcome = compute_support_update_from_evidence(
                    current_support=min(
                        max((prior_supports or {}).get(hypothesis_id, 0.5), 0.05),
                        0.95,
                    ),
                    evidence_classification=classification,
                    evidence_labels=evidence.labels,
                    expected_effect=prediction.expected_effect,
                    expected_range=comparable_range,
                    observed_delta=metric_delta if metric is not None else None,
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
                            metric_delta,
                            outcome.direction_matched,
                            outcome.magnitude_matched,
                            evidence_classification=classification,
                            metric=metric,
                            metric_ci=metric_ci,
                        ),
                        metric=metric or "unknown",
                        observed_metric_delta=metric_delta,
                        metric_ci=metric_ci,
                        evidence_classification=classification,
                        evidence_labels=evidence.labels,
                        updated=outcome.updated,
                    )
                )
            display_label = _display_hypothesis_label(hypothesis_id)
            if classification == EVIDENCE_STABLE_IMPROVEMENT:
                target_list = evidence_for
            elif classification == EVIDENCE_STABLE_DEGRADATION:
                target_list = evidence_against
            else:
                target_list = evidence_for
            target_list.append(
                EvidenceClaim(
                    claim=(
                        f"{display_label} 在本轮获得了一次对照检验；"
                        "支持度仅在稳定增益/退化时调整，"
                        f"当前证据分类为{EVIDENCE_CLASSIFICATION_LABELS.get(classification, classification)}。"
                    ),
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
    if result.robustness is not None:
        return result.robustness
    rmse_delta = result.comparison.rmse
    pearson_delta = result.comparison.pearson_r
    mixed_signal = (
        rmse_delta is not None
        and pearson_delta is not None
        and ((rmse_delta > 0 and pearson_delta < 0) or (rmse_delta < 0 and pearson_delta > 0))
    )
    concern = (
        "RMSE 与 Pearson r 方向冲突，不能仅凭点估计判定稳定。"
        if mixed_signal
        else "未运行配对日历块 bootstrap，无法判定主增益是否稳定。"
    )
    return RobustnessReport(
        bootstrap=BootstrapReport(stable=False, n_iterations=0, status="not_run"),
        time_slice=TimeSliceReport(stable=False),
        overall=RobustnessOverall(stable=False, concern=concern),
        recommendation="补充配对日历块 bootstrap 后再作方向性支持度结论。",
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
        json_dumps(evaluation.model_dump(mode="json", exclude_none=True), indent=2),
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
    evidence_classification: str | None = None,
    metric_ci: list[float] | None = None,
    observed_metric_delta: float | None = None,
) -> str:
    display_label = _display_hypothesis_label(hypothesis_id)
    if evidence_classification == EVIDENCE_IDENTICAL_PREDICTIONS:
        return (
            f"{display_label} 本轮两组预测完全一致（CI=[0,0]），"
            "未检出预测差异，不支持度判定；需先核查特征是否真正参与模型。"
        )
    if evidence_classification in {EVIDENCE_INCONCLUSIVE, EVIDENCE_NO_STABLE_GAIN}:
        return (
            f"{display_label} 主指标证据不足以稳定判定方向"
            f"（{EVIDENCE_CLASSIFICATION_LABELS.get(evidence_classification, evidence_classification)}），"
            "支持度暂不调整。"
        )
    if evidence_classification == EVIDENCE_STABLE_IMPROVEMENT:
        return f"{display_label} 主指标出现稳定改善，支持度小幅上调。"
    if evidence_classification == EVIDENCE_STABLE_DEGRADATION:
        return f"{display_label} 主指标出现稳定退化，支持度小幅下调。"
    if evidence_classification in {EVIDENCE_INVALID, EVIDENCE_NOT_EXECUTED}:
        return f"{display_label} 本轮实验无效或未完成，仅保留审计证据，支持度不变。"
    if direction_matched is False:
        return f"{display_label} 观测方向与预期不符，证据不支持该约定"
    if magnitude_matched is False:
        return f"{display_label} 方向正确，幅度不足预期"
    if direction_matched is True:
        return f"{display_label} 与预期一致，但需 bootstrap CI 确认后才调整支持度"
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
    first_assessment = next(
        (item for item in assessments if item.evidence_classification is not None),
        None,
    )
    evidence_classification = (
        first_assessment.evidence_classification
        if first_assessment is not None
        else None
    )
    rmse_ci = _rmse_ci_from_result(result)
    evidence_labels = (
        first_assessment.evidence_labels if first_assessment is not None else []
    )
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
        actual_delta = (
            assessment.observed_metric_delta
            if assessment is not None
            else None
        )
        hypothesis_rows.append(
            ConclusionHypothesisRow(
                hypothesis_id=hypothesis_id,
                statement=_display_hypothesis_label(hypothesis_id),
                predicted_direction=predicted_direction,
                predicted_range=predicted_range,
                actual_delta=actual_delta,
                direction_matched=direction_matched,
                magnitude_matched=magnitude_matched,
                conclusion=_hypothesis_layer_conclusion(
                    hypothesis_id,
                    direction_matched,
                    magnitude_matched,
                    predicted_direction,
                    actual_delta,
                    evidence_classification=(
                        assessment.evidence_classification
                        if assessment is not None
                        else evidence_classification
                    ),
                    metric_ci=(
                        assessment.metric_ci
                        if assessment is not None
                        else None
                    ),
                ),
                support_after=assessment.support_after if assessment is not None else None,
            )
        )

    core_question = main_question or protocol.scientific_objective or "宇宙线日影南北偏移是否可提前改善太阳风速度预测？"
    if evidence_classification == EVIDENCE_STABLE_IMPROVEMENT:
        answer = (
            "能。bootstrap RMSE 增益 95% CI 整体为正，"
            "日影南北偏移为太阳风速度预测提供超越对照组的稳定前置信息。"
        )
    elif evidence_classification == EVIDENCE_STABLE_DEGRADATION:
        answer = "暂不能。本轮检测到稳定负向 RMSE 增量，前置探针假设当前未获得支持。"
    elif evidence_classification == EVIDENCE_IDENTICAL_PREDICTIONS:
        answer = (
            "本轮暂无法回答：两组预测完全一致（RMSE 增益 CI=[0,0]），"
            "未检出任何预测差异。该结果不构成“日影无物理作用”的证据，"
            "需先核查特征是否真正进入模型、系数是否被压缩为零，"
            "或对照组与实验组是否意外使用了相同信息。"
        )
    elif evidence_classification in {EVIDENCE_INCONCLUSIVE, EVIDENCE_NO_STABLE_GAIN}:
        answer = (
            f"尚不能明确判断：点估计为 {observed_delta:+.4f}，"
            f"但{EVIDENCE_CLASSIFICATION_LABELS.get(evidence_classification, '稳健性证据不足')}，"
            "需补充或收敛 bootstrap 证据后再作方向性结论。"
        )
    elif evidence_classification in {EVIDENCE_INVALID, EVIDENCE_NOT_EXECUTED}:
        answer = "本轮实验无效或未完成，暂不对主科学问题做支持度结论。"
    elif observed_delta >= _CONCLUSION_THRESHOLD:
        answer = (
            "能。观测到可比较的正向预测技能增量，日影南北偏移为太阳风速度预测提供超越对照组的前置信息。"
        )
    elif observed_delta <= -_CONCLUSION_THRESHOLD:
        answer = "暂不能。当前实验组未优于对照组，前置探针假设在本轮未获得支持。"
    else:
        answer = "尚不能明确判断。观测增量接近零，需要结合路径轴与窗口轴证据继续收敛。"

    path_question = "该增量信息通过什么物理路径传递？"
    if evidence_classification in {
        EVIDENCE_IDENTICAL_PREDICTIONS,
        EVIDENCE_INCONCLUSIVE,
        EVIDENCE_NO_STABLE_GAIN,
        EVIDENCE_INVALID,
        EVIDENCE_NOT_EXECUTED,
    }:
        path_answer = "本轮主指标缺乏稳定方向证据，暂不对路径归属做结论。"
    elif design.probe_axis == "physical_path":
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
    if evidence_classification == EVIDENCE_IDENTICAL_PREDICTIONS:
        evidence_text += "，两组预测指标完全一致、未检出预测差异"
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
            experiment_type=result.experiment_type,
            runs=list(result.runs),
            contrasts=list(result.contrasts),
            diagnostics=list(result.diagnostics),
            decisive=(
                evidence_classification
                in {
                    EVIDENCE_STABLE_IMPROVEMENT,
                    EVIDENCE_STABLE_DEGRADATION,
                }
                and abs(observed_delta) >= _DECISIVE_THRESHOLD
            ),
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
            rmse_ci=rmse_ci,
            evidence_classification=evidence_classification,
            evidence_labels=evidence_labels,
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
    rmse_ci: list[float] | None = None,
    evidence_classification: str | None = None,
    evidence_labels: list[str] | None = None,
) -> ConclusionDataLayer:
    if baseline_rmse is not None and treatment_rmse is not None:
        rmse_text = (
            f"RMSE 由 {baseline_rmse:.4f} 变化至 {treatment_rmse:.4f}，"
            f"ΔRMSE={baseline_rmse - treatment_rmse:+.4f}。"
        )
        if (
            evidence_classification == EVIDENCE_IDENTICAL_PREDICTIONS
            and rmse_ci
            and len(rmse_ci) >= 2
            and float(rmse_ci[0]) == 0.0
            and float(rmse_ci[1]) == 0.0
        ):
            rmse_text += (
                "两组预测完全一致（bootstrap RMSE 增益 CI=[0,0]），"
                "当前未检出任何预测差异；"
                "这不代表日影没有物理作用，需核查特征是否实际参与模型、"
                "系数是否被压缩为零。"
            )
        if rmse_ci:
            rmse_text += (
                f"bootstrap RMSE 增益 95% CI=[{rmse_ci[0]:+.4f}, {rmse_ci[1]:+.4f}]，"
                "该区间决定是否认定稳定增益。"
            )
        else:
            rmse_text += "未取得 bootstrap CI，点估计不能单独确认稳定性。"
        if evidence_classification:
            rmse_text += (
                f"证据分类：{EVIDENCE_CLASSIFICATION_LABELS.get(evidence_classification, evidence_classification)}。"
            )
        rmse_text += "该变化反映整体误差水平，需结合数据覆盖与缺失对齐判断是否源自特征信息。"
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
    expected = str(run_name or "").strip().lower()
    for run in result.runs:
        if str(run.run_id or "").strip().lower() == expected:
            return run
        if str(run.name or "").strip().lower() == expected:
            return run
    return None


_METRIC_ALIASES = {
    "rmse": "rmse",
    "root_mean_squared_error": "rmse",
    "mae": "mae",
    "mean_absolute_error": "mae",
    "pearson_r": "pearson_r",
    "pearson": "pearson_r",
    "r": "pearson_r",
    "r2": "r2",
    "r_squared": "r2",
    "skill": "skill",
    "sst": "skill",
}


def _normalize_metric(metric: str | None) -> str | None:
    if metric is None:
        return None
    normalized = str(metric).strip().lower().replace("-", "_").replace(" ", "_")
    return _METRIC_ALIASES.get(normalized)


def _metric_label(metric: str | None) -> str:
    return {
        "rmse": "RMSE",
        "mae": "MAE",
        "pearson_r": "Pearson r",
        "r2": "R2",
        "skill": "Skill",
    }.get(_normalize_metric(metric) or "", "对应指标")


def _is_finite_number(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _valid_ci(ci) -> bool:
    if not ci or len(ci) < 2:
        return False
    try:
        lower, upper = float(ci[0]), float(ci[1])
    except (TypeError, ValueError):
        return False
    return math.isfinite(lower) and math.isfinite(upper)


def _metric_delta_from_result(
    result: ExperimentResult,
    metric: str | None,
) -> float | None:
    normalized = _normalize_metric(metric)
    if normalized is None:
        return None
    comparison = result.comparison
    if comparison is None:
        return None
    value = getattr(comparison, normalized, None)
    if value is None or not _is_finite_number(value):
        return None
    return float(value)


def _metric_ci_from_result(
    result: ExperimentResult,
    metric: str | None,
) -> list[float] | None:
    normalized = _normalize_metric(metric)
    if normalized is None:
        return None
    field = f"{normalized}_ci"
    bootstrap = result.robustness.bootstrap if result.robustness is not None else None
    if bootstrap is not None:
        ci = getattr(bootstrap, field, None)
        if _valid_ci(ci):
            return [float(ci[0]), float(ci[1])]
    for contrast in result.contrasts:
        if contrast.bootstrap_ci is None:
            continue
        ci = getattr(contrast.bootstrap_ci, field, None)
        if _valid_ci(ci):
            return [float(ci[0]), float(ci[1])]
    return None


def _rmse_ci_from_result(result: ExperimentResult) -> list[float] | None:
    return _metric_ci_from_result(result, "rmse")


def _assessment_reason(
    hypothesis_id: str,
    prediction: HypothesisPrediction | None,
    observed_delta: float | None,
    direction_matched: bool | str,
    magnitude_matched: bool | str,
    evidence_classification: str | None = None,
    metric: str | None = None,
    metric_ci: list[float] | None = None,
) -> str:
    display_label = _display_hypothesis_label(hypothesis_id)
    classification_text = EVIDENCE_CLASSIFICATION_LABELS.get(
        evidence_classification,
        "未取得明确证据分类",
    )
    metric_text = _metric_label(metric)
    if _valid_ci(metric_ci):
        ci_text = (
            f"{metric_text} 指标 bootstrap 95% CI="
            f"[{metric_ci[0]:+.4f}, {metric_ci[1]:+.4f}]"
        )
    elif metric is not None:
        ci_text = f"未取得 {metric_text} 指标的 bootstrap CI"
    else:
        ci_text = "未取得对应指标的 bootstrap CI"
    if observed_delta is None:
        delta_text = (
            f"未取得 {metric_text} 指标的观测增量"
            if metric is not None
            else "未取得对应指标的观测增量"
        )
    else:
        delta_text = f"观测到的 {metric_text} 指标增量为 {observed_delta:+.4f}"
    if prediction is None:
        return (
            f"{display_label} 缺少结构化预测范围；{ci_text}，"
            f"证据分类为{classification_text}，支持度暂不调整。"
        )
    return (
        f"{display_label} 预期效应为 {prediction.expected_effect}，"
        f"{delta_text}；"
        f"方向匹配情况为{_matched_label(direction_matched)}，"
        f"幅度匹配情况为{_matched_label(magnitude_matched)}；"
        f"{ci_text}，实验主证据分类为{classification_text}。"
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
    """Experiment-information audit after an experiment has real results.

    This is an informational audit only; it does not drive support updates.
    The main support update path uses RMSE point estimates plus bootstrap CI,
    while this audit follows the metric declared in the hypothesis prediction.
    """
    empty: dict[str, object] = {"kl": None, "prior": None, "posterior": None}
    if protocol is None or result is None or not protocol.hypothesis_predictions:
        return dict(empty)
    ranged = [
        (hypothesis_id, prediction)
        for hypothesis_id, prediction in protocol.hypothesis_predictions.items()
        if hypothesis_id in protocol.tested_hypotheses
        and prediction.expected_range is not None
        and len(prediction.expected_range) == 2
    ]
    if len(ranged) < 2:
        return dict(empty)
    audit_metrics = {
        _normalize_metric(prediction.metric)
        for _, prediction in ranged
    }
    if any(metric is None for metric in audit_metrics) or len(audit_metrics) != 1:
        # 区间所属指标缺失或混用单位时，不做正常似然审计，避免量纲混排。
        return dict(empty)
    audit_metric = next(iter(audit_metrics))
    observed_delta = _audit_delta_from_comparison(
        result.comparison,
        audit_metric,
    )
    if observed_delta is None:
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


def _audit_delta_from_comparison(
    comparison,
    metric: str | None,
) -> float | None:
    normalized = _normalize_metric(metric)
    if normalized is None or comparison is None:
        return None
    value = getattr(comparison, normalized, None)
    if value is None or not _is_finite_number(value):
        return None
    return float(value)


def _normal_density_value(x: float, mean: float, sigma: float) -> float:
    z = (x - mean) / max(sigma, EPSILON)
    return math.exp(-0.5 * z * z) / (max(sigma, EPSILON) * math.sqrt(2.0 * math.pi))
