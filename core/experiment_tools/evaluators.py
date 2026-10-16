"""Experiment-type-aware evaluation gate.

Only ``incremental_prediction`` may reuse the legacy automatic support update
path. Every other type stores its structured evidence but explicitly marks
the scientific assessment as pending instead of being treated as a single
``ΔPearson r`` support vote.
"""

from __future__ import annotations

from core.evaluation_unified import evaluate_experiment as evaluate_incremental
from core.support_update_rules import (
    EVIDENCE_IDENTICAL_PREDICTIONS,
    EVIDENCE_INCONCLUSIVE,
    EVIDENCE_NO_STABLE_GAIN,
    EVIDENCE_STABLE_DEGRADATION,
    EVIDENCE_STABLE_IMPROVEMENT,
)
from core.unified_schema import (
    ConclusionDataLayer,
    ConclusionExperimentLayer,
    ConclusionScientificLayer,
    ConclusionTrackingLayer,
    EvaluationResult,
    EvidenceClaim,
    EvidenceSummary,
    ExperimentProtocol,
    ExperimentResult,
    HypothesisAssessment,
    ModelSensitivityFamilyEvidence,
    PerformanceMetrics,
    RobustnessReport,
    ScientificEvaluation,
    ThreeLayerConclusion,
)
from core.experiment_tools.registry import normalize_model_family


def evaluate_experiment_result(
    result: ExperimentResult,
    *,
    protocol: ExperimentProtocol | None = None,
    remaining_uncertainties: list[dict] | None = None,
    main_question: str | None = None,
    prior_supports: dict[str, float] | None = None,
) -> EvaluationResult:
    """Dispatch evaluation by result.experiment_type."""

    if result.experiment_type == "incremental_prediction":
        return evaluate_incremental(
            result,
            protocol=protocol,
            remaining_uncertainties=remaining_uncertainties,
            main_question=main_question,
            prior_supports=prior_supports,
        )
    if result.experiment_type == "model_sensitivity":
        return _evaluate_model_sensitivity(
            result,
            protocol=protocol,
            remaining_uncertainties=remaining_uncertainties,
            main_question=main_question,
            prior_supports=prior_supports,
        )
    return _build_pending_scientific_evaluation(
        result,
        protocol=protocol,
        remaining_uncertainties=remaining_uncertainties,
        main_question=main_question,
    )


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value) if float(value) == float(value) else None
    except (TypeError, ValueError):
        return None


def _ci_crosses_zero(ci: list[float] | None) -> bool | None:
    """Return None when the interval is unusable, True when it crosses zero."""

    if not ci or len(ci) < 2:
        return None
    lower = _finite(ci[0])
    upper = _finite(ci[1])
    if lower is None or upper is None:
        return None
    if lower == 0.0 and upper == 0.0:
        return False
    return lower < 0.0 < upper or lower == 0.0 or upper == 0.0


def _ci_is_identical(ci: list[float] | None) -> bool:
    """True when the interval collapses to exactly [0, 0]."""

    if not ci or len(ci) < 2:
        return False
    lower = _finite(ci[0])
    upper = _finite(ci[1])
    return lower == 0.0 and upper == 0.0


def _family_for_contrast(
    result: ExperimentResult,
    contrast: object,
) -> str | None:
    run_index = {str(run.run_id).strip(): run for run in result.runs}
    run_ids = list(getattr(contrast, "run_ids", None) or [])
    for run_id in run_ids:
        run = run_index.get(str(run_id).strip())
        if run is not None:
            return normalize_model_family(str(run.model_family or "")) or "unknown"
    return None


def _family_evidences(result: ExperimentResult) -> list[dict[str, Any]]:
    """Group shadow-gain evidence per model family from executed contrasts."""

    per_family: dict[str, dict[str, Any]] = {}
    for contrast in result.contrasts:
        family = _family_for_contrast(result, contrast)
        key = family or "unknown"
        entry = per_family.setdefault(
            key,
            {
                "family": key,
                "contrast_id": getattr(contrast, "contrast_id", None),
                "point_gain_rmse": None,
                "point_gain_pearson_r": None,
                "rmse_ci": None,
                "pearson_ci": None,
                "crosses_zero": None,
                "evaluation": None,
            },
        )
        delta = getattr(contrast, "metrics_delta", None)
        if delta is not None:
            rmse = _finite(getattr(delta, "rmse", None))
            pearson = _finite(getattr(delta, "pearson_r", None))
            if rmse is not None:
                entry["point_gain_rmse"] = rmse
            if pearson is not None:
                entry["point_gain_pearson_r"] = pearson
        bootstrap = getattr(contrast, "bootstrap_ci", None)
        if bootstrap is not None:
            rmse_ci = getattr(bootstrap, "rmse_ci", None)
            pearson_ci = getattr(bootstrap, "pearson_r_ci", None)
            if rmse_ci:
                entry["rmse_ci"] = [v for v in rmse_ci[:2]]
            if pearson_ci:
                entry["pearson_ci"] = [v for v in pearson_ci[:2]]
        entry["crosses_zero"] = _ci_crosses_zero(entry["rmse_ci"])
        entry["evaluation"] = _classify_family_gain(entry)
    return list(per_family.values())


def _classify_family_gain(entry: dict[str, Any]) -> str:
    """Conservative classification for one model family's shadow gain."""

    ci = entry.get("rmse_ci")
    point = entry.get("point_gain_rmse")
    crosses_zero = entry.get("crosses_zero")
    if ci:
        if _ci_is_identical(ci) and (point is None or abs(float(point)) < 1e-9):
            return "identical_predictions"
        if crosses_zero is True:
            return "no_stable_gain"
        if crosses_zero is False and ci[0] > 0:
            return "stable_positive"
        if crosses_zero is False and ci[1] < 0:
            return "stable_negative"
        return "insufficient_blocks"
    if point is not None:
        return "point_positive" if point > 0 else "point_negative" if point < 0 else "point_zero"
    return "not_evaluated"


def _family_gain_label(evaluation: str) -> str:
    labels = {
        "stable_positive": "稳定正向增益",
        "stable_negative": "稳定负向增益",
        "no_stable_gain": "未检出稳定增益（CI 跨零）",
        "identical_predictions": "两组预测完全一致（无预测差异）",
        "insufficient_blocks": "CI 不可用 / 有效时间块不足",
        "point_positive": "点估计正向，无稳定证据",
        "point_negative": "点估计负向，无稳定证据",
        "point_zero": "点估计接近零",
        "not_evaluated": "未评估",
    }
    return labels.get(evaluation, evaluation)


def _model_sensitivity_evidence_classification(kind: str) -> str:
    mapping = {
        "stable_positive": EVIDENCE_STABLE_IMPROVEMENT,
        "stable_negative": EVIDENCE_STABLE_DEGRADATION,
        "identical_predictions": EVIDENCE_IDENTICAL_PREDICTIONS,
        "no_stable_gain": EVIDENCE_NO_STABLE_GAIN,
    }
    return mapping.get(kind, EVIDENCE_INCONCLUSIVE)


def _model_sensitivity_assessments(
    protocol: ExperimentProtocol | None,
    *,
    conclusion_kind: str,
    family_evidences: list[dict[str, Any]],
    prior_supports: dict[str, float] | None,
) -> list[HypothesisAssessment]:
    """Attach conservative per-hypothesis assessments without auto-updating support."""

    if protocol is None:
        return []
    evidence_classification = _model_sensitivity_evidence_classification(
        conclusion_kind
    )
    labels = [
        f"{entry['family']}: {_family_gain_label(entry['evaluation'])}"
        for entry in family_evidences
    ] or ["无可用家族对比结果"]
    assessments: list[HypothesisAssessment] = []
    for hypothesis_id in protocol.tested_hypotheses:
        before = min(
            max(float((prior_supports or {}).get(hypothesis_id, 0.5)), 0.05),
            0.95,
        )
        label_text = "；".join(labels) or "无可用家族结论"
        assessments.append(
            HypothesisAssessment(
                hypothesis_id=hypothesis_id,
                support_before=before,
                support_after=before,
                status="active",
                direction_matched=None,
                magnitude_matched=None,
                reason=(
                    f"模型敏感性实验仅提供结构化证据（{label_text}）："
                    "point/CI、方向与稳定证据分开记录，按保守门控不自动调整"
                    "支持度；待人工复核或使用按实验类型定义的正式证据规则写入。"
                ),
                metric="rmse",
                observed_metric_delta=None,
                metric_ci=None,
                evidence_classification=evidence_classification,
                evidence_labels=labels,
                updated=False,
            )
        )
    return assessments


def _evaluate_model_sensitivity(
    result: ExperimentResult,
    *,
    protocol: ExperimentProtocol | None,
    remaining_uncertainties: list[dict] | None,
    main_question: str | None,
    prior_supports: dict[str, float] | None,
) -> EvaluationResult:
    family_evidences = _family_evidences(result)
    classified = [entry["evaluation"] for entry in family_evidences]
    stable_positive = classified.count("stable_positive")
    stable_negative = classified.count("stable_negative")
    no_stable_gain = any(
        classification in {"no_stable_gain", "insufficient_blocks", "not_evaluated"}
        for classification in classified
    )
    all_identical = bool(classified) and all(
        classification == "identical_predictions" for classification in classified
    )
    stateful_stable = stable_positive > 0 or stable_negative > 0
    mixed_direction = stable_positive > 0 and stable_negative > 0
    point_gains = [
        entry["point_gain_rmse"]
        for entry in family_evidences
        if entry["point_gain_rmse"] is not None
    ]
    point_directions = {1 if value > 0 else -1 if value < 0 else 0 for value in point_gains}
    same_direction = len(point_directions) <= 1 if point_directions else None
    if all_identical:
        conclusion_kind = "identical_predictions"
    elif no_stable_gain:
        conclusion_kind = "no_stable_gain"
    elif stable_positive == len(family_evidences) and len(family_evidences) > 0:
        conclusion_kind = "stable_positive"
    elif stable_negative == len(family_evidences) and len(family_evidences) > 0:
        conclusion_kind = "stable_negative"
    elif mixed_direction:
        conclusion_kind = "mixed_direction"
    else:
        conclusion_kind = "not_evaluated"

    family_evidence_text = _format_family_evidence(family_evidences)
    overall_answer = _model_sensitivity_answer(conclusion_kind, same_direction, family_evidences)
    next_focus, suggestions = _model_sensitivity_next_actions(
        conclusion_kind,
        same_direction,
        classified,
    )
    new_evidence_for: list[EvidenceClaim] = []
    new_evidence_against: list[EvidenceClaim] = []
    if conclusion_kind == "stable_positive":
        new_evidence_for.append(
            EvidenceClaim(
                claim=(
                    f"模型敏感性实验显示所有模型家族均给出稳定正向日影增益"
                    f"（{family_evidence_text}）。"
                ),
                strength="medium",
                source="core.experiment_tools.evaluators:model_sensitivity",
            )
        )
    elif conclusion_kind == "stable_negative":
        new_evidence_against.append(
            EvidenceClaim(
                claim=(
                    f"模型敏感性实验显示所有模型家族均为稳定负向增益"
                    f"（{family_evidence_text}），需先核查数据与执行质量。"
                ),
                strength="medium",
                source="core.experiment_tools.evaluators:model_sensitivity",
            )
        )
    elif conclusion_kind == "identical_predictions":
        new_evidence_for.append(
            EvidenceClaim(
                claim=(
                    f"模型敏感性实验确认两组预测完全一致（CI=[0,0]）："
                    f"{family_evidence_text}。该结果不直接证明日影无物理作用，"
                    "需先核查特征是否真正进入模型或系数被压缩为零。"
                ),
                strength="medium",
                source="core.experiment_tools.evaluators:model_sensitivity",
            )
        )
    else:
        new_evidence_for.append(
            EvidenceClaim(
                claim=(
                    f"模型敏感性实验结果为“未检出稳定增益”："
                    f"{family_evidence_text}。点估计方向与稳定证据分开判断，"
                    "该结果不自动改变假设支持度。"
                ),
                strength="medium",
                source="core.experiment_tools.evaluators:model_sensitivity",
            )
        )

    evidence_summary = EvidenceSummary(
        new_evidence_for=new_evidence_for,
        new_evidence_against=new_evidence_against,
        remaining_uncertainties=list(remaining_uncertainties or []),
    )
    metrics = PerformanceMetrics(
        experiment_id=result.experiment_id,
        round_id=result.round_id,
        experiment_type=result.experiment_type,
        baseline_rmse=None,
        treatment_rmse=None,
        delta=result.comparison,
        runs=list(result.runs),
        contrasts=list(result.contrasts),
        diagnostics=list(result.diagnostics),
    )
    robustness_concern = (
        "模型敏感性按模型家族分别读取 CI，不合并成单一总体 CI。"
        if family_evidences
        else "未发现可用的 per-family contrast，无法计算家族内增益。"
    )
    robustness = result.robustness or RobustnessReport(
        overall={
            "stable": conclusion_kind == "stable_positive",
            "concern": robustness_concern,
        }
    )
    conclusion = ThreeLayerConclusion(
        experiment_layer=ConclusionExperimentLayer(
            experiment_id=result.experiment_id,
            design_summary=(
                f"实验类型=模型敏感性；模型家族={'/'.join(item['family'] for item in family_evidences) or '未识别'}；"
                f"runs={len(result.runs)}；contrasts={len(result.contrasts)}；"
                f"每家族按对照组/实验组配对计算日影增量。"
            ),
            probe_axis="model_sensitivity",
            experiment_type=result.experiment_type,
            runs=list(result.runs),
            contrasts=list(result.contrasts),
            diagnostics=list(result.diagnostics),
            model_sensitivity_families=[
                ModelSensitivityFamilyEvidence(
                    family=entry["family"],
                    contrast_id=entry["contrast_id"],
                    point_gain_rmse=entry["point_gain_rmse"],
                    point_gain_pearson_r=entry["point_gain_pearson_r"],
                    rmse_ci=entry["rmse_ci"],
                    pearson_ci=entry["pearson_ci"],
                    crosses_zero=entry["crosses_zero"],
                    evaluation=entry["evaluation"],
                    evaluation_label=_family_gain_label(entry["evaluation"]),
                )
                for entry in family_evidences
            ],
        ),
        scientific_layer=ConclusionScientificLayer(
            main_question=main_question
            or (protocol.scientific_objective if protocol else None)
            or "宇宙线日影南北偏移是否可提前改善太阳风速度预测？",
            answer=overall_answer,
            evidence_text=f"Per-family 日影增量：{family_evidence_text or '本结果未提供可用的家族内增量。'}",
        ),
        data_layer=ConclusionDataLayer(
            rmse_attribution=(
                "本结果来自模型敏感性评价器：分别按 Ridge / ElasticNet 家族计算"
                "对照臂与处理臂的 RMSE（首选）与 Pearson r（补充），"
                "并读取各家族的 bootstrap CI 判断稳定方向；CI 跨零一律写“未检出稳定增益”。"
            ),
            pearson_attribution=(
                "家族间点估计方向"
                + ("一致，" if same_direction is True else "不一致，" if same_direction is False else "暂无法判断，")
                + "但稳定证据单独判定。"
            ),
            skill_delta_meaning="模型敏感性不生成单一 Skill 结论；有效指标以各家族 metrics_delta 为准。",
            anomalies=[
                item
                for item in [
                    _family_anomaly(entry) for entry in family_evidences
                ]
                if item
            ],
            next_focus=next_focus,
        ),
        tracking_layer=ConclusionTrackingLayer(
            audit_items=[
                "未执行自动支持度增减：converged/supported 状态的判定必须由后续有方向的有效证据驱动。",
                "CI 跨零或缺少足够时间块时，结论固定为“未检出稳定增益”。",
                "模型敏感性只回答正则化/模型设置是否改变日影增量，不单独验证路径中介。",
                "尚未解决的路径问题保留在不确定性队列，不因点估计方向关闭。",
                f"suggestions:{'; '.join(suggestions)}",
            ],
            snapshot_refs=[f"experiment:{result.experiment_id}", *(entry["contrast_id"] for entry in family_evidences if entry["contrast_id"])],
        ),
    )
    return EvaluationResult(
        metrics=metrics,
        robustness=robustness,
        scientific=ScientificEvaluation(
            experiment_id=result.experiment_id,
            round_id=result.round_id,
            hypothesis_assessments=_model_sensitivity_assessments(
                protocol,
                conclusion_kind=conclusion_kind,
                family_evidences=family_evidences,
                prior_supports=prior_supports,
            ),
            disagreement_updates=[],
            evidence_summary=evidence_summary,
            three_layer_conclusion=conclusion,
        ),
        visualizations=result.visualizations,
    )


def _format_family_evidence(family_evidences: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for entry in family_evidences:
        family = entry["family"]
        point = entry["point_gain_rmse"]
        ci = entry["rmse_ci"]
        unit = "RMSE 增量"
        if ci:
            ci_text = f"[{ci[0]:.4g}, {ci[1]:.4g}]"
        else:
            ci_text = "无可用 CI"
        parts.append(f"{family}: {unit}={'-' if point is None else f'{point:.4g}'}, CI={ci_text}")
    return "；".join(parts) or "无"


def _family_anomaly(entry: dict[str, Any]) -> str | None:
    family = entry["family"]
    point = entry["point_gain_rmse"]
    ci = entry["rmse_ci"]
    if _ci_is_identical(ci):
        return (
            f"{family} 两组预测完全一致（CI=[0,0]），未检出任何预测差异；"
            "需核查该家族的对照/实验特征是否真实验证了日影变量。"
        )
    if point is None:
        return f"{family} 无法计算日影 RMSE 增量（常数预测或其他指标不可用）。"
    if ci and _ci_crosses_zero(ci) is True:
        return f"{family} 点估计 {point:.4g} 但 CI={ci} 跨零，不能判定稳定方向。"
    if not ci:
        return f"{family} 缺少 bootstrap CI，只能保留点估计参考。"
    return None


def _model_sensitivity_answer(
    conclusion_kind: str,
    same_direction: bool | None,
    family_evidences: list[dict[str, Any]],
) -> str:
    base = "宇宙线日影南北偏移在模型敏感性检验下"
    if conclusion_kind == "stable_positive":
        return (
            f"{base}获得稳定正向增益，且各模型家族方向一致；"
            "可作为“存在可复现预测增量”的暂定证据，仍需固定任务继续核验。"
        )
    if conclusion_kind == "stable_negative":
        return (
            f"{base}在全部已执行家族中均为稳定负向增益；"
            "应先核查样本对齐、特征方向和执行质量，暂不视为路径机制反证。"
        )
    if conclusion_kind == "mixed_direction":
        return (
            f"{base}在不同模型家族中呈现相反稳定方向；"
            "说明结论依赖模型设置，当前不能得出统一的日影增量结论。"
        )
    if conclusion_kind == "identical_predictions":
        return (
            f"{base}的当前证据为“两组预测完全一致”："
            "bootstrap RMSE 增益 CI 为 [0,0]，未检出任何预测差异。"
            "该结果不能直接证明日影没有物理作用，下一轮应优先核查"
            "特征是否真正进入模型、系数是否被压缩为零，再设计区分性实验。"
        )
    return (
        f"{base}的当前证据为“未检出稳定增益”"
        + (
            f"。虽然各家族点估计方向{'一致' if same_direction else '不一致'}，但 CI 宽、样本块少或缺少 CI，"
            "不足以判定稳定方向。"
            if same_direction is not None
            else "。点估计与稳定证据均不足以给出方向性结论。"
        )
        + " 下一轮应优先核验稳定性、正则化敏感性与数据质量。"
    )


def _model_sensitivity_next_actions(
    conclusion_kind: str,
    same_direction: bool | None,
    classified: list[str],
) -> tuple[str, list[str]]:
    suggestions: list[str] = []
    if conclusion_kind == "stable_positive":
        suggestions = [
            "在固定目标任务上做一次单因素正则化敏感性复核，确认增益不依赖 alpha/l1_ratio。",
            "记录各家族的系数与特征重要性，说明日影变量的贡献来源。",
            "下一轮保持特征集、预测超前与历史长度不变，只检验一个因素。",
        ]
        return "结论暂可复现，下一步做单因素稳定性复核。", suggestions
    if conclusion_kind == "stable_negative":
        suggestions = [
            "检查共同样本的缺失掩码、目标日标签和双臂特征是否按日历对齐。",
            "对比预测 CSV 的真实标签与报告目标，确认变量没有错标单位或错位。",
            "固定任务重跑一次，区分真实退化与执行批次差异。",
        ]
        return "先做数据质量与执行一致性核查，再决定是否调整特征方向。", suggestions
    if conclusion_kind == "identical_predictions":
        suggestions = [
            "对比对照组/实验组的预测 CSV，确认两组是否真的使用不同特征集，而不是读取了同一数据。",
            "检查日影特征的实际系数：若被 ElasticNet 筛零或处理后未生效，需单独记录并重跑。",
            "固定任务条件后补一个更少压缩的模型设置或显式特征注入检查，确认焦点变量真正进入模型。",
            "保留路径/中介不确定性的未解决状态，不能因“无差异”直接判定日影无用。",
        ]
        return "下一轮优先核查预测完全一致的原因，而不是把它当作日影无用的反证。", suggestions
    suggestions = [
        (
            "优先核验稳定性：固定特征集、时间窗口与预测超前，只比较两家族的同设置差异。"
            if same_direction is not False
            else "优先核验模型设置依赖：固定全部任务条件，比较 alpha/l1_ratio 敏感性。"
        ),
        "检查被 ElasticNet 筛零的系数与 Ridge 的系数差异，判断是否为正则化伪影。",
        "检查时间块样本数与 bootstrap 有效块数，不足时如实标记而不做方向性稳定判定。",
        "保留未解决的路径/中介问题，不因点估计方向而关闭。",
    ]
    return "下一轮优先核验稳定增益与正则化设置敏感性。", suggestions


def _build_pending_scientific_evaluation(
    result: ExperimentResult,
    *,
    protocol: ExperimentProtocol | None,
    remaining_uncertainties: list[dict] | None,
    main_question: str | None,
) -> EvaluationResult:
    experiment_type = result.experiment_type
    family_labels = sorted(
        {
            str(run.model_family or "未知模型")
            for run in result.runs
        }
    )
    metrics = PerformanceMetrics(
        experiment_id=result.experiment_id,
        round_id=result.round_id,
        experiment_type=experiment_type,
        delta=result.comparison,
        runs=list(result.runs),
        contrasts=list(result.contrasts),
        diagnostics=list(result.diagnostics),
    )
    robustness = result.robustness or RobustnessReport(
        overall={"stable": False, "concern": "该实验类型尚未配置自动稳健性汇总。"}
    )
    evidence = EvidenceSummary(
        new_evidence_for=[
            EvidenceClaim(
                claim=(
                    f"{experiment_type} 已保存结构化结果："
                    f"{len(result.runs)} 个 run（模型：{'/'.join(family_labels) or '未配置'}）、"
                    f"{len(result.contrasts)} 个 contrast、{len(result.diagnostics)} 个 diagnostic；"
                    "该类型尚未适配自动支持度更新，标记为待科学评价。"
                ),
                strength="medium",
                source="core.experiment_tools.evaluators",
            )
        ],
        new_evidence_against=[],
        remaining_uncertainties=list(remaining_uncertainties or []),
    )
    conclusion = ThreeLayerConclusion(
        experiment_layer=ConclusionExperimentLayer(
            experiment_id=result.experiment_id,
            design_summary=(
                f"实验类型={experiment_type}；runs={len(result.runs)}"
                f"（模型：{'/'.join(family_labels) or '未配置'}）；"
                f"contrasts={len(result.contrasts)}；diagnostics={len(result.diagnostics)}。"
            ),
            experiment_type=experiment_type,
            runs=list(result.runs),
            contrasts=list(result.contrasts),
            diagnostics=list(result.diagnostics),
        ),
        scientific_layer=ConclusionScientificLayer(
            main_question=main_question
            or (protocol.scientific_objective if protocol else None)
            or "宇宙线日影南北偏移是否可提前改善太阳风速度预测？",
            answer=(
                f"{experiment_type} 属于尚未适配自动科学结论的实验类型；"
                "本层由程序写入待评价标记，不生成默认支持度结论。"
            ),
            evidence_text=(
                "; ".join(
                    [
                        f"{item.contrast_id}: ΔRMSE={item.metrics_delta.rmse if item.metrics_delta else None}"
                        for item in result.contrasts[:3]
                    ]
                )
                if result.contrasts
                else "无已有对比。"
            ),
        ),
        data_layer=ConclusionDataLayer(
            rmse_attribution="程序只保存指标与结构，待科学评价后补充归因。",
            next_focus="先完成科学解释者/人工复核，再进行支持度判定。",
        ),
        tracking_layer=ConclusionTrackingLayer(
            audit_items=[
                "未适配自动更新：本类型不会转换为单一 Δr 支持票。",
                "结果已落盘，可被下一轮证据链引用。",
                "支持度更新需新评价器或人工复核确认。",
            ],
            snapshot_refs=[f"experiment:{result.experiment_id}"],
        ),
    )
    return EvaluationResult(
        metrics=metrics,
        robustness=robustness,
        scientific=ScientificEvaluation(
            experiment_id=result.experiment_id,
            round_id=result.round_id,
            hypothesis_assessments=[],
            disagreement_updates=[],
            evidence_summary=evidence,
            three_layer_conclusion=conclusion,
        ),
        visualizations=result.visualizations,
    )
