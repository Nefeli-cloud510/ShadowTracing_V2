"""Experiment-type-aware evaluation gate.

Only ``incremental_prediction`` may reuse the legacy automatic support update
path. Every other type stores its structured evidence but explicitly marks
the scientific assessment as pending instead of being treated as a single
``ΔPearson r`` support vote.
"""

from __future__ import annotations

from core.evaluation_unified import evaluate_experiment as evaluate_incremental
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
    PerformanceMetrics,
    RobustnessReport,
    ScientificEvaluation,
    ThreeLayerConclusion,
)


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
    return _build_pending_scientific_evaluation(
        result,
        protocol=protocol,
        remaining_uncertainties=remaining_uncertainties,
        main_question=main_question,
    )


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
