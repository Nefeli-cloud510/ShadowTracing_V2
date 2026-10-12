"""Registry of executable scientific experiment tool types.

Only tools whose executor, validator and evaluator are connected may be
declared available. Everything else must return an explicit unsupported
status instead of falling back to an incremental treatment comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentToolRecord:
    """One registered scientific check type and its declared capability."""

    experiment_type: str
    version: str
    required_data_fields: list[str]
    allowed_manipulated_factors: list[str]
    fixed_conditions: list[str]
    protocol_fields: list[str]
    output_fields: list[str]
    applicability: str
    refusal_reasons: list[str]
    retrain_required: bool
    executor_name: str | None
    evaluator_name: str | None
    available: bool = False
    notes: list[str] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        return {
            "experiment_type": self.experiment_type,
            "version": self.version,
            "available": self.available,
            "retrain_required": self.retrain_required,
            "required_data_fields": list(self.required_data_fields),
            "allowed_manipulated_factors": list(self.allowed_manipulated_factors),
            "fixed_conditions": list(self.fixed_conditions),
            "protocol_fields": list(self.protocol_fields),
            "output_fields": list(self.output_fields),
            "applicability": self.applicability,
            "refusal_reasons": list(self.refusal_reasons),
            "executor": self.executor_name,
            "evaluator": self.evaluator_name,
            "notes": list(self.notes),
        }


def _incremental_tool() -> ExperimentToolRecord:
    return ExperimentToolRecord(
        experiment_type="incremental_prediction",
        version="1.0.0",
        required_data_fields=["target", "baseline_features", "treatment_features"],
        allowed_manipulated_factors=["focus_feature", "baseline_feature_set"],
        fixed_conditions=[
            "target",
            "common_calendar_sample",
            "calendar_split",
            "model_family",
            "forecast_horizon_days",
        ],
        protocol_fields=[
            "experiment_type",
            "features.control",
            "features.treatment",
            "model",
            "forecast_horizon_days",
            "runs_spec",
            "contrasts_spec",
        ],
        output_fields=[
            "runs",
            "comparison",
            "contrasts",
            "robustness.bootstrap",
            "data_coverage",
        ],
        applicability="检验新增焦点特征在固定基线和共同样本之外是否提供额外预测信息。",
        refusal_reasons=[
            "对照组与实验组不是 base / base+focus 关系",
            "双臂样本未按共同日历对齐",
            "实验类型与执行结构不一致",
        ],
        retrain_required=True,
        executor_name="incremental_prediction",
        evaluator_name="incremental_prediction",
        available=True,
        notes=[
            "复用现有双臂共同样本执行器，不另写样本构造。",
            "评价只做粗粒度说明，不得把性能改善直接写成假设支持。",
        ],
    )


def _model_sensitivity_tool() -> ExperimentToolRecord:
    return ExperimentToolRecord(
        experiment_type="model_sensitivity",
        version="1.0.0",
        required_data_fields=["target", "baseline_features", "treatment_features", "model_families"],
        allowed_manipulated_factors=["model_family", "regularization_settings"],
        fixed_conditions=[
            "target",
            "common_calendar_sample",
            "calendar_split",
            "feature_sets",
            "forecast_horizon_days",
            "within_family_baseline/treatment_pairing",
        ],
        protocol_fields=[
            "experiment_type",
            "features.control",
            "features.treatment",
            "model.parameters.model_families",
            "runs_spec",
            "contrasts_spec",
        ],
        output_fields=["runs", "contrasts", "channel_gains", "warnings"],
        applicability="判断日影增量是否依赖模型家族或正则化设置；先在各自家族内比较加日影前后的增益。",
        refusal_reasons=[
            "缺少多个模型家族的双臂结构",
            "把 Ridge 加日影直接对比 ElasticNet 不加日影",
            "允许任意未登记模型名称",
        ],
        retrain_required=True,
        executor_name="model_sensitivity",
        evaluator_name="model_sensitivity",
        available=True,
        notes=[
            "首版仅支持 Ridge 与 ElasticNet，且每个家族内部必须有 baseline/treatment 配对。",
            "固定相同 alpha 只作为同一家族内部的单独敏感性子类型。",
        ],
    )


def _baseline_redundancy_tool() -> ExperimentToolRecord:
    return ExperimentToolRecord(
        experiment_type="baseline_redundancy",
        version="1.0.0",
        required_data_fields=["target", "M0_base", "M1_focus", "M2_enhanced", "M3_both"],
        allowed_manipulated_factors=["enhanced_baseline_variables"],
        fixed_conditions=[
            "target",
            "common_calendar_sample",
            "calendar_split",
            "focus_feature",
            "model_family",
        ],
        protocol_fields=[
            "experiment_type",
            "model.parameters.baseline_features",
            "model.parameters.enhanced_features",
            "model.parameters.focus_features",
            "runs_spec",
            "contrasts_spec",
        ],
        output_fields=["runs", "contrasts", "gain_change", "bootstrap_ci"],
        applicability="检验增强已有观测基线后，日影的增量信息是否仍保留。",
        refusal_reasons=[
            "缺失 M0/M1/M2/M3 任一配对结构",
            "四组未使用共同训练/评价样本",
            "把增量变化差异直接写成因果路径证明",
        ],
        retrain_required=True,
        executor_name=None,
        evaluator_name=None,
        available=False,
        notes=["阶段B接通：G_basic、G_enhanced、gain_change 使用同一次配对块 bootstrap 重算。"],
    )


def _temporal_sensitivity_tool() -> ExperimentToolRecord:
    return ExperimentToolRecord(
        experiment_type="temporal_sensitivity",
        version="1.0.0",
        required_data_fields=["target", "change_axis", "values", "alignment"],
        allowed_manipulated_factors=["forecast_horizon_days", "past_lag_days"],
        fixed_conditions=[
            "target",
            "common_calendar_sample",
            "calendar_split",
            "feature_sets",
            "model_family",
        ],
        protocol_fields=[
            "experiment_type",
            "model.parameters.change_axis",
            "model.parameters.values",
            "alignment",
            "runs_spec",
            "contrasts_spec",
        ],
        output_fields=["runs", "contrasts", "sample_counts", "date_ranges"],
        applicability="检验结论是否依赖预测提前量或输入历史长度；首版每次只改变一个因素。",
        refusal_reasons=[
            "同时改变提前量和历史长度",
            "LLM 自由扩大网格",
            "未固定 common_issue_date 或 common_target_date",
            "把不同提前量的原始 RMSE 直接当成同一任务比较",
        ],
        retrain_required=True,
        executor_name=None,
        evaluator_name=None,
        available=False,
        notes=["阶段B接通：仅允许在开发数据上选择设置，最终测试结果不得回灌到设置选择。"],
    )


def _conditional_effect_tool() -> ExperimentToolRecord:
    return ExperimentToolRecord(
        experiment_type="conditional_effect",
        version="1.0.0",
        required_data_fields=["target", "condition_field", "threshold_source", "group_gain_difference"],
        allowed_manipulated_factors=["condition_field", "threshold"],
        fixed_conditions=["target", "common_calendar_sample", "model_family", "feature_sets"],
        protocol_fields=[
            "experiment_type",
            "model.parameters.condition_field",
            "model.parameters.threshold_source",
            "diagnostics_spec",
        ],
        output_fields=["groups", "group_gains", "gain_difference", "block_coverage"],
        applicability="检验日影增量是否随事先定义且可解释的条件变化，优先复用已有预测结果。",
        refusal_reasons=[
            "阈值来自测试结果",
            "把用未来真实风速分组写成预报时可用的策略",
            "有效时间块不足仍强行判定",
        ],
        retrain_required=False,
        executor_name=None,
        evaluator_name=None,
        available=False,
        notes=["阶段B接通：G_low / G_high / gain_difference 使用共同块抽样。"],
    )


def _robustness_diagnostic_tool() -> ExperimentToolRecord:
    return ExperimentToolRecord(
        experiment_type="robustness_diagnostic",
        version="1.0.0",
        required_data_fields=["source_experiment_id", "prediction_files"],
        allowed_manipulated_factors=["bootstrap_block_size", "time_segments", "quality_groups"],
        fixed_conditions=["source_prediction_file_version", "sample_alignment", "metric_definitions"],
        protocol_fields=[
            "experiment_type",
            "model.parameters.source_experiment_id",
            "model.parameters.source_prediction_file",
            "diagnostics_spec",
        ],
        output_fields=["diagnostics", "warnings", "validity_status"],
        applicability="复用已有预测与训练产物评估可靠性、时间分段误差、偏差和误差源，不做无意义重训。",
        refusal_reasons=[
            "不指向明确 source_experiment_id 或预测文件版本",
            "用测试段残差删除难预测日期后重新宣称性能",
            "校准器/新模型变更未形成新协议",
        ],
        retrain_required=False,
        executor_name="robustness_diagnostic",
        evaluator_name="robustness_diagnostic",
        available=True,
        notes=[
            "诊断过程不修改预测；若需拟合校准器或改变模型，必须另建新训练协议。",
            "bootstrap 保持日历块结构与双臂配对。",
        ],
    )


EXPERIMENT_TOOL_INDEX: dict[str, ExperimentToolRecord] = {
    item.experiment_type: item
    for item in [
        _incremental_tool(),
        _baseline_redundancy_tool(),
        _model_sensitivity_tool(),
        _temporal_sensitivity_tool(),
        _conditional_effect_tool(),
        _robustness_diagnostic_tool(),
    ]
}

ALL_EXPERIMENT_TYPES = tuple(EXPERIMENT_TOOL_INDEX.keys())


def get_tool(experiment_type: str) -> ExperimentToolRecord | None:
    return EXPERIMENT_TOOL_INDEX.get(experiment_type)


def available_tool_summaries() -> list[dict[str, Any]]:
    return [tool.to_summary() for tool in EXPERIMENT_TOOL_INDEX.values() if tool.available]
