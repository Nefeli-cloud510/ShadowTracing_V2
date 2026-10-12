"""Structural validation for executable experiment protocols.

The goal is not to judge a design as scientifically "good", but to reject
designs whose declared experiment type cannot be expressed by the required
runs / contrasts / diagnostics structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.experiment_tools.registry import EXPERIMENT_TOOL_INDEX, get_tool
from core.unified_schema import (
    ContrastSpec,
    DiagnosticSpec,
    ExperimentProtocol,
    RunSpec,
)


@dataclass
class ProtocolValidationResult:
    valid: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    compiled_runs: list[RunSpec] = field(default_factory=list)
    compiled_contrasts: list[ContrastSpec] = field(default_factory=list)
    compiled_diagnostics: list[DiagnosticSpec] = field(default_factory=list)


def validate_experiment_protocol(
    protocol: ExperimentProtocol,
) -> ProtocolValidationResult:
    """Validate a protocol against the registered tool's structure contract."""

    experiment_type = protocol.experiment_type
    tool = get_tool(experiment_type)
    if tool is None:
        return ProtocolValidationResult(
            valid=False,
            reasons=[
                f"实验类型 {experiment_type!r} 未注册，禁止静默降级为普通增量实验。"
            ],
        )

    validators = {
        "baseline_benchmark": _validate_baseline_benchmark,
        "incremental_prediction": _validate_incremental,
        "model_sensitivity": _validate_model_sensitivity,
        "baseline_redundancy": _validate_baseline_redundancy,
        "temporal_sensitivity": _validate_temporal_sensitivity,
        "conditional_effect": _validate_conditional_effect,
        "robustness_diagnostic": _validate_robustness_diagnostic,
    }
    validator = validators.get(experiment_type)
    if validator is None:
        return ProtocolValidationResult(
            valid=True,
            warnings=[
                f"{experiment_type} 已有结构定义，但执行器尚未接通；调度器将返回 unsupported。"
            ],
        )
    return validator(protocol)


def _validate_baseline_benchmark(protocol: ExperimentProtocol) -> ProtocolValidationResult:
    control, treatment = _base_treatment_sets(protocol)
    reasons: list[str] = []
    if control:
        reasons.append("baseline_benchmark 是单臂基准实验，features.control 必须为空。")
    if not treatment:
        reasons.append("baseline_benchmark 缺少单臂基准特征 features.treatment。")

    family = _default_model_family(protocol)
    if family is None:
        reasons.append("默认模型家族必须是 Ridge 或 ElasticNet。")

    runs: list[RunSpec] = []
    if protocol.runs_spec:
        runs = [RunSpec.model_validate(item.model_dump()) for item in protocol.runs_spec]
        ids = {item.run_id for item in runs}
        if "baseline" not in ids:
            reasons.append("baseline_benchmark 的 runs_spec 必须包含名为 baseline 的 run。")
        for run in runs:
            if run.feature_columns and set(run.feature_columns) != set(treatment):
                reasons.append("baseline_benchmark 的 baseline run 特征列必须与 features.treatment 一致。")
    else:
        runs = [
            RunSpec(
                run_id="baseline",
                label="单臂基准",
                feature_columns=treatment,
                model_family=family or "elasticnet",
                parameters=dict(protocol.model.parameters),
                retrain=True,
            )
        ]

    return ProtocolValidationResult(
        valid=not reasons,
        reasons=reasons,
        warnings=list(protocol.warnings),
        compiled_runs=runs,
        compiled_contrasts=[ContrastSpec.model_validate(item.model_dump()) for item in protocol.contrasts_spec],
        compiled_diagnostics=[DiagnosticSpec.model_validate(item.model_dump()) for item in protocol.diagnostics_spec],
    )


def _normalize_family(name: str) -> str | None:
    normalized = name.strip().lower()
    if normalized == "enet":
        return "elasticnet"
    if normalized in {"ridge", "elasticnet"}:
        return normalized
    return None


def _base_treatment_sets(protocol: ExperimentProtocol) -> tuple[list[str], list[str]]:
    control = [str(item).strip() for item in protocol.features.control if str(item).strip()]
    treatment = [str(item).strip() for item in protocol.features.treatment if str(item).strip()]
    return control, treatment


def _default_model_family(protocol: ExperimentProtocol) -> str:
    family = protocol.model.parameters.get("model_family") or protocol.model.name or "ElasticNet"
    return _normalize_family(str(family)) or "elasticnet"


def _family_tag(family: str) -> str:
    return family


def _validate_incremental(protocol: ExperimentProtocol) -> ProtocolValidationResult:
    control, treatment = _base_treatment_sets(protocol)
    reasons: list[str] = []
    if not control:
        reasons.append("incremental_prediction 缺少对照组基线特征 features.control。")
    if not treatment:
        reasons.append("incremental_prediction 缺少实验组特征 features.treatment。")
    focus = [item for item in treatment if item not in control]
    if not focus:
        reasons.append("实验组未在对照组之外增加任何焦点特征，无法构成增量预测检验。")
    if control and treatment and not set(control).issubset(set(treatment)):
        reasons.append("实验组必须完整包含对照组基线特征，只能在其上增加焦点特征。")

    family = _default_model_family(protocol)
    if family is None:
        reasons.append("默认模型家族必须是 Ridge 或 ElasticNet。")

    runs: list[RunSpec] = []
    if protocol.runs_spec:
        runs = [RunSpec.model_validate(item.model_dump()) for item in protocol.runs_spec]
        run_ids = {item.run_id for item in runs}
        if not {"baseline", "treatment"}.issubset(run_ids):
            reasons.append("incremental_prediction 的 runs_spec 必须包含 baseline 与 treatment 两个 run。")
        for run in runs:
            if run.run_id == "baseline" and run.feature_columns and set(run.feature_columns) != set(control):
                reasons.append("baseline run 的特征列与 features.control 不一致。")
            if run.run_id == "treatment" and run.feature_columns and set(run.feature_columns) != set(treatment):
                reasons.append("treatment run 的特征列与 features.treatment 不一致。")
    else:
        runs = [
            RunSpec(
                run_id="baseline",
                label="对照组",
                feature_columns=control,
                model_family=family,
                parameters=dict(protocol.model.parameters),
                retrain=True,
            ),
            RunSpec(
                run_id="treatment",
                label="实验组",
                feature_columns=treatment,
                model_family=family,
                parameters=dict(protocol.model.parameters),
                retrain=True,
            ),
        ]

    contrasts: list[ContrastSpec] = []
    if protocol.contrasts_spec:
        contrasts = [ContrastSpec.model_validate(item.model_dump()) for item in protocol.contrasts_spec]
        for contrast in contrasts:
            if any(run_id not in {item.run_id for item in runs} for run_id in contrast.run_ids):
                reasons.append(f"contrast {contrast.contrast_id} 引用了 runs_spec 中不存在的 run_id。")
    else:
        contrasts = [
            ContrastSpec(
                contrast_id="baseline_vs_treatment",
                name="对照组 / 实验组主比较",
                kind="metric_delta",
                run_ids=["baseline", "treatment"],
                direction="mixed",
                note="增量预测主比较：误差类指标降低与相关类指标升高均视为增益方向。",
            )
        ]

    return ProtocolValidationResult(
        valid=not reasons,
        reasons=reasons,
        warnings=list(protocol.warnings),
        compiled_runs=runs,
        compiled_contrasts=contrasts,
        compiled_diagnostics=[DiagnosticSpec.model_validate(item.model_dump()) for item in protocol.diagnostics_spec],
    )


def _validate_model_sensitivity(protocol: ExperimentProtocol) -> ProtocolValidationResult:
    control, treatment = _base_treatment_sets(protocol)
    reasons: list[str] = []
    focus = [item for item in treatment if item not in control]
    if not control or not treatment:
        reasons.append("model_sensitivity 必须同时声明对照组与实验组特征。")
    if not focus:
        reasons.append("model_sensitivity 的实验组未在对照组之外增加焦点特征。")
    if control and treatment and not set(control).issubset(set(treatment)):
        reasons.append("每个模型家族内的 treatment 必须完整包含 baseline 特征。")

    raw_families = protocol.model.parameters.get("model_families")
    families: list[str] = []
    if not isinstance(raw_families, list) or not raw_families:
        reasons.append("model_sensitivity 必须通过 model.parameters.model_families 显式声明模型家族。")
    else:
        for value in raw_families:
            normalized = _normalize_family(str(value))
            if normalized is None:
                reasons.append(f"不支持的模型家族 {value!r}，首版仅允许 Ridge / ElasticNet。")
            elif normalized not in families:
                families.append(normalized)
        if len(families) < 2:
            reasons.append("model_sensitivity 至少需要两个已登记模型家族。")

    warnings = list(protocol.warnings)
    if isinstance(protocol.model.parameters.get("fixed_alpha"), (int, float)):
        warnings.append("固定相同 alpha 的比较只作为同一模型家族内部的单独敏感性子类型。")

    runs: list[RunSpec] = []
    if protocol.runs_spec:
        runs = [RunSpec.model_validate(item.model_dump()) for item in protocol.runs_spec]
        run_ids = {item.run_id for item in runs}
        for family in families:
            if f"{family}_baseline" not in run_ids or f"{family}_treatment" not in run_ids:
                reasons.append(f"模型家族 {family} 缺少 baseline/treatment 双臂 run。")
    else:
        for family in families:
            runs.append(
                RunSpec(
                    run_id=f"{family}_baseline",
                    label=f"{family} 对照组",
                    feature_columns=control,
                    model_family=family,
                    parameters=dict(protocol.model.parameters),
                    retrain=True,
                )
            )
            runs.append(
                RunSpec(
                    run_id=f"{family}_treatment",
                    label=f"{family} 实验组",
                    feature_columns=treatment,
                    model_family=family,
                    parameters=dict(protocol.model.parameters),
                    retrain=True,
                )
            )

    contrasts: list[ContrastSpec] = []
    if protocol.contrasts_spec:
        contrasts = [ContrastSpec.model_validate(item.model_dump()) for item in protocol.contrasts_spec]
    else:
        for family in families:
            contrasts.append(
                ContrastSpec(
                    contrast_id=f"gain_{family}",
                    name=f"{family} 家族内日影增量",
                    kind="incremental_gain",
                    run_ids=[f"{family}_baseline", f"{family}_treatment"],
                    direction="mixed",
                    note="只比较同一模型家族内部加日影前后；禁止跨家族错误归因。",
                )
            )
    return ProtocolValidationResult(
        valid=not reasons,
        reasons=reasons,
        warnings=warnings,
        compiled_runs=runs,
        compiled_contrasts=contrasts,
        compiled_diagnostics=[DiagnosticSpec.model_validate(item.model_dump()) for item in protocol.diagnostics_spec],
    )


def _validate_baseline_redundancy(protocol: ExperimentProtocol) -> ProtocolValidationResult:
    params = protocol.model.parameters
    base = [str(item).strip() for item in params.get("baseline_features", []) if str(item).strip()]
    enhanced = [str(item).strip() for item in params.get("enhanced_features", []) if str(item).strip()]
    focus = [str(item).strip() for item in params.get("focus_features", []) if str(item).strip()]
    reasons: list[str] = []
    if not base:
        reasons.append("baseline_redundancy 缺少 model.parameters.baseline_features（M0 基础基线）。")
    if not enhanced:
        reasons.append("baseline_redundancy 缺少 model.parameters.enhanced_features（M2 增强基线变量）。")
    if not focus:
        reasons.append("baseline_redundancy 缺少 model.parameters.focus_features（待检验焦点特征）。")
    if focus and enhanced and set(focus).issubset(set(enhanced)):
        reasons.append("焦点特征已包含在增强基线中，M3 与 M2 等价，无法检验增强后的增量。")

    runs: list[RunSpec] = []
    if protocol.runs_spec:
        runs = [RunSpec.model_validate(item.model_dump()) for item in protocol.runs_spec]
        ids = {item.run_id.lower() for item in runs}
        for required in ("m0", "m1", "m2", "m3"):
            if required not in ids:
                reasons.append(f"baseline_redundancy 必须包含 {required.upper()} 运行。")
        model_families = {
            item.run_id.split("_")[0]
            for item in runs
            if item.run_id.count("_") == 1 and item.run_id.endswith("_baseline")
        }
        if len(model_families) > 1:
            reasons.append("baseline_redundancy 必须冻结单一模型家族，不允许跨模型比较增量。")
    else:
        family = _default_model_family(protocol) or "elasticnet"
        definitions = {
            "m0": {"label": "M0 基础基线", "features": base},
            "m1": {"label": "M1 基础基线＋焦点", "features": base + focus},
            "m2": {"label": "M2 增强基线", "features": base + enhanced},
            "m3": {"label": "M3 增强基线＋焦点", "features": base + enhanced + focus},
        }
        for key, definition in definitions.items():
            runs.append(
                RunSpec(
                    run_id=key,
                    label=definition["label"],
                    feature_columns=definition["features"],
                    model_family=family,
                    parameters=dict(params),
                    retrain=True,
                )
            )

    contrasts: list[ContrastSpec] = []
    if protocol.contrasts_spec:
        contrasts = [ContrastSpec.model_validate(item.model_dump()) for item in protocol.contrasts_spec]
    else:
        contrasts = [
            ContrastSpec(
                contrast_id="gain_basic",
                name="G_basic = RMSE(M0) - RMSE(M1)",
                kind="incremental_gain",
                run_ids=["m0", "m1"],
                direction="reduce_error",
            ),
            ContrastSpec(
                contrast_id="gain_enhanced",
                name="G_enhanced = RMSE(M2) - RMSE(M3)",
                kind="incremental_gain",
                run_ids=["m2", "m3"],
                direction="reduce_error",
            ),
            ContrastSpec(
                contrast_id="gain_change",
                name="增益变化 = G_enhanced - G_basic",
                kind="gain_change",
                run_ids=[],
                contrast_ids=["gain_basic", "gain_enhanced"],
                direction="neutral",
                note="增益差异必须由同一次配对块 bootstrap 重算，不能仅凭两组显著性差宣布。",
            ),
        ]

    return ProtocolValidationResult(
        valid=not reasons,
        reasons=reasons,
        warnings=list(protocol.warnings),
        compiled_runs=runs,
        compiled_contrasts=contrasts,
        compiled_diagnostics=[DiagnosticSpec.model_validate(item.model_dump()) for item in protocol.diagnostics_spec],
    )


def _validate_temporal_sensitivity(protocol: ExperimentProtocol) -> ProtocolValidationResult:
    params = protocol.model.parameters
    control, treatment = _base_treatment_sets(protocol)
    change_axis = params.get("change_axis")
    values = params.get("values", [])
    reasons: list[str] = []
    if change_axis not in {"forecast_horizon_days", "past_lag_days"}:
        reasons.append("temporal_sensitivity 必须声明 change_axis 为 forecast_horizon_days 或 past_lag_days。")
    if not isinstance(values, list) or not values:
        reasons.append("temporal_sensitivity 必须给出有限且非空的 values 列表。")
    elif any(not isinstance(value, (int, float)) or not float(value).is_integer() or float(value) <= 0 for value in values):
        reasons.append("temporal_sensitivity 的 values 必须是有限正整数。")
    if len(values) != len(set(values)):
        reasons.append("temporal_sensitivity 的 values 存在重复。")
    if not control or not treatment:
        reasons.append("temporal_sensitivity 必须声明对照组/实验组特征。")

    alignment = protocol.alignment
    if (
        alignment is None
        or (not alignment.common_issue_date and not alignment.common_target_date)
        or (alignment.common_issue_date and alignment.common_target_date)
    ):
        reasons.append("temporal_sensitivity 必须在 alignment 中固定 common_issue_date 或 common_target_date 之一。")

    warnings = list(protocol.warnings)
    warnings.append(
        "不同提前量属于不同预测任务，不能按原始 RMSE 高低直接宣称某一方案更优；"
        "仅允许在开发数据上根据结果选择设置。"
    )

    runs: list[RunSpec] = []
    if protocol.runs_spec:
        runs = [RunSpec.model_validate(item.model_dump()) for item in protocol.runs_spec]
    else:
        family = _default_model_family(protocol) or "elasticnet"
        for value in values:
            tag = f"h{int(value)}" if change_axis == "forecast_horizon_days" else f"w{int(value)}"
            merged = dict(params)
            merged[change_axis] = int(value)
            runs.append(
                RunSpec(
                    run_id=f"{tag}_baseline",
                    label=f"{change_axis}={value} 对照组",
                    feature_columns=control,
                    model_family=family,
                    parameters=merged,
                    retrain=True,
                )
            )
            runs.append(
                RunSpec(
                    run_id=f"{tag}_treatment",
                    label=f"{change_axis}={value} 实验组",
                    feature_columns=treatment,
                    model_family=family,
                    parameters=merged,
                    retrain=True,
                )
            )

    contrasts: list[ContrastSpec] = []
    if protocol.contrasts_spec:
        contrasts = [ContrastSpec.model_validate(item.model_dump()) for item in protocol.contrasts_spec]
    else:
        for value in values:
            tag = f"h{int(value)}" if change_axis == "forecast_horizon_days" else f"w{int(value)}"
            contrasts.append(
                ContrastSpec(
                    contrast_id=f"gain_{tag}",
                    name=f"{change_axis}={value} 的日影增量",
                    kind="incremental_gain",
                    run_ids=[f"{tag}_baseline", f"{tag}_treatment"],
                    direction="mixed",
                )
            )

    return ProtocolValidationResult(
        valid=not reasons,
        reasons=reasons,
        warnings=warnings,
        compiled_runs=runs,
        compiled_contrasts=contrasts,
        compiled_diagnostics=[DiagnosticSpec.model_validate(item.model_dump()) for item in protocol.diagnostics_spec],
    )


def _validate_conditional_effect(protocol: ExperimentProtocol) -> ProtocolValidationResult:
    params = protocol.model.parameters
    condition_field = params.get("condition_field")
    threshold_source = params.get("threshold_source")
    reasons: list[str] = []
    warnings = list(protocol.warnings)
    if not condition_field:
        reasons.append("conditional_effect 必须声明 condition_field。")
    if threshold_source not in {"training_data", "issue_time_known", "future_post_hoc"}:
        reasons.append("conditional_effect 必须声明 threshold_source 为 training_data / issue_time_known / future_post_hoc。")
    elif threshold_source == "future_post_hoc":
        warnings.append("未来真实风速等条件只用于事后误差诊断，不得写成预报时可选择的策略。")
    if not protocol.contrasts_spec and "group_gain_difference" not in {
        item.kind for item in protocol.contrasts_spec
    }:
        reasons.append("conditional_effect 必须定义 group_gain_difference 对比。")

    return ProtocolValidationResult(
        valid=not reasons,
        reasons=reasons,
        warnings=warnings,
        compiled_runs=[RunSpec.model_validate(item.model_dump()) for item in protocol.runs_spec],
        compiled_contrasts=[ContrastSpec.model_validate(item.model_dump()) for item in protocol.contrasts_spec],
        compiled_diagnostics=[DiagnosticSpec.model_validate(item.model_dump()) for item in protocol.diagnostics_spec],
    )


def _validate_robustness_diagnostic(protocol: ExperimentProtocol) -> ProtocolValidationResult:
    params = protocol.model.parameters
    reasons: list[str] = []
    source_experiment_id = params.get("source_experiment_id")
    prediction_file = params.get("source_prediction_file")
    has_reused_runs = any(
        item.retrain is False and item.source_prediction_file is not None
        for item in protocol.runs_spec
    )
    if not source_experiment_id and not prediction_file and not has_reused_runs:
        reasons.append(
            "robustness_diagnostic 必须指向明确的 source_experiment_id 或复用的预测文件，禁止无来源诊断。"
        )
    diagnostics = [DiagnosticSpec.model_validate(item.model_dump()) for item in protocol.diagnostics_spec]
    if not diagnostics and not protocol.notes:
        reasons.append("robustness_diagnostic 至少需要一个 diagnostics_spec 条目。")

    return ProtocolValidationResult(
        valid=not reasons,
        reasons=reasons,
        warnings=list(protocol.warnings),
        compiled_runs=[RunSpec.model_validate(item.model_dump()) for item in protocol.runs_spec],
        compiled_contrasts=[ContrastSpec.model_validate(item.model_dump()) for item in protocol.contrasts_spec],
        compiled_diagnostics=diagnostics,
    )


def default_experiment_type() -> str:
    return "incremental_prediction"
