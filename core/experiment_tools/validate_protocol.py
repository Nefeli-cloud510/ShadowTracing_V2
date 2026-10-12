"""Structural validation for executable experiment protocols.

The goal is not to judge a design as scientifically "good", but to reject
designs whose declared experiment type cannot be expressed by the required
runs / contrasts / diagnostics structure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.experiment_tools.compiled_plan import (
    CompiledExperimentPlan,
    normalize_diagnostic_id,
)
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
    compiled_plan: CompiledExperimentPlan | None = None


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
    return _finalize_protocol_validation(protocol, validator(protocol))


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
    times = _validate_shared_time_parameters(protocol, reasons)
    _validate_no_diagnostics(protocol, reasons)
    if protocol.contrasts_spec:
        reasons.append("baseline_benchmark 是单臂基准实验，不允许定义跨 run 对比。")

    runs: list[RunSpec] = []
    if protocol.runs_spec:
        runs = [RunSpec.model_validate(item.model_dump()) for item in protocol.runs_spec]
        ids = [item.run_id for item in runs]
        if len(ids) != len(set(ids)):
            reasons.append("baseline_benchmark 的 runs_spec 存在重复 run_id。")
        if ids != ["baseline"]:
            reasons.append("baseline_benchmark 的 runs_spec 只能包含名为 baseline 的一个 run。")
        if "baseline" not in ids:
            reasons.append("baseline_benchmark 的 runs_spec 必须包含名为 baseline 的 run。")
        for run in runs:
            if run.feature_columns and set(run.feature_columns) != set(treatment):
                reasons.append("baseline_benchmark 的 baseline run 特征列必须与 features.treatment 一致。")
            if _normalize_family(run.model_family) != family:
                reasons.append(
                    f"baseline_benchmark 的 run 必须使用协议默认模型家族 {family}，"
                    f"收到 {run.model_family!r}。"
                )
            _validate_retrain_required(run, reasons)
            _validate_run_time_parameters(run, times, reasons)
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

    _validate_feature_groups_compatible(
        protocol,
        expected_baseline=treatment,
        expected_treatment=treatment,
        expected_mode="baseline_single_arm",
        reasons=reasons,
    )
    _validate_arm_time_windows_compatible(protocol, times, reasons)

    return ProtocolValidationResult(
        valid=not reasons,
        reasons=reasons,
        warnings=list(protocol.warnings),
        compiled_runs=runs,
        compiled_contrasts=[],
        compiled_diagnostics=[],
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


def _finalize_protocol_validation(
    protocol: ExperimentProtocol,
    result: ProtocolValidationResult,
) -> ProtocolValidationResult:
    """Attach the single authoritative plan to any structurally valid protocol."""

    if not result.valid:
        return result
    result.compiled_plan = CompiledExperimentPlan(
        experiment_type=protocol.experiment_type,
        target=protocol.target,
        experiment_id=protocol.experiment_id,
        round_id=protocol.round_id,
        model_parameters=dict(protocol.model.parameters),
        time_parameters={
            key: value
            for key, value in _time_parameters(protocol).items()
            if value is not None
        },
        runs=[
            RunSpec.model_validate(item.model_dump())
            for item in result.compiled_runs
        ],
        contrasts=[
            ContrastSpec.model_validate(item.model_dump())
            for item in result.compiled_contrasts
        ],
        diagnostics=[
            DiagnosticSpec.model_validate(item.model_dump())
            for item in result.compiled_diagnostics
        ],
        alignment=(
            protocol.alignment.model_copy(deep=True)
            if protocol.alignment is not None
            else None
        ),
    )
    return result


TIME_PARAMETER_KEYS = (
    "forecast_horizon_days",
    "past_lag_days",
    "window_size",
    "test_split_ratio",
)
TRAINING_CONTRAST_KINDS = {"metric_delta", "incremental_gain"}


def _time_parameters(protocol: ExperimentProtocol) -> dict[str, object]:
    params = protocol.model.parameters
    return {key: params.get(key) for key in TIME_PARAMETER_KEYS}


def _validate_shared_time_parameters(
    protocol: ExperimentProtocol,
    reasons: list[str],
) -> dict[str, object]:
    times = _time_parameters(protocol)
    for key in TIME_PARAMETER_KEYS:
        value = times.get(key)
        if value is None:
            reasons.append(
                f"普通训练实验必须显式固定 {key}，禁止执行器读取旧默认值。"
            )
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            reasons.append(f"{key} 必须是有限正数，收到 {value!r}。")
            continue
        if not math.isfinite(numeric) or numeric <= 0:
            reasons.append(f"{key} 必须是有限正数，收到 {value!r}。")
    return times


def _validate_run_time_parameters(
    run: RunSpec,
    times: dict[str, object],
    reasons: list[str],
) -> None:
    for key, canonical_value in times.items():
        if canonical_value is None:
            continue
        override = run.parameters.get(key)
        if override is None:
            continue
        try:
            same = float(override) == float(canonical_value)
        except (TypeError, ValueError):
            same = False
        if not same:
            reasons.append(
                f"run {run.run_id} 试图把共享时间参数 {key} 改为 {override}，"
                f"与编译计划固定值 {canonical_value} 冲突。"
            )


def _validate_retrain_required(run: RunSpec, reasons: list[str]) -> None:
    if not run.retrain:
        reasons.append(
            f"run {run.run_id} 禁止 retrain=False：普通训练实验必须重新训练并产出新预测。"
        )


def _validate_no_diagnostics(
    protocol: ExperimentProtocol,
    reasons: list[str],
) -> None:
    if protocol.diagnostics_spec:
        reasons.append(
            "普通训练实验禁止携带 diagnostics_spec；只有 robustness_diagnostic 允许声明诊断。"
        )


def _validate_contrast_references(
    contrasts: list[ContrastSpec],
    run_ids: set[str],
    reasons: list[str],
) -> None:
    seen: set[str] = set()
    contrast_ids = {item.contrast_id for item in contrasts}
    for contrast in contrasts:
        if contrast.contrast_id in seen:
            reasons.append(f"contrast_id {contrast.contrast_id!r} 重复。")
        seen.add(contrast.contrast_id)
        if contrast.kind not in TRAINING_CONTRAST_KINDS:
            reasons.append(
                f"contrast {contrast.contrast_id} 使用不支持的对比类型 {contrast.kind!r}，"
                "已接通训练实验只支持 metric_delta / incremental_gain。"
            )
        if contrast.contrast_ids:
            reasons.append(
                f"contrast {contrast.contrast_id} 不允许引用其他 contrast（嵌套对比未接通）。"
            )
        for run_id in contrast.run_ids:
            if run_id not in run_ids:
                reasons.append(
                    f"contrast {contrast.contrast_id} 引用了不存在的 run_id {run_id!r}。"
                )
        for target in contrast.contrast_ids:
            if target not in contrast_ids:
                reasons.append(
                    f"contrast {contrast.contrast_id} 引用了不存在的 contrast_id {target!r}。"
                )


def _family_arm(run_id: str) -> tuple[str, str] | None:
    parts = run_id.rsplit("_", 1)
    if len(parts) != 2:
        return None
    family, arm = parts
    normalized = _normalize_family(family)
    if normalized is None or arm not in {"baseline", "treatment"}:
        return None
    return normalized, arm


def _model_sensitivity_arm(
    run_id: str,
    families: list[str],
) -> tuple[str, str] | None:
    """Parse a run id against the declared model sensitivity families."""

    parts = run_id.rsplit("_", 1)
    if len(parts) != 2:
        return None
    family, arm = parts
    normalized = _normalize_family(family)
    if normalized not in families or arm not in {"baseline", "treatment"}:
        return None
    return normalized, arm


def _validate_feature_groups_compatible(
    protocol: ExperimentProtocol,
    *,
    expected_baseline: list[str],
    expected_treatment: list[str],
    expected_mode: str,
    reasons: list[str],
) -> None:
    groups = protocol.model.parameters.get("feature_groups")
    if not isinstance(groups, dict):
        return
    declared_mode = groups.get("mode")
    if declared_mode not in (None, expected_mode):
        reasons.append(
            f"feature_groups.mode={declared_mode!r} 与实验类型 {protocol.experiment_type} "
            f"要求的 {expected_mode!r} 冲突。"
        )
    for label, expected in (
        ("baseline", expected_baseline),
        ("treatment", expected_treatment),
    ):
        configured = groups.get(label)
        if isinstance(configured, list):
            normalized = [
                str(item).strip() for item in configured if str(item).strip()
            ]
            if normalized and set(normalized) != set(expected):
                reasons.append(
                    f"feature_groups.{label} 与 features.{label} 冲突："
                    f"配置 {normalized}，协议声明 {expected}。"
                )


def _validate_arm_time_windows_compatible(
    protocol: ExperimentProtocol,
    times: dict[str, object],
    reasons: list[str],
) -> None:
    arm_windows = protocol.model.parameters.get("arm_time_windows")
    if not isinstance(arm_windows, dict):
        return
    resolved: dict[str, dict[str, object]] = {}
    for label in ("baseline", "treatment"):
        override = arm_windows.get(label)
        if not isinstance(override, dict):
            continue
        for key, canonical_value in times.items():
            if canonical_value is None:
                continue
            override_value = override.get(key)
            if override_value is None:
                continue
            try:
                same = float(override_value) == float(canonical_value)
            except (TypeError, ValueError):
                same = False
            if not same:
                reasons.append(
                    f"arm_time_windows.{label}.{key} 与最终时间参数冲突："
                    f"配置 {override_value}，协议声明 {canonical_value}。"
                )
        resolved[label] = override
    if (
        "baseline" in resolved
        and "treatment" in resolved
        and resolved["baseline"] != resolved["treatment"]
    ):
        reasons.append(
            "arm_time_windows 中 baseline/treatment 必须一致，"
            "双臂必须在同一日历样本上比较。"
        )


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
    times = _validate_shared_time_parameters(protocol, reasons)
    _validate_no_diagnostics(protocol, reasons)

    runs: list[RunSpec] = []
    if protocol.runs_spec:
        runs = [RunSpec.model_validate(item.model_dump()) for item in protocol.runs_spec]
        run_ids = [item.run_id for item in runs]
        if len(run_ids) != len(set(run_ids)):
            reasons.append("incremental_prediction 的 runs_spec 存在重复 run_id。")
        if len(run_ids) != 2 or set(run_ids) != {"baseline", "treatment"}:
            reasons.append(
                "incremental_prediction 的 runs_spec 必须恰好包含 baseline 与 treatment 两个 run。"
            )
        for run in runs:
            expected_features = {
                "baseline": control,
                "treatment": treatment,
            }.get(run.run_id)
            if expected_features is not None and run.feature_columns and set(run.feature_columns) != set(expected_features):
                reasons.append(
                    f"{run.run_id} run 的特征列与 features.{run.run_id} 不一致："
                    f"收到 {run.feature_columns}，应为 {expected_features}。"
                )
            if _normalize_family(run.model_family) != family:
                reasons.append(
                    f"run {run.run_id} 必须使用协议默认模型家族 {family}，"
                    f"收到 {run.model_family!r}。"
                )
            _validate_retrain_required(run, reasons)
            _validate_run_time_parameters(run, times, reasons)
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
        _validate_contrast_references(
            contrasts,
            {item.run_id for item in runs},
            reasons,
        )
        for contrast in contrasts:
            if len(contrast.run_ids) != 2 or set(contrast.run_ids) != {"baseline", "treatment"}:
                reasons.append(
                    f"contrast {contrast.contrast_id} 必须精确引用 baseline 与 treatment 两个 run。"
                )
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

    _validate_feature_groups_compatible(
        protocol,
        expected_baseline=control,
        expected_treatment=treatment,
        expected_mode="comparative",
        reasons=reasons,
    )
    _validate_arm_time_windows_compatible(protocol, times, reasons)

    return ProtocolValidationResult(
        valid=not reasons,
        reasons=reasons,
        warnings=list(protocol.warnings),
        compiled_runs=runs,
        compiled_contrasts=contrasts,
        compiled_diagnostics=[],
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

    times = _validate_shared_time_parameters(protocol, reasons)
    _validate_no_diagnostics(protocol, reasons)
    warnings = list(protocol.warnings)
    if isinstance(protocol.model.parameters.get("fixed_alpha"), (int, float)):
        warnings.append("固定相同 alpha 的比较只作为同一模型家族内部的单独敏感性子类型。")

    runs: list[RunSpec] = []
    if protocol.runs_spec:
        runs = [RunSpec.model_validate(item.model_dump()) for item in protocol.runs_spec]
        run_ids = [item.run_id for item in runs]
        if len(run_ids) != len(set(run_ids)):
            reasons.append("model_sensitivity 的 runs_spec 存在重复 run_id。")
        for run in runs:
            parsed = _model_sensitivity_arm(run.run_id, families)
            if parsed is None:
                reasons.append(
                    f"run_id {run.run_id!r} 必须符合 {{family}}_baseline / {{family}}_treatment 命名。"
                )
            else:
                family, arm = parsed
                run_family = _normalize_family(run.model_family)
                if run_family != family:
                    reasons.append(
                        f"run {run.run_id} 声明家族 {family}，但 model_family={run.model_family!r} "
                        f"实际归一化为 {run_family}，模型家族与 run 命名必须一致。"
                    )
                expected_features = control if arm == "baseline" else treatment
                if run.feature_columns and set(run.feature_columns) != set(expected_features):
                    reasons.append(
                        f"run {run.run_id} 的特征列与 features.{arm} 不一致："
                        f"收到 {run.feature_columns}，应为 {expected_features}。"
                    )
            _validate_retrain_required(run, reasons)
            _validate_run_time_parameters(run, times, reasons)
        present_ids = set(run_ids)
        for family in families:
            for arm in ("baseline", "treatment"):
                expected_id = f"{family}_{arm}"
                if expected_id not in present_ids:
                    reasons.append(f"模型家族 {family} 缺少 {expected_id} run。")
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
        _validate_contrast_references(
            contrasts,
            {item.run_id for item in runs},
            reasons,
        )
        for contrast in contrasts:
            if len(contrast.run_ids) != 2:
                reasons.append(
                    f"contrast {contrast.contrast_id} 必须精确引用同一模型家族的两个 run。"
                )
                continue
            left = _model_sensitivity_arm(contrast.run_ids[0], families)
            right = _model_sensitivity_arm(contrast.run_ids[1], families)
            if left is None or right is None:
                reasons.append(
                    f"contrast {contrast.contrast_id} 引用了不符合模型敏感性命名的 run。"
                )
                continue
            if left[0] != right[0]:
                reasons.append(
                    f"contrast {contrast.contrast_id} 跨模型家族比较 "
                    f"({contrast.run_ids[0]} vs {contrast.run_ids[1]})，"
                    "首版禁止把家族间性能差异归因为日影增量。"
                )
            if {left[1], right[1]} != {"baseline", "treatment"}:
                reasons.append(
                    f"contrast {contrast.contrast_id} 必须比较同一家族内 "
                    "baseline 与 treatment，不允许 arm 内自比较。"
                )
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
    _validate_feature_groups_compatible(
        protocol,
        expected_baseline=control,
        expected_treatment=treatment,
        expected_mode="comparative",
        reasons=reasons,
    )
    _validate_arm_time_windows_compatible(protocol, times, reasons)

    return ProtocolValidationResult(
        valid=not reasons,
        reasons=reasons,
        warnings=warnings,
        compiled_runs=runs,
        compiled_contrasts=contrasts,
        compiled_diagnostics=[],
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
    if (
        not protocol.contrasts_spec
        or "group_gain_difference"
        not in {item.kind for item in protocol.contrasts_spec}
    ):
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
    supported_diagnostics = {
        "paired_calendar_bootstrap",
        "time_segments",
        "residual_diagnostics",
    }
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
    if not diagnostics:
        reasons.append("robustness_diagnostic 至少需要一个 diagnostics_spec 条目。")
        return ProtocolValidationResult(
            valid=not reasons,
            reasons=reasons,
            warnings=list(protocol.warnings),
            compiled_runs=[RunSpec.model_validate(item.model_dump()) for item in protocol.runs_spec],
            compiled_contrasts=[ContrastSpec.model_validate(item.model_dump()) for item in protocol.contrasts_spec],
            compiled_diagnostics=[],
        )
    for diagnostic in diagnostics:
        if normalize_diagnostic_id(diagnostic.diagnostic_id) not in supported_diagnostics:
            reasons.append(
                f"robustness_diagnostic 不支持诊断 {diagnostic.diagnostic_id!r}，"
                "只允许 paired_calendar_bootstrap / time_segments / residual_diagnostics。"
            )
    for run in protocol.runs_spec:
        if run.retrain:
            reasons.append(
                f"robustness_diagnostic 的 run {run.run_id} 必须 retrain=False，"
                "只能复用已有预测，禁止重新训练。"
            )

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
