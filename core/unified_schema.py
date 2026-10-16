from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MetricName = Literal["RMSE", "MAE", "Pearson_r", "Skill", "R2"]
MAX_ACTIVE_UNCERTAINTIES = 10
HypothesisStatus = Literal[
    "draft",
    "pending",
    "active",
    "observing",
    "converged",
    "supported",
    "partially_supported",
    "weakened",
    "pruned",
]
EvidenceType = Literal["literature", "observational_data", "experimental_result", "physical_prior", "expert_judgment"]
DecisionType = Literal[
    "task_approved",
    "task_revised",
    "hypothesis_generated",
    "supplemental_hypothesis_added",
    "hypothesis_tree_confirmed",
    "experiment_selection_requested",
    "experiment_approved",
    "experiment_rejected",
    "experiment_modified",
    "execution_failed",
    "protocol_generated",
    "stop_requested",
    "pause_requested",
    "round_review_requested",
    "scientific_interpreter_applied",
    "scientific_questioning_completed",
    "scientific_questioning_undone",
    "planner_input_prepared",
    "planner_output_generated",
    "planner_output_applied",
    "round_completed",
    "continue_next_round",
    "round_adjusted",
    "terminate_workflow",
]
UncertaintyStatus = Literal["identified", "active", "experimenting", "resolved", "deprecated"]
PriorityLevel = Literal["low", "medium", "high"]
ExpectedEffect = Literal["positive", "negative", "near_zero", "nonlinear", "unknown"]
ExecutionStatus = Literal["queued", "running", "completed", "failed", "aborted"]
StepStatus = Literal["pending", "in_progress", "completed", "failed", "skipped"]
ExperimentType = Literal[
    "incremental_prediction",
    "baseline_benchmark",
    "baseline_redundancy",
    "model_sensitivity",
    "temporal_sensitivity",
    "conditional_effect",
    "robustness_diagnostic",
    "unsupported",
]
ExperimentLifecycleStatus = Literal[
    "proposed",
    "validated",
    "approved",
    "running",
    "completed",
    "failed",
    "inconclusive",
]


class ShadowBaseModel(BaseModel):
    """Base configuration shared by all unified schemas."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    def to_pretty_json(self) -> str:
        return self.model_dump_json(indent=2, exclude_none=True)


class VariableBinding(ShadowBaseModel):
    x: str = Field(description="Primary explanatory variable")
    y: str = Field(description="Target variable")
    m_candidates: list[str] = Field(default_factory=list, description="Candidate mediator variables")


class ResearchQuestion(ShadowBaseModel):
    text: str = Field(min_length=1, description="Natural-language scientific question")
    target: str = Field(min_length=1, description="Primary optimization target")
    question_type: str = Field(min_length=1, description="Question type or task category")
    variables: VariableBinding | None = Field(default=None, description="Structured variable bindings")
    keywords: list[str] = Field(default_factory=list)
    clarified: bool = False


class EvaluationSpec(ShadowBaseModel):
    primary_metric: MetricName = Field(default="RMSE")
    secondary_metrics: list[MetricName] = Field(default_factory=list)
    visual_analysis: list[str] = Field(default_factory=list)

    @field_validator("secondary_metrics")
    @classmethod
    def unique_secondary_metrics(cls, value: list[MetricName]) -> list[MetricName]:
        return list(dict.fromkeys(value))


class ScientificConstraints(ShadowBaseModel):
    no_future_information: bool = True
    validation_feedback_allowed: bool = True
    final_test_blind: bool = True
    max_hypotheses_per_level: int | None = Field(default=None, ge=1)
    min_active_hypotheses: int | None = Field(default=None, ge=1)
    max_rounds: int | None = Field(default=None, ge=1)
    max_experiments_per_round: int | None = Field(default=None, ge=1)
    resource_budget: "ResourceBudget | None" = None
    notes: list[str] = Field(default_factory=list)


class ScientificTaskPayload(ShadowBaseModel):
    research_question: ResearchQuestion
    evaluation: EvaluationSpec
    constraints: ScientificConstraints
    data_sources: dict[str, Any] = Field(default_factory=dict)


class ScientificTask(ShadowBaseModel):
    schema_version: str = "1.0"
    message_type: Literal["scientific_task"] = "scientific_task"
    task_id: str = Field(min_length=1)
    producer: Literal["human_pi"] = "human_pi"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    payload: ScientificTaskPayload


class FieldDescriptor(ShadowBaseModel):
    field_name: str = Field(min_length=1)
    data_type: str = Field(min_length=1)
    physical_meaning: str = Field(min_length=1)
    physical_name: str | None = None
    display_name: str | None = None
    unit: str | None = None
    format: str | None = None
    range: list[float | int] | None = None
    missing_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    is_target: bool = False
    user_notes: str | None = None


class DataDictionary(ShadowBaseModel):
    dictionary_id: str = Field(min_length=1)
    generated_at: datetime
    version: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    total_samples: int = Field(ge=0)
    time_column: str = Field(min_length=1)
    time_format: str = Field(min_length=1)
    time_range: dict[str, str] | None = None
    fields: list[FieldDescriptor] = Field(default_factory=list)
    target_candidates: list[str] = Field(default_factory=list)
    feature_candidates: list[str] = Field(default_factory=list)
    user_supplementary: dict[str, str] = Field(default_factory=dict)

    @field_validator("fields")
    @classmethod
    def unique_field_names(cls, value: list[FieldDescriptor]) -> list[FieldDescriptor]:
        names = [item.field_name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("DataDictionary.fields contains duplicate field names")
        return value

    def get_available_fields(self) -> list[str]:
        return [item.field_name for item in self.fields]


class EvidenceItem(ShadowBaseModel):
    id: str = Field(min_length=1)
    type: EvidenceType
    description: str = Field(min_length=1)
    source: str = Field(min_length=1)
    added_at_round: int = Field(ge=0)
    support_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    experiment_id: str | None = None
    run_id: str | None = None
    contrast_id: str | None = None
    evidence_version: str | None = None
    evidence_key: str | None = None


class CritiqueRecord(ShadowBaseModel):
    id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    from_role: str = Field(min_length=1)
    added_at_round: int = Field(ge=0)
    resolved: bool = False
    experiment_id: str | None = None
    run_id: str | None = None
    contrast_id: str | None = None
    evidence_version: str | None = None
    evidence_key: str | None = None


class PredictionRecord(ShadowBaseModel):
    experiment_id: str = Field(min_length=1)
    metric: MetricName
    expected_direction: ExpectedEffect | str
    expected_range: list[float] | None = None
    actual_value: float | None = None
    matched: bool | Literal["partial"] | None = None

    @field_validator("expected_range")
    @classmethod
    def validate_expected_range(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != 2:
            raise ValueError("expected_range must contain exactly two values")
        return value


class SupportHistoryEntry(ShadowBaseModel):
    round: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    support_before: float | None = Field(default=None, ge=0.0, le=1.0)
    support_delta: float | None = None
    event: str = Field(min_length=1)
    evidence_source: Literal["advisory", "experimental", "human"] | None = None
    experiment_id: str | None = None
    run_id: str | None = None
    contrast_id: str | None = None
    evidence_version: str | None = None
    evidence_key: str | None = None
    evidence_identity: str | None = None
    evidence_classification: str | None = None
    direction_matched: bool | str | None = None


class HypothesisSourceSignal(ShadowBaseModel):
    signal_type: str = Field(min_length=1)
    source_id: str | None = None
    source_round: int | None = Field(default=None, ge=0)
    excerpt: str = Field(min_length=1)
    weight: float = Field(default=0.5, ge=0.0, le=1.0)


class HypothesisGenerationRationale(ShadowBaseModel):
    trigger: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    linked_uncertainties: list[str] = Field(default_factory=list)
    derived_features: list[str] = Field(default_factory=list)
    source_signals: list[HypothesisSourceSignal] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class HypothesisQuestioningRecord(ShadowBaseModel):
    round: int = Field(ge=0)
    impact_direction: str = Field(min_length=1)
    impact_strength: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    support_before: float = Field(ge=0.0, le=1.0)
    support_after: float = Field(ge=0.0, le=1.0)
    status: str = Field(min_length=1)
    source_type: Literal["advisory", "experimental", "human"] = "advisory"
    falsification_basis: str = ""


class HypothesisNode(ShadowBaseModel):
    hypothesis_id: str = Field(min_length=1)
    display_hypothesis_id: str | None = Field(
        default=None,
        description="user-facing hypothesis label such as H1/H2/H3",
    )
    statement: str = Field(min_length=1)
    display_statement: str | None = Field(
        default=None,
        description="user-facing statement built from physical variable display names",
    )
    level: int = Field(ge=1)
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    status: HypothesisStatus = "draft"
    support_score: float = Field(ge=0.0, le=1.0)
    support_history: list[SupportHistoryEntry] = Field(default_factory=list)
    questioning_records: list[HypothesisQuestioningRecord] = Field(default_factory=list)
    activation_condition: str = Field(min_length=1)
    activated_at_round: int | None = Field(default=None, ge=0)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    evidence_against: list[EvidenceItem] = Field(default_factory=list)
    critiques: list[CritiqueRecord] = Field(default_factory=list)
    predictions: list[PredictionRecord] = Field(default_factory=list)
    falsification_conditions: list[str] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    pruned_at_round: int | None = Field(default=None, ge=0)
    prune_reason: str | None = None
    created_at_round: int = Field(default=0, ge=0)
    updated_at_round: int = Field(default=0, ge=0)
    generation_rationale: HypothesisGenerationRationale | None = None


class TreeSummary(ShadowBaseModel):
    total_nodes: int = Field(ge=0)
    active_count: int = Field(ge=0)
    pruned_count: int = Field(ge=0)
    pending_count: int = Field(ge=0)


class LatestTreeUpdate(ShadowBaseModel):
    round: int = Field(ge=0)
    event: str = Field(min_length=1)
    description: str = Field(min_length=1)


class HypothesisTreeState(ShadowBaseModel):
    tree_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    current_round: int = Field(ge=0, default=0)
    root_question: str = Field(min_length=1)
    generated_at: datetime | None = Field(
        default=None,
        description="ISO timestamp recorded when the LLM generated the hypothesis proposal",
    )
    nodes: list[HypothesisNode] = Field(default_factory=list)
    active_hypotheses: list[str] = Field(default_factory=list)
    pruned_hypotheses: list[str] = Field(default_factory=list)
    pending_hypotheses: list[str] = Field(default_factory=list)
    latest_update: LatestTreeUpdate | None = None
    tree_summary: TreeSummary | None = None

    @field_validator("nodes")
    @classmethod
    def unique_hypothesis_ids(cls, value: list[HypothesisNode]) -> list[HypothesisNode]:
        ids = [item.hypothesis_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("HypothesisTreeState.nodes contains duplicate hypothesis_id")
        return value

    @model_validator(mode="after")
    def validate_indexes(self) -> "HypothesisTreeState":
        node_ids = {node.hypothesis_id for node in self.nodes}
        for group_name, items in (
            ("active_hypotheses", self.active_hypotheses),
            ("pruned_hypotheses", self.pruned_hypotheses),
            ("pending_hypotheses", self.pending_hypotheses),
        ):
            missing = set(items) - node_ids
            if missing:
                raise ValueError(f"{group_name} contains unknown hypotheses: {sorted(missing)}")
        return self

    def node_index(self) -> dict[str, HypothesisNode]:
        index: dict[str, HypothesisNode] = {}
        canonical_triggers = {
            "canonical_incremental_gain",
            "canonical_mediated_path",
            "canonical_beyond_path",
            "canonical_lead_time",
            "canonical_stability",
        }
        for node in self.nodes:
            index.setdefault(node.hypothesis_id, node)
        for node in self.nodes:
            alias = node.display_hypothesis_id
            if not alias or alias == node.hypothesis_id:
                continue
            trigger = (
                node.generation_rationale.trigger
                if node.generation_rationale and node.generation_rationale.trigger
                else ""
            )
            is_canonical = trigger in canonical_triggers
            if alias not in index or is_canonical:
                index[alias] = node
        return index


class UncertaintyDisagreement(ShadowBaseModel):
    hypothesis_id: str = Field(min_length=1)
    expected: str = Field(min_length=1)
    status: HypothesisStatus | str
    support_score: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale_trigger: str | None = None
    source_summary: str | None = None


class PriorityFactors(ShadowBaseModel):
    hypothesis_count: int = Field(ge=0)
    support_gap: float = Field(ge=0.0, le=1.0)
    expected_information_gain: float = Field(ge=0.0, le=1.0)
    estimated_cost: float = Field(ge=0.0, le=1.0)
    human_priority_override: float | None = Field(default=None, ge=0.0, le=1.0)


class UncertaintyHistoryEntry(ShadowBaseModel):
    round: int = Field(ge=0)
    event: str = Field(min_length=1)
    source: str = Field(min_length=1)
    description: str = Field(min_length=1)
    experiment_id: str | None = None
    run_id: str | None = None
    contrast_id: str | None = None
    evidence_version: str | None = None
    evidence_key: str | None = None


class UncertaintyRecord(ShadowBaseModel):
    uncertainty_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    description: str = Field(min_length=1)
    related_hypotheses: list[str] = Field(default_factory=list)
    disagreement: dict[str, UncertaintyDisagreement | dict[str, Any]] = Field(default_factory=dict)
    mining_sources: list[str] = Field(default_factory=list)
    status: UncertaintyStatus = "identified"
    priority: PriorityLevel = "medium"
    priority_factors: PriorityFactors | None = None
    created_at_round: int = Field(ge=0)
    created_by: str = Field(min_length=1)
    resolved_at_round: int | None = Field(default=None, ge=0)
    resolved_by: str | None = None
    resolution: str | None = None
    resolution_status: Literal["resolved", "partially_resolved", "unresolved"] | None = None
    resolving_experiment: str | None = None
    history: list[UncertaintyHistoryEntry] = Field(default_factory=list)
    alternative_resolutions: list[str] = Field(default_factory=list)
    notes: str | None = None


class PrioritizedUncertainty(ShadowBaseModel):
    uncertainty_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    priority_score: float = Field(ge=0.0, le=1.0)
    status: UncertaintyStatus | str
    estimated_resolution_round: int | None = Field(default=None, ge=0)


class UncertaintyPriorityQueue(ShadowBaseModel):
    last_updated: datetime
    current_round: int = Field(ge=0)
    queue: list[PrioritizedUncertainty] = Field(default_factory=list)
    resolved: list[str] = Field(default_factory=list)
    deprecated: list[str] = Field(default_factory=list)

    def top_uncertainties(self, limit: int = 3) -> list[PrioritizedUncertainty]:
        return sorted(self.queue, key=lambda item: item.priority_score, reverse=True)[:limit]


class ExperimentDesign(ShadowBaseModel):
    target: str = Field(min_length=1)
    control: list[str] = Field(default_factory=list)
    treatment: list[str] = Field(default_factory=list)
    lags: dict[str, list[int]] = Field(default_factory=dict)
    probe_axis: str | None = Field(
        default=None,
        description="which scientific axis this experiment discriminates: incremental_gain, physical_path, lead_time, robustness",
    )
    forecast_horizon_days: int | None = Field(default=None, ge=1)
    past_lag_days: int | None = Field(default=None, ge=1)
    window_size: int | None = Field(default=None, ge=1)
    control_lag_days: int | None = Field(default=None, ge=1)
    treatment_lag_days: int | None = Field(default=None, ge=1)
    control_forecast_horizon_days: int | None = Field(default=None, ge=1)
    treatment_forecast_horizon_days: int | None = Field(default=None, ge=1)
    notes: list[str] = Field(default_factory=list)
    design_focus: str | None = Field(default=None, description="raw feature name this candidate isolates")
    display_target: str | None = Field(default=None, description="user-facing target variable name")
    display_control: list[str] = Field(default_factory=list, description="user-facing control variable names")
    display_treatment: list[str] = Field(default_factory=list, description="user-facing treatment variable names")
    display_design_focus: str | None = Field(default=None, description="user-facing design focus name")
    model_families: list[str] = Field(
        default_factory=list,
        description="registered model families for model_sensitivity; Ridge / ElasticNet only",
    )
    runs_spec: list[RunSpec] = Field(default_factory=list)
    contrasts_spec: list[ContrastSpec] = Field(default_factory=list)
    diagnostics_spec: list[DiagnosticSpec] = Field(default_factory=list)
    alignment: ProtocolAlignmentSpec | None = Field(
        default=None,
        description="跨配置日期对齐规则，供时间敏感性等类型使用",
    )


class HypothesisPrediction(ShadowBaseModel):
    expected_effect: ExpectedEffect | str
    expected_range: list[float] | None = None
    metric: str | None = Field(
        default=None,
        description="metric on which the LLM expected_range is measured",
    )

    @field_validator("expected_range")
    @classmethod
    def validate_range(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != 2:
            raise ValueError("expected_range must contain exactly two values")
        return value


class EstimatedValue(ShadowBaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    rationale: str | None = None


class CandidateExperiment(ShadowBaseModel):
    experiment_id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    experiment_type: str | None = Field(
        default=None,
        description="registered executable experiment type; legacy candidates derive from type",
    )
    purpose: str = Field(min_length=1)
    scientific_question: str | None = None
    tested_hypotheses: list[str] = Field(default_factory=list)
    related_uncertainties: list[str] = Field(default_factory=list)
    disagreement_context: dict[str, dict[str, UncertaintyDisagreement | dict[str, Any]]] = Field(default_factory=dict)
    hypothesis_source_context: dict[str, HypothesisGenerationRationale | dict[str, Any]] = Field(default_factory=dict)
    design: ExperimentDesign
    hypothesis_predictions: dict[str, HypothesisPrediction] = Field(default_factory=dict)
    distinguishing_insight: str | None = None
    value_analysis: str | None = None
    estimated_information_gain: EstimatedValue | None = None
    estimated_performance_gain: EstimatedValue | None = None
    estimated_risk: EstimatedValue | None = None
    estimated_cost: EstimatedValue | None = None
    utility_score: float | None = Field(default=None, ge=0.0, le=1.0)
    requires_human_review: bool = False
    novelty: str | None = None
    responded_prior_finding: str | None = Field(
        default=None,
        description="本轮实验所回应的上一轮发现；没有历史实验时可为当前不确定性说明",
    )
    target_uncertainty_focus: str | None = Field(
        default=None,
        description="本轮实验试图解决的科学不确定性",
    )
    changed_factors: list[str] = Field(
        default_factory=list,
        description="相对上一轮实验改变的因素（模型家族、特征集、超前天数、历史长度等）",
    )
    held_fixed_factors: list[str] = Field(
        default_factory=list,
        description="相对上一轮实验保持固定的条件",
    )
    expected_outcome_branches: dict[str, str] = Field(
        default_factory=dict,
        description="不同结果对应下一步的预登记判断",
    )
    exploratory: bool = Field(
        default=False,
        description="同时改变多个因素时标记为探索性实验，不做单因素归因",
    )

    @model_validator(mode="after")
    def resolve_experiment_type(self) -> "CandidateExperiment":
        resolved = (
            str(self.experiment_type).strip()
            if self.experiment_type and str(self.experiment_type).strip()
            else (
                "baseline_benchmark"
                if self.type == "baseline_benchmark"
                else "incremental_prediction"
            )
        )
        if self.experiment_type != resolved:
            object.__setattr__(self, "experiment_type", resolved)
        return self


class RejectedCandidate(ShadowBaseModel):
    experiment_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    similar_to: str | None = None


class CandidateExperimentSet(ShadowBaseModel):
    round: int = Field(ge=0)
    generated_at: datetime
    candidates: list[CandidateExperiment] = Field(default_factory=list)
    rejected_candidates: list[RejectedCandidate] = Field(default_factory=list)
    minimum_required: int = Field(default=3, ge=1)
    current_count: int | None = Field(default=None, ge=0)
    note: str | None = None

    @field_validator("candidates")
    @classmethod
    def unique_experiment_ids(cls, value: list[CandidateExperiment]) -> list[CandidateExperiment]:
        ids = [item.experiment_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("CandidateExperimentSet contains duplicate experiment_id")
        return value

    @model_validator(mode="after")
    def populate_current_count(self) -> "CandidateExperimentSet":
        if self.current_count is None:
            self.current_count = len(self.candidates)
        return self

    def top_candidate(self) -> CandidateExperiment | None:
        scored = [item for item in self.candidates if item.utility_score is not None]
        if not scored:
            return None
        discriminative = [item for item in scored if item.design.control != item.design.treatment]
        if discriminative:
            return max(discriminative, key=lambda item: item.utility_score or 0.0)
        return max(scored, key=lambda item: item.utility_score or 0.0)


class ResourceBudget(ShadowBaseModel):
    token_budget: int | None = Field(default=None, ge=0)
    max_time_seconds_per_round: int | None = Field(default=None, ge=0)
    max_candidates: int | None = Field(default=None, ge=1)


class ModelSpec(ShadowBaseModel):
    name: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ExperimentStep(ShadowBaseModel):
    step: int = Field(ge=1)
    action: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ProtocolConstraints(ShadowBaseModel):
    no_future_information: bool = True
    target_immutable: bool = True


class ProtocolAlignmentSpec(ShadowBaseModel):
    """跨配置（如时间尺度敏感性）的日期对齐规则。"""

    common_issue_date: bool = False
    common_target_date: bool = False
    note: str | None = None


class RunSpec(ShadowBaseModel):
    """协议中一次具体模型训练与预测的定义。"""

    run_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    feature_columns: list[str] = Field(default_factory=list)
    model_family: str = Field(default="ElasticNet", min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    retrain: bool = True
    source_experiment_id: str | None = None
    source_execution_id: str | None = None
    source_prediction_file: str | None = None


class ContrastSpec(ShadowBaseModel):
    """协议中两个或更多 run 之间预先定义的比较。"""

    contrast_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: Literal["metric_delta", "incremental_gain", "gain_change", "group_gain_difference"] | str = (
        "metric_delta"
    )
    run_ids: list[str] = Field(default_factory=list)
    contrast_ids: list[str] = Field(default_factory=list)
    direction: Literal["reduce_error", "increase_correlation", "mixed", "neutral"] | str = "mixed"
    note: str | None = None

    @model_validator(mode="after")
    def require_at_least_one_reference(self) -> "ContrastSpec":
        if not self.run_ids and not self.contrast_ids:
            raise ValueError("ContrastSpec 至少需要 run_ids 或 contrast_ids 之一")
        return self


class DiagnosticSpec(ShadowBaseModel):
    """协议中基于已有 run 或已有预测结果的诊断。"""

    diagnostic_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    source_run_ids: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    retrain: bool = False
    note: str | None = None


class ExperimentProtocol(ShadowBaseModel):
    experiment_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    task_id: str = Field(min_length=1)
    target: str = Field(min_length=1)
    experiment_type: ExperimentType = "incremental_prediction"
    candidate_id: str | None = None
    source_evidence_ids: list[str] = Field(default_factory=list)
    scientific_objective: str | None = None
    forecast_horizon_days: int | None = Field(default=None, ge=1)
    features: ExperimentDesign
    model: ModelSpec = Field(default_factory=lambda: ModelSpec(name="ElasticNet"))
    runs_spec: list[RunSpec] = Field(default_factory=list)
    contrasts_spec: list[ContrastSpec] = Field(default_factory=list)
    diagnostics_spec: list[DiagnosticSpec] = Field(default_factory=list)
    protocol_version: str = "1.0"
    alignment: ProtocolAlignmentSpec | None = None
    tested_hypotheses: list[str] = Field(default_factory=list)
    target_uncertainties: list[str] = Field(default_factory=list)
    disagreement_context: dict[str, dict[str, UncertaintyDisagreement | dict[str, Any]]] = Field(default_factory=dict)
    hypothesis_source_context: dict[str, HypothesisGenerationRationale | dict[str, Any]] = Field(default_factory=dict)
    hypothesis_predictions: dict[str, HypothesisPrediction] = Field(default_factory=dict)
    evaluation_metrics: list[MetricName] = Field(default_factory=lambda: ["RMSE", "MAE", "Pearson_r", "Skill"])
    budget: ResourceBudget | None = None
    steps: list[ExperimentStep] = Field(default_factory=list)
    constraints: ProtocolConstraints = Field(default_factory=ProtocolConstraints)
    source_candidate_id: str | None = None
    allowed_conclusions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    status: ExperimentLifecycleStatus = "proposed"
    requires_human_review: bool = False
    notes: list[str] = Field(default_factory=list)
    responded_prior_finding: str | None = Field(
        default=None,
        description="协议所回应的上一轮发现",
    )
    target_uncertainty_focus: str | None = Field(
        default=None,
        description="协议试图解决的科学不确定性",
    )
    changed_factors: list[str] = Field(
        default_factory=list,
        description="相对上一轮实验改变的因素",
    )
    held_fixed_factors: list[str] = Field(
        default_factory=list,
        description="相对上一轮实验保持固定的条件",
    )
    expected_outcome_branches: dict[str, str] = Field(
        default_factory=dict,
        description="不同结果对应下一步的预登记判断",
    )
    exploratory: bool = Field(
        default=False,
        description="是否标记为探索性实验",
    )


class RunMetrics(ShadowBaseModel):
    rmse: float | None = None
    mae: float | None = None
    pearson_r: float | None = None
    skill: float | None = None
    r2: float | None = None
    unavailable_reasons: dict[str, str] = Field(
        default_factory=dict,
        description="按指标名记录不可计算的原因，如 constant_prediction",
    )


class ExperimentRun(ShadowBaseModel):
    run_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    prediction_artifact: str | None = None
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    model_family: str = Field(default="ElasticNet", min_length=1)
    feature_labels: list[str] = Field(default_factory=list)
    feature_columns: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    n_train: int | None = Field(default=None, ge=0)
    n_test: int | None = Field(default=None, ge=0)
    training_mode: Literal["retrained", "reused"] = "retrained"
    protocol_version: str | None = None


class MetricDelta(ShadowBaseModel):
    rmse: float | None = None
    mae: float | None = None
    pearson_r: float | None = None
    skill: float | None = None
    r2: float | None = None


class VisualizationArtifact(ShadowBaseModel):
    name: str = Field(min_length=1)
    path: str = Field(min_length=1)


class ExecutionSummary(ShadowBaseModel):
    duration_seconds: float | None = Field(default=None, ge=0.0)
    compute_seconds: float | None = Field(default=None, ge=0.0)
    status: ExecutionStatus
    message: str | None = None


class DataSourceCoverage(ShadowBaseModel):
    """单个实验臂内某个数据源的日历覆盖与缺失日处理审计信息。"""

    source: str = Field(min_length=1)
    run_id: str | None = None
    expected_days: int | None = Field(default=None, ge=0)
    observed_days: int | None = Field(default=None, ge=0)
    missing_days: int | None = Field(default=None, ge=0)
    coverage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    dropped_gap_windows: int | None = Field(default=None, ge=0)
    interpolated_days: int | None = Field(default=None, ge=0)
    note: str | None = None


class ExperimentResult(ShadowBaseModel):
    experiment_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    status: ExecutionStatus
    experiment_type: ExperimentType = "incremental_prediction"
    protocol_version: str | None = None
    runs: list[ExperimentRun] = Field(default_factory=list)
    comparison: MetricDelta = Field(default_factory=MetricDelta)
    contrasts: list["ExperimentContrast"] = Field(default_factory=list)
    diagnostics: list["ExperimentDiagnostic"] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)
    allowed_conclusions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    validity_status: Literal["valid", "uncertain", "invalid", "pending_scientific_review"] = "valid"
    warnings: list[str] = Field(default_factory=list)
    visualizations: list[VisualizationArtifact] = Field(default_factory=list)
    data_coverage: list[DataSourceCoverage] = Field(default_factory=list)
    execution: ExecutionSummary
    run_id: str | None = None
    robustness: RobustnessReport | None = None


class ExperimentContrast(ShadowBaseModel):
    """一次已执行的、协议预先定义的 run 比较。"""

    contrast_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: Literal["metric_delta", "incremental_gain", "gain_change", "group_gain_difference"] | str = (
        "metric_delta"
    )
    specification: dict[str, Any] = Field(default_factory=dict)
    run_ids: list[str] = Field(default_factory=list)
    contrast_ids: list[str] = Field(default_factory=list)
    metrics_delta: MetricDelta = Field(default_factory=MetricDelta)
    channel_gains: dict[str, float] = Field(default_factory=dict)
    bootstrap_ci: "BootstrapReport | None" = None
    note: str | None = None


class ExperimentDiagnostic(ShadowBaseModel):
    """一次已执行的、基于已有 run 或预测结果的诊断。"""

    diagnostic_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    source_run_ids: list[str] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    structured: dict[str, Any] = Field(default_factory=dict)
    status: Literal["completed", "failed", "not_run", "pending_scientific_review"] = "completed"
    note: str | None = None


class PerformanceMetrics(ShadowBaseModel):
    experiment_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    experiment_type: ExperimentType | None = Field(
        default=None,
        description="registered executable experiment type of the evaluated result",
    )
    baseline_rmse: float | None = None
    baseline_mae: float | None = None
    baseline_pearson_r: float | None = None
    baseline_skill: float | None = None
    treatment_rmse: float | None = None
    treatment_mae: float | None = None
    treatment_pearson_r: float | None = None
    treatment_skill: float | None = None
    pg_actual_signed: float | None = None
    pg_actual_clipped: float | None = Field(default=None, ge=0.0, le=1.0)
    delta: MetricDelta = Field(default_factory=MetricDelta)
    information_gain_kl: float | None = Field(
        default=None,
        ge=0.0,
        description="post-experiment KL information-gain audit from actual observed delta",
    )
    ig_prior_probs: dict[str, float] | None = None
    ig_posterior_probs: dict[str, float] | None = None
    runs: list[ExperimentRun] = Field(default_factory=list)
    contrasts: list[ExperimentContrast] = Field(default_factory=list)
    diagnostics: list[ExperimentDiagnostic] = Field(default_factory=list)


class BootstrapReport(ShadowBaseModel):
    pearson_r_ci: list[float] | None = None
    rmse_ci: list[float] | None = None
    mae_ci: list[float] | None = None
    skill_ci: list[float] | None = None
    stable: bool
    n_iterations: int | None = Field(default=None, ge=0)
    status: str | None = Field(default=None, description="completed / not_run / failed")
    block_size: int | None = Field(default=None, ge=1)
    n_nonempty_blocks: int | None = Field(default=None, ge=0)
    min_blocks: int | None = Field(default=None, ge=1)
    insufficient_blocks: bool = Field(
        default=False,
        description="有效日历块数低于预登记最小块数，不能作出方向性稳定结论",
    )
    seed: int | None = None
    point_gain_rmse: float | None = None
    point_gain_mae: float | None = None
    point_gain_pearson_r: float | None = None


class TimeSliceFold(ShadowBaseModel):
    fold: int = Field(ge=1)
    segment: str | None = None
    date_start: str | None = None
    date_end: str | None = None
    n_samples: int | None = Field(default=None, ge=0)
    pearson_r: float | None = None
    rmse: float | None = None
    skill: float | None = None
    baseline_rmse: float | None = None
    treatment_rmse: float | None = None
    baseline_bias: float | None = None
    treatment_bias: float | None = None


class TimeSliceReport(ShadowBaseModel):
    splits: list[TimeSliceFold] = Field(default_factory=list)
    std_pearson_r: float | None = None
    stable: bool
    status: str | None = Field(
        default=None,
        description="not_run / descriptive / stable / inconclusive",
    )
    reason: str | None = None
    metric: str | None = None


class RobustnessOverall(ShadowBaseModel):
    stable: bool
    concern: str | None = None


class RobustnessReport(ShadowBaseModel):
    bootstrap: BootstrapReport | None = None
    time_slice: TimeSliceReport | None = None
    overall: RobustnessOverall
    recommendation: str | None = None


class HypothesisAssessment(ShadowBaseModel):
    hypothesis_id: str = Field(min_length=1)
    support_before: float = Field(ge=0.0, le=1.0)
    support_after: float = Field(ge=0.0, le=1.0)
    status: HypothesisStatus | Literal["supported", "partially_supported", "weakened"]
    direction_matched: bool | Literal["partial"] | None = None
    magnitude_matched: bool | Literal["partial"] | None = None
    reason: str = Field(min_length=1)
    metric: str = "rmse"
    observed_metric_delta: float | None = None
    metric_ci: list[float] | None = None
    evidence_classification: str | None = None
    evidence_labels: list[str] = Field(default_factory=list)
    updated: bool = True
    run_id: str | None = None
    contrast_id: str | None = None


class EvidenceClaim(ShadowBaseModel):
    claim: str = Field(min_length=1)
    strength: PriorityLevel | str
    source: str | None = None


class EvidenceSummary(ShadowBaseModel):
    new_evidence_for: list[EvidenceClaim] = Field(default_factory=list)
    new_evidence_against: list[EvidenceClaim] = Field(default_factory=list)
    remaining_uncertainties: list[dict[str, Any]] = Field(default_factory=list)


class DisagreementUpdate(ShadowBaseModel):
    uncertainty_id: str = Field(min_length=1)
    compared_hypotheses: list[str] = Field(default_factory=list)
    support_span_before: float = Field(ge=0.0, le=1.0)
    support_span_after: float = Field(ge=0.0, le=1.0)
    narrowed: bool = False
    resolution_status: Literal["resolved", "partially_resolved", "unresolved"] = "unresolved"
    evidence_basis: Literal[
        "stable_experiment_evidence",
        "missing_valid_experiment_evidence",
    ] = "missing_valid_experiment_evidence"
    leading_hypothesis_id: str | None = None
    unresolved_hypotheses: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)


class ConclusionExperimentLayer(ShadowBaseModel):
    experiment_id: str = Field(min_length=1)
    design_summary: str = Field(min_length=1)
    probe_axis: str | None = None
    forecast_horizon_days: int | None = None
    baseline_rmse: float | None = None
    treatment_rmse: float | None = None
    baseline_pearson_r: float | None = None
    treatment_pearson_r: float | None = None
    skill_delta: float | None = None
    decisive: bool = False
    experiment_type: str | None = None
    runs: list[ExperimentRun] = Field(default_factory=list)
    contrasts: list[ExperimentContrast] = Field(default_factory=list)
    diagnostics: list[ExperimentDiagnostic] = Field(default_factory=list)
    model_sensitivity_families: list[ModelSensitivityFamilyEvidence] = Field(default_factory=list)


class ModelSensitivityFamilyEvidence(ShadowBaseModel):
    """One model family's shadow-gain point estimate, CI and conservative verdict."""

    family: str = Field(min_length=1)
    contrast_id: str | None = None
    point_gain_rmse: float | None = None
    point_gain_pearson_r: float | None = None
    rmse_ci: list[float] | None = None
    pearson_ci: list[float] | None = None
    crosses_zero: bool | None = None
    evaluation: str = Field(min_length=1)
    evaluation_label: str | None = None


class ConclusionHypothesisRow(ShadowBaseModel):
    hypothesis_id: str = Field(min_length=1)
    display_hypothesis_id: str | None = None
    statement: str = Field(min_length=1)
    predicted_direction: str | None = None
    predicted_range: list[float] | None = None
    actual_delta: float | None = None
    direction_matched: bool | str | None = None
    magnitude_matched: bool | str | None = None
    conclusion: str = Field(min_length=1)
    support_after: float | None = None


class ConclusionScientificLayer(ShadowBaseModel):
    main_question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    path_question: str | None = None
    path_answer: str | None = None
    evidence_text: str | None = None
    interpretation_source: str = "program_fallback"


class ConclusionChartAnalysis(ShadowBaseModel):
    """LLM description of one report chart, kept together with its image path."""

    chart_name: str = Field(min_length=1)
    chart_role: str = Field(min_length=1)
    chart_image_path: str | None = None
    description: str = Field(min_length=1)
    key_observations: list[str] = Field(default_factory=list)
    anomaly_or_insight: str | None = None


class ConclusionNextRoundSuggestion(ShadowBaseModel):
    """Real-LLM advice payload persisted for the next planning round."""

    evidence_summary: str | None = None
    resolved_uncertainties_this_round: list[str] = Field(default_factory=list)
    unresolved_uncertainties_todo: list[str] = Field(default_factory=list)
    remaining_uncertainty_analysis: str | None = None
    recommendation: Literal["continue", "adjust", "stop"] | None = None
    pi_decision_advice: str | None = None
    experiment_design_advice: str | None = None
    hypothesis_space_advice: str | None = None
    notes: list[str] = Field(default_factory=list)


class ConclusionDataLayer(ShadowBaseModel):
    """LLM numerical interpretation layer for round report."""

    rmse_attribution: str | None = None
    pearson_attribution: str | None = None
    skill_delta_meaning: str | None = None
    anomalies: list[str] = Field(default_factory=list)
    next_focus: str | None = None
    chart_analyses: list[ConclusionChartAnalysis] = Field(default_factory=list)
    comparison_analysis: str | None = None
    interpretation_source: str = "program_fallback"


class ConclusionTrackingLayer(ShadowBaseModel):
    """Program-generated audit and provenance layer for round report."""

    audit_items: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    snapshot_refs: list[str] = Field(default_factory=list)


class ThreeLayerConclusion(ShadowBaseModel):
    experiment_layer: ConclusionExperimentLayer
    hypothesis_layer: list[ConclusionHypothesisRow] = Field(default_factory=list)
    scientific_layer: ConclusionScientificLayer
    data_layer: ConclusionDataLayer | None = None
    tracking_layer: ConclusionTrackingLayer | None = None
    hypothesis_layer_summary: str | None = None
    overall_summary: str | None = None
    next_round_suggestion: ConclusionNextRoundSuggestion | None = None


class ScientificEvaluation(ShadowBaseModel):
    experiment_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    hypothesis_assessments: list[HypothesisAssessment] = Field(default_factory=list)
    disagreement_updates: list[DisagreementUpdate] = Field(default_factory=list)
    evidence_summary: EvidenceSummary = Field(default_factory=EvidenceSummary)
    three_layer_conclusion: ThreeLayerConclusion | None = None


class EvaluationResult(ShadowBaseModel):
    metrics: PerformanceMetrics
    robustness: RobustnessReport
    scientific: ScientificEvaluation
    visualizations: list[VisualizationArtifact] = Field(default_factory=list)

    def to_interpreter_payload(self) -> dict[str, Any]:
        return {
            "performance_metrics": self.metrics.model_dump(exclude_none=True),
            "robustness": self.robustness.model_dump(exclude_none=True),
            "hypothesis_assessments": [
                item.model_dump(exclude_none=True) for item in self.scientific.hypothesis_assessments
            ],
            "visualizations": [item.model_dump(exclude_none=True) for item in self.visualizations],
            "remaining_uncertainties": self.scientific.evidence_summary.remaining_uncertainties,
        }


class PlannerUncertaintyItem(ShadowBaseModel):
    uncertainty_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    priority: PriorityLevel | str
    priority_score: float | None = Field(default=None, ge=0.0, le=1.0)
    status: UncertaintyStatus | str | None = None
    resolution_status: Literal["resolved", "partially_resolved", "unresolved"] | None = None
    related_hypotheses: list[str] = Field(default_factory=list)


class HypothesisSnapshot(ShadowBaseModel):
    hypothesis_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    status: HypothesisStatus | str
    support_score: float = Field(ge=0.0, le=1.0)


class DataDictionarySummary(ShadowBaseModel):
    dictionary_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    time_column: str = Field(min_length=1)
    target_candidates: list[str] = Field(default_factory=list)
    feature_candidates: list[str] = Field(default_factory=list)
    display_time_column: str | None = None
    display_target_candidates: list[str] = Field(default_factory=list)
    display_feature_candidates: list[str] = Field(default_factory=list)
    raw_display_map: dict[str, str] = Field(default_factory=dict)
    display_raw_map: dict[str, str] = Field(default_factory=dict)


class PlannerEvaluationSummary(ShadowBaseModel):
    experiment_id: str | None = None
    round_id: int = Field(ge=0)
    baseline_rmse: float | None = None
    treatment_rmse: float | None = None
    baseline_pearson_r: float | None = None
    treatment_pearson_r: float | None = None
    delta_pearson_r: float | None = None
    delta_rmse: float | None = None
    pg_actual_signed: float | None = None
    pg_actual_clipped: float | None = Field(default=None, ge=0.0, le=1.0)
    key_findings: list[str] = Field(default_factory=list)
    robustness_recommendation: str | None = None
    stable: bool | None = None


class PlannerReasoningTraceItem(ShadowBaseModel):
    trace_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    related_hypotheses: list[str] = Field(default_factory=list)
    related_uncertainties: list[str] = Field(default_factory=list)
    referenced_variables: list[str] = Field(
        default_factory=list,
        description="Raw column/domain variable names referenced by this trace.",
    )
    support_delta: float | None = Field(default=None, ge=-1.0, le=1.0)
    priority_delta: float | None = Field(default=None, ge=-1.0, le=1.0)


class PlannerHumanFeedbackItem(ShadowBaseModel):
    feedback_id: str = Field(min_length=1)
    feedback_type: str = Field(min_length=1)
    content: str = Field(min_length=1)
    linked_reasoning_trace_ids: list[str] = Field(default_factory=list)
    related_experiment_id: str | None = None
    context_summary: str | None = None


class PlannerDisagreementUpdateItem(ShadowBaseModel):
    uncertainty_id: str = Field(min_length=1)
    resolution_status: Literal["resolved", "partially_resolved", "unresolved"]
    evidence_basis: Literal[
        "stable_experiment_evidence",
        "missing_valid_experiment_evidence",
    ] = "missing_valid_experiment_evidence"
    compared_hypotheses: list[str] = Field(default_factory=list)
    narrowed: bool = False
    leading_hypothesis_id: str | None = None
    unresolved_hypotheses: list[str] = Field(default_factory=list)
    support_span_before: float = Field(ge=0.0, le=1.0)
    support_span_after: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1)


class ReasoningPlannerInput(ShadowBaseModel):
    task_id: str = Field(min_length=1)
    source_round_id: int = Field(ge=0)
    next_round_id: int = Field(ge=0)
    source_experiment_id: str | None = None
    scientific_question: str = Field(min_length=1)
    target: str = Field(min_length=1)
    human_feedback: str | None = None
    evaluation_summary: PlannerEvaluationSummary
    unresolved_uncertainties: list[PlannerUncertaintyItem] = Field(default_factory=list)
    active_hypotheses: list[HypothesisSnapshot] = Field(default_factory=list)
    recent_hypothesis_assessments: list[HypothesisAssessment] = Field(default_factory=list)
    recent_disagreement_updates: list[PlannerDisagreementUpdateItem] = Field(default_factory=list)
    recent_reasoning_traces: list[PlannerReasoningTraceItem] = Field(default_factory=list)
    recent_human_feedback: list[PlannerHumanFeedbackItem] = Field(default_factory=list)
    data_dictionary_summary: DataDictionarySummary
    planning_constraints: dict[str, Any] = Field(default_factory=dict)
    planner_guidance: list[str] = Field(default_factory=list)
    llm_hypothesis_proposals: list[dict[str, Any]] = Field(
        default_factory=list,
        description="LLM-written final hypothesis texts for the next tree rebuild",
    )
    generated_at: datetime = Field(default_factory=datetime.now)

    def to_central_controller_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_round_id": self.source_round_id,
            "next_round_id": self.next_round_id,
            "scientific_question": self.scientific_question,
            "target": self.target,
            "human_feedback": self.human_feedback,
            "evaluation_summary": self.evaluation_summary.model_dump(exclude_none=True),
            "recent_hypothesis_assessments": [
                item.model_dump(exclude_none=True) for item in self.recent_hypothesis_assessments
            ],
            "recent_disagreement_updates": [
                item.model_dump(exclude_none=True) for item in self.recent_disagreement_updates
            ],
            "recent_human_feedback": [
                item.model_dump(exclude_none=True) for item in self.recent_human_feedback
            ],
            "planner_guidance": self.planner_guidance,
        }

    def to_experiment_planner_payload(self) -> dict[str, Any]:
        return {
            "source_experiment_id": self.source_experiment_id,
            "human_feedback": self.human_feedback,
            "unresolved_uncertainties": [
                item.model_dump(exclude_none=True) for item in self.unresolved_uncertainties
            ],
            "active_hypotheses": [
                item.model_dump(exclude_none=True) for item in self.active_hypotheses
            ],
            "recent_disagreement_updates": [
                item.model_dump(exclude_none=True) for item in self.recent_disagreement_updates
            ],
            "recent_reasoning_traces": [
                item.model_dump(exclude_none=True) for item in self.recent_reasoning_traces
            ],
            "recent_human_feedback": [
                item.model_dump(exclude_none=True) for item in self.recent_human_feedback
            ],
            "data_dictionary_summary": self.data_dictionary_summary.model_dump(exclude_none=True),
            "planner_guidance": self.planner_guidance,
        }

    def to_scientific_interpreter_payload(self) -> dict[str, Any]:
        return {
            "source_round_id": self.source_round_id,
            "evaluation_summary": self.evaluation_summary.model_dump(exclude_none=True),
            "recent_hypothesis_assessments": [
                item.model_dump(exclude_none=True) for item in self.recent_hypothesis_assessments
            ],
            "recent_disagreement_updates": [
                item.model_dump(exclude_none=True) for item in self.recent_disagreement_updates
            ],
            "unresolved_uncertainties": [
                item.model_dump(exclude_none=True) for item in self.unresolved_uncertainties
            ],
            "recent_reasoning_traces": [
                item.model_dump(exclude_none=True) for item in self.recent_reasoning_traces
            ],
            "planner_guidance": self.planner_guidance,
        }

    def merged_guidance_text(self) -> str | None:
        parts: list[str] = []
        if self.human_feedback and self._guidance_part_is_clean(self.human_feedback):
            parts.append(self.human_feedback)
        parts.extend(self.evaluation_summary.key_findings[:2])
        parts.extend(item.summary for item in self.recent_disagreement_updates[:2])
        parts.extend(
            item.summary
            for item in self.recent_reasoning_traces[:2]
            if self._guidance_part_is_clean(item.summary)
        )
        parts.extend(
            item.content
            for item in self.recent_human_feedback[:2]
            if self._guidance_part_is_clean(item.content)
        )
        parts.extend(
            item.question
            for item in self.unresolved_uncertainties[:2]
            if self._guidance_part_is_clean(item.question)
        )
        parts.extend(
            part for part in self.planner_guidance[:2] if self._guidance_part_is_clean(part)
        )
        return " ".join(part for part in parts if part) or None

    @staticmethod
    def _guidance_part_is_clean(part: str | None) -> bool:
        """Refuse malformed JSON fragments from ever reaching user-facing notes."""
        if not part:
            return False
        lowered = str(part).lower()
        corrupted_markers = (
            "yload",
            "\\\"",
            '"research_question"',
            '"current_round"',
            '"root_question"',
            '"active_hypotheses"',
            '"data_dictionary_summary"',
        )
        if any(marker in lowered for marker in corrupted_markers):
            return False
        if len(part) > 4000:
            return False
        return True


class PlannerCandidateSupplement(ShadowBaseModel):
    proposal_id: str = Field(min_length=1)
    candidate: CandidateExperiment
    rationale: str = Field(min_length=1)
    source: str = Field(default="programmatic_planner", min_length=1)


class InterpretationEnhancement(ShadowBaseModel):
    enhancement_id: str = Field(min_length=1)
    target_hypothesis_id: str | None = None
    interpretation: str = Field(min_length=1)
    evidence_basis: list[str] = Field(default_factory=list)
    related_uncertainties: list[str] = Field(default_factory=list)
    suggested_state_note: str | None = None
    impact_direction: Literal["supports", "weakens", "clarifies"] = "clarifies"
    impact_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    uncertainty_priority_action: Literal["increase", "decrease", "maintain"] = "maintain"
    run_id: str | None = None
    contrast_id: str | None = None
    evidence_version: str | None = None
    evidence_key: str | None = None


class ProtocolRefinementSuggestion(ShadowBaseModel):
    suggestion_id: str = Field(min_length=1)
    target_candidate_id: str | None = None
    refinement_type: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    suggested_model_parameters: dict[str, Any] = Field(default_factory=dict)
    suggested_feature_focus: list[str] = Field(default_factory=list)
    suggested_steps: list[ExperimentStep] = Field(default_factory=list)
    protocol_notes: list[str] = Field(default_factory=list)


class ReasoningPlannerOutput(ShadowBaseModel):
    task_id: str = Field(min_length=1)
    source_round_id: int = Field(ge=0)
    next_round_id: int = Field(ge=0)
    source_experiment_id: str | None = None
    planner_mode: Literal["programmatic", "llm"] = "programmatic"
    candidate_supplements: list[PlannerCandidateSupplement] = Field(default_factory=list)
    interpretation_enhancements: list[InterpretationEnhancement] = Field(default_factory=list)
    protocol_refinements: list[ProtocolRefinementSuggestion] = Field(default_factory=list)
    summary: str | None = None
    generated_at: datetime = Field(default_factory=datetime.now)

    def to_experiment_planner_payload(self) -> dict[str, Any]:
        return {
            "candidate_supplements": [
                item.model_dump(exclude_none=True) for item in self.candidate_supplements
            ],
            "protocol_refinements": [
                item.model_dump(exclude_none=True) for item in self.protocol_refinements
            ],
            "summary": self.summary,
        }

    def to_scientific_interpreter_payload(self) -> dict[str, Any]:
        return {
            "interpretation_enhancements": [
                item.model_dump(exclude_none=True) for item in self.interpretation_enhancements
            ],
            "summary": self.summary,
        }


class UncertaintyState(ShadowBaseModel):
    task_id: str = Field(min_length=1)
    current_round: int = Field(ge=0)
    last_updated: datetime
    records: list[UncertaintyRecord] = Field(default_factory=list)
    priority_queue: UncertaintyPriorityQueue | None = None

    @field_validator("records")
    @classmethod
    def unique_uncertainty_ids(cls, value: list[UncertaintyRecord]) -> list[UncertaintyRecord]:
        ids = [item.uncertainty_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("UncertaintyState.records contains duplicate uncertainty_id")
        return value

    def record_index(self) -> dict[str, UncertaintyRecord]:
        return {item.uncertainty_id: item for item in self.records}


class ExperimentMemoryEntry(ShadowBaseModel):
    experiment_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    experiment_type: str | None = Field(
        default=None,
        description="registered executable experiment type; used for comparable-history selection",
    )
    model_families: list[str] = Field(default_factory=list)
    status: ExecutionStatus
    tested_hypotheses: list[str] = Field(default_factory=list)
    target_uncertainties: list[str] = Field(default_factory=list)
    protocol_path: str | None = None
    result_path: str | None = None
    evaluation_path: str | None = None
    metrics_snapshot: PerformanceMetrics | None = None
    key_findings: list[str] = Field(default_factory=list)
    visualizations: list[str] = Field(default_factory=list)
    data_coverage: list[DataSourceCoverage] = Field(default_factory=list)
    reasoning_traces: list["ReasoningTraceEntry"] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ReasoningTraceEntry(ShadowBaseModel):
    trace_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    source: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    related_hypotheses: list[str] = Field(default_factory=list)
    related_uncertainties: list[str] = Field(default_factory=list)
    referenced_variables: list[str] = Field(
        default_factory=list,
        description="Raw column/domain variable names referenced by this trace.",
    )
    support_delta: float | None = Field(default=None, ge=-1.0, le=1.0)
    priority_delta: float | None = Field(default=None, ge=-1.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)
    experiment_id: str | None = None
    run_id: str | None = None
    contrast_id: str | None = None
    evidence_version: str | None = None
    evidence_key: str | None = None


class ExperimentMemoryState(ShadowBaseModel):
    task_id: str = Field(min_length=1)
    current_round: int = Field(ge=0)
    last_updated: datetime
    entries: list[ExperimentMemoryEntry] = Field(default_factory=list)

    @field_validator("entries")
    @classmethod
    def unique_experiment_entries(cls, value: list[ExperimentMemoryEntry]) -> list[ExperimentMemoryEntry]:
        ids = [item.experiment_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("ExperimentMemoryState.entries contains duplicate experiment_id")
        return value

    def entry_index(self) -> dict[str, ExperimentMemoryEntry]:
        return {item.experiment_id: item for item in self.entries}


class ClosureChecklistItem(ShadowBaseModel):
    item_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    completed: bool = False
    required: bool = True
    detail: str | None = None


class RoundHistoryEntry(ShadowBaseModel):
    round_id: int = Field(ge=0)
    status: Literal["in_progress", "completed", "failed", "stopped"] = "in_progress"
    source_experiment_id: str | None = None
    source_round_id: int | None = Field(default=None, ge=0)
    source_findings: list[str] = Field(default_factory=list)
    source_delta_snapshot: dict[str, Any] | None = None
    target_uncertainty_ids: list[str] = Field(default_factory=list)
    selection_rationale: str | None = None
    planned_changed_factors: list[str] = Field(default_factory=list)
    actual_changed_factors: list[str] = Field(default_factory=list)
    protocol_delta_notes: list[str] = Field(default_factory=list)
    candidate_id: str | None = None
    executed_experiment_id: str | None = None
    actual_config_snapshot: dict[str, Any] | None = None
    approved_candidate_id: str | None = None
    next_round_decision: Literal["continue", "adjust", "stop"] | None = None
    decision_summary: str | None = None
    human_feedback: str | None = None
    gating_ready: bool = False
    closure_checklist: list[ClosureChecklistItem] = Field(default_factory=list)
    tested_hypotheses: list[str] = Field(default_factory=list)
    target_uncertainties: list[str] = Field(default_factory=list)
    unresolved_uncertainties: list[str] = Field(default_factory=list)
    highlighted_hypotheses: list[str] = Field(default_factory=list)
    scientific_findings: list[str] = Field(default_factory=list)
    three_layer_conclusion: ThreeLayerConclusion | None = None
    failure_reason: str | None = None
    metrics_snapshot: PerformanceMetrics | None = None
    round_snapshot_path: str | None = None
    iterative_validations: list[dict[str, Any]] = Field(default_factory=list)
    iteration_input_sources: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class RoundHistoryState(ShadowBaseModel):
    task_id: str = Field(min_length=1)
    current_round: int = Field(ge=0)
    last_updated: datetime
    entries: list[RoundHistoryEntry] = Field(default_factory=list)

    @field_validator("entries")
    @classmethod
    def unique_round_entries(cls, value: list[RoundHistoryEntry]) -> list[RoundHistoryEntry]:
        ids = [item.round_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("RoundHistoryState.entries contains duplicate round_id")
        return value


class FailureRecord(ShadowBaseModel):
    failure_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    phase: str = Field(min_length=1)
    step: str = Field(min_length=1)
    experiment_id: str | None = None
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    traceback_excerpt: str | None = None
    can_continue: bool = True
    recorded_at: datetime = Field(default_factory=datetime.now)


class FailureHistoryState(ShadowBaseModel):
    task_id: str = Field(min_length=1)
    current_round: int = Field(ge=0)
    last_updated: datetime
    records: list[FailureRecord] = Field(default_factory=list)


class MetricsTimelineEntry(ShadowBaseModel):
    round_id: int = Field(ge=0)
    experiment_id: str | None = None
    execution_status: ExecutionStatus
    baseline_rmse: float | None = None
    treatment_rmse: float | None = None
    baseline_pearson_r: float | None = None
    treatment_pearson_r: float | None = None
    delta_rmse: float | None = None
    delta_pearson_r: float | None = None
    stable: bool | None = None
    support_mean: float | None = None
    support_max: float | None = None
    unresolved_uncertainty_count: int = Field(default=0, ge=0)
    recorded_at: datetime = Field(default_factory=datetime.now)


class MetricsTimelineState(ShadowBaseModel):
    task_id: str = Field(min_length=1)
    current_round: int = Field(ge=0)
    last_updated: datetime
    entries: list[MetricsTimelineEntry] = Field(default_factory=list)


class ProcessStep(ShadowBaseModel):
    name: str = Field(min_length=1)
    status: StepStatus
    requires_user_approval: bool = False
    notes: str | None = None


class ProcessPhase(ShadowBaseModel):
    status: StepStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    notes: str | None = None
    steps: dict[str, ProcessStep] = Field(default_factory=dict)


class UserSettings(ShadowBaseModel):
    auto_continue: bool = False
    stop_at_each_step: bool = True
    approval_required: list[str] = Field(default_factory=list)


class UserRequests(ShadowBaseModel):
    stop_requested: bool = False
    stop_at_phase: str | None = None
    pause_reason: str | None = None


class ProcessState(ShadowBaseModel):
    task_id: str = Field(min_length=1)
    current_round: int = Field(ge=0)
    current_stage: str = Field(min_length=1)
    current_phase: str | None = None
    current_step: str | None = None
    progress_percentage: int | None = Field(default=None, ge=0, le=100)
    started_at: datetime
    phase_sequence: list[str] = Field(default_factory=list)
    phases: dict[str, ProcessPhase] = Field(default_factory=dict)
    steps: dict[str, ProcessStep] = Field(default_factory=dict)
    user_settings: UserSettings = Field(default_factory=UserSettings)
    user_requests: UserRequests = Field(default_factory=UserRequests)


class DecisionEntry(ShadowBaseModel):
    decision_id: str = Field(default_factory=lambda: f"D{int(datetime.now().timestamp() * 1000)}", min_length=1)
    timestamp: datetime
    round_id: int | None = Field(default=None, ge=0)
    phase: str | None = None
    step: str | None = None
    decision_type: DecisionType
    made_by: Literal["human_pi", "system", "hypothesis_proposer_llm", "scientific_questioner_llm"]
    summary: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)


class HumanFeedbackEntry(ShadowBaseModel):
    feedback_id: str = Field(min_length=1)
    timestamp: datetime
    phase: str = Field(min_length=1)
    round_number: int = Field(ge=0)
    feedback_type: str = Field(min_length=1)
    content: str = Field(min_length=1)
    implemented_in_round: int | None = Field(default=None, ge=0)
    status: str = Field(default="recorded", min_length=1)
    related_experiment_id: str | None = None
    linked_reasoning_trace_ids: list[str] = Field(default_factory=list)
    context_summary: str | None = None


class StopHistoryEntry(ShadowBaseModel):
    stop_id: str = Field(min_length=1)
    timestamp: datetime
    type: Literal["pause", "stop"]
    phase: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    resumed_at: datetime | None = None


class DecisionLog(ShadowBaseModel):
    schema_version: str = "1.0"
    message_type: Literal["decision_log"] = "decision_log"
    task_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=datetime.now)
    decisions: list[DecisionEntry] = Field(default_factory=list)
    human_feedback: list[HumanFeedbackEntry] = Field(default_factory=list)
    stop_history: list[StopHistoryEntry] = Field(default_factory=list)


ExperimentContrast.model_rebuild()
ExperimentDesign.model_rebuild()
