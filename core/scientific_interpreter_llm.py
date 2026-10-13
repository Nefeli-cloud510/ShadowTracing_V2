from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from core.llm_gateway import LLMGateway
from core.planner_unified import ProgrammaticPlannerOutputBuilder
from core.unified_schema import (
    CandidateExperimentSet,
    ConclusionChartAnalysis,
    ConclusionDataLayer,
    ConclusionNextRoundSuggestion,
    EvaluationResult,
    InterpretationEnhancement,
    ReasoningPlannerInput,
)


class ChartAnalysisOutput(BaseModel):
    chart_name: str = ""
    chart_role: str = ""
    description: str = ""
    key_observations: list[str] = Field(default_factory=list)
    anomaly_or_insight: str | None = None


class NextRoundSuggestionOutput(BaseModel):
    evidence_summary: str | None = None
    resolved_uncertainties_this_round: list[str] = Field(default_factory=list)
    unresolved_uncertainties_todo: list[str] = Field(default_factory=list)
    remaining_uncertainty_analysis: str | None = None
    recommendation: str | None = None
    pi_decision_advice: str | None = None
    experiment_design_advice: str | None = None
    hypothesis_space_advice: str | None = None
    notes: list[str] = Field(default_factory=list)


class ScientificInterpreterLLMResponse(BaseModel):
    target_hypothesis_id: str | None = None
    interpretation: str = Field(min_length=1)
    related_uncertainty_ids: list[str] = Field(default_factory=list)
    impact_direction: str = Field(default="clarifies")
    impact_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    uncertainty_priority_action: str = Field(default="maintain")
    suggested_state_note: str | None = None
    rmse_attribution: str | None = None
    pearson_attribution: str | None = None
    skill_delta_meaning: str | None = None
    anomaly_identification: str | None = None
    next_focus: str | None = None
    conclusion_text: str | None = None
    path_answer: str | None = None
    hypothesis_layer_summary: str | None = None
    chart_analyses: list[ChartAnalysisOutput] = Field(default_factory=list)
    comparison_analysis: str | None = None
    overall_summary: str | None = None
    next_round_suggestion: NextRoundSuggestionOutput | None = None


class InterpreterRoundAnalysis(BaseModel):
    """Combined output used by the round-report four-layer writer."""

    enhancements: list[InterpretationEnhancement]
    data_layer: ConclusionDataLayer | None = None
    conclusion_text: str | None = None
    path_answer: str | None = None
    hypothesis_layer_summary: str | None = None
    overall_summary: str | None = None
    next_round_suggestion: ConclusionNextRoundSuggestion | None = None


class ScientificInterpreterLLM:
    """LLM scaffold for turning evaluation context into interpretation enhancements."""

    SYSTEM_PROMPT = """你是“逐影 Shadow Tracing”的科学解释者，负责把真实实验结果写成可供评审复核的轮次报告。

本轮实验包含两个臂：对照组（不含待验证变量的基准实验）与实验组（在对照组设计基础上加入待验证变量的对比实验）。你必须严格区分两组，不得互换名称，也不得把“实验组数值更好”直接等同于“假设被支持”。

你输出的是真实推理文本，不是模板填空。必须基于给定的指标、假设树、不确定性、实验历史与图表证据逐项分析，不能虚构不存在的 hypothesis_id 或 uncertainty_id，不许写“继续核对”“需要进一步分析”这类没有信息量的套话。

要求：
1. 数据层：对 RMSE、Pearson r 做方向与幅度归因，指出异常/不一致现象，给出下一轮应优先核查的数据特征；逐图分析四张真实图表（对照组时序、实验组时序、对照组散点、实验组散点），并做两组对比分析。
2. 假设层：结合程序给出的方向/幅度匹配、支持度变化与实验历史，给出综合解释，区分“模型拟合更好”与“假设获得支持”。
3. 科学问题层：回答主科学问题，并回答物理传递路径问题。
4. 下一轮建议：先标明“本轮已解决的不确定性”和“仍需解决的不确定性”；建议必须基于当前假设树状态，已被强支持的假设不再重复验证，而转向其剩余子问题；给出可执行但不过度设计的实验建议与对人工 PI 的决策意见。
5. 总体总结：用 3-5 句话概括本轮数据科学结论、假设层结论与下一轮建议。
任何叙述文本不得出现 H_*、U_*、status=、narrowed=、before_span= 等内部机器标识；提及假设用 H1..H5 或自然语言名称，提及不确定性用问题原文，内部 ID 只允许出现在指定 id 字段。
如果图表以图像形式提供，必须真正观察图像；如果退化为文本摘要，则基于摘要数值分析并说明通道退化，不得假装看到了图像。"""

    def __init__(
        self,
        *,
        gateway: LLMGateway | None = None,
        baseline_builder: ProgrammaticPlannerOutputBuilder | None = None,
    ) -> None:
        self.gateway = gateway or LLMGateway()
        self.baseline_builder = baseline_builder or ProgrammaticPlannerOutputBuilder()

    def build_interpretations(
        self,
        *,
        planner_input: ReasoningPlannerInput,
        candidate_plan: CandidateExperimentSet,
    ) -> list[InterpretationEnhancement]:
        return self.build_round_analysis(
            planner_input=planner_input,
            candidate_plan=candidate_plan,
        ).enhancements

    def build_round_analysis(
        self,
        *,
        planner_input: ReasoningPlannerInput,
        candidate_plan: CandidateExperimentSet,
        evaluation: EvaluationResult | None = None,
    ) -> InterpreterRoundAnalysis:
        baseline_output = self.baseline_builder.build(
            planner_input=planner_input,
            candidate_plan=candidate_plan,
        )
        baseline = baseline_output.interpretation_enhancements
        chart_context = self._build_chart_context(evaluation)
        image_paths = [item["path"] for item in chart_context]

        response: ScientificInterpreterLLMResponse | None = None
        if image_paths:
            try:
                response = self.gateway.generate_structured(
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=self._build_report_prompt(
                        planner_input=planner_input,
                        candidate_plan=candidate_plan,
                        chart_context=chart_context,
                        image_mode=True,
                    ),
                    response_model=ScientificInterpreterLLMResponse,
                    image_paths=image_paths,
                    fallback_factory=None,
                    role="科学解释者llm",
                )
            except Exception:
                # Vision may be unsupported by the active model; retry text-only
                # with program-derived chart summaries. Never silently use templates.
                response = self.gateway.generate_structured(
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=self._build_report_prompt(
                        planner_input=planner_input,
                        candidate_plan=candidate_plan,
                        chart_context=chart_context,
                        image_mode=False,
                    ),
                    response_model=ScientificInterpreterLLMResponse,
                    image_paths=None,
                    fallback_factory=None,
                    role="科学解释者llm",
                )
        if response is None:
            response = self.gateway.generate_structured(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=self._build_report_prompt(
                    planner_input=planner_input,
                    candidate_plan=candidate_plan,
                    chart_context=chart_context,
                    image_mode=False,
                ),
                response_model=ScientificInterpreterLLMResponse,
                image_paths=None,
                fallback_factory=None,
                role="科学解释者llm",
            )
        return InterpreterRoundAnalysis(
            enhancements=self._build_enhancement(planner_input, baseline, response),
            data_layer=self._build_data_layer(response, chart_context),
            conclusion_text=response.conclusion_text,
            path_answer=response.path_answer,
            hypothesis_layer_summary=response.hypothesis_layer_summary,
            overall_summary=response.overall_summary,
            next_round_suggestion=_build_next_round_suggestion(response.next_round_suggestion),
        )

    def _build_enhancement(
        self,
        planner_input: ReasoningPlannerInput,
        baseline: list[InterpretationEnhancement],
        response: ScientificInterpreterLLMResponse,
    ) -> list[InterpretationEnhancement]:
        llm_enhancement = InterpretationEnhancement(
            enhancement_id="LIE_INT_001",
            target_hypothesis_id=_resolve_hypothesis_id(planner_input, response.target_hypothesis_id),
            interpretation=response.interpretation,
            evidence_basis=planner_input.evaluation_summary.key_findings[:2],
            related_uncertainties=_resolve_uncertainty_ids(
                planner_input,
                response.related_uncertainty_ids,
            ),
            suggested_state_note=response.suggested_state_note,
            impact_direction=_normalize_impact_direction(response.impact_direction),
            impact_strength=response.impact_strength,
            confidence=response.confidence,
            uncertainty_priority_action=_normalize_priority_action(
                response.uncertainty_priority_action
            ),
        )
        return baseline[:1] + [
            llm_enhancement
        ]

    def _build_data_layer(
        self,
        response: ScientificInterpreterLLMResponse,
        chart_context: list[dict[str, str]],
    ) -> ConclusionDataLayer | None:
        path_by_role = {item["role"]: item["path"] for item in chart_context}
        chart_analyses: list[ConclusionChartAnalysis] = []
        for item in response.chart_analyses:
            role = item.chart_role
            if role not in path_by_role:
                continue
            chart_analyses.append(
                ConclusionChartAnalysis(
                    chart_name=item.chart_name,
                    chart_role=role,
                    chart_image_path=path_by_role[role],
                    description=item.description,
                    key_observations=item.key_observations,
                    anomaly_or_insight=item.anomaly_or_insight,
                )
            )
        anomalies = [item for item in [response.anomaly_identification] if item]
        anomalies.extend(
            insight
            for item in chart_analyses
            if item.anomaly_or_insight
            for insight in [item.anomaly_or_insight]
            if insight not in anomalies
        )
        if not any(
            [
                response.rmse_attribution,
                response.pearson_attribution,
                response.skill_delta_meaning,
                anomalies,
                response.next_focus,
                chart_analyses,
                response.comparison_analysis,
            ]
        ):
            return None
        return ConclusionDataLayer(
            rmse_attribution=response.rmse_attribution,
            pearson_attribution=response.pearson_attribution,
            skill_delta_meaning=response.skill_delta_meaning,
            anomalies=anomalies,
            next_focus=response.next_focus,
            chart_analyses=chart_analyses,
            comparison_analysis=response.comparison_analysis,
        )

    @staticmethod
    def _build_report_prompt(
        *,
        planner_input: ReasoningPlannerInput,
        candidate_plan: CandidateExperimentSet,
        chart_context: list[dict[str, str]],
        image_mode: bool,
    ) -> str:
        chart_lines: list[str] = []
        for item in chart_context:
            line = f"- {item['role']}：{item['path']}"
            if not image_mode:
                line += f"\n  数值摘要：{item['summary']}"
            chart_lines.append(line)
        mode_note = (
            "本次以真实图像形式提供 4 张图表，必须逐张观察并分析。"
            if image_mode
            else "本次视觉通道不可用，图表以程序统计的数值摘要提供；请基于摘要分析并保持诚实，不要声称看到了图像。"
        )
        return (
            "请结合以下上下文输出完整轮次报告解释。\n"
            f"context={json.dumps(planner_input.to_scientific_interpreter_payload(), ensure_ascii=False)}\n"
            f"candidate_plan={json.dumps(candidate_plan.model_dump(mode='json', exclude_none=True), ensure_ascii=False)[:4000]}\n"
            f"图表清单：\n" + "\n".join(chart_lines) + "\n"
            f"说明：{mode_note}\n"
            "输出字段必须包含：rmse_attribution、pearson_attribution、skill_delta_meaning、"
            "anomaly_identification、next_focus、hypothesis_layer_summary、conclusion_text、path_answer、"
            "chart_analyses（四项，chart_role 分别为 baseline_timeseries / treatment_timeseries / baseline_scatter / treatment_scatter）、"
            "comparison_analysis、overall_summary、next_round_suggestion。"
        )

    @staticmethod
    def _build_chart_context(
        evaluation: EvaluationResult | None,
    ) -> list[dict[str, str]]:
        if evaluation is None:
            return []
        wanted: dict[str, str] = {}
        for artifact in evaluation.visualizations:
            name = artifact.name.lower()
            path = artifact.path
            if not (name.endswith(".png") and ("timeseries" in name or "scatter" in name)):
                continue
            label = "baseline" if ("_baseline_" in name or "/baseline/" in path) else None
            if label is None:
                label = "treatment" if ("_treatment_" in name or "/treatment/" in path) else None
            if label is None:
                continue
            kind = "timeseries" if "timeseries" in name else "scatter"
            role = f"{label}_{kind}"
            if role not in wanted:
                wanted[role] = path
        results: list[dict[str, str]] = []
        for role in (
            "baseline_timeseries",
            "treatment_timeseries",
            "baseline_scatter",
            "treatment_scatter",
        ):
            path = wanted.get(role)
            if not path:
                continue
            results.append(
                {
                    "role": role,
                    "path": ScientificInterpreterLLM._resolve_relative_path(path),
                    "summary": ScientificInterpreterLLM._chart_text_summary(path),
                }
            )
        return results

    @staticmethod
    def _resolve_relative_path(path: str) -> str:
        candidate = Path(path)
        if candidate.is_file():
            return str(candidate)
        cwd_candidate = Path.cwd() / path
        if cwd_candidate.is_file():
            return str(cwd_candidate)
        return str(candidate)

    @staticmethod
    def _chart_text_summary(image_path: str) -> str:
        csv_path = ScientificInterpreterLLM._prediction_csv_path(image_path)
        if csv_path is None or not csv_path.is_file():
            return "未找到对应预测 CSV，无法提供数值摘要。"
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
        if not rows:
            return "预测 CSV 为空。"
        times: list[str] = []
        y_true: list[float] = []
        y_pred: list[float] = []
        for row in rows:
            if row.get("time"):
                times.append(row["time"])
            try:
                y_true.append(float(row["y_true"]))
                y_pred.append(float(row["y_pred"]))
            except (TypeError, ValueError):
                continue
        if not y_true:
            return "预测 CSV 未包含可用数值。"
        dates = [_parse_iso_time(item) for item in times if _parse_iso_time(item) is not None]
        gap_count = 0
        gap_ranges: list[str] = []
        for index in range(1, len(dates)):
            diff = dates[index] - dates[index - 1]
            if diff is not None and diff > timedelta(days=1, hours=6):
                gap_count += 1
                if len(gap_ranges) < 3:
                    gap_ranges.append(f"{dates[index - 1].date()}~{dates[index].date()}")
        residuals = [abs(a - b) for a, b in zip(y_true, y_pred)]
        return (
            f"样本数={len(rows)}；时间跨度={times[0]} ~ {times[-1]}；"
            f"日期缺口={gap_count} 处（{'/'.join(gap_ranges) if gap_ranges else '无'}）；"
            f"y_true max/mean/min={max(y_true):.3f}/{sum(y_true) / len(y_true):.3f}/{min(y_true):.3f}；"
            f"y_pred max/mean/min={max(y_pred):.3f}/{sum(y_pred) / len(y_pred):.3f}/{min(y_pred):.3f}；"
            f"绝对残差 max/mean={max(residuals):.3f}/{sum(residuals) / len(residuals):.3f}"
        )

    @staticmethod
    def _prediction_csv_path(image_path: str) -> Path | None:
        chart_path = Path(image_path)
        experiment_dir = chart_path.parent.parent
        label = chart_path.parent.name
        if label not in {"baseline", "treatment"}:
            return None
        experiment_id = experiment_dir.name
        candidate = experiment_dir / "predictions" / f"{experiment_id}_{label}.csv"
        return candidate if candidate.is_file() else None

    @staticmethod
    def _build_fallback_response(
        planner_input: ReasoningPlannerInput,
        baseline: list[InterpretationEnhancement],
    ) -> ScientificInterpreterLLMResponse:
        top_hypothesis = planner_input.active_hypotheses[0] if planner_input.active_hypotheses else None
        top_uncertainty = planner_input.unresolved_uncertainties[0] if planner_input.unresolved_uncertainties else None
        baseline_interpretation = baseline[0].interpretation if baseline else "当前仍需保守解释。"
        return ScientificInterpreterLLMResponse(
            target_hypothesis_id=top_hypothesis.hypothesis_id if top_hypothesis else None,
            interpretation=(
                f"{baseline_interpretation} 同时需要补充目标变量预测效果分析："
                f"关注 Pearson r 相对对照组的变化、RMSE 是否下降，以及当前结果对 Vsw 预测是否形成真实增益。"
            ),
            rmse_attribution=(
                f"RMSE 变化需要结合数据覆盖、特征覆盖与过拟合风险继续归因："
                f"当前 ΔRMSE={_format_delta(planner_input.evaluation_summary.delta_rmse)}，"
                "若训练集与测试集差异明显，指标变化更可能来自覆盖范围而非新增特征。"
            ),
            pearson_attribution=(
                f"Pearson r 反映整体趋势捕捉能力："
                f"当前 ΔPearson r={_format_delta(planner_input.evaluation_summary.delta_pearson_r)}，"
                "若改善较小，说明新增条件主要改变误差分布而非趋势相关性。"
            ),
            skill_delta_meaning=(
                "该增量表示实验组相对对照组的预测技能变化；"
                "只有同时满足幅度稳定与方向一致时，才具有实际预测意义。"
            ),
            anomaly_identification=(
                "需要继续核对 RMSE 与 Pearson r 是否出现方向不一致，"
                "并检查数据缺口、缺失对齐与异常样本是否集中在特定时间窗。"
            ),
            next_focus=(
                "下一轮优先围绕数据覆盖与路径变量的协同作用做区分实验，"
                "先排除数据质量问题再更新假设支持度。"
            ),
            conclusion_text=(
                "本轮实验结果需要先放在数据层做归因：当前指标增量尚未形成稳定的科学判断，"
                "建议结合样本覆盖与稳健性检验后再决定假设是否获得支持。"
            ),
            related_uncertainty_ids=[top_uncertainty.uncertainty_id] if top_uncertainty else [],
            impact_direction="supports"
            if planner_input.evaluation_summary.delta_pearson_r and planner_input.evaluation_summary.delta_pearson_r > 0
            else "clarifies",
            impact_strength=0.55,
            confidence=0.68,
            uncertainty_priority_action="maintain",
            suggested_state_note="scientific_interpreter_llm fallback",
        )


def _format_delta(value: float | None) -> str:
    if value is None:
        return "未取得"
    return f"{value:+.4f}"


def _resolve_hypothesis_id(
    planner_input: ReasoningPlannerInput,
    requested_id: str | None,
) -> str | None:
    valid_ids = {item.hypothesis_id for item in planner_input.active_hypotheses}
    if requested_id in valid_ids:
        return requested_id
    return planner_input.active_hypotheses[0].hypothesis_id if planner_input.active_hypotheses else None


def _resolve_uncertainty_ids(
    planner_input: ReasoningPlannerInput,
    requested_ids: list[str],
) -> list[str]:
    valid_ids = {item.uncertainty_id for item in planner_input.unresolved_uncertainties}
    resolved = [item for item in requested_ids if item in valid_ids]
    if resolved:
        return resolved
    return [item.uncertainty_id for item in planner_input.unresolved_uncertainties[:2]]


def _normalize_impact_direction(direction: str) -> str:
    if direction in {"supports", "weakens", "clarifies"}:
        return direction
    return "clarifies"


def _normalize_priority_action(action: str) -> str:
    if action in {"increase", "decrease", "maintain"}:
        return action
    return "maintain"


def _parse_iso_time(value: str) -> datetime | None:
    try:
        text = value.strip().replace("Z", "+00:00")
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _build_next_round_suggestion(
    output: NextRoundSuggestionOutput | None,
) -> ConclusionNextRoundSuggestion | None:
    if output is None:
        return None
    recommendation = output.recommendation
    if recommendation not in {"continue", "adjust", "stop"}:
        recommendation = None
    return ConclusionNextRoundSuggestion(
        evidence_summary=output.evidence_summary,
        resolved_uncertainties_this_round=output.resolved_uncertainties_this_round[:10],
        unresolved_uncertainties_todo=output.unresolved_uncertainties_todo[:10],
        remaining_uncertainty_analysis=output.remaining_uncertainty_analysis,
        recommendation=recommendation,
        pi_decision_advice=output.pi_decision_advice,
        experiment_design_advice=output.experiment_design_advice,
        hypothesis_space_advice=output.hypothesis_space_advice,
        notes=output.notes[:10],
    )
