from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel, Field

from core.display_text_cleaner import (
    DISPLAY_UNCERTAINTY_FALLBACK,
    clean_llm_text,
    deep_clean_text,
)
from core.experiment_tools.registry import (
    ALLOWED_MODEL_FAMILIES,
    get_tool,
    normalize_model_family,
    registered_tool_summaries,
)
from core.llm_gateway import LLMGateway
from core.prompt_rules import ELASTIC_NET_EXECUTION_RULE
from core.runtime_config import get_llm_model_for_role
from core.unified_schema import (
    CandidateExperiment,
    ContrastSpec,
    ExperimentStep,
    ProtocolRefinementSuggestion,
    RunSpec,
    ScientificTask,
)
from core.variable_semantic_service import VariableSemanticService


class ExperimentPlannerLLMResponse(BaseModel):
    rationale: str = Field(min_length=1)
    suggested_model_parameters: dict[str, Any] = Field(default_factory=dict)
    suggested_feature_focus: list[str] = Field(default_factory=list)
    protocol_notes: list[str] = Field(default_factory=list)
    extra_steps: list[dict[str, Any]] = Field(default_factory=list)


class CandidateProseLLMResponse(BaseModel):
    purpose: str = Field(min_length=1)
    distinguishing_insight: str = Field(min_length=1)
    value_analysis: str = Field(min_length=1)


class CandidateDesignLLMResponse(BaseModel):
    focus_display: str | None = Field(default=None)
    baseline_display: list[str] = Field(default_factory=list)
    treatment_display: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    experiment_type: str | None = Field(default=None)
    model_families: list[str] = Field(default_factory=list)


def _apply_model_sensitivity_design(design: ExperimentDesign) -> None:
    """Compile the shared Ridge/ElasticNet sensitivity structure into the design."""

    families: list[str] = []
    for name in design.model_families or ["ridge", "elasticnet"]:
        normalized = normalize_model_family(name)
        if normalized and normalized not in families:
            families.append(normalized)
    if len(families) < 2:
        families = ["ridge", "elasticnet"]
    design.model_families = families
    design.runs_spec = []
    for family in families:
        design.runs_spec.extend(
            [
                RunSpec(
                    run_id=f"{family}_baseline",
                    label=f"{family} 对照组",
                    feature_columns=list(design.control),
                    model_family=family,
                    retrain=True,
                ),
                RunSpec(
                    run_id=f"{family}_treatment",
                    label=f"{family} 实验组",
                    feature_columns=list(design.treatment),
                    model_family=family,
                    retrain=True,
                ),
            ]
        )
    design.contrasts_spec = [
        ContrastSpec(
            contrast_id=f"gain_{family}",
            name=f"{family} 家族内日影增量",
            kind="incremental_gain",
            run_ids=[f"{family}_baseline", f"{family}_treatment"],
            direction="mixed",
            note="只比较同一模型家族内部加日影前后；禁止跨家族错误归因。",
        )
        for family in families
    ]


class CandidateExperimentDesignerLLM:
    """Let the LLM choose the tested variables and 对照组/实验组 arms.

    The programmatic generator produces a schema-correct first draft; this role
    is responsible for the actual scientific variable selection, so candidates
    stop being produced by a fixed "primary is always the 对照组" rule.
    """

    @classmethod
    def _tool_catalog_text(cls) -> str:
        lines = []
        for item in registered_tool_summaries():
            availability = "可选" if item.get("available") else "当前不可选"
            reasons = (
                f"；限制：{'、'.join(item['refusal_reasons'])}"
                if item.get("refusal_reasons")
                else ""
            )
            lines.append(
                f"- {item['experiment_type']}（{availability}）："
                f"{item.get('applicability', '')}{reasons}"
            )
        return "可用实验工具目录（必须从登记类型中选择并输出 experiment_type）：\n" + "\n".join(lines)

    @classmethod
    def build_system_prompt(cls) -> str:
        return (
            """你是“逐影 Shadow Tracing”的候选实验设计者，负责为每条科学不确定性确定实验类型、变量选择与对照组/实验组结构。

你的选择必须遵守以下规则：
1. 默认实验类型是 incremental_prediction（增量预测实验）：对照组包含常规日地环境变量且不含待验证焦点变量，实验组在对照组全部变量基础上加入待验证焦点变量；
2. 当前执行层已接通 Ridge 与 ElasticNet 两个模型家族。默认增量实验使用 ElasticNet；
3. 只有当科学不确定性明确讨论“模型家族、线性/非线性、算法依赖、模型稳定性”时才选择 model_sensitivity（模型敏感性实验）。此时实验组变量仍保持“对照组+焦点变量”，模型家族必须同时包含 Ridge 与 ElasticNet，用于比较加日影前后的增益是否依赖模型；
4. baseline_benchmark 是单臂基准实验：只给出 treatment_display 作为被测基准特征组合，baseline_display 留空，不设置对照组/实验组对比；
5. 只能使用 data_dictionary 中出现的真实物理量展示名，禁止发明变量、虚构数据来源；
6. 待验证焦点变量必须出现在实验组变量列表中，且不能出现在对照组；
7. 对照组与实验组必须存在明确变量差异，且差异必须只来自待验证焦点变量；
8. 不要选择因变量作为自变量，不要选择时间列；
9. 若不确定性讨论的是某个太阳风特征（如行星际磁场、地磁指数）对主特征的影响，仍要优先检验主特征在固定常规对照组下的增量贡献；
10. 未登记或当前不可选的实验类型只能明确拒绝，禁止改标签后按 incremental_prediction 执行。

"""
            + cls._tool_catalog_text()
            + """

输出结构必须为：
{"focus_display": "<待验证焦点变量展示名>", "baseline_display": ["..."], "treatment_display": ["...", ...], "rationale": "<方向性设计理由>", "experiment_type": "<incremental_prediction|model_sensitivity|baseline_benchmark>", "model_families": ["ridge", "elasticnet"]}

若选择 incremental_prediction，model_families 可省略；若选择 model_sensitivity，model_families 必须为 ["ridge", "elasticnet"]。只输出严格 JSON，不要夹带解释。"""
        )

    def __init__(
        self,
        *,
        gateway: LLMGateway | None = None,
        max_attempts: int = 2,
    ) -> None:
        self.gateway = gateway or LLMGateway(
            model=get_llm_model_for_role("experiment_designer"),
            allow_fallback=False,
        )
        self.max_attempts = max_attempts

    def design(
        self,
        *,
        candidate: CandidateExperiment,
        task: ScientificTask,
        uncertainty_question: str | None,
        uncertainty_description: str | None,
        semantic_service: VariableSemanticService,
        node_index: dict[str, object],
        primary_x: str,
        round_id: int | None = None,
    ) -> bool:
        """Rewrite candidate.design using LLM-selected variables.

        Returns False when the LLM response is invalid or cannot improve on the
        programmatic draft. The existing draft is then kept as the fallback.
        """
        if not semantic_service.entries:
            return False
        target = candidate.design.target
        display_target = semantic_service.to_display(target)
        display_block = semantic_service.build_display_block([target])
        hypothesis_lines: list[str] = []
        for hypothesis_id in candidate.tested_hypotheses:
            node = node_index.get(hypothesis_id)
            label = (
                getattr(node, "display_hypothesis_id", None)
                or (f"H{node.level}" if node is not None and getattr(node, "level", None) else "")
                or "相关假设"
            )
            statement = getattr(node, "statement", None) or "未检索到假设陈述"
            hypothesis_lines.append(f"- {label}：{statement}")

        user_prompt = (
            f"科研问题={task.payload.research_question.text}\n"
            f"轮次={round_id if round_id is not None else '当前轮'}\n"
            f"候选实验={candidate.purpose or '本轮候选实验'}\n"
            f"实验类型={candidate.experiment_type or candidate.type}\n"
            f"因变量展示名={display_target}\n"
            f"科学不确定性问题={uncertainty_question or candidate.scientific_question or '无'}\n"
            f"科学不确定性描述={uncertainty_description or '无'}\n"
            f"被验证假设:\n{chr(10).join(hypothesis_lines) or '（无）'}\n"
            f"程序底稿：对照组={semantic_service.display_list(candidate.design.control) or '未配置'}；"
            f"实验组={semantic_service.display_list(candidate.design.treatment) or '未配置'}；"
            f"设计焦点={semantic_service.to_display(candidate.design.design_focus or primary_x)}\n"
            f"模型家族={'、'.join(candidate.design.model_families) if candidate.design.model_families else '未配置'}\n"
            f"data_dictionary={display_block}\n"
            "请只输出 JSON，不要夹带解释。"
        )

        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            try:
                response = self.gateway.generate_structured(
                    system_prompt=self.build_system_prompt(),
                    user_prompt=user_prompt,
                    response_model=CandidateDesignLLMResponse,
                    fallback_factory=None,
                    temperature=0.3,
                )
                return self._apply_response(
                    candidate=candidate,
                    response=response,
                    primary_x=primary_x,
                    semantic_service=semantic_service,
                )
            except Exception as exc:
                last_error = exc
        candidate.design.notes.append(
            f"llm_candidate_design_failed:{type(last_error).__name__ if last_error else 'unknown'}"
        )
        return False

    _ALLOWED_EXPERIMENT_TYPES = {
        "incremental_prediction",
        "model_sensitivity",
        "baseline_benchmark",
    }

    def _apply_response(
        self,
        *,
        candidate: CandidateExperiment,
        response: CandidateDesignLLMResponse,
        primary_x: str,
        semantic_service: VariableSemanticService,
    ) -> bool:
        design = candidate.design
        target = design.target
        raw_names = set(semantic_service.raw_names())
        control = semantic_service.raw_list(response.baseline_display)
        treatment = semantic_service.raw_list(response.treatment_display)
        focus = semantic_service.to_raw(response.focus_display)

        control = list(dict.fromkeys(item for item in control if item in raw_names))
        treatment = list(dict.fromkeys(item for item in treatment if item in raw_names))
        focus = focus if focus in raw_names else None

        requested_type = str(
            (
                response.experiment_type
                or candidate.experiment_type
                or candidate.type
                or "incremental_prediction"
            ).strip()
        )
        tool = get_tool(requested_type)
        if requested_type not in self._ALLOWED_EXPERIMENT_TYPES or tool is None:
            raise ValueError(
                f"LLM 请求的实验类型 {requested_type!r} 未登记，"
                "禁止改标签后按普通增量实验执行。"
            )
        if not tool.available:
            refusal = "；".join(tool.refusal_reasons) or "执行器尚未接通"
            raise ValueError(
                f"实验类型 {requested_type!r} 当前不可选（{refusal}），"
                "不允许改标签后按增量实验执行。"
            )
        if (
            candidate.experiment_type in {"model_sensitivity", "baseline_benchmark"}
            and requested_type != candidate.experiment_type
        ):
            raise ValueError(
                f"候选已登记为 {candidate.experiment_type}，"
                f"LLM 不能改标为 {requested_type} 后执行。"
            )

        if requested_type == "baseline_benchmark":
            if control:
                raise ValueError("基准对照实验不能设置对照组变量。")
            if not treatment:
                raise ValueError("基准对照实验至少需要一组单臂基准特征。")
            if focus is None:
                raise ValueError(
                    "基准对照实验的焦点变量必须来自真实变量库。"
                )
            if focus == target:
                raise ValueError("待验证焦点变量不能是因变量。")
            if focus not in treatment:
                raise ValueError("基准对照实验的特征组合必须包含焦点变量。")
        else:
            if focus is None:
                raise ValueError("LLM 选出的待验证焦点变量不在真实变量库中。")
            if focus == target:
                raise ValueError("待验证焦点变量不能是因变量。")
            if focus not in treatment:
                raise ValueError("实验组必须包含待验证焦点变量。")
            if focus in control:
                raise ValueError("对照组不得包含待验证焦点变量。")
            if not control or not treatment:
                raise ValueError("对照组与实验组都不能为空。")
            if set(control) == set(treatment):
                raise ValueError("对照组与实验组必须存在变量差异。")

        if requested_type == "model_sensitivity":
            families: list[str] = []
            for name in response.model_families or design.model_families or ["ridge", "elasticnet"]:
                normalized = normalize_model_family(name)
                if normalized and normalized not in families:
                    families.append(normalized)
            if len(families) < 2:
                families = ["ridge", "elasticnet"]
            if not all(family in ALLOWED_MODEL_FAMILIES for family in families):
                raise ValueError("模型敏感性实验仅允许 Ridge 与 ElasticNet 两个模型家族。")
            design.model_families = families
            design.control = control
            design.treatment = treatment
            _apply_model_sensitivity_design(design)
            candidate.experiment_type = "model_sensitivity"
        else:
            design.control = control
            design.treatment = treatment
            candidate.experiment_type = requested_type

        old_focus = design.design_focus
        if old_focus and old_focus != focus and old_focus in design.lags:
            lag_values = design.lags.pop(old_focus)
            if focus:
                design.lags[focus] = lag_values
        design.design_focus = focus
        design.notes = [
            note
            for note in design.notes
            if not note.startswith("design_focus:") and note != "llm_candidate_design_failed"
        ]
        design.notes.append(f"design_focus:{focus}")
        design.notes.append(
            f"llm_candidate_design:model={getattr(self.gateway, 'model', 'unknown')}"
        )
        if requested_type == "baseline_benchmark":
            candidate.type = "baseline_benchmark"
            design.notes.append("experiment_mode:baseline_single_arm")
        design.display_design_focus = semantic_service.to_display(focus) if focus else None
        design.display_control = semantic_service.display_list(design.control)
        design.display_treatment = semantic_service.display_list(design.treatment)
        if response.rationale:
            design.notes.append(f"llm_candidate_design_rationale:{response.rationale[:500]}")
        return True


class CandidateExperimentWriterLLM:
    """Write natural-language purpose, distinguishing insight, and value analysis
    for candidates."""

    ALLOWED_TECHNICAL_TOKENS = {
        "alpha",
        "l1_ratio",
        "Pearson",
        "Pearson_r",
        "RMSE",
        "ElasticNet",
        "Elastic_Net",
    }

    SYSTEM_PROMPT = """你是“逐影 Shadow Tracing”的候选实验科学撰稿人。

你接收的是一个已经结构化生成好的候选实验。你的任务不是重新设计实验，也不是输出程序式模板，
而是用严谨、自然、可执行的科学语言撰写三段文字：

1. purpose（实验目的）：说明本轮实验要判断哪个科学问题，实验结论将支持或削弱哪条假设。
2. distinguishing_insight（区分性思路）：措辞必须与用户消息中“实验类型”的运行结构匹配：
   - incremental_prediction：说明对照组（不加待验证变量）与实验组（加待验证变量）的差异，
     如何通过比较预测相关系数 / 均方根误差的增量区分竞争假设；
   - model_sensitivity：说明 Ridge 与 ElasticNet 两个模型家族分别在加入日影前后的
     预测相关系数 / 均方根误差变化，只比较同一家族内部的增量是否一致，不跨家族直接对比；
   - baseline_benchmark：说明单臂基准特征组合的性能快照与科学含义，不使用“对照组 / 实验组”对比句式。
3. value_analysis（实验综合价值数据分析）：只用用户消息中“预计算实验评估”给出的四个指标数值与计算说明，
   分五段自然语言书写，段与段之间用换行符分隔：
   - 信息增益分析：说明当前 IG 值在多大程度上代表着该实验能区分竞争假设；
   - 性能增益分析：说明 PG 的规则含义；若为第一轮或尚无历史实验结果，要明确说明 PG 置 0
     表示本轮未启用、不参与综合价值；
   - 风险分析：结合给出的风险分与 D1-D4 计算说明，明确指出当前哪一项风险最高、
     其数值是多少、为什么它推高了风险，以及整体风险属于什么水平；数值变化时
     措辞必须随之变化，不得在数值不支持时写“主要由...贡献”之类的固定句式；
   - 成本分析：结合给出的成本分与真实 LLM 费用计算说明，写出估算输入/输出 token、
     估算费用（元）、费用上限与 C1/C2/C3 的实际含义；不得写成脱离具体数字的套话；
   - 综合价值判断：综合 IG、PG、Risk、Cost 四项，用一两句话说明该实验是否值得执行及主要原因。

要求：
1. 只能使用 data_dictionary_display_summary 中出现的物理量展示名，严禁发明变量、虚构数据来源或
   设计当前 Ridge / ElasticNet 执行流程无法验证的机制。
2. purpose 不要写成长句复述不确定性 ID；要写成可阅读、可执行的实验目的。
3. distinguishing_insight 不要机械堆砌字段式的程序词，必须按实验类型的运行结构选择措辞：
   双臂增量实验使用“对照组 / 实验组”，模型敏感性实验按家族内增量撰写，单臂基准不使用对比句式。
4. 涉及评价指标时用自然短语（如“预测相关系数、均方根误差”）表述，不要在同一段重复堆叠
   Pearson_r / RMSE 这类字段式名称。
5. 不要在文本中暴露 candidate_id、uncertainty_id、hypothesis_id 等内部代号；假设可写作 H1、H2。
6. 不得虚构任何实验结果或观测数据，只写判断依据、预期与可证伪条件。
7. value_analysis 引用的所有数值必须严格来自“预计算实验评估”，不得自行编造或改写
   （包括 token 数、金额、C1/C2/C3 数值、D1-D4 数值）。
8. value_analysis 不得使用 Markdown 标题、列表符号、JSON 键或带括号的字段名，只用五段连贯的自然语言文本。
9. 只输出严格 JSON：{"purpose": "...", "distinguishing_insight": "...", "value_analysis": "..."}""" + "\n\n" + ELASTIC_NET_EXECUTION_RULE

    EXPECTED_EFFECT_LABELS = {
        "positive": "预期正增益",
        "negative": "预期负增益",
        "near_zero": "预期近零",
        "zero": "预期零增益",
    }

    def __init__(
        self,
        *,
        gateway: LLMGateway | None = None,
        max_attempts: int = 2,
    ) -> None:
        self.gateway = gateway or LLMGateway(
            model=get_llm_model_for_role("experiment_planner"),
            allow_fallback=False,
        )
        self.max_attempts = max_attempts

    def write_all(
        self,
        candidates: list[CandidateExperiment],
        *,
        task: ScientificTask,
        semantic_service: VariableSemanticService,
        node_index: dict[str, object],
        round_id: int | None = None,
        uncertainty_index: dict[str, Any] | None = None,
        max_workers: int = 4,
    ) -> None:
        if not candidates:
            return
        payloads = [
            (
                candidate,
                self._build_user_prompt(
                    task=task,
                    candidate=candidate,
                    semantic_service=semantic_service,
                    node_index=node_index,
                    round_id=round_id,
                    uncertainty_index=uncertainty_index,
                ),
                semantic_service,
                self.gateway,
            )
            for candidate in candidates
        ]
        with ThreadPoolExecutor(max_workers=min(max_workers, len(payloads))) as pool:
            for candidate, response in pool.map(self._write_one, payloads):
                candidate.purpose = response.purpose
                candidate.distinguishing_insight = response.distinguishing_insight
                candidate.value_analysis = response.value_analysis
                if not any(
                    note.startswith("llm_candidate_prose:")
                    for note in candidate.design.notes
                ):
                    candidate.design.notes.append(
                        f"llm_candidate_prose:model={getattr(self.gateway, 'model', 'unknown')}"
                    )

    @staticmethod
    def _write_one(
        payload: tuple[CandidateExperiment, str, VariableSemanticService, LLMGateway],
    ) -> tuple[CandidateExperiment, CandidateProseLLMResponse]:
        candidate, user_prompt, semantic_service, gateway = payload
        candidate_pass = CandidateExperimentWriterLLM(
            gateway=gateway,
        )
        return candidate, candidate_pass._generate(
            candidate=candidate,
            user_prompt=user_prompt,
            semantic_service=semantic_service,
        )

    def _generate(
        self,
        *,
        candidate: CandidateExperiment,
        user_prompt: str,
        semantic_service: VariableSemanticService,
    ) -> CandidateProseLLMResponse:
        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            try:
                response = self.gateway.generate_structured(
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    response_model=CandidateProseLLMResponse,
                    fallback_factory=None,
                    temperature=0.3,
                )
                self._validate_prose(candidate, response, semantic_service)
                return response
            except Exception as exc:
                last_error = exc
        raise last_error or RuntimeError("候选实验 LLM 撰稿失败。")

    def _build_user_prompt(
        self,
        *,
        task: ScientificTask,
        candidate: CandidateExperiment,
        semantic_service: VariableSemanticService,
        node_index: dict[str, object],
        round_id: int | None = None,
        uncertainty_index: dict[str, Any] | None = None,
    ) -> str:
        design = candidate.design
        target = design.display_target or semantic_service.to_display(design.target)
        control = design.display_control or semantic_service.display_list(design.control)
        treatment = design.display_treatment or semantic_service.display_list(design.treatment)
        focus = design.display_design_focus
        if not focus and design.design_focus:
            focus = semantic_service.to_display(design.design_focus)
        focus = focus or "（未指定独立焦点变量）"

        fixed_settings = self._format_fixed_settings(design)
        hypothesis_lines: list[str] = []
        for hypothesis_id in candidate.tested_hypotheses:
            node = node_index.get(hypothesis_id)
            label = ""
            if node is not None:
                label = (
                    getattr(node, "display_hypothesis_id", None)
                    or (f"H{node.level}" if getattr(node, "level", None) else "")
                )
            statement = getattr(node, "statement", None) or "未检索到假设陈述"
            prediction = candidate.hypothesis_predictions.get(hypothesis_id)
            hypothesis_lines.append(
                f"- {label or '相关假设'}：假设陈述={statement}；实验预期={self._format_prediction(prediction)}"
            )

        uncertainty_lines: list[str] = []
        for uncertainty_id in candidate.related_uncertainties:
            record = (uncertainty_index or {}).get(uncertainty_id)
            question = getattr(record, "question", None) if record is not None else None
            uncertainty_lines.append(f"- {question or DISPLAY_UNCERTAINTY_FALLBACK}")

        display_block = semantic_service.build_display_block([design.target]) if semantic_service.entries else ""
        experiment_type = candidate.experiment_type or candidate.type or "incremental_prediction"
        if experiment_type == "baseline_benchmark":
            arms_block = (
                f"单臂基准特征组合（无对照组）={'、'.join(treatment) or '未配置'}\n"
            )
        else:
            arms_block = (
                f"对照组（不加待验证变量）={'、'.join(control) or '未配置'}\n"
                f"实验组（加待验证变量）={'、'.join(treatment) or '未配置'}\n"
            )
        model_families_line = (
            f"模型家族={'、'.join(design.model_families) if design.model_families else '未配置（按实验类型默认）'}\n"
            if experiment_type == "model_sensitivity"
            else ""
        )
        return (
            f"科研问题={task.payload.research_question.text}\n"
            f"轮次={round_id if round_id is not None else '当前轮'}\n"
            f"候选实验={candidate.purpose or '本轮候选实验'}\n"
            f"实验类型={experiment_type}\n"
            f"因变量={target}\n"
            f"{arms_block}"
            f"待验证焦点变量={focus}\n"
            f"{model_families_line}"
            f"固定控制条件={fixed_settings}\n"
            f"被验证假设:\n{chr(10).join(hypothesis_lines) or '（无）'}\n"
            f"上游不确定性:\n{chr(10).join(uncertainty_lines) or '（无）'}\n"
            f"程序底稿_experiment_purpose={candidate.purpose}\n"
            f"程序底稿_distinguishing_insight={candidate.distinguishing_insight or ''}\n"
            f"预计算实验评估:\n{self._format_evaluation(candidate)}\n"
            f"data_dictionary_display_summary={display_block}\n"
            "请只输出 JSON，不要夹带解释。"
        )

    @staticmethod
    def _format_evaluation(candidate: CandidateExperiment) -> str:
        lines = [
            CandidateExperimentWriterLLM._format_estimate(
                candidate.estimated_information_gain,
                "IG（信息增益）",
                0.0,
            ),
            CandidateExperimentWriterLLM._format_estimate(
                candidate.estimated_performance_gain,
                "PG（预期性能增益）",
                0.0,
            ),
            CandidateExperimentWriterLLM._format_estimate(
                candidate.estimated_risk,
                "Risk（风险分）",
                0.5,
            ),
            CandidateExperimentWriterLLM._format_estimate(
                candidate.estimated_cost,
                "Cost（成本分）",
                0.5,
            ),
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_estimate(estimate: Any, label: str, default: float) -> str:
        value = getattr(estimate, "value", None)
        if value is None:
            value = default
        rationale = getattr(estimate, "rationale", None)
        return f"{label}={value:.4f}; 计算说明={rationale or '无'}"

    @staticmethod
    def _format_fixed_settings(design: Any) -> str:
        parts: list[str] = []
        if design.forecast_horizon_days:
            parts.append(f"预测超前 {design.forecast_horizon_days} 天")
        if design.past_lag_days:
            parts.append(f"历史滞后 {design.past_lag_days} 天")
        if design.window_size:
            parts.append(f"时间窗口 {design.window_size} 天")
        if design.control_lag_days and design.control_lag_days != design.treatment_lag_days:
            parts.append(f"对照组滞后 {design.control_lag_days} 天")
        if design.treatment_lag_days and design.treatment_lag_days != design.control_lag_days:
            parts.append(f"实验组滞后 {design.treatment_lag_days} 天")
        return "；".join(parts) or "除所列自变量外保持对照组设置一致"

    @staticmethod
    def _format_prediction(prediction: Any | None) -> str:
        if prediction is None:
            return "实验后按实际 Δ 结果判定"
        expected_effect = prediction.expected_effect if hasattr(prediction, "expected_effect") else None
        label = CandidateExperimentWriterLLM.EXPECTED_EFFECT_LABELS.get(
            str(expected_effect or ""),
            str(expected_effect or "按实验结果判定"),
        )
        expected_range = getattr(prediction, "expected_range", None)
        range_text = ""
        if isinstance(expected_range, list) and len(expected_range) >= 2:
            range_text = f" [{expected_range[0]:.3f}, {expected_range[1]:.3f}]"
        return f"{label}{range_text}"

    @staticmethod
    def _validate_prose(
        candidate: CandidateExperiment,
        response: CandidateProseLLMResponse,
        semantic_service: VariableSemanticService,
    ) -> None:
        purpose = (response.purpose or "").strip()
        insight = (response.distinguishing_insight or "").strip()
        analysis = (response.value_analysis or "").strip()
        if not purpose or not insight:
            raise ValueError(f"候选 {candidate.experiment_id} 的 LLM 文本字段为空。")
        if not analysis:
            raise ValueError(f"候选 {candidate.experiment_id} 的实验综合价值数据分析为空。")
        analysis_blocks = [block.strip() for block in re.split(r"\n+", analysis) if block.strip()]
        if len(analysis_blocks) < 5:
            raise ValueError(
                f"候选 {candidate.experiment_id} 的实验综合价值数据分析不足 5 段"
                f"（当前 {len(analysis_blocks)} 段）。"
            )
        if purpose == (candidate.purpose or "").strip():
            raise ValueError(
                f"候选 {candidate.experiment_id} 的实验目的仍是程序模板原句，禁止采用。"
            )
        combined = f"{purpose} {insight} {analysis}"
        allowed_display = set(
            semantic_service.display_list(
                [*candidate.design.control, *candidate.design.treatment, candidate.design.target]
            )
        )
        if allowed_display and not any(name and name in combined for name in allowed_display):
            raise ValueError(
                f"候选 {candidate.experiment_id} 的 LLM 文本未引用本实验任何真实变量"
                f"（应至少出现：{ '、'.join(sorted(allowed_display)) }）。"
            )
        raw_style_tokens = re.findall(
            r"(?<![A-Za-z0-9_\u4e00-\u9fff])[A-Za-z]{1,8}_[A-Za-z0-9_\-（）()\u4e00-\u9fff]{1,120}(?![A-Za-z0-9_\u4e00-\u9fff])",
            combined,
        )
        known_raw = set(semantic_service.raw_names())
        invalid = [
            token
            for token in raw_style_tokens
            if token not in known_raw
            and token not in CandidateExperimentWriterLLM.ALLOWED_TECHNICAL_TOKENS
        ]
        if invalid:
            raise ValueError(
                f"候选 {candidate.experiment_id} 的 LLM 文本出现变量库之外的字段式名称：{ '、'.join(invalid[:5]) }。"
            )


class ExperimentPlannerLLM:
    """Refine a chosen candidate into a more explicit executable protocol suggestion."""

    SYSTEM_PROMPT = """你是“逐影 Shadow Tracing”的实验规划者。

你接收的是已经通过人工审批候选实验的结构化上下文。你的职责不是绕过程序生成整份协议，
而是在当前 candidate 的基础上给出结构化、可校验、可执行的 refinement 建议。

要求：
1. 不得发明不存在的 candidate_id、hypothesis_id、uncertainty_id
2. 只输出严格 JSON
3. 数值参数只给模型参数建议，不做最终裁决
4. 所有建议必须适合作为 protocol refinement 附加到现有协议上
5. 若 candidate_type 是 baseline_benchmark，则按“单臂基准实验”理解，不要生成对照组/实验组对比建议
6. 若 candidate_type 不是 baseline_benchmark，则必须维持对照组与实验组存在明确变量差异，不能给出无差异对照建议
7. 必须参考 history_context 中的历史轮次结果与失败归因，结合科学经验自动给出更优的模型参数建议；
   建议范围尽量覆盖 window_size、max_lag_day、forecast_horizon_days、past_lag_days、
   test_split_ratio、alpha、l1_ratio，不要简单照抄上一轮""" + "\n\n" + ELASTIC_NET_EXECUTION_RULE

    def __init__(self, *, gateway: LLMGateway | None = None) -> None:
        self.gateway = gateway or LLMGateway()

    def refine(
        self,
        *,
        task: ScientificTask,
        candidate: CandidateExperiment,
        existing_refinements: list[ProtocolRefinementSuggestion] | None = None,
        semantic_service: VariableSemanticService | None = None,
        history_context: str | None = None,
    ) -> ProtocolRefinementSuggestion:
        response = self.gateway.generate_structured(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(
                task=task,
                candidate=candidate,
                existing_refinements=existing_refinements or [],
                semantic_service=semantic_service,
                history_context=history_context,
            ),
            response_model=ExperimentPlannerLLMResponse,
            fallback_factory=lambda: self._fallback(candidate, history_context=history_context),
        )
        semantic_service = semantic_service or VariableSemanticService()
        feature_focus_raw = semantic_service.raw_list(response.suggested_feature_focus)
        return ProtocolRefinementSuggestion(
            suggestion_id="EPL001",
            target_candidate_id=candidate.experiment_id,
            refinement_type="实验规划者llm",
            rationale=response.rationale,
            suggested_model_parameters=_filter_model_parameters(response.suggested_model_parameters),
            suggested_feature_focus=[
                feature
                for feature in feature_focus_raw
                if feature in candidate.design.treatment or feature in candidate.design.control
            ],
            suggested_steps=_coerce_steps(response.extra_steps),
            protocol_notes=["实验规划者llm"] + response.protocol_notes,
        )

    def _build_user_prompt(
        self,
        *,
        task: ScientificTask,
        candidate: CandidateExperiment,
        existing_refinements: list[ProtocolRefinementSuggestion],
        semantic_service: VariableSemanticService | None = None,
        history_context: str | None = None,
    ) -> str:
        display_block = ""
        if semantic_service and semantic_service.entries:
            display_block = (
                "data_dictionary_display_summary="
                f"{semantic_service.build_display_block()}\n"
                "注意：rationale、protocol_notes 等文本变量使用展示名；"
                "suggested_feature_focus 也必须使用展示名。\n"
            )
        return (
            f"scientific_task={task.payload.research_question.text}\n"
            f"candidate_type={candidate.experiment_type or candidate.type}\n"
            f"model_families={'、'.join(candidate.design.model_families) if candidate.design.model_families else '未配置'}\n"
            f"candidate_purpose={candidate.purpose}\n"
            "被验证假设与关联不确定性来自当前闭环落盘的候选实验，必须作为规划依据引用；"
            "输出文本不得复述内部代号。\n"
            f"control={candidate.design.control}\n"
            f"treatment={candidate.design.treatment}\n"
            f"candidate_notes={ '; '.join(clean_llm_text(note) for note in candidate.design.notes) or '无' }\n"
            f"existing_refinements={deep_clean_text([item.model_dump(exclude_none=True) for item in existing_refinements])}\n"
            f"history_context={history_context or '无历史轮次'}\n"
            f"{display_block}"
            "history_context 提供了最近实验的指标、结论与失败归因。请据此判断本轮模型参数是否需要调整：\n"
            "若上轮测试集样本过少可小幅降低 test_split_ratio；缺失记录按行剔除后样本偏少时可减小 past_lag_days；"
            "若超参数明显未调优可给出新的 alpha/l1_ratio。\n"
            "请输出 JSON: rationale, suggested_model_parameters, suggested_feature_focus, protocol_notes, extra_steps。"
        )

    @staticmethod
    def _fallback(
        candidate: CandidateExperiment,
        *,
        history_context: str | None = None,
    ) -> ExperimentPlannerLLMResponse:
        focus = candidate.design.treatment[:2]
        notes = [
            "实验规划者llm：本轮候选实验",
            "llm_planner_requests_pre_run_review",
        ]
        if candidate.related_uncertainties:
            notes.append("实验规划者llm：聚焦该候选关联科学不确定性")
        suggested_parameters: dict[str, Any] = {}
        history_text = history_context or ""
        if (
            "delta_pearson_r=-" in history_text
            or "treatment 未优于 baseline" in history_text
            or "处理组未优于对照组" in history_text
            or "实验组未优于对照组" in history_text
        ):
            suggested_parameters.update(
                {
                    "alpha": 0.12,
                    "l1_ratio": 0.7,
                    "test_split_ratio": 0.25,
                    "window_size": 10,
                    "max_lag_day": 3,
                    "past_lag_days": 10,
                }
            )
            notes.append("llm_planner_auto_tune:history_negative_delta")
        elif "validation" in candidate.type:
            suggested_parameters["alpha"] = 0.22
        extra_steps = [
            {
                "action": "planner_hypothesis_focus",
                "parameters": {
                    "candidate_id": candidate.experiment_id,
                    "tested_hypotheses": candidate.tested_hypotheses,
                },
            }
        ]
        return ExperimentPlannerLLMResponse(
            rationale="针对本轮候选实验，优先明确假设焦点和执行前检查，减少协议歧义。",
            suggested_model_parameters=suggested_parameters,
            suggested_feature_focus=focus,
            protocol_notes=notes,
            extra_steps=extra_steps,
        )


def _filter_model_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "window_size",
        "use_lag_feature",
        "max_lag_day",
        "forecast_horizon_days",
        "past_lag_days",
        "test_split_ratio",
        "alpha",
        "l1_ratio",
        "random_state",
    }
    filtered: dict[str, Any] = {}
    for key, value in parameters.items():
        if key not in allowed:
            continue
        if isinstance(value, (int, float, bool)):
            filtered[key] = value
    return filtered


def _coerce_steps(extra_steps: list[dict[str, Any]]) -> list[ExperimentStep]:
    steps: list[ExperimentStep] = []
    for index, item in enumerate(extra_steps, start=1):
        action = item.get("action")
        if not action:
            continue
        parameters = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
        steps.append(
            ExperimentStep(
                step=index,
                action=str(action),
                parameters=parameters,
            )
        )
    return steps
