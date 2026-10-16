"""Polish structured model-tuning records into natural report language."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from core.llm_gateway import LLMGateway
from core.prompt_rules import ELASTIC_NET_EXECUTION_RULE


PARAMETER_LABELS: dict[str, str] = {
    "window_size": "窗口大小（天）",
    "past_lag_days": "样本滞后（天）",
    "forecast_horizon_days": "预测视野（天）",
    "max_lag_day": "最大滞后构造（天）",
    "test_split_ratio": "迭代评估段比例",
    "alpha": "正则强度 alpha",
    "l1_ratio": "L1 占比 l1_ratio",
    "use_lag_feature": "是否启用滞后特征",
    "random_state": "随机种子",
}


class TuningNarrativeResponse(BaseModel):
    narrative: str = Field(min_length=1, description="自然、专业、可读的中文调参说明")
    parameter_explanation: str = Field(min_length=1, description="对关键模型参数作用的一句话解释")


class TuningNarrativeService:
    """Turn tuning_entries JSON into a natural-language report paragraph.

    The structured tuning record stays in the decision log for audit; the
    polished narrative is what the frontend shows to the PI and reviewers.
    """

    SYSTEM_PROMPT = """你是“逐影 Shadow Tracing”项目中负责把实验规划讲清楚的科学助手。

你只做一件事：把模型调参的结构化记录润色成自然、专业、有判断依据的中文报告文案。

要求：
1. 输出严格 JSON，字段为 narrative 与 parameter_explanation；narrative 不要逐字段罗列，不要使用“执行协议最终参数”这类机械话语。
2. 用 2 到 4 句中文说明本轮为什么要调、调了什么、预期影响。
3. 只能依据给定材料，不得发明数据；材料里没有上一轮结论时，不要虚构历史反馈。
4. 提到参数时使用“窗口大小”“滞后天数”“预测视野”“正则强度”等易读说法，必要时在括号里保留参数名。
5. 若材料表明调整来自上一轮实验反馈，要明确写成“基于上一轮……因此本轮……”。
6. 若本轮没有额外调参，就如实说明沿用了哪组核心设置，并说明它们的意义。
""" + "\n\n" + ELASTIC_NET_EXECUTION_RULE

    def __init__(self, *, gateway: LLMGateway | None = None) -> None:
        self.gateway = gateway or LLMGateway(temperature=0.35)

    def polish(
        self,
        *,
        candidate_id: str | None = None,
        scientific_objective: str | None = None,
        tuning_entries: list[dict[str, Any]] | None = None,
        plan_summary: str | None = None,
        history_feedback: str | None = None,
    ) -> dict[str, str]:
        entries = list(tuning_entries or [])
        fallback = self._fallback_response(
            candidate_id=candidate_id or "本轮实验",
            scientific_objective=scientific_objective,
            tuning_entries=entries,
            plan_summary=plan_summary,
            history_feedback=history_feedback,
        )
        response = self.gateway.generate_structured(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(
                candidate_id=candidate_id,
                scientific_objective=scientific_objective,
                tuning_entries=entries,
                plan_summary=plan_summary,
                history_feedback=history_feedback,
            ),
            response_model=TuningNarrativeResponse,
            fallback_factory=lambda: fallback,
            temperature=0.35,
            role="模型调参叙事llm",
        )
        return {
            "narrative": response.narrative,
            "parameter_explanation": response.parameter_explanation,
        }

    def _build_user_prompt(
        self,
        *,
        candidate_id: str | None,
        scientific_objective: str | None,
        tuning_entries: list[dict[str, Any]],
        plan_summary: str | None,
        history_feedback: str | None,
    ) -> str:
        entries_text = json.dumps(
            [
                {
                    "refinement_type": entry.get("refinement_type"),
                    "rationale": entry.get("rationale"),
                    "model_parameters": entry.get("model_parameters"),
                    "protocol_notes": (entry.get("protocol_notes") or [])[:6],
                }
                for entry in tuning_entries
            ],
            ensure_ascii=False,
            indent=2,
        )
        return (
            f"candidate_id={candidate_id or '未知'}\n"
            f"scientific_objective={scientific_objective or '未知'}\n"
            f"plan_summary={plan_summary or '无'}\n"
            f"history_feedback={history_feedback or '无'}\n"
            f"tuning_entries={entries_text}\n"
            "输出严格 JSON：{\"narrative\": \"2-4 句自然中文报告\", \"parameter_explanation\": \"一句话参数解释\"}。"
        )

    @staticmethod
    def _fallback_response(
        *,
        candidate_id: str,
        scientific_objective: str | None,
        tuning_entries: list[dict[str, Any]],
        plan_summary: str | None,
        history_feedback: str | None,
    ) -> TuningNarrativeResponse:
        merged: dict[str, Any] = {}
        for entry in tuning_entries:
            for key, value in (entry.get("model_parameters") or {}).items():
                merged[key] = value

        latest_keys = ("window_size", "past_lag_days", "forecast_horizon_days")
        time_parts: list[str] = []
        for key in latest_keys:
            if key in merged and merged[key] is not None:
                if key == "window_size":
                    time_parts.append(f"{merged[key]} 天窗口")
                elif key == "past_lag_days":
                    time_parts.append(f"{merged[key]} 天输入滞后")
                else:
                    time_parts.append(f"{merged[key]} 天预测视野")
        setting_parts: list[str] = []
        if time_parts:
            setting_parts.append("、".join(time_parts))
        if "max_lag_day" in merged and merged["max_lag_day"] is not None:
            setting_parts.append(f"最多构造 {merged['max_lag_day']} 天滞后特征")
        if "test_split_ratio" in merged and merged["test_split_ratio"] is not None:
            ratio = float(merged["test_split_ratio"])
            setting_parts.append(f"迭代评估段占比 {int(round(ratio * 100))}%")
        if "alpha" in merged or "l1_ratio" in merged:
            alpha_text = str(merged.get("alpha"))
            l1_text = str(merged.get("l1_ratio"))
            if alpha_text != "None" or l1_text != "None":
                setting_parts.append(
                    "弹性网络正则 alpha="
                    f"{alpha_text if alpha_text != 'None' else '0'}"
                    f"、l1_ratio={l1_text if l1_text != 'None' else '0.5'}"
                )

        if setting_parts:
            narrative = f"本轮为 {candidate_id} 的模型将关键设置定为：{'；'.join(setting_parts)}。"
        else:
            narrative = f"本轮为 {candidate_id} 未对模型参数做额外调整，沿用候选实验设计的默认时间窗与训练设置。"

        delta_match = re.search(r"delta_pearson_r=(-?\d+\.?\d*)", history_feedback or "")
        if delta_match:
            source_match = re.search(r"source_round=(\d+); experiment=([^;]+)", history_feedback or "")
            delta = float(delta_match.group(1))
            previous = source_match.group(2) if source_match else "前一实验"
            if delta < -0.005:
                narrative += f" 上一轮（{previous}）ΔPearson r={delta:+.3f}，新增变量尚未带来明显增益，本轮在此基础上调整时间窗与正则设置，继续考察弱信号是否来自更长历史结构。"
            elif delta > 0.005:
                narrative += f" 上一轮（{previous}）ΔPearson r={delta:+.3f}，新增变量已呈现正向贡献，本轮沿用相近口径检验该增益是否稳定、可解释。"
            else:
                narrative += f" 上一轮（{previous}）ΔPearson r={delta:+.3f}，信号仍处临界范围，本轮细化时间窗设置以确认其方向与量级。"
        else:
            objective = scientific_objective or "检验物理变量对太阳风速度预测的增量信息"
            if len(objective) > 80:
                objective = objective[:80] + "…"
            narrative += f" 该设置用于在控制未来信息泄漏的前提下，围绕“{objective}”比较对照组与实验组的预测能力。"

        parameter_text = (
            "窗口大小决定模型一次看到多少天历史样本，样本滞后与预测视野共同定义输入相对预测目标的时间差；"
            "迭代评估段比例用于防止滚动评估样本过少，正则强度则用于抑制高相关物理变量的过拟合。"
        )
        return TuningNarrativeResponse(
            narrative=narrative,
            parameter_explanation=parameter_text,
        )
