import json
from openai import OpenAI

from core.prompt_rules import ELASTIC_NET_EXECUTION_RULE
from core.runtime_config import (
    get_dasyscope_api_key,
    get_dasyscope_base_url,
    get_default_llm_model,
    load_project_env,
)


load_project_env()
DASHSCOPE_API_KEY = get_dasyscope_api_key()
client = None
if DASHSCOPE_API_KEY:
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=get_dasyscope_base_url(),
    )


SYSTEM_PROMPT = """你是一个科学实验规划智能体。

你的任务：
根据上一轮的真实实验结果，为太阳风速度预测任务设计下一轮实验方案。

你只能调整以下5个参数：
1. window_size: 整数，1-196，滑动窗口天数
2. use_lag_feature: true/false，是否使用时滞特征
3. max_lag_day: 整数，0-10，最大时滞天数
4. alpha: 小数，0.0-2.0，ElasticNet正则化强度
5. l1_ratio: 小数，0.0-1.0，L1/L2混合比例

你必须遵守：
1. 只输出严格JSON格式
2. 输出必须包含所有5个字段
3. 根据实验结果提出有依据的调整方案

示例输出：
{
    "window_size": 3,
    "use_lag_feature": true,
    "max_lag_day": 3,
    "alpha": 0.5,
    "l1_ratio": 0.5,
    "reasoning": "当前R²较低，建议加入时滞特征并调整窗口大小"
}
""" + "\n\n" + ELASTIC_NET_EXECUTION_RULE


def generate_next_plan(previous_result: dict) -> str:
    """
    根据上一轮结果生成下一轮计划。
    返回: JSON字符串
    """
    metrics = previous_result.get("metrics", {})
    config = previous_result.get("config", {})

    prompt = f"""这是上一轮实验的真实结果：

实验配置：
- window_size: {config.get('window_size', 'N/A')}
- use_lag_feature: {config.get('use_lag_feature', 'N/A')}
- max_lag_day: {config.get('max_lag_day', 'N/A')}
- alpha: {config.get('alpha', 'N/A')}
- l1_ratio: {config.get('l1_ratio', 'N/A')}

评估指标：
- 迭代评估段 RMSE: {metrics.get('test_rmse', 'N/A')}
- 迭代评估段 R²: {metrics.get('test_r2', 'N/A')}
- 迭代评估段 Pearson r: {metrics.get('test_pearson_r', 'N/A')}
- 对照组 RMSE: {metrics.get('baseline_test_rmse', 'N/A')}
- 特征数量: {metrics.get('n_features', 'N/A')}

请分析当前效果，并提出下一轮实验的配置，如进行入参和出参调整。
如果R²接近0或为负，说明模型没有学到有效模式，需要调整。
如果加入了ΔDec效果有提升，考虑调整lag值。

只输出JSON，不要包含任何其他文字。
"""

    if client is None:
        content = json.dumps(_build_local_plan(config, metrics), ensure_ascii=False, indent=2)
    else:
        try:
            response = client.chat.completions.create(
                model=get_default_llm_model(),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            content = response.choices[0].message.content
        except Exception:
            content = json.dumps(_build_local_plan(config, metrics), ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print("Qwen 生成的下一轮计划:")
    print(f"{'='*50}")
    print(content)

    return content


def _build_local_plan(config: dict, metrics: dict) -> dict:
    current_window = int(config.get("window_size", 27))
    current_lag = bool(config.get("use_lag_feature", False))
    current_max_lag = int(config.get("max_lag_day", 3))
    current_alpha = float(config.get("alpha", 0.5))
    current_l1_ratio = float(config.get("l1_ratio", 0.5))
    current_r2 = float(metrics.get("test_r2", 0.0) or 0.0)

    next_plan = {
        "window_size": current_window,
        "use_lag_feature": current_lag,
        "max_lag_day": current_max_lag,
        "alpha": current_alpha,
        "l1_ratio": current_l1_ratio,
        "reasoning": "本地启发式规划：保持当前参数。"
    }

    if current_r2 < 0.35:
        next_plan["window_size"] = max(12, min(40, current_window - 6))
        next_plan["use_lag_feature"] = True
        next_plan["max_lag_day"] = 2
        next_plan["alpha"] = round((current_alpha + 0.35) / 2, 3)
        next_plan["l1_ratio"] = round((current_l1_ratio + 0.65) / 2, 3)
        next_plan["reasoning"] = "当前R²偏低，本地策略建议缩短窗口、启用lag特征，并把正则参数向经验较优区域收缩。"
    elif current_r2 < 0.5:
        next_plan["window_size"] = max(14, min(32, current_window - 3))
        next_plan["use_lag_feature"] = True
        next_plan["max_lag_day"] = max(1, min(3, current_max_lag))
        next_plan["alpha"] = round((current_alpha + 0.4) / 2, 3)
        next_plan["l1_ratio"] = round((current_l1_ratio + 0.6) / 2, 3)
        next_plan["reasoning"] = "当前表现一般，本地策略做温和调参：略缩窗口，保留lag，并微调ElasticNet强度。"
    else:
        next_plan["window_size"] = max(18, min(28, current_window))
        next_plan["use_lag_feature"] = current_lag
        next_plan["max_lag_day"] = current_max_lag
        next_plan["alpha"] = round(current_alpha, 3)
        next_plan["l1_ratio"] = round(current_l1_ratio, 3)
        next_plan["reasoning"] = "当前表现已较稳定，本地策略保持核心参数不变，仅记录本轮结果。"

    return next_plan
