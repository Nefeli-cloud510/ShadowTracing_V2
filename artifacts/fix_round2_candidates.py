from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

from core.unified_schema import CandidateExperimentSet


STATE_DIR = Path(
    r"D:\ShadowTracing\runtime\live_session\sessions\session_20261016_235451_805278\state"
)
CANDIDATE_PATH = STATE_DIR / "candidate_experiments.json"


def main() -> None:
    raw = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    source = next(
        item
        for item in raw["candidates"]
        if item["experiment_id"] == "E_R02_05"
    )
    candidate = deepcopy(source)

    design = candidate["design"]
    design["control"] = ["SW Plasma Speed, km/s"]
    design["treatment"] = ["SW Plasma Speed, km/s", "p2_y中心值"]
    design["lags"] = {
        "SW Plasma Speed, km/s": [1, 2, 3],
        "p2_y中心值": [1, 2, 3],
    }
    design["forecast_horizon_days"] = 3
    design["past_lag_days"] = 3
    design["window_size"] = 14
    design["control_lag_days"] = 3
    design["treatment_lag_days"] = 3
    design["control_forecast_horizon_days"] = 3
    design["treatment_forecast_horizon_days"] = 3
    design["design_focus"] = "p2_y中心值"
    design["display_target"] = "SW Plasma Speed, km/s"
    design["display_control"] = ["SW Plasma Speed, km/s"]
    design["display_treatment"] = ["SW Plasma Speed, km/s", "p2_y中心值"]
    design["display_design_focus"] = "p2_y中心值"
    design["model_families"] = ["ridge", "elasticnet"]
    design["runs_spec"] = [
        {
            "run_id": "ridge_baseline",
            "label": "ridge 对照组：风速历史",
            "feature_columns": ["SW Plasma Speed, km/s"],
            "model_family": "ridge",
            "parameters": {},
            "retrain": True,
        },
        {
            "run_id": "ridge_treatment",
            "label": "ridge 实验组：风速历史+日影",
            "feature_columns": ["SW Plasma Speed, km/s", "p2_y中心值"],
            "model_family": "ridge",
            "parameters": {},
            "retrain": True,
        },
        {
            "run_id": "elasticnet_baseline",
            "label": "elasticnet 对照组：风速历史",
            "feature_columns": ["SW Plasma Speed, km/s"],
            "model_family": "elasticnet",
            "parameters": {},
            "retrain": True,
        },
        {
            "run_id": "elasticnet_treatment",
            "label": "elasticnet 实验组：风速历史+日影",
            "feature_columns": ["SW Plasma Speed, km/s", "p2_y中心值"],
            "model_family": "elasticnet",
            "parameters": {},
            "retrain": True,
        },
    ]
    design["contrasts_spec"] = [
        {
            "contrast_id": "gain_ridge",
            "name": "ridge 家族内日影增量",
            "kind": "incremental_gain",
            "run_ids": ["ridge_baseline", "ridge_treatment"],
            "contrast_ids": [],
            "direction": "mixed",
            "note": "固定风速历史基线，只比较同一模型家族内加入 p2_y中心值 前后的增益。",
        },
        {
            "contrast_id": "gain_elasticnet",
            "name": "elasticnet 家族内日影增量",
            "kind": "incremental_gain",
            "run_ids": ["elasticnet_baseline", "elasticnet_treatment"],
            "contrast_ids": [],
            "direction": "mixed",
            "note": "固定风速历史基线，只比较同一模型家族内加入 p2_y中心值 前后的增益。",
        },
    ]
    design["diagnostics_spec"] = []
    design["notes"] = [
        "generated_from:U_R02_01",
        "human_pi_override:baseline=SW Plasma Speed,km/s,time=3/3/14",
        "design_focus:p2_y中心值",
        "capability_boundary:model_sensitivity_仅限线性家族内的正则化与设置敏感性",
        "round2_fixed_time:forecast_horizon_days=3,past_lag_days=3,window_size=14",
        "round2_primary_metric:rmse_gain_with_bootstrap_ci",
        "simple_baselines:train_mean,persistence",
    ]

    candidate["experiment_id"] = "E_R02_01"
    candidate["type"] = "model_sensitivity"
    candidate["experiment_type"] = "model_sensitivity"
    candidate["purpose"] = (
        "在固定 forecast_horizon_days=3、past_lag_days=3、window_size=14、"
        "random_state=42、test_split_ratio=0.2 的条件下，把对照基线改为太阳风速度"
        "历史滞后项 SW Plasma Speed, km/s，实验组为相同风速历史滞后 + LHAASO 日影"
        "南北偏移 p2_y中心值；在同一模型家族内比较 Ridge 与 ElasticNet 的 RMSE 增益"
        "及其配对块 bootstrap CI，同时加入训练均值与最近可用风速持续性两类简单基线，"
        "判断日影是否在风速历史基线之外提供稳定增量信息。"
    )
    candidate["scientific_question"] = (
        "在太阳风速度历史滞后已经构成基线时，加入 LHAASO 宇宙线日影南北偏移"
        "p2_y中心值 是否能使 Ridge 或 ElasticNet 对 3 天后的太阳风速度预测产生"
        "越过 bootstrap 波动的稳定 RMSE 改善，还是该增益在共同测试期内证据不足？"
    )
    candidate["tested_hypotheses"] = [
        "H_shadow_incremental_gain",
        "H_by_mediated_path",
    ]
    candidate["related_uncertainties"] = ["U_R02_01", "U_R02_03"]
    candidate["disagreement_context"] = {
        "U_R02_01": {
            "H_shadow_incremental_gain": {
                "hypothesis_id": "H_shadow_incremental_gain",
                "expected": "positive",
                "status": "active",
                "support_score": 0.8919,
                "rationale_trigger": "llm_hypothesis_H1",
                "source_summary": "H1：风速历史基线加入日影后应有稳定正向 RMSE 增益。",
            }
        },
        "U_R02_03": {
            "H_by_mediated_path": {
                "hypothesis_id": "H_by_mediated_path",
                "expected": "near_zero",
                "status": "active",
                "support_score": 0.6069,
                "rationale_trigger": "llm_hypothesis_H2",
                "source_summary": "H2：日影增益可能只是与风速历史共线的正则化系数再分配。",
            }
        },
    }
    candidate["hypothesis_predictions"] = {
        "H_shadow_incremental_gain": {
            "expected_effect": "positive",
            "expected_range": [0.2, 2.5],
            "metric": "RMSE",
        },
        "H_by_mediated_path": {
            "expected_effect": "near_zero",
            "expected_range": [-2.0, 2.0],
            "metric": "RMSE",
        },
    }
    candidate["responded_prior_finding"] = (
        "E_R01_03 未检出稳定增益：Ridge ΔRMSE=-0.2519、ElasticNet ΔRMSE=-0.4863，"
        "bootstrap CI 均跨零且 stable=false；该轮基线为 BX, nT (GSE, GSM)，未把风速历史"
        "作为基线，因此本轮固定 3/3/14 并改为风速历史基线后检验日影增量。"
    )
    candidate["target_uncertainty_focus"] = (
        "在相同 3 天超前、3 天滞后和 14 天窗口下，太阳风速度历史滞后基线加入宇宙线"
        "日影南北偏移后，Ridge 与 ElasticNet 的测试集 RMSE 下降是否超过配对块 bootstrap"
        "置信区间下界，并且 Pearson_r 提升是否与 RMSE 下降同向。"
    )
    candidate["changed_factors"] = ["自变量特征集（BX 基线 → 风速历史基线）"]
    candidate["held_fixed_factors"] = [
        "目标变量",
        "预测超前 3 天",
        "历史长度 3 天",
        "时间窗口 14 天",
        "模型家族",
        "评价指标与日历对齐规则",
        "训练/评价时间划分",
        "随机种子与测试比例",
    ]
    candidate["exploratory"] = False

    original_candidates = raw["candidates"]
    raw["round"] = 2
    raw["candidates"] = [candidate]
    raw["rejected_candidates"] = [
        {
            "experiment_id": item["experiment_id"],
            "reason": (
                "人工拒绝：基线/时间设置违反第二轮固定条件，卷入 Scalar B 或 5/7/21"
            ),
            "similar_to": item["experiment_id"],
        }
        for item in original_candidates
        if item["experiment_id"] != "E_R02_01"
    ]
    raw["minimum_required"] = 1
    raw["current_count"] = 1
    raw["note"] = (
        "AI 生成的 6 个候选均违反 PI 约束（基线不是风速历史、时间不是 3/3/14、"
        "把 Scalar B 当日影特征或 runs_spec 为空）。候选集由 PI 人工修正为一个"
        "唯一可执行候选 E_R02_01，并进入审批。"
    )

    parsed = CandidateExperimentSet.model_validate(raw)
    payload = json.loads(parsed.model_dump_json())
    CANDIDATE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("patched candidates ->", CANDIDATE_PATH)
    print("ids:", [item["experiment_id"] for item in payload["candidates"]])


if __name__ == "__main__":
    sys.exit(main())
