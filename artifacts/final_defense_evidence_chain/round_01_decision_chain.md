# 第 1 轮决策链证据包

统一前缀：`D:\ShadowTracing\runtime\live_session\_rebuild_probe\`

## 1. 科学问题与竞争假设

- 根科学问题：`state/task.json -> payload.research_question.text`
  - “LHAASO观测到的宇宙线日影南北偏移是否包含能够提前改善未来太阳风速度预测的有效信息？”
- 本轮科学目标：`results/round_01/protocol.json -> scientific_objective`
  - 区分 `H_p2_y中心值_independent_gain` 与 `H_p2_y中心值_via_scalar_b_nt` 是否能在“宇宙线日影南北偏移、太阳风速度、行星际磁场总量”上通过对照实验直接区分。
- 竞争假设及预测范围：`results/round_01/protocol.json -> hypothesis_predictions`
  - `H_p2_y中心值_independent_gain`：预期 `positive`，区间 `[0.0022, 0.1158]`
  - `H_p2_y中心值_via_scalar_b_nt`：预期 `near_zero`，区间 `[-0.0656, 0.0656]`
- 假设来源：`results/round_01/protocol.json -> hypothesis_source_context`
  - `H_p2_y中心值_independent_gain`：`task_bootstrap`
  - `H_p2_y中心值_via_scalar_b_nt`：`mediator_bootstrap`（候选中介变量）

注意：两个区间都是面向“假设区分度”的 LLM 启发式预测，不是实验观测到的 Δr 区间，也没有经过概率校准。

## 2. 所选不确定性

- `U_hypothesisconflict_R01_02`：两假设分歧是否能在上述变量上通过对照实验直接区分。
- 来源：`results/round_01/protocol.json -> target_uncertainties`；`state/decision_log.json -> D003.details.plan_summary`。
- 第 1 轮结束时状态为 `partially_resolved`，不代表已关闭。

## 3. 候选实验及评分

候选 `E_R01_01`，类型 `incremental_gain`：

| 项 | 值 | 原始位置 |
| --- | ---: | --- |
| IG | 0.6663 | `state/decision_log.json -> D002`；`state/planner_output.json`；1-overlap 启发式，只比较 1 组区间，不是 Bayesian EIG |
| PG | 0.0 | 候选评分快照 |
| Risk | 0.2835 | 候选评分快照 |
| Cost | 0.1754 | 候选评分快照 |
| utility U | 0.2589 | `state/decision_log.json -> D002/D003.details.utility_score` |

选择理由：系统请求审批（D002），人类 PI 批准（D003），D004 生成执行协议，没有自动跳过人工审批。

## 4. 决策链

| 决策 | 行为 | 执行者 | 来源 |
| --- | --- | --- | --- |
| D001 | round 0 准备 planner input | system | `state/decision_log.json` |
| D002 | 请求审批 E_R01_01，U=0.2589 | system | `state/decision_log.json` |
| D003 | 批准 E_R01_01 | human_pi | `state/decision_log.json` |
| D004 | 生成执行协议 | system | `state/decision_log.json` |
| D005 | 科学解释者回写（IE001、LIE_INT_001） | system | `state/decision_log.json` |
| D006 | 请求 round review | system | `state/decision_log.json` |
| D007 | 批准进入下一轮 | human_pi | `state/decision_log.json` |

## 5. 实际输入与参数

完整协议：`results/round_01/protocol.json`

- 目标：`SW Plasma Speed, km/s`；预测时距 3 天；模型 ElasticNet。
- baseline 6 字段、treatment 7 字段，具体见 `data_provenance_audit.md`。
- `alpha=0.3, l1_ratio=0.5, lags={}, window_size=3, use_lag_feature=false, max_lag_day=0, random_state=42, test_split_ratio=0.2`。
- 主指标 `Pearson_r`，辅助 `RMSE`、`MAE`（`state/task.json -> payload.evaluation`）。
- 特征展开为 lag 0,1,2，不是逐变量稀疏滞后；预测 CSV 无 issue/feature/target date 列。

## 6. 真实结果

`results/round_01/result_unified.json`、`results/round_01/evaluation_unified.json`：

| 指标 | baseline | treatment | 增量 |
| --- | ---: | ---: | ---: |
| Pearson r | 0.39448958586943433 | 0.4119936080368105 | +0.017504022167376176 |
| RMSE, km/s | 49.25384857191404 | 48.42758229119842 | -0.8262662807156218 |
| MAE, km/s | 39.5986254968859 | 38.40749317745875 | -1.1911323194271546 |
| R2 | -0.05852141690524881 | -0.023304498564322484 | +0.03521691834092633 |

- 没有配对 bootstrap CI，没有 time-slice CI，Δr=+0.0175 只能是该 36 样本评价段的点估计增量。
- 预测文件：`results/round_01/E_R01_01/predictions/E_R01_01_baseline.csv` 与 `E_R01_01_treatment.csv`。

## 7. 假设判断

`results/round_01/evaluation_unified.json -> scientific.hypothesis_assessments`：

- `H_p2_y中心值_independent_gain`：0.5000 -> 0.5917，系统状态 `supported`，direction=true，magnitude=true。
- `H_p2_y中心值_via_scalar_b_nt`：0.5000 -> 0.5145，系统状态 `partially_supported`，direction=partial。
- 分歧更新：`U_hypothesisconflict_R01_02` 的 support_span 0.220 -> 0.0772，`partially_resolved`。

边界说明：这些分数包含系统状态机与科学解释者叠加成分，评价摘要已写明“间接检验”；这不是独立时段验证，不能等价于“日影增益稳定成立”。

## 8. 下一轮动作

- D007 人工批准继续；D008 程序化生成 5 个条件路径假设。
- 第 2 轮选中 `E_R02_01`，目标不确定性 `U_LLM_R01_03`，详见 `round_02_decision_chain.md`。
