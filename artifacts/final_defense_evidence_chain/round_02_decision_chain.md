# 第 2 轮决策链证据包

统一前缀：`D:\ShadowTracing\runtime\live_session\_rebuild_probe\`

## 1. 科学问题与竞争假设

- 本轮科学目标：`results/round_02/protocol.json -> scientific_objective`
  - “H_p2_y中心值_branch_by_nt_gse 关于 行星际磁场Y分量（GSE坐标） 与 太阳风速度 的解释是否仍受未建模条件约束？”
- 目标不确定性：`results/round_02/protocol.json -> target_uncertainties`，即 `U_LLM_R01_03`。
- 协议 `tested_hypotheses` 记 17 个节点；`state/round_history.json` 第 2 轮 entry 记 8 个。两处不一致，本证据包如实保留。

## 2. 候选实验及评分

候选 `E_R02_01`，类型 `distinguishing`：

| 项 | 值 | 说明 |
| --- | ---: | --- |
| IG | 0.2283 | 比较 136 组假设区间，1-overlap 启发式 |
| PG | 0.0268 | 实验前预期 |
| Risk | 0.3352 | 评分快照 |
| Cost | 0.1795 | 评分快照 |
| utility U | 0.0345 | `state/decision_log.json -> D012/D013` |

选择理由：系统请求（D012），人类 PI 批准（D013）。第 2 轮没有引入新的独立数据窗口，协议输入与第 1 轮相同。

## 3. 决策链

| 决策 | 行为 | 执行者 | 来源 |
| --- | --- | --- | --- |
| D008 | 第 2 轮程序化扩展假设树 | system | `state/decision_log.json` |
| D012 | 请求审批 E_R02_01，U=0.0345 | system | `state/decision_log.json` |
| D013 | 批准 E_R02_01 | human_pi | `state/decision_log.json` |
| D015 | 科学解释者回写，enhancement `IE001`、`LIE_INT_001` | system | `state/decision_log.json` |
| D016 | 请求 round review | system | `state/decision_log.json` |
| D017 | 批准进入下一轮 | human_pi | `state/decision_log.json` |
| D022 | 第 3 轮请求审批 E_R03_01，U=0.1047 | system | `state/decision_log.json`；未批准、未执行 |

D017 人工反馈原话：`修复候选实验设计去重：进入下一轮，验证实验设计已按不确定性来源分化。`（`state/decision_log.json -> D017.details.human_feedback`）

## 4. 实际输入与参数

完整协议：`results/round_02/protocol.json`

- 模型、特征、参数与第 1 轮完全一致：ElasticNet，`alpha=0.3, l1_ratio=0.5, lags={}, window_size=3`。
- 预测文件与第 1 轮同臂逐字节相同：
  - baseline SHA256 `142FB176EEE0E3C9ED7620AB5B7AA943C612FA903664A7EB232B8FDA0A450320`
  - treatment SHA256 `C9BF05CA62D5B71D87B9D82C2732248BF661F882EFE423766175ECC3CC703C6B`

## 5. 真实结果

`results/round_02/result_unified.json`、`results/round_02/evaluation_unified.json` 与第 1 轮完全相同（原因见预测文件逐字节一致）：

| 指标 | baseline | treatment | 增量 |
| --- | ---: | ---: | ---: |
| Pearson r | 0.39448958586943433 | 0.4119936080368105 | +0.017504022167376176 |
| RMSE, km/s | 49.25384857191404 | 48.42758229119842 | -0.8262662807156218 |
| MAE, km/s | 39.5986254968859 | 38.40749317745875 | -1.1911323194271546 |
| R2 | -0.05852141690524881 | -0.023304498564322484 | +0.03521691834092633 |

无 bootstrap CI、无 time-slice CI；`robustness.bootstrap.n_iterations=0`、`time_slice.splits=[]`。

## 6. 假设状态变化（含已知缺陷）

`state/hypothesis_tree.json`（`current_round=3`，22 节点）：

- `H_p2_y中心值_independent_gain`：score=1.0，status=`converged`。
- `H_p2_y中心值_via_scalar_b_nt`：score=0.9101，status=`active`。
- `H_p2_y中心值_branch_by_nt_gse`：score=0.8048，status=`active`。
- `H_p2_y中心值_null_competition`：score=0.6781，status=`active`。
- `H_p2_y中心值_branch_bz_nt_gse`：score=0.6406，status=`active`。
- 5 个 `_2` 条件节点：score=0.38，status=`pending`。

审计判定：`converged / 1.0` 主要由 LLM 解释增强（`IE001`、`LIE_INT_001`）与状态机叠加产生，不是实验直接证明；`state/planner_output.json -> interpretation_enhancements` 也包含“当前支持度为 1.00，结合上一轮表现”这类叠加表述。该状态不应作为日影稳定增益的实证证据。

## 7. 不确定性队列

`state/uncertainty_priority.json`：

- 当前队列 51 条，`resolved=[]`、`deprecated=[]`。
- top 三项：`U_LLM_R01_11`、`U_LLM_R01_12`、`U_LLM_R02_25`，`priority_score=0.854`。
- `U_LLM_R01_03` 在第 2 轮评价中被记为 `partially_resolved`（`results/round_02/evaluation_unified.json -> scientific.disagreement_updates`），但仍在当前队列 active，未关闭。

## 8. 下一轮动作

- D022 请求审批 `E_R03_01`（`distinguishing`，utility `0.1047`，目标 `U_LLM_R01_10`），当前处于 `awaiting_approval`。
- 当前 `state/candidate_experiments.json` 是 round 4 候选 `E_R04_01/02/03`，不能把它当成 D022 审批候选。
- 第 3 轮未执行、未关闭；不能引用任何第 3 轮结果。
