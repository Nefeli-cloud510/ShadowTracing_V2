# 最终答辩逐轮决策链证据包

本目录以最终答辩所指定的同一实时会话为基准，冻结两轮已执行实验的完整决策链。证据包只做只读归档与核验，不修改实时会话目录，不新增科学结论。

## 0. 会话与展示边界

- 实际会话目录：`D:\ShadowTracing\runtime\live_session\_rebuild_probe`
- 任务 ID：`ST_LIVE_20260903_201814`
- 冻结时会话状态：`awaiting_approval`，`currentRound=3`，消息为“第 3 轮候选实验已生成，等待审批。”
- `session_status.json` 中 `runRoot` 写的是 `D:\ShadowTracing\runtime\live_session\current`，与实际归档目录 `_rebuild_probe` 不一致；本证据包以实际目录为准，并保留该问题作为审计记录。
- 生成本证据包前代码版本：`08a15dc`
- 已执行实验：`E_R01_01`（第 1 轮）、`E_R02_01`（第 2 轮）
- 第 3 轮 `E_R03_01` 仍处于等待人工审批状态，未执行，不当作已完成闭环

展示语言只能使用以下事实：

- 同一任务、同一会话下两轮协议一致，两轮预测文件逐字节一致，指标可复核。
- 候选评分、人工审批、结果回写、假设状态更新链完整可追。
- 不能写“日影稳定增益”“跨时间稳定”“系统自主选出了最优实验”“已有文献已读”“不确定性已解决”。
- `H_p2_y中心值_independent_gain` 的 `converged / 1.0` 是状态机与 LLM 解释叠加造成的已知缺陷，不是实验直接证实。

## 1. 关键文件索引

| 内容 | 路径 |
| --- | --- |
| 实时会话状态 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\session_status.json` |
| 任务定义 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\state\task.json` |
| 决策日志 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\state\decision_log.json` |
| 第 1 轮协议 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\results\round_01\protocol.json` |
| 第 1 轮统一结果 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\results\round_01\result_unified.json` |
| 第 1 轮统一评价 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\results\round_01\evaluation_unified.json` |
| 第 2 轮协议 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\results\round_02\protocol.json` |
| 第 2 轮统一结果 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\results\round_02\result_unified.json` |
| 第 2 轮统一评价 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\results\round_02\evaluation_unified.json` |
| 假设树 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\state\hypothesis_tree.json` |
| 不确定性队列 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\state\uncertainty_priority.json` |
| 当前候选快照 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\state\candidate_experiments.json` |
| planner 输出 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\state\planner_output.json` |
| 第 1 轮归档 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\state\round_snapshots\round_01.json` |
| 第 2 轮归档 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\state\round_snapshots\round_02.json` |
| PDF 上传清单 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\state\upload_manifest.json` |
| 逐日预测 CSV | `D:\ShadowTracing\runtime\live_session\_rebuild_probe\results\round_01\E_R01_01\predictions\E_R01_01_baseline.csv` 等 4 个文件 |

## 2. 两轮指标

两轮协议的预测文件逐字节相同，因此两轮指标完全一致，以下数值直接来自 `result_unified.json` 与 `evaluation_unified.json`，可从逐日预测 CSV 独立重算。

| 指标 | baseline | treatment | 增量 |
| --- | ---: | ---: | ---: |
| RMSE, km/s | 49.25384857191404 | 48.42758229119842 | -0.8262662807156218 |
| MAE, km/s | 39.5986254968859 | 38.40749317745875 | -1.1911323194271546 |
| Pearson r | 0.39448958586943433 | 0.4119936080368105 | +0.017504022167376176 |
| R2 | -0.05852141690524881 | -0.023304498564322484 | +0.03521691834092633 |

任务主指标为 `Pearson_r`，辅助指标为 `RMSE`、`MAE`（`state/task.json`）。并没有计算配对 Δr 的 bootstrap CI，也没有 time-slice 证据，因此“Pearson r 提升 0.0175”只能作为该评估段的点估计增量。

## 3. 稳健性字段原样保留

`evaluation_unified.json` 中原文是：

- `robustness.bootstrap = {stable: true, n_iterations: 0}`
- `robustness.time_slice = {splits: "", stable: true}`
- `robustness.overall = {stable: true}`
- recommendation：可以进入下一轮假设解释，但建议补充 bootstrap/time-slice 稳健性检验。

`n_iterations=0`、`splits=""` 表示没有任何 bootstrap 或时间切片被执行，不能把这里的 `stable=true` 解释为稳定性验证通过。

## 4. 第 3 轮状态

`E_R03_01` 只存在于 `state/decision_log.json` 的 D022 和 `state/planner_output.json` 的 `LCP001` 中：

- 类型：`distinguishing`
- 目标不确定性：`U_LLM_R01_10`
- utility：`0.1047`
- 审批状态：D022 已由系统请求人工审批，未被批准，未执行

当前 `state/candidate_experiments.json` 是 round 4 候选（`E_R04_01/02/03`），生成时间 `2026-09-03T21:10:48`，不包含 D022 的审批快照。旧第二轮 IG=1、U=0.4466 的评分快照无法在当前归档中恢复，已作为不可完整追溯项保留。

## 5. 补充说明

- 第 2 轮 `round_history.json` 记 `tested_hypotheses` 为 8 个，第 2 轮 `protocol.json` 记 17 个；两者不一致，证据包如实保留，不能只引用其中一个。
- 上传的 6 个 PDF 在 `upload_manifest.json` 中全部为 `placeholder_only`，没有正文、chunk 或页码，文献证据只能标“未存档 / 不可追溯”。
- 训练均值、持续性、风速历史、风速历史＋日影等简单基线均未执行，标 `not_executed`。
- 本会话没有在 v3 CSV 上重跑，也没有在本证据包内重新做原始字段错位审计。

## 6. 相关文档

- `docs/final_demo_run_guide.md`
- `docs/group_meeting_evidence_chain.md`
- `artifacts/legacy_v3_migration/executor_acceptance/executor_sparse_lags_acceptance.md`
