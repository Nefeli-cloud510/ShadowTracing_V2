# 证据缺口清单

统一前缀：`D:\ShadowTracing\runtime\live_session\_rebuild_probe\`

## 1. 已确认缺口

| 编号 | 缺口 | 证据位置 | 影响与建议 |
| --- | --- | --- | --- |
| G01 | 未计算配对 Δr / ΔRMSE 的时间块 bootstrap CI | `results/round_01/evaluation_unified.json -> robustness.bootstrap.n_iterations=0`；`results/round_02/evaluation_unified.json` 同 | 只能讲“点估计 +0.0175”，不能讲“区间支持增益”；下一轮应登记块长、重抽样次数、种子与共同样本 |
| G02 | `robustness.stable=true` 但没有任何 bootstrap/time-slice 执行 | `evaluation_unified.json -> robustness`：`bootstrap.n_iterations=0`、`time_slice.splits=[]` | `stable=true` 只是状态字段，不是稳定性验证通过 |
| G03 | 两轮预测文件逐字节相同 | baseline SHA256 `142FB176...`；treatment SHA256 `C9BF05CA...`（两轮一致） | 不能把第二轮当作新的独立证据；跨轮证据链目前只证明“同一结果被记录两次” |
| G04 | 第 2 轮 tested_hypotheses 不一致 | `results/round_02/protocol.json` 17 个；`state/round_history.json` 第 2 轮 entry 8 个 | 报告中应同时说明两个口径，不能只引用其一 |
| G05 | PDF 只有占位记录，没有正文 | `state/upload_manifest.json`：6 个 PDF `extraction_status=placeholder_only`、`knowledge_sync_status=local_session_only` | 文献证据不可追溯；不能声称“已读论文” |
| G06 | 简单基线未执行 | 训练均值、持续性、Vsw 历史、Vsw 历史＋日影均 `not_executed` | 不能引用未运行的对照成绩 |
| G07 | 预测 CSV 缺日期三元组 | CSV 仅 `time,y_true,y_pred` | 无法逐样本证明 issue_date/feature_date/target_date 对齐；下一轮产物应补齐 |
| G08 | 旧第二轮 IG=1、U=0.4466 评分快照不可恢复 | 当前 `state/candidate_experiments.json` 为 round 4；`state/planner_output.json` 为 round 3 计划 | 只能标“不可完整追溯”，不能拼凑旧评分 |
| G09 | `H_p2_y中心值_independent_gain` 的 1.0/converged 是缺陷 | `state/hypothesis_tree.json`；`state/planner_output.json -> interpretation_enhancements` | 状态机与 LLM 解释叠加，不是实验直接证实 |
| G10 | `session_status.json` 的 `runRoot` 与实际目录不一致 | `session_status.json` 写 `runtime\live_session\current`；实际为 `_rebuild_probe` | 展示时以实际目录为准，并在审计中保留差异 |
| G11 | 第 3 轮候选未审批、未执行 | `state/decision_log.json -> D022`：`approved=false` | 第 3 轮无任何成绩可引用 |
| G12 | IG 是 1-overlap 启发式，不是校准 Bayesian EIG | `state/planner_output.json -> estimated_information_gain.rationale` | 报告中如实写“信息增益启发式评分” |
| G13 | 本会话未重做原始字段错位审计 | `state/task.json -> payload.data_sources`；本证据包只读引用 | 不能以本会话产物证明“原始字段错位已修复” |
| G14 | bootstrap 结果块构造、重抽样次数、种子和有效样本描述缺失 | 所有 evaluation/result 文件 | 下一轮稳健性检验必须登记这些参数 |

## 2. 建议的下一轮修复顺序

1. G07：执行协议和预测 CSV 增加 `issue_date/feature_date/target_date` 及共同样本掩码。
2. G01/G02/G14：在真实时间序列样本上执行时间块 bootstrap，提供块长、重复次数、种子与区间。
3. G04：统一 tested_hypotheses 的唯一口径，或明确区分“协议覆盖节点”与“实际更新节点”。
4. G05/G06：补 PDF 正文提取或改用真实检索片段；简单基线全部实际运行后才允许引用。
5. G09：收敛状态机加入“有效实验评价”门槛，未检验的假设不得自动 converged。
6. G08/G10/G11：补齐旧评分归档途径，修正 runRoot，第 3 轮批准后再执行。

## 3. 可展示结论建议

本证据包能够支撑的表达：

- 同一会话、同一任务的两轮协议与产物完整归档，指标、SHA、决策链可逐项核验。
- 候选评分、人工审批、结果回写、疑问状态更新的链路完整。
- 已确认缺口已经如实登记。

尚不能支撑的表达：

- “日影稳定改善太阳风速度预测”。
- “跨窗口/跨时段稳定性已验证”。
- “系统自主发现并选择了最优实验”。
- “上传文献已被模型阅读”。
- “不确定性已解决”。
