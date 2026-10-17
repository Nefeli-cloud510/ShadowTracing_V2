# 跨轮决策总表

## 1. 总表

`P0 = D:\ShadowTracing\runtime\live_session\_rebuild_probe`。每个单元格中的“来源”即原始文件路径；完整路径索引见第 2 节。

| 轮次 | 科学问题 | 竞争假设 | 所选不确定性 | 候选及 U 值 | 选择理由/人工干预 | 实际输入与参数 | Pearson r 与 Δr CI | 假设判断 | 下一轮动作 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 日影南北偏移是否提前改善太阳风速度预测；本轮聚焦独立增益 vs 磁场总量中介路径（来源：`P0\results\round_01\protocol.json`） | `H_p2_y中心值_independent_gain`（positive）、`H_p2_y中心值_via_scalar_b_nt`（near_zero）（来源：`P0\results\round_01\protocol.json`） | `U_hypothesisconflict_R01_02`（来源：`P0\results\round_01\protocol.json`） | `E_R01_01`；IG=0.6663、PG=0.0、Risk=0.2835、Cost=0.1754、U=0.2589（来源：`P0\state\decision_log.json`，评分公式见 `P0\state\planner_output.json`） | 系统 D002 请求、人类 D003 批准；D004 协议生成；D005 解释；D006 review；D007 人工批准继续（来源：`P0\state\decision_log.json`） | 目标 Vsw；时距 3 天；ElasticNet alpha=0.3、l1_ratio=0.5；baseline 6 字段、treatment 7 字段；lags={} 展开 0..2（来源：`P0\results\round_01\protocol.json`） | baseline r=0.39448958586943433；treatment r=0.4119936080368105；Δr=+0.017504022167376176；Δr CI：未计算（来源：`P0\results\round_01\evaluation_unified.json`） | independent_gain 0.5->0.5917 `supported`；via_scalar 0.5->0.5145 `partially_supported`；分歧 span 0.220->0.0772 `partially_resolved`；评价摘要标“间接检验”（来源：`P0\results\round_01\evaluation_unified.json`） | 人工批准继续（D007）；程序化扩展条件假设（D008）；第 2 轮选中 E_R02_01（D012/013）（来源：`P0\state\decision_log.json`） |
| 2 | `H_p2_y中心值_branch_by_nt_gse` 关于 BY GSE 的解释是否仍受未建模条件约束（来源：`P0\results\round_02\protocol.json`） | 协议列 17 个节点；`round_history` 记 8 个；两处不一致原样保留（来源：`P0\results\round_02\protocol.json`、`P0\state\round_history.json`） | `U_LLM_R01_03`（来源：`P0\results\round_02\protocol.json`） | `E_R02_01`；IG=0.2283、PG=0.0268、Risk=0.3352、Cost=0.1795、U=0.0345（来源：`P0\state\decision_log.json`，评分公式见 `P0\state\planner_output.json`） | 系统 D012 请求、人类 D013 批准；D015 解释；D016 review；D017 人工批准并反馈“修复候选实验设计去重”（来源：`P0\state\decision_log.json`） | 与第 1 轮输入、参数、预测文件完全一致；同臂 CSV 逐字节相同（来源：`P0\results\round_02\protocol.json`、`P0\results\round_02\E_R02_01\predictions\E_R02_01_baseline.csv`） | 指标与第 1 轮完全相同；Δr=+0.017504022167376176；Δr CI：未计算（来源：`P0\results\round_02\evaluation_unified.json`） | independent_gain 升至 1.0/`converged`（已知状态机/LLM 叠加缺陷，非独立实证）；via_scalar=0.9101、branch_by_nt_gse=0.8048、null_competition=0.6781 均 active（来源：`P0\state\hypothesis_tree.json`） | 人工批准继续（D017）；第 3 轮 D022 请求 E_R03_01（U=0.1047），未批准、未执行（来源：`P0\state\decision_log.json`） |
| 3 | 第 3 轮候选 `E_R03_01` 目标为 `U_LLM_R01_10`（来源：`P0\state\planner_output.json`） | 候选 tested_hypotheses 22 个（来源：`P0\state\planner_output.json`） | `U_LLM_R01_10`（来源：`P0\state\planner_output.json`） | `E_R03_01`；IG=0.3492、PG=0.0268、Risk=0.2845、Cost=0.1831、U=0.1047；仅存于 planner_output 与 D022（来源：`P0\state\planner_output.json`、`P0\state\decision_log.json`） | D022 系统请求审批；未批准（来源：`P0\state\decision_log.json`） | 无正式执行协议；不得引用任何第 3 轮结果 | 无结果；无 CI | 无假设更新可计入第 3 轮 | 等待人工审批；当前 `P0\state\candidate_experiments.json` 为 round 4 草案，不替代 D022 审批 |

## 2. 路径索引

| 键 | 原始文件 |
| --- | --- |
| P0 | `D:\ShadowTracing\runtime\live_session\_rebuild_probe` |
| round1 protocol | `P0\results\round_01\protocol.json` |
| round1 result | `P0\results\round_01\result_unified.json` |
| round1 evaluation | `P0\results\round_01\evaluation_unified.json` |
| round1 baseline CSV | `P0\results\round_01\E_R01_01\predictions\E_R01_01_baseline.csv` |
| round1 treatment CSV | `P0\results\round_01\E_R01_01\predictions\E_R01_01_treatment.csv` |
| round2 protocol | `P0\results\round_02\protocol.json` |
| round2 result | `P0\results\round_02\result_unified.json` |
| round2 evaluation | `P0\results\round_02\evaluation_unified.json` |
| round2 baseline CSV | `P0\results\round_02\E_R02_01\predictions\E_R02_01_baseline.csv` |
| round2 treatment CSV | `P0\results\round_02\E_R02_01\predictions\E_R02_01_treatment.csv` |
| decision log | `P0\state\decision_log.json` |
| round history | `P0\state\round_history.json` |
| hypothesis tree | `P0\state\hypothesis_tree.json` |
| uncertainty queue | `P0\state\uncertainty_priority.json` |
| candidate snapshot | `P0\state\candidate_experiments.json` |
| planner output | `P0\state\planner_output.json` |
| task | `P0\state\task.json` |
| data dictionary | `P0\state\data_dictionary.json` |
| upload manifest | `P0\state\upload_manifest.json` |

## 3. 必须保留的解释边界

- Δr 的 bootstrap CI 在所有记录中都不存在，表格中写“未计算”，不得补造区间。
- 两轮预测逐字节一致，因此两轮指标相同；这是同一预测产物被归档两次，不是第二轮独立复现。
- `H_p2_y中心值_independent_gain` 的 `1.0/converged` 是状态机与 LLM 解释叠加的已知缺陷，不能作为第 2 轮实验独立证实。
- 第 3 轮无结果、无假设更新，不是闭环中的实证轮次。
