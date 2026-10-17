# 会话定位审计：`session_20261016_235451_805278`

> 审计时间：2026-10-17。本文件用于回答“最终答辩证据包是否来自用户指定的同一会话”，不新增实验结果，不修改实时会话目录。

## 1. 结论

- 用户要求的会话 `session_20261016_235451_805278` 在 `runtime/live_session/sessions/` 下**不存在**。该目录当前只有 `session_20261017_073547_697152`。
- `session_20261017_073547_697152` 是唯一实时会话，状态为 `idle`、`currentRound=0`，没有科学问题、实验或结果，不能当作已执行两轮的真实会话。
- 现有最终答辩证据包 `artifacts/final_defense_evidence_chain/` 实际归档对象是 `runtime/live_session/_rebuild_probe`，任务 ID `ST_LIVE_20260903_201814`，不是 `session_20261016_235451_805278`。
- `_rebuild_probe` 中已执行的是 `E_R01_01`（第 1 轮）和 `E_R02_01`（第 2 轮），两轮预测文件逐字节相同，不能作为两个独立证据。
- `E_R01_03` 只存在于人工审批文件、候选规划和决策日志等文本层，`_rebuild_probe` 中没有与它对应的 prediction / result / evaluation 产物；不能把 `E_R01_03` 讲成已执行实验。
- 旧第二轮 `IG=1`、`U=0.4466` 的评分快照无法从现有归档恢复，只能标为“不可完整追溯”。

## 2. 会话识别表

| 会话 / 目录 | 位置 | task / 状态 | 实验 ID | 数据版本 | 模型 | 样本数 | 生成时间 |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| 用户指定会话 | `runtime/live_session/sessions/session_20261016_235451_805278` | 不存在；`artifacts/start_v3_rag_response.json` 只留下“正在初始化”记录 | 无 | 无实际运行文件 | `qwen3.8-flash` | 无 | 2026-10-16T23:54:51（启动占位） |
| 当前实时会话 | `runtime/live_session/sessions/session_20261017_073547_697152` | `idle`，`currentRound=0`，等待数据字典 | 无 | 无 | `qwen3.8-flash` | 无 | 2026-10-17T07:35:47 |
| 证据包实际归档 | `runtime/live_session/_rebuild_probe` | `ST_LIVE_20260903_201814`，`awaiting_approval`，`currentRound=3` | `E_R01_01`、`E_R02_01`；`E_R03_01` 未审批 | 冻结代码 `08a15dc`（见 `session_manifest.json`） | `qwen-plus` | 每个预测 CSV 36 行 | 2026-09-03T20:18:14 起 |

## 3. 为什么只能先把 `_rebuild_probe` 当作替代归档

`session_manifest.json` 的 `session_runtime_root_actual` 明确写的是 `D:\ShadowTracing\runtime\live_session\_rebuild_probe`。此前生成证据包时，项目里只有该目录带有完整的两轮 `protocol/result/evaluation/predictions` 文件，因此证据包以它为准。它满足“同一会话、两轮、决策链完整”的局部要求，但**不满足用户指定的 `session_20261016_235451_805278`**。

`session_20261016_235451_805278` 仍被多处清单引用：

- `artifacts/start_v3_rag_response.json`：会话启动占位，`runRoot` 指向该目录。
- `artifacts/confirm_hypotheses_v3.json`、`artifacts/confirm_hypotheses_r2_v3.json`：人工确认记录中的 `humanNotes` 引用该会话。
- `artifacts/legacy_v3_migration/legacy_v3_migration_manifest.json`、`artifacts/legacy_v3_migration/import_dry_run/*.json`、`artifacts/legacy_v3_migration/outer_backtest/*.json`：外部迁移/回测清单以该目录为 `session_root`。
- `artifacts/v4_flat_prediction_diagnosis/manifest.json`：诊断数据源路径引用该目录。
- `artifacts/hypothesis_tree_snapshots/hypothesis_tree_full_view_manifest.json`：假设树快照清单引用该目录。

这些引用说明该会话曾计划被建立，但实际目录没有保留下来；引用文件本身不能替代 session 目录中的运行产物。

## 4. 已核实的展示边界

### `_rebuild_probe` 两轮不是两次新实验

- `round_01/E_R01_01_baseline.csv` 与 `round_02/E_R02_01_baseline.csv` 的 SHA256 都是 `142FB176EEE0E3C9ED7620AB5B7AA943C612FA903664A7EB232B8FDA0A450320`。
- 两个 treatment CSV 的 SHA256 都是 `C9BF05CA62D5B71D87B9D82C2732248BF661F882EFE423766175ECC3CC703C6B`。
- 因此第 2 轮指标与第 1 轮完全相同：ΔPearson r `+0.017504022167376176`，ΔRMSE `-0.8262662807156218`。两轮只能解释为“同一预测产物被记录两次”，不是第二轮复现或独立回测。

### `E_R01_03 / E_R02_01 / E_R02_03` 的文本引用与执行 ID 不一致

- `artifacts/approve_round1_v3.json` 批准的是候选 ID `E_R01_03`：BX 基线、窗口 14、历史 3 天、提前 3 天；`_rebuild_probe` 没有该 ID 的 predictions / result / evaluation。
- `artifacts/approve_round2_e01.json` 批准的是候选 ID `E_R02_01`：风速历史基线、窗口 3/3/14；`_rebuild_probe` 虽有 `E_R02_01` 目录，但实际协议、参数、预测与 `E_R01_01` 完全一致，且不含 `SW Plasma Speed` 历史特征。
- `docs/group_meeting_evidence_chain.md` 第 2 节写的是 `E_R01_03` 与 `E_R02_03`（7/5/21 窗口），与 `_rebuild_probe` 的 `E_R01_01 / E_R02_01` 都不一致。这是文本层的候选 ID 混用，不能当作正式执行 ID 的证据链。

### 稳健性与评分字段

- `evaluation_unified.json` 原文为 `robustness.bootstrap.stable=true`、`n_iterations=0`；`time_slice.splits=""`、`stable=true`。`n_iterations=0` 表示 bootstrap 未运行，`stable=true` 不能解释为稳定性验证通过。
- 当前 `state/candidate_experiments.json` 是 round 4 草案；`state/planner_output.json` 是 round 3 计划；`state/decision_log.json` 的 D022 只请求审批 `E_R03_01`，未被批准、未执行。
- 旧第二轮 `IG=1`、`U=0.4466` 的候选评分原始快照未能恢复，不补造。

## 5. 可追溯的替代材料

- 两轮实际归档：`runtime/live_session/_rebuild_probe/results/round_01|round_02`（protocol、result_unified、evaluation_unified、逐日预测 CSV）。
- 假设树、不确定性队列、决策日志：`runtime/live_session/_rebuild_probe/state/`。
- 已归档证据包说明与缺口清单：`artifacts/final_defense_evidence_chain/README.md`、`gaps_inventory.md`、`final_cross_round_table.md`。
- 外部迁移诊断与数据指纹：`artifacts/legacy_v3_migration/`、`artifacts/v4_flat_prediction_diagnosis/manifest.json`。
- `E_R03_01` 逐变量稀疏滞后的执行器隔离验收：`artifacts/legacy_v3_migration/executor_acceptance/executor_sparse_lags_acceptance.md`。

## 6. 后续动作建议

1. 若正式会话 `session_20261016_235451_805278` 仍可能在其他备份位置存在，先恢复原始目录再整理证据包；找不到就明确标注“会话不存在，证据包使用 `_rebuild_probe` 替代”。
2. 展示时把所有实验 ID 写成“会话 ID＋轮次＋实验 ID”的完整形式，不以候选 ID 代替执行 ID。
3. 两轮需要独立证据时，应在新会话中实际执行两轮，而不是继续引用 `_rebuild_probe` 的重复预测文件。
