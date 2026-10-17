# 执行器稀疏滞后协议验收报告（2026-10-17）

## 结论

最后两项执行器验收通过：修复版 `E_R03_01` 能从候选草案原样转成正式执行协议并完成隔离执行；`bootstrap_robustness` 与 `selection_robustness` 保持为不可执行草案，不会被静默降级。

验收档案：`artifacts/legacy_v3_migration/executor_acceptance/executor_sparse_lags_acceptance.json`

## 1. `E_R03_01` 协议编译

- 实验类型：`model_sensitivity`
- 运行组：`ridge_baseline`、`ridge_treatment`、`elasticnet_baseline`、`elasticnet_treatment`
- 对比：`gain_ridge`、`gain_elasticnet`
- 目标变量：`SW Plasma Speed, km/s`

冻结的逐变量稀疏滞后原样保留：

| 特征 | 滞后 |
| --- | --- |
| `BY, nT (GSE)` | `[4,5,7,8]` |
| `BZ, nT (GSE)` | `[1,2]` |
| `Scalar B, nT` | `[1,2]` |
| `p2_y中心值` | `[1]` |
| `p1_x中心值` | `[9]` |

时间参数：`forecast_horizon_days=3`、`past_lag_days=9`、`window_size=9`。

## 2. 实际执行特征

在隔离临时项目中完成 4 个 run 的完整执行，归档 `run_id=E_R03_01_r1`。

- 磁场基线：每个模型 8 项特征，与冻结配置一致。
- 加双方向日影：每个模型 10 项特征，包含 `p2_y中心值_t-1` 与 `p1_x中心值_t-9`。
- 评价计划：`split_label=dev_backtest`，测试起点 `2021-08-26`。
- 执行状态：`completed`。

本次验收复制 v3 CSV 到临时目录执行，未改动实时会话目录。

## 3. 未注册类型检查

`bootstrap_robustness`、`selection_robustness` 均不通过：

- 不在 `ALL_EXPERIMENT_TYPES`
- 不在候选可选类型
- 无对应执行工具
- 映射器抛出“实验类型未注册，禁止静默降级为普通增量实验”错误

## 4. 边界说明

本次验收只验证“候选协议原样转正式协议并执行”：

- 未调用大模型规划。
- 当前执行器自动覆盖 dev 单窗口；`f1/f2` 是冻结外部回测清单中的只读参照，三窗口自动执行尚未接通。
- 不据此宣布跨窗口一致支持或任何性能结论。

## 5. 验收中发现并修复的问题

首次完整运行返回 false，根因不在执行器，而在验收脚本的期望集合构造：`set.update(字符串)` 会把完整特征名拆成单个字符。已改为 `set.add(...)`，修正后完整验收 `all_passed=True`。

## 6. 证据出处

- 验收脚本：`scripts/acceptance_executor_sparse_lags.py`
- 修复候选：`artifacts/legacy_v3_migration/import_dry_run/round_03_candidates_dry_run.fixed.json`
- 验收档案：`artifacts/legacy_v3_migration/executor_acceptance/executor_sparse_lags_acceptance.json`
- 验收档案生成时 HEAD：`65f3b98`
- 归档提交：`08a15dc fix: enforce per-variable sparse lags and reject unregistered draft types`
