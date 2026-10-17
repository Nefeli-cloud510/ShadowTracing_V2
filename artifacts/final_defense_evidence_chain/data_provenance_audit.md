# 数据来源与执行产物审计

本文件只做只读核验，不重跑实验，不修改实时会话目录。统一前缀为：

- 绝对路径：`D:\ShadowTracing\runtime\live_session\_rebuild_probe\`
- 仓库相对路径：`runtime/live_session/_rebuild_probe/`

## 1. 任务声明的数据源

任务定义见 `state/task.json`，数据字典见 `state/data_dictionary.json`：

| 项目 | 原始记录 | 审计状态 |
| --- | --- | --- |
| OMNI 源表 | `data/raw/out_of_ICME_raw_OMNI_Daily_DATA_1_.xlsx`，时间列 `TIME`，目标列 `SW Plasma Speed, km/s` | 只记录任务声明，未在本证据包内重读原 Excel |
| LHAASO 日影源表 | `data/raw/raw_out_of_ICME_sunshadow2.xlsx`，时间列 `TIME` | 只记录任务声明，未在本证据包内重读原 Excel |
| 目标字段 | `SW Plasma Speed, km/s`，物理含义“太阳风速度”，`is_target=true`，`category=target` | 与两轮协议目标一致 |
| 核心解释变量 | `p2_y中心值`，物理含义“宇宙线日影南北偏移”，`category=core_explanatory` | 与 treatment 特征一致 |
| 样本规模 | `data_dictionary.json` 记 `total_samples=364` | 未在本证据包内重算 |
| 时间范围 | `data_dictionary.json` 记 `TIME 20210321..20211002` | 未在本证据包内重算 |

原始来源记录在 `state/task.json -> payload.data_sources`；字段说明记录在 `state/data_dictionary.json -> fields` 与 `user_supplementary`。

## 2. 两轮协议使用同一套输入

两轮 `protocol.json` 中的共同输入相同：

| 输入 | 值 |
| --- | --- |
| target | `SW Plasma Speed, km/s` |
| forecast_horizon_days | 3 |
| control 字段（6 个） | `Scalar B, nT`、`BY, nT (GSE)`、`BZ, nT (GSE)`、`BY, nT (GSM)`、`BZ, nT (GSM)`、`Kp*10 index` |
| treatment 字段（7 个） | control 6 个 + `p2_y中心值` |
| model | `ElasticNet` |
| alpha | 0.3 |
| l1_ratio | 0.5 |
| lags | `{}` |
| window_size | 3 |
| use_lag_feature | false |
| max_lag_day | 0 |
| random_state | 42 |
| test_split_ratio | 0.2 |

协议原文：

- `results/round_01/protocol.json`
- `results/round_02/protocol.json`

数据字典中还存在 `p1_x中心值`（宇宙线日影东西偏移）、`Dst-index, nT`、`ap_index, nT`、`f10.7_index` 等字段，本会话两轮协议均未使用，不能凭数据字典推断它们进入了模型。

## 3. 特征展开口径（关键限制）

`lags={}` 与 `window_size=3` 按执行器口径展开为 lag `0,1,2`，不是逐变量稀疏滞后。因此：

- 本会话实际执行的不是冻结版 `E_R03_01` 协议中的“BY 4 项＋BZ 2 项＋Scalar B 2 项＋p2_y lag1＋p1_x lag9”结构。
- 不能把本会话的模型输入当作稀疏滞后版特征。
- 预测 CSV 只有 `time,y_true,y_pred` 三列，没有保存 `issue_date/feature_date/target_date`，不能从产物逐样本重建发报日、特征日和目标日的隔离核对。
- `protocol.json -> constraints.no_future_information=true` 是协议声明，不等于本证据包已经独立验证每个样本在发报时刻的特征可用性。

## 4. 预测 CSV 核验

两个轮次共 4 个逐日预测文件：

- `results/round_01/E_R01_01/predictions/E_R01_01_baseline.csv`
- `results/round_01/E_R01_01/predictions/E_R01_01_treatment.csv`
- `results/round_02/E_R02_01/predictions/E_R02_01_baseline.csv`
- `results/round_02/E_R02_01/predictions/E_R02_01_treatment.csv`

| 文件 | 行数 | 日期范围 | 缺失日期 | SHA256 |
| --- | ---: | --- | --- | --- |
| round_01 baseline | 36 | 2021-08-25 .. 2021-10-02 | 2021-08-27、2021-09-05、2021-09-21 | `142FB176EEE0E3C9ED7620AB5B7AA943C612FA903664A7EB232B8FDA0A450320` |
| round_01 treatment | 36 | 2021-08-25 .. 2021-10-02 | 同上 | `C9BF05CA62D5B71D87B9D82C2732248BF661F882EFE423766175ECC3CC703C6B` |
| round_02 baseline | 36 | 2021-08-25 .. 2021-10-02 | 同上 | `142FB176EEE0E3C9ED7620AB5B7AA943C612FA903664A7EB232B8FDA0A450320` |
| round_02 treatment | 36 | 2021-08-25 .. 2021-10-02 | 同上 | `C9BF05CA62D5B71D87B9D82C2732248BF661F882EFE423766175ECC3CC703C6B` |

两轮同臂文件逐字节一致，因此两轮指标完全一致。

## 5. 评价段标签分布

从逐日 CSV 直接统计：

| 量 | 取值 |
| --- | ---: |
| y_true 最小值 | 315 km/s |
| y_true 最大值 | 514 km/s |
| y_true 均值 | 390.11 km/s |
| y_pred 最小值 | 328.36 km/s |
| y_pred 最大值 | 460.01 km/s |
| y_pred 均值 | 404.21 km/s |
| 共同成对样本 | 36 |

预测均值域明显窄于真实值，说明预测曲线偏平；该项是预测表现限制，不是作图方式造成的。

## 6. 指标可重算性

`results/round_01/result_unified.json`、`results/round_01/evaluation_unified.json`、`results/round_02/result_unified.json`、`results/round_02/evaluation_unified.json` 中的指标一致：

| 指标 | baseline | treatment | 增量 |
| --- | ---: | ---: | ---: |
| Pearson r | 0.39448958586943433 | 0.4119936080368105 | +0.017504022167376176 |
| RMSE, km/s | 49.25384857191404 | 48.42758229119842 | -0.8262662807156218 |
| MAE, km/s | 39.5986254968859 | 38.40749317745875 | -1.1911323194271546 |
| R2 | -0.05852141690524881 | -0.023304498564322484 | +0.03521691834092633 |
| pg_actual_signed / clipped | - | - | 0.016775669408030432 |

没有计算配对时间块 bootstrap 的 Δr / ΔRMSE 区间，`gaps_inventory.md` 已列为缺口。

## 7. 未在本证据包内确认的内容

- 原始 Excel 字段是否仍存在错位，没有在本会话中重新逐列核对。
- 清洗转换后运行 CSV 与原始记录的逐日映射，没有在本证据包内重做。
- 标准化、缺失值标记、日期排序、重复日期、滞后构造和训练内选参是否严格满足时间隔离，没有在本证据包内重新验证。
- 训练均值、持续性、风速历史、风速历史＋日影等简单基线均 `not_executed`。

以上任一项都不能凭本文件自动判定为通过。
