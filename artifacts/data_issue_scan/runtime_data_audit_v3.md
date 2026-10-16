# 运行数据审计与第 2 轮零增益归因

> 审计对象：`session_20261013_202258_798542` 第 2 轮实验 `E_R02_03`
> 运行输入：`runtime/live_run_input/` 下的 v2/v3 文件
> 结论：第 2 轮确实仍在使用 v2 错位数据，因实际预测目标是质子密度，
> 该轮结果不能作为太阳风速度证据；零增益的直接原因是实验组日影特征
> 全部被 ElasticNet 正则化压为 0，不是显示精度问题。

## 1. 第 2 轮实际使用的数据版本

1. 会话数据目录只有 v2 文件，没有 v3 文件：
   - `runtime/live_session/sessions/session_20261013_202258_798542/data/raw/omni_2021_full_calendar_v2.csv`
   - `runtime/live_session/sessions/session_20261013_202258_798542/data/raw/lhaaso_shadow_pfss_2021_full_calendar_v2.csv`
2. `E_R02_03_r1/data_manifest.json` 记录来源为：
   - `data/raw/omni_2021_full_calendar_v2.csv`
   - `data/raw/lhaaso_shadow_pfss_2021_full_calendar_v2.csv`
3. `E_R02_03_r1/DATA_ISSUE_INVALIDATION.json` 标记：
   - `status = invalidated_for_wind_speed_evidence`
   - `declared_target = SW Plasma Speed, km/s`
   - `actual_prediction_target = SW Proton Density, N/cm^3`
   - `root_cause = v2 OMNI 整列错位`
4. 预测文件显示真实标签量级：
   - `y_true` 范围 2.6 到 16.5，量级符合质子密度（N/cm^3）
   - 预测范围 6.13 到 7.06
   - 若目标是太阳风速度，真实值应为约 300 到 900 km/s

结论：`E_R02_03` 的训练、特征构造和评价标签全部来自 v2 错位列；
报告声明目标为 `SW Plasma Speed, km/s`，实际建模对象是质子密度。

## 2. v3 运行输入审计

修复后文件：

| 用途 | 文件 | SHA-256 |
| --- | --- | --- |
| OMNI v3 | `D:\ShadowTracing\runtime\live_run_input\omni_2021_full_calendar_v3.csv` | `8106F44F06CE3A6E7BE8A51A6C44E2C004D31D08A63CC049AF60726DFD1496D5` |
| LHAASO+PFSS v3 | `D:\ShadowTracing\runtime\live_run_input\lhaaso_shadow_pfss_2021_full_calendar_v3.csv` | `338A06B01BF014364C31DAE4FDCAD527896487D8A2DB1F349F21B3DF54D6AE88` |

### 2.1 日期与覆盖

| 项目 | OMNI v3 | LHAASO+PFSS v3 |
| --- | --- | --- |
| 行数 | 196 | 196 |
| 日期范围 | 2021-03-21 至 2021-10-02 | 2021-03-21 至 2021-10-02 |
| 重复日期 | 0 | 0 |
| 缺失值 | 0 | 日影拟合 6 列各 8 天 NaN，PFSS 9 列无缺失 |

缺失日影拟合值的 8 天保留为 NaN，由样本层按共同日历处理，不提前整行删除。

### 2.2 字段与量纲

OMNI v3 列名对应物理量：

| 列名 | 物理量 | 单位 |
| --- | --- | --- |
| `SW Plasma Speed, km/s` | 太阳风速度 | km/s |
| `SW Proton Density, N/cm^3` | 质子密度 | /cm^3 |
| `SW Plasma Temperature, K` | 等离子体温度 | K |
| `Scalar B, nT` | 磁场总强度 | nT |
| `BX/BY/BZ ...` | 磁场分量 | nT |
| `Kp index` / `Dst-index, nT` / `ap_index, nT` | 地磁指数 | 见列名 |

LHAASO v3 日影列：

| 列名 | 物理量 | 单位 |
| --- | --- | --- |
| `p2_y中心值` | 日影南北偏移中心值 | deg |
| `p2_y中心值误差` | 对应误差 | deg |
| `p1_x中心值` | 东西偏移中心值 | deg |
| `p1_x中心值误差` | 对应误差 | deg |
| `p3 sigma_宽度` | 日影宽度 | deg |
| `_llf 最优卡方值` | 拟合卡方值 | 无量纲 |

### 2.3 运行输入统计

OMNI v3：

| 字段 | min | Q1 | 中位数 | Q3 | max |
| --- | --- | --- | --- | --- | --- |
| `SW Plasma Speed, km/s` | 298.0 | 346.75 | 380.5 | 422.5 | 615.0 |
| `SW Proton Density, N/cm^3` | 1.8 | 3.875 | 5.8 | 8.2 | 19.1 |
| `SW Plasma Temperature, K` | 13669 | 32541 | 50463 | 85158.75 | 273382 |
| `Scalar B, nT` | 1.7 | 3.6 | 4.6 | 5.925 | 12.6 |

LHAASO v3（可比日 188 天）：

| 字段 | min | Q1 | 中位数 | Q3 | max |
| --- | --- | --- | --- | --- | --- |
| `p2_y中心值` | -0.577 | -0.07875 | 0.0805 | 0.28825 | 2.65 |
| `p2_y中心值误差` | 0.043 | 0.067 | 0.079 | 0.101 | 0.5 |
| `ss_br_mean` | -0.046724 | -0.017115 | -0.000451 | 0.009899 | 0.053746 |

### 2.4 与权威数据逐字段核验

由 `artifacts/data_issue_scan/v3_acceptance_report.json` 记录：

| 对比 | 结果 |
| --- | --- |
| OMNI v3 vs 全日历 `calender_internal_2020_2024.xlsx` | 核心字段 196/196 天逐日精确一致 |
| LHAASO v3 拟合列 vs 全日历 | 可比日 188/188 逐日精确一致 |
| LHAASO v3 PFSS 9 列 vs `pfss_basic_features_daily_v1.csv` | 196/196 天逐日精确一致 |
| OMNI v3 风速 vs helioprior 原始表 | 182/182 天精确一致 |

helioprior 原始表中温度、密度的 9 天不一致，逐值核对后仍是同段单日错位：
例如该原始表 2021-09-08 的密度 `4.0` 等于 v3 2021-09-07 的密度，
连续 7 天如此；2021-09-23 的密度 `7.1` 等于 v3 2021-09-22 的密度。
判断为 helioprior 原始表该段内部日期偏移，不是 v3 错误；以用户指定的
新全日历为准。

### 2.5 物理范围与关系检查

`core/data_provenance.py::validate_merged_data()` 已接入
`models/elasticnet_runner.py::load_and_merge_data()`：

| 检查 | 结果 |
| --- | --- |
| `|Bx| <= Scalar B` | 0 失败 |
| `|By_GSE| <= Scalar B` | 0 失败 |
| `|Bz_GSE| <= Scalar B` | 0 失败 |
| 风速 150-1500 km/s | 0 失败 |
| 密度 0.01-100 /cm^3 | 0 失败 |
| 温度 1e4-5e6 K | 0 失败 |

任一标准物理列越界会抛错阻断实验。

## 3. 第 2 轮零增益的直接原因

### 3.1 协议与样本

- 实验类型：`incremental_prediction`
- 对照臂：`ss_br_mean`
- 实验臂：`ss_br_mean + p2_y中心值`
- 模型：ElasticNet，`alpha=0.3`、`l1_ratio=0.5`
- 窗口：21 天；训练 121 样本；共同测试 33 样本
- bootstrap：7 天块、4000 次、有效块 6 个

### 3.2 指标与逐样本预测

| 检查 | 值 |
| --- | --- |
| 训练样本 | 121 |
| 测试样本 | 33 |
| 两组 sample_id 是否一致 | True |
| 两组 y_true 是否一致 | True |
| 两组逐样本最大绝对预测差 | 0.0 |
| 预测完全不同的行数 | 0 |
| baseline RMSE / MAE / r / R2 | 3.596454 / 2.898880 / 0.064270 / -0.047355 |
| treatment RMSE / MAE / r / R2 | 完全相同 |
| RMSE 增益 CI | [0.0, 0.0] |

### 3.3 系数表

实验组 `实验组_coefficients.csv`：

| 特征 | 系数 |
| --- | --- |
| `ss_br_mean_t-6` | 0.191025 |
| `ss_br_mean_t` 至 `ss_br_mean_t-5` | 0.000000 |
| `p2_y中心值_t` 至 `p2_y中心值_t-6` | `-0.000000` / `0.000000` |

对照组 `对照组_coefficients.csv` 只保留 `ss_br_mean_t-6 = 0.191025`，
其余系数均为 0。

因此两组预测函数完全相同，不是小数位四舍五入造成的假零增益。
ElasticNet 在共同测试段没有给任何日影滞后项非零系数，实验组新增特征
没有任何预测贡献。

### 3.4 为什么这个结论不能用于太阳风科学判断

1. 第 2 轮实际目标标签是质子密度，不是太阳风速度。
2. 实验组使用的 `p2_y中心值` 在 v2 文件中实际是 `p1_x中心值误差`，
   不是日影南北偏移。
3. 该轮只有 33 个测试样本、单一正则化参数、单一窗口，且实验组日影
   特征被 L1 压零，属于模型设置和数据有效性双重未决。

正确表述：`E_R02_03` 只能说明错位 v2 数据、ElasticNet
`alpha=0.3/l1_ratio=0.5` 条件下没有检出“质子密度预测差”；
不能说明日影对太阳风速度预测无增量。

## 4. 旧结果失效范围

已经标记失效（保留原文件，不删除）：

- `E_R01_05`
- `E_R02_03`

失效影响：对应太阳风速度结论、由此调整的假设支持度、已解决/未解决
不确定性、后续规划依据均不能继续作为正式证据。相关记录见
`artifacts/data_issue_scan/legacy_evidence_invalidation.md`。

## 5. 下一项：v3 标准闭环

- 新建会话时必须上传 v3 文件，或依赖
  `scripts/live_workflow_server.py` 的 `resolve_corrected_data_files()`
  把 v2 上传名自动替换为 v3。
- 验收标准：第 1 轮协议目标 `SW Plasma Speed, km/s`；共同测试样本
  `y_true` 在约 300-900 km/s 区间；`data_manifest.json` 路径为
  `*_v3.csv`；指标不再出现“两组逐样本完全一致”的退化形式。
