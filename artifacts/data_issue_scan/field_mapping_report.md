# 2021 数据错位修复：字段映射与核验报告

> 结论：live 使用的 v2 两张 CSV 存在系统性错位。v3 已按用户指定的新全日历
> `D:\sunshadow\calender_internal_2020_2024.xlsx` 和
> `D:\astro_rearrange\data\analysis\pfss_basic_features_daily_v1.csv`
> 重建，并逐字段精确核验。旧 v2 文件保留不改，新文件另存为 v3。

## 1. OMNI：v2 列实际装了什么、v3 装回了什么

| v2 原列名 | v2 中实际物理量 | v3 正确物理量 | 单位 | v3 来源 | 核验方式 |
|---|---|---|---|---|---|
| `Scalar B, nT` | 全为 0（被置零） | 磁场总强度 Scalar B | nT | 全日历 xlsx | 196/196 与日历精确一致 |
| `BX, nT (GSE, GSM)` | Scalar B | BX | nT | 全日历 xlsx | 196/196 精确一致 |
| `BY, nT (GSE)` | BX | BY (GSE) | nT | 全日历 xlsx | 196/196 精确一致 |
| `BZ, nT (GSE)` | BY (GSE) | BZ (GSE) | nT | 全日历 xlsx | 196/196 精确一致 |
| `BY, nT (GSM)` | BZ (GSE) | BY (GSM) | nT | 全日历 xlsx | 196/196 精确一致 |
| `BZ, nT (GSM)` | BY (GSM) | BZ (GSM) | nT | 全日历 xlsx | 196/196 精确一致 |
| `SW Plasma Temperature, K` | RMS_BZ_GSE 等小值量 | 等离子体温度 | K | 全日历 xlsx | 196/196 精确一致 |
| `SW Proton Density, N/cm^3` | 等离子体温度 | 质子密度 | /cm^3 | 全日历 xlsx | 196/196 精确一致 |
| `SW Plasma Speed, km/s` | 质子密度 | 太阳风速度 | km/s | 全日历 xlsx | 196/196 精确一致 |
| `Kp index` | Flow pressure | Kp 指数 | 0-90 尺度 | 全日历 xlsx | 196/196 精确一致 |
| `Dst-index, nT` | Kp 指数 | Dst 指数 | nT | 全日历 xlsx | 196/196 精确一致 |
| `ap_index, nT` | Dst 指数 | ap 指数 | nT | 全日历 xlsx | 196/196 精确一致 |

该表同时得到相关核验支撑：v2 每个错位列与权威列的最大相关均为
`corr = 1.0`，说明是整表平移，不是随机错误。

## 2. LHAASO 日影：v2 列实际装了什么、v3 装回了什么

| v2 原列名 | v2 中实际物理量 | v3 正确物理量 | 单位 | v3 来源 | 核验方式 |
|---|---|---|---|---|---|
| `p2_y中心值` | p1_x 中心值误差 | p2_y 中心值 | deg | 全日历 xlsx | 188/188 可比日精确一致 |
| `p2_y中心值误差` | p2_y 中心值 | p2_y 中心值误差 | deg | 全日历 xlsx | 188/188 精确一致 |
| `p1_x中心值` | p0 误差 | p1_x 中心值 | deg | 全日历 xlsx | 188/188 精确一致 |
| `p1_x中心值误差` | p1_x 中心值 | p1_x 中心值误差 | deg | 全日历 xlsx | 188/188 精确一致 |
| `p3 sigma_宽度` | p2_y 中心值误差 | p3 sigma 宽度 | deg | 全日历 xlsx | 188/188 精确一致 |
| `_llf 最优卡方值` | p3 sigma 宽度误差 | 最优卡方值 | - | 全日历 xlsx | 188/188 精确一致 |

其中 p1_x/p2_y/p3/_llf 六列的错位也由 `corr = 1.0` 的相关核验确认。

## 3. PFSS 9 列

`l0_04utc_deg`、`photo_br_mean/photo_br_absmean/photo_n_lat/photo_n_lon`、
`ss_br_mean/ss_br_absmean/ss_n_lat/ss_n_lon` 在 v2 中已经与 v3 完全一致；
v3 按用户指定的 `pfss_basic_features_daily_v1.csv` 重新写入，
196/196 天逐字段精确一致。

## 4. 物理量与关系检查

执行前健康检查通过：

| 检查 | 结果 |
|---|---|
| `|Bx| <= Scalar B` | 0 失败 |
| `|By_GSE| <= Scalar B` | 0 失败 |
| `|Bz_GSE| <= Scalar B` | 0 失败 |
| 风速 150-1500 km/s | 0 失败 |
| 密度 0.01-100 /cm^3 | 0 失败 |
| 温度 1e4-5e6 K | 0 失败 |

这组检查直接落到执行链路：`models/elasticnet_runner.py` 的
`load_and_merge_data()` 现在会调用 `core/data_provenance.py`，
任一标准列出现越界或置零都会抛出错误并阻断实验。

## 5. 与另一份独立 OMNI 原始表的差异说明

v3 与 `D:\helioprior\data\out_of_ICME_raw_OMNI_Daily_DATA(1).xlsx`
比较时：

- 风速 182/182 精确一致。
- 温度、密度有 9/182 天不一致，集中在 20210828-20210903 和
  20210922-20210923；该原始表在这些日期的值接近日历表前一天的值，
  判断为原始表内部该段的单日错位，不是 v3 错位。
- 以用户指定的新全日历 `calender_internal_2020_2024.xlsx` 为准。

## 6. 修复产物与审计文件

修复后另存 / 生成的路径：

| 用途 | 文件 |
|---|---|
| OMNI v3 | `D:\ShadowTracing\runtime\live_run_input\omni_2021_full_calendar_v3.csv` |
| LHAASO+PFSS v3 | `D:\ShadowTracing\runtime\live_run_input\lhaaso_shadow_pfss_2021_full_calendar_v3.csv` |
| 字段逐项核验 | `D:\ShadowTracing\artifacts\data_issue_scan\field_verification_report.json` |
| v3 验收报告 | `D:\ShadowTracing\artifacts\data_issue_scan\v3_acceptance_report.json` |
| 物理量关系检查 | `D:\ShadowTracing\artifacts\data_issue_scan\v3_physical_relation_check.json` |
| 数据来源记录 | `D:\ShadowTracing\artifacts\data_issue_scan\v3_data_provenance.json` |
| v2/v3 逐日对照 | `D:\ShadowTracing\artifacts\data_issue_scan\omni_live_vs_authoritative_merged.csv` |
| v2/v3 逐日对照 | `D:\ShadowTracing\artifacts\data_issue_scan\shadow_live_vs_authoritative_merged.csv` |
| 本字段映射报告 | `D:\ShadowTracing\artifacts\data_issue_scan\field_mapping_report.md` |
