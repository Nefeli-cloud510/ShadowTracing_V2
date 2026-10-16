# 逐影 v3 图包 v1.0.0

生成时间：2026-10-16T14:12:10+08:00

来源运行：E_R02_03_r1，轮次 round_02，测试期 2021-08-26 至 2021-09-29，
测试共同样本 n=35，预测提前量由归档配置决定。

## 图片输出格式版本

- figure_format_version：v3-fig/1.0
- 画布：PNG，dpi=200，白底
- 字体：Microsoft YaHei / SimHei / Microsoft JhengHei
- 日期轴：真实日历日期，缺失日期不压缩、不插值
- 预测只加误差棒：只有实际计算过预测区间时才会绘制
- 主图配色：真实值深灰，基线与加入日影分别蓝、橙
- 逐日绝对误差变化：|实际-基线| − |实际-日影|，正值表示当日日影误差更小

## 当前 v3 闭环模型参数

- ridge_baseline: Ridge alpha=1000, 选参=auto, 收敛=否
- ridge_treatment: Ridge alpha=1000, 选参=auto, 收敛=否
- elasticnet_baseline: ElasticNet alpha=10, l1_ratio=0.3, 选参=auto, 收敛=是
- elasticnet_treatment: ElasticNet alpha=10, l1_ratio=0.1, 选参=auto, 收敛=是

## 文件清单

- figures/v3_gain_ci.png
- figures/v3_test_prediction_comparison.png
- figures/v3_daily_abs_error_delta.png（新增）
- figures/v3_combined_overview.png（新增组合版）
- figures/v3_appendix_full_calendar_overview.png
- figures/v3_appendix_training_quality.png
- figures/v3_appendix_parameters.png / .csv
- v3_daily_abs_error_delta.csv
- v3_model_parameters.csv
- version.json

所有图均从同一运行 E_R02_03_r1 的归档预测与指标计算，指标不来自 LLM 文本。
