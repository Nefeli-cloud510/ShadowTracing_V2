"""Shared constraints injected into scientific LLM role prompts."""

ELASTIC_NET_EXECUTION_RULE = (
    "当前实验执行层接入 Ridge 与 ElasticNet 两个线性正则化模型家族，"
    "可调项包括 `alpha`、`l1_ratio`、特征组合、历史滞后窗口与预测超前窗口等。"
    "Ridge 与 ElasticNet 都属于线性回归模型，"
    "只能用于检验线性模型中的正则化方式与模型设置敏感性，"
    "不能据此验证自变量与因变量之间的非线性机制。"
    "你生成的所有假设、质询、不确定性、候选实验与结果解释，"
    "必须只基于数据变量库中的真实物理量，"
    "并能被当前 Ridge / ElasticNet 执行流程实际执行、量化和比较（Pearson_r / RMSE）；"
    "必须符合物理规律，禁止发明变量、虚构数据来源或设计当前流程无法验证的机制；"
    "对暂不支持的实验类型或非线性机制，必须明确说明能力边界，"
    "禁止改名后套用普通增量实验执行。"
)
