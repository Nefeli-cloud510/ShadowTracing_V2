# 知识链审计报告：上传论文 → 读取 → 假设证据

审计时间：2026-10-14
审计会话：`runtime/live_session/sessions/session_20261013_202258_798542`

## 结论

断点属实：在此之前，网页上传的 PDF 只落盘了原始文件和一个占位文本，
`extraction_status` 为 `placeholder_only`，正文从未进入本地文献检索，
也没有进入远程百炼知识库。假设树中的证据因此反复出现
“当前上下文未提供可核验文献条目”。

本次已经完成两件事：新上传路径改为真实提取 PDF 正文；历史会话通过
`scripts/re_extract_knowledge_pdfs.py` 完成补提取。补提取后，本地
`RAGService.search_literature_knowledge()` 已能命中真实文章段落并返回可核对
的文件引用。

仍有两个缺口：远程百炼知识库完全未配置；已有假设树中写死的占位证据不会
自动追溯替换，需要一轮显示刷新或人工确认后由新的 RAG 上下文重新写入。

## 链路现状

```text
前端上传
  -> uploads/knowledge/<file>（原始 PDF 保留）
  -> outputs/pdf_texts/<stem>.txt（真实提取文本）
  -> RAGService.search_literature_knowledge()
  -> RAGContextBundle.literature_evidence
  -> 假设提出者 / 科学质询者 / 实验规划者提示词
```

关键调用点：

```text
core/rag_service.py: build_context_bundle(), search_literature_knowledge()
core/control_unified.py: 787 / 2820 / 2894 构建 RAG 上下文
core/hypothesis_proposer_llm.py / scientific_questioner_llm.py: 接收 RAGContextBundle
scripts/live_workflow_server.py: ingest_knowledge_files() 执行提取
```

## 文件明细

| 文件 | 原始大小 | 页数 | 提取字符数 | 引擎 | 修复后状态 |
|---|---:|---:|---:|---|---|
| 2013-ICRC-HAWC_and_Solar_Energetic_Transient_Events.pdf | 1.1 MB | 3 | 12,277 | pdfplumber | extracted |
| 2015-ICRC_Solar_Event_Simulations_using_the_HAWC_Scaler_System.pdf | 203 KB | 7 | 15,191 | pdfplumber | extracted |
| 2015-VAMOS-_A_pathfinder_for_the_HAWC_gamma-ray_observatory.pdf | 2.0 MB | 9 | 40,354 | pdfplumber | extracted |
| LHAASO-the_Inovation-Monitoring-daily_variation_of_IMF_1_.pdf | 4.2 MB | 7 | 37,560 | pdfplumber | extracted |
| Solar_and_Wind_Power_Forecasting.pdf | 470 KB | 9 | 19,220 | pdfplumber | extracted |
| 南云程-利用LHAASO实验的宇宙线日影研究太阳及日地空间磁场_1_.pdf | 8.0 MB | 162 | 132,251 | pdfplumber | extracted |

修复前这 6 个 `outputs/pdf_texts/*.txt` 都只有 123–223 字节的占位文本；原始 PDF
一直保留在 `uploads/knowledge/`，因此可以做无损补提取。补提取前的清单备份保存在
会话目录
`state/upload_manifest.pre_extract.20261014_110825.json`。

## 本地检索验证

用查询“LHAASO 宇宙线 日影 太阳风 行星际磁场”调用当前会话的本地文献检索：

```text
results = 2
南云程-利用LHAASO实验的宇宙线日影研究太阳及日地空间磁场_1_.txt | score = 1.0
LHAASO-the_Inovation-Monitoring-daily_variation_of_IMF_1_.txt | score = 0.2
```

返回的 `title` 是实际文件名，`citation` 是相对会话目录的 `outputs/pdf_texts/*.txt`
路径，`excerpt` 可追溯到文章中命中位置。也就是说，**本地“读取论文”断点已经打通**。

## 远程百炼知识库断点

当前 `.env` 只配置了：

```env
DASHSCOPE_API_KEY
DASHSCOPE_BASE_URL
BAILIAN_MODEL
```

`RAGService.search_bailian_knowledge_base()` 要真正发起远程检索，需要
`DASHSCOPE_API_KEY`、`BAILIAN_KNOWLEDGE_AGENT_ID` 和有效的
`BAILIAN_KNOWLEDGE_SEARCH_ENDPOINT`（或 `BAILIAN_WORKSPACE_ID` 推导默认端点）。
这些字段全部为空，因此远程命中数必然为 0。

另外，上传清单当前统一标记
`"knowledge_sync_status": "local_session_only"`，系统没有把上传文件写回百炼托管
知识库。即使以后配置远端，也应该先明确“是否允许自动上传”的权限边界。

## 入口提示：新会话与旧会话

新会话：只要重启后端使 `ingest_knowledge_files()` 的新代码生效，新上传 PDF 会直接
提取并标记为 `extracted`。

旧会话：使用

```powershell
python scripts/re_extract_knowledge_pdfs.py --root <session_root>
```

即可对尚未提取的 PDF 做同样处理，脚本会先备份 `upload_manifest.json`。

## 后续建议

1. 在假设证据结构中增加 `literature_source`（文件名、页面、检索分、原文片段），
   让“哪篇论文的哪句话支持了哪条假设”可以展示和核验。
2. 对已有会话增加一次显式“证据刷新”：用真实 RAG 上下文重写
   “当前上下文未提供可核验文献条目”类型的占位证据，而不是只让下一轮新证据使用。
3. 远程知识库若启用，网页端应持久化每次检索的 query、命中数、来源、耗时，
   形成 `rag_telemetry.json`，否则现网会话无法证明远程知识库真的参与过规划。
4. 对图片型扫描 PDF 需要 OCR 兜底；当前 6 个文件均能用 pdfplumber 提取，但
   后续如果上传纯扫描件，`extraction_status` 会如实保留 `placeholder_only`。
