# Final Demo Run Guide

This guide explains how to switch from local fallback mode to real Bailian
model / knowledge-base mode, and how to run the two-round closed-loop demo.

## 1. Where To Put The API Configuration

Create a new file at:

`D:\Shadow-Tracing-simple\.env`

The easiest way is:

1. Copy `D:\Shadow-Tracing-simple\.env.example`
2. Rename the copy to `.env`
3. Fill the values in that `.env`

The runtime loader is already wired in:
- `core/runtime_config.py`
- `core/llm_gateway.py`
- `core/rag_service.py`

You do **not** need to hardcode keys into Python files.

## 2. How To Configure The Bailian Model Key

Open:

`D:\Shadow-Tracing-simple\.env`

Fill these lines:

```env
DASHSCOPE_API_KEY=你的百炼模型Key
BAILIAN_MODEL=qwen-plus
```

Optional:

```env
DASHSCOPE_BASE_URL=https://llm-jz60biyiqkkwzssm.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
BAILIAN_API_KEY=你的百炼模型Key
```

Recommended rule:
- Prefer filling `DASHSCOPE_API_KEY`
- `BAILIAN_API_KEY` is kept as an alias for compatibility

Where it is consumed:
- `core/llm_gateway.py`

## 3. How To Configure The Knowledge Base ID

Open:

`D:\Shadow-Tracing-simple\.env`

Fill these lines:

```env
BAILIAN_WORKSPACE_ID=你的业务空间ID
BAILIAN_KNOWLEDGE_AGENT_ID=你已发布的知识检索服务ID
BAILIAN_KNOWLEDGE_AGENT_VERSION=
```

Where it is consumed:
- `core/rag_service.py`

Current request contract:

```json
{
  "agent_id": "aid-xxxxxxxxxxxxxxxx",
  "query": "DeltaDec Vsw By",
  "images": []
}
```

Current default request URL:

```text
https://{workspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/indices/knowledge/search
```

Current expected response contract:

```json
{
  "items": [
    {
      "id": "doc_001",
      "title": "title",
      "excerpt": "retrieved snippet",
      "score": 0.91,
      "citation": "source info"
    }
  ]
}
```

Also accepted:

```json
{
  "data": [
    {
      "id": "doc_001",
      "title": "title",
      "excerpt": "retrieved snippet",
      "score": 0.91,
      "citation": "source info"
    }
  ]
}
```

If your actual Bailian knowledge-base API format differs, adjust only:
- `RAGService._search_remote_knowledge_base()`

## 4. How To Switch `programmatic / llm`

Current switch point:
- `core/control_unified.py`
- `HumanControlService(..., planner_mode="programmatic" | "llm")`

Examples:

Programmatic mode:

```python
control = HumanControlService(repository, project_root=project_root, planner_mode="programmatic")
```

LLM mode:

```python
control = HumanControlService(repository, project_root=project_root, planner_mode="llm")
```

What changes in `llm` mode:
- enables `central_controller_llm`
- enables `scientific_interpreter_llm`
- enables `RAGService`
- enables `hypothesis_proposer_llm`
- enables `scientific_questioner_llm`
- enables `experiment_planner_llm`

What does **not** change:
- PI approval is still required
- protocol execution still goes through validated schema
- the audit chain is still written to `state/*.json`

## 5. How To Run The Two-Round Closed-Loop Demo

### Option A: Run the stubbed-agent closed-loop demo

Run:

```powershell
python -m unittest tests.test_full_llm_closed_loop -v
```

This uses stub agents (no real API key required) and verifies:
- `RAG -> proposer/questioner -> hypothesis_generation -> candidate -> protocol -> execution -> evaluation -> scientific_interpreter -> next planner_input -> next candidate`

It is an integration test, not a real-LLM test. Real DashScope/Bailian
coverage lives in Option C.

### Option B: Run all regression tests

Run:

```powershell
python -m unittest discover -s tests -v
```

### Option C: Run the optional real API smoke test

First set in `D:\Shadow-Tracing-simple\.env`:

```env
ENABLE_REAL_API_TESTS=1
```

Then run:

```powershell
python -m unittest tests.test_real_api_optional -v
```

Behavior:
- If key is missing, model smoke test is skipped
- If knowledge-base endpoint / id is missing, KB smoke test is skipped
- If all are configured, the test hits the real remote API

## 6. Files That Record The Demo Audit Trail

After the two-round demo, inspect:
- `state/hypothesis_tree.json`
- `state/uncertainties.json`
- `state/experiment_memory.json`
- `state/decision_log.json`
- `state/planner_input.json`
- `state/planner_output.json`

These files contain:
- hypothesis evolution
- uncertainty evolution
- LLM reasoning traces
- planner mode
- protocol refinements
- human decision checkpoints

## 7. Recommended Final Replacement Order

1. Fill `.env`
2. Verify `tests.test_real_api_optional`
3. Replace any remaining stubs in ad-hoc demo scripts
4. Run `tests.test_full_llm_closed_loop`
5. Review `state/*.json` audit outputs

## 8. If The Real API Call Fails

Check in this order:

1. `.env` is placed at `D:\Shadow-Tracing-simple\.env`
2. `DASHSCOPE_API_KEY` is not empty
3. `BAILIAN_WORKSPACE_ID` is not empty
4. `BAILIAN_KNOWLEDGE_AGENT_ID` is not empty
5. `BAILIAN_KNOWLEDGE_SEARCH_ENDPOINT` is reachable only if you override the default URL
6. `ENABLE_REAL_API_TESTS=1` only when you really want remote smoke tests
