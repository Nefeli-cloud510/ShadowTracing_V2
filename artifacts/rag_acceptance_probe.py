from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_real_api_two_round_demo import ObservedRAGService

root = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT
svc = ObservedRAGService(project_root=root)
print("configured:", svc.bailian_api_key is not None, svc.bailian_knowledge_agent_id, svc.bailian_knowledge_endpoint)
for q in [
    "LHAASO 日影 3.3天 行星际磁场",
    "日影 太阳风速度 预测",
    "太阳 阴影 LHAASO 磁场",
]:
    result = svc._search_remote_knowledge_base(q, max_results=3)
    print("Q:", q)
    print("count:", len(result))
    for item in result[:2]:
        print(json.dumps(item, ensure_ascii=False)[:500])
print("telemetry:", json.dumps(svc.telemetry(), ensure_ascii=False))
