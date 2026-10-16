from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.hypothesis_proposer_llm import (
    HypothesisProposerLLM,
    HypothesisProposerResponse,
    LlmEvidenceItem,
    LlmPrediction,
    ProposedHypothesis,
)
from core.llm_gateway import LLMGateway
from core.rag_service import RAGContextBundle
from core.unified_schema import ReasoningPlannerInput
from scripts.run_real_api_two_round_demo import ObservedRAGService


PROJECT_TERMS = [
    "LHAASO",
    "日影",
    "宇宙线太阳",
    "太阳阴影",
    "cosmic ray",
    "cosmic-ray",
    "DeltaDec",
]
IMF_TERMS = ["行星际磁场", "IMF", "磁场", "Bx", "By", "Bz", "magnetic field", "heliospheric"]
VSW_TERMS = ["太阳风速度", "太阳风", "Vsw", "solar wind"]
LEAD_TERMS = ["3.3", "3.3 天", "3.3天", "领先", "提前", "lead time"]
CLAIM_TERMS = ["准确预报", "可靠预报", "提高预报", "改善预报", "能够预测", "可预测"]
BOUNDARY_TERMS = ["尚未检验", "未检验", "不能证明", "有待验证", "尚需验证", "未验证", "not yet"]
POWER_TERMS = ["光伏", "风电", "太阳能功率", "风力发电", "solar power", "wind power", "photovoltaic"]


class RecordingGateway(LLMGateway):
    def __init__(self) -> None:
        super().__init__(
            model="fixture-model",
            api_key="fixture-key",
            client=object(),
            allow_fallback=False,
            use_cache=False,
        )
        self.prompts: list[dict[str, str]] = []

    def generate_structured(self, *, system_prompt, user_prompt, response_model, **kwargs):
        self.prompts.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "role": str(kwargs.get("role") or ""),
            }
        )
        proposals = []
        for index in range(1, 6):
            display_id = f"H{index}"
            proposals.append(
                ProposedHypothesis(
                    display_hypothesis_id=display_id,
                    statement=f"fixture hypothesis {display_id} about shadow IMF lead signals",
                    level=1 if index <= 2 else 2,
                    parent_id=None if index <= 2 else "H1",
                    activation_condition="fixture",
                    falsification="fixture falsification",
                    expected_effect="positive",
                    falsification_conditions=["fixture"],
                    alternative_explanations=["fixture"],
                    predictions=[
                        LlmPrediction(
                            experiment_condition="fixture",
                            expected_observable="Pearson_r",
                            expected_direction="positive",
                            expected_range=[0.02, 0.08],
                        )
                    ],
                    evidence_items=[LlmEvidenceItem(description="fixture", evidence_type="literature", source="fixture")],
                )
            )
        return response_model.model_validate(
            {
                "focus_features": [],
                "guidance_notes": ["fixture note"],
                "proposal_summaries": [f"proposal {index}" for index in range(1, 6)],
                "proposed_hypotheses": proposals,
                "source": "probe_recording_fixture",
                "model": "fixture-model",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )


def prepare_literature(session_root: Path) -> list[dict[str, object]]:
    target_dir = session_root / "outputs" / "pdf_texts"
    target_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        (REPO_ROOT / "outputs" / "_tmp_lhaaso_imf.txt", "lhaaso_imf_2021.txt"),
        (REPO_ROOT / "outputs" / "_tmp_nanyun.txt", "nanyun_2021.txt"),
        (REPO_ROOT / "outputs" / "_tmp_lizhe.txt", "lizhe_2021.txt"),
        (REPO_ROOT / "outputs" / "_tmp_thesis.txt", "thesis_2021.txt"),
    ]
    records: list[dict[str, object]] = []
    for source, target_name in sources:
        if not source.exists():
            continue
        target = target_dir / target_name
        shutil.copyfile(source, target)
        text = target.read_text(encoding="utf-8", errors="ignore")
        records.append(
            {
                "file_name": target_name,
                "file_type": ".txt",
                "size_bytes": len(text.encode("utf-8")),
                "uploaded_at": datetime.now().isoformat(),
                "storage_path": f"outputs/pdf_texts/{target_name}",
                "extracted_text_path": f"outputs/pdf_texts/{target_name}",
                "extraction_status": "ready",
                "extraction_engine": "pre_extracted_text",
                "character_count": len(text),
                "extraction_note": "由人工提取的论文正文文本直接纳入本地会话文献库。",
                "knowledge_sync_status": "local_session_only",
                "knowledge_sync_note": "本文件不写回百炼托管知识库，仅供会话内 RAG 检索。",
            }
        )
    state_dir = session_root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = state_dir / "upload_manifest.json"
    manifest_path.write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return records


def _text_of(item) -> str:
    data = item if isinstance(item, dict) else item.model_dump()
    return " ".join([str(data.get("title") or ""), str(data.get("excerpt") or "")])


def _flag(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _snippet(item, limit: int = 420) -> dict[str, object]:
    data = item if isinstance(item, dict) else item.model_dump()
    excerpt = str(data.get("excerpt") or "")
    return {
        "source_type": data.get("source_type"),
        "source_id": data.get("source_id"),
        "title": data.get("title"),
        "excerpt": excerpt[:limit],
        "excerpt_full_length": len(excerpt),
        "score": data.get("score"),
        "citation": data.get("citation"),
        "flags": {
            "project_related": _flag(_text_of(item), PROJECT_TERMS),
            "imf_field": _flag(excerpt, IMF_TERMS),
            "solar_wind": _flag(excerpt, VSW_TERMS),
            "lead_relation": _flag(excerpt, LEAD_TERMS),
            "prediction_claim": _flag(excerpt, CLAIM_TERMS),
            "claim_boundary_stated": _flag(excerpt, BOUNDARY_TERMS),
            "power_forecast_confusion": _flag(excerpt, POWER_TERMS),
        },
    }


def _query_acceptance(
    *,
    query: str,
    svc: ObservedRAGService,
    expected_note: str,
) -> dict[str, object]:
    local_lit = svc.search_literature_knowledge(query, max_results=3)
    remote = svc.search_bailian_knowledge_base(query, max_results=3)
    project = svc.search_project_knowledge(query, max_results=3)
    return {
        "query": query,
        "expected_note": expected_note,
        "project_hits": [_snippet(item) for item in project],
        "local_literature_hits": [_snippet(item) for item in local_lit],
        "remote_hits": [_snippet(item) for item in remote],
        "summary": {
            "project_count": len(project),
            "local_literature_count": len(local_lit),
            "remote_count": len(remote),
            "remote_has_project_related": any(_flag(_text_of(item), PROJECT_TERMS) for item in remote),
            "remote_has_power_forecast_confusion": any(_flag(_text_of(item), POWER_TERMS) for item in remote),
            "remote_imf_lead_found": any(
                _flag(_text_of(item), IMF_TERMS) and _flag(_text_of(item), LEAD_TERMS)
                for item in remote
            ),
            "local_imf_lead_found": any(
                _flag(_text_of(item), IMF_TERMS) and _flag(_text_of(item), LEAD_TERMS)
                for item in local_lit
            ),
            "prediction_claim_without_boundary": any(
                _flag(_text_of(item), CLAIM_TERMS) and not _flag(_text_of(item), BOUNDARY_TERMS)
                for item in [*local_lit, *remote]
            ),
        },
    }


def _prompt_chain_acceptance(
    *,
    query: str,
    svc: ObservedRAGService,
    fixture_path: Path,
) -> dict[str, object]:
    local_lit = svc.search_literature_knowledge(query, max_results=2)
    remote = svc.search_bailian_knowledge_base(query, max_results=2)
    bundle = RAGContextBundle(
        query=query,
        project_evidence=[],
        literature_evidence=local_lit,
        external_evidence=remote,
    )
    notes = bundle.guidance_notes()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    planner_input = ReasoningPlannerInput.model_validate(fixture)
    gateway = RecordingGateway()
    result = HypothesisProposerLLM(gateway=gateway).propose(
        planner_input=planner_input,
        rag_context=bundle,
    )
    prompts = gateway.prompts
    user_prompt = prompts[-1]["user_prompt"] if prompts else ""
    visible_notes = [note for note in notes if note in user_prompt]
    return {
        "query": query,
        "rag_note_count": len(notes),
        "rag_notes_visible_in_prompt": len(visible_notes),
        "prompt_length_chars": len(user_prompt),
        "role": prompts[-1]["role"] if prompts else None,
        "visible_note_fragments": [note[:260] for note in visible_notes],
        "proposal_count": len(result.proposed_hypotheses),
        "source": result.source,
        "model": result.model,
        "boundary_notes": [
            "候选实验设计 prompt 当前不直接注入 rag_notes；RAG 片段先进入假设提出者，"
            "再经 hypothesis -> candidate 链路由程序关联。"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--out", default=REPO_ROOT / "artifacts" / "rag_acceptance_20261016.json", type=Path)
    args = parser.parse_args()

    session_root = args.session_root.resolve()
    if not session_root.is_dir():
        print(f"session root missing: {session_root}")
        return 1

    ingestion = prepare_literature(session_root)
    svc = ObservedRAGService(project_root=session_root)
    queries = [
        {
            "id": "q1_lead_3_3d_observable",
            "query": "LHAASO 日影领先约 3.3 天，具体对应什么观测量？",
            "expected_note": "应定位到行星际磁场/IMF 的时间关联，不得改写成太阳风速度已能提前准确预报。",
        },
        {
            "id": "q2_paper_proves_vsw_gain",
            "query": "这篇论文是否证明日影能提高太阳风速度预测？",
            "expected_note": "应区分论文已有发现与本项目待检验问题，不能把文献支持当成实验验证。",
        },
        {
            "id": "q3_hypothesis_support",
            "query": "什么证据支持本轮某条假设？",
            "expected_note": "应返回可核对片段，并区分直接证据、间接依据与物理推测。",
        },
        {
            "id": "q4_uncovered_domain",
            "query": "量子引力对地磁暴预报的约束是什么？",
            "expected_note": "知识库未覆盖时应返回未找到足够依据，不能编造文献或结论。",
        },
        {
            "id": "q5_retrieval_to_candidate",
            "query": "请用检索到的 IMF 领先 3.3 天证据生成候选实验",
            "expected_note": "检查实际传给智能体的上下文，而不只看检索接口返回值。",
        },
    ]
    acceptance = [_query_acceptance(query=item["query"], svc=svc, expected_note=item["expected_note"]) for item in queries]
    prompt_chain = _prompt_chain_acceptance(
        query=queries[-1]["query"],
        svc=svc,
        fixture_path=args.fixture,
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_root": str(session_root),
        "ingestion": {
            "literature_files": [
                {
                    "file_name": record["file_name"],
                    "character_count": record["character_count"],
                    "extraction_status": record["extraction_status"],
                    "extracted_text_path": record["extracted_text_path"],
                }
                for record in ingestion
            ],
            "manifest_path": str(session_root / "state" / "upload_manifest.json"),
        },
        "remote_telemetry": svc.telemetry(),
        "queries": acceptance,
        "generation_chain": prompt_chain,
        "findings": {
            "stored_pending_prompt_chain": True,
            "remote_kb_reachable": svc.telemetry()["configured"] and svc.telemetry()["successful_attempts"] > 0,
            "local_literature_indexed": len(ingestion) > 0 and all(record["character_count"] > 100 for record in ingestion),
            "power_forecast_confusion_detected": any(
                item["summary"]["remote_has_power_forecast_confusion"] for item in acceptance
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report written: {args.out}")
    print(json.dumps(report["findings"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
