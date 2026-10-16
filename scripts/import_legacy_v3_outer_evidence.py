"""Import the frozen legacy-v3 outer backtest as external diagnostic evidence.

This import is deliberately not a closed-loop experiment writeback:

- It adds auditable ``observational_data`` evidence items with
  ``support_weight=0.0`` and never changes any hypothesis ``support_score``.
- It opens one new active uncertainty about cross-window reproducibility.
- It re-runs the procedural prioritizer and candidate generator as a dry run so
  the next-round plan can be inspected before any LLM planning or approval.
- None of the produced candidates are written into the live session's
  ``candidate_experiments.json``; they are only saved under the frozen artifact
  directory as a dry-run file.

Outputs:

- ``runtime/live_session/sessions/<active>/state/backups/...`` raw state backups
- ``artifacts/legacy_v3_migration/import_dry_run/legacy_v3_outer_import_report.json``
- ``artifacts/legacy_v3_migration/import_dry_run/round_03_candidates_dry_run.json``
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.decision_unified import CandidateExperimentGenerator, UncertaintyPrioritizer
from core.json_safe import json_dumps
from core.state_repository import UnifiedStateRepository
from core.unified_schema import (
    EvidenceItem,
    LatestTreeUpdate,
    UncertaintyHistoryEntry,
    UncertaintyRecord,
)

SESSIONS_ROOT = ROOT / "runtime" / "live_session" / "sessions"
SESSION_POINTER = SESSIONS_ROOT / "active_session.json"
ART_DIR = ROOT / "artifacts" / "legacy_v3_migration"
OUT_DIR = ART_DIR / "import_dry_run"
OUTER_DIR = ART_DIR / "outer_backtest"

SOURCE = "external_diagnostic:legacy_v3_outer_backtest"
EVIDENCE_KEY = f"{SOURCE}:20261017"
UNCERTAINTY_ID = "U_EXT_L3_01"
TARGET_ROUND = 3


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(payload, indent=2), encoding="utf-8")


def _active_session_root() -> Path:
    if not SESSION_POINTER.exists():
        raise SystemExit(f"缺少活跃会话指针: {SESSION_POINTER}")
    pointer = _read_json(SESSION_POINTER)
    session_name = pointer.get("session")
    if not session_name:
        raise SystemExit(f"active_session.json 未指向会话: {pointer}")
    session_root = SESSIONS_ROOT / session_name
    if not session_root.exists():
        raise SystemExit(f"会话目录不存在: {session_root}")
    return session_root


def _upsert_evidence(
    node,
    *,
    evidence_id: str,
    description: str,
    added_at_round: int,
) -> None:
    target = node.evidence_items
    target[:] = [
        item for item in target if getattr(item, "evidence_key", None) != EVIDENCE_KEY
    ]
    target.append(
        EvidenceItem(
            id=evidence_id,
            type="observational_data",
            description=description,
            source=SOURCE,
            added_at_round=added_at_round,
            support_weight=0.0,
            evidence_version="legacy_v3_outer_backtest:20261017",
            evidence_key=EVIDENCE_KEY,
        )
    )


def _upsert_uncertainty(uncertainty_state) -> UncertaintyRecord:
    index = uncertainty_state.record_index()
    record = index.get(UNCERTAINTY_ID)
    if record is None:
        record = UncertaintyRecord(
            uncertainty_id=UNCERTAINTY_ID,
            question=(
                "外部补充回测中，dev 段 mag -> mag_both（提前 3 天、训练内自动选参）"
                "的唯一不跨零 RMSE 改善未在 f1/f2 复现；这一局部信号是否具有跨时段稳定性？"
            ),
            description=(
                "冻结产物 outer_backtest_reproducibility_conclusion.json：dev n=35 增益 +0.481"
                " km/s、95% CI [0.014, 0.850]；f1 n=17 增益 -0.273 km/s、95% CI [-1.507, 0.507]；"
                "f2 n=23 增益 +0.530 km/s、95% CI [-0.327, 1.677]。两个早段窗口均跨零，"
                "dev 的 mag_both 改善未复现。当前只登记外部诊断证据与未决问题，不修改支持度。"
            ),
            related_hypotheses=["H_shadow_incremental_gain", "H_window_stability"],
            mining_sources=[SOURCE],
            status="active",
            priority="high",
            created_at_round=2,
            created_by="external_diagnostic_import",
            notes="external diagnostic dry-run; support scores untouched",
        )
        uncertainty_state.records.append(record)

    record.question = (
        "外部补充回测中，dev 段 mag -> mag_both（提前 3 天、训练内自动选参）"
        "的唯一不跨零 RMSE 改善未在 f1/f2 复现；这一局部信号是否具有跨时段稳定性？"
    )
    record.description = (
        "冻结产物 outer_backtest_reproducibility_conclusion.json：dev n=35 增益 +0.481"
        " km/s、95% CI [0.014, 0.850]；f1 n=17 增益 -0.273 km/s、95% CI [-1.507, 0.507]；"
        "f2 n=23 增益 +0.530 km/s、95% CI [-0.327, 1.677]。两个早段窗口均跨零，"
        "dev 的 mag_both 改善未复现。外部诊断导入不修改任何支持度。"
    )
    record.related_hypotheses = list(
        dict.fromkeys([*record.related_hypotheses, "H_shadow_incremental_gain", "H_window_stability"])
    )
    record.mining_sources = list(dict.fromkeys([*record.mining_sources, SOURCE]))
    record.status = "active"
    record.priority = "high"
    record.history[:] = [
        entry
        for entry in record.history
        if getattr(entry, "evidence_key", None) != EVIDENCE_KEY
    ]
    record.history.append(
        UncertaintyHistoryEntry(
            round=2,
            event="external_diagnostic_imported",
            source=SOURCE,
            description=(
                "legacy_v3 outer backtest frozen evidence imported; f1/f2 CI cross zero,"
                " dev signal not reproduced; no support score update"
            ),
            evidence_version="legacy_v3_outer_backtest:20261017",
            evidence_key=EVIDENCE_KEY,
        )
    )
    return record


def main() -> None:
    session_root = _active_session_root()
    repo = UnifiedStateRepository(session_root)

    backup_dir = session_root / "state" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in ("hypothesis_tree.json", "uncertainties.json"):
        source = session_root / "state" / name
        if source.exists():
            shutil.copy2(source, backup_dir / f"pre_ext_import_{stamp}_{name}")

    tree = repo.load_hypothesis_tree()
    uncertainties = repo.load_uncertainties()
    task = repo.load_task()
    dictionary = repo.load_data_dictionary()
    memory = repo.load_experiment_memory()

    before_scores = {
        node.hypothesis_id: node.support_score for node in tree.nodes
    }
    node_index = tree.node_index()

    evidence_plan = [
        (
            node_index["H_shadow_incremental_gain"],
            "EXT_L3_H1_20261017",
            (
                "外部诊断(legacy_v3_outer_backtest)：dev 段 mag -> mag_both、提前 3 天、"
                "训练内自动选参的 RMSE 点增益 +0.481 km/s、95% 配对块 bootstrap CI"
                " [0.014, 0.850]。该结果仅来自冻结的迭代评估段，support_weight=0.0，"
                "不修改支持度，也不宣称已经跨时段成立。"
            ),
        ),
        (
            node_index["H_window_stability"],
            "EXT_L3_H5_20261017",
            (
                "外部诊断(legacy_v3_outer_backtest)：预登记早段 f1 n=17 增益 -0.273"
                " km/s、95% CI [-1.507, 0.507]；f2 n=23 增益 +0.530 km/s、95% CI"
                " [-0.327, 1.677]。两个窗口 RMSE CI 均跨零，dev 的 mag_both 改善未复现；"
                "另有 2 组探索性不跨零结果方向相反，未做多重比较调整。support_weight=0.0。"
            ),
        ),
    ]
    for node, evidence_id, description in evidence_plan:
        if node is None:
            raise SystemExit(f"假设节点未找到: {evidence_plan[0][0].hypothesis_id}")
        _upsert_evidence(
            node,
            evidence_id=evidence_id,
            description=description,
            added_at_round=2,
        )

    record = _upsert_uncertainty(uncertainties)
    tree.latest_update = LatestTreeUpdate(
        round=2,
        event="external_diagnostic_imported",
        description=(
            "legacy_v3 outer backtest frozen as external diagnostic evidence;"
            " evidence support_weight=0.0, support scores unchanged"
        ),
    )
    tree.generated_at = tree.generated_at or datetime.now()

    repo.save_hypothesis_tree(tree)
    repo.save_uncertainties(uncertainties)

    prioritized = UncertaintyPrioritizer().prioritize(
        uncertainties,
        tree,
        target_round=TARGET_ROUND,
    )
    repo.save_uncertainties(uncertainties)

    candidate_set = CandidateExperimentGenerator().generate(
        task=task,
        data_dictionary=dictionary,
        hypothesis_tree=tree,
        uncertainty_state=uncertainties,
        experiment_memory=memory,
        target_round=TARGET_ROUND,
        limit=4,
    )

    after_scores = {
        node.hypothesis_id: node.support_score for node in tree.nodes
    }
    changed = {
        hypothesis_id: (before, after_scores[hypothesis_id])
        for hypothesis_id, before in before_scores.items()
        if before != after_scores.get(hypothesis_id)
    }
    if changed:
        raise SystemExit(f"导入脚本意外修改了支持度: {changed}")

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "source": SOURCE,
        "session_root": str(session_root),
        "status": "imported_as_external_diagnostic",
        "support_score_updates": changed,
        "declaration": (
            "This import records external diagnostic evidence only. It does not claim"
            " the system autonomously discovered these results, and it does not approve"
            " any candidate shown in the dry run."
        ),
        "evidence_items_added": [
            {
                "hypothesis_id": node.hypothesis_id,
                "evidence_id": item.id,
                "evidence_key": item.evidence_key,
                "type": item.type,
                "support_weight": item.support_weight,
            }
            for node in tree.nodes
            for item in node.evidence_items
            if item.evidence_key == EVIDENCE_KEY
        ],
        "uncertainty": {
            "uncertainty_id": record.uncertainty_id,
            "priority": record.priority,
            "status": record.status,
            "related_hypotheses": record.related_hypotheses,
            "priority_score": next(
                (
                    item.priority_score
                    for item in prioritized.queue
                    if item.uncertainty_id == UNCERTAINTY_ID
                ),
                None,
            ),
        },
        "priority_queue_top": [
            {
                "uncertainty_id": item.uncertainty_id,
                "priority_score": item.priority_score,
                "estimated_resolution_round": item.estimated_resolution_round,
            }
            for item in prioritized.queue
        ],
        "candidate_dry_run": {
            "target_round": candidate_set.round,
            "count": len(candidate_set.candidates),
            "minimum_required": candidate_set.minimum_required,
            "note": candidate_set.note,
            "written_to_live_state": False,
        },
        "source_artifacts": [
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in sorted(OUTER_DIR.glob("*"))
            if path.is_file()
        ],
    }

    candidate_payload: dict[str, Any] = {
        "declaration": (
            "Round-3 candidate dry run produced after importing the frozen external"
            " outer-backtest evidence. This is not approval, execution, or a claim that"
            " the system discovered the diagnostic by itself."
        ),
        "source": SOURCE,
        "session_root": str(session_root),
        "candidate_set": candidate_set.model_dump(mode="json", exclude_none=True),
    }

    report_path = out_dir / "legacy_v3_outer_import_report.json"
    candidates_path = out_dir / "round_03_candidates_dry_run.json"
    _write_json(report_path, report)
    _write_json(candidates_path, candidate_payload)

    print(f"session_root={session_root}")
    print(f"support_score_updates={changed or 'none'}")
    print(f"uncertainty={UNCERTAINTY_ID} priority_score="
          f"{report['uncertainty']['priority_score']}")
    print("candidates:")
    for candidate in candidate_set.candidates:
        print(
            f"  {candidate.experiment_id} | {candidate.experiment_type} | "
            f"{candidate.scientific_question}"
        )
    print(f"report={report_path.relative_to(ROOT).as_posix()}")
    print(f"candidates={candidates_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
