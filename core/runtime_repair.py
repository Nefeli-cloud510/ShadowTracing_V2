"""Maintenance helpers for legacy live-session artifacts.

The three-layer conclusion was introduced after early rounds were persisted.
This module re-derives the missing conclusion from already existing protocol,
result and evaluation files, then writes it back to the round reports.  It is
intended to be run once from the repository root:

    python -m core.runtime_repair --live-root runtime/live_session/current
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.evaluation_unified import recover_three_layer_conclusion
from core.json_safe import json_dumps
from core.state_repository import UnifiedStateRepository
from core.state_updater import translate_three_layer_conclusion
from core.unified_schema import (
    EvaluationResult,
    ExperimentProtocol,
    ExperimentResult,
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def backfill_three_layer_conclusion(live_root: str | Path) -> list[dict[str, Any]]:
    live_root = Path(live_root)
    repository = UnifiedStateRepository(live_root)
    round_history = repository.load_round_history()
    results: list[dict[str, Any]] = []

    for entry in round_history.entries:
        round_dir = live_root / "results" / f"round_{entry.round_id:02d}"
        protocol_payload = _load_json(round_dir / "protocol.json")
        result_payload = _load_json(round_dir / "result_unified.json")
        evaluation_payload = _load_json(round_dir / "evaluation_unified.json")

        if not (protocol_payload and result_payload and evaluation_payload):
            results.append(
                {
                    "round_id": entry.round_id,
                    "status": "skipped",
                    "reason": "缺少 protocol/result/evaluation 文件",
                }
            )
            continue

        protocol = ExperimentProtocol.model_validate(protocol_payload)
        result = ExperimentResult.model_validate(result_payload)
        evaluation = EvaluationResult.model_validate(evaluation_payload)
        if evaluation.scientific.three_layer_conclusion is not None:
            results.append(
                {
                    "round_id": entry.round_id,
                    "status": "already_present",
                    "experiment_id": protocol.experiment_id,
                }
            )
            continue

        main_question = protocol.scientific_objective
        try:
            task = repository.load_task()
            main_question = (
                task.payload.research_question.text
                or protocol.scientific_objective
            )
        except Exception:
            pass

        conclusion = recover_three_layer_conclusion(
            evaluation,
            protocol,
            result=result,
            main_question=main_question,
        )
        if conclusion is None:
            results.append(
                {
                    "round_id": entry.round_id,
                    "status": "skipped",
                    "reason": "无法基于现有产物重建三层结论",
                }
            )
            continue

        evaluation.scientific.three_layer_conclusion = conclusion
        translate_three_layer_conclusion(repository, evaluation)
        evaluation_path = round_dir / "evaluation_unified.json"
        evaluation_path.write_text(
            json_dumps(
                evaluation.model_dump(mode="json", exclude_none=True),
                indent=2,
            ),
            encoding="utf-8",
        )
        entry.three_layer_conclusion = (
            evaluation.scientific.three_layer_conclusion.model_copy(deep=True)
        )
        results.append(
            {
                "round_id": entry.round_id,
                "status": "backfilled",
                "experiment_id": protocol.experiment_id,
            }
        )

    repository.save_round_history(round_history)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing three-layer conclusions.")
    parser.add_argument(
        "--live-root",
        default="runtime/live_session/current",
        help="live session root, defaults to runtime/live_session/current",
    )
    args = parser.parse_args()
    results = backfill_three_layer_conclusion(args.live_root)
    print(json_dumps(results, indent=2))


if __name__ == "__main__":
    main()
