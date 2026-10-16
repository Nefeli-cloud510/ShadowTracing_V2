"""Idempotently backfill cross-round audit fields for the active live session."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.json_safe import json_dumps
from core.round_history_helpers import populate_cross_round_fields
from core.unified_schema import ExperimentProtocol, RoundHistoryState


SESSION_ROOT = (
    REPO_ROOT
    / "runtime"
    / "live_session"
    / "sessions"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, value: dict) -> None:
    path.write_text(json_dumps(value), encoding="utf-8")


def _populate_history(history: RoundHistoryState, live_root: Path) -> None:
    for entry in history.entries:
        protocol_path = (
            live_root
            / "results"
            / f"round_{entry.round_id:02d}"
            / "protocol.json"
        )
        if not protocol_path.is_file():
            continue
        previous_entry = next(
            (
                item
                for item in history.entries
                if item.round_id == entry.round_id - 1
            ),
            None,
        )
        protocol = ExperimentProtocol.model_validate(_load_json(protocol_path))
        populate_cross_round_fields(entry, protocol, previous_entry)
        entry.updated_at = datetime.now()
    history.last_updated = datetime.now()


def main() -> None:
    pointer = _load_json(SESSION_ROOT / "active_session.json")
    session_name = str(pointer.get("session") or "").strip()
    if not session_name:
        raise SystemExit("active_session.json 中缺少 session 指针，无法定位 live 会话。")
    live_root = SESSION_ROOT / session_name
    state_dir = live_root / "state"
    history_path = state_dir / "round_history.json"
    if not history_path.is_file():
        raise SystemExit(f"未找到 round_history: {history_path}")

    history = RoundHistoryState.model_validate(_load_json(history_path))
    _populate_history(history, live_root)
    _save_json(history_path, history.model_dump(mode="json"))
    print(f"已更新 round_history: {history_path}")

    snapshot_dir = state_dir / "round_snapshots"
    if snapshot_dir.is_dir():
        for snapshot_path in sorted(snapshot_dir.glob("round_*.json")):
            snapshot = _load_json(snapshot_path)
            artifacts = snapshot.get("artifacts") or {}
            old_history = artifacts.get("round_history")
            if not isinstance(old_history, dict) or not old_history.get("entries"):
                continue
            snapshot_history = RoundHistoryState.model_validate(old_history)
            _populate_history(snapshot_history, live_root)
            artifacts["round_history"] = snapshot_history.model_dump(mode="json")
            snapshot["artifacts"] = artifacts
            _save_json(snapshot_path, snapshot)
            print(f"已同步快照: {snapshot_path}")


if __name__ == "__main__":
    main()
