import json
import time

import requests


BASE = "http://127.0.0.1:8765"


def state_key(snapshot: dict) -> tuple:
    return (
        snapshot.get("status"),
        snapshot.get("stage"),
        snapshot.get("planning_status"),
        snapshot.get("message"),
    )


def main() -> int:
    previous = None
    while True:
        try:
            response = requests.get(f"{BASE}/api/session", timeout=10)
            response.raise_for_status()
            snapshot = response.json()
        except Exception as exc:
            print(f"poll-error: {exc}", flush=True)
            time.sleep(10)
            continue
        key = state_key(snapshot)
        if key != previous:
            print(
                json.dumps(
                    {
                        "status": snapshot.get("status"),
                        "stage": snapshot.get("stage"),
                        "planning_status": snapshot.get("planning_status"),
                        "currentRound": snapshot.get("currentRound"),
                        "message": snapshot.get("message"),
                        "runRoot": snapshot.get("runRoot"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            previous = key
        if snapshot.get("status") == "failed":
            print("WORKFLOW_FAILED", flush=True)
            return 1
        terminal_stages = {
            "hypothesis_review",
            "awaiting_scientific_questioning",
            "uncertainty_pending",
            "approval_pending",
            "awaiting_round_decision",
            "completed",
        }
        if snapshot.get("stage") in terminal_stages:
            print("STAGE_WAITING", flush=True)
            return 0
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
