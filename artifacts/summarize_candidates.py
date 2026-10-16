from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(
            r"D:\ShadowTracing\runtime\live_session\sessions\session_20261016_235451_805278\state\candidate_experiments.json"
        )
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    print("round", data.get("round"), "minimum_required", data.get("minimum_required"), "current_count", data.get("current_count"))
    for candidate in data.get("candidates", []):
        design = candidate.get("design", {})
        print("---")
        print(
            candidate.get("experiment_id"),
            "| type:", candidate.get("experiment_type"),
            "| focus:", design.get("design_focus"),
            "| target:", design.get("target"),
        )
        print("control:", design.get("control"))
        print("treatment:", design.get("treatment"))
        print(
            "horizon/lag/window:",
            design.get("forecast_horizon_days"),
            design.get("past_lag_days"),
            design.get("window_size"),
        )
        print(
            "control/treatment lags:",
            design.get("control_lag_days"),
            design.get("treatment_lag_days"),
        )
        print("model_families:", design.get("model_families"))
        print("runs_spec:", json.dumps(design.get("runs_spec"), ensure_ascii=False)[:700])
        print("contrasts_spec:", json.dumps(design.get("contrasts_spec"), ensure_ascii=False)[:700])


if __name__ == "__main__":
    main()
