"""Clean LLM prose leaked into round-2 candidate variable fields."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.state_repository import UnifiedStateRepository
from core.variable_semantic_service import VariableSemanticService

LIVE_ROOT = Path(
    r"D:\ShadowTracing\runtime\live_session\sessions\session_20261015_132955_705440"
)

PROSE_MARKERS = [
    "基线特征仅保留太阳风速度历史滞后",
    "新增特征仅保留当前宇宙线日影南北偏移",
    "不引入非线性交互",
    "不引入未支持的新物理量",
    "不同时扩展滞后窗口或超前窗口",
]

CLEAN_FEATURES = {
    "E_R02_03": {
        "control": ["ss_br_mean", "BX, nT (GSE, GSM)"],
        "treatment": ["ss_br_mean", "BX, nT (GSE, GSM)", "p2_y中心值"],
    },
    "E_R02_04": {
        "control": ["SW Plasma Speed, km/s"],
        "treatment": ["SW Plasma Speed, km/s", "p2_y中心值"],
    },
}


def _is_prose(feature: str) -> bool:
    return any(marker in feature for marker in PROSE_MARKERS)


def main() -> None:
    repo = UnifiedStateRepository(LIVE_ROOT)
    candidate_set = repo.load_candidate_experiments()
    semantic = VariableSemanticService.from_data_dictionary(repo.load_data_dictionary())
    changed: list[dict[str, object]] = []

    for candidate in candidate_set.candidates:
        if candidate.experiment_id not in CLEAN_FEATURES:
            continue
        design = candidate.design
        clean = CLEAN_FEATURES[candidate.experiment_id]
        design.control = [item for item in clean["control"] if item and not _is_prose(item)]
        design.treatment = [item for item in clean["treatment"] if item and not _is_prose(item)]
        design.design_focus = "p2_y中心值"
        design.display_control = semantic.display_list(design.control)
        design.display_treatment = semantic.display_list(design.treatment)
        design.display_design_focus = semantic.to_display("p2_y中心值")
        if "repair:round2_candidate_prose_sanitized" not in design.notes:
            design.notes.append("repair:round2_candidate_prose_sanitized")
        if design.runs_spec:
            baseline = design.control
            treatment = design.treatment
            for run_spec in design.runs_spec:
                if run_spec.run_id.endswith("treatment"):
                    run_spec.feature_columns = list(treatment)
                elif run_spec.run_id.endswith("baseline"):
                    run_spec.feature_columns = list(baseline)
        changed.append(
            {
                "experiment_id": candidate.experiment_id,
                "control": design.control,
                "treatment": design.treatment,
                "runs_spec": [
                    {
                        "run_id": run.run_id,
                        "feature_columns": run.feature_columns,
                        "model_family": run.model_family,
                    }
                    for run in design.runs_spec
                ],
                "design_focus": design.design_focus,
                "display_control": design.display_control,
                "display_treatment": design.display_treatment,
            }
        )

    if not changed:
        print("No round-2 candidates needed repair.")
        return
    repo.save_candidate_experiments(candidate_set)
    print(json.dumps(changed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
