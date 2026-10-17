#!/usr/bin/env python
"""Isolated acceptance: frozen sparse-lag protocol executed exactly as declared.

This script does not touch the live session: it copies the real v3 input CSVs
into a temporary project root, compiles E_R03_01 into a formal
``model_sensitivity`` protocol, and runs all four family/arm pairs in that
temporary root. It also verifies that ``bootstrap_robustness`` and
``selection_robustness`` remain unregistered draft types and cannot be
silently relabelled into an incremental experiment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.control_unified import CandidateProtocolMapper
from core.experiment_tools.dispatcher import ExperimentDispatcher
from core.experiment_tools.registry import (
    ALL_EXPERIMENT_TYPES,
    candidate_selectable_types,
    get_tool,
)
from core.experiment_tools.validate_protocol import validate_experiment_protocol
from core.harness_unified import UnifiedExperimentHarness
from core.protocol_bridge import ElasticNetDataSourceConfig
from core.unified_schema import (
    CandidateExperiment,
    CandidateExperimentSet,
    EvaluationSpec,
    ExperimentDesign,
    ResearchQuestion,
    ScientificConstraints,
    ScientificTask,
    ScientificTaskPayload,
    VariableBinding,
)


PYTHON_BIN = Path(sys.executable).resolve()

FIXED_CANDIDATE_FILE = (
    REPO_ROOT
    / "artifacts"
    / "legacy_v3_migration"
    / "import_dry_run"
    / "round_03_candidates_dry_run.fixed.json"
)
OUTPUT_FILE = (
    REPO_ROOT
    / "artifacts"
    / "legacy_v3_migration"
    / "executor_acceptance"
    / "executor_sparse_lags_acceptance.json"
)
OMNI_SRC = REPO_ROOT / "runtime" / "live_run_input" / "omni_2021_full_calendar_v3.csv"
LHAASO_SRC = (
    REPO_ROOT / "runtime" / "live_run_input" / "lhaaso_shadow_pfss_2021_full_calendar_v3.csv"
)

FROZEN_LAGS = {
    "BY, nT (GSE)": [4, 5, 7, 8],
    "BZ, nT (GSE)": [1, 2],
    "Scalar B, nT": [1, 2],
    "p2_y中心值": [1],
    "p1_x中心值": [9],
}
BASELINE_FEATURES = [
    "BY, nT (GSE)",
    "BZ, nT (GSE)",
    "Scalar B, nT",
]
TREATMENT_FEATURES = [
    *BASELINE_FEATURES,
    "p2_y中心值",
    "p1_x中心值",
]


def git_head() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
        )
        return value.strip()
    except Exception:
        return "unknown"


def load_task() -> ScientificTask:
    return ScientificTask(
        task_id="ST_ACCEPTANCE_SPARSE_LAGS",
        payload=ScientificTaskPayload(
            research_question=ResearchQuestion(
                text="冻结外部回测中双方向日影增益是否能在冻结特征协议下跨时段重复",
                target="SW Plasma Speed, km/s",
                question_type="forecasting",
                variables=VariableBinding(
                    x="p2_y中心值,p1_x中心值",
                    y="SW Plasma Speed, km/s",
                ),
            ),
            evaluation=EvaluationSpec(primary_metric="RMSE"),
            constraints=ScientificConstraints(),
        ),
    )


def load_fixed_candidate_set() -> CandidateExperimentSet:
    payload = json.loads(FIXED_CANDIDATE_FILE.read_text(encoding="utf-8"))
    return CandidateExperimentSet.model_validate(payload["candidate_set"])


def feature_name(column: str, lag: int) -> str:
    return f"{column}_t-{lag}"


def expected_feature_sets() -> dict[str, set[str]]:
    baseline: set[str] = set()
    for column in BASELINE_FEATURES:
        baseline.update(feature_name(column, lag) for lag in FROZEN_LAGS[column])
    treatment = set(baseline)
    treatment.add(feature_name("p2_y中心值", 1))
    treatment.add(feature_name("p1_x中心值", 9))
    return {"baseline": baseline, "treatment": treatment}


def compile_protocol_smoke() -> dict:
    candidate_set = load_fixed_candidate_set()
    candidate = next(
        item for item in candidate_set.candidates if item.experiment_id == "E_R03_01"
    )
    task = load_task()
    protocol = CandidateProtocolMapper().to_protocol(
        task=task,
        candidate=candidate,
        round_id=3,
        model_name="ElasticNet",
    )
    validation = validate_experiment_protocol(protocol)
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    record("candidate_type_is_model_sensitivity", protocol.experiment_type == "model_sensitivity")
    record(
        "protocol_run_count_is_four",
        len(protocol.runs_spec) == 4,
        [item.run_id for item in protocol.runs_spec],
    )
    record(
        "protocol_contrast_count_is_two",
        len(protocol.contrasts_spec) == 2,
        [item.contrast_id for item in protocol.contrasts_spec],
    )
    record(
        "protocol_frozen_lags_preserved",
        {key: list(values) for key, values in (protocol.features.lags or {}).items()}
        == FROZEN_LAGS,
        protocol.features.lags,
    )
    record("protocol_validation_valid", validation.valid, "; ".join(validation.reasons))
    if validation.compiled_plan is not None:
        compiled = validation.compiled_plan
        record(
            "compiled_plan_has_four_runs",
            len(compiled.runs) == 4,
            [item.run_id for item in compiled.runs],
        )
        record(
            "compiled_plan_has_two_contrasts",
            len(compiled.contrasts) == 2,
            [item.contrast_id for item in compiled.contrasts],
        )
        record(
            "compiled_plan_feature_lags_exact",
            compiled.feature_lags == FROZEN_LAGS,
            compiled.feature_lags,
        )
        record(
            "compiled_time_parameters_exact",
            {
                "forecast_horizon_days": compiled.time_parameters.get("forecast_horizon_days"),
                "past_lag_days": compiled.time_parameters.get("past_lag_days"),
                "window_size": compiled.time_parameters.get("window_size"),
            }
            == {"forecast_horizon_days": 3, "past_lag_days": 9, "window_size": 9},
            compiled.time_parameters,
        )
    else:
        record("compiled_plan_present", False, "validation produced no compiled plan")

    return {
        "protocol_experiment_id": protocol.experiment_id,
        "experiment_type": protocol.experiment_type,
        "task_id": protocol.task_id,
        "round_id": protocol.round_id,
        "target": protocol.target,
        "protocol_features": {
            "control": list(protocol.features.control),
            "treatment": list(protocol.features.treatment),
            "lags": {
                key: list(values)
                for key, values in (protocol.features.lags or {}).items()
            },
            "forecast_horizon_days": protocol.features.forecast_horizon_days,
            "past_lag_days": protocol.features.past_lag_days,
            "window_size": protocol.features.window_size,
        },
        "model_parameters": {
            "model_families": list(protocol.model.parameters.get("model_families", [])),
            "window_size": protocol.model.parameters.get("window_size"),
            "forecast_horizon_days": protocol.model.parameters.get(
                "forecast_horizon_days"
            ),
            "past_lag_days": protocol.model.parameters.get("past_lag_days"),
            "param_search_mode": protocol.model.parameters.get("param_search_mode"),
        },
        "checks": checks,
        "all_passed": all(item["passed"] for item in checks),
        "compiled_plan": (
            validation.compiled_plan.to_dict()
            if validation.compiled_plan is not None
            else None
        ),
    }


def check_draft_types() -> dict:
    task = load_task()
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    for experiment_type in ("bootstrap_robustness", "selection_robustness"):
        record(
            f"{experiment_type}_not_in_ALL_EXPERIMENT_TYPES",
            experiment_type not in ALL_EXPERIMENT_TYPES,
            sorted(ALL_EXPERIMENT_TYPES),
        )
        record(
            f"{experiment_type}_not_selectable",
            experiment_type not in candidate_selectable_types(),
            sorted(candidate_selectable_types()),
        )
        record(
            f"{experiment_type}_tool_absent",
            get_tool(experiment_type) is None,
            str(get_tool(experiment_type)),
        )
        candidate = CandidateExperiment(
            experiment_id=f"E_DRAFT_{experiment_type.upper()}",
            type="distinguishing",
            experiment_type=experiment_type,
            purpose="跨时段稳定性草案：仅保留为不可执行候选",
            design=ExperimentDesign(
                target="SW Plasma Speed, km/s",
                control=list(BASELINE_FEATURES),
                treatment=list(TREATMENT_FEATURES),
                design_focus="p2_y中心值,p1_x中心值",
            ),
        )
        try:
            CandidateProtocolMapper().to_protocol(
                task=task,
                candidate=candidate,
                round_id=3,
            )
            raised_expected_error = False
            error_message = "未按预期抛出 ValueError"
        except ValueError as exc:
            raised_expected_error = "未注册" in str(exc)
            error_message = str(exc)
        record(
            f"{experiment_type}_mapper_rejects_with_unregistered",
            raised_expected_error,
            error_message,
        )
    return {
        "checks": checks,
        "all_passed": all(item["passed"] for item in checks),
        "selectable_types": sorted(candidate_selectable_types()),
    }


def run_four_runs_in_isolated_root() -> dict:
    with tempfile.TemporaryDirectory(prefix="shadow_acceptance_sparse_") as raw_temp:
        temp_root = Path(raw_temp)
        (temp_root / "config").mkdir(parents=True, exist_ok=True)
        (temp_root / "outputs").mkdir(parents=True, exist_ok=True)
        (temp_root / "results").mkdir(parents=True, exist_ok=True)

        omni_path = temp_root / "omni_2021_full_calendar_v3.csv"
        lhaaso_path = temp_root / "lhaaso_shadow_pfss_2021_full_calendar_v3.csv"
        shutil.copy2(OMNI_SRC, omni_path)
        shutil.copy2(LHAASO_SRC, lhaaso_path)

        (temp_root / "config" / "evaluation_plan.json").write_text(
            json.dumps(
                {
                    "schema_version": "evaluation-plan-v1",
                    "split_label": "dev_backtest",
                    "test_ratio": 0.2,
                    "n_internal_folds": 2,
                    "test_start": "2021-08-26",
                    "notes": [
                        "isolated executor acceptance: dev single-window execution only; "
                        "f1/f2 are frozen external backtest references, not auto-executed here."
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        candidate_set = load_fixed_candidate_set()
        candidate = next(
            item for item in candidate_set.candidates if item.experiment_id == "E_R03_01"
        )
        task = load_task()
        protocol = CandidateProtocolMapper().to_protocol(
            task=task,
            candidate=candidate,
            round_id=3,
            model_name="ElasticNet",
        )
        validation = validate_experiment_protocol(protocol)
        if not validation.valid or validation.compiled_plan is None:
            raise RuntimeError(
                "protocol validation failed before isolated execution: "
                + "; ".join(validation.reasons)
            )

        data_source = ElasticNetDataSourceConfig(
            omni_file_path=omni_path,
            lhaaso_file_path=lhaaso_path,
            time_column="TIME",
            target_column="SW Plasma Speed, km/s",
            test_split_ratio=0.2,
            random_state=42,
        )
        harness = UnifiedExperimentHarness(
            data_source=data_source,
            project_root=temp_root,
            run_shap=False,
        )
        started = time.perf_counter()
        result = ExperimentDispatcher(harness).execute(protocol)
        duration_seconds = round(time.perf_counter() - started, 3)
        if result.status != "completed":
            raise RuntimeError(
                f"isolated execution failed: status={result.status} "
                f"validity={result.validity_status} warnings={result.warnings}"
            )
        if result.run_id is None:
            raise RuntimeError("isolated execution returned no run_id")

        archive_dir = (
            temp_root
            / "results"
            / f"round_{protocol.round_id:02d}"
            / "archive"
            / result.run_id
        )
        compiled = json.loads(
            (archive_dir / "compiled_plan.json").read_text(encoding="utf-8")
        )
        eval_plan = json.loads(
            (archive_dir / "evaluation_plan.json").read_text(encoding="utf-8")
        )
        archived_protocol = json.loads(
            (archive_dir / "experiment_protocol.json").read_text(encoding="utf-8")
        )
        checks: list[dict] = []

        def record(name: str, passed: bool, detail: str = "") -> None:
            checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

        record("execution_completed", result.status == "completed", result.status)
        record(
            "archived_compiled_feature_lags_exact",
            compiled.get("feature_lags") == FROZEN_LAGS,
            compiled.get("feature_lags"),
        )
        archived_runs = compiled.get("runs", [])
        record(
            "archived_compiled_has_four_runs",
            len(archived_runs) == 4,
            [item.get("run_id") for item in archived_runs],
        )
        record(
            "archived_protocol_lags_exact",
            (archived_protocol.get("features") or {}).get("lags") == FROZEN_LAGS,
            (archived_protocol.get("features") or {}).get("lags"),
        )
        record(
            "evaluation_split_label_dev_backtest",
            eval_plan.get("split_label") == "dev_backtest",
            eval_plan.get("split_label"),
        )
        record(
            "evaluation_test_start_2021_08_26",
            eval_plan.get("test_start") == "2021-08-26",
            eval_plan.get("test_start"),
        )

        expected = expected_feature_sets()
        coefficient_summary: dict[str, dict] = {}
        for run_id in ("ridge_baseline", "ridge_treatment", "elasticnet_baseline", "elasticnet_treatment"):
            coef_path = archive_dir / run_id / "coefficients.csv"
            if not coef_path.exists():
                record(f"{run_id}_coefficients_present", False, str(coef_path))
                continue
            features = _read_feature_column(coef_path)
            arm = "baseline" if run_id.endswith("_baseline") else "treatment"
            coefficient_summary[run_id] = {
                "n_features": len(features),
                "features": sorted(features),
            }
            record(
                f"{run_id}_feature_count_matches",
                len(features) == (8 if arm == "baseline" else 10),
                f"n={len(features)}",
            )
            record(
                f"{run_id}_feature_set_matches",
                set(features) == expected[arm],
                sorted(features),
            )
            for required_feature in (
                "p2_y中心值_t-1",
                "p1_x中心值_t-9",
            ):
                if arm == "treatment":
                    record(
                        f"{run_id}_contains_{required_feature}",
                        required_feature in set(features),
                        required_feature,
                    )

        predictions_dir = archive_dir / "predictions.csv"
        record("predictions_csv_present", predictions_dir.exists(), str(predictions_dir))

        return {
            "temporary_project_root": str(temp_root),
            "run_id": result.run_id,
            "duration_seconds": duration_seconds,
            "execution_status": result.status,
            "evaluation_plan": eval_plan,
            "archived_compiled_plan": {
                "experiment_type": compiled.get("experiment_type"),
                "runs": [item.get("run_id") for item in archived_runs],
                "contrasts": [item.get("contrast_id") for item in compiled.get("contrasts", [])],
                "feature_lags": compiled.get("feature_lags"),
                "time_parameters": compiled.get("time_parameters"),
            },
            "coefficient_summary": coefficient_summary,
            "checks": checks,
            "all_passed": all(item["passed"] for item in checks),
        }


def _read_feature_column(coef_path: Path) -> list[str]:
    import pandas as pd

    frame = pd.read_csv(coef_path)
    return [str(item) for item in frame["feature"].tolist()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="compile/validate only")
    args = parser.parse_args()

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_head": git_head(),
        "fixed_candidate_file": str(FIXED_CANDIDATE_FILE),
        "execution_boundary": (
            "执行器当前架构只自动执行 dev 单窗口；f1/f2 是冻结外部回测清单中的"
            "只读参照，三窗口自动执行尚未接通。"
        ),
        "process_boundary": (
            "本次验收不调用大模型规划：E_R03_01 为研究者导入外部诊断后生成的"
            "候选草案，由程序化映射转正式协议并执行；不宣称跨窗口一致支持。"
        ),
        "protocol_smoke": compile_protocol_smoke(),
        "draft_type_checks": check_draft_types(),
    }

    if not args.smoke:
        summary["isolated_execution"] = run_four_runs_in_isolated_root()

    all_passed = all(
        item["all_passed"]
        for item in (summary["protocol_smoke"], summary["draft_type_checks"])
    )
    if not args.smoke:
        all_passed = all_passed and summary["isolated_execution"]["all_passed"]
    summary["all_passed"] = bool(all_passed)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"acceptance all_passed={summary['all_passed']} "
        f"smoke={args.smoke} output={OUTPUT_FILE}"
    )
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
