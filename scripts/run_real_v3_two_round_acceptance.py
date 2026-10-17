#!/usr/bin/env python
"""Real two-round acceptance on the frozen v3 protocol, isolated from live state.

Round 1 runs the frozen E_R03_01 model_sensitivity protocol with training-internal
auto parameter selection.
Round 2 reuses the same frozen features/lags/split but fixes each model family to
the parameters actually selected in round 1, testing whether the shadow gain
depends on automatic selection. All decisions are recorded as PI override, not
as autonomous model choices.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.control_unified import CandidateProtocolMapper
from core.experiment_tools.dispatcher import ExperimentDispatcher
from core.experiment_tools.validate_protocol import validate_experiment_protocol
from core.harness_unified import UnifiedExperimentHarness
from core.protocol_bridge import ElasticNetDataSourceConfig
from core.unified_schema import (
    CandidateExperiment,
    CandidateExperimentSet,
    EvaluationSpec,
    ResearchQuestion,
    ScientificConstraints,
    ScientificTask,
    ScientificTaskPayload,
    VariableBinding,
)


FIXED_CANDIDATE_FILE = (
    REPO_ROOT
    / "artifacts"
    / "legacy_v3_migration"
    / "import_dry_run"
    / "round_03_candidates_dry_run.fixed.json"
)
OMNI_SRC = REPO_ROOT / "runtime" / "live_run_input" / "omni_2021_full_calendar_v3.csv"
LHAASO_SRC = (
    REPO_ROOT / "runtime" / "live_run_input" / "lhaaso_shadow_pfss_2021_full_calendar_v3.csv"
)
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "real_v3_two_round"


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def make_task() -> ScientificTask:
    return ScientificTask(
        task_id="ST_REAL_V3_TWO_ROUND",
        payload=ScientificTaskPayload(
            research_question=ResearchQuestion(
                text="双方向日影的增量是否在冻结协议与固定选参对照下保持或消失",
                target="SW Plasma Speed, km/s",
                question_type="forecasting",
                variables=VariableBinding(
                    x="p2_y中心值,p1_x中心值",
                    y="SW Plasma Speed, km/s",
                ),
            ),
            evaluation=EvaluationSpec(
                primary_metric="Pearson_r",
                secondary_metrics=["RMSE", "MAE", "R2"],
            ),
            constraints=ScientificConstraints(
                no_future_information=True,
                validation_feedback_allowed=True,
                final_test_blind=True,
                max_rounds=2,
                max_experiments_per_round=1,
            ),
        ),
    )


def load_fixed_candidate(experiment_id: str = "E_R03_01") -> CandidateExperiment:
    payload = json.loads(FIXED_CANDIDATE_FILE.read_text(encoding="utf-8"))
    candidate_set = CandidateExperimentSet.model_validate(payload["candidate_set"])
    return next(
        item for item in candidate_set.candidates if item.experiment_id == experiment_id
    )


def prepare_project_root(run_root: Path) -> Path:
    project_root = run_root / "project"
    (project_root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (project_root / "config").mkdir(parents=True, exist_ok=True)
    (project_root / "outputs").mkdir(parents=True, exist_ok=True)
    (project_root / "results").mkdir(parents=True, exist_ok=True)
    shutil.copy2(OMNI_SRC, project_root / "data" / "raw" / "omni.csv")
    shutil.copy2(LHAASO_SRC, project_root / "data" / "raw" / "lhaaso.csv")
    (project_root / "config" / "evaluation_plan.json").write_text(
        json.dumps(
            {
                "schema_version": "evaluation-plan-v1",
                "split_label": "dev_backtest",
                "test_ratio": 0.2,
                "n_internal_folds": 2,
                "test_start": "2021-08-26",
                "notes": [
                    "real_v3_two_round acceptance: dev single-window execution.",
                    "f1/f2 remain frozen read-only external references.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return project_root


def compile_protocol(
    task: ScientificTask,
    candidate: CandidateExperiment,
    *,
    round_id: int,
    model_parameters: dict | None = None,
) -> tuple:
    protocol = CandidateProtocolMapper().to_protocol(
        task=task,
        candidate=candidate,
        round_id=round_id,
        model_name="ElasticNet",
        model_parameters=model_parameters,
    )
    validation = validate_experiment_protocol(protocol)
    if not validation.valid or validation.compiled_plan is None:
        raise RuntimeError(
            "protocol validation failed: " + "; ".join(validation.reasons)
        )
    return protocol, validation


def execute_protocol(project_root: Path, protocol, validation) -> dict:
    data_source = ElasticNetDataSourceConfig(
        omni_file_path=project_root / "data" / "raw" / "omni.csv",
        lhaaso_file_path=project_root / "data" / "raw" / "lhaaso.csv",
        time_column="TIME",
        target_column="SW Plasma Speed, km/s",
        test_split_ratio=0.2,
        random_state=42,
    )
    harness = UnifiedExperimentHarness(
        data_source=data_source,
        project_root=project_root,
        run_shap=False,
    )
    result = ExperimentDispatcher(harness).execute(protocol)
    if result.status != "completed":
        raise RuntimeError(
            f"execution failed: status={result.status} "
            f"validity={result.validity_status} warnings={result.warnings}"
        )
    if result.run_id is None:
        raise RuntimeError("execution returned no run_id")
    archive_dir = (
        harness.results_dir
        / f"round_{protocol.round_id:02d}"
        / "archive"
        / result.run_id
    )
    return {
        "result": result,
        "protocol": protocol,
        "validation": validation,
        "harness": harness,
        "archive_dir": archive_dir,
    }


def selected_params_from_archive(archive_dir: Path) -> dict[str, dict[str, float | None]]:
    import pandas as pd

    rows_path = archive_dir / "cv_results.csv"
    if not rows_path.exists():
        metrics_path = archive_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            return {
                run["run_id"]: {
                    "alpha": run.get("selected_alpha"),
                    "l1_ratio": run.get("selected_l1_ratio"),
                }
                for run in metrics.get("runs", [])
            }
        raise FileNotFoundError(f"no selected params archive: {archive_dir}")
    frame = pd.read_csv(rows_path)
    out: dict[str, dict[str, float | None]] = {}
    for run_id, group in frame.groupby("run_id"):
        alpha_values = group["selected_alpha"].dropna().unique()
        l1_values = group["selected_l1_ratio"].dropna().unique()
        out[str(run_id)] = {
            "alpha": float(alpha_values[0]) if len(alpha_values) else None,
            "l1_ratio": float(l1_values[0]) if len(l1_values) else None,
        }
    return out


def make_round2_candidate(
    round1_candidate: CandidateExperiment,
    selected: dict[str, dict[str, float | None]],
    round1_summary: dict,
) -> CandidateExperiment:
    candidate = round1_candidate.model_copy(deep=True)
    candidate.experiment_id = "E_R03_02_FIXED_PARAM"
    candidate.purpose = (
        "承接第一轮冻结自动选参结果：固定每臂在第一轮实际选中的 alpha/l1_ratio，"
        "再以完全相同特征、稀疏滞后、3 天提前与共同样本重复双臂对照，"
        "检验日影增量是否依赖训练内自动选参。"
    )
    candidate.scientific_question = "日影增量是否依赖第一轮训练内自动选参结果？"
    candidate.requires_human_review = True
    candidate.changed_factors = [
        "选参规则：第一轮训练内自动选参 -> 第二轮固定第一轮每臂实际参数",
        "其余冻结特征、稀疏滞后、提前量、共同样本与模型家族保持不变",
    ]
    candidate.held_fixed_factors = list(round1_candidate.held_fixed_factors)
    candidate.responded_prior_finding = (
        "第一轮冻结 E_R03_01 dev 窗口结果："
        + json.dumps(round1_summary.get("contrasts", []), ensure_ascii=False)
    )
    candidate.estimated_information_gain = round1_candidate.estimated_information_gain
    candidate.next_round_kind = "new_hypothesis_test"

    family_params = {
        "ridge": {
            "alpha": selected["ridge_baseline"]["alpha"],
            "l1_ratio": selected["ridge_baseline"]["l1_ratio"],
        },
        "elasticnet": {
            "alpha": selected["elasticnet_baseline"]["alpha"],
            "l1_ratio": selected["elasticnet_baseline"]["l1_ratio"],
        },
    }
    assert family_params["ridge"]["alpha"] == selected["ridge_treatment"]["alpha"]
    assert (
        family_params["elasticnet"]["alpha"]
        == selected["elasticnet_treatment"]["alpha"]
    )
    if candidate.design.runs_spec:
        for run in candidate.design.runs_spec:
            family = run.run_id.rsplit("_", 1)[0]
            params = {"param_search_mode": "fixed"}
            if family_params[family]["alpha"] is not None:
                params["alpha"] = float(family_params[family]["alpha"])
            if family == "elasticnet" and family_params[family]["l1_ratio"] is not None:
                params["l1_ratio"] = float(family_params[family]["l1_ratio"])
            run.parameters = params
    return candidate


def human_decisions(round_id: int, note: str) -> dict:
    return {
        "round_id": round_id,
        "decision_source": "human_pi_override",
        "note": note,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def build_summary(execution: dict, *, round_label: str) -> dict:
    result = execution["result"]
    protocol = execution["protocol"]
    archive_dir = execution["archive_dir"]
    metrics_path = archive_dir / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else {}
    )
    runs = []
    for run in result.runs:
        runs.append(
            {
                "run_id": run.run_id,
                "model_family": run.model_family,
                "n_train": run.n_train,
                "n_test": run.n_test,
                "pearson_r": run.metrics.pearson_r,
                "rmse": run.metrics.rmse,
                "mae": run.metrics.mae,
                "r2": run.metrics.r2,
                "unavailable_reasons": (
                    run.metrics.unavailable_reasons if run.metrics.unavailable_reasons else {}
                ),
                "parameters": run.parameters,
                "prediction_artifact": run.prediction_artifact,
            }
        )
    contrasts = []
    for contrast in result.contrasts:
        bootstrap = contrast.bootstrap_ci
        contrasts.append(
            {
                "contrast_id": contrast.contrast_id,
                "run_ids": contrast.run_ids,
                "delta_pearson_r": contrast.metrics_delta.pearson_r,
                "delta_rmse": contrast.metrics_delta.rmse,
                "delta_mae": contrast.metrics_delta.mae,
                "pearson_r_ci": bootstrap.pearson_r_ci if bootstrap else None,
                "rmse_ci": bootstrap.rmse_ci if bootstrap else None,
                "n_iterations": bootstrap.n_iterations if bootstrap else None,
                "block_size": bootstrap.block_size if bootstrap else None,
                "n_nonempty_blocks": bootstrap.n_nonempty_blocks if bootstrap else None,
                "seed": bootstrap.seed if bootstrap else None,
                "insufficient_blocks": (
                    bootstrap.insufficient_blocks if bootstrap else None
                ),
            }
        )
    return {
        "round_id": protocol.round_id,
        "round_label": round_label,
        "experiment_id": protocol.experiment_id,
        "run_id": result.run_id,
        "archive_dir": str(archive_dir.relative_to(REPO_ROOT)),
        "summary_metrics": metrics,
        "runs": runs,
        "contrasts": contrasts,
        "validity_status": result.validity_status,
        "warnings": result.warnings,
    }


def persist_json(execution: dict, run_root: Path, name: str) -> None:
    payload = {
        "protocol": execution["protocol"].model_dump(mode="json", exclude_none=True),
        "result": execution["result"].model_dump(mode="json", exclude_none=True),
    }
    (run_root / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    started_at = datetime.now(timezone.utc)
    stamp = started_at.strftime("%Y%m%d_%H%M%S")
    run_root = OUTPUT_ROOT / stamp
    run_root.mkdir(parents=True, exist_ok=False)
    project_root = prepare_project_root(run_root)

    records: list[dict] = []
    try:
        task = make_task()
        round1_candidate = load_fixed_candidate()
        protocol1, validation1 = compile_protocol(
            task, round1_candidate, round_id=1
        )
        exec1 = execute_protocol(project_root, protocol1, validation1)
        persist_json(exec1, run_root, "round1_execution.json")
        summary1 = build_summary(exec1, round_label="round1_auto_selection")
        (run_root / "round1_summary.json").write_text(
            json.dumps(summary1, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append(
            {
                **human_decisions(
                    1,
                    "PI 批准执行冻结 E_R03_01；自动选参为冻结协议，不属于模型自主选择。",
                ),
                "experiment_id": protocol1.experiment_id,
            }
        )

        selected = selected_params_from_archive(exec1["archive_dir"])
        (run_root / "round1_selected_params.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        round2_candidate = make_round2_candidate(
            round1_candidate, selected, summary1
        )
        protocol2, validation2 = compile_protocol(
            task, round2_candidate, round_id=2
        )
        exec2 = execute_protocol(project_root, protocol2, validation2)
        persist_json(exec2, run_root, "round2_execution.json")
        summary2 = build_summary(exec2, round_label="round2_fixed_params")
        (run_root / "round2_summary.json").write_text(
            json.dumps(summary2, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append(
            {
                **human_decisions(
                    2,
                    "PI 基于第一轮实际选参结果显式指定固定参数复核；"
                    "该选择不是模型自主规划，记为 human_pi_override。",
                ),
                "experiment_id": protocol2.experiment_id,
            }
        )

        manifest = {
            "generated_at": started_at.isoformat(),
            "repo_head": git_head(),
            "session_id": f"RV3TR_{stamp}",
            "task_id": task.task_id,
            "evaluation_primary_metric": "Pearson_r",
            "evaluation_secondary_metrics": ["RMSE", "MAE", "R2"],
            "data_sources": {
                "omni": str(OMNI_SRC.name),
                "lhaaso": str(LHAASO_SRC.name),
                "copied_to": "project/data/raw",
            },
            "decision_log": records,
            "rounds": {
                "round1": "round1_summary.json",
                "round2": "round2_summary.json",
            },
            "artifacts_required": [
                "round1_execution.json",
                "round2_execution.json",
                "prediction CSV archives in each round archive dir",
            ],
            "boundary": (
                "dev_backtest is an iterative evaluation segment; f1/f2 remain "
                "frozen external read-only references. CI crossing zero is "
                "insufficient evidence, not a failed run. No positive-gain "
                "target was imposed."
            ),
        }
        (run_root / "session_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[done] real two-round acceptance root: {run_root}")
        print(json.dumps({"round1": summary1, "round2": summary2}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        import traceback

        (run_root / "failure_report.json").write_text(
            json.dumps(
                {
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "decision_log": records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
