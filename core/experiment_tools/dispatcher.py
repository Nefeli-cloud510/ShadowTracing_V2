"""Dispatcher that maps a validated protocol to its registered executor."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from core.experiment_tools.compiled_plan import normalize_diagnostic_id
from core.experiment_tools.registry import get_tool
from core.experiment_tools.validate_protocol import validate_experiment_protocol
from core.harness_unified import UnifiedExperimentHarness
from core.result_unified import build_metric_delta, build_run_metrics, export_result_bundle
from core.robustness_diagnostics import (
    empty_bootstrap_summary,
    paired_calendar_block_bootstrap,
    summary_to_robustness_report,
)
from core.unified_schema import (
    DataSourceCoverage,
    DiagnosticSpec,
    ExecutionSummary,
    ExperimentDiagnostic,
    ExperimentProtocol,
    ExperimentResult,
    ExperimentRun,
    RunSpec,
    TimeSliceFold,
    TimeSliceReport,
)


class _PredictionCandidate(NamedTuple):
    path: Path
    kind: str
    execution_id: str | None = None
    matched_ids: tuple[str, ...] = ()


class ExperimentDispatcher:
    """Execute validated structured protocols through the shared harness."""

    def __init__(self, harness: UnifiedExperimentHarness) -> None:
        self.harness = harness
        self.project_root = harness.project_root

    def execute(self, protocol: ExperimentProtocol) -> ExperimentResult:
        validation = validate_experiment_protocol(protocol)
        tool = get_tool(protocol.experiment_type)
        if not validation.valid:
            return self._invalid_result(protocol, validation.reasons)
        if tool is None or not tool.available:
            return self._unsupported_result(protocol, validation)

        if protocol.experiment_type == "incremental_prediction":
            result = self.harness.run_compiled(protocol, validation.compiled_plan)
        elif protocol.experiment_type == "baseline_benchmark":
            result = self.harness.run_compiled(protocol, validation.compiled_plan)
        elif protocol.experiment_type == "model_sensitivity":
            result = self._run_model_sensitivity(protocol, validation)
        elif protocol.experiment_type == "robustness_diagnostic":
            result = self._run_robustness_diagnostic(protocol, validation)
        else:
            return self._unsupported_result(protocol, validation)
        return result

    def _run_model_sensitivity(self, protocol: ExperimentProtocol, validation) -> ExperimentResult:
        result = self.harness.run_compiled(protocol, validation.compiled_plan)
        gain_rows = []
        for contrast in result.contrasts:
            gain_rows.append(
                {
                    "contrast_id": contrast.contrast_id,
                    "run_ids": contrast.run_ids,
                    "delta_rmse": contrast.metrics_delta.rmse,
                    "delta_pearson_r": contrast.metrics_delta.pearson_r,
                }
            )
        result.diagnostics.append(
            ExperimentDiagnostic(
                diagnostic_id="family_gain_summary",
                name="各模型家族内日影增量汇总",
                source_run_ids=[run.run_id for run in result.runs],
                structured={
                    "gains": gain_rows,
                    "rule": "只比较同一模型家族内部 baseline/treatment；跨家族不得错误归因。",
                },
                status="completed",
            )
        )
        round_dir = self.harness.results_dir / f"round_{protocol.round_id:02d}"
        export_result_bundle(
            result,
            protocol,
            round_dir=round_dir,
            latest_protocol_path=self.harness.latest_protocol_path,
            latest_metrics_path=self.harness.latest_metrics_path,
            write_latest=not self.harness.is_blind_sealed(),
        )
        return result

    def _run_robustness_diagnostic(
        self,
        protocol: ExperimentProtocol,
        validation,
    ) -> ExperimentResult:
        run_ids: list[str] = [run.run_id for run in protocol.runs_spec] or [
            "baseline",
            "treatment",
        ]
        if len(run_ids) != 2:
            return self._invalid_result(
                protocol,
                [
                    "robustness_diagnostic 必须恰好包含两个 retrain=False 的复用 run，"
                    "当前 runs_spec 为空或数量不是两个。"
                ],
            )
        try:
            base_frame, treat_frame, source_meta = self._resolve_source_predictions(
                protocol=protocol,
                run_ids=run_ids,
            )
        except (FileNotFoundError, ValueError, AssertionError) as exc:
            return self._invalid_result(protocol, [str(exc)])

        diagnostics_specs = [
            DiagnosticSpec.model_validate(item.model_dump())
            for item in protocol.diagnostics_spec
        ]
        normalized_specs = {
            normalized: spec
            for spec in diagnostics_specs
            if (normalized := normalize_diagnostic_id(spec.diagnostic_id))
        }

        started = time.perf_counter()
        diagnostics: list[ExperimentDiagnostic] = []
        bootstrap_summary = empty_bootstrap_summary(
            "diagnostics_spec 未声明 paired_calendar_bootstrap。"
        )

        bootstrap_spec = normalized_specs.get("paired_calendar_bootstrap")
        if bootstrap_spec is not None:
            params = bootstrap_spec.parameters or {}
            block_size = int(params.get("block_size", 7))
            n_draws = int(params.get("n_draws", 4000))
            seed = int(params.get("seed", 20261008))
            bootstrap_summary = paired_calendar_block_bootstrap(
                base_frame,
                treat_frame,
                block_size=block_size,
                n_draws=n_draws,
                seed=seed,
            )
            diagnostics.append(
                ExperimentDiagnostic(
                    diagnostic_id="paired_calendar_bootstrap",
                    name=bootstrap_spec.name,
                    source_run_ids=list(run_ids),
                    structured=bootstrap_summary,
                    status="completed",
                )
            )

        robustness = summary_to_robustness_report(bootstrap_summary)

        if "time_segments" in normalized_specs:
            time_segment_payload = self._time_segment_diagnostics(
                base_frame,
                treat_frame,
            )
            diagnostics.append(
                ExperimentDiagnostic(
                    diagnostic_id="time_segments",
                    name=normalized_specs["time_segments"].name,
                    source_run_ids=list(run_ids),
                    structured=time_segment_payload,
                    status="completed",
                )
            )
            time_slice = TimeSliceReport(
                splits=[
                    TimeSliceFold(
                        fold=index + 1,
                        segment=segment.get("segment"),
                        date_start=segment.get("date_start"),
                        date_end=segment.get("date_end"),
                        n_samples=segment.get("n_samples"),
                        baseline_rmse=segment.get("baseline_rmse"),
                        treatment_rmse=segment.get("treatment_rmse"),
                        baseline_bias=segment.get("baseline_bias"),
                        treatment_bias=segment.get("treatment_bias"),
                    )
                    for index, segment in enumerate(
                        time_segment_payload.get("segments") or []
                    )
                ],
                stable=False,
                status="descriptive",
                metric="RMSE",
                reason="已执行 early/middle/late 描述性诊断，未按预登记规则作稳定性判定",
            )
            recommendation = robustness.recommendation or ""
            stale_phrase = "未运行分时段检验，不据此作分时段稳定性结论。"
            if stale_phrase in recommendation:
                recommendation = recommendation.replace(
                    stale_phrase,
                    "已执行描述性分时段诊断，未按预登记规则作稳定性判定。",
                )
            else:
                recommendation = (
                    recommendation.rstrip("。；")
                    + "；已执行描述性分时段诊断，未按预登记规则作稳定性判定。"
                )
            robustness = robustness.model_copy(
                update={
                    "time_slice": time_slice,
                    "recommendation": recommendation,
                }
            )
        if "residual_diagnostics" in normalized_specs:
            diagnostics.append(
                ExperimentDiagnostic(
                    diagnostic_id="residual_diagnostics",
                    name=normalized_specs["residual_diagnostics"].name,
                    source_run_ids=list(run_ids),
                    structured=self._residual_diagnostics(
                        base_frame,
                        treat_frame,
                    ),
                    status="completed",
                )
            )
        duration_seconds = time.perf_counter() - started

        base_metrics = _metrics_from_frame(base_frame)
        treat_metrics = _metrics_from_frame(treat_frame)
        label_by_run = {run.run_id: run.label for run in protocol.runs_spec}
        runs = [
            ExperimentRun(
                run_id=run_ids[0],
                name=label_by_run.get(run_ids[0], "对照组（复用预测）"),
                prediction_artifact=source_meta["frames"][0]["path"],
                metrics=build_run_metrics(base_metrics),
                model_family=str(protocol.model.name or "reused"),
                feature_labels=[],
                feature_columns=[],
                parameters={
                    "source_experiment_id": source_meta["frames"][0][
                        "source_experiment_id"
                    ],
                    "source_execution_id": source_meta["frames"][0].get(
                        "source_execution_id"
                    ),
                    "source_round_id": source_meta["frames"][0]["source_round_id"],
                    "source_run_id": source_meta["frames"][0].get("source_run_id"),
                    "prediction_artifact_hash": source_meta["frames"][0][
                        "prediction_artifact_hash"
                    ],
                },
                n_train=None,
                n_test=int(len(base_frame)),
                training_mode="reused",
                protocol_version=protocol.protocol_version,
            ),
            ExperimentRun(
                run_id=run_ids[1],
                name=label_by_run.get(run_ids[1], "实验组（复用预测）"),
                prediction_artifact=source_meta["frames"][1]["path"],
                metrics=build_run_metrics(treat_metrics),
                model_family=str(protocol.model.name or "reused"),
                feature_labels=[],
                feature_columns=[],
                parameters={
                    "source_experiment_id": source_meta["frames"][1][
                        "source_experiment_id"
                    ],
                    "source_execution_id": source_meta["frames"][1].get(
                        "source_execution_id"
                    ),
                    "source_round_id": source_meta["frames"][1]["source_round_id"],
                    "source_run_id": source_meta["frames"][1].get("source_run_id"),
                    "prediction_artifact_hash": source_meta["frames"][1][
                        "prediction_artifact_hash"
                    ],
                },
                n_train=None,
                n_test=int(len(treat_frame)),
                training_mode="reused",
                protocol_version=protocol.protocol_version,
            ),
        ]

        warnings = ["复用已有预测结果，未重新训练模型。"]
        if bootstrap_summary.get("status") == "completed":
            if not bootstrap_summary.get("stable_rmse"):
                warnings.append(
                    "不确定性的 bootstrap CI 跨越零，结论方向待科学评价。"
                )
            if not bootstrap_summary.get("n_valid_pearson_draws"):
                warnings.append(
                    "有效 Pearson 重采样次数为 0，相关系数区间不可用。"
                )
        else:
            warnings.append(
                f"bootstrap 未运行：{bootstrap_summary.get('reason') or '未配置'}。"
            )

        result = ExperimentResult(
            experiment_id=protocol.experiment_id,
            round_id=protocol.round_id,
            status="completed",
            experiment_type=protocol.experiment_type,
            protocol_version=protocol.protocol_version,
            runs=runs,
            comparison=build_metric_delta(base_metrics, treat_metrics),
            diagnostics=diagnostics,
            source_evidence_ids=list(protocol.source_evidence_ids),
            allowed_conclusions=list(protocol.allowed_conclusions),
            unresolved_questions=list(protocol.unresolved_questions),
            validity_status="valid",
            warnings=warnings,
            visualizations=[],
            data_coverage=[
                DataSourceCoverage(
                    source=(
                        source_meta["frames"][0].get("source_experiment_id")
                        or "source_predictions"
                    ),
                    run_id=runs[0].run_id,
                    observed_days=int(len(base_frame)),
                    note=(
                        f"复用预测文件 {source_meta['frames'][0]['path']}，"
                        "SHA-256 已核验。"
                    ),
                ),
                DataSourceCoverage(
                    source=(
                        source_meta["frames"][1].get("source_experiment_id")
                        or "source_predictions"
                    ),
                    run_id=runs[1].run_id,
                    observed_days=int(len(treat_frame)),
                    note=(
                        f"复用预测文件 {source_meta['frames'][1]['path']}，"
                        "SHA-256 已核验。"
                    ),
                ),
            ],
            execution=ExecutionSummary(
                duration_seconds=duration_seconds,
                compute_seconds=duration_seconds,
                status="completed",
            ),
            robustness=robustness,
        )
        round_dir = self.harness.results_dir / f"round_{protocol.round_id:02d}"
        export_result_bundle(
            result,
            protocol,
            round_dir=round_dir,
            latest_protocol_path=self.harness.latest_protocol_path,
            latest_metrics_path=self.harness.latest_metrics_path,
            write_latest=not self.harness.is_blind_sealed(),
        )
        return result

    def _resolve_source_predictions(
        self,
        *,
        protocol: ExperimentProtocol,
        run_ids: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        defaults = dict(protocol.model.parameters)
        refs: list[dict] = []
        for index, run_id in enumerate(run_ids):
            run_spec = next(
                (item for item in protocol.runs_spec if item.run_id == run_id),
                None,
            )
            ref = dict(defaults)
            if run_spec is not None:
                ref.update(run_spec.parameters)
                if run_spec.source_experiment_id:
                    ref["source_experiment_id"] = run_spec.source_experiment_id
                if run_spec.source_execution_id:
                    ref["source_execution_id"] = run_spec.source_execution_id
                if run_spec.source_prediction_file:
                    ref["source_prediction_file"] = run_spec.source_prediction_file
            ref["_run_id"] = run_id
            ref["_arm_hint"] = "baseline" if index == 0 else "treatment"
            refs.append(ref)

        loaded = [self._load_source_frame(protocol, ref) for ref in refs]
        base_frame, base_meta = loaded[0]
        treat_frame, treat_meta = loaded[1]
        self._assert_paired_frames(base_frame, treat_frame)
        return (
            base_frame,
            treat_frame,
            {"frames": [base_meta, treat_meta]},
        )

    def _load_source_frame(
        self,
        protocol: ExperimentProtocol,
        ref: dict,
    ) -> tuple[pd.DataFrame, dict]:
        source_experiment_id = str(ref.get("source_experiment_id") or "legacy")
        raw_round = ref.get("source_round_id") or protocol.round_id
        try:
            source_round_id = int(raw_round)
        except (TypeError, ValueError):
            source_round_id = protocol.round_id

        prediction_file = ref.get("source_prediction_file")
        if prediction_file:
            path = Path(str(prediction_file))
            if not path.is_absolute():
                path = self.project_root / path
            if not path.exists():
                raise FileNotFoundError(
                    f"robustness_diagnostic 找不到来源预测文件 {path}。"
                )
            frame = self._read_prediction_file(path, ref)
        else:
            path, frame = self._search_source_predictions(
                protocol,
                ref=ref,
                source_experiment_id=source_experiment_id,
                source_round_id=source_round_id,
            )

        self._verify_artifact_hash(path, ref.get("prediction_artifact_hash"))
        frame = self._normalize_frame(frame)
        self._validate_source_metadata(protocol, ref, path, frame)
        return frame, {
            "path": str(path.resolve()),
            "source_experiment_id": source_experiment_id,
            "source_execution_id": ref.get("source_execution_id")
            or self._execution_id_from_path(path),
            "source_round_id": source_round_id,
            "source_run_id": ref.get("source_run_id"),
            "prediction_artifact_hash": self._file_sha256(path),
            "run_id": str(ref.get("_run_id") or ""),
        }

    @staticmethod
    def _execution_id_from_path(path: Path) -> str | None:
        parts = path.resolve().parts
        try:
            index = parts.index("archive")
        except ValueError:
            return None
        if index + 1 < len(parts):
            return parts[index + 1]
        return None

    @staticmethod
    def _read_prediction_file(path: Path, ref: dict) -> pd.DataFrame:
        frame = pd.read_csv(path)
        if "arm" in frame.columns:
            preferred = list(
                dict.fromkeys(
                    str(item)
                    for item in (
                        ref.get("source_run_id"),
                        ref.get("_run_id"),
                        ref.get("_arm_hint"),
                    )
                    if item is not None
                )
            )
            available = sorted(
                {str(item) for item in frame["arm"].dropna().unique()}
            )
            matched = [name for name in preferred if name in available]
            if len(matched) == 1:
                frame = frame[frame["arm"].astype(str) == matched[0]].reset_index(
                    drop=True
                )
            elif len(matched) > 1:
                raise ValueError(
                    f"来源预测文件 {path} 的 arm={available} 与多个候选 {preferred} 匹配，"
                    "必须用 source_run_id 唯一指定该 run 对应的臂。"
                )
            else:
                raise ValueError(
                    f"来源预测文件 {path} 的 arm={available} 中没有 run "
                    f"{ref.get('_run_id') or ref.get('_arm_hint')} 对应的臂。"
                )
        required = {"sample_id", "issue_date", "target_date", "y_true", "y_pred"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"来源预测文件 {path} 缺少必要列：{sorted(missing)}。"
            )
        return frame

    def _search_source_predictions(
        self,
        protocol: ExperimentProtocol,
        *,
        ref: dict,
        source_experiment_id: str,
        source_round_id: int,
    ) -> tuple[Path, pd.DataFrame]:
        for round_dir in self._round_dirs(protocol, source_round_id):
            candidates = [
                *self._direct_prediction_candidates(
                    round_dir,
                    source_experiment_id,
                    ref,
                ),
                *self._archive_prediction_candidates(
                    round_dir,
                    source_experiment_id,
                    ref,
                ),
            ]
            existing = [candidate for candidate in candidates if candidate.path.exists()]
            if not existing:
                continue
            candidate = self._resolve_unique_prediction_candidate(
                existing,
                round_dir,
                ref,
            )
            return candidate.path, self._read_prediction_file(candidate.path, ref)
        raise FileNotFoundError(
            f"robustness_diagnostic 找不到来源实验 {source_experiment_id!r} 的预测文件"
            f"（source_round_id={source_round_id or protocol.round_id}），"
            "已搜索 results/round_*/ 下的 per-run 与 archive 预测。"
        )

    def _round_dirs(
        self,
        protocol: ExperimentProtocol,
        source_round_id: int,
    ) -> list[Path]:
        if source_round_id:
            return [
                self.harness.results_dir / f"round_{int(source_round_id):02d}"
            ]
        current = [self.harness.results_dir / f"round_{protocol.round_id:02d}"]
        previous = [
            self.harness.results_dir / f"round_{index:02d}"
            for index in range(protocol.round_id - 1, 0, -1)
        ]
        return [*current, *previous]

    def _direct_prediction_candidates(
        self,
        round_dir: Path,
        source_experiment_id: str,
        ref: dict,
    ) -> list[_PredictionCandidate]:
        names = list(
            dict.fromkeys(
                str(item)
                for item in (
                    ref.get("source_run_id"),
                    ref.get("_run_id"),
                    ref.get("_arm_hint"),
                )
                if item is not None
            )
        )
        candidates: list[Path] = []
        for name in names:
            candidates.extend(
                [
                    round_dir
                    / source_experiment_id
                    / "predictions"
                    / f"{source_experiment_id}_{name}.csv",
                    round_dir
                    / source_experiment_id
                    / f"{source_experiment_id}_{name}.csv",
                    round_dir / f"{source_experiment_id}_{name}.csv",
                ]
            )
        return [
            _PredictionCandidate(path=path, kind="direct")
            for path in candidates
        ]

    def _archive_prediction_candidates(
        self,
        round_dir: Path,
        source_experiment_id: str,
        ref: dict,
    ) -> list[_PredictionCandidate]:
        archive_root = round_dir / "archive"
        if not archive_root.exists():
            return []
        matches: list[_PredictionCandidate] = []
        for archive_dir in sorted(archive_root.iterdir()):
            if not archive_dir.is_dir():
                continue
            if not (archive_dir / "predictions.csv").exists():
                continue
            if not self._archive_matches_experiment(
                archive_dir,
                source_experiment_id,
            ):
                continue
            info = self._archive_info(archive_dir)
            if not self._archive_matches_ref(info, ref):
                continue
            matches.append(
                _PredictionCandidate(
                    path=archive_dir / "predictions.csv",
                    kind="archive",
                    execution_id=str(info["execution_id"]),
                    matched_ids=tuple(sorted(str(item) for item in info["run_ids"])),
                )
            )
        return matches

    @staticmethod
    def _archive_matches_experiment(
        archive_dir: Path,
        source_experiment_id: str,
    ) -> bool:
        for meta_name in ("experiment_protocol.json", "metrics.json"):
            meta_path = archive_dir / meta_name
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if meta.get("experiment_id") == source_experiment_id:
                return True
        return False

    @staticmethod
    def _archive_info(archive_dir: Path) -> dict[str, object]:
        run_ids: set[str] = set()
        execution_id = archive_dir.name
        metrics_path = archive_dir / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                metrics = {}
            metric_run_id = metrics.get("run_id")
            if metric_run_id:
                execution_id = str(metric_run_id)
                run_ids.add(str(metric_run_id))
            for run in metrics.get("runs") or []:
                if run.get("run_id"):
                    run_ids.add(str(run["run_id"]))

        frame_path = archive_dir / "predictions.csv"
        if frame_path.exists():
            try:
                arms = pd.read_csv(frame_path, usecols=["arm"])
                run_ids.update(str(item) for item in arms["arm"].dropna().unique())
            except (ValueError, OSError):
                pass

        return {
            "execution_id": execution_id,
            "run_ids": run_ids,
        }

    @staticmethod
    def _archive_matches_ref(info: dict[str, object], ref: dict) -> bool:
        run_ids = set(info["run_ids"])
        requested_execution = ref.get("source_execution_id")
        if requested_execution and str(info["execution_id"]) != str(
            requested_execution
        ):
            return False
        requested = [
            str(item)
            for item in (
                ref.get("source_run_id"),
                ref.get("_run_id"),
                ref.get("_arm_hint"),
            )
            if item is not None
        ]
        return any(item in run_ids for item in requested)

    def _resolve_unique_prediction_candidate(
        self,
        candidates: list[_PredictionCandidate],
        round_dir: Path,
        ref: dict,
    ) -> _PredictionCandidate:
        selected = self._filter_execution_candidates(candidates, ref)
        selected = self._filter_hash_candidates(selected, ref)
        if len(selected) == 1:
            return selected[0]

        by_sha: dict[str, list[_PredictionCandidate]] = {}
        for candidate in selected:
            by_sha.setdefault(self._file_sha256(candidate.path), []).append(
                candidate
            )
        if len(by_sha) == 1:
            return self._prefer_canonical_candidate(
                next(iter(by_sha.values())),
                round_dir,
            )

        by_content: dict[str, list[_PredictionCandidate]] = {}
        for candidate in selected:
            by_content.setdefault(
                self._prediction_content_digest(candidate.path, ref),
                [],
            ).append(candidate)
        if len(by_content) == 1:
            return self._prefer_canonical_candidate(
                next(iter(by_content.values())),
                round_dir,
            )

        detail = "；".join(
            (
                f"{candidate.path}（kind={candidate.kind}, "
                f"execution_id={candidate.execution_id or '--'}, "
                f"sha256={self._file_sha256(candidate.path)}）"
            )
            for candidate in selected
        )
        raise ValueError(
            "来源实验定位存在歧义，匹配到多个科学上不同的预测文件版本："
            + detail
            + "。请用 source_execution_id 或 prediction_artifact_hash 精确指定。"
        )

    def _filter_execution_candidates(
        self,
        candidates: list[_PredictionCandidate],
        ref: dict,
    ) -> list[_PredictionCandidate]:
        requested = ref.get("source_execution_id")
        if not requested:
            return list(candidates)
        matched = [
            candidate
            for candidate in candidates
            if candidate.kind == "archive"
            and candidate.execution_id == str(requested)
        ]
        if matched:
            return matched
        available = sorted(
            {
                str(candidate.execution_id)
                for candidate in candidates
                if candidate.execution_id
            }
        )
        hint = (
            "；可用 execution_id：" + ", ".join(available)
            if available
            else "未找到任何 archive"
        )
        raise ValueError(
            f"来源实验定位失败：source_execution_id={requested} 未匹配任何 "
            f"archive（{hint}）。"
        )

    def _filter_hash_candidates(
        self,
        candidates: list[_PredictionCandidate],
        ref: dict,
    ) -> list[_PredictionCandidate]:
        expected_raw = ref.get("prediction_artifact_hash")
        if not expected_raw:
            return list(candidates)
        expected = str(expected_raw).strip().lower()
        if expected.startswith("sha256:"):
            expected = expected[len("sha256:") :]
        matched = [
            candidate
            for candidate in candidates
            if self._file_sha256(candidate.path) == expected
        ]
        if matched:
            return matched
        detail = "；".join(
            f"{candidate.path}（实际 {self._file_sha256(candidate.path)}）"
            for candidate in candidates
        )
        raise ValueError(
            "来源预测文件 SHA-256 与 prediction_artifact_hash 不一致："
            f"期望 {expected}；候选：{detail}。"
        )

    def _prefer_canonical_candidate(
        self,
        candidates: list[_PredictionCandidate],
        round_dir: Path,
    ) -> _PredictionCandidate:
        archives = [candidate for candidate in candidates if candidate.kind == "archive"]
        if not archives:
            return candidates[0]
        latest_path = round_dir / "archive" / "latest_run.json"
        if latest_path.exists():
            try:
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                latest = {}
            latest_run_id = str(latest.get("run_id") or "")
            for candidate in archives:
                if candidate.execution_id == latest_run_id:
                    return candidate
        return archives[0]

    @classmethod
    def _prediction_content_digest(cls, path: Path, ref: dict) -> str:
        frame = cls._read_prediction_file(path, ref).copy()
        frame = frame.sort_values("sample_id", kind="stable", ignore_index=True)
        columns = ["sample_id", "issue_date", "target_date", "y_true", "y_pred"]
        payload = frame.loc[:, columns].to_csv(
            index=False,
            encoding="utf-8",
            float_format="%.12g",
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _verify_artifact_hash(self, path: Path, expected_hash) -> None:
        if not expected_hash:
            return
        expected = str(expected_hash).strip().lower()
        if expected.startswith("sha256:"):
            expected = expected[len("sha256:") :]
        actual = self._file_sha256(path)
        if expected != actual:
            raise ValueError(
                f"来源预测文件 {path} 的 SHA-256 与 prediction_artifact_hash 不一致："
                f"期望 {expected}，实际 {actual}。"
            )

    def _validate_source_metadata(
        self,
        protocol: ExperimentProtocol,
        ref: dict,
        path: Path,
        frame: pd.DataFrame,
    ) -> None:
        declared_horizon = ref.get("forecast_horizon_days")
        if declared_horizon is None:
            declared_horizon = protocol.model.parameters.get(
                "forecast_horizon_days"
            )
        expected_horizon = (
            int(declared_horizon)
            if declared_horizon is not None
            else protocol.forecast_horizon_days
        )
        if expected_horizon is not None:
            issue = pd.to_datetime(frame["issue_date"]).dt.normalize()
            target = pd.to_datetime(frame["target_date"]).dt.normalize()
            horizons = (target - issue).dt.days
            mismatched = int(np.count_nonzero(horizons != int(expected_horizon)))
            if mismatched:
                raise ValueError(
                    f"来源预测 {path} 的目标提前量与协议不一致：期望 "
                    f"{expected_horizon} 天，{mismatched} 行不匹配。"
                )

        base_dir = path.parent
        manifest_path = base_dir / "sample_manifest.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path, dtype={"sample_id": str})
            frame_ids = set(frame["sample_id"].astype(str))
            manifest_ids = set(manifest["sample_id"].astype(str))
            missing_ids = frame_ids - manifest_ids
            if missing_ids:
                raise ValueError(
                    f"来源预测 {path} 有 {len(missing_ids)} 个 sample_id 不在 "
                    "sample_manifest.csv 中，预测行与样本清单不一致。"
                )
            covered = manifest[manifest["sample_id"].astype(str).isin(frame_ids)]
            if "split" in covered.columns:
                non_test = sorted(
                    {
                        str(item)
                        for item in covered["split"].astype(str).dropna().unique()
                        if str(item) != "test"
                    }
                )
                if non_test:
                    raise ValueError(
                        f"来源预测 {path} 引用了非 test 样本，split={non_test}。"
                    )
            if (
                expected_horizon is not None
                and "horizon_days" in covered.columns
            ):
                horizons = {
                    int(item)
                    for item in pd.to_numeric(
                        covered["horizon_days"],
                        errors="coerce",
                    ).dropna()
                }
                if horizons and horizons != {int(expected_horizon)}:
                    raise ValueError(
                        f"来源预测 {path} 的 sample_manifest horizon_days="
                        f"{sorted(horizons)} 与协议 {expected_horizon} 不一致。"
                    )

        for meta_name in ("prediction_protocol.json", "experiment_protocol.json"):
            meta_path = base_dir / meta_name
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            source_target = meta.get("target")
            if source_target and source_target != protocol.target:
                raise ValueError(
                    f"来源预测 {path} 的目标字段 {source_target!r} 与当前协议 "
                    f"{protocol.target!r} 不一致。"
                )
            source_params = (meta.get("model") or {}).get("parameters") or {}
            for key in ("forecast_horizon_days", "past_lag_days"):
                declared = ref.get(key)
                if declared is None:
                    declared = protocol.model.parameters.get(key)
                actual = source_params.get(key)
                if (
                    declared is not None
                    and actual is not None
                    and float(actual) != float(declared)
                ):
                    raise ValueError(
                        f"来源预测 {path} 的 {key}={actual} 与当前协议 "
                        f"{declared} 不一致。"
                    )

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame[
            ["sample_id", "issue_date", "target_date", "y_true", "y_pred"]
        ].copy()
        normalized["sample_id"] = normalized["sample_id"].astype(str)
        return normalized.reset_index(drop=True)

    @staticmethod
    def _assert_paired_frames(
        base_frame: pd.DataFrame,
        treat_frame: pd.DataFrame,
    ) -> None:
        if len(base_frame) == 0 or len(treat_frame) == 0:
            raise ValueError("来源预测配对后存在空臂，无法执行 bootstrap/诊断。")
        if len(base_frame) != len(treat_frame):
            raise ValueError(
                "配对来源预测样本数不一致："
                f"{len(base_frame)} vs {len(treat_frame)}。"
            )
        base_ids = list(base_frame["sample_id"].astype(str))
        treat_ids = list(treat_frame["sample_id"].astype(str))
        if base_ids != treat_ids:
            raise ValueError(
                "配对来源预测 sample_id 顺序/集合不一致，禁止隐式重排后比较增益。"
            )
        if list(base_frame["target_date"]) != list(treat_frame["target_date"]):
            raise ValueError(
                "配对来源预测 target_date 顺序/集合不一致，禁止隐式重排后比较增益。"
            )
        if not np.allclose(
            base_frame["y_true"].to_numpy(dtype=float),
            treat_frame["y_true"].to_numpy(dtype=float),
        ):
            raise ValueError(
                "配对来源预测 y_true 不一致，公共真实标签语义被破坏。"
            )

    @staticmethod
    def _time_segment_diagnostics(
        base_frame: pd.DataFrame,
        treat_frame: pd.DataFrame,
    ) -> dict:
        segments = []
        dates = pd.to_datetime(treat_frame["target_date"])
        labels = ["early", "middle", "late"]
        bounds = np.quantile(
            np.arange(len(dates)),
            [1 / 3, 2 / 3],
        ).astype(int)
        for label, start, end in [
            (labels[0], 0, bounds[0]),
            (labels[1], bounds[0], bounds[1]),
            (labels[2], bounds[1], len(dates)),
        ]:
            if end <= start:
                continue
            seg_base = base_frame.iloc[start:end]
            seg_treat = treat_frame.iloc[start:end]
            base_metrics = _metrics_from_frame(seg_base)
            treat_metrics = _metrics_from_frame(seg_treat)
            segments.append(
                {
                    "segment": label,
                    "n_samples": int(end - start),
                    "date_start": str(dates.iloc[start].date()),
                    "date_end": str(dates.iloc[end - 1].date()),
                    "baseline_rmse": base_metrics["test_rmse"],
                    "treatment_rmse": treat_metrics["test_rmse"],
                    "baseline_bias": _mean_bias(seg_base),
                    "treatment_bias": _mean_bias(seg_treat),
                }
            )
        return {"segments": segments, "note": "仅按日历顺序分段，不用于删除难预测日期。"}

    @staticmethod
    def _residual_diagnostics(
        base_frame: pd.DataFrame,
        treat_frame: pd.DataFrame,
    ) -> dict:
        base_metrics = _metrics_from_frame(base_frame)
        treat_metrics = _metrics_from_frame(treat_frame)
        return {
            "baseline": {
                **base_metrics,
                "mean_bias": _mean_bias(base_frame),
                "prediction_range": [
                    float(base_frame["y_pred"].min()),
                    float(base_frame["y_pred"].max()),
                ],
            },
            "treatment": {
                **treat_metrics,
                "mean_bias": _mean_bias(treat_frame),
                "prediction_range": [
                    float(treat_frame["y_pred"].min()),
                    float(treat_frame["y_pred"].max()),
                ],
            },
        }

    def _invalid_result(
        self,
        protocol: ExperimentProtocol,
        reasons: list[str],
    ) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=protocol.experiment_id,
            round_id=protocol.round_id,
            status="failed",
            experiment_type=protocol.experiment_type,
            protocol_version=protocol.protocol_version,
            validity_status="invalid",
            warnings=list(reasons),
            execution=ExecutionSummary(
                status="failed",
                message="协议结构校验失败；" + "；".join(reasons),
            ),
        )

    def _unsupported_result(
        self,
        protocol: ExperimentProtocol,
        validation,
    ) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=protocol.experiment_id,
            round_id=protocol.round_id,
            status="failed",
            experiment_type=protocol.experiment_type,
            protocol_version=protocol.protocol_version,
            validity_status="pending_scientific_review",
            warnings=[
                f"实验类型 {protocol.experiment_type} 尚未接通执行器，禁止静默降级。"
                + ("；".join(validation.warnings) if validation.warnings else "")
            ],
            execution=ExecutionSummary(
                status="failed",
                message=f"unsupported experiment_type={protocol.experiment_type}",
            ),
        )


def _metrics_from_frame(frame: pd.DataFrame) -> dict:
    y_true = frame["y_true"].astype(float).to_numpy()
    y_pred = frame["y_pred"].astype(float).to_numpy()
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    with np.errstate(invalid="ignore", divide="ignore"):
        pearson_raw = float(np.corrcoef(y_true, y_pred)[0, 1])
    pearson_r = (
        pearson_raw if np.isfinite(pearson_raw) else None
    )
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {
        "test_rmse": rmse,
        "test_mae": mae,
        "test_pearson_r": pearson_r,
        "test_r2": r2,
    }


def _mean_bias(frame: pd.DataFrame) -> float:
    y_true = frame["y_true"].astype(float).to_numpy()
    y_pred = frame["y_pred"].astype(float).to_numpy()
    return float(np.mean(y_true - y_pred))
