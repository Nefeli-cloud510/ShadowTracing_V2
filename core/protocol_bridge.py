from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from pydantic import Field

from core.experiment_tools.compiled_plan import CompiledExperimentPlan
from core.schema import ExperimentConfig as LegacyAdjustmentConfig
from core.unified_schema import (
    ExperimentProtocol,
    ModelSpec,
    RunSpec,
    ShadowBaseModel,
)
from models.elasticnet_runner import (
    ElasticNetConfig as RunnerElasticNetConfig,
    ExperimentConfig as RunnerExperimentConfig,
    FileConfig as RunnerFileConfig,
    TimeWindowConfig as RunnerTimeWindowConfig,
)


class ElasticNetDataSourceConfig(ShadowBaseModel):
    """Static data binding needed by the real ElasticNet runner."""

    omni_file_path: Path
    lhaaso_file_path: Path
    extra_file_paths: list[Path] = []
    sparse_sources: list[str] = Field(default_factory=list, description="需要缺失记录审计的源名称")
    time_column: str = "dataset_id"
    target_column: str = "SW Plasma Speed, km/s"
    column_aliases: dict[str, str] = {}
    test_split_ratio: float = 0.2
    random_state: int = 42


class ElasticNetProtocolBridge:
    """Convert unified experiment protocols into the runner's config schema."""

    def __init__(self, data_source: ElasticNetDataSourceConfig) -> None:
        self.data_source = data_source

    def validate_feature_columns(self, feature_columns: Iterable[str]) -> list[str]:
        feature_list = list(dict.fromkeys(feature_columns))
        if not feature_list:
            raise ValueError("feature_columns cannot be empty")

        available_columns = self._available_columns()
        valid_columns: list[str] = []
        for name in feature_list:
            resolved = self._resolve_column_name(name, available_columns)
            if resolved is not None and resolved not in valid_columns:
                valid_columns.append(resolved)
        if not valid_columns:
            missing = [name for name in feature_list if self._resolve_column_name(name, available_columns) is None]
            raise KeyError(f"特征列在数据源中不存在: {missing}")
        return valid_columns

    def _resolve_feature_lags(
        self,
        feature_lags: dict[str, list[int]] | None,
    ) -> dict[str, list[int]] | None:
        """Resolve sparse-lag keys through the data dictionary aliases.

        Protocol lags may name physical-meaning display labels such as
        ``BY, nT (GSE)`` while the merged frame stores the canonical field
        name. Resolve both forms to the actual frame column so the runner
        never silently falls back to a uniform lag window.
        """

        if not feature_lags:
            return None
        available_columns = self._available_columns()
        resolved: dict[str, list[int]] = {}
        for key, values in feature_lags.items():
            actual = self._resolve_column_name(str(key), available_columns)
            if actual is not None:
                resolved[actual] = [int(value) for value in values]
        if not resolved:
            raise ValueError(
                f"feature_lags 未解析到任何可用特征列: {list(feature_lags)}"
            )
        return resolved

    def build_runner_config(
        self,
        protocol: ExperimentProtocol,
        feature_columns: list[str],
        arm_overrides: dict | None = None,
    ) -> RunnerExperimentConfig:
        feature_columns = self.validate_feature_columns(feature_columns)
        model_parameters = protocol.model.parameters
        available_columns = self._available_columns()
        target_column = self._resolve_column_name(self.data_source.target_column or protocol.target, available_columns) or protocol.target
        overrides = arm_overrides or {}
        window_size = int(overrides.get("window_size", model_parameters.get("window_size", 3)))
        forecast_horizon_days = int(
            self._resolve_forecast_horizon_days(
                protocol=protocol,
                overrides=overrides,
                model_parameters=model_parameters,
            )
        )
        past_lag_days = overrides.get("past_lag_days", model_parameters.get("past_lag_days"))
        return self._assemble_runner_config(
            target_column=target_column,
            feature_columns=feature_columns,
            window_size=window_size,
            forecast_horizon_days=forecast_horizon_days,
            past_lag_days=past_lag_days,
            feature_lags=protocol.features.lags or None,
            test_split_ratio=float(
                model_parameters.get(
                    "test_split_ratio",
                    self.data_source.test_split_ratio,
                )
            ),
            model_family=self._resolve_model_family(protocol, model_parameters),
            model_parameters=model_parameters,
        )

    def build_runner_config_from_plan(
        self,
        compiled_plan: CompiledExperimentPlan,
        run_spec: RunSpec,
        feature_columns: list[str],
    ) -> RunnerExperimentConfig:
        """Build the runner config exclusively from the compiled plan.

        The executor must never re-derive time windows from the legacy
        protocol fields; this method rejects plans that lack the authoritative
        time parameters instead of falling back to hidden defaults.
        """

        feature_columns = self.validate_feature_columns(feature_columns)
        time_parameters = dict(compiled_plan.time_parameters or {})
        required_time_keys = (
            "forecast_horizon_days",
            "past_lag_days",
            "window_size",
            "test_split_ratio",
        )
        missing = [
            key
            for key in required_time_keys
            if time_parameters.get(key) is None
        ]
        if missing:
            raise ValueError(
                f"编译计划缺少权威时间参数: {missing}，"
                "执行器禁止从旧协议重新推导。"
            )

        model_parameters = dict(compiled_plan.model_parameters or {})
        model_parameters.update(dict(run_spec.parameters or {}))
        for legacy_key in ("feature_groups", "arm_time_windows", "model_families"):
            model_parameters.pop(legacy_key, None)
        # 编译计划中的共享时间参数是唯一权威值，即使 run 级覆盖残留同名
        # 字段也不能再次改变执行参数。
        model_parameters.update(time_parameters)

        available_columns = self._available_columns()
        target = compiled_plan.target or self.data_source.target_column
        target_column = (
            self._resolve_column_name(
                self.data_source.target_column or target,
                available_columns,
            )
            or target
        )
        return self._assemble_runner_config(
            target_column=target_column,
            feature_columns=feature_columns,
            window_size=int(time_parameters["window_size"]),
            forecast_horizon_days=int(time_parameters["forecast_horizon_days"]),
            past_lag_days=(
                int(time_parameters["past_lag_days"])
                if time_parameters["past_lag_days"] is not None
                else None
            ),
            feature_lags=(
                dict(compiled_plan.feature_lags or {})
                or time_parameters.get("feature_lags")
                or None
            ),
            test_split_ratio=float(time_parameters["test_split_ratio"]),
            model_family=self._normalize_model_family_value(
                run_spec.model_family
                or model_parameters.get("model_family")
            ),
            model_parameters=model_parameters,
        )

    def _assemble_runner_config(
        self,
        *,
        target_column: str,
        feature_columns: list[str],
        window_size: int,
        forecast_horizon_days: int,
        past_lag_days: int | None,
        feature_lags: dict[str, list[int]] | None,
        test_split_ratio: float,
        model_family: str,
        model_parameters: dict,
    ) -> RunnerExperimentConfig:
        return RunnerExperimentConfig(
            file_config=RunnerFileConfig(
                omni_file_path=self.data_source.omni_file_path,
                lhaaso_file_path=self.data_source.lhaaso_file_path,
                extra_file_paths=self.data_source.extra_file_paths,
                time_column=self.data_source.time_column,
                target_column=target_column,
                feature_columns=feature_columns,
                sparse_sources=self.data_source.sparse_sources,
            ),
            time_window_config=RunnerTimeWindowConfig(
                window_size=window_size,
                test_split_ratio=test_split_ratio,
                use_lag_feature=bool(model_parameters.get("use_lag_feature", False)),
                max_lag_day=int(model_parameters.get("max_lag_day", 0)),
                forecast_horizon_days=forecast_horizon_days,
                past_lag_days=(
                    int(past_lag_days)
                    if past_lag_days is not None
                    else None
                ),
                feature_lags=self._resolve_feature_lags(feature_lags),
            ),
            elasticnet_config=RunnerElasticNetConfig(
                model_family=model_family,
                alpha=float(model_parameters.get("alpha", 0.5)),
                l1_ratio=float(model_parameters.get("l1_ratio", 0.5)),
                random_state=int(
                    model_parameters.get(
                        "random_state",
                        self.data_source.random_state,
                    )
                ),
                alpha_grid=list(
                    model_parameters.get(
                        "alpha_grid",
                        [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                    )
                ),
                l1_ratio_grid=list(
                    model_parameters.get(
                        "l1_ratio_grid",
                        [0.1, 0.3, 0.5, 0.7, 0.9],
                    )
                ),
                param_search_mode=str(
                    model_parameters.get("param_search_mode", "auto")
                ),
                max_param_search_combinations=int(
                    model_parameters.get("max_param_search_combinations", 30)
                ),
                fit_intercept=bool(
                    model_parameters.get("fit_intercept", True)
                ),
                ridge_solver=str(
                    model_parameters.get("ridge_solver", "svd")
                ),
            ),
        )

    @staticmethod
    def _resolve_forecast_horizon_days(
        *,
        protocol: ExperimentProtocol,
        overrides: dict,
        model_parameters: dict,
    ) -> int:
        design = protocol.features
        candidates = [
            overrides.get("forecast_horizon_days"),
            model_parameters.get("forecast_horizon_days"),
            design.forecast_horizon_days,
            design.control_forecast_horizon_days,
            design.treatment_forecast_horizon_days,
            protocol.forecast_horizon_days,
        ]
        for value in candidates:
            if value is not None:
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed >= 1:
                    return parsed
        return 1

    @staticmethod
    def _resolve_model_family(protocol: ExperimentProtocol, model_parameters: dict) -> str:
        parameter_family = model_parameters.get("model_family")
        family = parameter_family or protocol.model.name or "ElasticNet"
        return ElasticNetProtocolBridge._normalize_model_family_value(family)

    @staticmethod
    def _normalize_model_family_value(family: str | None) -> str:
        normalized = str(family or "ElasticNet").strip().lower()
        if normalized not in {"ridge", "elasticnet", "enet"}:
            raise ValueError(f"未知模型族: {family}")
        if normalized == "enet":
            normalized = "elasticnet"
        return normalized

    def _available_columns(self) -> set[str]:
        paths = [self.data_source.omni_file_path, self.data_source.lhaaso_file_path, *self.data_source.extra_file_paths]
        available_columns: set[str] = set()
        for path in paths:
            available_columns.update(self._read_columns(path))
        return available_columns

    def _resolve_column_name(self, name: str | None, available_columns: set[str]) -> str | None:
        if not name:
            return None
        if name in available_columns:
            return name
        alias = self.data_source.column_aliases.get(name)
        if alias and alias in available_columns:
            return alias
        return None

    @staticmethod
    def _read_columns(path: Path) -> list[str]:
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path, nrows=0).columns.tolist()
        if suffix == ".csv":
            return pd.read_csv(path, nrows=0).columns.tolist()
        if suffix == ".txt":
            return pd.read_csv(path, sep=r"\s+", engine="python", nrows=0).columns.tolist()
        raise ValueError(f"暂不支持的数据格式: {path}")


def legacy_config_to_protocol(
    config: LegacyAdjustmentConfig,
    *,
    round_id: int,
    task_id: str,
    target: str,
    control_features: list[str],
    treatment_features: list[str],
    experiment_id: str = "LEGACY_E01",
) -> ExperimentProtocol:
    """Adapt the old minimal config into the unified protocol format."""

    model_parameters = config.model_dump(exclude={"reasoning"})
    model_parameters["random_state"] = 42
    model_parameters["test_split_ratio"] = 0.2

    model_parameters["forecast_horizon_days"] = 1
    return ExperimentProtocol(
        experiment_id=experiment_id,
        round_id=round_id,
        task_id=task_id,
        target=target,
        forecast_horizon_days=1,
        features={
            "target": target,
            "control": control_features,
            "treatment": treatment_features,
            "lags": {},
            "forecast_horizon_days": 1,
        },
        model=ModelSpec(
            name="ElasticNet",
            parameters=model_parameters,
        ),
        next_round_kind="recompute",
        notes=[config.reasoning] if config.reasoning else [],
    )
