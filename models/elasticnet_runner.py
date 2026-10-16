import json
import math
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from pydantic import BaseModel, Field, ValidationError
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.calendar_samples import (
    CalendarSample,
    CalendarSplit,
    SamplePlan,
    build_sample_plan,
    make_calendar_split,
)
from core.data_provenance import (
    DataProvenanceError,
    DataValidationReport,
    validate_merged_data,
)
from core.json_safe import json_dumps

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update(
    {
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
    }
)


# 输入配置校验

class FileConfig(BaseModel):
    omni_file_path: Path
    lhaaso_file_path: Path
    extra_file_paths: list[Path] = []
    time_column: str = "dataset_id"
    target_column: str
    feature_columns: list[str]
    sparse_sources: list[str] = Field(default_factory=list, description="需要缺失记录审计的源名称")

    def model_post_init(self, __context):
        if not self.omni_file_path.exists():
            raise FileNotFoundError(f"OMNI文件不存在 {self.omni_file_path}")
        if not self.lhaaso_file_path.exists():
            raise FileNotFoundError(f"LHAASO文件不存在 {self.lhaaso_file_path}")
        for path in self.extra_file_paths:
            if not path.exists():
                raise FileNotFoundError(f"额外数据文件不存在 {path}")


class TimeWindowConfig(BaseModel):
    window_size: int = Field(ge=1, le=196, description="历史窗口天数，1~196")
    test_split_ratio: float = Field(ge=0.1, le=0.5)
    use_lag_feature: bool = True
    max_lag_day: int = Field(default=0, ge=0, le=196)
    forecast_horizon_days: int = Field(default=1, ge=1, le=30, description="预测超前天数，>=1 的严格预测回测")
    past_lag_days: int | None = Field(default=None, ge=1, le=196, description="仅保留最近 N 天滞后特征")


class ElasticNetConfig(BaseModel):
    """线性正则化模型配置，Ridge 与 ElasticNet 共用。"""

    model_family: Literal["ridge", "elasticnet"] = "ridge"
    alpha: float = 1.0
    l1_ratio: float = 0.5
    random_state: int = 42
    alpha_grid: list[float] = Field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    )
    l1_ratio_grid: list[float] = Field(
        default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9]
    )
    param_search_mode: Literal["auto", "fixed"] = Field(
        default="auto",
        description="auto=训练内网格选参；fixed=固定使用 alpha/l1_ratio 作为对照",
    )
    max_param_search_combinations: int = Field(default=30, ge=1, le=200)
    fit_intercept: bool = True
    ridge_solver: str = "svd"


# 兼容旧调用：Ridge 升级后仍然允许旧代码以 ElasticNetConfig 名义使用。
ModelConfig = ElasticNetConfig


class ExperimentConfig(BaseModel):
    file_config: FileConfig
    time_window_config: TimeWindowConfig
    elasticnet_config: ElasticNetConfig


@dataclass
class DataCoverageReport:
    """单个数据源的日历覆盖统计。"""

    source: str
    expected_days: int = 0
    observed_days: int = 0
    missing_days: int = 0
    coverage_ratio: float = 1.0
    interpolated_days: int = 0
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "expected_days": self.expected_days,
            "observed_days": self.observed_days,
            "missing_days": self.missing_days,
            "coverage_ratio": self.coverage_ratio,
            "interpolated_days": self.interpolated_days,
            "note": self.note,
        }


@dataclass
class MergedLoadResult:
    """数据合并结果，附带来源覆盖与稀疏源识别信息。"""

    X: pd.DataFrame
    y: pd.Series
    time: pd.Series
    coverage: list[DataCoverageReport]
    active_sparse_sources: list[str]
    merged_sources: list[str]
    missing_counts: dict[str, int] = field(default_factory=dict)
    data_validation: DataValidationReport | None = None


@dataclass
class SlidingWindowResult:
    """Calendar-aligned sliding window result.

    History and forecast horizon are calendar differences computed by
    ``core.calendar_samples``; a horizon is never a record-count offset.
    """

    X_all: np.ndarray
    y_all: np.ndarray
    time_all: np.ndarray
    feature_names: list[str]
    dropped_gap_windows: int = 0
    total_windows: int = 0
    origins_all: np.ndarray | None = None
    sample_ids_all: np.ndarray | None = None
    records_all: list[CalendarSample] | None = None
    sample_plan: SamplePlan | None = None
    window_mode: str = "calendar"
    retained_before_filter: int = 0
    common_sample_excluded: int = 0


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    time_train: np.ndarray
    time_test: np.ndarray
    origin_train: np.ndarray | None = None
    origin_test: np.ndarray | None = None
    sample_ids_train: list[str] = field(default_factory=list)
    sample_ids_test: list[str] = field(default_factory=list)
    cv_folds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    split_date: pd.Timestamp | None = None
    test_start: pd.Timestamp | None = None
    block_boundaries: list[pd.Timestamp] = field(default_factory=list)
    boundary_excluded_count: int = 0
    train_mask: np.ndarray | None = None


def load_llm_json_config(json_path: str | Path) -> ExperimentConfig:
    try:
        json_path = Path(json_path)
        raw_config = json_path.read_text(encoding="utf-8")
        config_dict = json.loads(raw_config)
        return ExperimentConfig.model_validate(config_dict)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"配置文件不存在: {json_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件不是合法JSON: {e}") from e
    except ValidationError as e:
        raise ValueError(f"LLM输出配置校验失败: {e}") from e


# 数据读取与合并

def _read_table(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".txt":
        return pd.read_csv(file_path, sep=r"\s+", engine="python")
    raise ValueError(f"暂不支持的文件格式: {file_path}")


def _normalize_time_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        normalized = pd.to_datetime(series, errors="coerce")
    else:
        cleaned = series.astype(str).str.replace(r"_x000d_", "", regex=True).str.strip()
        exact_dates = cleaned.str.fullmatch(r"\d{8}", na=False)
        yyyymmdd = cleaned.where(exact_dates, other=pd.NA).str.extract(
            r"(\d{8})", expand=False
        )
        normalized = pd.to_datetime(yyyymmdd, format="%Y%m%d", errors="coerce")
        remaining = cleaned[yyyymmdd.isna()]
        if not remaining.empty:
            parsed_rest = pd.to_datetime(remaining, errors="coerce")
            normalized.loc[remaining.index] = parsed_rest

    if normalized.isna().any():
        bad_values = series[normalized.isna()].head(5).tolist()
        raise ValueError(f"TIME列存在无法解析的日期值: {bad_values}")
    return normalized


def load_and_merge_data(cfg: ExperimentConfig) -> MergedLoadResult:
    file_cfg = cfg.file_config
    unique_sources: list[tuple[str, Path]] = []
    seen_paths: set[str] = set()
    for path in [
        file_cfg.omni_file_path,
        file_cfg.lhaaso_file_path,
        *file_cfg.extra_file_paths,
    ]:
        resolved = str(path.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        unique_sources.append((path.stem, path))
    data_frames = [(label, _read_table(path)) for label, path in unique_sources]

    normalized_frames: list[pd.DataFrame] = []
    for label, frame in data_frames:
        if file_cfg.time_column not in frame.columns:
            raise KeyError(f"{label}数据缺少时间列: {file_cfg.time_column}")
        next_frame = frame.copy()
        next_frame[file_cfg.time_column] = _normalize_time_column(next_frame[file_cfg.time_column])
        normalized_frames.append(next_frame)

    source_index = {label: idx for idx, (label, _) in enumerate(data_frames)}
    source_columns = {
        label: set(normalized_frames[idx].columns)
        for label, idx in source_index.items()
    }

    # 只合并当前实验臂实际使用到的源：目标列来源 + 特征列来源。
    required_labels: set[str] = set()
    target_sources = [
        label for label, idx in source_index.items()
        if file_cfg.target_column in source_columns[label]
    ]
    if not target_sources:
        raise KeyError(f"没有任何数据源包含目标列: {file_cfg.target_column}")
    required_labels.update(target_sources)
    for col in file_cfg.feature_columns:
        matches = [label for label, idx in source_index.items() if col in source_columns[label]]
        required_labels.update(matches)

    ordered_sources = [
        label for label, _ in unique_sources if label in required_labels
    ]
    if not ordered_sources:
        raise KeyError("没有可用的数据源")

    # 期望日历范围只按本轮实际参与合并的源计算。
    ordered_dates: set[pd.Timestamp] = set()
    for label in ordered_sources:
        frame = normalized_frames[source_index[label]]
        ordered_dates.update(
            pd.to_datetime(frame[file_cfg.time_column]).dt.normalize().unique()
        )
    if not ordered_dates:
        raise ValueError("本轮使用的数据源均为空，无法确定日历范围")
    expected_days = int((max(ordered_dates) - min(ordered_dates)).days + 1)

    coverage_reports: list[DataCoverageReport] = []
    for label in ordered_sources:
        frame = normalized_frames[source_index[label]]
        observed_dates = pd.to_datetime(frame[file_cfg.time_column]).dt.normalize().unique()
        observed_days = int(len(observed_dates))
        missing_days = max(0, expected_days - observed_days)
        coverage_ratio = observed_days / expected_days if expected_days else 1.0
        coverage_reports.append(
            DataCoverageReport(
                source=label,
                expected_days=expected_days,
                observed_days=observed_days,
                missing_days=missing_days,
                coverage_ratio=round(coverage_ratio, 6),
                note="缺失记录保留至日历样本层判定，不提前删除整行" if missing_days else None,
            )
        )

    configured_sparse = set(file_cfg.sparse_sources)
    active_sparse_sources = (
        [
            report.source
            for report in coverage_reports
            if report.missing_days > 0 and report.source in configured_sparse
        ]
        if configured_sparse
        else []
    )

    df_merge = normalized_frames[source_index[ordered_sources[0]]]
    for label in ordered_sources[1:]:
        df_merge = pd.merge(
            df_merge,
            normalized_frames[source_index[label]],
            on=file_cfg.time_column,
            how="outer",
        )

    missing_features = [col for col in file_cfg.feature_columns if col not in df_merge.columns]
    if missing_features:
        raise KeyError(f"合并后的数据缺少特征列: {missing_features}")
    if file_cfg.target_column not in df_merge.columns:
        raise KeyError(f"合并后的数据缺少目标列: {file_cfg.target_column}")

    validation_report = validate_merged_data(
        df_merge,
        source_labels=ordered_sources,
        time_column=file_cfg.time_column,
    )

    selected_columns = file_cfg.feature_columns + [file_cfg.target_column]
    missing_counts = {
        str(column): int(count)
        for column, count in df_merge[selected_columns].isna().sum().items()
    }
    df_merge = (
        df_merge
        .sort_values(file_cfg.time_column)
        .reset_index(drop=True)
    )

    X = df_merge[file_cfg.feature_columns].astype(float).copy()
    y = df_merge[file_cfg.target_column].astype(float).copy()
    time_raw = df_merge[file_cfg.time_column].copy()
    print(
        f"数据合并完成，日历行数: {len(df_merge)}；"
        f"缺失记录审计源: {active_sparse_sources or '无'}"
    )
    return MergedLoadResult(
        X=X,
        y=y,
        time=time_raw,
        coverage=coverage_reports,
        active_sparse_sources=active_sparse_sources,
        merged_sources=ordered_sources,
        missing_counts=missing_counts,
        data_validation=validation_report,
    )


# 时间窗构造与数据集切分

def _sample_key(record: CalendarSample) -> tuple[str, str, int]:
    return (
        record.issue_date.date().isoformat(),
        record.target_date.date().isoformat(),
        int(record.horizon_days),
    )


def build_sliding_window(
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    time_raw: pd.Series,
    tw_cfg: TimeWindowConfig,
    only_keys: set[tuple[str, str, int]] | None = None,
    feature_columns: list[str] | None = None,
) -> SlidingWindowResult:
    """Build calendar-aligned samples on the real calendar axis.

    ``only_keys`` accepts the same compact keys used by
    ``core.calendar_samples.CalendarSample`` so both arms can share one frozen
    common sample set.
    """

    past_lag_days = tw_cfg.past_lag_days or tw_cfg.window_size
    forecast_horizon = tw_cfg.forecast_horizon_days
    columns = list(
        dict.fromkeys(
            str(item)
            for item in (feature_columns if feature_columns is not None else X_raw.columns)
        )
    )
    plan = build_sample_plan(
        X_raw,
        y_raw,
        time_raw,
        feature_columns=columns,
        target_name=y_raw.name or "target",
        past_lag_days=past_lag_days,
        forecast_horizon_days=forecast_horizon,
    )
    records = plan.retained_records()
    retained_before_filter = len(records)
    if only_keys is not None:
        records = [record for record in records if _sample_key(record) in only_keys]
    if not records:
        detail = (
            f"calendar plan retained 0 samples "
            f"(candidate={plan.candidate_count}, dropped={plan.dropped_count})"
        )
        raise ValueError(detail)

    frame = X_raw.copy()
    frame["_hp_time"] = pd.to_datetime(
        pd.Series(time_raw, index=X_raw.index), errors="coerce"
    ).dt.normalize()
    frame["_hp_y"] = pd.to_numeric(y_raw, errors="coerce")
    frame = frame.sort_values("_hp_time", kind="mergesort").reset_index(drop=True)
    arr_x = frame[columns].to_numpy(dtype=float)
    arr_y = frame["_hp_y"].to_numpy(dtype=float)
    date_to_index = {
        pd.Timestamp(value).normalize(): index
        for index, value in enumerate(frame["_hp_time"])
    }

    X_all, y_all, time_all, origins_all, sample_ids_all = [], [], [], [], []
    for record in records:
        values: list[float] = []
        for spec in record.feature_specs:
            feature_date = record.issue_date + pd.Timedelta(days=int(spec.offset_days))
            row_index = date_to_index[pd.Timestamp(feature_date).normalize()]
            values.append(float(arr_x[row_index, columns.index(spec.variable)]))
        target_index = date_to_index[pd.Timestamp(record.target_date).normalize()]
        X_all.append(values)
        y_all.append(float(arr_y[target_index]))
        time_all.append(record.target_date.to_datetime64())
        origins_all.append(record.issue_date.to_datetime64())
        sample_ids_all.append(record.sample_id)

    return SlidingWindowResult(
        X_all=np.asarray(X_all, dtype=float),
        y_all=np.asarray(y_all, dtype=float),
        time_all=np.asarray(time_all),
        feature_names=[spec.feature_name() for spec in plan.feature_specs],
        dropped_gap_windows=max(0, plan.candidate_count - retained_before_filter),
        total_windows=plan.candidate_count,
        origins_all=np.asarray(origins_all),
        sample_ids_all=np.asarray(sample_ids_all),
        records_all=records,
        sample_plan=plan,
        window_mode="calendar",
        retained_before_filter=retained_before_filter,
        common_sample_excluded=max(0, retained_before_filter - len(records)),
    )


class _IndexFoldCV:
    """GridSearchCV-compatible wrapper for frozen calendar folds."""

    def __init__(self, folds: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.folds = folds

    def split(self, X, y=None, groups=None):
        return iter(self.folds)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self.folds)


def _split_by_target_date(
    time_all: np.ndarray,
    test_rate: float,
) -> tuple[np.ndarray, pd.Timestamp]:
    """Legacy-compatible target-date split; both arms share the boundary."""

    dates = pd.Series(pd.to_datetime(pd.Series(time_all), errors="coerce")).dt.normalize()
    if dates.isna().any():
        raise ValueError("split 输入存在无法解析的目标日期")
    unique_dates = np.unique(dates.to_numpy()).tolist()
    if len(unique_dates) < 2:
        raise ValueError("训练/测试按日历切分时至少需要两个不同的目标日期")
    n_train = max(
        1,
        min(
            len(unique_dates) - 1,
            int(round(len(unique_dates) * (1.0 - test_rate))),
        ),
    )
    split_date = pd.Timestamp(unique_dates[n_train - 1]).normalize()
    train_mask = np.asarray(dates.dt.normalize() <= split_date, dtype=bool)
    return train_mask, split_date


def split_dataset_by_calendar(
    X_all: np.ndarray,
    y_all: np.ndarray,
    time_all: np.ndarray,
    origins_all: np.ndarray | None,
    sample_ids_all: np.ndarray | list[str],
    records: list[CalendarSample],
    test_rate: float,
    feature_names: list[str],
    n_internal_folds: int = 2,
    calendar_split: CalendarSplit | None = None,
) -> DatasetSplit:
    """Freeze one common calendar split shared by both experiment arms."""

    if len(X_all) != len(records):
        raise ValueError("X_all 与 records 长度不一致，无法按共同日历样本划分")
    if calendar_split is None:
        calendar_split = make_calendar_split(
            records,
            test_ratio=test_rate,
            n_internal_folds=n_internal_folds,
        )
    train_ids = {record.sample_id for record in calendar_split.train}
    test_ids = {record.sample_id for record in calendar_split.test}
    boundary_ids = {
        record.sample_id for record in calendar_split.boundary_excluded
    }
    sample_ids = list(sample_ids_all)
    excluded_in_split = np.asarray(
        [item in boundary_ids for item in sample_ids], dtype=bool
    )
    if excluded_in_split.any():
        keep_mask = ~excluded_in_split
        sample_ids = list(np.asarray(sample_ids)[keep_mask])
        X_arr = np.asarray(X_all, dtype=float)[keep_mask]
        y_arr = np.asarray(y_all, dtype=float)[keep_mask]
        time_arr = np.asarray(time_all)[keep_mask]
        origins_arr = (
            np.asarray(origins_all)[keep_mask]
            if origins_all is not None
            else None
        )
    else:
        X_arr = np.asarray(X_all, dtype=float)
        y_arr = np.asarray(y_all, dtype=float)
        time_arr = np.asarray(time_all)
        origins_arr = np.asarray(origins_all) if origins_all is not None else None

    train_mask = np.asarray(
        [item in train_ids for item in sample_ids], dtype=bool
    )
    test_mask = np.asarray(
        [item in test_ids for item in sample_ids], dtype=bool
    )
    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            "日历切分后的训练或测试样本为空，无法继续训练"
        )

    X_train = pd.DataFrame(X_arr[train_mask], columns=feature_names)
    X_test = pd.DataFrame(X_arr[test_mask], columns=feature_names)
    train_id_to_index = {
        sample_id: index
        for index, sample_id in enumerate(np.asarray(sample_ids)[train_mask])
    }
    cv_folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_train, fold_eval in calendar_split.cv_folds:
        train_index = np.asarray(
            [train_id_to_index[record.sample_id] for record in fold_train],
            dtype=int,
        )
        eval_index = np.asarray(
            [train_id_to_index[record.sample_id] for record in fold_eval],
            dtype=int,
        )
        if len(train_index) and len(eval_index):
            cv_folds.append((train_index, eval_index))
    if not cv_folds:
        raise ValueError("日历内部折为空，无法进行训练内选参")

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_arr[train_mask],
        y_test=y_arr[test_mask],
        time_train=time_arr[train_mask],
        time_test=time_arr[test_mask],
        origin_train=(
            origins_arr[train_mask]
            if origins_arr is not None
            else None
        ),
        origin_test=(
            origins_arr[test_mask]
            if origins_arr is not None
            else None
        ),
        sample_ids_train=list(np.asarray(sample_ids)[train_mask]),
        sample_ids_test=list(np.asarray(sample_ids)[test_mask]),
        cv_folds=cv_folds,
        split_date=calendar_split.test_start,
        test_start=calendar_split.test_start,
        block_boundaries=calendar_split.block_boundaries,
        boundary_excluded_count=len(calendar_split.boundary_excluded),
        train_mask=train_mask,
    )


def split_dataset_by_date(
    X_all: np.ndarray,
    y_all: np.ndarray,
    time_all: np.ndarray,
    test_rate: float,
    feature_names: list[str],
    origins_all: np.ndarray | None = None,
) -> DatasetSplit:
    """Split by target calendar date; kept for legacy callers."""

    train_mask, split_date = _split_by_target_date(time_all, test_rate)
    X_arr = np.asarray(X_all, dtype=float)
    y_arr = np.asarray(y_all, dtype=float)
    origins = np.asarray(origins_all) if origins_all is not None else None
    return DatasetSplit(
        X_train=pd.DataFrame(X_arr[train_mask], columns=feature_names),
        X_test=pd.DataFrame(X_arr[~train_mask], columns=feature_names),
        y_train=y_arr[train_mask],
        y_test=y_arr[~train_mask],
        time_train=np.asarray(time_all)[train_mask],
        time_test=np.asarray(time_all)[~train_mask],
        origin_train=origins[train_mask] if origins is not None else None,
        origin_test=origins[~train_mask] if origins is not None else None,
        split_date=split_date,
        train_mask=train_mask,
    )


def split_and_scale_dataset(
    X_all: np.ndarray,
    y_all: np.ndarray,
    time_all: np.ndarray,
    test_rate: float,
    feature_names: list[str],
    origins_all: np.ndarray | None = None,
):
    split = split_dataset_by_date(
        X_all,
        y_all,
        time_all,
        test_rate,
        feature_names,
        origins_all=origins_all,
    )
    # Pipeline 内部完成标准化；这里保留旧返回值形态。
    scaler = StandardScaler().fit(split.X_train)
    return (
        split.X_train,
        split.X_test,
        split.y_train,
        split.y_test,
        split.time_train,
        split.time_test,
        scaler,
    )


# 模型训练与评估

DEFAULT_MODEL_FAMILY: Literal["ridge", "elasticnet"] = "ridge"
MODEL_DISPLAY_NAMES = {"ridge": "Ridge", "elasticnet": "ElasticNet"}


@dataclass
class ParamSearchInfo:
    """结构化训练内选参记录，供 cv_results 与诊断审计使用。"""

    model_family: str
    search_mode: str
    selected_alpha: float
    selected_l1_ratio: float | None
    converged: bool
    search_completed: bool
    boundary_hit: bool
    solver_converged: bool | None
    grid_size: int
    rows: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_family": self.model_family,
            "search_mode": self.search_mode,
            "selected_alpha": self.selected_alpha,
            "selected_l1_ratio": self.selected_l1_ratio,
            "converged": self.converged,
            "search_completed": self.search_completed,
            "boundary_hit": self.boundary_hit,
            "solver_converged": self.solver_converged,
            "boundary_note": self.boundary_note(),
            "grid_size": self.grid_size,
            "n_rows": len(self.rows),
        }

    def boundary_note(self) -> str:
        """分别说明搜索完整、边界命中与求解器收敛三类状态。"""
        if self.search_mode != "auto" or self.grid_size <= 1:
            return "固定参数对照，不涉及网格搜索"
        notes = []
        if not self.search_completed:
            notes.append("受 max_param_search_combinations 限制，未枚举完整网格")
        if self.boundary_hit:
            notes.append("最优参数落在网格边缘，需扩网格或改用其他选参方式")
        if self.solver_converged is None:
            notes.append("求解器为解析/封闭解，无迭代收敛状态")
        elif not self.solver_converged:
            notes.append("求解器在 max_iter 处停止，未满足收敛条件")
        return "；".join(notes) if notes else "网格枚举完整，未命中网格边缘，求解器收敛"


def resolve_model_family(cfg: ElasticNetConfig) -> Literal["ridge", "elasticnet"]:
    family = getattr(cfg, "model_family", None) or DEFAULT_MODEL_FAMILY
    normalized = str(family).strip().lower()
    if normalized not in {"ridge", "elasticnet", "enet"}:
        raise ValueError(f"未知模型族: {family}")
    if normalized == "enet":
        normalized = "elasticnet"
    return normalized


def make_model(
    cfg: ElasticNetConfig,
    alpha_override: float | None = None,
    l1_ratio_override: float | None = None,
) -> BaseEstimator:
    family = resolve_model_family(cfg)
    alpha = cfg.alpha if alpha_override is None else alpha_override
    l1_ratio = cfg.l1_ratio if l1_ratio_override is None else l1_ratio_override
    if family == "ridge":
        return Ridge(
            alpha=alpha,
            fit_intercept=cfg.fit_intercept,
            solver=cfg.ridge_solver,
            random_state=cfg.random_state,
        )
    return ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=cfg.random_state,
        max_iter=10000,
        fit_intercept=cfg.fit_intercept,
    )


def resolve_alpha_for_model(model: BaseEstimator, default: float) -> float:
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
    return float(getattr(estimator, "alpha", default))


def select_best_alpha(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    cfg: ElasticNetConfig,
    cv_folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> float:
    """向后兼容包装：返回训练内选出的 alpha。"""

    return select_best_params(Xtr, ytr, cfg, cv_folds=cv_folds).selected_alpha


def select_best_params(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    cfg: ElasticNetConfig,
    cv_folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> ParamSearchInfo:
    """在冻结的日历折上为 Ridge/ElasticNet 做对等的训练内选参。

    scaler 始终位于 Pipeline 内部，每折只在训练部分拟合。ElasticNet 与原
    Ridge 使用完全相同的折集合与评分指标，避免“模型族差异”混入“选参策略
    差异”。``param_search_mode="fixed"`` 时不做搜索，固定参数直接训练，
    用于单独的参数对照敏感性实验。
    """

    family = resolve_model_family(cfg)
    alpha_grid = [
        float(value)
        for value in getattr(cfg, "alpha_grid", [])
        if float(value) > 0
    ]
    if family == "ridge":
        l1_ratios: list[float] = []
    else:
        l1_ratios = [
            float(value)
            for value in getattr(cfg, "l1_ratio_grid", [])
            if 0.0 < float(value) < 1.0
        ]

    fixed_alpha = float(cfg.alpha)
    fixed_l1_ratio = (
        (float(cfg.l1_ratio) if 0.0 < float(cfg.l1_ratio) < 1.0 else 0.5)
        if family == "elasticnet"
        else None
    )
    full_l1_ratios = list(l1_ratios)
    search_mode = str(getattr(cfg, "param_search_mode", "auto") or "auto").lower()
    can_search = (
        search_mode != "fixed"
        and len(alpha_grid) >= 2
        and len(Xtr) >= 4
        and (family == "ridge" or len(l1_ratios) >= 1)
    )
    if cv_folds is not None and not cv_folds:
        can_search = False
    if not can_search:
        return ParamSearchInfo(
            model_family=family,
            search_mode="fixed",
            selected_alpha=fixed_alpha,
            selected_l1_ratio=fixed_l1_ratio,
            converged=True,
            search_completed=False,
            boundary_hit=False,
            solver_converged=True,
            grid_size=1,
        )

    param_grid: dict[str, list[float]] = {"model__alpha": alpha_grid}
    if family == "elasticnet":
        param_grid["model__l1_ratio"] = l1_ratios
    combination_count = 1
    for values in param_grid.values():
        combination_count *= len(values)
    max_combinations = int(getattr(cfg, "max_param_search_combinations", 30) or 30)
    effective_alpha_grid = list(alpha_grid)
    if combination_count > max_combinations:
        limit = max(max_combinations // max(len(l1_ratios), 1), 1)
        effective_alpha_grid = alpha_grid[:limit]
        param_grid["model__alpha"] = effective_alpha_grid

    if cv_folds is not None:
        cv = _IndexFoldCV(cv_folds)
    else:
        n_splits = min(2, len(Xtr) // 2)
        if n_splits < 2:
            return ParamSearchInfo(
                model_family=family,
                search_mode="fixed",
                selected_alpha=fixed_alpha,
                selected_l1_ratio=fixed_l1_ratio,
                converged=True,
                search_completed=False,
                boundary_hit=False,
                solver_converged=True,
                grid_size=1,
            )
        cv = TimeSeriesSplit(n_splits=n_splits)

    model = make_model(cfg)
    search = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("model", model)]),
        param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        refit=True,
        error_score="raise",
        n_jobs=1,
    )
    search.fit(Xtr, ytr)
    selected_alpha = float(search.best_params_["model__alpha"])
    selected_l1_ratio = (
        float(search.best_params_["model__l1_ratio"])
        if family == "elasticnet"
        else None
    )
    estimator = search.best_estimator_.named_steps["model"]
    solver_converged: bool | None = True
    if hasattr(estimator, "n_iter_") and hasattr(estimator, "max_iter"):
        raw_n_iter = getattr(estimator, "n_iter_", None)
        if raw_n_iter is None:
            solver_converged = None
        else:
            solver_converged = int(raw_n_iter) < int(
                getattr(estimator, "max_iter") or 0
            )
    effective_l1_ratios = (
        [float(value) for value in search.cv_results_["param_model__l1_ratio"]]
        if family == "elasticnet"
        else []
    )
    boundary_hit = bool(
        selected_alpha <= min(effective_alpha_grid)
        or selected_alpha >= max(effective_alpha_grid)
        or (
            effective_l1_ratios
            and (
                selected_l1_ratio <= min(effective_l1_ratios)
                or selected_l1_ratio >= max(effective_l1_ratios)
            )
        )
    )
    full_combination_count = len(alpha_grid) * max(len(full_l1_ratios), 1)
    search_completed = bool(
        combination_count <= max_combinations
        and len(effective_alpha_grid) == len(alpha_grid)
    )
    rows = _grid_rows_from_search(
        search,
        family=family,
        alpha_grid=effective_alpha_grid,
        l1_ratios=l1_ratios,
        feature_columns=list(Xtr.columns),
        Xtr=Xtr,
        ytr=ytr,
    )
    return ParamSearchInfo(
        model_family=family,
        search_mode="auto",
        selected_alpha=selected_alpha,
        selected_l1_ratio=selected_l1_ratio,
        converged=True if solver_converged is None else solver_converged,
        search_completed=search_completed,
        boundary_hit=boundary_hit,
        solver_converged=solver_converged,
        grid_size=len(rows),
        rows=rows,
    )


def _grid_rows_from_search(
    search: GridSearchCV,
    *,
    family: str,
    alpha_grid: list[float],
    l1_ratios: list[float],
    feature_columns: list[str],
    Xtr: pd.DataFrame | None = None,
    ytr: np.ndarray | None = None,
) -> list[dict]:
    """把 GridSearchCV 的 cv_results_ 展开为按原始网格序排列的逐行记录。

    sklearn 的 cv_results_ 按 rank 排序；这里按参数网格索引恢复枚举顺序，
    保证组合顺序可复现、可审计。
    """
    cv_results = getattr(search, "cv_results_", None)
    if cv_results is None:
        return []
    alphas_raw = np.asarray(
        [float(value) for value in cv_results["param_model__alpha"]]
    )
    l1_ratios_raw = (
        np.asarray(
            [float(value) for value in cv_results["param_model__l1_ratio"]]
        )
        if family == "elasticnet"
        else None
    )
    alpha_to_index = {
        float(value): index for index, value in enumerate(alpha_grid)
    }
    l1_to_index = (
        {float(value): index for index, value in enumerate(l1_ratios)}
        if l1_ratios_raw is not None
        else None
    )
    order = sorted(
        range(len(alphas_raw)),
        key=lambda index: (
            alpha_to_index.get(float(alphas_raw[index]), index),
            (
                l1_to_index.get(float(l1_ratios_raw[index]), index)
                if l1_to_index is not None
                else 0
            ),
        ),
    )
    alphas = alphas_raw[order]
    l1_ratios = l1_ratios_raw[order] if l1_ratios_raw is not None else None
    ranks = np.asarray(cv_results["rank_test_score"])[order]
    mean_neg_mse = np.asarray(cv_results["mean_test_score"], dtype=float)[order]
    std_scores = np.asarray(cv_results["std_test_score"], dtype=float)[order]
    split_scores = [
        np.asarray(cv_results[f"split{split}_test_score"], dtype=float)[order]
        for split in range(search.n_splits_)
    ]
    train_target = np.asarray(ytr, dtype=float) if ytr is not None else None
    best_train_pred = (
        np.asarray(search.best_estimator_.predict(Xtr), dtype=float)
        if Xtr is not None and len(Xtr)
        else None
    )
    if best_train_pred is not None and train_target is not None:
        train_error = best_train_pred - train_target
        train_rmse_best = float(np.sqrt(np.mean(np.square(train_error))))
        train_mae_best = float(np.mean(np.abs(train_error)))
        pred_std_best = float(np.std(best_train_pred))
    else:
        train_rmse_best = None
        train_mae_best = None
        pred_std_best = None
    try:
        best_pipeline_coef = np.asarray(
            search.best_estimator_.named_steps["model"].coef_,
            dtype=float,
        ).ravel()
    except (AttributeError, TypeError, ValueError):
        best_pipeline_coef = None
    best_coef: list[float] | None = None
    if best_pipeline_coef is not None and len(best_pipeline_coef) == len(feature_columns):
        best_coef = [float(value) for value in best_pipeline_coef]
    rows: list[dict] = []
    for index in range(len(order)):
        row: dict = {
            "rank": int(ranks[index]),
            "param_alpha": float(alphas[index]),
            "param_l1_ratio": (
                None if l1_ratios is None else float(l1_ratios[index])
            ),
            "mean_test_neg_mse": float(mean_neg_mse[index]),
            "mean_test_rmse": float(
                np.sqrt(max(-float(mean_neg_mse[index]), 0.0))
            ),
            "std_test_score": float(std_scores[index]),
            "train_rmse_best_params": train_rmse_best,
            "train_mae_best_params": train_mae_best,
            "prediction_std_best_params": pred_std_best,
            "refit_train_rmse": train_rmse_best,
            "refit_train_mae": train_mae_best,
            "refit_prediction_std": pred_std_best,
        }
        for split, scores in enumerate(split_scores):
            row[f"split{split}_test_neg_mse"] = float(scores[index])
            row[f"split{split}_train_rmse_best_params"] = train_rmse_best
        if best_coef is not None:
            for column_index, feature_name in enumerate(feature_columns):
                row[f"refit_coef_{feature_name}"] = best_coef[column_index]
        rows.append(row)
    return rows


def train_elasticnet_model_with_search(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    enet_cfg: ElasticNetConfig,
    cv_folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[BaseEstimator, ParamSearchInfo]:
    search = select_best_params(Xtr, ytr, enet_cfg, cv_folds=cv_folds)
    model = make_model(
        enet_cfg,
        alpha_override=search.selected_alpha,
        l1_ratio_override=search.selected_l1_ratio,
    )
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipeline.fit(Xtr, ytr)
    return pipeline, search


def train_elasticnet_model(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    enet_cfg: ElasticNetConfig,
    cv_folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> BaseEstimator:
    model, _ = train_elasticnet_model_with_search(
        Xtr,
        ytr,
        enet_cfg,
        cv_folds=cv_folds,
    )
    return model


# 旧入口的兼容别名；新代码建议使用 train_elasticnet_model。
train_model = train_elasticnet_model


def evaluate_model(
    model: BaseEstimator,
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    ytr: np.ndarray,
    yte: np.ndarray,
):
    ytr_pred = model.predict(Xtr)
    yte_pred = model.predict(Xte)
    baseline_pred = np.full_like(yte, ytr.mean(), dtype=float)
    if len(yte) > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            pearson_raw = float(np.corrcoef(yte, yte_pred)[0, 1])
        pearson_r = pearson_raw if np.isfinite(pearson_raw) else None
    else:
        pearson_r = None
    unavailable_reasons: dict[str, str] = {}
    if pearson_r is None:
        if len(yte) <= 1:
            unavailable_reasons["pearson_r"] = "insufficient_samples"
        elif float(np.std(yte)) <= 1e-12:
            unavailable_reasons["pearson_r"] = "constant_target"
        elif float(np.std(yte_pred)) <= 1e-12:
            unavailable_reasons["pearson_r"] = "constant_prediction"
        else:
            unavailable_reasons["pearson_r"] = "undefined"

    metrics = {
        "train_rmse": float(np.sqrt(mean_squared_error(ytr, ytr_pred))),
        "test_rmse": float(np.sqrt(mean_squared_error(yte, yte_pred))),
        "train_mae": float(mean_absolute_error(ytr, ytr_pred)),
        "test_mae": float(mean_absolute_error(yte, yte_pred)),
        "train_r2": float(r2_score(ytr, ytr_pred)),
        "test_r2": float(r2_score(yte, yte_pred)),
        "test_pearson_r": pearson_r,
        "unavailable_reasons": unavailable_reasons,
        "baseline_test_rmse": float(np.sqrt(mean_squared_error(yte, baseline_pred))),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
    }
    return metrics, ytr_pred, yte_pred


def make_coefficient_table(model: BaseEstimator, feat_names: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
    coef_df = pd.DataFrame(
        {
            "feature": feat_names,
            "coefficient": estimator.coef_,
        }
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    return coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


# 图表与残差分析

def _format_metric_text(value, digits: int) -> str:
    """Format a metric for plot or CLI text, showing N/A when undefined."""

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return "N/A"
    if not math.isfinite(numeric):
        return "N/A"
    return f"{numeric:.{int(digits)}f}"


def plot_feature_coefficients(
    coef_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 15,
    *,
    model_family: str = DEFAULT_MODEL_FAMILY,
):
    plot_df = coef_df.head(top_n).sort_values("coefficient")
    plt.figure(figsize=(10, 6), dpi=300)
    colors = ["#c0392b" if v < 0 else "#2980b9" for v in plot_df["coefficient"]]
    plt.barh(plot_df["feature"], plot_df["coefficient"], color=colors)
    plt.xlabel("系数")
    plt.ylabel("特征")
    display = MODEL_DISPLAY_NAMES.get(model_family, model_family)
    plt.title(f"{display} 特征系数（Top {len(plot_df)}）")
    plt.tight_layout()
    prefix = "elasticnet" if model_family == "elasticnet" else "ridge"
    save_path = output_dir / f"{prefix}_coefficients.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
    output_dir: Path,
    *,
    experiment_label: str = "",
    train_true: np.ndarray | None = None,
    train_pred: np.ndarray | None = None,
    model_family: str = DEFAULT_MODEL_FAMILY,
):
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    if train_true is not None and train_pred is not None and len(train_true) > 0:
        min_val = min(min_val, float(np.min(train_true)), float(np.min(train_pred)))
        max_val = max(max_val, float(np.max(train_true)), float(np.max(train_pred)))
    plt.figure(figsize=(7, 6), dpi=300)
    if train_true is not None and train_pred is not None and len(train_true) > 0:
        plt.scatter(train_true, train_pred, alpha=0.32, s=16, c="#8aa5c2", label="训练集")
    plt.scatter(y_true, y_pred, alpha=0.7, s=26, c="#2c3e50")
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    text_info = (
        f"R2 = {_format_metric_text(metrics.get('test_r2'), 3)}\n"
        f"RMSE = {_format_metric_text(metrics.get('test_rmse'), 2)}\n"
        f"r = {_format_metric_text(metrics.get('test_pearson_r'), 3)}"
    )
    plt.text(
        0.03,
        0.97,
        text_info,
        transform=plt.gca().transAxes,
        va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )
    plt.xlabel("真实太阳风速度 (km/s)")
    plt.ylabel("预测太阳风速度 (km/s)")
    title = (
        f"{experiment_label} 预测 vs 真实 线性散点图"
        if experiment_label
        else "预测 vs 真实 线性散点图"
    )
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    prefix = "elasticnet" if model_family == "elasticnet" else "ridge"
    save_path = output_dir / f"{prefix}_scatter.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_prediction_series(
    time_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    *,
    experiment_label: str = "",
    time_train: np.ndarray | None = None,
    train_true: np.ndarray | None = None,
    train_pred: np.ndarray | None = None,
    x_limits: tuple | None = None,
    split_time=None,
    model_family: str = DEFAULT_MODEL_FAMILY,
):
    plt.figure(figsize=(9.25, 5), dpi=300)
    ax = plt.gca()
    if time_train is not None and train_true is not None and train_pred is not None and len(time_train) > 0:
        ax.plot(
            time_train,
            train_true,
            linestyle="-",
            linewidth=1.2,
            alpha=0.7,
            color="#7f9ab3",
            label="真实值（训练集）",
        )
        ax.plot(
            time_train,
            train_pred,
            linestyle="-",
            linewidth=1.2,
            alpha=0.7,
            color="#d4a557",
            label="预测值（训练集）",
        )
    if split_time is not None:
        ax.axvline(split_time, color="#c0392b", linestyle=":", lw=1.6, label="训练 / 测试分割")
    ax.plot(
        time_test,
        y_true,
        linestyle="-",
        linewidth=1.6,
        alpha=0.9,
        color="#2c3e50",
        label="真实值",
    )
    ax.plot(
        time_test,
        y_pred,
        linestyle="-",
        linewidth=1.6,
        alpha=0.9,
        color="#c0392b",
        label="预测值",
    )
    if x_limits is not None:
        ax.set_xlim(x_limits)
    ax.set_xlabel("时间")
    ax.set_ylabel("太阳风速度 (km/s)")
    if x_limits is not None:
        start_label = pd.Timestamp(x_limits[0]).strftime("%Y-%m-%d")
        end_label = pd.Timestamp(x_limits[1]).strftime("%Y-%m-%d")
        subtitle = f"完整数据区间 {start_label} - {end_label}"
    else:
        subtitle = "完整数据区间"
    title = (
        f"{experiment_label} 时间轴预测对照 · {subtitle}"
        if experiment_label
        else f"时间轴预测对照 · {subtitle}"
    )
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", color="#d8dee8", alpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(framealpha=0.9, edgecolor="#dbe1ea", fontsize=8, loc="upper left")
    plt.tight_layout()
    prefix = "elasticnet" if model_family == "elasticnet" else "ridge"
    save_path = output_dir / f"{prefix}_timeseries.png"
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_residual_scatter(
    y_pred: np.ndarray,
    residual: np.ndarray,
    output_dir: Path,
    *,
    model_family: str = DEFAULT_MODEL_FAMILY,
):
    plt.figure(figsize=(7, 5), dpi=300)
    plt.scatter(y_pred, residual, alpha=0.7, s=24, c="#34495e")
    plt.axhline(y=0, color="red", linestyle="--", lw=2)
    plt.xlabel("预测太阳风速度 (km/s)")
    plt.ylabel("残差（真实值 - 预测值）")
    plt.title("残差散点图")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    prefix = "elasticnet" if model_family == "elasticnet" else "ridge"
    save_path = output_dir / f"{prefix}_residual_scatter.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_residual_histogram(
    residual: np.ndarray,
    output_dir: Path,
    *,
    model_family: str = DEFAULT_MODEL_FAMILY,
):
    plt.figure(figsize=(7, 5), dpi=300)
    sns.histplot(residual, bins=20, kde=True, color="#34495e")
    plt.xlabel("残差（真实值 - 预测值）")
    plt.ylabel("频数")
    plt.title("残差分布直方图")
    plt.tight_layout()
    prefix = "elasticnet" if model_family == "elasticnet" else "ridge"
    save_path = output_dir / f"{prefix}_residual_hist.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


# SHAP 解释

def run_shap_analysis(
    model: BaseEstimator,
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    output_dir: Path,
    *,
    model_family: str = DEFAULT_MODEL_FAMILY,
):
    def split_feature_lag(feature):
        name, _, lag_text = str(feature).rpartition("_t-")
        if lag_text.isdigit():
            return name, int(lag_text)
        return str(feature), None

    explainer = shap.Explainer(model.predict, Xtr)
    shap_values = explainer(Xte)
    prefix = "elasticnet" if model_family == "elasticnet" else "ridge"

    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_values, Xte, show=False)
    plt.tight_layout()
    summary_path = output_dir / f"{prefix}_shap_summary.png"
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_values, Xte, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = output_dir / f"{prefix}_shap_bar.png"
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 8), dpi=300)
    shap.plots.waterfall(shap_values[0], show=False, max_display=15)
    plt.tight_layout()
    waterfall_path = output_dir / f"{prefix}_shap_waterfall_0.png"
    plt.savefig(waterfall_path, bbox_inches="tight")
    plt.close()

    shap_importance = pd.DataFrame(
        {
            "feature": Xte.columns,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_importance["variable"] = shap_importance["feature"].map(
        lambda item: split_feature_lag(item)[0]
    )
    shap_importance["lag"] = shap_importance["feature"].map(
        lambda item: split_feature_lag(item)[1]
    )
    variable_importance = (
        shap_importance
        .groupby("variable", as_index=False)["mean_abs_shap"]
        .mean()
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    lag_profile = (
        shap_importance.loc[shap_importance["lag"].notna()]
        .groupby("lag", as_index=False)["mean_abs_shap"]
        .mean()
        .sort_values("lag")
        .reset_index(drop=True)
    )
    shap_importance_path = output_dir / f"{prefix}_shap_importance.csv"
    variable_importance_path = output_dir / f"{prefix}_shap_variable_importance.csv"
    lag_profile_path = output_dir / f"{prefix}_shap_lag_profile.csv"
    shap_importance.to_csv(shap_importance_path, index=False, encoding="utf-8-sig")
    variable_importance.to_csv(variable_importance_path, index=False, encoding="utf-8-sig")
    lag_profile.to_csv(lag_profile_path, index=False, encoding="utf-8-sig")

    return {
        "summary_path": summary_path,
        "bar_path": bar_path,
        "waterfall_path": waterfall_path,
        "importance": shap_importance,
        "importance_path": shap_importance_path,
        "variable_importance": variable_importance,
        "variable_importance_path": variable_importance_path,
        "lag_profile": lag_profile,
        "lag_profile_path": lag_profile_path,
    }


# 主流程

def main(config_path: str | None = None):
    if not config_path:
        print("用法: python -m models.elasticnet_runner <config.json>")
        return
    cfg = load_llm_json_config(config_path)
    output_dir = Path(config_path).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = load_and_merge_data(cfg)
    window_result = build_sliding_window(
        merged.X,
        merged.y,
        merged.time,
        cfg.time_window_config,
    )
    split = split_dataset_by_calendar(
        window_result.X_all,
        window_result.y_all,
        window_result.time_all,
        window_result.origins_all,
        window_result.sample_ids_all,
        window_result.records_all or [],
        cfg.time_window_config.test_split_ratio,
        window_result.feature_names,
    )

    model_family = resolve_model_family(cfg.elasticnet_config)
    display_name = MODEL_DISPLAY_NAMES[model_family]
    output_prefix = "elasticnet" if model_family == "elasticnet" else "ridge"

    model = train_elasticnet_model(
        split.X_train,
        split.y_train,
        cfg.elasticnet_config,
        cv_folds=split.cv_folds,
    )
    metrics, ytr_pred, yte_pred = evaluate_model(
        model,
        split.X_train,
        split.X_test,
        split.y_train,
        split.y_test,
    )
    coef_df = make_coefficient_table(model, window_result.feature_names)
    coef_df.to_csv(
        output_dir / f"{output_prefix}_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics["model_family"] = model_family
    metrics["alpha_selected"] = resolve_alpha_for_model(
        model,
        cfg.elasticnet_config.alpha,
    )
    (output_dir / f"{output_prefix}_metrics.json").write_text(
        json_dumps(metrics, indent=2),
        encoding="utf-8",
    )

    residual = split.y_test - yte_pred
    full_time = np.concatenate([split.time_train, split.time_test])
    full_y = np.concatenate([split.y_train, split.y_test])
    full_y_pred = np.concatenate([ytr_pred, yte_pred])
    x_limits = (full_time.min(), full_time.max()) if len(full_time) > 0 else None
    split_time = split.time_test[0] if len(split.time_test) > 0 else None
    figure_paths = {
        "coefficients": plot_feature_coefficients(
            coef_df, output_dir, model_family=model_family
        ),
        "scatter": plot_prediction_scatter(
            split.y_test,
            yte_pred,
            metrics,
            output_dir,
            experiment_label=display_name,
            train_true=split.y_train,
            train_pred=ytr_pred,
            model_family=model_family,
        ),
        "timeseries": plot_prediction_series(
            split.time_test,
            split.y_test,
            yte_pred,
            output_dir,
            experiment_label=display_name,
            time_train=split.time_train,
            train_true=split.y_train,
            train_pred=ytr_pred,
            x_limits=x_limits,
            split_time=split_time,
            model_family=model_family,
        ),
        "residual_scatter": plot_residual_scatter(
            yte_pred,
            residual,
            output_dir,
            model_family=model_family,
        ),
        "residual_hist": plot_residual_histogram(
            residual,
            output_dir,
            model_family=model_family,
        ),
    }
    shap_result = run_shap_analysis(
        model,
        split.X_train,
        split.X_test,
        output_dir,
        model_family=model_family,
    )

    print(f"{display_name} 训练完成")
    print(f"滑动窗口后样本数: {len(window_result.X_all)}")
    print(f"滑动窗口共 {window_result.total_windows} 个")
    print(f"训练段 RMSE: {_format_metric_text(metrics['train_rmse'], 3)}")
    print(f"迭代评估段 RMSE: {_format_metric_text(metrics['test_rmse'], 3)}")
    print(f"训练段 R2: {_format_metric_text(metrics['train_r2'], 3)}")
    print(f"迭代评估段 R2: {_format_metric_text(metrics['test_r2'], 3)}")
    print(f"迭代评估段 Pearson r: {_format_metric_text(metrics['test_pearson_r'], 3)}")
    print(f"迭代评估段均值对照组 RMSE: {_format_metric_text(metrics['baseline_test_rmse'], 3)}")
    print(f"选定的 alpha: {metrics['alpha_selected']:.6g}")
    print("系数绝对值 Top 特征:")
    print(coef_df.head(10).to_string(index=False))

    return {
        "config": cfg.model_dump(mode="json"),
        "metrics": metrics,
        "coefficients": coef_df,
        "shap_importance": shap_result["importance"],
        "model": model,
        "model_family": model_family,
        "time_train": split.time_train,
        "time_test": split.time_test,
        "y_train": split.y_train,
        "y_test": split.y_test,
        "y_train_pred": ytr_pred,
        "y_test_pred": yte_pred,
        "figure_paths": {name: str(path) for name, path in figure_paths.items()},
        "shap_paths": {
            "summary": shap_result["summary_path"],
            "bar": shap_result["bar_path"],
            "waterfall": shap_result["waterfall_path"],
            "importance_csv": shap_result["importance_path"],
        },
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
