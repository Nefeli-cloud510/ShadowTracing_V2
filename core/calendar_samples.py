from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    """A calendar-shifted feature: ``variable`` observed ``offset_days`` relative
    to the forecast issue date (0 = issue date, -1 = previous calendar day)."""

    variable: str
    offset_days: int = 0

    def feature_name(self) -> str:
        if self.offset_days == 0:
            return f"{self.variable}_t"
        return f"{self.variable}_t-{abs(int(self.offset_days))}"

    def to_dict(self) -> dict[str, Any]:
        return {"variable": self.variable, "offset_days": int(self.offset_days)}


@dataclass(frozen=True)
class CalendarSample:
    sample_id: str
    issue_date: pd.Timestamp
    target_date: pd.Timestamp
    horizon_days: int
    feature_specs: tuple[FeatureSpec, ...]
    feature_dates: tuple[pd.Timestamp, ...]
    sample_condition: str = "complete_case"
    drop_reason: str | None = None

    @property
    def compact_key(self) -> tuple[str, str, int]:
        return (
            self.issue_date.date().isoformat(),
            self.target_date.date().isoformat(),
            int(self.horizon_days),
        )

    @property
    def feature_names(self) -> list[str]:
        return [spec.feature_name() for spec in self.feature_specs]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "issue_date": self.issue_date.date().isoformat(),
            "target_date": self.target_date.date().isoformat(),
            "horizon_days": int(self.horizon_days),
            "feature_specs": [spec.to_dict() for spec in self.feature_specs],
            "feature_dates": [value.date().isoformat() for value in self.feature_dates],
            "feature_names": self.feature_names,
            "sample_condition": self.sample_condition,
            "drop_reason": self.drop_reason,
        }


@dataclass
class SamplePlan:
    """Complete calendar sample plan with retention/exclusion bookkeeping.

    ``records`` contains every candidate; a record is retained only when the
    union of the comparison's required feature dates and the target date are
    present and non-null. Missing values are never imputed.
    """

    records: list[CalendarSample]
    feature_specs: tuple[FeatureSpec, ...]
    feature_columns: list[str]
    target_name: str
    past_lag_days: int
    forecast_horizon_days: int
    original_days: int = 0
    quality_available_days: int = 0
    field_drop_counts: dict[str, int] = field(default_factory=dict)
    generated_at: str = ""
    date_start: str | None = None
    date_end: str | None = None
    data_version: str = "shadow-calendar-v1"

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()

    @property
    def candidate_count(self) -> int:
        return len(self.records)

    @property
    def retained_count(self) -> int:
        return sum(1 for record in self.records if record.drop_reason is None)

    @property
    def dropped_count(self) -> int:
        return self.candidate_count - self.retained_count

    def retained_records(self) -> list[CalendarSample]:
        return sorted(
            (record for record in self.records if record.drop_reason is None),
            key=lambda record: (record.issue_date, record.sample_id),
        )

    def drop_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            reason = record.drop_reason or "retained"
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def stats(self) -> dict[str, Any]:
        return {
            "data_version": self.data_version,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "original_days": self.original_days,
            "quality_available_days": self.quality_available_days,
            "candidate_count": self.candidate_count,
            "retained_count": self.retained_count,
            "dropped_count": self.dropped_count,
            "drop_reasons": self.drop_reason_counts(),
            "field_drop_counts": self.field_drop_counts,
            "past_lag_days": self.past_lag_days,
            "forecast_horizon_days": self.forecast_horizon_days,
            "generated_at": self.generated_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.stats(),
            "feature_specs": [spec.to_dict() for spec in self.feature_specs],
            "feature_columns": list(self.feature_columns),
            "target_name": self.target_name,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True)
class FrozenEvaluationPlan:
    """Frozen evaluation partition persisted once per project.

    ``test_start`` fixes the first issue date of the evaluation segment so a
    changed feature set cannot silently move the boundary. ``split_label`` is
    deliberately conservative: when there is no sealed, never-inspected final
    interval, the run is recorded as ``dev_backtest`` instead of pretending to
    be a blind final test.
    """

    schema_version: str = "evaluation-plan-v1"
    split_label: str = "dev_backtest"
    test_ratio: float = 0.2
    n_internal_folds: int = 2
    test_start: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: list[str] = field(default_factory=list)

    ALLOWED_LABELS: ClassVar[tuple[str, ...]] = ("dev_backtest", "final_test_blind")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "split_label": self.split_label,
            "test_ratio": float(self.test_ratio),
            "n_internal_folds": int(self.n_internal_folds),
            "test_start": self.test_start,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FrozenEvaluationPlan:
        label = str(payload.get("split_label") or "dev_backtest")
        if label not in cls.ALLOWED_LABELS:
            raise ValueError(f"评估计划 split_label 非法: {label}")
        return cls(
            schema_version=str(payload.get("schema_version") or "evaluation-plan-v1"),
            split_label=label,
            test_ratio=float(payload.get("test_ratio") or 0.2),
            n_internal_folds=int(payload.get("n_internal_folds") or 2),
            test_start=str(payload["test_start"]) if payload.get("test_start") else None,
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            notes=list(payload.get("notes") or []),
        )

    def with_test_start(
        self,
        test_start: pd.Timestamp | str | None,
    ) -> FrozenEvaluationPlan:
        if test_start is None:
            return self
        return FrozenEvaluationPlan(
            schema_version=self.schema_version,
            split_label=self.split_label,
            test_ratio=self.test_ratio,
            n_internal_folds=self.n_internal_folds,
            test_start=str(pd.Timestamp(test_start).normalize().date().isoformat()),
            created_at=self.created_at,
            updated_at=datetime.now().isoformat(),
            notes=list(self.notes),
        )


def evaluation_plan_path(project_root: Path | str) -> Path:
    root = Path(project_root)
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "evaluation_plan.json"


def compute_default_test_start(
    records: list[CalendarSample],
    *,
    test_ratio: float = 0.2,
    n_internal_folds: int = 2,
) -> pd.Timestamp:
    """Derive the initial evaluation boundary using the legacy fraction rule."""

    retained = [
        record
        for record in records
        if record.drop_reason is None
    ]
    issues = _unique_issue_dates(retained)
    min_dates = int(n_internal_folds) + 2
    if len(issues) < min_dates:
        raise ValueError(
            f"共同样本仅有 {len(issues)} 个 issue 日期，少于划分为 "
            f"{n_internal_folds} 个内部折 + 测试段所需的最小 {min_dates}"
        )
    test_start_idx = int(round(len(issues) * (1.0 - float(test_ratio))))
    test_start_idx = max(
        int(n_internal_folds) + 1,
        min(len(issues) - 1, test_start_idx),
    )
    return issues[test_start_idx]


def load_or_freeze_evaluation_plan(
    project_root: Path | str,
    *,
    test_ratio: float = 0.2,
    n_internal_folds: int = 2,
    requested_split_label: str | None = None,
    sealed_final_test: bool = False,
) -> FrozenEvaluationPlan:
    """Load a project-level frozen evaluation plan, creating it on first use.

    An existing file is authoritative and every candidate must reuse the same
    boundary. A new ``final_test_blind`` label is only honored when the caller
    can prove the interval is sealed and uninspected; otherwise the plan is
    recorded as ``dev_backtest`` so the system never relabels inspected data as
    a blind test.
    """

    path = evaluation_plan_path(project_root)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"评估计划无法解析 {path}: {exc}") from exc
        plan = FrozenEvaluationPlan.from_dict(payload)
        if not math.isclose(
            plan.test_ratio,
            float(test_ratio),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"已冻结评估计划 test_ratio={plan.test_ratio}，"
                f"与本次请求 {test_ratio} 冲突，禁止静默重切。"
            )
        if plan.n_internal_folds != int(n_internal_folds):
            raise ValueError(
                f"已冻结评估计划 n_internal_folds={plan.n_internal_folds}，"
                f"与本次请求 {n_internal_folds} 冲突，禁止静默重切。"
            )
        return plan

    label = str(requested_split_label or "dev_backtest")
    notes = [
        "评估分区在项目首次执行时冻结，后续候选实验不得移动 test_start。",
        "缺失值在冻结的测试段内按每个实验各自的共同日历样本处理，不得越过分区边界。",
    ]
    if label == "final_test_blind" and not sealed_final_test:
        label = "dev_backtest"
        notes.append(
            "请求 final_test_blind，但没有封存且从未查看的最终测试区间证据；"
            "按 dev_backtest 如实记录，不把已查看数据伪称为盲测。"
        )
    if label not in FrozenEvaluationPlan.ALLOWED_LABELS:
        raise ValueError(f"split_label 非法: {label}")

    plan = FrozenEvaluationPlan(
        split_label=str(label),
        test_ratio=float(test_ratio),
        n_internal_folds=int(n_internal_folds),
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        notes=notes,
    )
    return freeze_evaluation_plan(project_root, plan)


def freeze_evaluation_plan(
    project_root: Path | str,
    plan: FrozenEvaluationPlan,
) -> FrozenEvaluationPlan:
    path = evaluation_plan_path(project_root)
    path.write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return plan


def is_blind_sealed(project_root: Path | str) -> bool:
    """Whether the persisted project plan is a sealed final-test partition."""

    path = evaluation_plan_path(project_root)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return FrozenEvaluationPlan.from_dict(payload).split_label == "final_test_blind"
    except (OSError, ValueError):
        return False


def _normalize_dates(time_raw: pd.Series) -> pd.Series:
    series = pd.Series(time_raw, index=time_raw.index)
    if pd.api.types.is_datetime64_any_dtype(series):
        times = pd.to_datetime(series, errors="coerce")
    else:
        cleaned = (
            series.astype(str)
            .str.replace(r"_x000d_", "", regex=True)
            .str.strip()
        )
        yyyymmdd = cleaned.str.extract(r"(\d{8})", expand=False)
        times = pd.to_datetime(yyyymmdd, format="%Y%m%d", errors="coerce")
        if times.isna().all():
            times = pd.to_datetime(cleaned, errors="coerce")
    if times.isna().any():
        bad_count = int(times.isna().sum())
        raise ValueError(f"时间列存在 {bad_count} 个无法解析/空白的日期，不能构造日历样本")
    return times.dt.normalize()


def check_unique_calendar_dates(time_raw: pd.Series) -> None:
    """Reject duplicate calendar days explicitly instead of keeping the first."""

    times = _normalize_dates(time_raw)
    counts = times.value_counts()
    duplicates = counts[counts > 1]
    if not duplicates.empty:
        detail = "; ".join(
            f"{pd.Timestamp(day).date().isoformat()} x{int(count)}"
            for day, count in duplicates.head(10).items()
        )
        total_duplicates = int((counts - 1).sum())
        raise ValueError(
            f"日历索引不允许重复日期（共 {total_duplicates} 个多余行）：{detail}"
        )


def build_sample_plan(
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    time_raw: pd.Series,
    *,
    feature_columns: list[str],
    target_name: str,
    past_lag_days: int,
    forecast_horizon_days: int,
) -> SamplePlan:
    """Construct the comparison-level complete-case calendar sample plan.

    Feature dates are ``issue_date - lag`` for ``lag`` in ``0..past_lag_days-1``;
    the target is ``issue_date + forecast_horizon_days`` calendar days. A record
    is retained only when both arms' union feature columns and the target date
    are available; missing values are never interpolated.
    """

    if not isinstance(X_raw, pd.DataFrame) or X_raw.shape[1] == 0:
        raise ValueError("X_raw 必须是非空 DataFrame")
    feature_columns = list(dict.fromkeys(str(item) for item in feature_columns))
    missing_cols = [column for column in feature_columns if column not in X_raw.columns]
    if missing_cols:
        raise KeyError(f"日历样本计划缺少特征列: {missing_cols}")
    if len(y_raw) != len(X_raw) or len(time_raw) != len(X_raw):
        raise ValueError("X_raw / y_raw / time_raw 行数必须一致")

    past_lag_days = int(past_lag_days)
    forecast_horizon_days = int(forecast_horizon_days)
    if past_lag_days < 1:
        raise ValueError("past_lag_days 必须 >= 1")
    if forecast_horizon_days < 1:
        raise ValueError("forecast_horizon_days 必须 >= 1（同天标签不属于严格预测回测）")

    check_unique_calendar_dates(time_raw)
    times = _normalize_dates(time_raw)

    feature_values = X_raw[feature_columns].astype(float)
    numeric_ok = feature_values.notna().to_numpy(dtype=bool)
    target_values = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
    target_ok = np.isfinite(target_values)
    quality_available_days = int(
        np.count_nonzero(numeric_ok.all(axis=1) & target_ok)
    )

    specs = tuple(
        FeatureSpec(variable=column, offset_days=-lag)
        for column in feature_columns
        for lag in range(0, past_lag_days)
    )
    feature_names = [spec.feature_name() for spec in specs]

    ordered = pd.DataFrame(
        {
            "_hp_time": times.to_numpy(),
            "_hp_orig_index": np.arange(len(X_raw)),
        }
    ).sort_values("_hp_time", kind="mergesort")
    dates = ordered["_hp_time"].to_numpy()
    original_rows = ordered["_hp_orig_index"].to_numpy(dtype=int)
    # 排序后特征/目标掩码必须与日期同步，否则输入行顺序会改变样本计划。
    numeric_ok = numeric_ok[original_rows]
    target_ok = target_ok[original_rows]
    date_to_index = {
        pd.Timestamp(value).normalize(): index for index, value in enumerate(dates)
    }

    records: list[CalendarSample] = []
    field_drop_counts: dict[str, int] = {}
    for origin_index, origin_value in enumerate(dates):
        origin = pd.Timestamp(origin_value).normalize()
        feature_dates = tuple(
            origin - pd.Timedelta(days=lag) for lag in range(0, past_lag_days)
        )
        target_date = origin + pd.Timedelta(days=forecast_horizon_days)
        sample_id = f"{origin.date().isoformat()}__h{forecast_horizon_days}"

        common_args: dict[str, Any] = {
            "sample_id": sample_id,
            "issue_date": origin,
            "target_date": target_date,
            "horizon_days": forecast_horizon_days,
            "feature_specs": specs,
            "feature_dates": feature_dates,
        }
        history_dates = [
            pd.Timestamp(value).normalize() for value in feature_dates
        ]
        history_complete = all(
            value in date_to_index for value in history_dates
        )
        if not history_complete:
            records.append(
                CalendarSample(
                    **common_args,
                    sample_condition="incomplete_case",
                    drop_reason="history_gap",
                )
            )
            continue

        feature_rows = [date_to_index[value] for value in history_dates]
        missing_rows = [
            row_index
            for row_index in feature_rows
            if not bool(numeric_ok[row_index].all())
        ]
        if missing_rows:
            for col_idx, column in enumerate(feature_columns):
                for row_index in feature_rows:
                    if not bool(numeric_ok[row_index, col_idx]):
                        field_drop_counts[column] = field_drop_counts.get(column, 0) + 1
            records.append(
                CalendarSample(
                    **common_args,
                    sample_condition="incomplete_case",
                    drop_reason="feature_missing",
                )
            )
            continue

        if pd.Timestamp(target_date).normalize() not in date_to_index:
            records.append(
                CalendarSample(
                    **common_args,
                    sample_condition="incomplete_case",
                    drop_reason="target_unavailable",
                )
            )
            continue
        target_index = date_to_index[pd.Timestamp(target_date).normalize()]
        if not bool(target_ok[target_index]):
            field_drop_counts[target_name] = field_drop_counts.get(target_name, 0) + 1
            records.append(
                CalendarSample(
                    **common_args,
                    sample_condition="incomplete_case",
                    drop_reason="target_unavailable",
                )
            )
            continue

        records.append(CalendarSample(**common_args, sample_condition="complete_case"))

    unique_dates = pd.Series(dates).dt.normalize()
    return SamplePlan(
        records=records,
        feature_specs=specs,
        feature_columns=feature_columns,
        target_name=str(target_name),
        past_lag_days=past_lag_days,
        forecast_horizon_days=forecast_horizon_days,
        original_days=int(len(unique_dates)),
        quality_available_days=quality_available_days,
        field_drop_counts=field_drop_counts,
        date_start=unique_dates.min().date().isoformat() if len(unique_dates) else None,
        date_end=unique_dates.max().date().isoformat() if len(unique_dates) else None,
    )


@dataclass(frozen=True)
class CalendarSplit:
    """Frozen calendar split shared by both experiment arms.

    ``train`` / ``test`` are the retained common samples used by both arms.
    ``boundary_excluded`` contains records whose issue date is before the test
    segment but whose target date is not available at the test cut-off; they are
    never used in training or testing.
    ``cv_folds`` are expanding-window folds expressed against the *train* list.
    """

    train: list[CalendarSample]
    test: list[CalendarSample]
    boundary_excluded: list[CalendarSample]
    cv_folds: list[tuple[list[CalendarSample], list[CalendarSample]]]
    test_start: pd.Timestamp
    block_boundaries: list[pd.Timestamp]
    test_ratio: float
    data_version: str = "shadow-split-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_version": self.data_version,
            "test_start": self.test_start.date().isoformat(),
            "block_boundaries": [value.date().isoformat() for value in self.block_boundaries],
            "test_ratio": self.test_ratio,
            "train_count": len(self.train),
            "test_count": len(self.test),
            "boundary_excluded_count": len(self.boundary_excluded),
            "cv_folds": [
                {
                    "fold": index + 1,
                    "train_count": len(train),
                    "eval_count": len(eval_samples),
                    "train_targets_strictly_before_eval_issue": bool(
                        train
                        and eval_samples
                        and all(item.target_date < eval_samples[0].issue_date for item in train)
                    ),
                }
                for index, (train, eval_samples) in enumerate(self.cv_folds)
            ],
        }


def _unique_issue_dates(records: list[CalendarSample]) -> list[pd.Timestamp]:
    return sorted({record.issue_date for record in records})


def _split_list_by_count(values: list, n_blocks: int) -> list[list]:
    if n_blocks < 1:
        raise ValueError("n_blocks 必须 >= 1")
    if n_blocks > len(values):
        raise ValueError(
            f"无法把 {len(values)} 个 issue 日期切成 {n_blocks} 块"
        )
    blocks: list[list] = []
    for index in range(n_blocks):
        start = int(round(len(values) * index / n_blocks))
        end = int(round(len(values) * (index + 1) / n_blocks))
        blocks.append(values[start:end])
    return blocks


def make_calendar_split(
    records: list[CalendarSample],
    *,
    test_ratio: float = 0.2,
    n_internal_folds: int = 2,
    test_start: pd.Timestamp | str | None = None,
) -> CalendarSplit:
    """Create the frozen common-sample split and internal expanding folds.

    The 196-day public case has a single target year, so the independent
    scripts' year-based folds are adapted to contiguous issue-date blocks:
    fold 1 trains on block 1 and evaluates block 2; fold 2 trains on blocks
    1+2 and evaluates block 3; the final model trains on blocks 1..3 and is
    tested on the last block. Every fold and the final model assert that all
    training target dates are strictly before the evaluation segment's first
    issue date.
    """

    retained = sorted(
        (record for record in records if record.drop_reason is None),
        key=lambda record: (record.issue_date, record.sample_id),
    )
    if not retained:
        raise ValueError("没有可用的共同样本，无法划分训练/测试")
    issues = _unique_issue_dates(retained)
    min_dates = n_internal_folds + 2
    if len(issues) < min_dates:
        raise ValueError(
            f"共同样本仅有 {len(issues)} 个 issue 日期，少于划分为 "
            f"{n_internal_folds} 个内部折 + 测试段所需的最小 {min_dates}"
        )
    if test_start is not None:
        frozen_test_start = pd.Timestamp(test_start).normalize()
        train_issues = [value for value in issues if value < frozen_test_start]
        if len(train_issues) < n_internal_folds + 1:
            raise ValueError(
                f"冻结测试起点 {frozen_test_start.date().isoformat()} 过早，"
                f"只有 {len(train_issues)} 个训练 issue 日期，不足 "
                f"{n_internal_folds + 1} 个内部折块。"
            )
        test_issues = [value for value in issues if value >= frozen_test_start]
        if not test_issues:
            raise ValueError(
                f"冻结测试起点 {frozen_test_start.date().isoformat()} 晚于全部 "
                f"{len(issues)} 个 issue 日期，测试段为空。"
            )
        test_start = frozen_test_start
    else:
        test_start_idx = int(round(len(issues) * (1.0 - test_ratio)))
        test_start_idx = max(
            n_internal_folds + 1,
            min(len(issues) - 1, test_start_idx),
        )
        test_start = issues[test_start_idx]

    issue_to_record: dict[pd.Timestamp, list[CalendarSample]] = {}
    for record in retained:
        issue_to_record.setdefault(record.issue_date, []).append(record)

    train_issues = [value for value in issues if value < test_start]
    blocks = _split_list_by_count(train_issues, n_internal_folds + 1)
    block_records = [
        [record for issue_date in block for record in issue_to_record.get(issue_date, [])]
        for block in blocks
    ]

    cv_folds: list[tuple[list[CalendarSample], list[CalendarSample]]] = []
    for fold_index in range(n_internal_folds):
        eval_block = block_records[fold_index + 1]
        eval_start = eval_block[0].issue_date
        train_records: list[CalendarSample] = []
        for earlier_block in block_records[: fold_index + 1]:
            train_records.extend(
                record
                for record in earlier_block
                if record.target_date < eval_start
            )
        if not train_records or not eval_block:
            raise ValueError(
                f"内部折 {fold_index + 1} 训练或评估为空，无法进行训练内选参"
            )
        for record in train_records:
            if record.target_date >= eval_start:
                raise AssertionError(
                    f"内部折 {fold_index + 1} 训练标签晚于评估首个 issue_date"
                )
        cv_folds.append((train_records, eval_block))

    test_records: list[CalendarSample] = []
    for record in retained:
        if record.issue_date >= test_start:
            test_records.append(record)
    boundary_excluded = [
        record
        for record in retained
        if record.issue_date < test_start and record.target_date >= test_start
    ]

    final_train_records = [
        record
        for record in retained
        if record.issue_date < test_start and record.target_date < test_start
    ]
    for record in final_train_records:
        if record.target_date >= test_start:
            raise AssertionError("最终训练标签晚于测试段首个 issue_date")

    # GridSearchCV receives folds against the final train list.
    train_id_to_index = {
        record.sample_id: index for index, record in enumerate(final_train_records)
    }
    normalized_folds: list[tuple[list[CalendarSample], list[CalendarSample]]] = []
    for train_samples, eval_samples in cv_folds:
        normalized_folds.append(
            (
                [record for record in train_samples if record.sample_id in train_id_to_index],
                [record for record in eval_samples if record.sample_id in train_id_to_index],
            )
        )

    return CalendarSplit(
        train=final_train_records,
        test=test_records,
        boundary_excluded=boundary_excluded,
        cv_folds=normalized_folds,
        test_start=test_start,
        block_boundaries=[block[0] for block in blocks if block] + [test_start],
        test_ratio=test_ratio,
    )


def records_to_dataframe(records: list[CalendarSample], *, split: str | None = None) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    rows = []
    for record in records:
        row = {
            "sample_id": record.sample_id,
            "issue_date": record.issue_date.date().isoformat(),
            "target_date": record.target_date.date().isoformat(),
            "horizon_days": record.horizon_days,
            "split": split or "",
        }
        for spec in record.feature_specs:
            row[spec.feature_name()] = 1
        rows.append(row)
    return pd.DataFrame(rows)
