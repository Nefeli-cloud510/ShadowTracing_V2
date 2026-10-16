"""Execution-time data health checks for the live experiment pipeline.

The 2021 live CSV pair was found to contain a systematic one-column shift.
These checks are intentionally conservative: a malformed or physically
implausible field should block execution instead of silently feeding a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


class DataProvenanceError(RuntimeError):
    """Raised when merged live data fails a pre-execution health check."""


@dataclass
class DataValidationReport:
    source_labels: list[str] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> dict:
        return {
            "source_labels": list(self.source_labels),
            "passed": self.passed,
            "checks": list(self.checks),
        }


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _record(
    report: DataValidationReport,
    check: str,
    failures: pd.Series | Iterable[bool],
    frame: pd.DataFrame,
    value_col: str | None = None,
    tolerance: float = 1e-6,
) -> None:
    """Append one check, keeping date-level detail for human review."""
    mask = np.asarray(failures, dtype=bool)
    failing_dates = []
    if mask.any():
        time_col = "TIME" if "TIME" in frame.columns else frame.columns[0]
        failing_dates = [str(value) for value in frame.loc[mask, time_col].tolist()]
    bad_values = []
    if value_col is not None and mask.any():
        raw = frame.loc[mask, value_col].tolist()
        bad_values = [None if pd.isna(value) else float(value) for value in raw[:8]]
    report.checks.append(
        {
            "check": check,
            "fail_count": int(mask.sum()),
            "tolerance": tolerance,
            "bad_dates": failing_dates[:20],
            "bad_values_sample": bad_values,
        }
    )
    if mask.any():
        report.passed = False


def validate_merged_data(
    frame: pd.DataFrame,
    source_labels: Iterable[str],
    time_column: str = "TIME",
) -> DataValidationReport:
    """Validate standard OMNI/Solar-wind columns after all sources are merged.

    Missing values are skipped by range/relation checks; the checks use the
    column names that appear in the v2/v3 live input CSVs. Execution is blocked
    when a standard field carries physically impossible values, which is the
    signature left by the original one-column shift.
    """
    report = DataValidationReport(source_labels=list(source_labels))
    frame = frame.copy()
    if time_column not in frame.columns:
        time_column = frame.columns[0]

    numeric_columns = [
        column
        for column in frame.columns
        if column
        in {
            "Scalar B, nT",
            "BX, nT (GSE, GSM)",
            "BY, nT (GSE)",
            "BZ, nT (GSE)",
            "BY, nT (GSM)",
            "BZ, nT (GSM)",
            "SW Plasma Temperature, K",
            "SW Proton Density, N/cm^3",
            "SW Plasma Speed, km/s",
            "Kp index",
            "Dst-index, nT",
            "ap_index, nT",
            "p2_y中心值",
            "p2_y中心值误差",
            "p1_x中心值",
            "p1_x中心值误差",
            "p3 sigma_宽度",
            "_llf 最优卡方值",
        }
    ]
    for column in numeric_columns:
        frame[column] = _numeric(frame, column)

    # The original shift filled Scalar B with zeros while keeping non-zero
    # data in the following fields. A 100% zero-filled multi-hundred-row B
    # field is never a realistic OMNI daily record.
    if "Scalar B, nT" in frame.columns:
        scalar_b = frame["Scalar B, nT"].astype(float)
        valid_b = scalar_b.notna()
        non_zero = scalar_b.abs() > 1e-9
        if valid_b.sum() >= 30 and non_zero.sum() == 0:
            _record(
                report,
                "Scalar B, nT 非零值数量",
                pd.Series([True] * len(frame), index=frame.index),
                frame,
                value_col=None,
            )
        elif valid_b.sum() > 0:
            _record(
                report,
                "Scalar B, nT > 0 或缺失",
                (scalar_b <= 0) & valid_b,
                frame,
                value_col="Scalar B, nT",
            )

    component_checks = [
        ("BX, nT (GSE, GSM)", "|Bx| <= Scalar B"),
        ("BY, nT (GSE)", "|By_GSE| <= Scalar B"),
        ("BZ, nT (GSE)", "|Bz_GSE| <= Scalar B"),
    ]
    if "Scalar B, nT" in frame.columns:
        scalar_b = frame["Scalar B, nT"].astype(float)
        for column, label in component_checks:
            if column not in frame.columns:
                continue
            values = frame[column].astype(float)
            paired = values.notna() & scalar_b.notna()
            failures = pd.Series(False, index=frame.index)
            if paired.any():
                failures = (values.abs() > scalar_b + 1e-6) & paired
            _record(report, label, failures, frame, value_col=column)

    range_checks = [
        ("SW Plasma Speed, km/s", 150.0, 1500.0, "风速 150-1500 km/s"),
        ("SW Proton Density, N/cm^3", 0.01, 100.0, "密度 0.01-100 /cm^3"),
        ("SW Plasma Temperature, K", 1e4, 5e6, "温度 1e4-5e6 K"),
    ]
    for column, lower, upper, label in range_checks:
        if column not in frame.columns:
            continue
        values = frame[column].astype(float)
        valid = values.notna()
        failures = pd.Series(False, index=frame.index)
        if valid.any():
            failures = valid & ((values < lower) | (values > upper))
        _record(report, label, failures, frame, value_col=column)

    if not report.passed:
        failed = [check["check"] for check in report.checks if check["fail_count"] > 0]
        detail_lines = ["执行前数据校验失败，已阻断本次实验。"]
        for name in failed:
            detail_lines.append(f"- {name}")
        raise DataProvenanceError("\n".join(detail_lines))
    return report
