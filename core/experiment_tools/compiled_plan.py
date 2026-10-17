"""The single authoritative execution plan produced by protocol validation.

Validation compiles an ``ExperimentProtocol`` into exactly one plan. Every
executor consumes this plan and must not re-read legacy duplicate
configuration such as ``feature_groups`` / ``arm_time_windows``, silently
change model families, or report a different feature set than the one that
was actually trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.unified_schema import (
    ContrastSpec,
    DiagnosticSpec,
    ProtocolAlignmentSpec,
    RunSpec,
)


@dataclass
class CompiledExperimentPlan:
    """Authoritative run / contrast / diagnostic structure for one protocol.

    ``time_parameters`` holds the shared window, horizon, lag and split that
    every run in a normal training experiment must respect; per-run overrides
    are validated to match these values before execution.
    """

    experiment_type: str
    target: str
    experiment_id: str
    round_id: int
    model_parameters: dict[str, Any] = field(default_factory=dict)
    time_parameters: dict[str, Any] = field(default_factory=dict)
    runs: list[RunSpec] = field(default_factory=list)
    contrasts: list[ContrastSpec] = field(default_factory=list)
    diagnostics: list[DiagnosticSpec] = field(default_factory=list)
    alignment: ProtocolAlignmentSpec | None = None
    feature_lags: dict[str, list[int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_type": self.experiment_type,
            "target": self.target,
            "experiment_id": self.experiment_id,
            "round_id": self.round_id,
            "model_parameters": dict(self.model_parameters),
            "time_parameters": dict(self.time_parameters),
            "feature_lags": {
                key: list(values)
                for key, values in self.feature_lags.items()
            },
            "runs": [item.model_dump(mode="json", exclude_none=True) for item in self.runs],
            "contrasts": [
                item.model_dump(mode="json", exclude_none=True) for item in self.contrasts
            ],
            "diagnostics": [
                item.model_dump(mode="json", exclude_none=True) for item in self.diagnostics
            ],
            "alignment": (
                self.alignment.model_dump(mode="json", exclude_none=True)
                if self.alignment is not None
                else None
            ),
        }


def normalize_diagnostic_id(diagnostic_id: str) -> str | None:
    """Canonicalize supported diagnostic names, returning None for unknown ones."""

    normalized = str(diagnostic_id or "").strip().lower()
    aliases = {
        "paired_bootstrap": "paired_calendar_bootstrap",
        "paired_calendar_bootstrap": "paired_calendar_bootstrap",
        "bootstrap": "paired_calendar_bootstrap",
        "time_segment": "time_segments",
        "time_segments": "time_segments",
        "residual": "residual_diagnostics",
        "residual_diagnostics": "residual_diagnostics",
    }
    return aliases.get(normalized)
