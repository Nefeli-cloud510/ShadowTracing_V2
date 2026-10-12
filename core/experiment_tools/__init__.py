"""Structured scientific experiment tools for ShadowTracing.

Each candidate protocol is compiled into runs, contrasts and diagnostics
before execution. Unadapted experiment types are never silently collapsed
into an ordinary incremental test.
"""

from core.experiment_tools.registry import (
    EXPERIMENT_TOOL_INDEX,
    ExperimentToolRecord,
    available_tool_summaries,
    get_tool,
)
from core.experiment_tools.validate_protocol import (
    ProtocolValidationResult,
    validate_experiment_protocol,
)

__all__ = [
    "EXPERIMENT_TOOL_INDEX",
    "ExperimentToolRecord",
    "ProtocolValidationResult",
    "available_tool_summaries",
    "get_tool",
    "validate_experiment_protocol",
]
