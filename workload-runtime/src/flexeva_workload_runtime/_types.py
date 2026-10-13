"""Shared invocation and result types for frozen workload execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class WorkloadExecutionError(ValueError):
    """The invocation violates its declared workload execution contract."""


@dataclass(frozen=True)
class WorkloadInvocation:
    lane: int | str
    arguments: Mapping[str, Any]
    globals: Mapping[str, Any] = field(default_factory=dict)
    mode: str = "instrumented"


@dataclass(frozen=True)
class WorkloadRunResult:
    output: Any
    audit: Mapping[str, Any]
