"""Backend-free contracts for a source-owned, structurally selected Code region.

These records describe source structure, not a completed dynamic execution or
an evaluator checkpoint. No training-specific concept is part of the contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionSelector:
    """Select a unique with-body beneath a particular enclosing for statement."""

    callee: str
    for_target: str
    for_iter: str


@dataclass(frozen=True)
class MutableRegion:
    region_id: str
    files: tuple[str, ...]
    entrypoint: str
    selector: RegionSelector | None = None

    def __post_init__(self) -> None:
        if not self.region_id or not self.entrypoint or not self.files:
            raise ValueError("region requires an identity, owned source files and one entrypoint")
        if len(set(self.files)) != len(self.files):
            raise ValueError("region source files must be unique")
