"""Portable Code input evidence, independent of execution and layer adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ..errors import PipelineContractError
from ..model import StateRef


def safe_input(value: Any) -> Any:
    """Copy JSON-shaped evidence without invoking user conversion or repr hooks."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (tuple, list):
        return tuple(safe_input(item) for item in value)
    if type(value) in (dict, MappingProxyType):
        if not all(type(key) is str for key in value):
            raise PipelineContractError("dynamic input evidence requires string mapping keys")
        return {key: safe_input(item) for key, item in sorted(value.items())}
    raise PipelineContractError("dynamic input evidence must contain finite JSON values, not live objects")


@dataclass(frozen=True)
class DynamicInputEvidence:
    """An input provider's explicit scope; completeness is never inferred here.

    ``producer_inputs`` pairs an actual producer instance ID with the consumed
    symbol. It is not a lexical site ID or a guessed previous event. Source
    inputs describe selected pre-window bindings, not every program input.
    """

    producer_inputs: tuple[tuple[str, str], ...] = ()
    semantic_inputs: Mapping[str, Any] = field(default_factory=dict)
    source_inputs: Mapping[str, Any] = field(default_factory=dict)
    state_refs: tuple[StateRef[Any], ...] = ()
    inputs_complete: bool = False
    unresolved: tuple[str, ...] = ()

    def checked(self) -> DynamicInputEvidence:
        if type(self.inputs_complete) is not bool:
            raise PipelineContractError("dynamic input completeness must be an explicit Boolean")
        if any(not isinstance(ref, StateRef) for ref in self.state_refs):
            raise PipelineContractError("dynamic state inputs must be managed StateRefs")
        if any(not producer or not symbol or type(producer) is not str or type(symbol) is not str
               for producer, symbol in self.producer_inputs):
            raise PipelineContractError("dynamic dependencies require producer instance and symbol")
        if self.inputs_complete and self.unresolved:
            raise PipelineContractError("unresolved dynamic inputs cannot be marked complete")
        return DynamicInputEvidence(
            tuple(sorted(set(self.producer_inputs))), safe_input(self.semantic_inputs),
            safe_input(self.source_inputs), tuple(self.state_refs), self.inputs_complete,
            tuple(self.unresolved),
        )
