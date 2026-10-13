"""Evaluator-owned classification of actual mutable-source interfaces.

Source ownership identifies candidate interfaces. The evaluator's contract
decides which of those interfaces is an emission for its transform. Resolution
must not execute the target, inspect backend internals, or retain borrowed
operands. Input completeness and reuse validity remain separate obligations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ..errors import PipelineContractError
from .inputs import safe_input


class CodeCallTarget(Protocol):
    """Borrowed call boundary supplied by the optional workload runtime.

    For a protocol operation, ``operation`` names the interface and ``callable``
    is its Python operator function, not a guessed reflected special method.
    Only the evaluator can declare that this interface covers its evaluation.
    None of these live fields is a persistent Code event or checkpoint.
    """

    site: Mapping[str, Any]
    lane: int | str
    callable: Callable[..., Any] | None
    receiver: Any
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    operation: str | None


@dataclass(frozen=True)
class EmissionBinding:
    """Stable transform/operation names; deliberately no callable or result."""

    transform_id: str
    operation_id: str

    def __post_init__(self) -> None:
        if any(type(value) is not str or not value
               for value in (self.transform_id, self.operation_id)):
            raise PipelineContractError("emission binding requires nonempty string identities")


class CodeEmissionContract(Protocol):
    """A transform author's deterministic interface classification contract.

    ``None`` means this interface is ordinary host execution for this evaluator;
    it does not mean the interface is pure. A binding does not prove complete
    inputs, output equivalence, native completion, or permission to skip a call.
    Classification-relevant configuration must be in the identity. Resolution
    uses already evaluated operands and must neither invoke nor mutate them.

    An optional ``summarize_body(target)`` returns a nonempty portable mapping
    only for an emission whose complete source body the evaluator can treat as
    atomic. Its policy must be versioned in ``code_contract_identity``. The
    runtime still executes the body and Ground, but suppresses internal Code
    observations and interface observers. The outer observer must cover native
    effects; the evaluator must cover hidden/mutable inputs and escaping writes,
    and supply an explicit rank transfer for the outer call. Code records the
    entered frozen definitions and Ground facts in the emission's evidence.
    This is not permission to skip workload execution or assume purity.
    """

    @property
    def code_contract_identity(self) -> Mapping[str, Any]: ...

    def resolve_emission(self, target: CodeCallTarget) -> EmissionBinding | None: ...


def checked_contract_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Copy portable contract configuration without calling user conversions."""

    if not isinstance(identity, Mapping) or not identity:
        raise PipelineContractError("Code emission contract requires a nonempty mapping identity")
    # safe_input admits only ordinary dict/MappingProxyType at every level;
    # custom objects cannot quietly turn themselves into persistent evidence.
    return dict(safe_input(identity))
