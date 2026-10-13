"""Extensible rank-taint rules, independent of emissions and reuse evidence.

Rules cover rank-related return/selected-field propagation. They must not
invoke the workload, mutate or retain borrowed operands. Version every
semantic rule change in its portable identity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..errors import PipelineContractError
from .inputs import safe_input


@dataclass(frozen=True)
class TaintObservation:
    """Borrowed operation; result is valid only when completed is true.

    ``operation='observed-value'`` requests a read-only fallback projection for
    an already evaluated rank-relevant value (definition, argument, loop binding
    or return). Its callable is None, args contains that value, and result is
    the same value with completed=True. A matching rule supplies
    ``projections['result']``; it cannot change the previously established taint.
    The observer must neither evaluate the source again nor retain the value.
    """

    site: Mapping[str, Any]
    lane: int | str
    callable: Any
    operation: str | None
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    argument_taints: tuple[bool, ...]
    keyword_taints: Mapping[str, bool]
    receiver_tainted: bool
    control_tainted: bool
    result: Any
    completed: bool

    @property
    def input_tainted(self):
        return self.receiver_tainted or any(self.argument_taints) or any(self.keyword_taints.values())


@dataclass(frozen=True)
class TaintTransfer:
    """A rank transfer, not proof of purity, complete inputs or reuse.

    writes names source-visible access paths; the rule must cover aliases
    affecting Lane. projections supplies portable rank-relevant views of opaque
    values under arg:N/kw:name/result, not evaluator replay checkpoints.
    """

    return_tainted: bool
    writes: Mapping[str, bool] = field(default_factory=dict)
    projections: Mapping[str, Any] = field(default_factory=dict)

    def checked(self):
        if type(self.return_tainted) is not bool or any(
            type(key) is not str or not key or type(value) is not bool for key, value in self.writes.items()
        ):
            raise PipelineContractError("taint transfer requires explicit Boolean return/field influences")
        return TaintTransfer(self.return_tainted, safe_input(self.writes), safe_input(self.projections))


@dataclass(frozen=True)
class CodeTaintRule:
    identity: Mapping[str, Any]
    resolve: Callable[[TaintObservation], TaintTransfer | None]


@dataclass(frozen=True)
class CodeTaintPolicy:
    sources: tuple[str, ...] = ("rank", "rank_id")
    rules: tuple[CodeTaintRule, ...] = ()

    @property
    def analysis_identity(self):
        if (not isinstance(self.sources, tuple)
                or any(type(name) is not str or not name.isidentifier() for name in self.sources)
                or len(set(self.sources)) != len(self.sources)
                or not isinstance(self.rules, tuple)):
            raise PipelineContractError("taint policy requires unique source names and an immutable rule tuple")
        identities = []
        for rule in self.rules:
            if not isinstance(rule, CodeTaintRule) or not callable(rule.resolve) or not rule.identity:
                raise PipelineContractError("taint rules require a resolver and nonempty versioned identity")
            identities.append(safe_input(rule.identity))
        return {"analysis": "cfg-taint-v1", "defaults": "scalar-python-v1",
                "sources": sorted(self.sources), "rules": identities}

    def resolve(self, observation):
        matches = []
        for rule in self.rules:
            result = rule.resolve(observation)
            if result is not None:
                if not isinstance(result, TaintTransfer):
                    raise PipelineContractError("taint rule must return TaintTransfer or None")
                matches.append(result.checked())
        if len(matches) > 1:
            raise PipelineContractError("multiple taint rules cover the same operation")
        return matches[0] if matches else None


def checked_analysis_identity(value):
    value = safe_input(value)
    if (type(value) is not dict or set(value) != {"analysis", "defaults", "sources", "rules"}
            or value["analysis"] != "cfg-taint-v1" or value["defaults"] != "scalar-python-v1"
            or type(value["sources"]) is not tuple or type(value["rules"]) is not tuple
            or any(type(name) is not str or not name.isidentifier() for name in value["sources"])
            or tuple(sorted(set(value["sources"]))) != value["sources"]
            or any(type(rule) is not dict or not rule for rule in value["rules"])):
        raise PipelineContractError("Code requires an explicit cfg-taint-v1 analysis identity")
    return value
