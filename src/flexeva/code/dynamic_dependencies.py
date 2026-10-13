"""Execution-local reaching definitions for dynamic Code occurrences.

This tracker proves primitive source bindings and actual producer instances.
Object aliasing, mutation and foreign hidden effects remain explicit open input
obligations. It neither serializes objects nor treats equal scalar values as
proof that two values came from the same producer.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import BuiltinFunctionType, CodeType, FunctionType, MethodType, ModuleType
from typing import Any

from ..errors import PipelineContractError
from ..model import stable_digest
from .inputs import DynamicInputEvidence, safe_input


def _code_value(value):
    """Immutable code constants only; no callable invocation or object repr."""
    if isinstance(value, CodeType):
        return {"bytecode": value.co_code.hex(), "exceptiontable": getattr(value, "co_exceptiontable", b"").hex(),
                "constants": tuple(_code_value(item) for item in value.co_consts),
                "names": value.co_names, "varnames": value.co_varnames,
                "freevars": value.co_freevars, "cellvars": value.co_cellvars,
                "arguments": (value.co_argcount, value.co_posonlyargcount, value.co_kwonlyargcount),
                "flags": value.co_flags}
    if type(value) is bytes:
        return {"bytes": value.hex()}
    if type(value) in (tuple, frozenset):
        items = tuple(_code_value(item) for item in value)
        return {"frozen_set": tuple(sorted(items, key=stable_digest))} if isinstance(value, frozenset) else items
    if value is Ellipsis:
        return {"literal": "Ellipsis"}
    if type(value) is complex:
        return {"complex": (safe_input(value.real), safe_input(value.imag))}
    return safe_input(value)


def _callable_input(function) -> Mapping[str, Any] | None:
    original = function.__func__ if isinstance(function, MethodType) else function
    if isinstance(original, FunctionType):
        def immutable(value):
            if type(value) is tuple:
                return tuple(immutable(item) for item in value)
            if value is None or type(value) in (str, bool, int, float):
                return safe_input(value)
            raise PipelineContractError("callable binding has opaque or mutable captured state")
        closure = tuple(immutable(cell.cell_contents) for cell in (original.__closure__ or ()))
        return {"module": safe_input(original.__module__), "qualname": safe_input(original.__qualname__),
                "code": stable_digest(_code_value(original.__code__)),
                "defaults": immutable(original.__defaults__), "closure": closure,
                "keyword_defaults": {key: immutable(value) for key, value in
                                     (original.__kwdefaults__ or {}).items()}}
    if isinstance(original, BuiltinFunctionType):
        return {"module": safe_input(original.__module__), "qualname": safe_input(original.__qualname__),
                "runtime": sys.version}
    return None


@dataclass
class _Origin:
    producers: set[tuple[str, str]] = field(default_factory=set)
    values: dict[str, Any] = field(default_factory=dict)
    unknown: set[str] = field(default_factory=set)

    def include(self, other: _Origin) -> None:
        self.producers.update(other.producers)
        self.values.update(other.values)
        self.unknown.update(other.unknown)


@dataclass
class _Call:
    caller: tuple[Any, ...]
    site: Mapping[str, Any]
    parameters: dict[str, _Origin]
    targets: tuple[str, ...] = ()
    token: Any = None
    return_origin: _Origin | None = None
    return_order: int = -1


class DynamicDependencyTracker:
    """Optional runtime sink hooks plus a conservative input-evidence provider.

    ``foreign_inputs_complete`` is an evaluator-owned proof callback, not a
    source option. A true result covers foreign effects only; it cannot repair
    unknown source bindings, aliases or operands reported by this tracker.
    ``foreign_callable_identity`` may describe an exact evaluator-bound callable
    object that the evaluator recognizes. It never overrides mutable/default
    closure failures of ordinary Python functions or unknown source inputs.
    """

    def __init__(self, *, foreign_inputs_complete=None, foreign_callable_identity=None) -> None:
        self.foreign_inputs_complete = foreign_inputs_complete
        self.foreign_callable_identity = foreign_callable_identity
        self._definitions: dict[tuple[Any, ...], dict[str, _Origin]] = {}
        self._controls: dict[tuple[Any, ...], str] = {}
        self._completed: dict[tuple[Any, ...], dict[str, tuple[int, Mapping[str, Any], _Origin]]] = {}
        self._last_block: dict[tuple[Any, ...], int] = {}
        self._calls: ContextVar[tuple[_Call, ...]] = ContextVar("dynamic_code_calls", default=())
        self._active: dict[tuple[Any, ...], _Call] = {}
        self._parents: dict[tuple[Any, ...], tuple[Any, ...]] = {}

    @staticmethod
    def _scope(lane, path, owner) -> tuple[Any, ...]:
        return (type(lane).__name__, lane, tuple(path), owner)

    def _occurrence_scope(self, occurrence) -> tuple[Any, ...]:
        return self._scope(occurrence.lane, occurrence.inputs.get("scope_path", ()),
                           occurrence.site.get("owner"))

    @staticmethod
    def _primitive(value) -> bool:
        # Containers can alias/mutate even when their current contents happen
        # to be scalars. Only immutable primitive values close a source binding.
        return value is None or type(value) in (str, bool, int, float)

    def _lookup(self, name: str, scope, bindings: Mapping[str, Any]) -> _Origin:
        definitions = self._definitions.get(scope, {})
        if name in definitions:
            return definitions[name]
        parent = self._parents.get(scope)
        if parent is not None and name in self._definitions.get(parent, {}):
            return self._definitions[parent][name]
        result = _Origin()
        if name in bindings and self._primitive(bindings[name]):
            try:
                result.values[name] = safe_input(bindings[name])
            except PipelineContractError:
                result.unknown.add(f"non-finite source input: {name}")
        elif name in bindings and isinstance(bindings[name], (ModuleType, FunctionType, MethodType)):
            # Actual call operands and foreign-effects coverage are handled
            # separately. A module's arbitrary attributes are never read here.
            result.unknown.add(f"callable/module binding is not value-complete: {name}")
        else:
            root = name.split(".", 1)[0]
            if root in definitions:
                result.include(definitions[root])
            result.unknown.add(f"source binding or alias is not value-complete: {name}")
        return result

    @staticmethod
    def _contains(outer, inner) -> bool:
        first, second = outer.get("span", {}), inner.get("span", {})
        if not first or not second or first.get("path") != second.get("path"):
            return False
        start = (first["start_line"], first["start_column"])
        end = (first["end_line"], first["end_column"])
        return start <= (second["start_line"], second["start_column"]) and (
            second["end_line"], second["end_column"]) <= end

    def __call__(self, occurrence) -> DynamicInputEvidence:
        inputs, site = occurrence.inputs, occurrence.site
        scope = self._occurrence_scope(occurrence)
        parent = site.get("lexical_parent")
        if parent and ".<" in site.get("owner", ""):
            self._parents[scope] = self._scope(occurrence.lane,
                                              inputs.get("scope_path", ()), parent)
        result = _Origin()
        if "scope_path" not in inputs or "source_bindings" not in inputs:
            result.unknown.add("runtime source-binding hooks are absent")
        bindings = inputs.get("source_bindings", {})
        uses = set(site.get("uses", ())) | set(site.get("access_paths", ()))
        # AST Name loads inside an Attribute are not independent reads of the
        # complete argument namespace. Keep its selected fields scoped.
        for name in tuple(uses):
            if "." not in name and any(other.startswith(name + ".") for other in uses):
                uses.discard(name)
        callee = site.get("callee") or ""
        if occurrence.kind == "emission" and callee:
            # A direct namespace lookup is covered by the actual callable plus
            # evaluator effects proof. Receiver objects still remain unknown.
            root = callee.split(".", 1)[0]
            if isinstance(bindings.get(root), (ModuleType, FunctionType)):
                uses.discard(root)
                uses.discard(callee)
        if occurrence.kind in ("block", "emission"):
            lower = self._last_block.get((scope, occurrence.site_id), -1)
            for order, producer_site, origin in self._completed.get(scope, {}).values():
                if order > lower and self._contains(site, producer_site):
                    result.include(origin)
                    # The completed call already owns the callable/effects
                    # obligation. Its residual assignment reads the result,
                    # not every attribute of the imported module again.
                    produced_callee = producer_site.get("callee") or ""
                    root = produced_callee.split(".", 1)[0]
                    if isinstance(bindings.get(root), (ModuleType, FunctionType)):
                        uses.discard(root)
                        uses.discard(produced_callee)
        for name in sorted(uses):
            result.include(self._lookup(name, scope, bindings))
        if scope in self._controls:
            result.producers.add((self._controls[scope], "$control"))
        if occurrence.kind == "emission":
            try:
                callable_input = _callable_input(inputs.get("callable"))
            except (PipelineContractError, ValueError):
                callable_input = None
            function = inputs.get("callable")
            if (callable_input is None and function is not None
                    and not isinstance(function, (FunctionType, MethodType, BuiltinFunctionType))
                    and self.foreign_callable_identity is not None):
                supplied = self.foreign_callable_identity(occurrence)
                if supplied is not None:
                    if not isinstance(supplied, Mapping) or not supplied:
                        raise PipelineContractError("evaluator callable identity requires a nonempty safe mapping")
                    callable_input = {"evaluator_bound_callable": safe_input(supplied)}
            if callable_input is None:
                result.unknown.add("actual callable implementation identity is not covered")
            else:
                result.values["$callable"] = callable_input
            for position, value in enumerate(inputs.get("args", ())):
                self._operand(result, f"$argument:{position}", value)
            for name, value in inputs.get("kwargs", {}).items():
                self._operand(result, f"$keyword:{name}", value)
            if inputs.get("receiver") is not None:
                self._operand(result, "$receiver", inputs["receiver"])
            if not (self.foreign_inputs_complete is not None
                    and self.foreign_inputs_complete(occurrence) is True):
                result.unknown.add("foreign hidden effects are not covered")
        if "completed_value" in inputs:
            self._operand(result, "$completed_value", inputs["completed_value"])
        if any("." in name or "[" in name for name in site.get("defs", ())):
            result.unknown.add("attribute/container write alias effects are not covered")
        return DynamicInputEvidence(
            producer_inputs=tuple(sorted(result.producers)), semantic_inputs=result.values,
            inputs_complete=not result.unknown, unresolved=tuple(sorted(result.unknown)),
        )

    def _operand(self, result: _Origin, name: str, value: Any) -> None:
        if self._primitive(value):
            try:
                result.values[name] = safe_input(value)
            except PipelineContractError:
                result.unknown.add(f"non-finite operand: {name}")
        else:
            result.unknown.add(f"opaque or aliasable operand: {name}")

    def complete(self, occurrence, *, result=None, error=None) -> None:
        if error is not None:
            return
        scope, site = self._occurrence_scope(occurrence), occurrence.site
        producer = _Origin(producers={(occurrence.instance_id, "$result")})
        # Actual completed expression instances are associated through source
        # containment, not equal values, pointer hashes, or predecessor guesses.
        self._completed.setdefault(scope, {})[occurrence.site_id] = (
            occurrence.lane_order, site, producer,
        )
        self._last_block[(scope, occurrence.site_id)] = occurrence.lane_order
        if occurrence.kind == "block":
            values = occurrence.inputs.get("defined_values", {})
            definitions = self._definitions.setdefault(scope, {})
            for name in site.get("defs", ()):
                origin = _Origin(producers={(occurrence.instance_id, name)})
                if name not in values or not self._primitive(values[name]):
                    origin.unknown.add(f"defined value alias effects are not covered: {name}")
                definitions[name] = origin
            role = site.get("role")
            if role in ("branch-test", "loop-test", "assert-test", "loop-next"):
                self._controls[scope] = occurrence.instance_id
            if role == "return" and scope in self._active:
                self._active[scope].return_origin = producer
                self._active[scope].return_order = occurrence.lane_order

    def bind_call(self, site, scope_path, lane, function, args, kwargs,
                  source_bindings):
        scope = self._scope(lane, scope_path, site.get("owner"))
        parameters: dict[str, _Origin] = {}
        original = function.__func__ if isinstance(function, MethodType) else function
        if isinstance(original, FunctionType):
            code = original.__code__
            names = code.co_varnames[:code.co_argcount]
            actual = tuple(args)
            argument_spec = site.get("argument_uses", {})
            argument_uses = argument_spec.get("positional", ())
            if isinstance(function, MethodType):
                actual = (function.__self__, *actual)
                argument_uses = ((site.get("callee", "").rsplit(".", 1)[0],), *argument_uses)
            for position, (name, value) in enumerate(zip(names, actual)):
                origin = _Origin()
                if position < len(argument_uses):
                    for symbol in argument_uses[position]:
                        origin.include(self._lookup(symbol, scope, source_bindings))
                else:
                    origin.unknown.add(f"call argument producer binding is missing: {name}")
                self._operand(origin, f"$parameter:{name}", value)
                parameters[name] = origin
            keyword_uses = argument_spec.get("keywords", {})
            for name, value in kwargs.items():
                origin = _Origin()
                if name in keyword_uses:
                    for symbol in keyword_uses[name]:
                        origin.include(self._lookup(symbol, scope, source_bindings))
                else:
                    origin.unknown.add(f"keyword producer binding is missing: {name}")
                self._operand(origin, f"$parameter:{name}", value)
                parameters[name] = origin
        targets = ()
        if isinstance(original, FunctionType):
            targets = (original.__module__ + "." + original.__qualname__.replace(".<locals>", ""),)
        call = _Call(scope, site, parameters, targets)
        call.token = self._calls.set((*self._calls.get(), call))
        return call

    def end_call(self, token: _Call, *, result=None, error=None) -> None:
        self._calls.reset(token.token)
        if error is None and token.return_origin is not None:
            # The return block is the real producer; retain caller call-site
            # containment so a subsequent assignment can consume that return.
            self._completed.setdefault(token.caller, {})[token.site.get("site_id", str(token.site.get("span")))] = (
                token.return_order, token.site, token.return_origin,
            )

    def activate(self, owner, scope_path, lane, parameters):
        scope = self._scope(lane, scope_path, owner)
        calls = self._calls.get()
        definitions = self._definitions.setdefault(scope, {})
        if calls and owner in calls[-1].targets:
            call = calls[-1]
            self._active[scope] = call
            for name in parameters:
                definitions[name] = call.parameters.get(name, _Origin(
                    unknown={f"actual parameter has no caller provenance: {name}"},
                ))
        return scope

    def deactivate(self, token) -> None:
        self._active.pop(token, None)
        self._definitions.pop(token, None)
        self._controls.pop(token, None)
        completed = self._completed.pop(token, {})
        for site_id in completed:
            self._last_block.pop((token, site_id), None)
        self._parents.pop(token, None)
