"""Execute a closed Code window under developer-declared emission bindings.

Source hooks are observations of real execution, not RAS event boundaries. The
sink aggregates host observations; only a resolved emission closes their block.
This module knows neither Core state nor evaluator reuse decisions.
"""

from __future__ import annotations

import json
import math
import operator
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import Any

from ._hooks import SOURCE_PROTOCOL, DynamicOccurrence, _identity, _Pending, _SourceHooks
from ._loading import _run_frozen_source
from ._types import WorkloadExecutionError, WorkloadInvocation, WorkloadRunResult
from .dynamic_operations import _BINARY, _COMPARE, _INPLACE, _UNARY

PROTOCOL_VERSION = "code-window-v2"

# These name the outer Python protocol operation, not a guessed reflected slot.
_OPERATIONS = {
    **{name: value[0] for name, value in _BINARY.items()},
    **{name: value[0] for name, value in _COMPARE.items()},
    **_INPLACE, **_UNARY,
    "truth": bool, "not": operator.not_, "is": operator.is_, "is_not": operator.is_not,
    "in": operator.contains, "not_in": operator.contains,
    "getitem": operator.getitem, "setitem_rhs_first": operator.setitem,
    "getattr": getattr, "setattr_rhs_first": setattr,
    "iter": iter, "next": next,
}


@dataclass(frozen=True)
class CodeCallTarget:
    """Transient structural target consumed by an external emission resolver.

    ``operation`` and ``args`` use the source protocol's operand order (including
    RHS-first stores). ``callable`` identifies that protocol's outer operation;
    the resolver must not invoke it or its operands to discover dispatch.
    ``source_owned`` identifies explicit calls into the frozen source. Binding
    such a call does not suppress its body hooks or nested interface resolution.
    """

    site: Mapping[str, Any]
    lane: int | str
    callable: Callable[..., Any]
    receiver: Any
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    operation: str | None = None
    source_owned: bool = False


@dataclass(frozen=True)
class MissingValue:
    """Backend-declared scalar/JSON fact; the original operation still executes."""

    query: Any
    inputs: Any
    value_type: str = "number"


@dataclass(frozen=True)
class CodeControlPoint:
    point_id: str
    lane: int | str
    site_id: str


def _json_value(value):
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) in (tuple, list):
        return all(_json_value(item) for item in value)
    return (type(value) is dict and all(type(key) is str for key in value)
            and all(_json_value(item) for item in value.values()))


class CodeWindowDispatcher(_SourceHooks):
    """Reuse source/protocol hooks with bounded, closed-window observations."""

    def __init__(self, sites, *, lane, sink, resolve_emission, owned_files,
                 region_id=None, region_filter=None, callback_context_provider=None,
                 interface_observer=None, missing_value=None, complete_value=None,
                 region_observer=None, materialize_value=None, summarize_body=None,
                 source_callable=None):
        if not callable(resolve_emission):
            raise WorkloadExecutionError("a Code window requires an emission resolver")
        if interface_observer is not None and not callable(interface_observer):
            raise WorkloadExecutionError("interface observer must be callable")
        for callback in (region_observer, materialize_value, summarize_body):
            if callback is not None and not callable(callback):
                raise WorkloadExecutionError("region observer and value materializer must be callable")
        for name in ("observe_block", "flush_block"):
            if not callable(getattr(sink, name, None)):
                raise WorkloadExecutionError(f"Code window sink requires {name}")
        super().__init__(
            sites, lane=lane, sink=sink, owned_files=owned_files,
            region_id=region_id, region_filter=region_filter,
            callback_context_provider=callback_context_provider,
        )
        self.resolve_emission = resolve_emission
        self.source_callable = source_callable
        self.interface_observer = interface_observer
        self.missing_value, self.complete_value = missing_value, complete_value
        self.region_observer, self.materialize_value = region_observer, materialize_value
        self.summarize_body = summarize_body
        self._value_visits = Counter()
        self._block_identity: tuple[str, int] | None = None
        self._event_order = 0
        self._summary: Counter[str] = Counter()
        self._selected_region_json: str | None = None

    def _reserve(self, kind: str) -> tuple[str, int]:
        region = self._region.get()
        if region is None:
            raise WorkloadExecutionError("a Code event requires an active region")
        order = self._event_order
        self._event_order += 1
        identity = (type(self.lane).__name__, self.lane, region.region_id,
                    region.key, region.occurrence, order, kind)
        self._summary[kind + "_count"] += 1
        return "code-window:" + _identity(identity), order

    def _observation(self, site_id: str, kind: str, identity: tuple[str, int],
                     inputs: Mapping[str, Any] | None = None) -> DynamicOccurrence:
        region = self._region.get()
        if region is None:
            raise WorkloadExecutionError("a Code event requires an active region")
        site = self._site(site_id)
        order = self._lane_order
        self._lane_order += 1
        pending = self._pending.get()
        live = {
            "scope_path": self._path.get(), "source_bindings": self._bindings(site),
            **dict(inputs or {}), "event_order": identity[1],
        }
        if kind == "block":
            live["defined_values"] = self._bindings(site, definitions=True)
        return DynamicOccurrence(
            instance_id=identity[0], site_id=site_id, kind=kind, lane=self.lane,
            region_id=region.region_id, region_activation=region.key,
            region_occurrence=region.occurrence,
            activation_path=(*self._path.get(), *self._calls.get()),
            occurrence_index=identity[1], lane_order=order,
            parent_instance_id=pending[-1].occurrence.instance_id if pending else None,
            site=site, inputs=MappingProxyType(live),
        )

    def _observe_block(self, site_id: str, inputs=None, *, result=None) -> None:
        if self.body_summary is not None:
            return
        with self._state_lock:
            if self._block_identity is None:
                self._block_identity = self._reserve("block")
            occurrence = self._observation(site_id, "block", self._block_identity, inputs)
            self.sink.observe_block(occurrence, result=result)
            self._summary["host_step_count"] += 1

    def _flush_block(self) -> None:
        with self._state_lock:
            if self._block_identity is not None:
                self.sink.flush_block()
                self._block_identity = None

    def _region_scope(self):
        if self.region_observer is None:
            return nullcontext()
        frame = self._source_frame()
        try:
            bindings = MappingProxyType(dict(frame.f_locals)) if frame is not None else MappingProxyType({})
        finally:
            del frame
        # The observer must freeze consumed values here, not retain these borrowed locals.
        return self.region_observer(self.export_context(), bindings)

    @contextmanager
    def region(self, region_id: str, activation_key: Any = None) -> Iterator[None]:
        try:
            with super().region(region_id, activation_key):
                if self.capture_active:
                    with self._state_lock:
                        if self._summary["region_count"]:
                            raise WorkloadExecutionError(
                                "Code window requires one actual region activation per lane; "
                                "select one activation with region_filter"
                            )
                        region = self._region.get()
                        self._selected_region_json = json.dumps({
                            "region_id": region.region_id,
                            "activation": region.key,
                            "occurrence": region.occurrence,
                        }, sort_keys=True, allow_nan=False)
                        self._summary["region_count"] += 1
                if not self.capture_active:
                    yield
                    return
                completed = False
                with self._region_scope():
                    yield
                    self._flush_block()
                    completed = True
                if not completed:
                    raise WorkloadExecutionError("region observer suppressed a workload failure")
        except BaseException:
            with self._state_lock:
                self._summary["failed_region_count"] += 1
            raise

    @contextmanager
    def _event(self, site_id: str, kind: str, inputs=None, *, body_summary=None):
        if kind != "emission" or not inputs or "emission_binding" not in inputs:
            raise WorkloadExecutionError("a Code emission requires a resolved callable binding")
        with self._state_lock:
            self._flush_block()
            occurrence = self._observation(site_id, kind, self._reserve(kind), inputs)
            pending = _Pending(occurrence, self.sink.arrive(occurrence))
            if body_summary is not None:
                pending.body_summary = {"contract": body_summary, "owners": set()}
            token = self._pending.set((*self._pending.get(), pending))
            self._live_pending[occurrence.instance_id] = pending
            self._open_events += 1
        try:
            yield pending
        except BaseException as error:
            with self._state_lock:
                self._summary["failed_emission_count"] += 1
            with suppress(BaseException):
                self.sink.complete(pending.completion_handle, occurrence, result=None, error=error)
            raise
        else:
            if pending.body_summary is not None:
                self.sink.record_body_summary(pending.completion_handle, pending.body_summary)
            self.sink.complete(pending.completion_handle, occurrence, result=pending.result, error=None)
            with self._state_lock:
                self._summary["completed_emission_count"] += 1
        finally:
            with self._state_lock:
                self._open_events -= 1
                del self._live_pending[occurrence.instance_id]
            self._pending.reset(token)

    def _target(self, site_id, inputs):
        operation = inputs.get("operation", self._site(site_id).get("operation"))
        function = inputs.get("callable")
        args = tuple(inputs.get("args", ()))
        receiver = inputs.get("receiver")
        if function is None and operation in _OPERATIONS:
            function = _OPERATIONS[operation]
            receiver = (args[1] if operation in ("setitem_rhs_first", "setattr_rhs_first", "in", "not_in")
                        else args[0]) if args else None
        if function is None:
            return None
        return CodeCallTarget(
            self._site(site_id), self.lane, function, receiver, args,
            MappingProxyType(dict(inputs.get("kwargs", {}))), operation,
            inputs.get("source_owned", False),
        )

    def _binding(self, target):
        if target is None:
            return None
        binding = self.resolve_emission(target)
        if binding is None:
            return None
        result = {name: getattr(binding, name, None) for name in ("transform_id", "operation_id")}
        if any(type(value) is not str or not value for value in result.values()):
            raise WorkloadExecutionError("emission binding requires nonempty transform_id and operation_id")
        return result

    def _execute_interface(self, target, continuation):
        """Borrow the target for one scope; only the runtime executes the call.

        Observers neither receive a continuation nor select a return value.
        A context manager cannot silently suppress a workload failure.
        """
        if target is None or self.interface_observer is None:
            return continuation()
        completed = False
        with self.interface_observer(target):
            value = continuation()
            completed = True
        if not completed:
            raise WorkloadExecutionError("interface observer suppressed a workload failure")
        return value

    def operation(self, site_id, op_name, /, *operands, **kwargs):
        if self.body_summary is None or op_name not in _OPERATIONS or op_name in {"iter", "next"} or kwargs:
            return super().operation(site_id, op_name, *operands, **kwargs)
        # Python's operator performs the real reflected/in-place dispatch.
        # An atomic body needs no separate classification of its type slots.
        function, actual = _OPERATIONS[op_name], operands
        if op_name in {"setitem_rhs_first", "setattr_rhs_first"}:
            actual = (operands[1], operands[2], operands[0])
        elif op_name in {"in", "not_in"}:
            actual = (operands[1], operands[0])
        def execute():
            value = function(*actual)
            return not value if op_name == "not_in" else value
        return self._perform(site_id, "emission", execute, {"operation": op_name, "args": operands})

    def _perform(self, site_id, kind, continuation, inputs=None):
        if not self.capture_active or kind == "call":
            return continuation()
        if kind not in {"block", "emission"}:
            raise WorkloadExecutionError(f"invalid executed operation kind: {kind!r}")
        self._site(site_id)
        live = dict(inputs or {})
        target = self._target(site_id, live)
        summarized = self.body_summary is not None
        binding = None if summarized else self._binding(target)
        summary = self.summarize_body(target) if binding is not None and self.summarize_body else None
        if summary is not None:
            if (type(summary) is not dict or not summary or not _json_value(summary)
                    or not callable(getattr(self.sink, "record_body_summary", None))):
                raise WorkloadExecutionError("body summary requires a nonempty portable contract and a summary sink")
            summary = json.loads(json.dumps(summary, allow_nan=False))
        missing = self.missing_value(target) if target is not None and self.missing_value else None
        point = None
        if missing is not None:
            if not isinstance(missing, MissingValue) or not callable(self.complete_value):
                raise WorkloadExecutionError("declared missing value requires a completion service")
            region = self._region.get()
            identity = (type(self.lane).__name__, self.lane, site_id, self._path.get(), self._calls.get(),
                        region.region_id, region.key, region.occurrence)
            key = _identity(identity)
            visit = self._value_visits[key]
            self._value_visits[key] += 1
            point = CodeControlPoint("code-point:" + _identity((identity, visit)), self.lane, site_id)

        def execute():
            value = continuation() if summarized else self._execute_interface(target, continuation)
            # The observer has closed; the original call succeeded exactly once.
            if missing is not None:
                original = value
                value = self.complete_value(point, missing)
                valid = {"number": type(value) in (int, float) and math.isfinite(value),
                         "bool": type(value) is bool, "int": type(value) is int,
                         "str": type(value) is str,
                         "json": _json_value(value)}.get(missing.value_type, False)
                if not valid:
                    raise WorkloadExecutionError("Ground completion returned an invalid or missing scalar/JSON fact")
                if self.materialize_value is not None:
                    value = self.materialize_value(target, missing, value, original)
            return value
        if binding is not None:
            live["emission_binding"] = binding
            if summary is not None:
                live["body_summary"] = summary
            with self._event(site_id, "emission", live, body_summary=summary) as pending:
                pending.result = execute()
                return pending.result
        value = execute()
        # Owned body/return hooks already describe this host call's producers.
        if not summarized and (target is None or not target.source_owned):
            self._observe_block(site_id, {**live, "completed_value": value}, result=value)
        return value

    def block(self, site_id, inputs=None) -> None:
        if self.capture_active:
            if self._site(site_id)["kind"] != "block":
                raise WorkloadExecutionError("block hook requires a block site")
            self._observe_block(site_id, inputs)

    def predicate(self, site_id, value):
        result = self._perform(site_id, self._truth_kind(value), lambda: bool(value),
                               {"operation": "truth", "args": (value,)})
        if type(result) is not bool:
            raise WorkloadExecutionError("predicate operation must return a bool")
        # The host branch decision remains in the following block even when a
        # developer explicitly binds an otherwise builtin truth operation.
        self._perform(site_id, "block", lambda: result,
                      {"operation": "predicate", "args": (result,)})
        return result

    def finish(self) -> dict[str, Any]:
        with self._state_lock:
            if self._open_events or self._open_regions or self._open_contexts:
                raise WorkloadExecutionError("Code window ended with open source scopes")
            if self._block_identity is not None or self._summary["failed_region_count"]:
                raise WorkloadExecutionError("Code window did not close successfully")
            self._closed = True
            return {
                "protocol_version": PROTOCOL_VERSION, "measured": True, "closed": True,
                "event_count": self._event_order,
                "region_count": self._summary["region_count"],
                "selected_region": (json.loads(self._selected_region_json)
                                    if self._selected_region_json is not None else None),
                **{key: self._summary[key] for key in (
                    "block_count", "emission_count", "host_step_count",
                    "completed_emission_count", "failed_emission_count",
                )},
            }


def run_code_window_workload(
    envelope: Mapping[str, Any], invocation: WorkloadInvocation, *, sink: Any,
    resolve_emission: Callable[[CodeCallTarget], Any],
    module_bindings: Mapping[str, ModuleType] | None = None,
    region_filter: Callable[[str, Any], bool] | None = None,
    callback_context_provider: Callable[[str], Any] | None = None,
    interface_observer: Callable[[CodeCallTarget], AbstractContextManager[Any]] | None = None,
    missing_value: Callable[[CodeCallTarget], MissingValue | None] | None = None,
    complete_value: Callable[[CodeControlPoint, MissingValue], Any] | None = None,
    region_observer: Callable[..., AbstractContextManager[Any]] | None = None,
    materialize_value: Callable[[CodeCallTarget, MissingValue, Any, Any], Any] | None = None,
    summarize_body: Callable[[CodeCallTarget], Mapping[str, Any] | None] | None = None,
    source_callable: Callable[[Any], Any] | None = None,
) -> WorkloadRunResult:
    """Execute original calls once and return only a closed-window count audit."""
    if envelope.get("protocol_version") != SOURCE_PROTOCOL:
        raise WorkloadExecutionError("Code windows require code-dynamic-v1 source instrumentation")
    if invocation.mode not in {"instrumented", "original"}:
        raise WorkloadExecutionError(f"unknown workload mode: {invocation.mode!r}")
    hook_name = envelope.get("hook_name", "__flexeva_dynamic__")
    if not isinstance(hook_name, str) or not hook_name.isidentifier():
        raise WorkloadExecutionError("dynamic hook must be a Python identifier")
    dispatcher = CodeWindowDispatcher(
        envelope["sites"], lane=invocation.lane, sink=sink, resolve_emission=resolve_emission,
        owned_files=envelope["source"]["files"], region_id=envelope.get("region_id"),
        region_filter=region_filter, callback_context_provider=callback_context_provider,
        interface_observer=interface_observer,
        missing_value=missing_value, complete_value=complete_value,
        region_observer=region_observer, materialize_value=materialize_value,
        summarize_body=summarize_body, source_callable=source_callable,
    )
    return _run_frozen_source(envelope, invocation, dispatcher, module_bindings=module_bindings)
