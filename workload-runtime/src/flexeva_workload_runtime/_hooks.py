"""Private source hooks for closed Code-window execution.

The hook base manages actual Python ownership, borrowed def-use notifications,
and explicit callback lifetimes. Subclasses own observation and completion;
there is no evaluator coverage, returned-state registry, or reuse decision here.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import PurePosixPath
from threading import RLock
from types import FunctionType, MappingProxyType
from typing import Any

from ._types import WorkloadExecutionError
from .dynamic_operations import OperationsMixin

SOURCE_PROTOCOL = "code-dynamic-v1"
_CURRENT_DISPATCHER: ContextVar[_SourceHooks | None] = ContextVar(
    "flexeva_current_dynamic_dispatcher", default=None,
)


def _identity(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise WorkloadExecutionError("dynamic identity requires JSON-safe scalar keys") from error
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DynamicOccurrence:
    """A transient source observation; ``inputs`` are borrowed, not a checkpoint.

    A sink/evaluator must turn consumed inputs into its own semantic fingerprints
    and managed records. The runtime never hashes, copies, or pickles tensors.
    ``lane_order`` is local execution order, never a coordinator's RPC order.
    """

    instance_id: str
    site_id: str
    kind: str
    lane: int | str
    region_id: str
    region_activation: Any
    region_occurrence: int
    activation_path: tuple[str, ...]
    occurrence_index: int
    lane_order: int
    parent_instance_id: str | None
    site: Mapping[str, Any]
    inputs: Mapping[str, Any]


@dataclass(frozen=True)
class DynamicExecutionContext:
    """Immutable, process-local logical callback context, not reusable state.

    No ticket, occurrence payload, tensor, Python frame, or ``Context`` is held
    here. Pending identities resolve to borrowed live objects inside the owning
    dispatcher. ``dispatcher_id`` is a lifetime guard, never a RAS fingerprint.
    The JSON string keeps mutable region activation keys out of the snapshot.
    """

    dispatcher_id: str
    lane: int | str
    region_id: str
    region_activation_json: str
    region_occurrence: int
    activation_path: tuple[str, ...]
    call_path: tuple[str, ...]
    pending_instance_ids: tuple[str, ...]


def current_execution_context() -> DynamicExecutionContext | None:
    """Export the context of the *current* invocation for an adapter bridge.

    An adapter may propagate this value through its execution engine, for
    example task-local storage. It must export anew for backward/recomputation
    rather than retaining a callback's forward-registration context.
    """
    dispatcher = _CURRENT_DISPATCHER.get()
    return None if dispatcher is None else dispatcher.export_context()


@dataclass(frozen=True)
class _Region:
    region_id: str
    key: Any
    occurrence: int


@dataclass
class _Pending:
    """One live emission's local completion, never reusable evaluator state."""

    occurrence: DynamicOccurrence
    completion_handle: Any
    result: Any = None
    body_summary: Any = None


class _SourceHooks(OperationsMixin):
    """Execute source hooks; subclasses define block and emission observations."""

    def __init__(
        self, sites: Mapping[str, Mapping[str, Any]], *, lane: int | str,
        sink: Any,
        owned_files: Mapping[str, str], region_id: str | None = None,
        region_filter: Callable[[str, Any], bool] | None = None,
        callback_context_provider: Callable[[str], DynamicExecutionContext | None] | None = None,
    ) -> None:
        if isinstance(lane, bool) or not isinstance(lane, (int, str)):
            raise WorkloadExecutionError("a dynamic lane must be int or str, excluding bool")
        if not callable(getattr(sink, "arrive", None)) or not callable(
            getattr(sink, "complete", None)
        ):
            raise WorkloadExecutionError("dynamic execution requires an arrive/complete sink")
        if callback_context_provider is not None and not callable(callback_context_provider):
            raise WorkloadExecutionError("callback context provider must be callable")
        self.sites: dict[str, Mapping[str, Any]] = {}
        for site_id, site in sites.items():
            if not isinstance(site_id, str) or not site_id or not isinstance(site, Mapping):
                raise WorkloadExecutionError("dynamic sites require nonempty IDs and descriptors")
            if site.get("kind") not in {"block", "emission", "call"}:
                raise WorkloadExecutionError(f"invalid dynamic site kind: {site_id!r}")
            self.sites[site_id] = MappingProxyType(dict(site))
        self.lane, self.sink = lane, sink
        self.region_id, self.region_filter = region_id, region_filter
        self.callback_context_provider = callback_context_provider
        self._dispatcher_id = uuid.uuid4().hex
        self._state_lock = RLock()
        self._live_regions: set[tuple[str, str, int]] = set()
        self._live_pending: dict[str, _Pending] = {}
        self._open_contexts = 0
        self._filenames = {f"<flexeva-frozen>/{path}" for path in owned_files}
        self._modules = set()
        for path in owned_files:
            parts = list(PurePosixPath(path).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            self._modules.add(".".join(parts))
        self._region: ContextVar[_Region | None] = ContextVar("flexeva_dynamic_region", default=None)
        self._path: ContextVar[tuple[str, ...]] = ContextVar("flexeva_dynamic_path", default=())
        self._calls: ContextVar[tuple[str, ...]] = ContextVar("flexeva_dynamic_calls", default=())
        self._pending: ContextVar[tuple[_Pending, ...]] = ContextVar(
            "flexeva_dynamic_pending", default=(),
        )
        self._counts: Counter = Counter()
        self._region_counts: Counter = Counter()
        self._lane_order = 0
        self._open_events = 0
        self._open_regions = 0
        self._closed = False

    @contextmanager
    def region(self, region_id: str, activation_key: Any = None) -> Iterator[None]:
        """Mark an actual configured region activation; outside code still runs."""
        if self._closed:
            raise WorkloadExecutionError("dynamic execution already finished")
        if not isinstance(region_id, str) or not region_id:
            raise WorkloadExecutionError("region identity must be a nonempty string")
        if self.region_id is not None and region_id != self.region_id:
            raise WorkloadExecutionError("source entered an undeclared mutable region")
        if self._region.get() is not None:
            raise WorkloadExecutionError("nested mutable regions require a separate explicit contract")
        key_digest = _identity(activation_key)
        counter_key = (region_id, key_digest)
        with self._state_lock:
            ordinal = self._region_counts[counter_key]
            self._region_counts[counter_key] += 1
        if self.region_filter is not None and not self.region_filter(region_id, activation_key):
            yield
            return
        region_key = (region_id, key_digest, ordinal)
        with self._state_lock:
            if self._closed:
                raise WorkloadExecutionError("dynamic execution already finished")
            self._open_regions += 1
            self._live_regions.add(region_key)
        token = self._region.set(_Region(region_id, activation_key, ordinal))
        path_token = self._path.set(())
        calls_token = self._calls.set(())
        dispatcher_token = _CURRENT_DISPATCHER.set(self)
        try:
            yield
        finally:
            with self._state_lock:
                self._open_regions -= 1
                self._live_regions.remove(region_key)
            _CURRENT_DISPATCHER.reset(dispatcher_token)
            self._calls.reset(calls_token)
            self._path.reset(path_token)
            self._region.reset(token)

    def export_context(self) -> DynamicExecutionContext | None:
        """Snapshot this execution context without borrowing its mutable objects."""
        region = self._region.get()
        if region is None:
            return None
        snapshot = DynamicExecutionContext(
            dispatcher_id=self._dispatcher_id, lane=self.lane,
            region_id=region.region_id,
            region_activation_json=json.dumps(
                region.key, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ),
            region_occurrence=region.occurrence,
            activation_path=self._path.get(), call_path=self._calls.get(),
            pending_instance_ids=tuple(
                pending.occurrence.instance_id for pending in self._pending.get()
            ),
        )
        with self._state_lock:
            self._context_bindings(snapshot)
        return snapshot

    def _context_bindings(self, snapshot: DynamicExecutionContext) -> tuple[_Region, tuple[_Pending, ...]]:
        """Validate and resolve live bindings while holding ``_state_lock``."""
        if not isinstance(snapshot, DynamicExecutionContext):
            raise WorkloadExecutionError("callback provider returned an invalid execution context")
        if self._closed or snapshot.dispatcher_id != self._dispatcher_id:
            raise WorkloadExecutionError("callback execution context belongs to a different or closed dispatcher")
        if type(snapshot.lane) is not type(self.lane) or snapshot.lane != self.lane:
            raise WorkloadExecutionError("callback execution context belongs to a different lane")
        try:
            key = json.loads(snapshot.region_activation_json)
        except (TypeError, ValueError) as error:
            raise WorkloadExecutionError("callback execution context has an invalid region key") from error
        region_key = (snapshot.region_id, _identity(key), snapshot.region_occurrence)
        if region_key not in self._live_regions:
            raise WorkloadExecutionError("callback execution context belongs to an inactive region")
        try:
            pending = tuple(self._live_pending[event_id] for event_id in snapshot.pending_instance_ids)
        except KeyError as error:
            raise WorkloadExecutionError("callback execution context references a completed emission") from error
        return _Region(snapshot.region_id, key, snapshot.region_occurrence), pending

    @contextmanager
    def context_scope(self, snapshot: DynamicExecutionContext) -> Iterator[None]:
        """Restore logical bindings for one callback, independently per context.

        This does not reuse ``contextvars.Context.run`` across threads and never
        copies live evaluator values. Callers must join their callback work
        before finalizing the execution. Stale event/region bindings fail closed.
        """
        with self._state_lock:
            region, pending = self._context_bindings(snapshot)
            self._open_contexts += 1
        region_token = self._region.set(region)
        path_token = self._path.set(snapshot.activation_path)
        calls_token = self._calls.set(snapshot.call_path)
        pending_token = self._pending.set(pending)
        dispatcher_token = _CURRENT_DISPATCHER.set(self)
        try:
            yield
        finally:
            _CURRENT_DISPATCHER.reset(dispatcher_token)
            self._pending.reset(pending_token)
            self._calls.reset(calls_token)
            self._path.reset(path_token)
            self._region.reset(region_token)
            with self._state_lock:
                self._open_contexts -= 1

    def _activate_sink(self, owner, parameters):
        summary = self.body_summary
        if summary is not None:
            with self._state_lock:
                summary["owners"].add(owner)
            return None
        activate = getattr(self.sink, "activate", None)
        if activate is None:
            return None
        if parameters is None:
            frame = self._source_frame()
            try:
                if frame is None:
                    parameters = {}
                else:
                    values = inspect.getargvalues(frame)
                    names = [*values.args, *([values.varargs] if values.varargs else []),
                             *([values.keywords] if values.keywords else [])]
                    parameters = {name: values.locals[name] for name in names}
            finally:
                del frame
        # Frame locals and parameter objects die with this helper, before yield.
        return activate(owner, self._path.get(), self.lane, parameters)

    @contextmanager
    def activate(self, owner: str, parameters=None) -> Iterator[None]:
        if not isinstance(owner, str) or not owner:
            raise WorkloadExecutionError("activation owner must be a nonempty string")
        region = self._region.get()
        if region is None:
            snapshot = (self.callback_context_provider(owner)
                        if self.callback_context_provider is not None else None)
            if snapshot is not None:
                with self.context_scope(snapshot), self.activate(owner, parameters):
                    del parameters
                    yield
                return
            with self._state_lock:
                active = bool(self._open_regions)
            if active:
                raise WorkloadExecutionError("owned callback has no propagated execution context")
            yield
            return
        parent, calls = self._path.get(), self._calls.get()
        key = (region.region_id, _identity(region.key), region.occurrence, parent, calls, owner)
        with self._state_lock:
            ordinal = self._counts[key]
            self._counts[key] += 1
        component = "activation:" + _identity((owner, calls, ordinal))[:24]
        token = self._path.set((*parent, component))
        activation_token = None
        summarized = self.body_summary is not None
        try:
            activation_token = self._activate_sink(owner, parameters)
            del parameters
            yield
        finally:
            try:
                deactivate = getattr(self.sink, "deactivate", None)
                if deactivate is not None and not summarized:
                    if sys.exc_info()[0] is not None:
                        with suppress(BaseException):
                            deactivate(activation_token)
                    else:
                        deactivate(activation_token)
            finally:
                self._path.reset(token)

    def _source_frame(self):
        """Find the immediate owned caller without building/formating a stack."""
        frame = inspect.currentframe()
        try:
            frame = frame.f_back
            while frame is not None and frame.f_code.co_filename not in self._filenames:
                frame = frame.f_back
            return frame
        finally:
            del frame

    def _bindings(self, site: Mapping[str, Any], *, definitions: bool = False) -> dict[str, Any]:
        fields = ("defs",) if definitions else ("uses", "defs")
        names = {
            name.split(".", 1)[0] for field in fields for name in site.get(field, ())
            if isinstance(name, str) and name.split(".", 1)[0].isidentifier()
        }
        frame = self._source_frame()
        try:
            if frame is None:
                return {}
            local, global_ = frame.f_locals, frame.f_globals
            return {name: local[name] if name in local else global_[name]
                    for name in names if name in local or name in global_}
        finally:
            del frame

    def _site(self, site_id: str) -> Mapping[str, Any]:
        try:
            return self.sites[site_id]
        except KeyError as error:
            raise WorkloadExecutionError(f"unknown dynamic site: {site_id!r}") from error

    @contextmanager
    def owner(self, site_id: str) -> Iterator[None]:
        """Track a source-owned AST scope without declaring an emission."""
        if not self.capture_active:
            yield
            return
        self._site(site_id)
        token = self._calls.set((*self._calls.get(), site_id))
        try:
            yield
        finally:
            self._calls.reset(token)

    def _source_owned(self, function: Callable[..., Any]) -> bool:
        resolver = getattr(self, "source_callable", None)
        if resolver is not None:
            function = resolver(function) or function
        if isinstance(function, type):
            return inspect.getattr_static(function, "__module__", None) in self._modules
        actual = function.__func__ if inspect.ismethod(function) else function
        # Only unwrap our own transparent instrumentation wrapper. Unwrapping an
        # arbitrary library decorator would incorrectly hide its external code.
        if isinstance(actual, FunctionType):
            actual = actual.__dict__.get("__flexeva_generator_source__", actual)
        code = actual.__code__ if isinstance(actual, FunctionType) else None
        if code is None:
            actual = inspect.getattr_static(type(function), "__call__", None)
            code = actual.__code__ if isinstance(actual, FunctionType) else None
        return code is not None and code.co_filename in self._filenames

    def invoke(self, site_id: str, function: Callable[..., Any], /,
               *args: Any, **kwargs: Any) -> Any:
        """Invoke already-evaluated operands exactly once under actual ownership."""
        if self._region.get() is None:
            return function(*args, **kwargs)
        site = self._site(site_id)
        if site["kind"] not in {"call", "emission"}:
            raise WorkloadExecutionError("invoke hook requires a call or emission site")
        token = self._calls.set((*self._calls.get(), site_id))
        try:
            owned = self._source_owned(function)

            def call():
                if not owned or self.body_summary is not None:
                    return function(*args, **kwargs)
                bind_call = getattr(self.sink, "bind_call", None)
                call_token = (
                    bind_call(site, self._path.get(), self.lane,
                              (getattr(self, "source_callable", None)(function) or function)
                              if getattr(self, "source_callable", None) else function, args, kwargs,
                              self._bindings(site)) if bind_call is not None else None
                )
                end_call = getattr(self.sink, "end_call", None)
                try:
                    value = function(*args, **kwargs)
                except BaseException as error:
                    if end_call is not None:
                        with suppress(BaseException):
                            end_call(call_token, result=None, error=error)
                    raise
                else:
                    if end_call is not None:
                        end_call(call_token, result=value, error=None)
                    return value

            inputs = {
                "args": args, "kwargs": MappingProxyType(dict(kwargs)),
                "callable": function,
                "receiver": function.__self__ if inspect.ismethod(function)
                or inspect.isbuiltin(function) else None,
                "source_owned": owned,
            }
            return self._perform(site_id, "emission", call, inputs)
        finally:
            self._calls.reset(token)

    def generator(self, owner: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Bind each resume, never hold a source activation across ``yield``.

        The proxy preserves next/send/throw/close, not concrete generator type or
        ``inspect.isgeneratorfunction``. Source that needs those identities must
        not be admitted by the exporter under this protocol.
        """
        def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
            if not inspect.isgeneratorfunction(function):
                raise WorkloadExecutionError("generator hook requires a generator function")

            @wraps(function)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                bound = inspect.signature(function).bind(*args, **kwargs)
                bound.apply_defaults()
                return _GeneratorProxy(self, owner, function(*args, **kwargs), bound.arguments)

            wrapped.__flexeva_generator_source__ = function
            return wrapped
        return decorate

    def yielded(self, site_id: str, value: Any) -> Any:
        """Report the evaluated yield transfer without wrapping its lexical scope."""
        self.block(site_id)
        return value

    def completed(self, site_id: str, value: Any) -> Any:
        """Report an already-evaluated residual/return expression with its value."""
        return self._perform(site_id, "block", lambda: value)

    def delegated(self, site_id: str, iterable: Any) -> Any:
        """Retain Python's native yield-from send/throw/close delegation."""
        # A native yield-from is deliberately used rather than a next-only proxy:
        # it forwards send/throw/close and propagates StopIteration.value.
        def delegate():
            result = yield from iterable
            self.completed(site_id, result)
            return result
        return delegate()

    @property
    def body_summary(self):
        """A live enclosing summary, including explicitly propagated callbacks."""
        for pending in reversed(self._pending.get()):
            if pending.body_summary is not None:
                return pending.body_summary
        return None

    @property
    def capture_active(self) -> bool:
        """Whether this context is inside an evaluation-selected source region."""
        return self._region.get() is not None

class _GeneratorProxy:
    def __init__(self, dispatcher: _SourceHooks, owner: str, generator: Any,
                 parameters: Mapping[str, Any]) -> None:
        self._dispatcher, self._owner, self._generator = dispatcher, owner, generator
        self._parameters = parameters

    def __iter__(self):
        return self

    def __next__(self):
        with self._dispatcher.activate(self._owner, self._parameters):
            return next(self._generator)

    def send(self, value):
        with self._dispatcher.activate(self._owner, self._parameters):
            return self._generator.send(value)

    def throw(self, *args):
        with self._dispatcher.activate(self._owner, self._parameters):
            return self._generator.throw(*args)

    def close(self):
        with self._dispatcher.activate(self._owner, self._parameters):
            return self._generator.close()
