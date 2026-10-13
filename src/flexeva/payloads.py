"""Internal backing-state plumbing; only backend Payload interprets a locator.

There is no decoder registry, write-through view or mutable event API here.
"""

import json
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import fields, is_dataclass
from functools import lru_cache

from .errors import PipelineContractError, StateStoreContractError
from .model import (
    BackendState,
    BackendStateRecord,
    EventPayload,
    PayloadBinding,
    StateRef,
    _canonical,
)


@lru_cache(maxsize=128)
def _field_names(kind):
    return tuple(sorted(field.name for field in fields(kind)))


def _input_snapshot(value):
    """Immutable canonical structure for callback guards, not a content address."""
    if value is None or type(value) is bool:
        return value
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, int):
        return int, int.__int__(value)
    if isinstance(value, float):
        # Preserve int/float and signed-zero distinctions; NaN stays comparable.
        return float, float.__repr__(value)
    if is_dataclass(value) and not isinstance(value, type):
        names = _field_names(type(value))
        return dict, names, tuple(_input_snapshot(getattr(value, name)) for name in names)
    if isinstance(value, Mapping):
        # Canonical string keys can collide: keep the original last-value rule.
        normalized = {str(key): _input_snapshot(item)
                      for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        return dict, tuple(normalized), tuple(normalized.values())
    if isinstance(value, (tuple, list)):
        return list, tuple(_input_snapshot(item) for item in value)
    # Bytes, Paths, sets, Enums and unusual mappings retain canonical handling
    # and rejection rules; ordinary Event/Layer guards need no JSON or SHA.
    return _input_snapshot(json.loads(json.dumps(_canonical(value), sort_keys=True)))


_payload_cache = ContextVar("flexeva_payload_operation_cache", default=None)


@contextmanager
def _payload_scope():
    """Backend-owned temporary results; no cache survives this layer operation."""
    cache = {}
    token = _payload_cache.set(cache)
    try:
        yield
    finally:
        cache.clear()
        _payload_cache.reset(token)


def backing_value(event, context=None):
    """Backend helper: obtain native storage, without interpreting its locator."""
    binding = event.binding
    if not isinstance(binding, PayloadBinding):
        raise PipelineContractError("event requires a PayloadBinding; recapture old events")
    if isinstance(binding.state, BackendState):
        return deepcopy(binding.state.value)
    if context is None or context.state is None:
        raise PipelineContractError("bound Payload requires its read-only state context")
    record = context.state.get(binding.state)
    if not isinstance(record, BackendStateRecord) or record.layer_id != event.layer_id:
        raise StateStoreContractError("Payload backing disagrees with its event layer")
    return record.value


def backing_context(event, context):
    ref = event.binding.context_ref
    if ref is not None:
        if context is None or context.state is None:
            raise PipelineContractError("bound Payload requires its read-only state context")
        record = context.state.get(ref)
        if not isinstance(record, BackendStateRecord) or record.layer_id != event.layer_id:
            raise StateStoreContractError("Payload context disagrees with its event layer")
        return record.value
    root = getattr(context, "root", None)
    return root.evidence if root is not None else None


def binding_refs(event):
    binding = event.binding
    if not isinstance(binding, PayloadBinding):
        raise StateStoreContractError("event requires a Payload binding")
    if isinstance(binding.state, BackendState):
        refs = binding.state.state_refs
        if not isinstance(refs, tuple):
            raise StateStoreContractError("inline backend state requires a reference tuple")
    elif isinstance(binding.state, StateRef):
        refs = (binding.state,)
    else:
        raise StateStoreContractError("Payload requires inline partition state or a backing reference")
    refs += ((binding.context_ref,) if binding.context_ref is not None else ())
    if any(not isinstance(ref, StateRef) for ref in refs):
        raise StateStoreContractError("Payload references must be StateRef values")
    return refs


def checked_payload(result):
    if (not isinstance(result, EventPayload) or not isinstance(result.fingerprint, str)
            or not result.fingerprint):
        raise PipelineContractError("Payload must return actual content and a semantic fingerprint")
    return result


def invoke_payload(adapter, event, context):
    """Audit callback purity in debug mode; always check its return schema."""
    debug = getattr(context, "debug", False)
    before = _input_snapshot(event) if debug else None
    result = adapter.payload(event, context)
    if debug and _input_snapshot(event) != before:
        raise PipelineContractError("Payload mutated its event input")
    return checked_payload(result)
