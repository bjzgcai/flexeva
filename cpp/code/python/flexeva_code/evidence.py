"""Portable evidence without inspecting or retaining workload objects."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from types import MappingProxyType


class CodeContractError(ValueError):
    """The frontend cannot certify the requested construction."""


def portable(value):
    """Detach finite JSON values; never invoke arbitrary conversion hooks."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        return tuple(portable(item) for item in value)
    if type(value) in (dict, MappingProxyType):
        if any(type(key) is not str for key in value):
            raise CodeContractError("evidence requires string mapping keys")
        return {key: portable(item) for key, item in sorted(value.items())}
    raise CodeContractError("evidence must be finite portable values, not live objects")


def encode(value):
    return json.dumps(portable(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def freeze(value):
    return _freeze_portable(portable(value))


def _freeze_portable(value):
    """Freeze the already detached tree without validating each subtree again."""
    if type(value) is dict:
        return MappingProxyType({key: _freeze_portable(item) for key, item in value.items()})
    if type(value) is tuple:
        return tuple(_freeze_portable(item) for item in value)
    return value


def identity(value, what):
    if not isinstance(value, Mapping) or not value:
        raise CodeContractError(f"{what} requires a nonempty versioned policy identity")
    # portable intentionally rejects arbitrary Mapping implementations.
    return freeze(value)
