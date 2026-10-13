"""Single-evaluation Python protocol helpers for dynamic source instrumentation.

Operands are evaluated in the caller, not inside synthetic lambdas. Standard
``operator`` functions retain Python's reflected and in-place dispatch. This
module classifies source ownership only; the caller decides whether an actual
foreign operation is bound to an emission.
"""

from __future__ import annotations

import inspect
import operator
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import (
    BuiltinFunctionType,
    FunctionType,
    GetSetDescriptorType,
    MemberDescriptorType,
    MethodDescriptorType,
    TracebackType,
    WrapperDescriptorType,
)
from typing import Any

from ._types import WorkloadExecutionError

_BINARY = {
    "add": (operator.add, "add"), "sub": (operator.sub, "sub"),
    "mul": (operator.mul, "mul"), "matmul": (operator.matmul, "matmul"),
    "truediv": (operator.truediv, "truediv"), "floordiv": (operator.floordiv, "floordiv"),
    "mod": (operator.mod, "mod"), "pow": (operator.pow, "pow"),
    "lshift": (operator.lshift, "lshift"), "rshift": (operator.rshift, "rshift"),
    "and": (operator.and_, "and"), "or": (operator.or_, "or"), "xor": (operator.xor, "xor"),
}
_INPLACE = {
    "iadd": operator.iadd, "isub": operator.isub, "imul": operator.imul,
    "imatmul": operator.imatmul, "itruediv": operator.itruediv,
    "ifloordiv": operator.ifloordiv, "imod": operator.imod, "ipow": operator.ipow,
    "ilshift": operator.ilshift, "irshift": operator.irshift,
    "iand": operator.iand, "ior": operator.ior, "ixor": operator.ixor,
}
_UNARY = {"neg": operator.neg, "pos": operator.pos, "invert": operator.invert}
_COMPARE = {
    "lt": (operator.lt, "gt"), "le": (operator.le, "ge"),
    "eq": (operator.eq, "eq"), "ne": (operator.ne, "ne"),
    "gt": (operator.gt, "lt"), "ge": (operator.ge, "le"),
}
_ATOMIC = (type(None), bool, int, float, complex, str, bytes)
_CONTAINERS = (tuple, list, dict, set, frozenset)
_HOST_TYPES = (*_ATOMIC, *_CONTAINERS, bytearray, range, slice, object,
               type(iter(())), type(iter([])), type(iter({})), type(iter(set())),
               type(iter(range(0))), type(iter("")), type(iter(b"")))
_MISSING = object()


def _is_exact(value: Any, types: tuple[type, ...]) -> bool:
    return any(type(value) is candidate for candidate in types)


def _special(value: Any, name: str) -> Any:
    """Read a type slot without executing an instance or metaclass descriptor."""

    return inspect.getattr_static(type(value), name, _MISSING)


def _primitive_tree(value: Any, seen: set[int] | None = None) -> bool:
    """Check only exact builtins whose elements may participate in comparisons."""

    if _is_exact(value, (*_ATOMIC, bytearray, range)):
        return True
    if not _is_exact(value, _CONTAINERS):
        return False
    seen = set() if seen is None else seen
    if id(value) in seen:
        return True
    seen.add(id(value))
    items = (*value.keys(), *value.values()) if type(value) is dict else value
    return all(_primitive_tree(item, seen) for item in items)


def _bind_special(value: Any, name: str) -> Callable[..., Any]:
    """Perform one real special-method descriptor binding, like ``with`` does."""

    descriptor = _special(value, name)
    if descriptor is _MISSING:
        raise TypeError(f"'{type(value).__name__}' object does not support {name}")
    binder = inspect.getattr_static(type(descriptor), "__get__", _MISSING)
    return descriptor if binder is _MISSING else binder(descriptor, value, type(value))


@dataclass
class _AugmentedTarget:
    hook: Any
    op_site: str
    set_site: str
    operation: str
    container: Any
    key: Any
    value: Any
    attribute: bool = False

    def __call__(self, rhs: Any) -> None:
        value = self.hook.operation(self.op_site, self.operation, self.value, rhs)
        store = "setattr_rhs_first" if self.attribute else "setitem_rhs_first"
        self.hook.operation(self.set_site, store, value, self.container, self.key)


@dataclass(frozen=True)
class _NextResult:
    """Transient iterator outcome, not a reusable iterator checkpoint."""

    value: Any = None
    exhausted: StopIteration | None = None


def _next_result(value: Any, default: Any = _MISSING) -> _NextResult:
    if default is not _MISSING:
        return _NextResult(next(value, default))
    try:
        return _NextResult(next(value))
    except StopIteration as error:
        return _NextResult(exhausted=error)


class _ComparisonOperand:
    """Keep native Compare's stack, laziness and original rich result values."""

    def __init__(self, hook: Any, sites: tuple[str, ...], index: int, value: Any) -> None:
        self.hook, self.sites, self.index, self.value = hook, sites, index, value

    def _compare(self, name: str, other: _ComparisonOperand) -> Any:
        if type(other) is not _ComparisonOperand:
            raise WorkloadExecutionError("chained comparison operands must share instrumentation")
        return self.hook.operation(self.sites[self.index], name, self.value, other.value)

    def __lt__(self, other: _ComparisonOperand) -> Any:
        return self._compare("lt", other)

    def __le__(self, other: _ComparisonOperand) -> Any:
        return self._compare("le", other)

    def __eq__(self, other: _ComparisonOperand) -> Any:
        return self._compare("eq", other)

    def __ne__(self, other: _ComparisonOperand) -> Any:
        return self._compare("ne", other)

    def __gt__(self, other: _ComparisonOperand) -> Any:
        return self._compare("gt", other)

    def __ge__(self, other: _ComparisonOperand) -> Any:
        return self._compare("ge", other)


class _Context:
    def __init__(self, hook: Any, enter_site: str, exit_site: str, value: Any) -> None:
        self.hook, self.enter_site, self.exit_site, self.value = hook, enter_site, exit_site, value
        self.exit: Callable[..., Any] | None = None

    def __enter__(self) -> Any:
        # CPython resolves enter before exit, and both before calling enter.
        enter = _bind_special(self.value, "__enter__")
        self.exit = _bind_special(self.value, "__exit__")
        return self.hook.invoke(self.enter_site, enter)

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> Any:
        assert self.exit is not None
        # Returning the original result leaves suppression truth-testing to
        # Python's with statement, including non-bool protocol return values.
        return self.hook.invoke(self.exit_site, self.exit, exc_type, exc, traceback)


class OperationsMixin:
    """Operation hooks mixed into the Core-facing dynamic dispatcher.

    The host must implement ``_source_owned(callable)``, ``invoke(...)`` and
    ``_perform(site, kind, continuation, inputs)``. No reuse state lives here.
    """

    def _protocol_kind(self, candidates: list[Any]) -> str:
        if not getattr(self, "capture_active", True):
            return "call"
        kinds: set[str] = set()
        for candidate in candidates:
            if candidate is _MISSING:
                continue
            if isinstance(candidate, (staticmethod, classmethod)):
                candidate = candidate.__func__
            if not _is_exact(candidate, (
                FunctionType, BuiltinFunctionType, GetSetDescriptorType, MemberDescriptorType,
                MethodDescriptorType, WrapperDescriptorType,
            )) and _special(candidate, "__get__") is not _MISSING:
                raise WorkloadExecutionError(
                    "custom special-method descriptors require explicit dynamic binding coverage"
                )
            if self._source_owned(candidate):
                kinds.add("call")
                continue
            owner = candidate.__objclass__ if _is_exact(candidate, (
                GetSetDescriptorType, MemberDescriptorType, MethodDescriptorType, WrapperDescriptorType,
            )) else None
            if any(owner is host_type for host_type in _HOST_TYPES):
                continue
            kinds.add("emission")
        if kinds == {"call", "emission"}:
            raise WorkloadExecutionError(
                "mixed source-owned/external operator fallback has unresolved actual ownership"
            )
        return "emission" if "emission" in kinds else "call" if "call" in kinds else "block"

    def _binary_protocols(self, name: str, left: Any, right: Any) -> list[Any]:
        protocols = [_special(left, f"__{name}__")]
        # Same-type binary operators never invoke the reflected method. For
        # different types operator.* supplies subtype priority and fallback.
        if type(left) is not type(right):
            protocols.append(_special(right, f"__r{name}__"))
        if name == "mul":
            sequences = (tuple, list, str, bytes, bytearray)
            if _is_exact(left, sequences):
                protocols.append(_special(right, "__index__"))
            elif _is_exact(right, sequences):
                protocols.append(_special(left, "__index__"))
        return protocols

    def _truth_kind(self, value: Any) -> str:
        method = _special(value, "__bool__")
        return self._protocol_kind([
            method if method is not _MISSING else _special(value, "__len__"),
        ])

    def _attribute_protocols(self, value: Any, name: Any, *, store: bool) -> list[Any]:
        method = "__setattr__" if store else "__getattribute__"
        protocols = [_special(value, method)]
        if type(name) is not str:
            # getattr/setattr itself will raise with the original arguments.
            return protocols
        descriptor = inspect.getattr_static(type(value), name, _MISSING)
        if isinstance(descriptor, property):
            accessor = descriptor.fset if store else descriptor.fget
            if accessor is not None:
                protocols.append(accessor)
        elif descriptor is not _MISSING:
            special = _special(descriptor, "__set__" if store else "__get__")
            if special is not _MISSING:
                protocols.append(special)
        if not store:
            protocols.append(_special(value, "__getattr__"))
        return protocols

    def operation(self, site_id: str, op_name: str, /, *operands: Any, _next_default: Any = _MISSING) -> Any:
        """Execute a pre-evaluated Python operation once, preserving its value."""

        protocols: list[Any]
        if op_name in _BINARY:
            left, right = operands
            function, name = _BINARY[op_name]
            protocols = self._binary_protocols(name, left, right)
            continuation = lambda: function(left, right)
        elif op_name in _INPLACE:
            left, right = operands
            protocols = [_special(left, f"__{op_name}__"), *self._binary_protocols(op_name[1:], left, right)]
            continuation = lambda: _INPLACE[op_name](left, right)
        elif op_name in _UNARY:
            (value,) = operands
            protocols = [_special(value, f"__{op_name}__")]
            continuation = lambda: _UNARY[op_name](value)
        elif op_name in ("not", "truth"):
            (value,) = operands
            method = _special(value, "__bool__")
            protocols = [method if method is not _MISSING else _special(value, "__len__")]
            continuation = lambda: operator.not_(value) if op_name == "not" else bool(value)
        elif op_name in _COMPARE:
            left, right = operands
            if getattr(self, "capture_active", True) and any(
                _is_exact(value, _CONTAINERS) and not _primitive_tree(value) for value in operands
            ):
                raise WorkloadExecutionError("container comparison with non-primitive elements requires explicit protocol coverage")
            function, reflected = _COMPARE[op_name]
            protocols = [_special(left, f"__{op_name}__"), _special(right, f"__{reflected}__")]
            continuation = lambda: function(left, right)
        elif op_name in ("is", "is_not"):
            left, right = operands
            protocols = []
            continuation = lambda: (left is right) if op_name == "is" else (left is not right)
        elif op_name in ("in", "not_in"):
            left, right = operands
            content = tuple(right.keys()) if type(right) is dict else right
            if getattr(self, "capture_active", True) and _is_exact(right, _CONTAINERS) and (
                not _primitive_tree(content) or not _primitive_tree(left)
            ):
                raise WorkloadExecutionError("container membership with non-primitive elements requires explicit protocol coverage")
            contains = _special(right, "__contains__")
            protocols = [contains] if contains is not _MISSING else [
                _special(right, "__iter__"), _special(right, "__getitem__"),
            ]
            continuation = lambda: operator.contains(right, left) if op_name == "in" else not operator.contains(right, left)
        elif op_name in ("getitem", "setitem_rhs_first"):
            if op_name == "getitem":
                container, key = operands
                protocols = [_special(container, "__getitem__")]
                continuation = lambda: operator.getitem(container, key)
            else:
                rhs, container, key = operands
                protocols = [_special(container, "__setitem__")]
                continuation = lambda: operator.setitem(container, key, rhs)
            if getattr(self, "capture_active", True) and type(container) is dict and (
                not _primitive_tree(key) or any(not _primitive_tree(item) for item in container)
            ):
                raise WorkloadExecutionError("builtin dict access with custom keys requires explicit protocol coverage")
            if _is_exact(container, (tuple, list, str, bytes, bytearray, range)):
                keys = (key.start, key.stop, key.step) if type(key) is slice else (key,)
                protocols.extend(_special(value, "__index__") for value in keys if value is not None)
        elif op_name in ("getattr", "setattr_rhs_first"):
            if op_name == "getattr":
                value, name = operands
                continuation = lambda: getattr(value, name)
            else:
                rhs, value, name = operands
                continuation = lambda: setattr(value, name, rhs)
            protocols = self._attribute_protocols(value, name, store=op_name == "setattr_rhs_first")
        elif op_name in ("iter", "next"):
            (value,) = operands
            protocols = [_special(value, "__iter__" if op_name == "iter" else "__next__")]
            if op_name == "iter" and protocols[0] is _MISSING:
                protocols = [_special(value, "__getitem__")]
            continuation = lambda: iter(value) if op_name == "iter" else _next_result(value, _next_default)
        else:
            raise WorkloadExecutionError(f"unsupported dynamic operation: {op_name!r}")
        kind = self._protocol_kind(protocols)
        result = self._perform(site_id, kind, continuation, {"operation": op_name, "args": operands})
        if op_name == "next":
            if not isinstance(result, _NextResult):
                raise WorkloadExecutionError("next coverage must preserve the iterator outcome contract")
            if result.exhausted is not None:
                raise result.exhausted
            return result.value
        return result

    @staticmethod
    def slice_key(start: Any, stop: Any, step: Any) -> slice:
        return slice(start, stop, step)

    def augmented(self, get_site: str, op_site: str, set_site: str, op_name: str,
                  container: Any, key: Any) -> _AugmentedTarget:
        """Read target now; Python evaluates the outer call's RHS afterwards."""

        if op_name not in _INPLACE:
            raise WorkloadExecutionError("augmented target requires an in-place operator")
        value = self.operation(get_site, "getitem", container, key)
        return _AugmentedTarget(self, op_site, set_site, op_name, container, key, value)

    def augmented_attribute(self, get_site: str, op_site: str, set_site: str, op_name: str,
                            value: Any, name: str) -> _AugmentedTarget:
        if op_name not in _INPLACE:
            raise WorkloadExecutionError("augmented target requires an in-place operator")
        current = self.operation(get_site, "getattr", value, name)
        return _AugmentedTarget(self, op_site, set_site, op_name, value, name, current, True)

    def iterate(self, site_id: str, iterable: Any) -> Iterator[Any]:
        iterator = self.operation(site_id, "iter", iterable)
        end = object()
        while True:
            # FOR_ITER consumes exhaustion in C. Boxing/re-raising its Python
            # traceback would keep this iterator and its values in a cycle.
            value = self.operation(site_id, "next", iterator, _next_default=end)
            if value is end:
                return
            yield value

    def context(self, enter_site: str, exit_site: str, value: Any) -> _Context:
        return _Context(self, enter_site, exit_site, value)

    def chain_operand(self, site_ids: tuple[str, ...], operand_index: int, value: Any) -> _ComparisonOperand:
        if not 0 <= operand_index <= len(site_ids):
            raise WorkloadExecutionError("invalid chained comparison operand index")
        return _ComparisonOperand(self, tuple(site_ids), operand_index, value)

    @staticmethod
    def chain_result(value: Any) -> Any:
        # Each rich comparison returned its original result, so the native
        # Compare expression already produced exactly the required value.
        return value
