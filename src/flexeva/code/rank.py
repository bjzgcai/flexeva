"""Resolve CFG rank influences with actual, already-evaluated observations."""

from __future__ import annotations

import ast
import inspect
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import FunctionType, MethodType

from ..errors import PipelineContractError
from .cfg import defined_names
from .inputs import safe_input
from .taint import CodeTaintPolicy, TaintObservation, TaintTransfer

_SCALARS = (type(None), bool, int, float, str)
_BUILTINS = (abs, int, float, bool, str, len, min, max, round, range, tuple, list, sum, enumerate, zip, print)


def _portable(value):
    if type(value) is range:
        return {"range": (value.start, value.stop, value.step)}
    return safe_input(value)


@dataclass
class _Scope:
    owner: str
    parameters: dict = field(default_factory=dict)
    calls: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)
    projections: dict = field(default_factory=dict)


@dataclass
class _Call:
    scope: tuple
    site: dict
    parameters: dict
    targets: tuple
    return_tainted: bool = False


class RankTracker:
    def __init__(self, analysis=None, policy=None):
        self.analysis, self.policy = analysis, policy or CodeTaintPolicy()
        self.scopes = {}
        self.calls = ContextVar("cfg_rank_calls", default=())

    def _scope(self, path, owner):
        facts = self.scopes.setdefault((tuple(path), owner), _Scope(owner))
        if self.analysis is not None and owner in self.analysis.lexical_contexts:
            parent, i = self.analysis.lexical_contexts[owner]
            outer = self._scope(path, parent)
            names = self.analysis.inputs[parent][i]
            for name, deps in names.items():
                facts.parameters[name] = facts.parameters.get(name, False) or self._tainted(deps, outer)
            for name, tainted in outer.parameters.items():
                if name not in names:
                    facts.parameters[name] = facts.parameters.get(name, False) or tainted
        return facts

    def activate(self, owner, scope, lane, parameters):
        facts = self._scope(scope, owner)
        calls = self.calls.get()
        if calls and owner in calls[-1].targets:
            facts.parameters.update(calls[-1].parameters)
        for name in parameters:
            facts.parameters.setdefault(name, name in self.policy.sources)

    def deactivate(self, scope):
        for key in tuple(self.scopes):
            if key[0] == tuple(scope):
                self.scopes.pop(key)

    def _environment(self, site):
        if self.analysis is None:
            return {}, frozenset()
        owner, key = site["owner"], site.get("site_id")
        i = self.analysis.site_nodes.get(key)
        if i is None:
            raise PipelineContractError(f"Code CFG has no source-site mapping: {owner}:{key}")
        return self.analysis.inputs[owner][i], self.analysis.control_inputs[owner][i]

    def _tainted(self, dependencies, facts):
        pending, seen = list(dependencies), set()
        while pending:
            dependency = pending.pop()
            if dependency in seen:
                continue
            seen.add(dependency)
            kind, name = dependency.split(":", 1)
            if kind == "input" and facts.parameters.get(name, name in self.policy.sources):
                return True
            if kind == "call":
                if facts.calls.get(name, False):
                    return True
                if name not in facts.calls and self.analysis is not None:
                    pending.extend(self.analysis.call_inputs.get(name, ()))
            if kind == "field" and facts.fields.get(name, False):
                return True
        return False

    def _expression_tainted(self, node, site, facts):
        if self.analysis is None:
            return False
        names, _ = self._environment(site)
        return self._tainted(self.analysis.expression_inputs(site["owner"], node, names), facts)

    def enter_call(self, site, scope, function, args, kwargs):
        facts = self._scope(scope, site["owner"])
        self._check_control(site, facts)
        syntax = self.analysis.expressions.get(site.get("site_id")) if self.analysis else None
        positional, keywords = self._arguments(syntax, site, facts, len(args), kwargs)
        positional = list(positional)
        original = function.__func__ if isinstance(function, MethodType) else function
        parameters, targets = {}, ()
        if isinstance(original, FunctionType):
            if isinstance(function, MethodType):
                positional.insert(0, self._expression_tainted(syntax.func.value, site, facts)
                                  if isinstance(syntax, ast.Call) and isinstance(syntax.func, ast.Attribute) else False)
            names = original.__code__.co_varnames[:original.__code__.co_argcount]
            parameters.update(zip(names, positional))
            parameters.update(keywords)
            code = original.__code__
            extra = code.co_argcount + code.co_kwonlyargcount
            if code.co_flags & inspect.CO_VARARGS:
                parameters[code.co_varnames[extra]] = any(positional[len(names):])
                extra += 1
            if code.co_flags & inspect.CO_VARKEYWORDS:
                parameters[code.co_varnames[extra]] = any(
                    tainted for name, tainted in keywords.items()
                    if name not in code.co_varnames[:code.co_argcount + code.co_kwonlyargcount])
            targets = (original.__module__ + "." + original.__qualname__.replace(".<locals>", ""),)
        call = _Call(tuple(scope), dict(site), parameters, targets)
        self.calls.set((*self.calls.get(), call))
        return call

    def _arguments(self, syntax, site, facts, count, kwargs):
        if not isinstance(syntax, ast.Call):
            return (), {}
        positional = [self._expression_tainted(arg, site, facts) for arg in syntax.args]
        if any(isinstance(arg, ast.Starred) for arg in syntax.args):
            # Already-expanded scalar values resolve any over-approximation;
            # inspecting/unpacking the original operands again is unnecessary.
            positional = [any(positional)] * count
        keywords = {kw.arg: self._expression_tainted(kw.value, site, facts)
                    for kw in syntax.keywords if kw.arg is not None}
        splat = any(self._expression_tainted(kw.value, site, facts)
                    for kw in syntax.keywords if kw.arg is None)
        return tuple(positional), {name: keywords.get(name, splat) for name in kwargs}

    def leave_call(self, token, *, error=None):
        if error is None:
            facts = self._scope(token.scope, token.site["owner"])
            key = token.site["site_id"]
            facts.calls[key] = facts.calls.get(key, False) or token.return_tainted
        self.calls.set(self.calls.get()[:-1])

    def _operation(self, occurrence, syntax, facts, result, completed, *, record_values=True):
        inputs, site = occurrence.inputs, {**occurrence.site, "site_id": occurrence.site_id}
        args, kwargs = tuple(inputs.get("args", ())), inputs.get("kwargs", {})
        if isinstance(syntax, ast.Call):
            parameters = syntax.args
            keywords = {kw.arg: kw.value for kw in syntax.keywords}
            receiver = syntax.func.value if isinstance(syntax.func, ast.Attribute) else None
        else:
            parameters = [child for child in ast.iter_child_nodes(syntax) if isinstance(child, ast.expr)] if syntax else []
            keywords, receiver = {}, None
        if isinstance(syntax, ast.Compare) and site.get("role", "").startswith("comparison-"):
            position = int(site["role"].split("-")[1])
            parameters = [syntax.left, *syntax.comparators][position:position + 2]
        mutation = inputs.get("operation") in {"setitem_rhs_first", "setattr_rhs_first"}
        if mutation:
            graph = self.analysis.graphs[site["owner"]]
            assignment = graph.nodes[self.analysis.site_nodes[occurrence.site_id]].syntax
            rhs = assignment.value if isinstance(assignment, (ast.Assign, ast.AnnAssign, ast.AugAssign)) else None
            parameters = [rhs, syntax.value, syntax.slice if isinstance(syntax, ast.Subscript) else None]
        if isinstance(syntax, ast.Call):
            taints, kw_taints = self._arguments(syntax, site, facts, len(args), kwargs)
        else:
            taints = tuple(self._expression_tainted(parameters[i], site, facts)
                           if i < len(parameters) else self._expression_tainted(syntax, site, facts)
                           for i in range(len(args)))
            kw_taints = {name: self._expression_tainted(keywords.get(name), site, facts) for name in kwargs}
        _, control = self._environment(site)
        observation = TaintObservation(site, occurrence.lane, inputs.get("callable"),
            inputs.get("operation"), args, kwargs, taints, kw_taints,
            self._expression_tainted(receiver, site, facts), self._tainted(control, facts), result, completed)
        transfer = self.policy.resolve(observation)
        if inputs.get("body_summary") is not None and transfer is None:
            self._missing(occurrence, "summarized body requires an explicit rank transfer rule")
        if transfer is None:
            function, operation = observation.callable, observation.operation
            builtin = any(function is item for item in _BUILTINS)
            scalar_operation = operation is not None and not mutation and all(
                type(arg) in (*_SCALARS, tuple, list, dict, range, slice) for arg in args)
            # Uniform opaque calls stay opaque to rank analysis; this says
            # nothing about effects or reuse. Extra rank sources need a rule.
            if (observation.input_tainted or observation.control_tainted) and not (builtin or scalar_operation):
                self._missing(occurrence, "external rank propagation needs a taint rule")
            transfer = TaintTransfer(False if function is print else observation.input_tainted)
        key = occurrence.site_id
        facts.calls[key] = facts.calls.get(key, False) or transfer.return_tainted
        if completed:
            for name, tainted in transfer.writes.items():
                facts.fields[name] = facts.fields.get(name, False) or tainted
            if "result" in transfer.projections:
                facts.projections[key] = transfer.projections["result"]
        self._check_control(site, facts)
        if not record_values:
            return ()
        values = []
        for i, (arg, tainted) in enumerate(zip(args, taints)):
            if tainted:
                values.append((f"arg:{i}", self._value(occurrence, arg, f"arg:{i}", transfer)))
        for name, arg in kwargs.items():
            if kw_taints[name]:
                values.append(("kw:" + name, self._value(occurrence, arg, "kw:" + name, transfer)))
        if completed and transfer.return_tainted:
            values.append(("result", self._value(occurrence, result, "result", transfer)))
        return tuple(values)

    def _check_control(self, site, facts):
        if self.analysis is None:
            return
        names, _ = self._environment(site)
        graph = self.analysis.graphs[site["owner"]]
        node = graph.nodes[self.analysis.site_nodes[site["site_id"]]]
        if node.kind == "opaque":
            dependencies = set().union(*(self.analysis.expression_inputs(site["owner"], item, names)
                for item in ast.walk(node.syntax) if isinstance(item, ast.expr)))
            if self._tainted(dependencies, facts):
                raise PipelineContractError(
                    f"unresolved Code taint at {site.get('span')}: {site['site_id']}: "
                    "rank-dependent compound control is outside the default CFG scope"
                )

    @staticmethod
    def _missing(occurrence, reason):
        span = occurrence.site.get("span", {})
        raise PipelineContractError(
            f"unresolved Code taint at {span}: {occurrence.site_id}: {reason}; extend CodeTaintPolicy"
        )

    def _value(self, occurrence, value, label, transfer=None):
        if transfer is not None and label in transfer.projections:
            return safe_input(transfer.projections[label])
        try:
            return _portable(value)
        except PipelineContractError:
            pass
        projection = self.policy.resolve(TaintObservation(
            {**occurrence.site, "site_id": occurrence.site_id}, occurrence.lane,
            None, "observed-value", (value,), {}, (True,), {}, False, False, value, True))
        if projection is not None and "result" in projection.projections:
            return projection.projections["result"]  # policy.resolve already checks finite JSON
        self._missing(occurrence, f"{label} needs a portable rank-relevant projection")

    def observe(self, occurrence, result=None, *, completed=True):
        if self.analysis is None:
            # Low-level fragment assembly cannot be admitted as a closed root
            # without a frozen program and checked construction evidence.
            return ()
        site = {**occurrence.site, "site_id": occurrence.site_id}
        facts = self._scope(occurrence.inputs.get("scope_path", ()), site["owner"])
        syntax = self.analysis.expressions.get(occurrence.site_id)
        names, control = self._environment(site)
        graph = self.analysis.graphs[site["owner"]]
        node = graph.nodes[self.analysis.site_nodes[occurrence.site_id]]
        self._check_control(site, facts)
        # Iterators are not Code parameters. Bind hooks expose each scalar
        # element and its CFG loop identity without retaining live iterators.
        if site.get("role") == "loop-next":
            return ()
        if "args" in occurrence.inputs and ("callable" in occurrence.inputs or "operation" in occurrence.inputs):
            return self._operation(occurrence, syntax, facts, result, completed)
        values = []
        if isinstance(syntax, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            dependencies = self.analysis.expression_inputs(site["owner"], syntax.value, names) | control
            if isinstance(syntax, ast.AugAssign):
                dependencies |= self.analysis.expression_inputs(site["owner"], syntax.target, names)
            tainted = self._tainted(dependencies, facts)
            targets = syntax.targets if isinstance(syntax, ast.Assign) else (syntax.target,)
            for target in targets:
                for name in sorted(defined_names(target)):
                    if tainted:
                        bindings = occurrence.inputs.get("defined_values", {})
                        if name not in bindings:
                            self._missing(occurrence, f"missing observed definition: {name}")
                        projections = {key[5:]: facts.projections[key[5:]] for key in dependencies
                                       if key.startswith("call:") and key[5:] in facts.projections}
                        transfer = (TaintTransfer(True, projections={"def:" + name: next(iter(projections.values()))})
                                    if len(projections) == 1 and type(bindings[name]) not in (*_SCALARS, tuple, list, dict, range)
                                    else None)
                        values.append(("def:" + name, self._value(occurrence, bindings[name], "def:" + name, transfer)))
        elif site.get("role") in {"loop-bind", "comprehension-bind"}:
            loop = graph.nodes[node.loops[-1]]
            dependencies = self.analysis.expression_inputs(site["owner"], loop.expression, names) | control
            if self._tainted(dependencies, facts):
                for name in site.get("defs", ()):
                    bindings = occurrence.inputs.get("defined_values", {})
                    if name not in bindings:
                        self._missing(occurrence, f"missing observed loop binding: {name}")
                    values.append(("bind:" + name, self._value(occurrence, bindings[name], "bind:" + name)))
        elif isinstance(syntax, ast.Return):
            tainted = self._tainted(self.analysis.expression_inputs(site["owner"], syntax.value, names) | control, facts)
            calls = self.calls.get()
            if calls and site["owner"] in calls[-1].targets:
                calls[-1].return_tainted |= tainted
            if tainted:
                values.append(("return", self._value(occurrence, result, "return")))
        return tuple(values)

    def complete(self, occurrence, result):
        """Finish emission return propagation; the event describes entry."""
        if self.analysis is not None:
            facts = self._scope(occurrence.inputs.get("scope_path", ()), occurrence.site["owner"])
            self._operation(occurrence, self.analysis.expressions.get(occurrence.site_id), facts, result, True,
                            record_values=False)
