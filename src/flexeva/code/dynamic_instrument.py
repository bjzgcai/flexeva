"""AST instrumentation that keeps the workload's Python execution in charge.

Hooks receive already-evaluated call operands. No pre-run, copied model method,
manual observation declaration, or evaluator implementation is introduced.
"""

from __future__ import annotations

import ast
import copy
from collections import Counter

from ..model import SourceFile, stable_digest
from .dynamic import DynamicCodeSite
from .region_source import SourceIndex


class DynamicSourceError(ValueError):
    """The exporter cannot instrument a source construct without changing it."""


_BINARY = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.MatMult: "matmul",
           ast.Div: "truediv", ast.FloorDiv: "floordiv", ast.Mod: "mod", ast.Pow: "pow",
           ast.LShift: "lshift", ast.RShift: "rshift", ast.BitAnd: "and",
           ast.BitOr: "or", ast.BitXor: "xor"}
_UNARY = {ast.UAdd: "pos", ast.USub: "neg", ast.Invert: "invert", ast.Not: "not"}
_COMPARE = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le",
            ast.Gt: "gt", ast.GtE: "ge", ast.Is: "is", ast.IsNot: "is_not",
            ast.In: "in", ast.NotIn: "not_in"}
_FRAME_INTRINSICS = {"super", "locals", "globals", "vars", "dir"}


def _location(node):
    return (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)


def _generator(function):
    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, (ast.Yield, ast.YieldFrom)) or walk(child):
                return True
        return False
    return walk(function)


def _symbols(node):
    defs, uses, paths = set(), set(), set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            (defs if isinstance(item.ctx, (ast.Store, ast.Del)) else uses).add(item.id)
        elif isinstance(item, (ast.Attribute, ast.Subscript)):
            root = item
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if isinstance(root, ast.Name):
                paths.add(ast.unparse(item))
                uses.add(root.id)
                if isinstance(item.ctx, (ast.Store, ast.Del)):
                    defs.add(root.id)
    if isinstance(node, ast.AugAssign):
        uses.update(name.id for name in ast.walk(node.target) if isinstance(name, ast.Name))
    return tuple(sorted(defs)), tuple(sorted(uses)), tuple(sorted(paths))


class _Instrumenter(ast.NodeTransformer):
    def __init__(self, index: SourceIndex, module: str, hook_name: str):
        self.index, self.module, self.hook_name = index, module, hook_name
        self.owner = module
        self.in_function = False
        self.in_generator = False
        self.sites: list[DynamicCodeSite] = []
        self.counts: Counter = Counter()
        self.lexical_parents = {name: definition.parent for name, definition in index.definitions.items()}
        self.selected_location = _location(index.selector_node)
        self.selected = False

    def fail(self, node, reason):
        raise DynamicSourceError(f"{self.index.module_for(self.owner).path}:{node.lineno}: {reason}")

    def hook(self, method, *args, at=None, keywords=()):
        call = ast.Call(ast.Attribute(ast.Name(self.hook_name, ast.Load()), method, ast.Load()),
                        list(args), list(keywords))
        return ast.copy_location(call, at) if at is not None else call

    def site(self, node, *, kind="block", role="statement", operation=None, callee=None, ownership="runtime"):
        normalized = ast.dump(node, include_attributes=False)
        key = (self.owner, kind, role, normalized)
        ordinal = self.counts[key]
        self.counts[key] += 1
        identity = "dynamic:" + stable_digest((self.index.region.region_id, *key, ordinal))[:24]
        defs, uses, paths = _symbols(node)
        argument_uses = {}
        if isinstance(node, ast.Call):
            argument_uses = {
                "positional": [_symbols(arg)[1] for arg in node.args],
                "keywords": {kw.arg: _symbols(kw.value)[1] for kw in node.keywords if kw.arg is not None},
                "starred_positions": [i for i, arg in enumerate(node.args) if isinstance(arg, ast.Starred)],
                "keyword_splat": [_symbols(kw.value)[1] for kw in node.keywords if kw.arg is None],
                "callable": _symbols(node.func)[1],
                "receiver": _symbols(node.func.value)[1] if isinstance(node.func, ast.Attribute) else (),
            }
        self.sites.append(DynamicCodeSite(identity, kind, self.owner,
                                         self.index.span(node, self.owner), self.index.text(node, self.owner),
                                         normalized, operation, callee, ownership, role, defs, uses, paths, argument_uses,
                                         self.lexical_parents.get(self.owner)))
        return identity

    def marker(self, site, node):
        return ast.copy_location(ast.Expr(self.hook("block", ast.Constant(site), at=node)), node)

    def context(self, expression, body, node):
        return ast.copy_location(ast.With([ast.withitem(expression)], body), node)

    def suite(self, body):
        output = []
        for statement in body:
            value = self.visit(statement)
            output.extend(value if isinstance(value, list) else [value] if value is not None else [])
        return output

    def _simple(self, original, changed, *, marker=True):
        if not self.in_function:
            return changed
        identity = self.site(original)
        body = [changed]
        if marker:
            body.append(self.marker(identity, original))
        # Yield is allowed to suspend only the original generator, never a
        # source-owner context manager in the consumer's execution context.
        if any(isinstance(item, (ast.Yield, ast.YieldFrom)) for item in ast.walk(original)):
            return body
        return self.context(self.hook("owner", ast.Constant(identity), at=original), body, original)

    def visit_Module(self, node):
        node.body = self.suite(node.body)
        return node

    def visit_ClassDef(self, node):
        outer, in_function = self.owner, self.in_function
        node.decorator_list = [self.visit(item) for item in node.decorator_list]
        node.bases = [self.visit(item) for item in node.bases]
        node.keywords = [ast.keyword(arg=item.arg, value=self.visit(item.value)) for item in node.keywords]
        self.owner, self.in_function = f"{outer}.{node.name}", False
        node.body = self.suite(node.body)
        self.owner, self.in_function = outer, in_function
        return node

    def visit_AsyncFunctionDef(self, node):
        self.fail(node, "async functions require an async dynamic runtime protocol")

    def visit_FunctionDef(self, node):
        original = copy.deepcopy(node)
        previous = self.owner, self.in_function, self.in_generator
        # Decorator/default expressions evaluate in the enclosing lexical
        # scope. Annotations remain unchanged, including postponed annotations.
        node.decorator_list = [self.visit(item) for item in node.decorator_list]
        node.args.defaults = [self.visit(item) for item in node.args.defaults]
        node.args.kw_defaults = [self.visit(item) if item is not None else None for item in node.args.kw_defaults]
        self.owner, self.in_function, self.in_generator = f"{previous[0]}.{node.name}", True, _generator(node)
        docstring = []
        source_body = node.body
        if source_body and isinstance(source_body[0], ast.Expr) and isinstance(source_body[0].value, ast.Constant) and isinstance(source_body[0].value.value, str):
            docstring, source_body = source_body[:1], source_body[1:]
        body = self.suite(source_body)
        if self.owner == self.index.root.name and self.index.region.selector is None:
            body = [self.context(self.hook("region", ast.Constant(self.index.region.region_id), at=node), body, node)]
            self.selected = True
        if self.in_generator:
            # Applied before user decorators (e.g. contextmanager), preserving
            # the original body and its send/throw/close behavior.
            node.decorator_list.append(self.hook("generator", ast.Constant(self.owner), at=node))
            node.body = docstring + body
        else:
            node.body = docstring + [self.context(self.hook("activate", ast.Constant(self.owner), at=node), body, node)]
        self.owner, self.in_function, self.in_generator = previous
        if self.in_function:
            return [node, self.marker(self.site(original, role="definition-bind"), original)]
        return node

    def visit_Call(self, node):
        original = copy.deepcopy(node)
        if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "compile"}:
            self.fail(node, "dynamic code compilation/execution is outside the frozen source contract")
        node.func = self.visit(node.func)
        node.args = [self.visit(item) for item in node.args]
        node.keywords = [ast.keyword(arg=item.arg, value=self.visit(item.value)) for item in node.keywords]
        if not self.in_function:
            return node
        if isinstance(original.func, ast.Name) and original.func.id in _FRAME_INTRINSICS:
            # These interpreter intrinsics must see the original caller frame.
            self.site(original, kind="block", role="caller-frame", callee=original.func.id,
                      ownership="host-interpreter")
            return node
        identity = self.site(original, kind="call", role="call", callee=ast.unparse(original.func))
        return self.hook("invoke", ast.Constant(identity), node.func, *node.args,
                         keywords=node.keywords, at=original)

    def visit_Lambda(self, node):
        outer = self.owner
        key = (outer, "lambda", ast.dump(node, include_attributes=False))
        ordinal = self.counts[key]
        self.counts[key] += 1
        node.args.defaults = [self.visit(item) for item in node.args.defaults]
        node.args.kw_defaults = [self.visit(item) if item is not None else None for item in node.args.kw_defaults]
        self.owner = f"{outer}.<lambda>:{stable_digest((key, ordinal))[:16]}"
        self.lexical_parents[self.owner] = outer
        node.body = self.visit(node.body)
        self.owner = outer
        return node

    def visit_BinOp(self, node):
        if not self.in_function:
            return self.generic_visit(node)
        identity = self.site(node, kind="call", role="operator", operation=_BINARY[type(node.op)])
        return self.hook("operation", ast.Constant(identity), ast.Constant(_BINARY[type(node.op)]),
                         self.visit(node.left), self.visit(node.right), at=node)

    def visit_UnaryOp(self, node):
        if not self.in_function:
            return self.generic_visit(node)
        operation = _UNARY[type(node.op)]
        identity = self.site(node, kind="call", role="operator", operation=operation)
        return self.hook("operation", ast.Constant(identity), ast.Constant(operation), self.visit(node.operand), at=node)

    def visit_Compare(self, node):
        if not self.in_function:
            return self.generic_visit(node)
        if len(node.ops) > 1:
            if any(isinstance(op, (ast.Is, ast.IsNot, ast.In, ast.NotIn)) for op in node.ops):
                self.fail(node, "chained identity/membership comparisons require a separate protocol lowering")
            sites = [self.site(node, kind="call", role=f"comparison-{i}", operation=_COMPARE[type(op)])
                     for i, op in enumerate(node.ops)]
            ids = ast.Tuple([ast.Constant(site) for site in sites], ast.Load())
            operands = [node.left, *node.comparators]
            wrapped = [self.hook("chain_operand", ids, ast.Constant(i), self.visit(value), at=value)
                       for i, value in enumerate(operands)]
            changed = ast.copy_location(ast.Compare(wrapped[0], node.ops, wrapped[1:]), node)
            return self.hook("chain_result", changed, at=node)
        operation = _COMPARE[type(node.ops[0])]
        identity = self.site(node, kind="call", role="operator", operation=operation)
        return self.hook("operation", ast.Constant(identity), ast.Constant(operation),
                         self.visit(node.left), self.visit(node.comparators[0]), at=node)

    def key(self, node):
        if isinstance(node, ast.Slice):
            return self.hook("slice_key", *(self.visit(item) if item is not None else ast.Constant(None)
                                            for item in (node.lower, node.upper, node.step)), at=node)
        if isinstance(node, ast.Tuple):
            return ast.copy_location(ast.Tuple([self.key(item) for item in node.elts], ast.Load()), node)
        return self.visit(node)

    def visit_Subscript(self, node):
        if not self.in_function or not isinstance(node.ctx, ast.Load):
            return self.generic_visit(node)
        identity = self.site(node, kind="call", role="operator", operation="getitem")
        return self.hook("operation", ast.Constant(identity), ast.Constant("getitem"),
                         self.visit(node.value), self.key(node.slice), at=node)

    def visit_Assign(self, node):
        original = copy.deepcopy(node)
        if self.in_function and any(isinstance(target, (ast.Subscript, ast.Attribute)) for target in node.targets):
            if len(node.targets) != 1:
                self.fail(node, "chained assignment with an attribute/item target needs explicit evaluation-order lowering")
            target = node.targets[0]
            operation = "setitem_rhs_first" if isinstance(target, ast.Subscript) else "setattr_rhs_first"
            identity = self.site(target, kind="call", role="store", operation=operation)
            key = self.key(target.slice) if isinstance(target, ast.Subscript) else ast.Constant(target.attr)
            changed = ast.copy_location(ast.Expr(self.hook("operation", ast.Constant(identity), ast.Constant(operation),
                                                          self.visit(node.value), self.visit(target.value), key, at=node)), node)
            return self._simple(original, changed)
        node.value = self.visit(node.value)
        # Name/tuple targets are kept as Python assignment syntax. No temporary
        # local is introduced into locals(), closure cells or exception scopes.
        return self._simple(original, node)

    def visit_AnnAssign(self, node):
        original = copy.deepcopy(node)
        if isinstance(node.target, (ast.Subscript, ast.Attribute)) and self.in_function and node.value is not None:
            self.fail(node, "annotated attribute/item store requires an explicit annotation-preserving lowering")
        if node.value is not None:
            node.value = self.visit(node.value)
        return self._simple(original, node)

    def visit_AugAssign(self, node):
        if not self.in_function:
            return self.generic_visit(node)
        original = copy.deepcopy(node)
        operation = "i" + _BINARY[type(node.op)].rstrip("_")
        identity = self.site(node, kind="call", role="inplace-operator", operation=operation)
        if isinstance(node.target, ast.Name):
            changed = ast.Assign([node.target], self.hook("operation", ast.Constant(identity), ast.Constant(operation),
                                                        ast.copy_location(ast.Name(node.target.id, ast.Load()), node.target),
                                                        self.visit(node.value), at=node))
        elif isinstance(node.target, (ast.Subscript, ast.Attribute)):
            target = node.target
            get_operation = "getitem" if isinstance(target, ast.Subscript) else "getattr"
            set_operation = "setitem" if isinstance(target, ast.Subscript) else "setattr"
            getter = self.site(target, kind="call", role="augmented-load", operation=get_operation)
            setter = self.site(target, kind="call", role="augmented-store", operation=set_operation)
            key = self.key(target.slice) if isinstance(target, ast.Subscript) else ast.Constant(target.attr)
            method = "augmented_attribute" if isinstance(target, ast.Attribute) else "augmented"
            capture = self.hook(method, ast.Constant(getter), ast.Constant(identity), ast.Constant(setter),
                                ast.Constant(operation), self.visit(target.value), key,
                                at=node)
            changed = ast.Expr(ast.Call(capture, [self.visit(node.value)], []))
        else:
            self.fail(node, "unsupported augmented assignment target")
        return self._simple(original, ast.copy_location(changed, node))

    def visit_Expr(self, node):
        original = copy.deepcopy(node)
        node.value = self.visit(node.value)
        if isinstance(original.value, (ast.Call, ast.Yield, ast.YieldFrom)):
            return node
        return self._simple(original, node)

    def visit_Return(self, node):
        if not self.in_function:
            return node
        identity = self.site(node, role="return")
        node.value = self.hook("completed", ast.Constant(identity),
                               self.visit(node.value) if node.value is not None else ast.Constant(None), at=node)
        return node

    def visit_Raise(self, node):
        identity = self.site(node, role="raise")
        node.exc = self.visit(node.exc) if node.exc is not None else None
        node.cause = self.visit(node.cause) if node.cause is not None else None
        # The owner helper is a try/finally context, not an except handler, so a
        # bare raise keeps the original active exception.
        if node.exc is not None:
            if node.cause is not None:
                node.cause = self.hook("completed", ast.Constant(identity), node.cause, at=node)
            else:
                node.exc = self.hook("completed", ast.Constant(identity), node.exc, at=node)
            return node
        return [self.marker(identity, node), node]

    def visit_Break(self, node):
        return [self.marker(self.site(node, role="break"), node), node]

    def visit_Continue(self, node):
        return [self.marker(self.site(node, role="continue"), node), node]

    def visit_Pass(self, node):
        return self._simple(copy.deepcopy(node), node)

    def visit_Import(self, node):
        return self._simple(copy.deepcopy(node), node)

    def visit_ImportFrom(self, node):
        return self._simple(copy.deepcopy(node), node)

    def visit_Delete(self, node):
        if any(isinstance(target, (ast.Subscript, ast.Attribute)) for target in node.targets):
            self.fail(node, "attribute/item deletion needs an explicit protocol scope")
        return self._simple(copy.deepcopy(node), node)

    def visit_If(self, node):
        identity = self.site(node.test, role="branch-test")
        node.test = self.hook("predicate", ast.Constant(identity), self.visit(node.test), at=node.test)
        node.body, node.orelse = self.suite(node.body), self.suite(node.orelse)
        return node

    def visit_While(self, node):
        identity = self.site(node.test, role="loop-test")
        node.test = self.hook("predicate", ast.Constant(identity), self.visit(node.test), at=node.test)
        node.body, node.orelse = self.suite(node.body), self.suite(node.orelse)
        return node

    def visit_For(self, node):
        identity = self.site(node.iter, role="loop-next")
        binding = self.site(node.target, role="loop-bind")
        node.iter = self.hook("iterate", ast.Constant(identity), self.visit(node.iter), at=node.iter)
        node.body = [self.marker(binding, node.target), *self.suite(node.body)]
        node.orelse = self.suite(node.orelse)
        return node

    def _comprehension(self, node):
        outer = self.owner
        key = (outer, type(node).__name__, ast.dump(node, include_attributes=False))
        ordinal = self.counts[key]
        self.counts[key] += 1
        # The first iterable is evaluated in the enclosing scope; Python owns
        # the target/filters/element scope thereafter.
        first = self.visit(node.generators[0].iter)
        self.owner = f"{outer}.<{type(node).__name__.lower()}>:{stable_digest((key, ordinal))[:16]}"
        self.lexical_parents[self.owner] = outer
        for position, generator in enumerate(node.generators):
            if generator.is_async:
                self.fail(node, "async comprehensions require an async dynamic runtime protocol")
            generator.iter = first if position == 0 else self.visit(generator.iter)
            binding = self.site(generator.target, role="comprehension-bind")
            generator.ifs = [self.hook("completed", ast.Constant(binding), ast.Constant(True), at=generator.target),
                             *(self.visit(condition) for condition in generator.ifs)]
        if isinstance(node, ast.DictComp):
            node.key, node.value = self.visit(node.key), self.visit(node.value)
        else:
            node.elt = self.visit(node.elt)
        self.owner = outer
        return node

    visit_ListComp = _comprehension
    visit_SetComp = _comprehension
    visit_DictComp = _comprehension
    visit_GeneratorExp = _comprehension

    def visit_With(self, node):
        selected = self.owner == self.index.root.name and _location(node) == self.selected_location and self.index.region.selector is not None
        items = []
        for item in node.items:
            expression = item.context_expr
            enter = self.site(expression, kind="call", role="context-enter", operation="__enter__")
            exit_site = self.site(expression, kind="call", role="context-exit", operation="__exit__")
            wrapped = self.hook("context", ast.Constant(enter), ast.Constant(exit_site), self.visit(expression), at=expression)
            items.append(ast.withitem(wrapped, item.optional_vars))
        node.items = items
        node.body = self.suite(node.body)
        if selected:
            key = copy.deepcopy(self.index.enclosing_loop.target)
            for item in ast.walk(key):
                if hasattr(item, "ctx"):
                    item.ctx = ast.Load()
            node.body = [self.context(self.hook("region", ast.Constant(self.index.region.region_id), key, at=node), node.body, node)]
            self.selected = True
        return node

    def visit_Try(self, node):
        node.body = self.suite(node.body)
        for handler in node.handlers:
            handler.body = self.suite(handler.body)
        node.orelse, node.finalbody = self.suite(node.orelse), self.suite(node.finalbody)
        return node

    def visit_Assert(self, node):
        # Assertions, including message evaluation and -O removal, stay native
        # syntax. The completed marker disappears with an optimized assertion.
        original = copy.deepcopy(node)
        identity = self.site(node.test, role="assert-test")
        node.test = self.hook("predicate", ast.Constant(identity), self.visit(node.test), at=node.test)
        if node.msg is not None:
            node.msg = self.visit(node.msg)
        return self._simple(original, node, marker=False)

    def visit_Yield(self, node):
        identity = self.site(node, role="yield")
        node.value = self.hook("yielded", ast.Constant(identity),
                               self.visit(node.value) if node.value is not None else ast.Constant(None), at=node)
        return node

    def visit_YieldFrom(self, node):
        identity = self.site(node, role="yield-from")
        node.value = self.hook("delegated", ast.Constant(identity), self.visit(node.value), at=node)
        return node


def instrument(index: SourceIndex, hook_name: str):
    files, sites, selected = [], [], False
    for name, module in sorted(index.modules.items()):
        transformer = _Instrumenter(index, name, hook_name)
        tree = transformer.visit(copy.deepcopy(module.tree))
        ast.fix_missing_locations(tree)
        # Compilation validates transformed syntax without executing imports,
        # decorators, source code, or evaluator initialization.
        compile(tree, f"<flexeva-frozen>/{module.path}", "exec")
        files.append(SourceFile.from_text(module.path, ast.unparse(tree) + "\n"))
        sites.extend(transformer.sites)
        selected = selected or transformer.selected
    if not selected:
        raise DynamicSourceError("the requested region was not instrumented")
    return tuple(files), tuple(sites)
