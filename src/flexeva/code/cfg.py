"""Code-owned control flow and finite, symbolic rank influence.

Graphs describe frozen Python, not RAS events. Edges model normal synchronous
control transfers; unsupported compound statements remain explicit opaque
regions, never approximate exception/async graphs. The worklist computes
reaching influences with strong assignments, control dependence and loop
feedback. Runtime observations resolve the resulting input/call symbols.
"""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..model import stable_digest
from .region_source import SourceIndex


def location(node):
    return (node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset)


def contains(outer, inner):
    a, b = location(outer)
    c, d = location(inner)
    return a <= c and d <= b


def defined_names(node):
    return {item.id for item in ast.walk(node)
            if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del))}


def syntax_identity(node):
    """Version-neutral AST facts (3.13's ast.dump omits empty fields)."""
    if isinstance(node, ast.AST):
        return (type(node).__name__, tuple(
            (name, syntax_identity(value)) for name, value in ast.iter_fields(node)
            if value is not None and value != []))
    if isinstance(node, list):
        return tuple(syntax_identity(item) for item in node)
    return node


@dataclass(frozen=True)
class CFGNode:
    kind: str
    syntax: ast.AST | None = field(compare=False, repr=False)
    loops: tuple[int, ...] = ()

    @property
    def expression(self):
        node = self.syntax
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Return)):
            return node.value
        if isinstance(node, (ast.If, ast.While, ast.Assert)):
            return node.test
        if isinstance(node, ast.For):
            return node.iter
        return node if isinstance(node, ast.expr) else None


@dataclass(frozen=True)
class CFGEdge:
    source: int
    target: int
    kind: str


class FunctionCFG:
    def __init__(self, owner, body, parameters=()):
        self.owner, self.parameters = owner, tuple(parameters)
        self.nodes, self.edges = [], []
        self.exit = self._node("exit")
        start = self._suite(body, self.exit, ())
        self.entry = self._node("entry")
        self._edge(self.entry, start)
        self.successors = {i: set() for i in range(len(self.nodes))}
        self.predecessors = {i: set() for i in range(len(self.nodes))}
        for edge in self.edges:
            self.successors[edge.source].add(edge.target)
            self.predecessors[edge.target].add(edge.source)
        self.reachable, pending = set(), [self.entry]
        while pending:
            current = pending.pop()
            if current not in self.reachable:
                self.reachable.add(current)
                pending.extend(self.successors[current])
        self.controls = self._control_dependencies()

    def _node(self, kind, syntax=None, loops=()):
        self.nodes.append(CFGNode(kind, syntax, loops))
        return len(self.nodes) - 1

    def _edge(self, source, target, kind="normal"):
        self.edges.append(CFGEdge(source, target, kind))

    def _suite(self, body, follow, loops):
        for statement in reversed(body):
            follow = self._statement(statement, follow, loops)
        return follow

    def _statement(self, node, follow, loops):
        nesting = tuple(item[0] for item in loops)
        if isinstance(node, ast.If):
            head = self._node("branch", node, nesting)
            join = self._node("join", loops=nesting)
            self._edge(join, follow)
            self._edge(head, self._suite(node.body, join, loops), "true")
            self._edge(head, self._suite(node.orelse, join, loops), "false")
            return head
        if isinstance(node, (ast.For, ast.While)):
            head = self._node("loop", node, nesting)
            after = self._node("join", loops=nesting)
            self._edge(after, follow)
            # Exhaustion/false enters else; break bypasses it.
            normal = self._suite(node.orelse, after, loops)
            nested = (*loops, (head, after))
            back = self._node("backedge", loops=(*nesting, head))
            self._edge(back, head, "backedge")
            body = self._suite(node.body, back, nested)
            if isinstance(node, ast.For):
                bind = self._node("loop-bind", node.target, (*nesting, head))
                self._edge(bind, body)
                body = bind
            self._edge(head, body, "next" if isinstance(node, ast.For) else "true")
            self._edge(head, normal, "exhausted" if isinstance(node, ast.For) else "false")
            return head
        if isinstance(node, (ast.Break, ast.Continue)):
            kind = "break" if isinstance(node, ast.Break) else "continue"
            current = self._node(kind, node, nesting)
            self._edge(current, loops[-1][1 if kind == "break" else 0], kind)
            return current
        if isinstance(node, ast.Return):
            current = self._node("return", node, nesting)
            self._edge(current, self.exit, "return")
            return current
        if isinstance(node, ast.With):
            # Enter/exit are actual interfaces. Nonlocal suppression/unwinding
            # is outside this normal-flow model and checked by the collector.
            leave = self._node("context-exit", node, nesting)
            self._edge(leave, follow)
            enter = self._node("context-enter", node, nesting)
            self._edge(enter, self._suite(node.body, leave, loops))
            return enter
        opaque = isinstance(node, (ast.Try, ast.Match)) or type(node).__name__ == "TryStar" or any(
            isinstance(item, (ast.Yield, ast.YieldFrom, ast.Await)) for item in ast.walk(node)
        )
        current = self._node("opaque" if opaque else "statement", node, nesting)
        self._edge(current, follow)
        return current

    def _control_dependencies(self):
        live = self.reachable
        post = {i: ({i} if i == self.exit else set(live)) for i in live}
        changed = True
        while changed:
            changed = False
            for i in sorted(live):
                if i == self.exit:
                    continue
                successors = self.successors[i] & live
                value = {i} | (set.intersection(*(post[j] for j in successors)) if successors else set())
                if value != post[i]:
                    post[i], changed = value, True
        controls = {i: set() for i in range(len(self.nodes))}
        for edge in self.edges:
            if edge.source in live and edge.target in live:
                for dependent in post[edge.target] - post[edge.source]:
                    controls[dependent].add(edge.source)
        return controls

    @property
    def digest(self):
        return stable_digest((self.owner, self.parameters, tuple(
            (node.kind, syntax_identity(node.syntax), node.loops)
            for node in self.nodes), tuple(self.edges)))


class CodeControlFlow:
    """One immutable-source analysis index shared by all invocation collectors."""

    def __init__(self, program, sources=("rank", "rank_id")):
        self.sources = tuple(sources)
        self.graphs, self.sites, self.expressions, self.call_tokens = {}, {}, {}, {}
        self.site_nodes, self.loop_paths, self.loop_ticks = {}, {}, {}
        self.inputs, self.control_inputs, self.call_inputs = {}, {}, {}
        self.lexical_contexts, self.outer_iterators = {}, {}
        lexical_roots = {}
        index = SourceIndex(program.source, program.region)
        positions, lexical_nodes = defaultdict(list), defaultdict(list)
        for module in index.modules.values():
            # Preserve ast.walk order, including nodes sharing a source span.
            # These lookup tables live only for this construction.
            for node in ast.walk(module.tree):
                if hasattr(node, "lineno"):
                    positions[module.name, location(node)].append(node)
                if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda)):
                    lexical_nodes[module.name, type(node).__name__.lower()].append(node)
        for owner, definition in index.definitions.items():
            node = definition.node
            if isinstance(node, ast.FunctionDef):
                parameters = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
                self.graphs[owner] = FunctionCFG(owner, node.body, parameters)
        for site in program.sites:
            self.sites[site.site_id] = site
            module = index.module_for(site.owner)
            candidates = positions.get((module.name, ((site.span.start_line, site.span.start_column),
                                                       (site.span.end_line, site.span.end_column))), ())
            self.expressions[site.site_id] = next((node for node in candidates
                if ast.dump(node, include_attributes=False) == site.normalized), candidates[0] if candidates else None)
            if site.kind == "call":
                key = (site.owner, site.span.start_line, site.span.start_column,
                       site.span.end_line, site.span.end_column)
                self.call_tokens.setdefault(key, set()).add("call:" + site.site_id)
        # Comprehensions have native Python lexical scopes; lower their normal
        # flow to the same for/if machinery, retaining original source spans.
        for site in program.sites:
            if site.owner in self.graphs:
                continue
            module = index.module_for(site.owner)
            marker = next((name for name in ("listcomp", "setcomp", "dictcomp", "generatorexp", "lambda")
                           if f".<{name}>:" in site.owner), None)
            syntax = self.expressions[site.site_id]
            candidates = [node for node in lexical_nodes.get((module.name, marker), ())
                          if syntax is not None and contains(node, syntax)]
            if not candidates:
                continue
            node = min(candidates, key=lambda item: item.end_lineno - item.lineno)
            lexical_roots[site.owner] = (site.lexical_parent, node)
            if isinstance(node, ast.Lambda):
                body = [ast.copy_location(ast.Return(node.body), node.body)]
                parameters = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args)]
            else:
                first, last = location(node.generators[0].iter)
                self.outer_iterators[(site.owner, *first, *last)] = set().union(
                    *(defined_names(generator.target) for generator in node.generators))
                values = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
                body = [ast.copy_location(ast.Expr(value), value) for value in values]
                for generator in reversed(node.generators):
                    for condition in reversed(generator.ifs):
                        body = [ast.copy_location(ast.If(condition, body, []), condition)]
                    body = [ast.copy_location(ast.For(generator.target, generator.iter, body, [], None), node)]
                parameters = ()
            self.graphs[site.owner] = FunctionCFG(site.owner, body, parameters)
        for site in program.sites:
            graph, syntax = self.graphs.get(site.owner), self.expressions[site.site_id]
            if graph is None or syntax is None:
                continue
            candidates = []
            for i, node in enumerate(graph.nodes):
                target = node.expression if node.kind in {"branch", "loop"} else node.syntax
                if target is not None and contains(target, syntax):
                    exact_role = ((node.kind == "loop-bind" and site.role in {"loop-bind", "comprehension-bind"})
                                  or node.kind == site.role)
                    candidates.append((not exact_role, target.end_lineno - target.lineno,
                                       target.end_col_offset - target.col_offset, i))
            if candidates:
                self.site_nodes[site.site_id] = min(candidates)[-1]
        self._index_loops()
        for owner, graph in self.graphs.items():
            self.inputs[owner], self.control_inputs[owner] = self._solve(graph)
        for owner, (parent, syntax) in lexical_roots.items():
            if parent not in self.graphs:
                continue
            candidates = [(node.syntax.end_lineno - node.syntax.lineno,
                           node.syntax.end_col_offset - node.syntax.col_offset, i)
                          for i, node in enumerate(self.graphs[parent].nodes)
                          if node.syntax is not None and contains(node.syntax, syntax)]
            if candidates:
                self.lexical_contexts[owner] = parent, min(candidates)[-1]
        for site_id, i in self.site_nodes.items():
            site, syntax = self.sites[site_id], self.expressions[site_id]
            if not isinstance(syntax, ast.Call) or site.kind != "call":
                continue
            names = self.inputs[site.owner][i]
            operands = [*syntax.args, *(kw.value for kw in syntax.keywords)]
            if isinstance(syntax.func, ast.Attribute):
                operands.append(syntax.func.value)
            deps = set().union(*(self.expression_inputs(site.owner, arg, names) for arg in operands))
            callee = site.callee or ""
            candidates = (site.owner + "." + callee, site.owner.rsplit(".", 1)[0] + "." + callee, callee)
            definition = next((index.definitions[name] for name in candidates if name in index.definitions), None)
            if definition is not None and isinstance(definition.node, ast.FunctionDef):
                parameters = self.graphs[definition.name].parameters
                deps.update("input:" + item.id for item in ast.walk(definition.node)
                            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
                            and item.id in self.sources and item.id not in parameters)
            # Source-level input influence also covers computations before the
            # selected window. An observed return summary refines this default.
            self.call_inputs[site_id] = frozenset(deps)

    def expression_inputs(self, owner, node, names):
        if node is None:
            return frozenset()
        if isinstance(node, ast.expr):
            first, last = location(node)
            captured = self.outer_iterators.get((owner, *first, *last), ())
            if captured:
                names = {name: deps for name, deps in names.items() if name not in captured}
        if isinstance(node, ast.Name):
            return names.get(node.id, frozenset(("input:" + node.id,)))
        if isinstance(node, ast.Constant):
            return frozenset()
        if isinstance(node, ast.Call):
            first, last = location(node)
            tokens = self.call_tokens.get((owner, *first, *last))
            if tokens:
                # The actual callee/owned return supplies this transfer,
                # including an explicitly rank-uniform result.
                return frozenset(tokens)
        children = [child for child in ast.iter_child_nodes(node)
                    if isinstance(child, (ast.expr, ast.keyword, ast.comprehension))]
        result = set().union(*(self.expression_inputs(owner, child, names) for child in children))
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            result.add("field:" + ast.unparse(node))
        if isinstance(node, ast.expr):
            first, last = location(node)
            result.update(self.call_tokens.get((owner, *first, *last), ()))
        return frozenset(result)

    def _solve(self, graph):
        # Local bottom is different from a free input. Treating an as-yet
        # unvisited loop definition as a new input breaks monotonicity.
        local = set(graph.parameters)
        for node in graph.nodes:
            syntax = node.syntax
            if isinstance(syntax, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = syntax.targets if isinstance(syntax, ast.Assign) else (syntax.target,)
                for target in targets:
                    local.update(defined_names(target))
            elif node.kind == "loop-bind":
                local.update(defined_names(syntax))
        bottom = dict.fromkeys(local, frozenset())
        incoming = {i: dict(bottom) for i in range(len(graph.nodes))}
        outgoing = {i: dict(bottom) for i in range(len(graph.nodes))}
        controls = {i: frozenset() for i in range(len(graph.nodes))}
        pending, queued = deque(sorted(graph.reachable)), set(graph.reachable)
        while pending:
            i = pending.popleft()
            queued.discard(i)
            node = graph.nodes[i]
            names = dict(bottom)
            for predecessor in graph.predecessors[i] & graph.reachable:
                for name, deps in outgoing[predecessor].items():
                    names[name] = names.get(name, frozenset()) | deps
            if i == graph.entry:
                names.update((name, frozenset(("input:" + name,))) for name in graph.parameters)
            pc = set()
            for controller in graph.controls[i]:
                pc.update(self.expression_inputs(graph.owner, graph.nodes[controller].expression, incoming[controller]))
                pc.update(controls[controller])
            # Loop termination/continuation controls its induction and carried
            # state, including the value observed outside a zero/partial loop.
            for loop in node.loops:
                pc.update(self.expression_inputs(graph.owner, graph.nodes[loop].expression, incoming[loop]))
                for j, transfer in enumerate(graph.nodes):
                    if transfer.kind in {"break", "continue"} and transfer.loops[-1:] == (loop,):
                        pc.update(controls[j])
            result = dict(names)
            syntax = node.syntax
            if isinstance(syntax, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                deps = self.expression_inputs(graph.owner, syntax.value, names) | pc
                targets = syntax.targets if isinstance(syntax, ast.Assign) else (syntax.target,)
                for target in targets:
                    if isinstance(syntax, ast.AugAssign):
                        deps |= self.expression_inputs(graph.owner, target, names)
                    for name in defined_names(target):
                        result[name] = frozenset(deps) | ({"input:" + name} if name in self.sources else set())
            elif node.kind == "loop-bind":
                loop = graph.nodes[node.loops[-1]]
                deps = self.expression_inputs(graph.owner, loop.expression, incoming[node.loops[-1]]) | pc
                for name in defined_names(syntax):
                    result[name] = frozenset(deps)
            if names != incoming[i] or result != outgoing[i] or frozenset(pc) != controls[i]:
                incoming[i], outgoing[i], controls[i] = names, result, frozenset(pc)
                # Control dependence and loop feedback cross ordinary successor
                # edges. The finite symbol domain ensures worklist convergence.
                for other in sorted(graph.reachable - queued):
                    pending.append(other)
                    queued.add(other)
        return incoming, controls

    def _index_loops(self):
        ticks = {}
        for site_id, i in self.site_nodes.items():
            site, graph = self.sites[site_id], self.graphs[self.sites[site_id].owner]
            node = graph.nodes[i]
            if site.role in {"loop-bind", "comprehension-bind"} and node.kind == "loop-bind":
                ticks[site.owner, node.loops[-1]] = site_id
                self.loop_ticks[site_id] = site.role
            elif site.role == "loop-test" and node.kind == "loop":
                ticks[site.owner, i] = site_id
                self.loop_ticks[site_id] = site.role
        for site_id, i in self.site_nodes.items():
            site, graph = self.sites[site_id], self.graphs[self.sites[site_id].owner]
            node = graph.nodes[i]
            loops = (*node.loops, i) if node.kind == "loop" else node.loops
            self.loop_paths[site_id] = tuple(ticks[site.owner, loop] for loop in loops
                                           if (site.owner, loop) in ticks)

    @property
    def digest(self):
        return stable_digest((tuple((owner, graph.digest) for owner, graph in sorted(self.graphs.items())),
                              self.site_nodes, self.loop_paths))
