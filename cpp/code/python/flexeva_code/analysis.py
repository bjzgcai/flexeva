"""Reuse the source-only Python CFG analysis for the native Code frontend.

This module never imports frozen workload files. It retains analysis metadata,
not a Python Core, Store or Trace. Dynamic execution is an independent operation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType

from .evidence import CodeContractError, encode, freeze, identity, portable


@dataclass(frozen=True)
class Definition:
    name: str
    path: str
    semantic_digest: str
    callees: tuple[str, ...]
    parameter_paths: tuple[str, ...]
    obligations: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisProgram:
    """Source metadata for the shared CFG, deliberately not executable code."""
    source: object
    region: object
    sites: tuple
    owned_definitions: tuple


@dataclass(frozen=True)
class SourcePlan:
    source_digest: str
    cfg_digest: str
    entrypoint: str
    region_id: str
    policy_identity: object
    rank_sources: tuple[str, ...]
    definitions: object
    program: object = field(repr=False, compare=False)
    analysis: object = field(repr=False, compare=False)
    schema: str = "native-code-analysis-v1"

    def closure(self, definitions):
        """Resolve declared roots and statically known owned callees."""
        pending, result = list(definitions), set()
        if not pending:
            raise CodeContractError("source evidence requires owned definition roots")
        while pending:
            name = pending.pop()
            if name not in self.definitions:
                raise CodeContractError(f"unknown frozen source definition: {name}")
            if name not in result:
                result.add(name)
                pending.extend(self.definitions[name].callees)
        return tuple(sorted(result))

    def obligations_for(self, definitions):
        return tuple(sorted({item for name in self.closure(definitions)
                             for item in self.definitions[name].obligations}))

    def evidence_for(self, definitions, parameters, *, ground_refs=()):
        """Local content evidence; membership and whole-source digest stay out.

        The evaluator declares the complete parameter projection. CFG influence
        alone does not prove that opaque effects or numeric values are irrelevant.
        This value does not itself authorize sharing or skipping execution.
        """
        return freeze({
            "schema": "native-code-input-v1",
            "analysis_policy": self.policy_identity,
            "sources": tuple((name, self.definitions[name].semantic_digest)
                             for name in self.closure(definitions)),
            "parameters": portable(parameters),
            "ground_refs": portable(ground_refs),
        })


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return base + "." + node.attr if base else None
    return None


def _definition_body(node):
    """Walk one definition; nested definitions have their own source identity."""
    pending = list(node.body)
    while pending:
        item = pending.pop()
        yield item
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            pending.extend(ast.iter_child_nodes(item))


def _summaries(index):
    from flexeva.code.cfg import syntax_identity
    from flexeva.model import stable_digest

    aliases, module_bindings, module_effects = {}, {}, {}
    for module in index.modules.values():
        bindings, source_bindings, effects = {}, {}, []

        def bind(key, node):
            source_bindings.setdefault(key, []).append(node)

        for node in module.tree.body:
            initializer_expressions = []
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                initializer_expressions = [node.value]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                initializer_expressions = [*node.decorator_list, *node.args.defaults,
                                           *node.args.kw_defaults, node.returns]
                initializer_expressions.extend(arg.annotation for arg in
                    (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs))
            elif isinstance(node, ast.ClassDef):
                initializer_expressions = [*node.decorator_list, *node.bases,
                                           *(keyword.value for keyword in node.keywords)]
            for expression in initializer_expressions:
                if expression is not None and any(isinstance(item, (ast.Call, ast.Attribute, ast.Subscript,
                        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.NamedExpr))
                        for item in ast.walk(expression)):
                    effects.append(syntax_identity(expression))
            if isinstance(node, ast.Import):
                bindings.update((item.asname or item.name.split(".")[0],
                                 item.name if item.asname else item.name.split(".")[0])
                                for item in node.names)
                for item in node.names:
                    bind(item.asname or item.name.split(".")[0], node)
            elif isinstance(node, ast.ImportFrom):
                prefix = ".".join(module.name.split(".")[:-node.level]) if node.level else ""
                owner = ".".join(part for part in (prefix, node.module) if part)
                bindings.update((item.asname or item.name, owner + "." + item.name)
                                for item in node.names if item.name != "*")
                for item in node.names:
                    bind(item.asname or item.name, node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                for target in targets:
                    for item in ast.walk(target):
                        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                            bind(item.id, node)
                    if any(isinstance(item, (ast.Attribute, ast.Subscript)) for item in ast.walk(target)):
                        effects.append(syntax_identity(node))
            elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)):
                    # Source alone cannot establish which future read is
                    # affected by an initializer call, alias mutation or control
                    # statement. Keep its exact syntax as a conservative module
                    # input, rather than silently losing its effects.
                    effects.append(syntax_identity(node))
        aliases[module.name] = bindings
        module_bindings[module.name] = source_bindings
        module_effects[module.name] = tuple(effects)
    result = {}
    for name, definition in index.definitions.items():
        callees, paths, obligations = set(), set(), set()
        node = definition.node
        parameters = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args,
                      *node.args.kwonlyargs)} if hasattr(node, "args") else set()
        lexical_body = tuple(_definition_body(node))
        rebound = {item.id for item in lexical_body if isinstance(item, ast.Name)
                   and isinstance(item.ctx, (ast.Store, ast.Del))}
        for item in lexical_body:
            if isinstance(item, (ast.Import, ast.ImportFrom)):
                rebound.update(alias.asname or alias.name.split(".")[0] for alias in item.names)
            elif isinstance(item, ast.ExceptHandler) and item.name:
                rebound.add(item.name)
        if isinstance(node, ast.ClassDef):
            # Analyze methods in their own lexical scopes so a method argument
            # cannot accidentally resolve to a same-named module function.
            callees.update(definition.name + "." + item.name for item in node.body
                           if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        body = lexical_body
        for item in body:
            if isinstance(item, ast.Call):
                target = _call_name(item.func)
                candidates = []
                base = target.split(".", 1)[0] if target else None
                shadowed = base in rebound or (base in parameters and base not in ("self", "cls"))
                # A module rebinding can replace an otherwise owned definition.
                # Import aliases without such a collision remain resolvable.
                if (base in module_bindings[definition.module]
                        and definition.module + "." + base in index.definitions):
                    shadowed = True
                if target and not shadowed:
                    candidates.extend((name + "." + target,
                                       definition.parent + "." + target,
                                       definition.module + "." + target, target))
                    if target.startswith(("self.", "cls.")):
                        candidates.append(definition.parent + "." + target.split(".", 1)[1])
                    base, *rest = target.split(".")
                    if base in aliases[definition.module]:
                        candidates.append(".".join((aliases[definition.module][base], *rest)))
                callee = next((key for key in candidates if key in index.definitions), None)
                if callee:
                    callees.add(callee)
                else:
                    obligations.add(f"opaque-call:{name}:{item.lineno}:{item.col_offset}:{target or '<dynamic>'}")
            if isinstance(item, (ast.Attribute, ast.Subscript)):
                path = ast.unparse(item)
                if path.split(".", 1)[0].split("[", 1)[0] in parameters:
                    paths.add(path)
                if isinstance(item.ctx, (ast.Store, ast.Del)):
                    obligations.add(f"mutable-alias:{name}:{item.lineno}:{item.col_offset}:{path}")
            if isinstance(item, (ast.Try, ast.AsyncFor, ast.AsyncWith, ast.Yield,
                                 ast.YieldFrom, ast.Await, ast.Match)):
                obligations.add(f"opaque-control:{name}:{item.lineno}:{item.col_offset}:{type(item).__name__}")
        # Globals and imported bindings are part of a function's source input.
        # Hash only bindings read by this definition, recursively following
        # module-level aliases/constants; do not smear unrelated globals over
        # every Code event.
        bindings, used, pending = module_bindings[definition.module], {}, [
            item.id for item in ast.walk(node) if isinstance(item, ast.Name)
            and isinstance(item.ctx, ast.Load) and item.id not in parameters]
        while pending:
            binding = pending.pop()
            if binding in used or binding not in bindings:
                continue
            syntaxes = bindings[binding]
            used[binding] = tuple(syntax_identity(syntax) for syntax in syntaxes)
            pending.extend(item.id for syntax in syntaxes for item in ast.walk(syntax)
                           if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load))
        if module_effects[definition.module]:
            obligations.add(f"module-initializer-effects:{definition.module}")
        semantic = (syntax_identity(node), tuple(sorted(used.items())), module_effects[definition.module])
        result[name] = Definition(name, definition.path, stable_digest(semantic),
                                  tuple(sorted(callees)), tuple(sorted(paths)),
                                  tuple(sorted(obligations)))
    return MappingProxyType(result)


def _analysis_program(source, region, index):
    # Reuse the existing source-site/symbol extraction but never run its AST
    # rewriting pass. Source analysis must also work for valid Python forms
    # that the optional instrumented executor cannot transparently rewrite.
    from flexeva.code.dynamic_instrument import _Instrumenter

    sites = []
    for module in index.modules.values():
        builder = _Instrumenter(index, module.name, "__flexeva_analysis__")

        class Sites(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                previous = builder.owner
                builder.owner = previous + "." + node.name
                for child in node.body:
                    self.visit(child)
                builder.owner = previous

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, node):
                previous = builder.owner
                builder.owner = previous + "." + node.name
                for child in node.body:
                    self.visit(child)
                builder.owner = previous

            def generic_visit(self, node):
                if hasattr(node, "lineno") and isinstance(node, (ast.stmt, ast.expr)):
                    role = "statement"
                    if isinstance(node, ast.Call):
                        builder.site(node, kind="call", callee=_call_name(node.func))
                    else:
                        builder.site(node, role=role)
                    if isinstance(node, ast.For):
                        builder.site(node.target, role="loop-bind")
                    elif isinstance(node, ast.While):
                        builder.site(node.test, role="loop-test")
                super().generic_visit(node)

        Sites().visit(module.tree)
        sites.extend(builder.sites)
    definitions = tuple({"owner": name, "source_digest": item.semantic_digest}
                        for name, item in _summaries(index).items())
    return AnalysisProgram(source, region, tuple(sites), definitions)


@lru_cache(maxsize=32)
def _analyze(files, entrypoint, region_id, policy_bytes, rank_sources):
    import json
    from flexeva.model import SourceFile, SourceSnapshot
    from flexeva.code.cfg import CodeControlFlow
    from flexeva.code.region_model import MutableRegion
    from flexeva.code.region_source import SourceIndex

    source = SourceSnapshot.create(tuple(SourceFile.from_text(path, text) for path, text in files))
    region = MutableRegion(region_id, tuple(path for path, _ in files), entrypoint)
    index = SourceIndex(source, region)
    program = _analysis_program(source, region, index)
    analysis = CodeControlFlow(program, rank_sources)
    return SourcePlan(source.digest, analysis.digest, entrypoint, region_id,
                      freeze(json.loads(policy_bytes)), rank_sources,
                      _summaries(index), program, analysis)


def analyze_source(sources, entrypoint, *, region_id="workload", policy_identity,
                   rank_sources=("rank", "rank_id")):
    """Parse/index/instrument frozen text without executing or importing it."""
    policy = identity(policy_identity, "source analysis")
    if not sources or any(type(path) is not str or type(text) is not str
                          for path, text in sources.items()):
        raise CodeContractError("frozen sources require nonempty relative-path/text mappings")
    if not rank_sources or any(type(name) is not str or not name.isidentifier() for name in rank_sources):
        raise CodeContractError("rank sources require explicit identifier names")
    return _analyze(tuple(sorted(sources.items())), entrypoint, region_id, encode(policy),
                    tuple(sorted(set(rank_sources))))


def clear_analysis_cache():
    _analyze.cache_clear()
