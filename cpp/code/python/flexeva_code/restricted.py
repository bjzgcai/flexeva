"""Source-only rank grouping for the explicitly restricted execution contract.

No model construction, user callable execution, symbolic object evaluation or
alias analysis occurs here. Unresolved rank transfer is an all-rank fallback.
"""
import ast
import operator
from dataclasses import replace
from .evidence import digest
from .ranks import RankEquivalenceEvidence

_BINARY = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
           ast.FloorDiv:operator.floordiv, ast.Mod:operator.mod, ast.Div:operator.truediv}
_COMPARE = {ast.Eq:operator.eq, ast.NotEq:operator.ne, ast.Lt:operator.lt,
            ast.LtE:operator.le, ast.Gt:operator.gt, ast.GtE:operator.ge}


def restricted_rank_evidence(source, arguments, *, call_supported, argument_evidence=None,
                             constructor_supported=None):
    """Return certificates for a frozen, fixed-context restricted source window.

    call_supported is the evaluator's finite effect vocabulary. Unknown calls,
    unsupported scalar expressions, or a rank crossing an owned call boundary
    leave an explicit obligation; callers then execute every logical rank.
    """
    trees = [ast.parse(item.text) for item in source.program.source.files]
    owned = {node.name for tree in trees for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    methods = {node.name for tree in trees for cls in ast.walk(tree)
               if isinstance(cls, ast.ClassDef) for node in cls.body if isinstance(node, ast.FunctionDef)}
    definitions = tuple(source.definitions)
    required = source.obligations_for(definitions)
    def constructed(node):
        while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('to', 'train', 'eval'):
            node = node.func.value
        return isinstance(node, ast.Call) and (ast.unparse(node.func) in owned
            or constructor_supported is not None and constructor_supported(ast.unparse(node.func)))

    unresolved = set()
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                frozen = {d.split(".", 1)[0] for d in definitions}
                if any(m.split(".", 1)[0] not in {"torch", "math", *frozen} for m in modules):
                    unresolved.add("restricted rank analysis: unregistered imported context")
            if isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
                unresolved.add('restricted rank analysis: unsupported source control')
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            leaf = name.rsplit('.',1)[-1]
            if leaf == "get_world_size" and (node.args or node.keywords):
                unresolved.add("restricted rank analysis: subgroup membership needs explicit mapping")
            if (leaf not in owned and leaf not in methods and not name.startswith('self.')
                    and not call_supported(name)):
                # Local model values are tracked from constructors below.
                constructors = {target.id for t in trees for assign in ast.walk(t)
                    if isinstance(assign, ast.Assign) and constructed(assign.value)
                    for target in assign.targets if isinstance(target, ast.Name)}
                if name not in constructors:
                    unresolved.add('restricted rank analysis: unknown call ' + name)

    def rank_call(node):
        return isinstance(node, ast.Call) and ast.unparse(node.func).rsplit(".", 1)[-1] == "get_rank"

    def evidence(rank):
        values = arguments[rank]
        reasons = set(unresolved)
        env = dict(values) if type(values) is dict else {}
        tainted = {'rank'} if 'rank' in env else set()
        signature = []
        rank_sensitive = bool(tainted) or any(rank_call(n)
                                             for tree in trees for n in ast.walk(tree))
        if rank_sensitive:
            for tree in trees:
                for n in ast.walk(tree):
                    if isinstance(n, (ast.AugAssign, ast.NamedExpr, ast.Try, ast.With, ast.While, ast.For)):
                        reasons.add("restricted rank analysis: unsupported scalar control transfer " + type(n).__name__)
                    if isinstance(n, (ast.If, ast.IfExp)) and any(isinstance(child, (ast.Assign, ast.AnnAssign)) for child in ast.walk(n)):
                        reasons.add("restricted rank analysis: conditional scalar assignment")
                    if isinstance(n, ast.Assign) and any(not isinstance(target, ast.Name) for target in n.targets):
                        reasons.add("restricted rank analysis: non-scalar assignment target")
                    if rank_call(n) and (n.args or n.keywords):
                        reasons.add("restricted rank analysis: subgroup rank needs explicit mapping")
        def has_rank(node):
            return any(isinstance(n, ast.Name) and n.id in tainted or
                       rank_call(n)
                       for n in ast.walk(node))
        def evaluate(node):
            if isinstance(node, ast.Constant): return node.value
            if isinstance(node, ast.Name): return env[node.id]
            if isinstance(node, (ast.Tuple,ast.List)): return tuple(evaluate(v) for v in node.elts)
            if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
                return _BINARY[type(node.op)](evaluate(node.left),evaluate(node.right))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op,(ast.Not,ast.USub,ast.UAdd)):
                value=evaluate(node.operand)
                return not value if isinstance(node.op,ast.Not) else -value if isinstance(node.op,ast.USub) else +value
            if isinstance(node, ast.Compare):
                left=evaluate(node.left)
                for op,rightnode in zip(node.ops,node.comparators):
                    right=evaluate(rightnode)
                    if not _COMPARE[type(op)](left,right): return False
                    left=right
                return True
            if isinstance(node, ast.BoolOp):
                vals=[evaluate(v) for v in node.values]
                return all(vals) if isinstance(node.op,ast.And) else any(vals)
            if isinstance(node, ast.Call):
                name=ast.unparse(node.func)
                if rank_call(node):
                    if node.args or node.keywords: raise ValueError('group rank requires explicit mapping')
                    return rank
                if name in ('min','max','int','bool'):
                    return {'min':min,'max':max,'int':int,'bool':bool}[name](*(evaluate(a) for a in node.args))
            raise ValueError('unresolved scalar expression')
        def record(node):
            if not has_rank(node): return
            try: signature.append((ast.dump(node,include_attributes=False),evaluate(node)))
            except (KeyError,ValueError,TypeError,ZeroDivisionError):
                reasons.add('restricted rank analysis: unresolved rank expression ' + ast.unparse(node))
        for tree in trees:
            # Source order suffices for this bounded scalar analysis; branch
            # assignments are conservatively rejected when values differ.
            for node in ast.walk(tree):
                if isinstance(node,(ast.Assign,ast.AnnAssign)) and node.value is not None:
                    targets=node.targets if isinstance(node,ast.Assign) else (node.target,)
                    for target in targets:
                        if isinstance(target,ast.Name) and has_rank(node.value):
                            tainted.add(target.id)
                            try: env[target.id]=evaluate(node.value)
                            except (KeyError,ValueError,TypeError,ZeroDivisionError):
                                reasons.add('restricted rank analysis: unresolved rank assignment')
                if isinstance(node,(ast.If,ast.While,ast.IfExp)): record(node.test)
                elif isinstance(node,ast.For): record(node.iter)
                elif isinstance(node,ast.Return) and node.value is not None: record(node.value)
                elif isinstance(node,ast.Call):
                    if rank_call(node): continue
                    for argument in (*node.args,*(kw.value for kw in node.keywords)): record(argument)
        canonical = argument_evidence[rank] if argument_evidence is not None else values
        params = {name:value for name,value in canonical.items() if name != 'rank'} if type(canonical) is dict else canonical
        return RankEquivalenceEvidence(definitions, params, tuple(signature), {'rank':rank},
            complete=not reasons, unresolved=tuple(sorted(reasons)),
            covered_obligations=required if not reasons else (),
            class_identity={'pending_cohort':True})
    certificates = {rank:evidence(rank) for rank in arguments}
    cohorts = {}
    for rank, certificate in certificates.items():
        key = digest((certificate.semantic_inputs, certificate.control_signature))
        cohorts.setdefault(key, []).append(rank)
    for members in cohorts.values():
        for rank in members:
            certificates[rank] = replace(certificates[rank], class_identity={'representative':min(members)})
    return certificates.__getitem__
