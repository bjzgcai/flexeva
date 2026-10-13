"""Frozen source indexing and exact region selection without importing a workload."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import PurePosixPath

from ..model import SourceSnapshot, SourceSpan, stable_digest
from .region_model import MutableRegion


def module_name(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def normalized(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


@dataclass(frozen=True)
class SourceModule:
    name: str
    path: str
    text: str
    tree: ast.Module


@dataclass(frozen=True)
class SourceDefinition:
    name: str
    module: str
    path: str
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    parent: str


class SourceIndex:
    """A source ownership index; being indexed does not emit a region event."""

    def __init__(self, source: SourceSnapshot, region: MutableRegion):
        self.source = source
        self.region = region
        by_path = {item.path: item for item in source.files}
        if len(by_path) != len(source.files):
            raise ValueError("duplicate frozen source paths")
        if any(item.digest != stable_digest(item.text) for item in source.files):
            raise ValueError("frozen source file digest does not match its text")
        if source.digest != SourceSnapshot.create(source.files).digest:
            raise ValueError("frozen source snapshot digest does not match its files")
        missing = set(region.files) - by_path.keys()
        if missing:
            raise ValueError(f"missing owned source files: {sorted(missing)}")
        self.modules: dict[str, SourceModule] = {}
        self.definitions: dict[str, SourceDefinition] = {}
        for path in region.files:
            if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
                raise ValueError("owned source paths must be relative and cannot escape the snapshot")
            item = by_path[path]
            name = module_name(path)
            if name in self.modules:
                raise ValueError(f"duplicate module identity: {name}")
            tree = ast.parse(item.text, filename=path)
            self.modules[name] = SourceModule(name, path, item.text, tree)
            self._index(tree.body, name, name, path)
        root = self.definitions.get(region.entrypoint)
        if root is None or not isinstance(root.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError(f"region entrypoint is not an owned function: {region.entrypoint}")
        self.root = root
        self.selector_node: ast.AST = root.node
        self.body: tuple[ast.stmt, ...] = tuple(root.node.body)
        self.enclosing_loop: ast.For | None = None
        if region.selector is not None:
            self._select()
        if not self.body:
            raise ValueError("region body is empty")
        self.root_span = SourceSpan(
            root.path, self.body[0].lineno, self.body[0].col_offset,
            self.body[-1].end_lineno, self.body[-1].end_col_offset,
        )

    def _index(self, body, parent: str, module: str, path: str) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{parent}.{node.name}"
                if name in self.definitions:
                    raise ValueError(f"ambiguous source definition: {name}")
                self.definitions[name] = SourceDefinition(name, module, path, node, parent)
                self._index(node.body, name, module, path)
            else:
                for _, value in ast.iter_fields(node):
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, ast.stmt):
                                self._index([item], parent, module, path)
                            elif isinstance(item, (ast.ExceptHandler, ast.match_case)):
                                self._index(item.body, parent, module, path)

    @staticmethod
    def _walk_body(body):
        """Walk one lexical function body, not nested functions or classes."""
        for node in body:
            yield node
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for _, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.stmt):
                            yield from SourceIndex._walk_body([item])
                        elif isinstance(item, (ast.ExceptHandler, ast.match_case)):
                            yield from SourceIndex._walk_body(item.body)

    def _select(self) -> None:
        selector = self.region.selector
        target = normalized(ast.parse(selector.for_target, mode="eval").body)
        iterable = normalized(ast.parse(selector.for_iter, mode="eval").body)
        matches = []
        for node in self._walk_body(self.root.node.body):
            if not isinstance(node, ast.For):
                continue
            # Ignore the AST's Store/Load distinction for the bound loop target.
            if normalized(ast.parse(ast.unparse(node.target), mode="eval").body) != target:
                continue
            if normalized(node.iter) != iterable:
                continue
            for item in self._walk_body(node.body):
                if isinstance(item, ast.With) and any(
                    isinstance(part.context_expr, ast.Call)
                    and ast.unparse(part.context_expr.func) == selector.callee
                    for part in item.items
                ):
                    matches.append((node, item))
        if len(matches) != 1:
            raise ValueError(f"region selector must match exactly once; found {len(matches)}")
        self.enclosing_loop, self.selector_node = matches[0]
        self.body = tuple(self.selector_node.body)

    def module_for(self, owner: str) -> SourceModule:
        if owner in self.definitions:
            return self.modules[self.definitions[owner].module]
        if owner in self.modules:
            return self.modules[owner]
        candidates = [name for name in self.modules if owner.startswith(name + ".")]
        if not candidates:
            raise ValueError(f"unknown source owner: {owner}")
        return self.modules[max(candidates, key=len)]

    def span(self, node: ast.AST, owner: str) -> SourceSpan:
        return SourceSpan(self.module_for(owner).path, node.lineno, node.col_offset,
                          node.end_lineno, node.end_col_offset)

    def text(self, node: ast.AST, owner: str) -> str:
        return ast.get_source_segment(self.module_for(owner).text, node) or ast.unparse(node)
