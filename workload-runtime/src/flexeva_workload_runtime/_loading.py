"""Backend-neutral loading and cleanup for a single frozen source invocation."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import inspect
import sys
import threading
from collections.abc import Callable, Mapping
from pathlib import PurePosixPath
from types import ModuleType
from typing import Any

from ._types import WorkloadExecutionError, WorkloadInvocation, WorkloadRunResult


class FrozenSourceLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Load declared source and explicit model modules without rereading files."""

    def __init__(
        self, files: Mapping[str, str], dispatch_name: str, dispatch: Any, *,
        globals: Mapping[str, Any] | None = None,
        module_bindings: Mapping[str, ModuleType] | None = None,
        allow_existing_namespaces: bool = False,
    ) -> None:
        self.modules: dict[str, tuple[str, str, bool]] = {}
        self.packages: set[str] = set()
        self.bindings: dict[str, ModuleType] = {}
        for name, module in (module_bindings or {}).items():
            if not isinstance(name, str) or not all(part.isidentifier() for part in name.split(".")) or not isinstance(module, ModuleType):
                raise WorkloadExecutionError("external bindings require named module objects")
            # Import machinery sets metadata and child attributes on packages.
            # Never apply those writes to the caller's reusable module template.
            local = ModuleType(name)
            local.__dict__.update(module.__dict__)
            local.__name__ = name
            self.bindings[name] = local
        for name, module in self.bindings.items():
            parent, _, child = name.rpartition(".")
            if parent in self.bindings:
                setattr(self.bindings[parent], child, module)
        self.dispatch_name, self.dispatch = dispatch_name, dispatch
        self.globals = dict(globals or {})
        if not dispatch_name.isidentifier():
            raise WorkloadExecutionError("reserved hook name must be a Python identifier")
        if dispatch_name in self.globals:
            raise WorkloadExecutionError("host globals cannot override the source hook")
        for path, source in files.items():
            pure = PurePosixPath(path)
            if pure.is_absolute() or ".." in pure.parts or pure.suffix != ".py":
                raise WorkloadExecutionError(f"expected a relative Python source path: {path!r}")
            parts = list(pure.with_suffix("").parts)
            package = parts[-1] == "__init__"
            if package:
                parts.pop()
            name = ".".join(parts)
            if not name or not all(part.isidentifier() for part in parts):
                raise WorkloadExecutionError(f"source has no importable module name: {path!r}")
            if name in self.modules or name in self.bindings:
                raise WorkloadExecutionError(f"duplicate frozen module: {name}")
            if not isinstance(source, str):
                raise WorkloadExecutionError("source contents must be strings")
            self.modules[name] = (path, source, package)
            self.packages.update(".".join(parts[:i]) for i in range(1, len(parts)))
        for name, module in self.bindings.items():
            if not all(part.isidentifier() for part in name.split(".")) or not isinstance(module, ModuleType):
                raise WorkloadExecutionError("external bindings require named module objects")
            parts = name.split(".")
            self.packages.update(".".join(parts[:i]) for i in range(1, len(parts)))
        if allow_existing_namespaces:
            # Dynamic source bundles retain original package paths (for example
            # examples/training) while their caller may already import examples.
            # Reuse only ancestor packages, never a frozen or bound module itself.
            retained = (self.packages & sys.modules.keys()) - self.modules.keys() - self.bindings.keys()
            if any(not hasattr(sys.modules[name], "__path__") for name in retained):
                raise WorkloadExecutionError("frozen source ancestor conflicts with a non-package")
            self.packages.difference_update(retained)
        if self.owned_names & sys.modules.keys():
            raise WorkloadExecutionError("frozen module conflicts with an already imported dependency")

    @property
    def owned_names(self) -> set[str]:
        return self.modules.keys() | self.bindings.keys() | self.packages

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname in self.modules:
            return importlib.util.spec_from_loader(fullname, self, is_package=self.modules[fullname][2])
        if fullname in self.bindings or fullname in self.packages:
            return importlib.util.spec_from_loader(fullname, self, is_package=fullname in self.packages)
        return None

    def create_module(self, spec):
        return self.bindings.get(spec.name)

    def exec_module(self, module):
        if module.__name__ not in self.modules:
            return
        path, source, _ = self.modules[module.__name__]
        module.__file__ = f"<flexeva-frozen>/{path}"
        module.__dict__.update(self.globals)
        module.__dict__[self.dispatch_name] = self.dispatch
        exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102


_IMPORT_LOCK = threading.RLock()


def _invoke_host(function: Callable[..., Any], arguments: Mapping[str, Any]) -> Any:
    """Honor Python's fixed positional-only and keyword parameter semantics."""
    signature = inspect.signature(function)
    keywords = dict(arguments)
    positional: list[Any] = []
    for parameter in signature.parameters.values():
        if parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
            continue
        if parameter.name in keywords:
            positional.append(keywords.pop(parameter.name))
        elif parameter.default is not inspect.Parameter.empty:
            positional.append(parameter.default)
        else:
            raise WorkloadExecutionError(f"missing workload argument: {parameter.name!r}")
    signature.bind(*positional, **keywords)
    return function(*positional, **keywords)


def _run_frozen_source(
    program: Mapping[str, Any], invocation: WorkloadInvocation,
    dispatcher: Any, *,
    module_bindings: Mapping[str, ModuleType] | None = None,
) -> WorkloadRunResult:
    """Shared source loading/cleanup; the caller supplies execution semantics."""
    originals = program["source"]["files"]
    original = invocation.mode == "original"
    hook_name = program.get("hook_name", "__flexeva_dynamic__")
    entrypoint = program["entrypoint"]
    if not isinstance(entrypoint, str) or "." not in entrypoint:
        raise WorkloadExecutionError("entrypoint must be module.function")
    module_name, function_name = entrypoint.rsplit(".", 1)
    with _IMPORT_LOCK:
        loader = FrozenSourceLoader(
            originals if original else program["instrumented_files"], hook_name, dispatcher,
            globals=invocation.globals, module_bindings=module_bindings,
            allow_existing_namespaces=True,
        )
        if module_name not in loader.modules:
            raise WorkloadExecutionError("entrypoint is absent from the frozen source set")
        absent = object()
        parent_attributes = []
        for name in loader.owned_names:
            parent, _, child = name.rpartition(".")
            if parent in sys.modules:
                namespace = sys.modules[parent].__dict__
                parent_attributes.append((namespace, child, namespace.get(child, absent)))
        sys.meta_path.insert(0, loader)
        try:
            module = importlib.import_module(module_name)
            function = getattr(module, function_name)
            output = _invoke_host(function, invocation.arguments)
            audit = dispatcher.finish()
            audit["measured"] = not original
            return WorkloadRunResult(output, audit)
        finally:
            sys.meta_path.remove(loader)
            for name in loader.owned_names:
                sys.modules.pop(name, None)
            for namespace, child, previous in parent_attributes:
                if previous is absent:
                    namespace.pop(child, None)
                else:
                    namespace[child] = previous
