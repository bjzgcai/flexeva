"""Code-owned export of a frozen, dynamically instrumented Python region.

The exported sites are source locations, not a Code RAS. The shared workload
runtime constructs block/emission occurrences only when this source executes.
Export itself neither imports nor executes the workload or an evaluator.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from functools import cached_property
from typing import Any

from ..errors import PipelineContractError
from ..model import SourceFile, SourceSnapshot, SourceSpan, stable_digest
from .region_model import MutableRegion, RegionSelector
from .region_source import SourceIndex


def _same_digest_metadata(value, saved):
    """Compare private metadata snapshots without weakening typed digest checks."""
    kind = type(value)
    if value is saved and kind in (str, bytes, int, float, bool, type(None)):
        return True
    if kind is not type(saved):
        return False
    if kind in (str, bytes, int, bool, type(None)):
        return value == saved
    if kind is float:
        return value.hex() == saved.hex()  # Includes the sign of zero.
    if kind is dict:
        if not value:
            return not saved
        return len(value) == len(saved) and all(
            _same_digest_metadata(key, old_key) and _same_digest_metadata(item, old_item)
            for (key, item), (old_key, old_item) in zip(value.items(), saved.items()))
    if kind in (tuple, list):
        if not value:
            return not saved
        return len(value) == len(saved) and all(
            _same_digest_metadata(item, old) for item, old in zip(value, saved))
    # Unusual metadata still goes through the canonical digest and its checks.
    return False


@dataclass(frozen=True)
class DynamicCodeSite:
    site_id: str
    kind: str
    owner: str
    span: SourceSpan
    source_text: str
    normalized: str
    operation: str | None = None
    callee: str | None = None
    ownership: str = "runtime"
    role: str = "statement"
    defs: tuple[str, ...] = ()
    uses: tuple[str, ...] = ()
    access_paths: tuple[str, ...] = ()
    argument_uses: Mapping[str, Any] = field(default_factory=dict)
    lexical_parent: str | None = None


@dataclass(frozen=True)
class DynamicCodeProgram:
    source: SourceSnapshot
    files: tuple[SourceFile, ...]
    region: MutableRegion
    sites: tuple[DynamicCodeSite, ...]
    owned_definitions: tuple[Mapping[str, Any], ...]
    root_span: SourceSpan
    protocol_version: str = "code-dynamic-v1"
    hook_name: str = "__flexeva_dynamic__"

    @property
    def entrypoint(self) -> str:
        return self.region.entrypoint

    @cached_property
    def _mutable_digest_parts(self):
        # Frozen records/tuples cannot replace their children. Track the mutable
        # subtrees instead of re-serializing all source and instrumented text on
        # every event. Keep nested metadata edits observable; frozen is shallow.
        def mutable(value):
            if type(value) in (str, bytes, int, float, bool, type(None)):
                return ()
            if type(value) is tuple:
                return tuple(part for item in value for part in mutable(item))
            if is_dataclass(value) and value.__dataclass_params__.frozen:
                return tuple(part for item in fields(value)
                             for part in mutable(getattr(value, item.name)))
            return (value,)

        return mutable(self)

    @property
    def program_digest(self) -> str:
        parts = self._mutable_digest_parts
        cached = self.__dict__.get("_program_digest_cache")
        if cached is None or not _same_digest_metadata(parts, cached[0]):
            digest = stable_digest(self)
            cached = (deepcopy(parts), digest)
            object.__setattr__(self, "_program_digest_cache", cached)
        return cached[1]


def export_dynamic_region(
    source: SourceSnapshot,
    region: MutableRegion,
    *,
    hook_name: str = "__flexeva_dynamic__",
) -> DynamicCodeProgram:
    """Freeze executable instrumentation without running a static or dry run.

    The source-owned definition index determines the selected source body.
    Actual callable ownership is resolved when a call occurs, including
    foreign ``Module.__call__`` wrappers that reenter owned ``forward`` code.
    """
    from .dynamic_instrument import instrument

    if not hook_name.isidentifier():
        raise ValueError("dynamic hook name must be a Python identifier")
    index = SourceIndex(source, region)
    for module in index.modules.values():
        for node in ast.walk(module.tree):
            names = []
            if isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.arg):
                names.append(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.alias):
                names.append(node.asname or node.name.split(".")[0])
            if hook_name in names:
                raise ValueError(f"source collides with reserved dynamic hook: {module.path}:{node.lineno}")
    files, sites = instrument(index, hook_name)
    definitions = tuple({"owner": definition.name, "module": definition.module,
                         "path": definition.path, "kind": type(definition.node).__name__,
                         "parameters": [arg.arg for arg in (*definition.node.args.posonlyargs, *definition.node.args.args,
                                                             *definition.node.args.kwonlyargs)]
                         if isinstance(definition.node, (ast.FunctionDef, ast.AsyncFunctionDef)) else [],
                         "source_digest": stable_digest(ast.dump(definition.node, include_attributes=False))}
                        for _, definition in sorted(index.definitions.items()))
    return DynamicCodeProgram(source, files, region, sites, definitions,
                              index.root_span, hook_name=hook_name)


def dynamic_program_envelope(program: DynamicCodeProgram) -> dict[str, Any]:
    """Return the JSON-compatible code-dynamic-v1 worker descriptor."""
    instrumented = {item.path: item.text for item in program.files}
    sites = {}
    for site in program.sites:
        value = asdict(site)
        value["path"] = site.span.path
        sites[site.site_id] = value
    return {
        "protocol_version": program.protocol_version,
        "source": {"files": {item.path: item.text for item in program.source.files},
                   "digest": program.source.digest},
        "files": instrumented,
        "instrumented_files": instrumented,
        "file_order": [item.path for item in program.files],
        "entrypoint": program.entrypoint,
        "region_id": program.region.region_id,
        "region": asdict(program.region),
        "root_span": asdict(program.root_span),
        "hook_name": program.hook_name,
        "sites": sites,
        "site_order": [site.site_id for site in program.sites],
        "owned_definitions": [dict(item) for item in program.owned_definitions],
        "execution_notes": [
            "Sites are static source locations; the runtime creates actual block/emission occurrences.",
            "Caller-frame interpreter intrinsics remain in their original source frame.",
            "Generator send/throw/close are preserved through a per-resume proxy; concrete generator type and isgeneratorfunction reflection are not preserved.",
        ],
        "program_digest": program.program_digest,
    }


def restore_dynamic_program(envelope: Mapping[str, Any]) -> DynamicCodeProgram:
    """Decode and validate a frozen Code descriptor without executing it.

    No source analysis or instrumentation is repeated. SourceSnapshot defines
    canonical source-file ordering; the explicit order arrays recover the
    instrumented-file and site tuples even after sorted-key JSON transport.
    Nested JSON lists and tuples have identical ``stable_digest`` semantics.
    This restores immutable Code metadata, not evaluator or workload state.
    """
    if not isinstance(envelope, Mapping):
        raise PipelineContractError("frozen dynamic program must be a JSON mapping")
    try:
        # Copy the entire transport surface before constructing typed records;
        # no live object or caller-owned mutable mapping is retained.
        payload = json.loads(json.dumps(dict(envelope), allow_nan=False))
        if payload.get("protocol_version") != "code-dynamic-v1":
            raise PipelineContractError("frozen dynamic program requires code-dynamic-v1")

        def file_map(value):
            if (type(value) is not dict or not value
                    or any(type(path) is not str or not path or type(text) is not str
                           for path, text in value.items())):
                raise PipelineContractError("frozen dynamic files require paths and source strings")
            return value

        def order(keys, values, label):
            if (type(keys) is not list or any(type(key) is not str for key in keys)
                    or len(keys) != len(set(keys)) or set(keys) != set(values)):
                raise PipelineContractError(
                    f"frozen dynamic {label} must uniquely cover its complete key set"
                )
            return keys

        source_files = file_map(payload["source"]["files"])
        source = SourceSnapshot.create(tuple(
            SourceFile.from_text(path, text) for path, text in source_files.items()
        ))
        if source.digest != payload["source"]["digest"]:
            raise PipelineContractError("dynamic source snapshot is corrupt")
        instrumented = file_map(payload["files"])
        if instrumented != payload["instrumented_files"]:
            raise PipelineContractError("frozen dynamic files and instrumented_files disagree")
        files = tuple(SourceFile.from_text(path, instrumented[path])
                      for path in order(payload["file_order"], instrumented, "file_order"))

        sites = payload["sites"]
        if type(sites) is not dict:
            raise PipelineContractError("frozen dynamic sites must be a mapping")
        decoded_sites = []
        for key in order(payload["site_order"], sites, "site_order"):
            value = dict(sites[key])
            if value["site_id"] != key or value.pop("path") != value["span"]["path"]:
                raise PipelineContractError("frozen dynamic site identity or source path disagrees")
            value["span"] = SourceSpan(**value["span"])
            for field_name in ("defs", "uses", "access_paths"):
                names = value[field_name]
                if type(names) is not list or any(type(name) is not str for name in names):
                    raise PipelineContractError("frozen dynamic site names must be string arrays")
                value[field_name] = tuple(names)
            decoded_sites.append(DynamicCodeSite(**value))

        region = dict(payload["region"])
        owned_paths = region["files"]
        if type(owned_paths) is not list or any(type(path) is not str for path in owned_paths):
            raise PipelineContractError("frozen dynamic region files must be a string array")
        region["files"] = tuple(owned_paths)
        if region["selector"] is not None:
            region["selector"] = RegionSelector(**region["selector"])
        definitions = payload["owned_definitions"]
        if type(definitions) is not list or any(type(item) is not dict for item in definitions):
            raise PipelineContractError("frozen dynamic owned definitions must be object records")
        program = DynamicCodeProgram(
            source, files, MutableRegion(**region), tuple(decoded_sites), tuple(definitions),
            SourceSpan(**payload["root_span"]), protocol_version=payload["protocol_version"],
            hook_name=payload["hook_name"],
        )
        if program.program_digest != payload["program_digest"]:
            raise PipelineContractError("frozen dynamic program digest mismatch")
        canonical = json.loads(json.dumps(dynamic_program_envelope(program), allow_nan=False))
        if canonical != payload:
            raise PipelineContractError("frozen dynamic descriptor differs from its canonical envelope")
        return program
    except (KeyError, TypeError, ValueError) as error:
        raise PipelineContractError("malformed frozen dynamic program descriptor") from error
