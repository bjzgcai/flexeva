"""Borrow a native Code context inside the current Python emission.

The generated extension keeps normal PyTorch/pybind entrypoints. This adapter
adds observations to the existing sink; it never constructs an evaluator Session.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import ctypes as C
import hashlib
import json
import sys
from pathlib import Path

from .evidence import CodeContractError, encode


class _ScalarValue(C.Union):
    _fields_ = [("boolean", C.c_uint8), ("i32", C.c_int32), ("u32", C.c_uint32),
                ("i64", C.c_int64), ("u64", C.c_uint64), ("f32", C.c_float), ("f64", C.c_double)]


class _Scalar(C.Structure):
    _fields_ = [("kind", C.c_uint32), ("reserved", C.c_uint32), ("value", _ScalarValue)]


class _Invocation(C.Structure):
    _fields_ = [("abi_version", C.c_uint32), ("rank", C.c_uint32),
                ("world_size", C.c_uint32), ("reserved", C.c_uint32),
                ("arguments", C.POINTER(_Scalar)), ("size", C.c_size_t),
                ("result", C.POINTER(_Scalar))]


class _Path(C.Structure):
    _fields_ = [("site", C.c_uint64), ("ordinal", C.c_uint64),
                ("kind", C.c_uint32), ("reserved", C.c_uint32)]


class _Occurrence(C.Structure):
    _fields_ = [("site", C.c_uint64), ("ordinal", C.c_uint64), ("order", C.c_uint64),
                ("kind", C.c_uint32), ("rank", C.c_uint32),
                ("path", C.POINTER(_Path)), ("path_size", C.c_size_t)]


_Begin = C.CFUNCTYPE(C.c_int, C.c_void_p, C.POINTER(_Occurrence), C.POINTER(_Scalar), C.c_size_t, C.POINTER(C.c_uint64))
_End = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_uint64)
_Branch = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_uint64, C.c_int)
_Definition = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_uint64, _Scalar)
_Ground = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_uint64, C.POINTER(_Scalar), C.c_size_t, C.POINTER(_Scalar))
_Communicator = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint32)
_Fail = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_char_p)


class _Callbacks(C.Structure):
    _fields_ = [("abi_version", C.c_uint32), ("reserved", C.c_uint32),
                ("begin", _Begin), ("end", _End), ("branch", _Branch),
                ("definition", _Definition), ("ground", _Ground),
                ("communicator", _Communicator), ("fail", _Fail)]


def _scalar(value):
    fields = {1: "boolean", 2: "i32", 3: "u32", 4: "i64", 5: "u64", 6: "f32", 7: "f64"}
    if value.reserved or value.kind not in fields:
        raise CodeContractError("invalid scalar from native Code runtime")
    return value.kind, getattr(value.value, fields[value.kind])


class NativeObserverFactory:
    """Register callable -> generated extension artifact (plan.json/workload.so).

    Include ``identity`` in workload registration evidence. Artifact bytes are
    pinned when this factory is constructed and checked before attachment.
    """
    def __init__(self, runtime_library, extensions):
        self.runtime = C.CDLL(str(Path(runtime_library).resolve()), mode=C.RTLD_GLOBAL)
        self.runtime.flexeva_code_current_occurrence_v1.restype = C.POINTER(_Occurrence)
        self.runtime.flexeva_code_attach_v1.argtypes = [C.POINTER(_Invocation), C.POINTER(_Callbacks), C.c_void_p, C.POINTER(C.c_void_p)]
        self.runtime.flexeva_code_attach_v1.restype = C.c_int
        self.runtime.flexeva_code_detach_v1.argtypes = [C.c_void_p]
        self.runtime.flexeva_code_detach_v1.restype = C.c_int
        self.runtime.flexeva_code_fail_v1.argtypes = [C.c_void_p, C.c_char_p]
        self.runtime.flexeva_code_fail_v1.restype = C.c_int
        self.extensions = {}
        for target, artifact in extensions.items():
            path = Path(artifact).resolve()
            plan_path = path / "plan.json" if path.is_dir() else path.with_name("plan.json")
            library = path / "workload.so" if path.is_dir() else path
            plan = json.loads(plan_path.read_text())
            loaded = C.CDLL(str(library))
            loaded.flexeva_code_plan_v1.restype = C.c_char_p
            if json.loads(loaded.flexeva_code_plan_v1()) != plan:
                raise CodeContractError("native loaded plan differs from the registered artifact")
            module_path = getattr(sys.modules.get(getattr(target, "__module__", "")), "__file__", None)
            if module_path is None or not library.samefile(module_path):
                raise CodeContractError("native callable does not belong to the registered artifact")
            stat = library.stat()
            fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            self.extensions[target] = (plan, library,
                                       hashlib.sha256(library.read_bytes()).hexdigest(), fingerprint, loaded)
        self.identity = tuple(sorted((plan["source_digest"], digest)
                                     for plan, _, digest, _, _ in self.extensions.values()))

    def __call__(self, rank, sink, writer):
        def observe(target):
            registered = self.extensions.get(getattr(target, "callable", target))
            if registered is None:
                return nullcontext()
            return self._attach(rank, sink, writer, registered)
        return observe

    @contextmanager
    def _attach(self, rank, sink, writer, registered):
        plan, library, digest, fingerprint, _ = registered
        stat = library.stat()
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != fingerprint:
            raise CodeContractError("native extension changed after workload registration")
        if not writer.stack:
            raise CodeContractError("native extension must execute inside a Python emission")
        parent_key = writer.stack[-1]
        parent = writer.event_id(parent_key)
        sites = {int(site["id"]): site for site in plan["sites"]}
        complete = bool(plan.get("local_reuse_complete"))
        stack, writers, errors, state = [], {}, [], {"last": 0, "predicates": {}}

        def protect(function):
            def guarded(*args):
                try:
                    function(*args)
                    return 0
                except BaseException as error:
                    errors.append(error)
                    return -1
            return guarded

        def begin(_, pointer, inputs, size, token):
            occurrence = pointer.contents
            site = sites[occurrence.site]
            path = tuple((sites[occurrence.path[i].site]["coordinate"], occurrence.path[i].ordinal,
                          occurrence.path[i].kind) for i in range(occurrence.path_size))
            key = ("cpp-extension", parent_key, path)
            identity = writer.event_id(key)
            dependencies = [parent]
            dependencies += [writers[v] for v in site.get("uses", ()) if v in writers]
            if state["last"]:
                dependencies.append(state["last"])
            if stack:
                dependencies.append(stack[-1][0])
            dependencies = tuple(sorted(set(dependencies)))
            evidence = encode(("cpp-entry-v1", site["semantic_digest"], plan["environment_identity"],
                               tuple(_scalar(inputs[i]) for i in range(size)), tuple(sorted(state["predicates"].items()))))
            packet = encode((key, occurrence.order, evidence.hex()))
            for logical_rank in writer.group.active_ranks:
                row = writer._row(key, evidence, complete=complete and site["kind"] != "cleanup", reusable=complete and site["kind"] != "cleanup",
                                  trace=occurrence.kind in (1, 5),
                                  site=("cpp", site["coordinate"]), rank=logical_rank)
                row["inputs"] = dependencies
                sink.declare_region(row)
            sink.begin_region(identity, rank)
            stack.append((identity, site, dependencies, packet))
            token[0] = identity

        def end(_, identity):
            if not stack or stack[-1][0] != identity:
                raise CodeContractError("native extension scope mismatch")
            _, site, dependencies, packet = stack.pop()
            sink.code_occurrence(identity, packet, (), dependencies)
            sink.end_region()
            for var in site.get("defines", ()):
                writers[var] = identity
            if site.get("variable"):
                writers[site["variable"]] = identity
            if site.get("state_effect"):
                state["last"] = identity

        def fail(_, message):
            errors.append(CodeContractError(message.decode(errors="replace") if message else "native extension failed"))
            return 0

        def unsupported(*_):
            raise CodeContractError("native extension scalar control/Ground requires a registered bridge rule")

        def event(site, values):
            occurrence = self.runtime.flexeva_code_current_occurrence_v1()
            if not occurrence or occurrence.contents.site != site:
                raise CodeContractError("native scalar event has no active occurrence")
            token = C.c_uint64()
            begin(None, occurrence, values, len(values), C.pointer(token))
            end(None, token.value)

        def branch(_, site, taken):
            state["predicates"][site] = bool(taken)
            value = _Scalar();value.kind = 1;value.value.boolean = bool(taken)
            event(site, (_Scalar * 1)(value))

        def definition(_, site, value):
            event(site, (_Scalar * 1)(value))

        def communicator(_, handle, membership, local):
            sink.communicator(handle, membership or 1, local)

        callbacks = _Callbacks(1, 0, _Begin(protect(begin)), _End(protect(end)),
            _Branch(protect(branch)), _Definition(protect(definition)),
            _Ground(protect(unsupported)), _Communicator(protect(communicator)), _Fail(fail))
        invocation = _Invocation(1, rank, max(writer.workload.ranks) + 1, 0, None, 0, None)
        handle = C.c_void_p()
        if self.runtime.flexeva_code_attach_v1(C.byref(invocation), C.byref(callbacks), None, C.byref(handle)):
            raise errors[0] if errors else CodeContractError("could not attach native Code context")
        try:
            yield
        except BaseException as error:
            self.runtime.flexeva_code_fail_v1(handle, str(error).encode())
            if errors:
                raise errors[0] from error
            raise
        finally:
            status = self.runtime.flexeva_code_detach_v1(handle)
            if status and not errors:
                errors.append(CodeContractError("native Code context did not close cleanly"))
        if errors:
            raise errors[0]
