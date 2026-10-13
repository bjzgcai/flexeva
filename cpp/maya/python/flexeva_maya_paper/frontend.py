"""Frozen Python execution wired to the native Code and Maya writers.

Registration describes evaluator rules; only execute invokes the workload.
The Python objects below never retain a CUDA trace or a second RAS graph.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from types import FunctionType, MethodType, ModuleType, GeneratorType

from flexeva_code import (CodeContractError, analyze_source,
                          capture_instrumented, encode, digest, plan_rank_execution)
from flexeva_code.writer import CodeCaptureWriter, CodeEvidenceDictionary, decode_code_packet
from flexeva_code.registration import CodeRegistration
from flexeva.code.dynamic_dependencies import DynamicDependencyTracker, _Origin, _code_value
from flexeva.code.emission import EmissionBinding
from flexeva.code.inputs import DynamicInputEvidence
from flexeva_workload_runtime import WorkloadInvocation


@dataclass(frozen=True)
class CallRule:
    """Explicit foreign-call semantics, separate from source and rank analysis."""
    identity: str
    matches: object
    inputs: object


@dataclass(frozen=True)
class GroundCall:
    registration_id: int
    matches: object
    inputs: object
    handler: object
    select: object
    value_type: str = "json"


class MayaRules:
    """Registered CUDA-producing calls plus explicit workload extensions."""
    def __init__(self, calls=(), *, missing_value=None, materialize_value=None,
                 grounds=(), ground_calls=(), torch_calls=True, autograd_geometry=True):
        self.calls = tuple(calls)
        self._missing_value = missing_value
        self.materialize_value = materialize_value
        self.ground_calls = tuple(ground_calls)
        self.grounds = (*tuple(grounds), *(item.handler.registration(registration_id=item.registration_id)
                                         for item in self.ground_calls))
        self.missing_value = self._missing if missing_value or self.ground_calls else None
        self.torch_calls = torch_calls
        self.autograd_geometry = autograd_geometry
        self.code_contract_identity = {"backend": "native-maya", "version": 2,
            "environment": "restricted-mutable-region-v1",
            "allocator": "uncached-process-context-v1",
            "calls": tuple(rule.identity for rule in self.calls),
            "torch_calls": torch_calls, "autograd_geometry": autograd_geometry,
            "native_extent": "same-emission-v1",
            "grounds": tuple((item["id"], item["identity"].hex()) for item in self.grounds)}

    def ground_call(self, target):
        found = [rule for rule in self.ground_calls if rule.matches(target)]
        if len(found) > 1:
            raise CodeContractError("multiple workload Ground call rules match")
        return found[0] if found else None

    def _missing(self, target):
        rule = self.ground_call(target)
        if rule is not None:
            from flexeva_workload_runtime.code_window import MissingValue
            value = rule.inputs(target)
            return MissingValue({"registration": rule.registration_id},
                                {"bytes": value.evidence_bytes().hex()}, rule.value_type)
        return self._missing_value(target) if self._missing_value else None

    def rule(self, target):
        found = [rule for rule in self.calls if rule.matches(target)]
        if len(found) > 1:
            raise CodeContractError("multiple Maya call rules match one interface")
        return found[0] if found else None

    def prepare_runtime(self, devices):
        """Initialize the fixed CUDA library context outside the modeled trace.

        This bounded preparation executes no user model, optimizer or step and
        remains inside the measured capture callback.
        """
        import torch
        for device in devices:
            with torch.cuda.device(device):
                # The FakeCUDA evaluation process uses a fixed uncached
                # allocator policy. Process-local pool growth (including live
                # returned tensors) must not change equivalent rank traces.
                # This is deliberately persistent in that evaluation process;
                # Python exposes no getter for safely restoring prior policy.
                torch.cuda.init()
                torch.cuda.memory.caching_allocator_enable(False)
                matrix = torch.empty((8, 8), device=device)
                product = torch.mm(matrix, matrix)
                del product, matrix

    @staticmethod
    def source_constructor_supported(name):
        """Known nn.Module constructors whose instances preserve call semantics."""
        return name.rsplit(".", 1)[-1] in {
            "Linear", "Embedding", "LayerNorm", "ModuleList", "Sequential",
            "ReLU", "GELU", "Dropout", "Flatten", "MSELoss", "CrossEntropyLoss"}

    def source_call_supported(self, name):
        """Finite static counterpart of registered ordinary PyTorch effects."""
        leaf = name.rsplit(".", 1)[-1]
        return leaf in {
            "super", "range", "len", "tuple", "list", "dict", "int", "float", "bool", "min", "max", "enumerate", "zip",
            "__init__", "empty", "zeros", "zeros_like", "ones", "ones_like", "randn", "randn_like", "rand", "randint",
            "arange", "full", "full_like", "eye", "stack", "cat", "mm", "bmm", "matmul", "add", "mul", "sub",
            "relu", "sigmoid", "tanh", "sum", "mean", "softmax", "log_softmax", "gelu", "silu", "linear",
            "layer_norm", "embedding", "mse_loss", "cross_entropy", "dropout", "manual_seed", "manual_seed_all",
            "set_device", "empty_cache", "synchronize", "Linear", "Embedding", "LayerNorm", "ModuleList", "Sequential",
            "ReLU", "GELU", "Dropout", "Flatten", "MSELoss", "CrossEntropyLoss", "SGD", "Adam", "AdamW",
            "view", "reshape", "unbind", "transpose", "contiguous", "float", "detach", "div_", "to", "size", "stride",
            "dim", "numel", "square", "pow", "clone", "flatten", "permute", "unsqueeze", "squeeze", "expand", "repeat",
            "chunk", "split", "zero_", "copy_", "add_", "mul_", "requires_grad_", "backward", "step", "zero_grad",
            "parameters", "train", "eval", "get_rank", "get_world_size", "is_initialized", "is_available",
            "init_process_group", "destroy_process_group", "new_group", "all_reduce", "broadcast",
            "all_gather", "all_gather_into_tensor", "reduce_scatter", "reduce_scatter_tensor",
            "all_to_all", "all_to_all_single", "send", "recv", "isend", "irecv", "batch_isend_irecv",
            "P2POp", "barrier", "wait"}

    def source_callable(self, function):
        """Borrow the registered Module.forward for provenance, execute Module.__call__."""
        import torch
        if isinstance(function, torch.nn.Module):
            forward = getattr(type(function), "forward", None)
            if isinstance(forward, FunctionType) and forward.__code__.co_filename.startswith("<flexeva-frozen>/"):
                if any(vars(function).get(name) for name in
                       ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks")):
                    raise CodeContractError("module hooks require a registered callback rule")
                return function.forward
        return None

    def effect_identity(self, target):
        """An audited geometry contract, independent of emission detection."""
        registered = getattr(target.callable, "__flexeva_effect__", None)
        if registered is not None:
            return {"registered": registered}
        rule = self.rule(target)
        if rule is not None:
            return {"registered": rule.identity}
        ground = self.ground_call(target)
        if ground is not None:
            return {"ground": ground.registration_id}
        if not self.torch_calls:
            return None
        import torch
        from torch.nn import functional as F
        fn = target.callable
        receiver = target.receiver if target.receiver is not None else getattr(fn, "__self__", None)
        name = getattr(fn, "__name__", "")
        # These routines have dispatch/launch geometry determined by their
        # explicit scalar options, tensor metadata and resource aliases.
        functions = (torch.empty, torch.zeros, torch.zeros_like, torch.randint,
                     torch.arange, torch.ones, torch.ones_like, torch.randn, torch.randn_like,
                     torch.rand, torch.full, torch.full_like, torch.eye, torch.stack, torch.cat,
                     torch.mm, torch.bmm, torch.matmul, torch.add, torch.mul, torch.sub,
                     torch.relu, torch.sigmoid, torch.tanh, torch.sum, torch.mean,
                     F.softmax, F.log_softmax, F.gelu, F.relu, F.silu, F.linear,
                     F.layer_norm, F.embedding, F.mse_loss, F.dropout,
                     F.cross_entropy, torch.manual_seed, torch.cuda.set_device,
                     torch.cuda.manual_seed_all, torch.cuda.empty_cache,
                     torch.cuda.synchronize, torch.nn.Linear, torch.nn.Embedding,
                     torch.nn.LayerNorm, torch.nn.ModuleList, torch.nn.Sequential,
                     torch.nn.ReLU, torch.nn.GELU, torch.nn.Dropout, torch.nn.Flatten,
                     torch.nn.MSELoss, torch.nn.CrossEntropyLoss,
                     torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW)
        if any(fn is item for item in functions):
            return {"torch_geometry": getattr(fn, "__module__", "") + "." + name}
        if type(fn) in (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm,
                        torch.nn.ReLU, torch.nn.GELU, torch.nn.Dropout, torch.nn.Flatten,
                        torch.nn.MSELoss, torch.nn.CrossEntropyLoss):
            return {"torch_module": type(fn).__qualname__}
        tensor_method = (isinstance(receiver, torch.Tensor)
                         and getattr(type(receiver), name, None) is getattr(torch.Tensor, name, None))
        if tensor_method and name in (
                "view", "reshape", "unbind", "transpose", "contiguous", "float",
                "detach", "div_", "to", "size", "stride", "dim", "numel",
                "sum", "mean", "square", "pow", "relu", "sigmoid", "tanh",
                "softmax", "log_softmax", "clone", "flatten", "permute",
                "unsqueeze", "squeeze", "expand", "repeat", "chunk", "split",
                "add", "mul", "sub", "matmul", "bmm", "mm", "zero_", "copy_",
                "add_", "mul_", "requires_grad_"):
            return {"tensor_geometry": name}
        if tensor_method and name == "backward" and self.autograd_geometry:
            return {"tensor_geometry": "registered-autograd-backward"}
        if type(receiver) in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW) and name in ("step", "zero_grad"):
            if name == "step" and (target.args or target.kwargs.get("closure") is not None):
                return None
            return {"optimizer_geometry": type(receiver).__name__ + "." + name}
        if (name == "__init__" and getattr(fn, "__func__", None) is torch.nn.Module.__init__):
            return {"module_geometry": "Module.__init__"}
        if isinstance(receiver, torch.nn.Module) and name in ("to", "parameters", "train", "eval", "zero_grad"):
            # Custom overrides remain ordinary foreign calls.
            if getattr(type(receiver), name, None) is getattr(torch.nn.Module, name):
                return {"module_geometry": name}
        if target.operation in ("add", "sub", "mul", "truediv", "matmul", "neg", "pos", "pow", "getitem"):
            if any(isinstance(item, torch.Tensor) for item in target.args):
                return {"tensor_operator": target.operation}
        return None

    def resolve_emission(self, target):
        if getattr(target.callable, "__flexeva_effect__", None) is not None:
            return EmissionBinding("maya", "registered-native")
        if self.ground_call(target) is not None:
            return EmissionBinding("maya", "ground-consumer")
        if self.rule(target) is not None:
            return EmissionBinding("maya", self.rule(target).identity)
        if not self.torch_calls or getattr(target, "source_owned", False):
            return None
        import torch
        fn = target.callable
        numeric_conversion = fn in (bool, int, float) and any(isinstance(v, torch.Tensor) for v in target.args)
        masked_index = (target.operation == "getitem" and len(target.args) > 1
                        and any(isinstance(v, torch.Tensor) and v.dtype in (torch.bool, torch.uint8)
                                for v in (target.args[1] if isinstance(target.args[1], tuple) else (target.args[1],))))
        if numeric_conversion or masked_index:
            if self.missing_value is None or self.missing_value(target) is None:
                raise CodeContractError("tensor numerical control requires a registered Ground rule")
            return EmissionBinding("maya", "ground-consumer")
        if isinstance(fn, torch.nn.Module):
            forward = getattr(type(fn), "forward", None)
            if isinstance(forward, FunctionType) and forward.__code__.co_filename.startswith("<flexeva-frozen>/"):
                return None
        module = getattr(fn, "__module__", None) or type(fn).__module__
        receiver = getattr(fn, "__self__", None)
        if (module == "torch" or module.startswith("torch.")
                or isinstance(receiver, (torch.Tensor, torch.optim.Optimizer))
                or target.operation is not None and any(isinstance(value, torch.Tensor)
                    for value in (*target.args, *target.kwargs.values()))):
            if ((target.operation in ("truth", "not") and any(isinstance(v, torch.Tensor) for v in target.args))
                    or isinstance(receiver, torch.Tensor) and getattr(fn, "__name__", "") in
                    ("item", "tolist", "numpy", "__bool__", "__int__", "__float__", "nonzero", "unique", "masked_select")
                    or any(fn is item for item in (torch.nonzero, torch.unique, torch.unique_consecutive,
                                                     torch.masked_select, torch.bincount))
                    or name_is_dynamic_repeat(target, torch)):
                if self.missing_value is None or self.missing_value(target) is None:
                    raise CodeContractError("tensor numerical control requires a registered Ground rule")
            return EmissionBinding("maya", "torch-native")
        return None


def name_is_dynamic_repeat(target, torch):
    if getattr(target.callable, "__name__", "") != "repeat_interleave":
        return False
    args = target.args
    bound = isinstance(getattr(target.callable, "__self__", None), torch.Tensor)
    repeats = args[0] if bound and args else args[1] if len(args) > 1 else args[0] if args else None
    return isinstance(repeats, torch.Tensor) or isinstance(target.kwargs.get("repeats"), torch.Tensor)


@lru_cache(maxsize=256)
def _initializer_identity(code):
    return digest(_code_value(code))


def operand(value, *, seen=None, aliases=None, tensor_values=True):
    """Borrowed geometry/state evidence; no CUDA numerical reads or repr calls."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CodeContractError("non-finite frontend input")
        return value
    if type(value) is slice:
        return {"slice": tuple(operand(v, aliases=aliases, tensor_values=tensor_values) for v in (value.start, value.stop, value.step))}
    if type(value) in (tuple, list):
        return {type(value).__name__: tuple(operand(v, seen=seen, aliases=aliases, tensor_values=tensor_values) for v in value)}
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: operand(v, seen=seen, aliases=aliases, tensor_values=tensor_values) for k, v in value.items()}
    import torch
    if isinstance(value, torch.Size):
        return {"shape": tuple(value)}
    if isinstance(value, torch.Tensor):
        result = {"tensor": tuple(value.shape), "stride": tuple(value.stride()),
                  "dtype": str(value.dtype), "device": str(value.device),
                  "offset": value.storage_offset(), "version": value._version,
                  "requires_grad": value.requires_grad}
        if aliases is not None:
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            result["storage_alias"] = aliases.setdefault(key, len(aliases))
            result["storage_bytes"] = storage.nbytes()
        if value.device.type == "cpu" and tensor_values:
            result["bytes"] = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes().hex()
        elif value.device.type not in ("cpu", "cuda"):
            raise CodeContractError("unsupported tensor device")
        return result
    if isinstance(value, GeneratorType) and value.gi_code in (
            torch.nn.Module.parameters.__code__, torch.nn.Module.named_parameters.__code__):
        return {"parameter_iterator": value.gi_code.co_name}
    if isinstance(value, type):
        if issubclass(value, torch.nn.Module):
            initializer = value.__init__
            return {"module_class": value.__qualname__, "initializer": _initializer_identity(initializer.__code__) if isinstance(initializer, FunctionType) else str(type(initializer))}
        if value in (tuple, list, dict, int, float, bool, range, super):
            return {"builtin_type": value.__qualname__}
    if type(value) in (torch.dtype, torch.device):
        return str(value)
    if type(value) in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
        return {"optimizer": type(value).__name__, "groups": tuple({k: operand(v, aliases=aliases, tensor_values=tensor_values) for k, v in group.items()
                if k != "params"} for group in value.param_groups),
                "parameters": tuple(operand(p, aliases=aliases, tensor_values=tensor_values) for group in value.param_groups for p in group["params"]),
                "gradients": tuple(operand(p.grad, aliases=aliases, tensor_values=tensor_values) for group in value.param_groups for p in group["params"]),
                "state": tuple(operand(value.state.get(p, {}), aliases=aliases, tensor_values=tensor_values) for group in value.param_groups for p in group["params"])}
    if isinstance(value, torch.nn.Module):
        seen = set() if seen is None else seen
        if id(value) in seen:
            return {"shared_module": type(value).__qualname__}
        seen.add(id(value))
        state = vars(value)
        if any(state.get(name) for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks")):
            raise CodeContractError("module hooks require a registered callback rule")
        result = {"module": type(value).__qualname__, "training": state.get("training"),
                  "parameters": {k: operand(v, aliases=aliases, tensor_values=tensor_values) for k, v in state.get("_parameters", {}).items()},
                  "buffers": {k: operand(v, aliases=aliases, tensor_values=tensor_values) for k, v in state.get("_buffers", {}).items()},
                  "children": {k: operand(v, seen=seen, aliases=aliases, tensor_values=tensor_values) for k, v in state.get("_modules", {}).items()},
                  "attributes": {k: operand(v, seen=seen, aliases=aliases, tensor_values=tensor_values) for k, v in vars(value).items()
                                 if not k.startswith("_") and k != "training"}}
        seen.remove(id(value))
        return result
    if isinstance(value, ModuleType):
        return {"module_binding": value.__name__}
    if isinstance(value, (FunctionType, MethodType)):
        fn = getattr(value, "__func__", value)
        return {"function": fn.__module__ + "." + fn.__qualname__,
                "code": hashlib.sha256(fn.__code__.co_code).hexdigest()}
    snapshot = getattr(type(value), "frontend_inputs", None)
    if snapshot is not None:
        return snapshot(value)
    raise CodeContractError("uncovered frontend operand: " + type(value).__qualname__)


class MayaInputs(DynamicDependencyTracker):
    """Keep actual reaching definitions; evaluator rules cover native operands."""
    def __init__(self, rules, source):
        super().__init__()
        self.rules, self.source = rules, source

    @staticmethod
    def _restricted_value(value):
        """The accepted environment excludes arbitrary aliasing and object effects.

        Source-owned modules are handles whose body reads are tracked separately;
        snapshotting the whole model on every host statement would be quadratic.
        Native module/optimizer calls still receive their explicit full geometry.
        """
        import torch
        if isinstance(value, torch.nn.Module):
            return {"module": type(value).__qualname__, "training": value.training}
        if isinstance(value, torch.optim.Optimizer):
            if type(value) not in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
                raise CodeContractError("unregistered optimizer effect")
            return {"optimizer": type(value).__name__}
        return operand(value, tensor_values=False)

    @staticmethod
    def _binding_value(name, bindings):
        # Only ordinary fields in the restricted source environment. No user
        # descriptors, arbitrary getattr, eval, or cross-object alias registry.
        parts = name.split(".")
        if not all(part.isidentifier() for part in parts) or parts[0] not in bindings:
            raise CodeContractError("non-simple source access")
        value = bindings[parts[0]]
        for field in parts[1:]:
            import torch
            if isinstance(value, torch.nn.Module):
                state = vars(value)
                if field in state:
                    value = state[field]
                else:
                    for key in ("_modules", "_parameters", "_buffers"):
                        if field in state.get(key, {}):
                            value = state[key][field]
                            break
                    else:
                        raise CodeContractError("non-field module access")
            elif isinstance(value, torch.Tensor) and field in ("shape", "dtype", "device", "requires_grad", "grad"):
                value = getattr(value, field)
            elif type(value) is dict and field in value:
                value = value[field]
            else:
                raise CodeContractError("non-simple source field")
        return value

    def _lookup(self, name, scope, bindings):
        origin = super()._lookup(name, scope, bindings)
        try:
            evidence = self._restricted_value(self._binding_value(name, bindings))
        except (CodeContractError, TypeError, ValueError, AttributeError):
            return origin
        root = name.split(".", 1)[0]
        covered = ("source binding or alias is not value-complete: ",
                   "defined value alias effects are not covered: ",
                   "callable/module binding is not value-complete: ",
                   "opaque or aliasable operand: $parameter:")
        unknown = {reason for reason in origin.unknown
                   if not any(reason == prefix + root or reason == prefix + name
                              for prefix in covered)}
        return _Origin(set(origin.producers), {**origin.values, name: evidence}, unknown)

    def _operand(self, result, name, value):
        try:
            result.values[name] = self._restricted_value(value)
        except (CodeContractError, TypeError, ValueError, AttributeError):
            super()._operand(result, name, value)

    def complete(self, occurrence, *, result=None, error=None):
        super().complete(occurrence, result=result, error=error)
        if error is not None or occurrence.kind != "block":
            return
        definitions = self._definitions.get(self._occurrence_scope(occurrence), {})
        for name, value in occurrence.inputs.get("defined_values", {}).items():
            try:
                self._restricted_value(value)
            except (CodeContractError, TypeError, ValueError, AttributeError):
                continue
            if name in definitions:
                definitions[name].unknown.discard("defined value alias effects are not covered: " + name)

    def __call__(self, occurrence):
        base = super().__call__(occurrence)
        site = occurrence.site
        inputs = {"site": site.get("normalized", site.get("source_text")),
                  "control": {}, "bindings": {}}
        unresolved = set(base.unresolved)
        aliases = {}
        covered_bindings = set()
        bindings = occurrence.inputs.get("source_bindings", {})
        # Each executed source read participates, without serializing unrelated
        # locals or performing arbitrary property access for dotted symbols.
        for name in site.get("uses", ()):
            if name in bindings:
                if occurrence.kind == "block" and not self._primitive(bindings[name]):
                    # An opaque host-state write is already unresolved in the
                    # base tracker. Do not duplicate whole models/containers
                    # in blocks that cannot certify reuse; actual consumers
                    # snapshot their explicit operands when they are reached.
                    continue
                try:
                    inputs["bindings"][name] = self._restricted_value(bindings[name])
                    covered_bindings.add(name)
                except CodeContractError as error:
                    unresolved.add(str(error))
        if occurrence.kind == "emission":
            from flexeva_workload_runtime.code_window import CodeCallTarget
            live = occurrence.inputs
            target = CodeCallTarget(site, occurrence.lane, live.get("callable"),
                live.get("receiver"), live.get("args", ()), live.get("kwargs", {}),
                live.get("operation"), live.get("source_owned", False))
            rule = self.rules.rule(target)
            effect = self.rules.effect_identity(target)
            call_covered = False
            try:
                if rule is not None:
                    inputs["call"] = rule.inputs(target)
                else:
                    inputs["call"] = {"receiver": operand(target.receiver, aliases=aliases, tensor_values=False),
                        "args": operand(target.args, aliases=aliases, tensor_values=False),
                        "kwargs": operand(dict(target.kwargs), aliases=aliases, tensor_values=False),
                        "callable": (getattr(target.callable, "__module__", None),
                                     getattr(target.callable, "__qualname__", type(target.callable).__qualname__))}
                    import torch
                    if isinstance(target.callable, torch.nn.Module):
                        inputs["call"]["module_state"] = operand(target.callable,
                            aliases=aliases, tensor_values=False)
                if "completed_value" in live:
                    inputs["completed_value"] = operand(live["completed_value"], aliases=aliases, tensor_values=False)
                call_covered = effect is not None
            except (CodeContractError, TypeError, ValueError, AttributeError) as error:
                unresolved.add(str(error))
            inputs["effect_rule"] = effect
            if call_covered:
                # Cover only obligations discharged by this registered rule
                # and the snapshots above. Keep unknown source state intact.
                unresolved = {reason for reason in unresolved if not (
                    reason in ("foreign hidden effects are not covered",
                               "actual callable implementation identity is not covered")
                    or reason.startswith("opaque or aliasable operand: ")
                    or any(reason == "source binding or alias is not value-complete: " + name
                           or reason.startswith("source binding or alias is not value-complete: " + name + ".")
                           or reason == "callable/module binding is not value-complete: " + name
                           for name in covered_bindings))}
            else:
                unresolved.add("registered native input/effect rule is absent or incomplete")
        return DynamicInputEvidence(base.producer_inputs, inputs,
            {"analysis": self.source.policy_identity, "rule": self.rules.code_contract_identity},
            base.state_refs, not unresolved, tuple(sorted(unresolved)))


@dataclass(frozen=True)
class WorkloadRegistration(CodeRegistration):
    rules: MayaRules
    identity: str = "maya-python-source-v2"

    def new_session(self, native, **options):
        return native.new_session(grounds=self.rules.grounds, **options)


class PythonWorkload:
    """A source candidate; registration and captured execution are distinct."""
    @classmethod
    def from_script(cls, sources, script_path, **options):
        """Select a plain script as a window, preserving module global bindings.

        Imports, literal configuration and definitions retain their original
        order as fixed context. The following executable statements form the
        window; a main guard selects its body. Interleaved executable setup and
        later definitions require an explicit function window.
        """
        import ast
        from pathlib import PurePosixPath
        if script_path not in sources:
            raise CodeContractError("script path is absent from frozen sources")
        tree = ast.parse(sources[script_path], filename=script_path)
        context, statements, assigned = [], [], set()
        definitions = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
        cutoff = max((index for index, node in enumerate(tree.body) if isinstance(node, definitions)), default=-1)
        for index, node in enumerate(tree.body):
            if index <= cutoff:
                allowed = isinstance(node, definitions) or (isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                    try:
                        ast.literal_eval(node.value)
                        allowed = True
                    except (ValueError, TypeError):
                        pass
                if not allowed:
                    raise CodeContractError(f"{script_path}:{node.lineno}: executable statement before a later definition; select a function window")
                context.append(node)
                continue
            if isinstance(node, ast.If) and ast.dump(node.test, include_attributes=False) in (
                    ast.dump(ast.parse("__name__ == '__main__'", mode="eval").body, include_attributes=False),
                    ast.dump(ast.parse("'__main__' == __name__", mode="eval").body, include_attributes=False)):
                statements.extend(node.body)
            else:
                statements.append(node)
        for node in statements:
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    assigned.add(child.id)
        name = "__flexeva_script_window__"
        if any(getattr(node, "name", None) == name for node in tree.body):
            raise CodeContractError("reserved script window function already exists")
        body = ([ast.Global(sorted(assigned))] if assigned else []) + (statements or [ast.Pass()])
        function = ast.FunctionDef(name=name, args=ast.arguments(posonlyargs=[], args=[], vararg=None,
            kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=body, decorator_list=[])
        tree.body = [*context, function]
        ast.fix_missing_locations(tree)
        frozen = dict(sources)
        frozen[script_path] = ast.unparse(tree)
        module = str(PurePosixPath(script_path).with_suffix("")).replace("/", ".")
        options.setdefault("arguments", lambda rank: {})
        return cls(frozen, module + "." + name, **options)

    def __init__(self, sources, entrypoint, *, arguments, ranks=(0,), registration=None,
                 rank_evidence=None, rank_policy=None, share=True, communicators=(),
                 rank_projections=(), module_bindings=None):
        self.sources = dict(sources)
        self.entrypoint, self.arguments, self.ranks = entrypoint, arguments, tuple(ranks)
        self.registration = registration or WorkloadRegistration(MayaRules())
        self.module_bindings = module_bindings
        policy = {"frontend": self.registration.identity,
                  "emissions": self.registration.rules.code_contract_identity,
                  "native_observer": getattr(self.registration.interface_observer_factory, "identity", None),
                  "fixed_context": self.registration.context_identity}
        self.source_plan = analyze_source(self.sources, entrypoint, policy_identity=policy)
        self._invocation_arguments = {rank: self.arguments(rank) for rank in self.ranks}
        self._argument_evidence = {rank: operand(value, aliases={})
                                  for rank, value in self._invocation_arguments.items()}
        if rank_evidence is None:
            from flexeva_code.restricted import restricted_rank_evidence
            rank_evidence = restricted_rank_evidence(self.source_plan, self._invocation_arguments,
                call_supported=self.registration.rules.source_call_supported,
                argument_evidence=self._argument_evidence,
                constructor_supported=self.registration.rules.source_constructor_supported)
        self.rank_plan = plan_rank_execution(self.source_plan, self.ranks, rank_evidence,
            policy_identity=rank_policy or policy, share=share)
        self.communicators, self.rank_projections = tuple(communicators), tuple(rank_projections)
        self.last_execution = None
        import ast
        import torch
        devices = set()
        def cuda_devices(value):
            if isinstance(value, torch.Tensor):
                value = value.device
            if isinstance(value, torch.device):
                value = str(value)
            if type(value) is str and (value == "cuda" or value.startswith("cuda:") and value[5:].isdigit()):
                devices.add(value if ":" in value else "cuda:0")
            elif type(value) is dict:
                for item in value.values(): cuda_devices(item)
            elif type(value) in (tuple, list):
                for item in value: cuda_devices(item)
        for value in self._invocation_arguments.values(): cuda_devices(value)
        for text in self.sources.values():
            for node in ast.walk(ast.parse(text)):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if keyword.arg == "device" and isinstance(keyword.value, ast.Constant):
                        cuda_devices(keyword.value.value)
                if isinstance(node.func, ast.Attribute) and node.func.attr in ("to", "device"):
                    if node.args and isinstance(node.args[0], ast.Constant):
                        cuda_devices(node.args[0].value)
        self.runtime_devices = tuple(sorted(devices))

    def describe(self):
        if self._invocation_arguments is None:
            self._invocation_arguments = {rank: self.arguments(rank) for rank in self.ranks}
            self._argument_evidence = {rank: operand(value, aliases={})
                                      for rank, value in self._invocation_arguments.items()}
        arguments = {str(rank): value for rank, value in self._argument_evidence.items()}
        return {"dynamic_regions": True, "regions": [], "dependencies": [],
            "configuration": encode(arguments), "communicators": self.communicators,
            "rank_projections": self.rank_projections,
            "window_semantic": encode({"sources": self.sources, "entrypoint": self.entrypoint,
                                        "registration": self.registration.identity,
                                        "modeled_function": self.registration.modeled_function,
                                        "analysis_policy": self.source_plan.policy_identity,
                                        "arguments": arguments,
                                        "rules": self.registration.rules.code_contract_identity}),
            "window_complete": self.registration.complete_invocation and not self.registration.rules.grounds
                and all(group.reusable for group in self.rank_plan.groups)
                and (self.registration.execution_context_factory is None or self.registration.context_identity is not None),
            "frontend_evidence": encode({"sources": self.sources, "entrypoint": self.entrypoint,
                "rank_plan": self.rank_plan.as_dict(), "rules": self.registration.rules.code_contract_identity})}

    def execute(self, sink):
        from flexeva.code.taint import CodeTaintPolicy
        if self._invocation_arguments is None:
            self.describe()
        dictionary = CodeEvidenceDictionary(sink)
        writers = {rank: CodeCaptureWriter(sink, self, rank, dictionary)
                   for rank in self.rank_plan.representatives}
        outputs, audits, preparation = {}, {}, []
        rules = self.registration.rules
        def publish(event):
            writers[event.actual.representative].publish(event)
        def observed(rank, output, audit):
            outputs[rank], audits[rank] = output, audit
        def invocation(rank):
            value = self._invocation_arguments[rank]
            if operand(value, aliases={}) != self._argument_evidence[rank]:
                raise CodeContractError("invocation arguments changed after candidate description")
            return WorkloadInvocation(rank, value)
        def complete(point, missing):
            rule = next((item for item in rules.ground_calls
                         if item.registration_id == missing.query.get("registration")), None)
            if rule is None:
                raise CodeContractError("missing registered workload Ground completion")
            inputs = rule.handler.input_type.from_bytes(bytes.fromhex(missing.inputs["bytes"]))
            writer = writers[point.lane]
            if not writer.stack:
                raise CodeContractError("Ground has no active emission")
            request = rule.handler.request(inputs,
                request_id=sink.identity(0x434f444504, encode(point.point_id)),
                point_id=sink.identity(0x434f444505, encode(point.site_id)),
                requested_lanes=writer.group.active_ranks)
            resolution = sink.ground(request)
            writer.ground_facts.setdefault(writer.stack[-1], []).append(resolution["fact_id"])
            return rule.select(rule.handler.consume(resolution))
        def region_observer(rank):
            @contextmanager
            def scope(context, bindings):
                import time
                with writers[rank].invocation(context, bindings):
                    started = time.perf_counter()
                    rules.prepare_runtime(self.runtime_devices)
                    preparation.append({"rank": rank, "devices": self.runtime_devices,
                                        "seconds": time.perf_counter() - started})
                    yield
            return scope

        @contextmanager
        def execution_context(rank):
            import torch
            with writers[rank].auxiliary("runtime-lifetime"), torch.autograd.set_multithreading_enabled(False):
                stack = ExitStack()
                try:
                    if self.registration.execution_context_factory is not None:
                        with writers[rank].auxiliary("context-enter"):
                            stack.enter_context(self.registration.execution_context_factory(rank, sink))
                    yield
                finally:
                    if self.registration.execution_context_factory is not None:
                        import sys
                        failed = sys.exc_info()[0] is not None
                        try:
                            with writers[rank].auxiliary("context-exit"):
                                stack.close()
                        except BaseException:
                            if not failed:
                                raise

        summary = capture_instrumented(self.source_plan, self.rank_plan,
            invocation, rules, publish,
            input_provider_factory=lambda rank: MayaInputs(rules, self.source_plan),
            module_bindings_factory=self.module_bindings,
            lifecycle_factory=lambda rank: writers[rank],
            execution_context_factory=execution_context,
            interface_observer_factory=(lambda rank: self.registration.interface_observer_factory(
                rank, sink, writers[rank])) if self.registration.interface_observer_factory else None,
            region_observer_factory=region_observer,
            missing_value_factory=(lambda rank: rules.missing_value) if rules.missing_value else None,
            complete_value=complete if rules.missing_value else None,
            materialize_value_factory=(lambda rank: rules.materialize_value) if rules.materialize_value else None,
            taint_policy=CodeTaintPolicy(sources=()), result_observer=observed)
        self.last_execution = {"summary": summary, "outputs": outputs, "audits": audits,
                               "runtime_preparation": preparation}
        return self.last_execution
