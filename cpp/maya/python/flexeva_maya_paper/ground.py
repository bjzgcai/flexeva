"""Numerical Ground rules for explicit, immutable CPU input boundaries.

These rules do not read FakeCUDA tensor contents or execute a training prefix.
The caller binds the rule to its Code site and passes the resulting fact to the
same workload invocation. Core owns request matching, parent reuse and atomic
publication; this module contains only input encoding and numerical handlers.
"""

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
from math import isfinite, prod
import struct
import sys


SCALAR_QUERY = "maya.scalar-control.v1"
MOE_QUERY = "maya.moe-route.v1"
SCALAR_INPUT_SCHEMA = 0x4D530001
SCALAR_FACT_SCHEMA = 0x4D530002
MOE_INPUT_SCHEMA = 0x4D4F0001
MOE_FACT_SCHEMA = 0x4D4F0002
_MAGIC = b"maya-ground-cpu-v1\x00"


@dataclass(frozen=True)
class FrozenCpuTensor:
    """A detached CPU tensor snapshot; no live object or filename is retained."""

    shape: tuple[int, ...]
    dtype: str
    data: bytes

    def __post_init__(self):
        widths = {"torch.float32": 4, "torch.int64": 8}
        if (type(self.shape) is not tuple or any(type(n) is not int or n < 0 for n in self.shape)
                or self.dtype not in widths or type(self.data) is not bytes
                or len(self.data) != prod(self.shape) * widths[self.dtype]):
            raise ValueError("CPU tensor bytes must match the declared shape and dtype")
        if sys.byteorder != "little":
            raise ValueError("Ground CPU tensor encoding requires little-endian input")

    @classmethod
    def freeze(cls, tensor):
        import torch

        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.dtype not in (torch.float32, torch.int64) or tensor.layout != torch.strided):
            raise ValueError("Ground requires real strided CPU float32 or int64 tensors")
        tensor = tensor.detach().resolve_conj().resolve_neg().contiguous()
        return cls(tuple(tensor.shape), str(tensor.dtype), tensor.numpy().tobytes())

    def thaw(self):
        import torch

        dtype = {"torch.float32": torch.float32, "torch.int64": torch.int64}[self.dtype]
        if not self.data:
            return torch.empty(self.shape, dtype=dtype)
        # Copy into an owned writable buffer; mutating a thawed tensor cannot
        # change the immutable evidence or a subsequent consumer's tensor.
        return torch.frombuffer(bytearray(self.data), dtype=dtype).reshape(self.shape)


def _pack(value):
    """Canonical metadata plus raw tensor bytes, without pickle or base64."""
    bodies = []

    def encode(item):
        if isinstance(item, FrozenCpuTensor):
            bodies.append(item.data)
            return {"$cpu_tensor": [item.shape, item.dtype, len(item.data)]}
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise TypeError("Ground metadata keys must be strings")
            return {key: encode(item[key]) for key in sorted(item)}
        if isinstance(item, (tuple, list)):
            return [encode(entry) for entry in item]
        if type(item) not in (str, int, float, bool, type(None)):
            raise TypeError("unsupported Ground metadata type")
        return item

    header = json.dumps(encode(value), sort_keys=True, separators=(",", ":"),
                        allow_nan=False).encode()
    return _MAGIC + struct.pack("<Q", len(header)) + header + b"".join(bodies)


def _unpack(packet):
    if type(packet) is not bytes or not packet.startswith(_MAGIC) or len(packet) < len(_MAGIC) + 8:
        raise ValueError("invalid Ground CPU packet")
    size = struct.unpack_from("<Q", packet, len(_MAGIC))[0]
    offset = len(_MAGIC) + 8 + size
    if offset > len(packet):
        raise ValueError("truncated Ground CPU metadata")
    value = json.loads(packet[len(_MAGIC) + 8:offset])

    def decode(item):
        nonlocal offset
        if isinstance(item, dict):
            if set(item) == {"$cpu_tensor"}:
                shape, dtype, length = item["$cpu_tensor"]
                if type(length) is not int or length < 0 or offset + length > len(packet):
                    raise ValueError("truncated Ground CPU tensor")
                result = FrozenCpuTensor(tuple(shape), dtype, packet[offset:offset + length])
                offset += length
                return result
            return {key: decode(entry) for key, entry in item.items()}
        if isinstance(item, list):
            return tuple(decode(entry) for entry in item)
        return item

    result = decode(value)
    if offset != len(packet):
        raise ValueError("unclaimed bytes in Ground CPU packet")
    return result


def _float32(value, name):
    import torch

    if not isinstance(value, FrozenCpuTensor) or value.dtype != "torch.float32":
        raise ValueError(f"{name} must be a frozen float32 CPU numerical input")
    result = value.thaw()
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain finite numerical values")
    return result


def _finite_number(value, name):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class ScalarControlInputs:
    """Exact input of sum(activation * scale) > threshold in float32."""

    activation: FrozenCpuTensor
    scale: float = 1.0
    threshold: float = 0.0

    @cached_property
    def _evidence(self):
        return _pack({"query": SCALAR_QUERY, "activation": self.activation,
                      "scale": self.scale, "threshold": self.threshold})

    def evidence_bytes(self):
        return self._evidence

    @classmethod
    def from_bytes(cls, packet):
        values = _unpack(packet)
        if not isinstance(values, dict) or values.pop("query", None) != SCALAR_QUERY:
            raise ValueError("not scalar-control Ground evidence")
        return cls(**values)


@dataclass(frozen=True)
class MoeRouteInputs:
    """Actual router boundary, with numerical and ownership dependencies.

    expert_owner contains contiguous EP-local ranks, as in the remote rule.
    The evaluator separately maps those ranks into a logical communicator.
    """

    activation: FrozenCpuTensor
    gate_weight: FrozenCpuTensor
    top_k: int
    capacity_factor: float
    expert_owner: tuple[int, ...]
    projection: str = "linear"

    def __post_init__(self):
        if type(self.expert_owner) is not tuple:
            raise ValueError("expert_owner must be an immutable tuple")

    @cached_property
    def _evidence(self):
        return _pack({"query": MOE_QUERY, "activation": self.activation,
                      "gate_weight": self.gate_weight, "top_k": self.top_k,
                      "capacity_factor": self.capacity_factor, "expert_owner": self.expert_owner,
                      "projection": self.projection})

    def evidence_bytes(self):
        return self._evidence

    @classmethod
    def from_bytes(cls, packet):
        values = _unpack(packet)
        if not isinstance(values, dict) or values.pop("query", None) != MOE_QUERY:
            raise ValueError("not MoE-route Ground evidence")
        return cls(**values)


@dataclass(frozen=True)
class GroundFact:
    """Immutable numerical result and its complete semantic evidence.

    ``input_digest`` is a lookup aid only: Core must compare input evidence (or
    its immutable content identity), not treat the digest as an equality proof.
    Each access to ``value`` gives the consumer a separate decoded view.
    """

    schema: str
    rule_identity: bytes
    input_evidence: bytes
    payload: bytes
    reusable: bool = True

    @property
    def input_digest(self):
        return hashlib.sha256(self.input_evidence).hexdigest()

    @property
    def value(self):
        return _unpack(self.payload)


def _torch_policy():
    import torch

    return (str(torch.__version__), torch.version.git_version, torch.get_num_threads(),
            torch.get_float32_matmul_precision(), torch.backends.mkldnn.enabled,
            torch.are_deterministic_algorithms_enabled())


class _CpuGround:
    def __init__(self):
        self.torch_policy = _torch_policy()

    @property
    def rule_identity(self):
        return _pack({"rule": self.rule, "torch_policy": self.torch_policy})

    def matches(self, query):
        # Must also run before a parent cache hit; changing numerical policy
        # cannot inherit a fact established under the old registration.
        self.validate_policy()
        return query == self.query

    def validate_policy(self):
        if _torch_policy() != self.torch_policy:
            raise RuntimeError("Torch numerical policy changed after Ground registration")

    def __call__(self, evidence):
        """Core callback adapter: one complete evidence packet, one fact."""
        return self.resolve(self.input_type.from_bytes(evidence))

    def request(self, inputs, *, request_id, point_id, requested_lanes):
        """Build a request for ``sink.ground`` from a frozen numerical input.

        Source-site and Lane binding remain explicit frontend responsibilities.
        Core compares the complete input payload, not just input_semantic.
        """
        self.validate_policy()
        if not isinstance(inputs, self.input_type):
            raise TypeError("Ground request has the wrong numerical input type")
        lanes = tuple(requested_lanes)
        if (type(request_id) is not int or request_id <= 0
                or type(point_id) is not int or point_id <= 0
                or not lanes or any(type(rank) is not int or rank < 0 for rank in lanes)
                or len(set(lanes)) != len(lanes)):
            raise ValueError("Ground needs positive request/site IDs and distinct logical ranks")
        return {"request_id": request_id, "point_id": point_id, "requested_lanes": tuple(sorted(lanes)),
                "query": self.query.encode(), "inputs": inputs.evidence_bytes(),
                "input_schema": self.input_schema, "input_semantic": self.query.encode()}

    def registration(self, *, registration_id, layer=1):
        """Adapt the numerical rule to the native Session Ground callbacks."""
        self.validate_policy()
        if type(registration_id) is not int or registration_id <= 0:
            raise ValueError("Ground registration ID must be positive")

        def matches(request):
            self.validate_policy()
            return request["query"] == self.query.encode()

        def handle(request):
            if not matches(request) or request["input_semantic"] != self.query.encode():
                raise ValueError("Ground request does not match the registered numerical schema")
            fact = self(bytes(request["inputs"]))
            return {"request_id": request["request_id"], "payload": fact.payload,
                    "semantic": fact.payload, "reusable": fact.reusable}

        return {"id": registration_id, "layer": layer, "identity": self.rule_identity,
                "result_schema": self.fact_schema, "matches": matches, "handler": handle}

    @staticmethod
    def consume(resolution):
        """Read the pinned Core payload used by Code and native execution."""
        return _unpack(bytes(resolution["payload"]))


class ScalarControlGround(_CpuGround):
    query = SCALAR_QUERY
    schema = "maya.scalar-control.fact.v1"
    rule = "cpu-fp32-scaled-sum-strict-greater-v1"
    input_type = ScalarControlInputs
    input_schema = SCALAR_INPUT_SCHEMA
    fact_schema = SCALAR_FACT_SCHEMA

    def resolve(self, inputs):
        import torch

        self.validate_policy()
        if not isinstance(inputs, ScalarControlInputs):
            raise TypeError("scalar-control Ground requires ScalarControlInputs")
        _finite_number(inputs.scale, "scale")
        _finite_number(inputs.threshold, "threshold")
        activation = _float32(inputs.activation, "activation")
        if not activation.numel():
            raise ValueError("scalar-control Ground requires a nonempty activation")
        with torch.no_grad(), torch.autocast("cpu", enabled=False):
            value = (activation * inputs.scale).sum().item()
        if not isfinite(value):
            raise ValueError("scalar-control Ground produced a nonfinite sum")
        branch = value > inputs.threshold
        result = {"value": value, "predicate": branch, "arm": "large" if branch else "small"}
        return GroundFact(self.schema, self.rule_identity, inputs.evidence_bytes(), _pack(result))


class MoeRouteGround(_CpuGround):
    query = MOE_QUERY
    schema = "maya.moe-route.fact.v1"
    rule = "cpu-fp32-softmax-topk-token-major-capacity-v1"
    input_type = MoeRouteInputs
    input_schema = MOE_INPUT_SCHEMA
    fact_schema = MOE_FACT_SCHEMA

    def resolve(self, inputs):
        import torch
        from torch.nn import functional

        self.validate_policy()
        if not isinstance(inputs, MoeRouteInputs):
            raise TypeError("MoE Ground requires MoeRouteInputs")
        activation = _float32(inputs.activation, "activation")
        weight = _float32(inputs.gate_weight, "gate_weight")
        if (activation.ndim not in (2, 3) or weight.ndim != 2
                or activation.shape[-1] != weight.shape[1] or activation.shape[-1] == 0):
            raise ValueError("router needs compatible [tokens, hidden] and [experts, hidden] inputs")
        activation = activation.reshape(-1, activation.shape[-1])
        tokens, experts = activation.shape[0], weight.shape[0]
        if not tokens or type(inputs.top_k) is not int or not 1 <= inputs.top_k <= experts:
            raise ValueError("router needs tokens and 1 <= top_k <= num_experts")
        _finite_number(inputs.capacity_factor, "capacity_factor")
        if inputs.capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")
        owners = inputs.expert_owner
        if (len(owners) != experts or any(type(rank) is not int or not 0 <= rank < experts for rank in owners)
                or set(owners) != set(range(max(owners, default=-1) + 1))):
            raise ValueError("expert owners must cover contiguous logical ranks")
        with torch.no_grad(), torch.autocast("cpu", enabled=False):
            if inputs.projection == "linear":
                logits = functional.linear(activation, weight)
            elif inputs.projection == "transposed-matmul":
                logits = torch.matmul(weight, activation.transpose(0, 1)).transpose(0, 1)
            else:
                raise ValueError("unknown gate projection policy")
            if not torch.isfinite(logits).all():
                raise ValueError("router produced nonfinite logits")
            probabilities = functional.softmax(logits, dim=-1)
            gates, indices = torch.topk(probabilities, inputs.top_k, dim=-1)
            gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-9)
        capacity_value = inputs.capacity_factor * tokens * inputs.top_k / experts
        if not isfinite(capacity_value):
            raise ValueError("router capacity overflows")
        capacity = max(1, int(capacity_value))
        counts = [0] * experts
        expert_tokens = [[] for _ in range(experts)]
        destination_tokens = [set() for _ in range(max(owners) + 1)]
        positions = [[-1] * inputs.top_k for _ in range(tokens)]
        # Preserve the remote workload's token-major admission order. Two
        # selected experts with the same owner send a token only once there.
        for token, selected in enumerate(indices.tolist()):
            for choice, expert in enumerate(selected):
                if counts[expert] < capacity:
                    positions[token][choice] = counts[expert]
                    counts[expert] += 1
                    expert_tokens[expert].append(token)
                    destination_tokens[owners[expert]].add(token)
        result = {
            "expert_indices": FrozenCpuTensor.freeze(indices), "weights": FrozenCpuTensor.freeze(gates),
            "positions": FrozenCpuTensor.freeze(torch.tensor(positions, dtype=torch.int64)),
            "expert_counts": tuple(counts), "expert_token_indices": tuple(tuple(row) for row in expert_tokens),
            "capacity": capacity, "dropped_assignments": tokens * inputs.top_k - sum(counts),
            "send_counts": tuple(len(row) for row in destination_tokens),
            "active_ranks": tuple(rank for rank, row in enumerate(destination_tokens) if row),
        }
        return GroundFact(self.schema, self.rule_identity, inputs.evidence_bytes(), _pack(result))
