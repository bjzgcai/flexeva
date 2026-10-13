"""Scoped, fixed-group torch.distributed API over in-process FakeNCCL.

This adapter records communication geometry; it never exchanges or invents
Tensor values. Numerical host decisions require the workload's Ground rules.
Install the scope before importing frozen workload modules so imported aliases
refer to these functions. Existing ProcessGroups and background threads are not
supported by this serial execution contract.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import ctypes
import sys

from .logical_dist import UniqueId


@dataclass(frozen=True)
class LogicalGroup:
    members: tuple
    identity: int

    def frontend_inputs(self):
        return {"logical_group": self.identity, "members": self.members}


@dataclass(frozen=True)
class LogicalP2POp:
    op: object
    tensor: object
    peer: int
    group: object
    tag: int

    def frontend_inputs(self):
        return {"logical_p2p": self.op.__name__, "peer": self.peer,
                "group": None if self.group is None else self.group.frontend_inputs(),
                "tag": self.tag, "shape": tuple(self.tensor.shape),
                "dtype": str(self.tensor.dtype), "device": str(self.tensor.device)}


class LogicalWork:
    def __init__(self, owner, stream):
        self.owner, self.stream = owner, stream
        self.complete = False

    def frontend_inputs(self):
        return {"logical_work": 1, "complete": self.complete}

    def wait(self, timeout=None):
        if timeout is not None:
            raise ValueError("logical work supports an unconditional wait; timeout decisions require Ground")
        self.owner._active()
        self.owner._check(self.owner.cuda.cudaStreamSynchronize(self.stream))
        self.complete = True
        return True

    def is_completed(self):
        if not self.complete:
            raise RuntimeError("logical communication polling requires Ground or an explicit wait")
        return True

    def is_success(self):
        return self.is_completed()


def _effect(function):
    function.__flexeva_effect__ = {"logical_distributed": function.__name__, "version": 1}
    return function


class FixedDistributed:
    """One fixed world and its registered subgroups, reused across logical ranks."""
    _names = ("init_process_group", "destroy_process_group", "is_initialized", "is_available",
              "is_nccl_available", "get_rank", "get_world_size", "get_backend", "new_group",
              "get_process_group_ranks", "get_global_rank", "get_group_rank", "all_reduce",
              "broadcast", "all_gather", "all_gather_into_tensor", "reduce_scatter",
              "reduce_scatter_tensor", "send", "recv", "isend", "irecv", "batch_isend_irecv",
              "all_to_all_single", "all_to_all", "barrier", "P2POp")

    def __init__(self, groups, sink, *, world_size=None, library="libnccl.so.2"):
        entries = tuple(groups)
        if world_size is None:
            world_size = max((rank for group in entries for rank in group["members"]), default=-1) + 1
        if type(world_size) is not int or world_size < 1:
            raise ValueError("logical distributed world_size must be positive")
        world = tuple(range(world_size))
        self.groups = {world: LogicalGroup(world, 0)}
        for entry in entries:
            members = tuple(entry["members"])
            if not members or len(set(members)) != len(members) or any(type(r) is not int or r not in world for r in members):
                raise ValueError("invalid fixed communicator membership")
            if members not in self.groups:
                self.groups[members] = LogicalGroup(members, len(self.groups))
        self.world, self.sink = self.groups[world], sink
        self.rank, self.handles, self.identities = None, {}, {}
        self.initialized, self.installed = False, False
        self.lib = ctypes.CDLL(library)
        self.cuda = ctypes.CDLL("libcudart.so.12")
        self.lib.fakecuda_ras_active.argtypes = []
        self.lib.fakecuda_ras_active.restype = ctypes.c_int
        p, n, i = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        signatures = {
            "ncclGetUniqueId": [ctypes.POINTER(UniqueId)],
            "ncclCommInitRank": [ctypes.POINTER(p), i, UniqueId, i],
            "ncclCommDestroy": [p], "ncclGroupStart": [], "ncclGroupEnd": [],
            "ncclAllReduce": [p, p, n, i, i, p, p],
            "ncclBroadcast": [p, p, n, i, i, p, p],
            "ncclAllGather": [p, p, n, i, p, p],
            "ncclReduceScatter": [p, p, n, i, i, p, p],
            "ncclSend": [p, n, i, i, p, p], "ncclRecv": [p, n, i, i, p, p]}
        for name, arguments in signatures.items():
            getattr(self.lib, name).argtypes = arguments
            getattr(self.lib, name).restype = i
        self.cuda.cudaStreamSynchronize.argtypes = [p]
        self.cuda.cudaStreamSynchronize.restype = i

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"logical distributed native operation failed: {status}")

    def _active(self):
        if self.rank is None or not self.initialized:
            raise RuntimeError("logical distributed operation outside an initialized rank scope")

    def _group(self, group=None, *, require_member=True):
        self._active()
        if group is None:
            return self.world
        if type(group) is not LogicalGroup or self.groups.get(group.members) != group:
            raise ValueError("unregistered logical process group")
        if require_member and self.rank not in group.members:
            raise ValueError("active rank is outside the logical process group")
        return group

    def _handle(self, group):
        if group.identity not in self.handles:
            uid = self.identities.get(group.identity)
            if uid is None:
                uid = UniqueId()
                # Fixed registration supplies identity. Avoid a one-time UID
                # native event at the first representative rank: all logical
                # ranks must execute the same protocol structure.
                value = b"flexeva-fixed-python:" + group.identity.to_bytes(8, "little")
                ctypes.memmove(ctypes.addressof(uid), value, len(value))
                self.identities[group.identity] = uid
            handle = ctypes.c_void_p()
            local = group.members.index(self.rank)
            self._check(self.lib.ncclCommInitRank(ctypes.byref(handle), len(group.members), uid, local))
            self.handles[group.identity] = handle
            self.sink.communicator(handle.value, group.members, local)
        return self.handles[group.identity]

    def _tensor(self, tensor, group=None):
        import torch
        selected = self._group(group)
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("logical distributed communication requires a contiguous CUDA tensor")
        dtype = {torch.int8: 0, torch.uint8: 1, torch.int32: 2, torch.int64: 4,
                 torch.float16: 6, torch.float32: 7, torch.float64: 8, torch.bfloat16: 9}.get(tensor.dtype)
        if dtype is None:
            raise ValueError(f"unsupported logical NCCL dtype {tensor.dtype}")
        return selected, self._handle(selected), dtype, torch.cuda.current_stream(tensor.device).cuda_stream

    @staticmethod
    def _compatible(output, tensor, count):
        if (not output.is_cuda or not output.is_contiguous() or output.dtype != tensor.dtype
                or output.device != tensor.device or output.numel() != count):
            raise ValueError("logical collective output geometry mismatch")

    def _reduction(self, op):
        import torch.distributed as dist
        if op is None:
            return 0
        values = (dist.ReduceOp.SUM, dist.ReduceOp.PRODUCT, dist.ReduceOp.MAX, dist.ReduceOp.MIN, dist.ReduceOp.AVG)
        for index, value in enumerate(values):
            if op == value:
                return index
        raise ValueError("unsupported logical reduction operation")

    def _finish(self, stream, asynchronous):
        work = LogicalWork(self, stream)
        if asynchronous:
            return work
        work.wait()
        return None

    @contextmanager
    def rank_scope(self, rank):
        import torch.distributed as dist
        if not self.lib.fakecuda_ras_active():
            raise RuntimeError("logical distributed scope requires active direct capture")
        if self.installed or self.rank is not None or rank not in self.world.members:
            raise ValueError("invalid or nested logical rank scope")
        modules = (dist, dist.distributed_c10d)
        originals = {(module, name): getattr(module, name)
                     for module in modules for name in self._names if hasattr(module, name)}
        self.rank, self.initialized, self.installed = rank, True, True
        for module, name in originals:
            setattr(module, name, getattr(self, name))
        try:
            yield self
        finally:
            propagating = sys.exc_info()[0] is not None
            cleanup_error = None
            for handle in self.handles.values():
                try:
                    self._check(self.lib.ncclCommDestroy(handle))
                except Exception as error:
                    cleanup_error = cleanup_error or error
            self.handles.clear()
            for (module, name), value in originals.items():
                setattr(module, name, value)
            self.rank, self.initialized, self.installed = None, False, False
            if cleanup_error is not None and not propagating:
                raise cleanup_error

    def frontend_inputs(self):
        return {"logical_distributed": 1, "world_size": len(self.world.members),
                "rank": self.rank, "groups": tuple(self.groups)}

    @_effect
    def init_process_group(self, backend=None, init_method=None, timeout=None, world_size=-1, rank=-1,
                           store=None, group_name="", pg_options=None, device_id=None):
        if self.rank is None or backend not in (None, "nccl") or store is not None or pg_options is not None:
            raise ValueError("only registered fixed NCCL groups are supported")
        if world_size not in (-1, len(self.world.members)) or rank not in (-1, self.rank):
            raise ValueError("process group arguments disagree with registered logical world")
        self.initialized = True

    @_effect
    def destroy_process_group(self, group=None):
        selected = self._group(group)
        handle = self.handles.pop(selected.identity, None)
        if handle is not None:
            self._check(self.lib.ncclCommDestroy(handle))
        if group is None:
            self.initialized = False

    @_effect
    def is_initialized(self): return self.initialized
    @_effect
    def is_available(self): return True
    @_effect
    def is_nccl_available(self): return True
    @_effect
    def get_rank(self, group=None):
        members = self._group(group, require_member=False).members
        return members.index(self.rank) if self.rank in members else -1
    @_effect
    def get_world_size(self, group=None):
        members = self._group(group, require_member=False).members
        return len(members) if self.rank in members else -1
    @_effect
    def get_backend(self, group=None): self._group(group); return "nccl"
    @_effect
    def get_process_group_ranks(self, group): return list(self._group(group, require_member=False).members)
    @_effect
    def get_global_rank(self, group, group_rank):
        members = self._group(group, require_member=False).members
        if type(group_rank) is not int or not 0 <= group_rank < len(members):
            raise ValueError("group rank outside registered communicator")
        return members[group_rank]
    @_effect
    def get_group_rank(self, group, global_rank): return self._group(group).members.index(global_rank)

    @_effect
    def new_group(self, ranks=None, timeout=None, backend=None, pg_options=None, use_local_synchronization=False, group_desc=None):
        self._active()
        members = self.world.members if ranks is None else tuple(ranks)
        if backend not in (None, "nccl") or pg_options is not None or members not in self.groups:
            raise ValueError("new_group requires a pre-registered fixed NCCL membership")
        return self.groups[members]

    @_effect
    def all_reduce(self, tensor, op=None, group=None, async_op=False):
        _, handle, dtype, stream = self._tensor(tensor, group)
        self._check(self.lib.ncclAllReduce(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(), dtype, self._reduction(op), handle, stream))
        return self._finish(stream, async_op)

    @_effect
    def broadcast(self, tensor, src=None, group=None, async_op=False, group_src=None):
        selected, handle, dtype, stream = self._tensor(tensor, group)
        if (src is None) == (group_src is None):
            raise ValueError("broadcast requires exactly one global src or group_src")
        root = selected.members.index(src) if src is not None else group_src
        self._check(self.lib.ncclBroadcast(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(), dtype, root, handle, stream))
        return self._finish(stream, async_op)

    @_effect
    def all_gather_into_tensor(self, output_tensor, input_tensor, group=None, async_op=False):
        selected, handle, dtype, stream = self._tensor(input_tensor, group)
        self._compatible(output_tensor, input_tensor, input_tensor.numel() * len(selected.members))
        self._check(self.lib.ncclAllGather(input_tensor.data_ptr(), output_tensor.data_ptr(), input_tensor.numel(), dtype, handle, stream))
        return self._finish(stream, async_op)

    @_effect
    def all_gather(self, tensor_list, tensor, group=None, async_op=False):
        import torch
        selected = self._group(group)
        if len(tensor_list) != len(selected.members): raise ValueError("all_gather requires one output per group member")
        for item in tensor_list: self._compatible(item, tensor, tensor.numel())
        packed = torch.empty((len(tensor_list), tensor.numel()), dtype=tensor.dtype, device=tensor.device)
        work = self.all_gather_into_tensor(packed, tensor, group, True)
        for index, item in enumerate(tensor_list): item.copy_(packed[index].view_as(item))
        return work if async_op else (work.wait() and None)

    @_effect
    def reduce_scatter_tensor(self, output, input, op=None, group=None, async_op=False):
        selected, handle, dtype, stream = self._tensor(input, group)
        self._compatible(output, input, input.numel() // len(selected.members))
        if input.numel() != output.numel() * len(selected.members): raise ValueError("reduce_scatter input count mismatch")
        self._check(self.lib.ncclReduceScatter(input.data_ptr(), output.data_ptr(), output.numel(), dtype, self._reduction(op), handle, stream))
        return self._finish(stream, async_op)

    @_effect
    def reduce_scatter(self, output, input_list, op=None, group=None, async_op=False):
        import torch
        selected = self._group(group)
        if len(input_list) != len(selected.members): raise ValueError("reduce_scatter requires one input per member")
        for item in input_list: self._compatible(item, output, output.numel())
        packed = torch.cat([item.reshape(-1) for item in input_list])
        return self.reduce_scatter_tensor(output, packed, op, group, async_op)

    def _transfer(self, tensor, peer, group, tag, send):
        selected, handle, dtype, stream = self._tensor(tensor, group)
        if tag != 0: raise ValueError("logical NCCL does not support nonzero message tags")
        local = selected.members.index(peer)
        function = self.lib.ncclSend if send else self.lib.ncclRecv
        self._check(function(tensor.data_ptr(), tensor.numel(), dtype, local, handle, stream))
        return LogicalWork(self, stream)

    @_effect
    def isend(self, tensor, dst=None, group=None, tag=0, group_dst=None):
        if (dst is None) == (group_dst is None): raise ValueError("isend requires exactly one peer")
        peer = dst if dst is not None else self._group(group).members[group_dst]
        return self._transfer(tensor, peer, group, tag, True)
    @_effect
    def irecv(self, tensor, src=None, group=None, tag=0, group_src=None):
        if (src is None) == (group_src is None): raise ValueError("irecv requires an explicit peer")
        peer = src if src is not None else self._group(group).members[group_src]
        return self._transfer(tensor, peer, group, tag, False)
    @_effect
    def send(self, tensor, dst=None, group=None, tag=0, group_dst=None):
        self.isend(tensor, dst, group, tag, group_dst).wait()
    @_effect
    def recv(self, tensor, src=None, group=None, tag=0, group_src=None):
        self.irecv(tensor, src, group, tag, group_src).wait()
        return src if src is not None else self._group(group).members[group_src]

    @_effect
    def P2POp(self, op, tensor, peer=None, group=None, tag=0, group_peer=None):
        if getattr(op, "__self__", None) is not self or op.__name__ not in ("isend", "irecv"):
            raise ValueError("P2POp requires scoped isend or irecv")
        if (peer is None) == (group_peer is None):
            raise ValueError("P2POp requires exactly one explicit peer")
        selected = self._group(group)
        peer = peer if peer is not None else selected.members[group_peer]
        if peer not in selected.members or tag != 0:
            raise ValueError("P2POp requires a registered peer and zero tag")
        return LogicalP2POp(op, tensor, peer, group, tag)

    @contextmanager
    def _native_group(self):
        self._check(self.lib.ncclGroupStart())
        try:
            yield
        except BaseException:
            # Always unwind native nesting without hiding the original cause
            # if capture was poisoned by the failed operation.
            self.lib.ncclGroupEnd()
            raise
        else:
            self._check(self.lib.ncclGroupEnd())

    @_effect
    def batch_isend_irecv(self, p2p_op_list):
        self._active()
        with self._native_group():
            works = []
            for item in p2p_op_list:
                name = getattr(item.op, "__name__", "")
                if name not in ("isend", "irecv"): raise ValueError("only explicit isend/irecv may be batched")
                works.append(self._transfer(item.tensor, item.peer, item.group, item.tag, name == "isend"))
            return works

    @_effect
    def all_to_all_single(self, output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False):
        selected, handle, dtype, stream = self._tensor(input, group)
        self._compatible(output, input, output.numel())
        if input.dim() == 0 or output.dim() == 0 or tuple(input.shape[1:]) != tuple(output.shape[1:]):
            raise ValueError("all_to_all_single requires compatible row geometry")
        size = len(selected.members)
        def splits(value, rows):
            if value is None or len(value) == 0:
                if rows % size: raise ValueError("equal all-to-all split requires divisible rows")
                return (rows // size,) * size
            value = tuple(value)
            if len(value) != size or any(type(n) is not int or n < 0 for n in value) or sum(value) != rows:
                raise ValueError("invalid all-to-all split sizes")
            return value
        sends, recvs = splits(input_split_sizes, input.shape[0]), splits(output_split_sizes, output.shape[0])
        row = 1
        for dimension in input.shape[1:]: row *= dimension
        width = input.element_size()
        send_offset = recv_offset = 0
        with self._native_group():
            for peer, (send_rows, recv_rows) in enumerate(zip(sends, recvs)):
                if send_rows:
                    self._check(self.lib.ncclSend(input.data_ptr() + send_offset * row * width, send_rows * row, dtype, peer, handle, stream))
                if recv_rows:
                    self._check(self.lib.ncclRecv(output.data_ptr() + recv_offset * row * width, recv_rows * row, dtype, peer, handle, stream))
                send_offset += send_rows; recv_offset += recv_rows
        return self._finish(stream, async_op)

    @_effect
    def all_to_all(self, output_tensor_list, input_tensor_list, group=None, async_op=False):
        import torch
        size = self.get_world_size(group)
        if len(input_tensor_list) != size or len(output_tensor_list) != size:
            raise ValueError("all_to_all requires one tensor per group member")
        first = input_tensor_list[0]
        for item in (*input_tensor_list, *output_tensor_list): self._compatible(item, first, item.numel())
        packed = torch.cat([item.reshape(-1) for item in input_tensor_list])
        result = torch.empty(sum(item.numel() for item in output_tensor_list), device=first.device, dtype=first.dtype)
        work = self.all_to_all_single(result, packed, [item.numel() for item in output_tensor_list],
                                     [item.numel() for item in input_tensor_list], group, True)
        position = 0
        for item in output_tensor_list:
            item.copy_(result[position:position + item.numel()].view_as(item)); position += item.numel()
        return work if async_op else (work.wait() and None)

    @_effect
    def barrier(self, group=None, async_op=False, device_ids=None):
        import torch
        if device_ids is not None and len(device_ids) != 1:
            raise ValueError("logical barrier requires one device")
        device = device_ids[0] if device_ids else torch.cuda.current_device()
        empty = torch.empty(0, dtype=torch.float32, device=f"cuda:{device}")
        return self.all_reduce(empty, group=group, async_op=async_op)


# Work methods participate in the same effect registry as distributed calls.
for _method in (LogicalWork.wait, LogicalWork.is_completed, LogicalWork.is_success):
    _effect(_method)
