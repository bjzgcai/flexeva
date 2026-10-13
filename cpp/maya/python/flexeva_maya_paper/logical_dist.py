"""Sequential logical-rank NCCL calls; no ProcessGroup or numeric reduction."""

import ctypes


class UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_ubyte * 128)]


class LogicalDist:
    def __init__(self, groups, sink, *, cpu=False):
        self.groups = groups
        self.sink = sink
        self.cpu = cpu
        self.identities = {}
        self.handles = {}
        self.rank = None
        self.calls = 0
        if cpu:
            return
        self.lib = ctypes.CDLL("libnccl.so.2")
        p = ctypes.c_void_p
        self.lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(UniqueId)]
        self.lib.ncclCommInitRank.argtypes = [ctypes.POINTER(p), ctypes.c_int, UniqueId, ctypes.c_int]
        self.lib.ncclCommDestroy.argtypes = [p]
        self.lib.ncclAllReduce.argtypes = [p, p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, p, p]
        self.lib.ncclSend.argtypes = [p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, p, p]
        self.lib.ncclRecv.argtypes = [p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, p, p]
        self.lib.ncclGroupStart.argtypes = []
        self.lib.ncclGroupEnd.argtypes = []

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"FakeCUDA NCCL operation returned {status}")

    def frontend_inputs(self):
        # Native handles are represented by the separately registered logical
        # communicator projection, never by Python object or pointer identity.
        return {"logical_dist": 1, "cpu": self.cpu,
                "groups": tuple((group["kind"], len(group["members"])) for group in self.groups
                                if self.rank is not None and self.rank in group["members"])}

    def begin_rank(self, rank):
        if self.rank is not None or self.handles:
            raise RuntimeError("logical rank already active")
        self.rank = rank
        for group in self.groups:
            members = group["members"]
            if rank not in members or len(members) == 1:
                continue
            group_id = group["id"]
            local_rank = members.index(rank)
            if self.cpu:
                self.handles[group_id] = group_id
                continue
            if group_id not in self.identities:
                uid = UniqueId()
                self._check(self.lib.ncclGetUniqueId(ctypes.byref(uid)))
                self.identities[group_id] = uid
            handle = ctypes.c_void_p()
            self._check(self.lib.ncclCommInitRank(ctypes.byref(handle), len(members),
                                                self.identities[group_id], local_rank))
            self.handles[group_id] = handle
            self.sink.communicator(handle.value, tuple(members), local_rank)

    def end_rank(self):
        try:
            if not self.cpu:
                for handle in self.handles.values():
                    self._check(self.lib.ncclCommDestroy(handle))
        finally:
            self.handles.clear()
            self.rank = None

    def group(self, kind):
        return next(group for group in self.groups
                    if group["kind"] == kind and self.rank in group["members"])

    def _arguments(self, tensor, kind):
        group = self.group(kind)
        if len(group["members"]) == 1:
            return group, None, None, None
        if not tensor.is_contiguous():
            raise ValueError("logical NCCL requires contiguous tensors")
        dtype = {"torch.float32": 7, "torch.bfloat16": 9, "torch.float16": 6}.get(str(tensor.dtype))
        if dtype is None:
            raise ValueError(f"unsupported logical NCCL dtype {tensor.dtype}")
        if self.cpu:
            return group, self.handles[group["id"]], dtype, None
        if not tensor.is_cuda:
            raise ValueError("native logical NCCL requires a CUDA tensor")
        import torch
        stream = torch.cuda.current_stream(tensor.device).cuda_stream
        return group, self.handles[group["id"]], dtype, stream

    def all_reduce(self, tensor, kind):
        group, handle, dtype, stream = self._arguments(tensor, kind)
        if len(group["members"]) > 1:
            self.calls += 1
            if not self.cpu:
                pointer = tensor.data_ptr()
                self._check(self.lib.ncclAllReduce(pointer, pointer, tensor.numel(),
                                                   dtype, 0, handle, stream))
        return tensor

    def transfer(self, tensor, peer, *, send):
        self.exchange(((tensor, peer, send),))

    def exchange(self, transfers):
        """Commit complementary 1F1B operations together before either waits."""
        prepared = []
        for tensor, peer, send in transfers:
            group, handle, dtype, stream = self._arguments(tensor, "pp")
            if peer not in group["members"]:
                raise ValueError("pipeline peer outside PP group")
            prepared.append((tensor, group["members"].index(peer), send, handle, dtype, stream))
        if not prepared:
            return
        self.calls += len(prepared)
        if not self.cpu:
            self._check(self.lib.ncclGroupStart())
            try:
                for tensor, peer, send, handle, dtype, stream in prepared:
                    fn = self.lib.ncclSend if send else self.lib.ncclRecv
                    self._check(fn(tensor.data_ptr(), tensor.numel(), dtype, peer, handle, stream))
            finally:
                self._check(self.lib.ncclGroupEnd())
