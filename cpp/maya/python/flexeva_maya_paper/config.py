"""Frozen artifact geometry and explicitly identified candidate changes."""

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class GptConfig:
    name: str
    global_batch_size: int
    seq_len: int
    hidden_size: int
    num_layers: int
    num_heads: int
    vocab_size: int
    tp: int
    pp: int
    dp: int
    micro_batches: int
    dtype: str = "bf16"
    seed: int = 1337
    learning_rate: float = 1e-4

    def __post_init__(self):
        sizes = (self.global_batch_size, self.seq_len, self.hidden_size,
                 self.num_layers, self.num_heads, self.vocab_size,
                 self.tp, self.pp, self.dp, self.micro_batches)
        if any(type(n) is not int or n < 1 for n in sizes):
            raise ValueError("GPT dimensions must be positive integers")
        if self.hidden_size % self.num_heads or self.num_heads % self.tp:
            raise ValueError("GPT hidden/head/tensor-parallel geometry is incompatible")
        if self.dtype not in ("fp32", "bf16"):
            raise ValueError("unsupported GPT dtype")

    @property
    def world_size(self):
        return self.tp * self.pp * self.dp

    @property
    def micro_batch_size(self):
        # Preserve artifact local_micro_batch_size(). The active Figure 7
        # configuration keeps both the nominal and effective batch at 512.
        denominator = self.dp * self.micro_batches
        return max(1, (self.global_batch_size + denominator - 1) // denominator)

    def coordinates(self, rank):
        if not 0 <= rank < self.world_size:
            raise ValueError("logical rank outside configured world")
        return rank // (self.tp * self.pp), (rank // self.pp) % self.tp, rank % self.pp

    def global_rank(self, dp_rank, tp_rank, pp_rank):
        return dp_rank * self.tp * self.pp + tp_rank * self.pp + pp_rank

    def layer_count(self, pp_rank):
        return self.num_layers // self.pp + int(pp_rank < self.num_layers % self.pp)

    def as_dict(self):
        return {**asdict(self), "world_size": self.world_size,
                "micro_batch_size": self.micro_batch_size,
                "effective_global_batch_size": self.micro_batch_size * self.dp * self.micro_batches,
                "steps": 1}

    def frontend_inputs(self):
        return self.as_dict()


def configuration(name="smoke", candidate="P0"):
    if name == "paper":
        # script/e3/figure7/measure_figure7_production.py: ANCHOR and R4.
        result = GptConfig("paper", 512, 2048, 2560, 32, 32, 32000, 1, 8, 2, 256)
    elif name == "smoke":
        result = GptConfig("smoke", 4, 8, 32, 2, 4, 64, 1, 2, 2, 2)
    else:
        raise ValueError("config must be smoke or paper")
    if candidate == "P4":
        result = replace(result, tp=2, dp=1, micro_batches=result.micro_batches * 2)
    elif candidate not in ("P0", "P1", "P2", "P3", "optimizer_patch"):
        raise ValueError("unknown GPT candidate")
    return result


def group_manifest(config):
    groups = []
    for kind_index, kind in enumerate(("tp", "pp", "dp"), 1):
        seen = set()
        for rank in range(config.world_size):
            d, t, p = config.coordinates(rank)
            if kind == "tp":
                members = tuple(config.global_rank(d, x, p) for x in range(config.tp))
                coordinates = d, p
            elif kind == "pp":
                members = tuple(config.global_rank(d, t, x) for x in range(config.pp))
                coordinates = d, t
            else:
                members = tuple(config.global_rank(x, t, p) for x in range(config.dp))
                coordinates = t, p
            if members not in seen:
                seen.add(members)
                a, b = coordinates
                group_id = 3 * ((a + b) * (a + b + 1) // 2 + b) + kind_index
                if group_id >= 1 << 64:
                    raise ValueError("logical communicator identity exceeds uint64")
                groups.append({"id": group_id, "kind": kind, "members": members})
    return groups
