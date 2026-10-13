"""GPT compute structure adapted from the artifact's synthetic Megatron model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ColumnParallelLinear(nn.Module):

    def __init__(self, inputs, outputs, tp):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs // tp, bias=False)

    def forward(self, tensor):
        return self.linear(tensor)

class RowParallelLinear(nn.Module):

    def __init__(self, inputs, outputs, tp, communication):
        super().__init__()
        self.linear = nn.Linear(inputs // tp, outputs, bias=False)
        self.communication = communication

    def forward(self, tensor):
        return self.communication.all_reduce(self.linear(tensor), 'tp')

class MegatronAttention(nn.Module):

    def __init__(self, config, communication):
        super().__init__()
        self.local_heads = config.num_heads // config.tp
        self.head_dim = config.hidden_size // config.num_heads
        self.local_hidden = self.local_heads * self.head_dim
        self.qkv = ColumnParallelLinear(config.hidden_size, config.hidden_size * 3, config.tp)
        self.out_proj = RowParallelLinear(config.hidden_size, config.hidden_size, config.tp, communication)

    def forward(self, tensor):
        batch, sequence, _ = tensor.shape
        qkv = self.qkv(tensor).view(batch, sequence, 3, self.local_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = (item.transpose(1, 2) for item in (query, key, value))
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.head_dim ** (-0.5)
        attention = F.softmax(attention, dim=-1)
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).contiguous().view(batch, sequence, self.local_hidden)
        return self.out_proj(output)

class MegatronMLP(nn.Module):

    def __init__(self, config, communication):
        super().__init__()
        self.up = ColumnParallelLinear(config.hidden_size, config.hidden_size * 4, config.tp)
        self.down = RowParallelLinear(config.hidden_size * 4, config.hidden_size, config.tp, communication)

    def forward(self, tensor):
        return self.down(F.gelu(self.up(tensor)))

class MegatronBlock(nn.Module):

    def __init__(self, config, communication):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attention = MegatronAttention(config, communication)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MegatronMLP(config, communication)

    def forward(self, tensor):
        tensor = tensor + self.attention(self.ln1(tensor))
        tensor = tensor + self.mlp(self.ln2(tensor))
        return tensor

class PipelineStageModel(nn.Module):

    def __init__(self, config, pp_rank, communication):
        super().__init__()
        self.pp_rank, self.pp = (pp_rank, config.pp)
        if pp_rank == 0:
            self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
            self.pos_emb = nn.Embedding(config.seq_len, config.hidden_size)
        else:
            self.token_emb = None
            self.pos_emb = None
        self.blocks = nn.ModuleList((MegatronBlock(config, communication) for _ in range(config.layer_count(pp_rank))))
        if pp_rank == config.pp - 1:
            self.final_norm = nn.LayerNorm(config.hidden_size)
            self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.final_norm = None
            self.head = None

    def forward(self, *, input_ids=None, hidden=None):
        if self.pp_rank == 0:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            hidden = self.token_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            hidden = block(hidden)
        if self.pp_rank == self.pp - 1:
            hidden = self.head(self.final_norm(hidden))
        return hidden

def run_step(config, rank, model, optimizer, communication, device, dtype):
    """Original rank-local 1F1B computation order; native replay joins ranks."""
    from collections import deque
    dp_rank, tp_rank, pp_rank = config.coordinates(rank)
    previous = config.global_rank(dp_rank, tp_rank, pp_rank - 1) if pp_rank else None
    following = config.global_rank(dp_rank, tp_rank, pp_rank + 1) if pp_rank + 1 < config.pp else None
    batch = config.micro_batch_size
    shape = (batch, config.seq_len, config.hidden_size)
    optimizer.zero_grad(set_to_none=True)
    queue = deque()

    def activation_input():
        if previous is None:
            return None
        return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)

    def forward(activation=None, *, received=False, send_output=True):
        inputs = torch.randint(0, config.vocab_size, (batch, config.seq_len), device=device) if pp_rank == 0 else None
        if previous is not None and (not received):
            activation = activation_input()
            communication.transfer(activation, previous, send=False)
        output = model(input_ids=inputs, hidden=activation)
        loss = None
        if following is None:
            labels = torch.randint(0, config.vocab_size, (batch, config.seq_len), device=device)
            loss = F.cross_entropy(output.float().reshape(-1, config.vocab_size), labels.reshape(-1))
        elif send_output:
            communication.transfer(output.detach(), following, send=True)
        return (activation, output, loss)

    def backward(context, gradient=None, *, received=False, send_input=True):
        activation, output, loss = context
        if loss is not None:
            loss.backward()
        else:
            if not received:
                gradient = torch.zeros_like(output)
                communication.transfer(gradient, following, send=False)
            output.backward(gradient)
        if previous is not None:
            gradient = activation.grad
            if gradient is None:
                gradient = torch.zeros_like(activation)
            if send_input:
                communication.transfer(gradient.detach(), previous, send=True)
            return gradient.detach()
        return None
    warmup = min(config.pp - pp_rank - 1, config.micro_batches)
    for _ in range(warmup):
        queue.append(forward())
    prefetched = None
    for step in range(warmup, config.micro_batches):
        current = forward(prefetched, received=prefetched is not None, send_output=False)
        queue.append(current)
        oldest = queue.popleft()
        gradient = None
        if following is not None:
            gradient = torch.zeros_like(oldest[1])
            communication.exchange(((current[1].detach(), following, True), (gradient, following, False)))
        input_gradient = backward(oldest, gradient, received=following is not None, send_input=False)
        prefetched = None
        if previous is not None:
            has_next = step + 1 < config.micro_batches
            if has_next:
                prefetched = activation_input()
            transfers = [(input_gradient, previous, True)]
            if prefetched is not None:
                transfers.append((prefetched, previous, False))
            communication.exchange(transfers)
    while queue:
        backward(queue.popleft())
    if config.dp > 1:
        for parameter in model.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            communication.all_reduce(parameter.grad, 'dp')
            parameter.grad.div_(config.dp)
    optimizer.step()
