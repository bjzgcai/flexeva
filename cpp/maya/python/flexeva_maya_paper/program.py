"""Ordinary frozen GPT rank program; it contains no capture or region markers."""

import gc
import torch
from flexeva_maya_paper.config import GptConfig
from _flexeva_gpt_bridge import communication
from .model import PipelineStageModel, run_step
from .optimizer import make_optimizer


def run_rank(settings, rank, device):
    config = GptConfig(**settings)
    dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float32
    torch.manual_seed(config.seed + rank)
    if device != "cpu":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(config.seed + rank)
        with torch.no_grad():
            matrix = torch.empty((8, 8), device=device, dtype=dtype)
            torch.mm(matrix, matrix)
            del matrix
        torch.cuda.empty_cache()
    communication.begin_rank(rank)
    _, _, stage = config.coordinates(rank)
    model = PipelineStageModel(config, stage, communication)
    model = model.to(device=device, dtype=dtype)
    optimizer = make_optimizer(torch, model.parameters(), config.learning_rate)
    try:
        with torch.autograd.set_multithreading_enabled(False):
            run_step(config, rank, model, optimizer, communication, device, dtype)
        if device != "cpu":
            torch.cuda.synchronize(device)
    finally:
        del optimizer, model
        gc.collect()
        communication.end_rank()
    return {"model_initializations": 1, "optimizer_initializations": 1, "step_executions": 1}
