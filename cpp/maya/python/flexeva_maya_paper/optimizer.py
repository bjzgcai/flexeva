def make_optimizer(torch, parameters, learning_rate):
    return torch.optim.AdamW(parameters, lr=learning_rate, foreach=True, fused=False)
