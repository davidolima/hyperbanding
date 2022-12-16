import functools
import torch.optim.lr_scheduler as lr_scheduler

def linear_warmup(
    optimizer,
    warmup_steps: int,
    start_factor: float=0.0,
    end_factor: float=1.0
):
    def _multiplier(
        step: int,
        warmup_steps: int,
        start_factor: float=0.0,
        end_factor: float=1.0
    ):
        if step < warmup_steps:
            return ((end_factor - start_factor) / warmup_steps) * step + start_factor
        return end_factor

    lambda_linear_warmup = functools.partial(
        _multiplier,
        warmup_steps=warmup_steps,
        start_factor=start_factor,
        end_factor=end_factor
    )

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_linear_warmup)
