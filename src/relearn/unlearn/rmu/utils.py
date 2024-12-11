import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
from typing import Dict


def forward_with_cache(
    model: AutoModelForCausalLM,
    inputs: Dict,
    module: nn.Module,
    no_grad: bool = True,
):
    # define a tensor with the size of our cached activations
    cache = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None

    hook_handle = module.register_forward_hook(hook)

    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)

    hook_handle.remove()

    return cache[0]
