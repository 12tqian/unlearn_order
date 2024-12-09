import os
import datetime

import numpy as np
import torch
from transformers import AdamW
import tqdm as tqdm


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from torch.optim import AdamW
from typing import List, Dict
from copy import deepcopy
from torch.utils.data import DataLoader


from .utils import load_model, get_params, forward_with_cache, get_data


def get_params(model: AutoModelForCausalLM, layers: List[int], module_names: List[str]):
    params = []
    for layer in layers:
        for name, param in model.model.layers[layer].named_parameters():
            if any([module_name in name for module_name in module_names]):
                params.append(param)
    return params


def train_rmu(
    model: AutoModelForCausalLM,
    frozen_model: AutoModelForCausalLM,
    forget_train_records: List[Dict],
    retain_train_records: List[Dict],
    magnitude: float = 20,
    retain_alpha: float = 100,
    lr: float = 5e-5,
    batch_size: int = 4,
    max_batches: int = 150,
    activation_layer: int = 7,
    train_layers: List[int] = [5, 6, 7],
    param_names: List[str] = ["down_proj"],
    n_epochs: int = 1,
):

    model.train()

    params = get_params(model, train_layers, param_names)

    for param in model.parameters():
        param.requires_grad = False

    for param in params:
        param.requires_grad = True

    optimizer = AdamW(params, lr=lr)

    rng_vec = torch.rand(
        1, 1, model.config.hidden_size, dtype=model.dtype, device=model.device
    )
    control_vec = rng_vec / torch.norm(rng_vec) * magnitude

    n_batches = min(max_batches, len(forget_train_records), len(retain_train_records))

    n_done = 0

    for _ in range(n_epochs):
        forget_dataloader = DataLoader(
            forget_train_records, batch_size=batch_size, shuffle=True
        )
        retain_dataloader = DataLoader(
            retain_train_records, batch_size=batch_size, shuffle=True
        )

        for forget_batch, retain_batch in (
            pbar := tqdm(zip(forget_dataloader, retain_dataloader))
        ):
            if n_done >= n_batches:
                break

            model.zero_grad()

            forget_input_ids = forget_batch["input_ids"].to(model.device)
            forget_activations = forward_with_cache(
                model, forget_input_ids, layer=activation_layer, no_grad=False
            ).to(model.device)

            retain_input_ids = retain_batch["input_ids"].to(model.device)
            retain_activations = forward_with_cache(
                model, retain_input_ids, layer=activation_layer, no_grad=False
            ).to(model.device)
            frozen_retain_activations = forward_with_cache(
                frozen_model, retain_input_ids, layer=activation_layer, no_grad=True
            ).to(model.device)

            unlearn_loss = torch.nn.functional.mse_loss(forget_activations, control_vec)
            retain_loss = (
                torch.nn.functional.mse_loss(
                    retain_activations, frozen_retain_activations
                )
                * retain_alpha
            )

            loss = unlearn_loss + retain_loss

            loss.backward()
            optimizer.step()

            n_done += 1

            pbar.set_postfix(
                {
                    "Loss": loss.detach().item(),
                    "Unlearn Loss": unlearn_loss.detach().item(),
                    "Retain Loss": retain_loss.detach().item(),
                }
            )

    return model
