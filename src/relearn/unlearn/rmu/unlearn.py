import os
import datetime

import numpy as np
import torch
from transformers import AdamW
from tqdm import tqdm


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from torch.optim import AdamW
from typing import List, Dict
from copy import deepcopy
from torch.utils.data import DataLoader
from relearn.evaluate.mcq import evaluate

from .utils import forward_with_cache


def get_params(model: AutoModelForCausalLM, layers: List[int], module_names: List[str]):
    params = []
    for layer in layers:
        for name, param in model.model.layers[layer].named_parameters():
            if any([module_name in name for module_name in module_names]):
                params.append(param)
    return params


def _train_rmu(
    model: AutoModelForCausalLM,
    forget_train_records: List[Dict],
    retain_train_records: List[Dict],
    eval_records_dict: Dict[str, List[Dict]],
    magnitude: float = 20,
    forget_alpha: float = 100,
    lr: float = 5e-5,
    batch_size: int = 4,
    max_batches: int = 150,
    activation_layer: int = 7,
    train_layers: List[int] = [5, 6, 7],
    param_names: List[str] = ["down_proj"],
    n_epochs: int = 1,
    eval_at_start: bool = True,
):
    def run_eval(prefix: str):
        if eval:
            for eval_name, eval_records in eval_records_dict.items():
                acc = evaluate(model, eval_records, batch_size=8, normalize_loss=False)
                print(f"{prefix} {eval_name} Accuracy: {acc}")

    if eval_at_start:
        run_eval("Start")
    frozen_model = deepcopy(model)
    frozen_model.eval()

    for param in frozen_model.parameters():
        param.requires_grad = False

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

    updated_module = model.model.layers[activation_layer]
    frozen_module = frozen_model.model.layers[activation_layer]

    n_batches = min(max_batches, len(forget_train_records), len(retain_train_records))
    n_done = 0

    for epoch in range(n_epochs):
        if n_done >= n_batches:
            break

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

            forget_inputs = {
                "input_ids": forget_batch["input_ids"].to(model.device),
                "labels": forget_batch["labels"].to(model.device),
                "attention_mask": forget_batch["attention_mask"].to(model.device),
            }
            forget_activations = forward_with_cache(
                model, forget_inputs, module=updated_module, no_grad=False
            ).to(model.device)

            retain_inputs = {
                "input_ids": retain_batch["input_ids"].to(model.device),
                "labels": retain_batch["labels"].to(model.device),
                "attention_mask": retain_batch["attention_mask"].to(model.device),
            }
            retain_activations = forward_with_cache(
                model, retain_inputs, module=updated_module, no_grad=False
            ).to(model.device)
            frozen_retain_activations = forward_with_cache(
                frozen_model, retain_inputs, module=frozen_module, no_grad=True
            ).to(model.device)

            unlearn_loss = (
                torch.nn.functional.mse_loss(forget_activations, control_vec)
                * forget_alpha
            )
            retain_loss = torch.nn.functional.mse_loss(
                retain_activations, frozen_retain_activations
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

        run_eval(f"Epoch {epoch}")

    for param in model.parameters():
        param.requires_grad = True

    return model


def train_rmu(
    model: AutoModelForCausalLM,
    forget_train_records: Dict[str, List[Dict]],
    retain_train_records: Dict[str, List[Dict]],
    eval_records_dict: Dict[str, List[Dict]],
    magnitude: float = 20,
    forget_alpha: float = 100,
    lr: float = 5e-5,
    batch_size: int = 4,
    max_batches: int = 150,
    activation_layer: int = 7,
    train_layers: List[int] = [5, 6, 7],
    param_names: List[str] = ["down_proj"],
    n_epochs: int = 1,
    eval_at_start: bool = True,
):
    def run_eval(prefix: str):
        if eval:
            for eval_name, eval_records in eval_records_dict.items():
                acc = evaluate(model, eval_records, batch_size=8, normalize_loss=False)
                print(f"{prefix} {eval_name} Accuracy: {acc}")

    if eval_at_start:
        run_eval("Start")
    frozen_model = deepcopy(model)
    frozen_model.eval()

    for param in frozen_model.parameters():
        param.requires_grad = False

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

    updated_module = model.model.layers[activation_layer]
    frozen_module = frozen_model.model.layers[activation_layer]

    n_batches = max_batches
    n_done = 0

    for epoch in range(n_epochs):
        if n_done >= n_batches:
            break

        forget_dataloaders = {
            key: iter(
                DataLoader(
                    forget_train_records[key], batch_size=batch_size, shuffle=True
                )
            )
            for key in forget_train_records
        }
        retain_dataloaders = {
            key: iter(
                DataLoader(
                    retain_train_records[key], batch_size=batch_size, shuffle=True
                )
            )
            for key in retain_train_records
        }

        min_length = min(len(forget_dataloaders[key]) for key in forget_dataloaders)
        min_length = min(
            min_length, min(len(retain_dataloaders[key]) for key in retain_dataloaders)
        )
        # keep iterating till next epoch

        pbar = tqdm(range(min(min_length, n_batches)))

        while True:
            if n_done >= n_batches:
                break

            try:
                forget_batches = {
                    key: next(forget_dataloaders[key]) for key in forget_dataloaders
                }
                retain_batches = {
                    key: next(retain_dataloaders[key]) for key in retain_dataloaders
                }
            except StopIteration:
                break

            losses = {}

            for key in forget_batches:
                forget_batch = forget_batches[key]

                forget_inputs = {
                    "input_ids": forget_batch["input_ids"].to(model.device),
                    "labels": forget_batch["labels"].to(model.device),
                    "attention_mask": forget_batch["attention_mask"].to(model.device),
                }
                forget_activations = forward_with_cache(
                    model, forget_inputs, module=updated_module, no_grad=False
                ).to(model.device)

                unlearn_loss = (
                    torch.nn.functional.mse_loss(forget_activations, control_vec)
                    * forget_alpha
                )
                unlearn_loss.backward()

                losses[f"forget_{key}"] = unlearn_loss.detach().item()

            for key in retain_batches:
                retain_batch = retain_batches[key]

                retain_inputs = {
                    "input_ids": retain_batch["input_ids"].to(model.device),
                    "labels": retain_batch["labels"].to(model.device),
                    "attention_mask": retain_batch["attention_mask"].to(model.device),
                }

                retain_activations = forward_with_cache(
                    model, retain_inputs, module=updated_module, no_grad=False
                ).to(model.device)

                frozen_retain_activations = forward_with_cache(
                    frozen_model, retain_inputs, module=frozen_module, no_grad=True
                ).to(model.device)

                retain_loss = torch.nn.functional.mse_loss(
                    retain_activations, frozen_retain_activations
                )

                retain_loss.backward()

                losses[f"retain_{key}"] = retain_loss.detach().item()

            optimizer.step()
            optimizer.zero_grad()

            n_done += 1

            pbar.set_postfix(losses)
            pbar.update(1)

        run_eval(f"Epoch {epoch}")

    for param in model.parameters():
        param.requires_grad = True

    return model
