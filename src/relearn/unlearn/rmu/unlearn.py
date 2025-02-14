import os
import datetime

import numpy as np
import torch
from transformers import AdamW
from tqdm import tqdm
from torch import nn


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from torch.optim import AdamW
from typing import List, Dict
from copy import deepcopy
from torch.utils.data import DataLoader
from relearn.evaluate import run_eval
import wandb
from .utils import forward_with_cache


def get_params(model: AutoModelForCausalLM, layers: List[int], module_names: List[str]):
    params = []
    for layer in layers:
        for name, param in model.model.layers[layer].named_parameters():
            if any([module_name in name for module_name in module_names]):
                params.append(param)
    return params


def log_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    table: wandb.Table,
    epoch: int,
    key: str,
    batch: Dict,
    max_extra_length: int = 30,
):
    inputs = {
        "input_ids": batch["input_ids"].to(model.device),
        "labels": batch["labels"].to(model.device),
        "attention_mask": batch["attention_mask"].to(model.device),
    }
    original_pad_token_id = tokenizer.pad_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    outputs = model.generate(
        **inputs,
        max_length=inputs["input_ids"].shape[-1] + max_extra_length,
        do_sample=False,
    )
    input_texts = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    completions = []
    for input_text, output_text in zip(input_texts, output_texts):
        completions.append(output_text[len(input_text) :])
    prompts = input_texts
    for prompt, completion in zip(prompts, completions):
        table.add_data(epoch, prompt, completion)
    tokenizer.pad_token_id = original_pad_token_id

    new_table = wandb.Table(columns=table.columns, data=table.data)
    wandb.log({f"{key}/examples": new_table}, commit=False)

    return table


def train_rmu(
    model: AutoModelForCausalLM,
    forget_train_records: Dict[str, List[Dict]],
    retain_train_records: Dict[str, List[Dict]],
    eval_records_dict: Dict[str, List[Dict]],
    magnitude: float = 20,
    forget_alphas: Dict[str, float] = {},
    retain_alphas: Dict[str, float] = {},
    lr: float = 5e-5,
    batch_size: int = 4,
    max_batches: int = 150,
    activation_layer: int = 7,
    train_layers: List[int] = [5, 6, 7],
    param_names: List[str] = ["down_proj"],
    n_epochs: int = 1,
    eval_at_start: bool = True,
    do_eval: bool = True,
    use_wandb: bool = False,
    debug: bool = False,
    tokenizer: AutoTokenizer = None,
    monitor_name: str = None,
    monitor_threshold: float = 0.28,
    base_epoch: int = 0,
    control_vecs_init: Dict[str, torch.Tensor] = None,
    return_control_vecs: bool = False,
    print_evals: bool  = False
):
    if max_batches is None:
        max_batches = int(1e9)

    for key in forget_train_records:
        assert (
            key in forget_alphas
        ), f"{key} not in forget_train_records {forget_train_records.keys()}"
    for key in retain_train_records:
        assert (
            key in retain_alphas
        ), f"{key} not in retain_train_records {retain_train_records.keys()}"

    tables = {}

    if eval_at_start:
        res = run_eval(model, tokenizer, eval_records_dict, -1)
        if use_wandb:
            wandb.log(res)
        if print_evals:
            print(res)

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

    control_vecs = {}
    for key in forget_train_records:
        rng_vec = torch.rand(
            1, 1, model.config.hidden_size, dtype=model.dtype, device=model.device
        )
        control_vecs[key] = rng_vec / torch.norm(rng_vec) * magnitude
    
    if control_vecs_init is not None:
        control_vecs.update(control_vecs_init)

    updated_module = model.model.layers[activation_layer]
    frozen_module = frozen_model.model.layers[activation_layer]

    n_batches = max_batches
    global_step = 0

    if use_wandb:
        wandb.define_metric("global_step")

        for key in forget_train_records:
            wandb.define_metric(f"{key}/forget_loss", step_metric="global_step")
            wandb.define_metric(
                f"{key}/frozen_forget_activations.norm", step_metric="global_step"
            )
            wandb.define_metric(
                f"{key}/forget_activations.norm", step_metric="global_step"
            )
            wandb.define_metric(f"{key}/unlearn_cosine", step_metric="global_step")
            tables[key] = wandb.Table(columns=["epoch", "prompt", "completion"])

        for key in retain_train_records:
            wandb.define_metric(f"{key}/retain_loss", step_metric="global_step")
            wandb.define_metric(
                f"{key}/retain_activations.norm", step_metric="global_step"
            )
            wandb.define_metric(
                f"{key}/frozen_retain_activations.norm", step_metric="global_step"
            )
            wandb.define_metric(f"{key}/retain_cosine", step_metric="global_step")
            tables[key] = wandb.Table(columns=["epoch", "prompt", "completion"])

        wandb.define_metric("epoch")

        for eval_name in eval_records_dict:
            wandb.define_metric(f"{eval_name}/acc", step_metric="epoch")

    for epoch in range(base_epoch, n_epochs + base_epoch):

        if global_step >= n_batches:
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
            if global_step >= n_batches:
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

            log_dict = {}
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

                control_vec = control_vecs[key]

                forget_loss = torch.nn.functional.mse_loss(
                    forget_activations, control_vec
                )

                if key in forget_alphas:
                    forget_loss = forget_loss * forget_alphas[key]

                forget_loss.backward()

                losses[f"{key}/forget_loss"] = forget_loss.detach().item()

                if use_wandb and debug:
                    frozen_forget_activations = forward_with_cache(
                        frozen_model, forget_inputs, module=frozen_module, no_grad=True
                    ).to(model.device)
                    unlearn_cosine = torch.nn.functional.cosine_similarity(
                        forget_activations, frozen_forget_activations, dim=-1
                    ).mean()

                    log_dict[f"{key}/frozen_forget_activations.norm"] = torch.mean(
                        frozen_forget_activations.norm(dim=-1).mean(dim=1), dim=0
                    ).item()
                    log_dict[f"{key}/forget_activations.norm"] = torch.mean(
                        forget_activations.norm(dim=-1).mean(dim=1), dim=0
                    ).item()
                    log_dict[f"{key}/unlearn_cosine"] = unlearn_cosine.item()

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

                if key in retain_alphas:
                    retain_loss = retain_loss * retain_alphas[key]

                retain_loss.backward()

                losses[f"{key}/retain_loss"] = retain_loss.detach().item()

                if use_wandb and debug:
                    frozen_retain_activations = forward_with_cache(
                        frozen_model, retain_inputs, module=frozen_module, no_grad=True
                    ).to(model.device)
                    retain_cosine = torch.nn.functional.cosine_similarity(
                        retain_activations, frozen_retain_activations, dim=-1
                    ).mean()

                    log_dict[f"{key}/retain_activations.norm"] = torch.mean(
                        retain_activations.norm(dim=-1).mean(dim=1), dim=0
                    ).item()
                    log_dict[f"{key}/frozen_retain_activations.norm"] = torch.mean(
                        frozen_retain_activations.norm(dim=-1).mean(dim=1), dim=0
                    ).item()
                    log_dict[f"{key}/retain_cosine"] = retain_cosine.item()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            pbar.set_postfix(losses)

            if use_wandb:
                log_dict.update(losses)
                log_dict["global_step"] = global_step
                log_dict["epoch"] = epoch

                wandb.log(log_dict)

            pbar.update(1)

        pbar.close()

        res = run_eval(model, tokenizer, eval_records_dict, epoch)
        if use_wandb:
            wandb.log(res)
        if print_evals:
            print(res)

        if monitor_name is not None:
            if res[monitor_name] < monitor_threshold:
                break

    for param in model.parameters():
        param.requires_grad = True

    if return_control_vecs:
        return model, control_vecs
    
    return model
