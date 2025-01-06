from pathlib import Path
import dotenv
import sys
import os

env_file = "../.env"

if os.path.exists(env_file):
    dotenv.load_dotenv(env_file, verbose=True)
    print("Loaded environment variables from .env file.")

cwd = os.getcwd()
# for some reason appending to PATH you need it to be string
sys.path.append(str(Path(cwd).parent / "src"))


import wandb
from datasets import load_dataset
from relearn.datasets.utils import (
    load_dataset as local_load_dataset,
    DATASETS_DICT,
    Datasets,
)
from relearn.datasets.corpus import process as process_corpus
from relearn.datasets.mcq import process as process_mcq
from pathlib import Path


from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import torch
from research_tools.utils import set_seed
import os


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
from relearn.evaluate.mcq import evaluate
import wandb
from relearn.unlearn.rmu.utils import forward_with_cache


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
from relearn.evaluate.mcq import evaluate
import wandb
from relearn.unlearn.rmu.utils import forward_with_cache


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
    verbose: bool = False,
    debug: bool = False,
    tokenizer: AutoTokenizer = None,
):
    if max_batches is None:
        max_batches = int(1e9)

    for key in forget_alphas:
        assert (
            key in forget_train_records
        ), f"{key} not in forget_train_records {forget_train_records.keys()}"

    for key in retain_alphas:
        assert (
            key in retain_train_records
        ), f"{key} not in retain_train_records {retain_train_records.keys()}"

    tables = {}

    def run_eval(epoch: int, best_acc):
        if do_eval:
            log_dict = {"epoch": epoch}

            for eval_name, eval_records in eval_records_dict.items():
                acc = evaluate(model, eval_records, batch_size=8, normalize_loss=False)

                log_dict[f"{eval_name}/acc"] = acc

            if verbose:
                wandb.log(log_dict)
                if log_dict["B/acc"] >= 0.44 and log_dict["retain/acc"] >= 0.55:
                    if best_acc > log_dict["A/acc"]:
                        best_acc = log_dict["A/acc"]
                else:
                    return None

                for key in forget_batches:
                    tables[key] = log_examples(
                        model, tokenizer, tables[key], epoch, key, forget_batches[key]
                    )
                for key in retain_batches:
                    tables[key] = log_examples(
                        model, tokenizer, tables[key], epoch, key, retain_batches[key]
                    )

            return best_acc

    best_acc = 1
    if eval_at_start:
        new_acc = run_eval(-1, best_acc)
        if new_acc is None:
            return best_acc

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

    updated_module = model.model.layers[activation_layer]
    frozen_module = frozen_model.model.layers[activation_layer]

    n_batches = max_batches
    global_step = 0

    if verbose:
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

    for epoch in range(n_epochs):

        # HERE
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

                if verbose and debug:
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

                if verbose and debug:
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

            if verbose:
                log_dict.update(losses)
                log_dict["global_step"] = global_step
                log_dict["epoch"] = epoch

                wandb.log(log_dict)

            pbar.update(1)

        pbar.close()

        new_acc = run_eval(epoch, best_acc)

        if new_acc is None:
            return best_acc

        best_acc = new_acc
        # example completions

    for param in model.parameters():
        param.requires_grad = True

    return best_acc


def get_records(tokenizer):

    dataset_config = DATASETS_DICT[Datasets.WMDP]

    # retain_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    data_dir = Path("../data")

    retain_train = local_load_dataset(data_dir, dataset_config["retain_files"])
    retain_val = local_load_dataset(data_dir, dataset_config["val_retain_files"])

    unlearn_files = dataset_config["unlearn_files"]
    val_unlearn_files = dataset_config["val_unlearn_files"]

    n_val_files = 4
    max_length = 512

    forget_train_1 = local_load_dataset(data_dir, unlearn_files[:n_val_files])
    forget_train_2 = local_load_dataset(data_dir, unlearn_files[n_val_files:])

    forget_val_1 = local_load_dataset(data_dir, val_unlearn_files[:n_val_files])
    forget_val_2 = local_load_dataset(data_dir, val_unlearn_files[n_val_files:])

    forget_train_1_records = process_corpus(forget_train_1, tokenizer, max_length)
    forget_train_2_records = process_corpus(forget_train_2, tokenizer, max_length)
    retain_train_records = process_corpus(retain_train, tokenizer, max_length)
    forget_train_records = forget_train_1_records + forget_train_2_records

    forget_train_mcq_1_records = process_mcq(
        forget_val_1, tokenizer, max_length, expand_choices=False
    )
    forget_train_mcq_2_records = process_mcq(
        forget_val_2, tokenizer, max_length, expand_choices=False
    )
    forget_train_mcq_records = forget_train_mcq_1_records + forget_train_mcq_2_records

    forget_val_1_records = process_mcq(forget_val_1, tokenizer, max_length)
    forget_val_2_records = process_mcq(forget_val_2, tokenizer, max_length)
    retain_val_records = process_mcq(retain_val, tokenizer, max_length)
    forget_val_records = forget_val_1_records + forget_val_2_records
    return (
        forget_train_1_records,
        forget_train_2_records,
        retain_train_records,
        forget_train_records,
        forget_train_mcq_records,
        forget_val_1_records,
        forget_val_2_records,
        retain_val_records,
        forget_val_records,
    )


def objective():
    wandb.init()
    config = wandb.config
    set_seed(42)

    hf_access_token = os.getenv("HUGGINGFACE_API_KEY")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == torch.device("cuda")
    model_id = "HuggingFaceH4/zephyr-7b-beta"

    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=hf_access_token,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    (
        forget_train_1_records,
        forget_train_2_records,
        retain_train_records,
        forget_train_records,
        forget_train_mcq_records,
        forget_val_1_records,
        forget_val_2_records,
        retain_val_records,
        forget_val_records,
    ) = get_records(tokenizer)
    eval_dict = {
        "A": forget_val_1_records,
        "B": forget_val_2_records,
        "retain": retain_val_records,
    }

    best_acc = train_rmu(
        model,
        {"A": forget_train_1_records},
        {"B": forget_train_2_records, "retain": retain_train_records},
        eval_records_dict=eval_dict,
        n_epochs=12,
        magnitude=6.5,
        lr=1e-5,
        forget_alphas={"A": config.A_alpha},
        retain_alphas={"B": config.B_alpha, "retain": 1},
        eval_at_start=False,
        max_batches=None,
        verbose=True,
        debug=False,
        tokenizer=tokenizer,
    )
    wandb.log({"score": best_acc})
    return best_acc


def main():
    sweep_id = "62er7wi9"
    wandb.agent(sweep_id, objective, count=10, project="relearn", entity="12tqian")


if __name__ == "__main__":
    main()
