import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
from tqdm import tqdm
from relearn.utils import fix_seq_len
import torch.nn.functional as F
from relearn.evaluate import run_eval
import wandb


def train_step_rtt(
    model: AutoModelForCausalLM,
    batch: Dict,
):
    batch = fix_seq_len(
        batch, ["input_ids", "attention_mask", "labels", "completion_mask"]
    )

    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    labels = batch["labels"].to(model.device)
    completion_mask = batch["completion_mask"].to(model.device)

    # only train on completions for multiple choice
    labels[~completion_mask.bool()] = -100

    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = output.loss

    loss.backward()

    loss = loss.detach().item()

    loss_dict = {
        "loss": loss,
    }

    return loss_dict


def train_epoch_rtt(
    model: AutoModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    mcq_records: List[Dict],
    batch_size: int = 4,
    log_steps: int = 50,
    grad_accum_steps: int = 1,
    use_wandb: bool = True,
):
    model.train()
    dataloader = DataLoader(mcq_records, batch_size=batch_size, shuffle=True)

    loss_traj = []
    for step, batch in tqdm(enumerate(dataloader)):
        loss_dict = train_step_rtt(model, batch)

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        loss_traj.append(loss_dict)
        if (step + 1) % log_steps == 0:
            if use_wandb:
                loss_dict["global_step"] = epoch * len(dataloader) + step
                wandb.log(loss_dict)
            else:
                print(f"Epoch {epoch}, Step {step}, Loss {loss_dict['loss']}")

    return loss_traj


def train_rtt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    n_epochs: int,
    mcq_records: List[Dict],
    eval_records_dict: Dict[str, List[Dict]],
    batch_size: int = 4,
    lr: float = 3e-5,
    log_steps: int = 50,
    eval_at_start: bool = True,
    grad_accum_steps: int = 1,
    use_wandb: bool = True,
):
    if eval_at_start:
        res = run_eval(model, tokenizer, eval_records_dict, -1)
        if use_wandb:
            wandb.log(res)
        else:
            print(res)

    if use_wandb:
        wandb.define_metric("global_step")

        wandb.define_metric("loss", step_metric="global_step")
        wandb.define_metric("epoch")

        for eval_name in eval_records_dict:
            wandb.define_metric(f"{eval_name}/acc", step_metric="epoch")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        train_epoch_rtt(
            model,
            optimizer,
            epoch,
            mcq_records,
            batch_size,
            log_steps=log_steps,
            grad_accum_steps=grad_accum_steps,
        )

        res = run_eval(model, tokenizer, eval_records_dict, epoch)
        if use_wandb:
            wandb.log(res)
        else:
            print(res)

    return model
