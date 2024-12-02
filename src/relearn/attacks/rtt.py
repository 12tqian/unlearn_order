import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
from tqdm import tqdm
from relearn.utils import fix_seq_len
import torch.nn.functional as F
from relearn.evaluate.mcq import evaluate


def train_step_ft(
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


def train_epoch_ft(
    model: AutoModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    mcq_records: List[Dict],
    batch_size: int = 4,
    lr: float = 3e-5,
    log_steps: int = 50,
    grad_accum_steps: int = 1,
):
    model.train()
    dataloader = DataLoader(mcq_records, batch_size=batch_size, shuffle=True)

    loss_traj = []
    for step, batch in tqdm(enumerate(dataloader)):
        loss_dict = train_step_ft(model, batch)

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        loss_traj.append(loss_dict)
        if (step + 1) % log_steps == 0:
            print(f"Epoch {epoch}, Step {step}, Loss {loss_dict['loss']}")

    return loss_traj


def train_ft(
    model: AutoModelForCausalLM,
    n_epochs: int,
    mcq_records: List[Dict],
    eval_records_dict: Dict[str, List[Dict]],
    batch_size: int = 4,
    lr: float = 3e-5,
    log_steps: int = 50,
    eval_at_start: bool = True,
    grad_accum_steps: int = 1,
):
    def run_eval(prefix: str):
        if eval:
            for eval_name, eval_records in eval_records_dict.items():
                acc = evaluate(model, eval_records, batch_size=8, normalize_loss=False)
                print(f"{prefix} {eval_name} Accuracy: {acc}")

    if eval_at_start:
        run_eval("Start")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        train_epoch_ft(
            model,
            optimizer,
            epoch,
            mcq_records,
            batch_size,
            lr=lr,
            log_steps=log_steps,
            grad_accum_steps=grad_accum_steps,
        )

        run_eval(f"Epoch {epoch}")

    return model
