import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
from tqdm import tqdm
from relearn.utils import fix_seq_len
import torch.nn.functional as F


def evaluate(
    model: AutoModelForCausalLM,
    records: List[Dict],
    batch_size: int = 8,
    n_choices: int = 4,
    normalize_loss: bool = False,
):
    # round up to nearest multiple of n_choices
    batch_size = (batch_size + n_choices - 1) // n_choices * n_choices
    original_fields = ["input_ids", "attention_mask", "labels"]

    aux_fields = ["answer", "completion_byte_len", "completion_mask", "length"]
    fields = original_fields + aux_fields
    dataset = [{k: v for k, v in rec.items() if k in fields} for rec in records]

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = fix_seq_len(batch, original_fields + ["completion_mask"])
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            completion_mask = batch["completion_mask"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device).bool()
            completion_byte_len = batch["completion_byte_len"].to(model.device)

            answers = batch["answer"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            labels[~completion_mask.bool()] = -100
            labels[~attention_mask.bool()] = -100

            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
                reduction="none",
            )
            loss = loss.view(shifted_labels.size(0), shifted_labels.size(1))
            loss_by_sample = loss.sum(dim=1)

            if normalize_loss:
                loss_by_sample = loss_by_sample / completion_byte_len
            loss_by_sample = loss_by_sample.view(-1, n_choices)

            answers = answers[::n_choices]

            pred = loss_by_sample.argmin(dim=1)

            correct += (pred == answers).sum().item()
            total += len(answers)

    accuracy = correct / total
    return accuracy


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
    forget_val_1_records: List[Dict],
    forget_val_2_records: List[Dict],
    batch_size: int = 4,
    lr: float = 3e-5,
    log_steps: int = 50,
    eval_at_start: bool = True,
    grad_accum_steps: int = 1,
):
    if eval_at_start:
        forget_acc_1 = evaluate(
            model, forget_val_1_records, batch_size=8, normalize_loss=False
        )
        forget_acc_2 = evaluate(
            model, forget_val_2_records, batch_size=8, normalize_loss=False
        )

        print(f"Initial Forget accuracy 1: {forget_acc_1}")
        print(f"Initial Forget accuracy 2: {forget_acc_2}")

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
        forget_acc_1 = evaluate(
            model, forget_val_1_records, batch_size=8, normalize_loss=False
        )
        forget_acc_2 = evaluate(
            model, forget_val_2_records, batch_size=8, normalize_loss=False
        )

        print(f"Epoch {epoch}, Forget accuracy 1: {forget_acc_1}")
        print(f"Epoch {epoch}, Forget accuracy 2: {forget_acc_2}")

    return model
