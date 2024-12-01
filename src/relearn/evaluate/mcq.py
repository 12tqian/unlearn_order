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
