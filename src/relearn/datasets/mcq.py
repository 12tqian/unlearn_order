import torch
from typing import Dict, List, Callable
from copy import deepcopy
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer
from relearn.datasets.format import create_prompt, create_prompt_letter_answer


def create_mcq(
    record: Dict,
    tokenizer: AutoTokenizer,
    max_length: int,
    context: str = "",
    completion_func: Callable = create_prompt_letter_answer,
):
    record["question"] = context + record["question"]
    text = completion_func(record)
    prompt = create_prompt(record)

    completion = text[len(prompt) :]

    record["prompt"] = prompt
    record["text"] = text
    record["completion"] = completion

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    completion_ids = tokenizer(
        completion, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]

    input_ids = torch.cat([prompt_ids, completion_ids])
    labels = input_ids.clone()

    attention_mask = torch.ones_like(input_ids)
    prompt_mask = torch.zeros_like(input_ids)
    prompt_mask[: len(prompt_ids)] = 1

    completion_mask = torch.zeros_like(input_ids)
    completion_mask[len(prompt_ids) :] = 1

    # pad to max length on left
    seq_len = input_ids.size(0)
    pad_len = max_length - seq_len
    padding = (0, pad_len)

    input_ids = F.pad(input_ids, padding, value=0)
    labels = F.pad(labels, padding, value=-100)
    attention_mask = F.pad(attention_mask, padding, value=0)
    prompt_mask = F.pad(prompt_mask, padding, value=0)
    completion_mask = F.pad(completion_mask, padding, value=0)

    record["input_ids"] = input_ids
    record["labels"] = labels
    record["attention_mask"] = attention_mask
    record["completion_byte_len"] = len(completion.encode("utf-8"))
    record["prompt_mask"] = prompt_mask
    record["completion_mask"] = completion_mask
    record["length"] = seq_len

    return record


def expand_mcq_records(records: List[Dict], expand_choices: bool = True, **kwargs):
    new_records = []
    for rec in records:
        n_choices = len(rec["choices"])

        if not expand_choices:
            new_rec = deepcopy(rec)
            new_rec = create_mcq(new_rec, **kwargs)
            new_records.append(new_rec)
            continue

        for i in range(n_choices):
            new_rec = deepcopy(rec)
            actual_answer = rec["answer"]
            new_rec["answer"] = i
            new_rec = create_mcq(new_rec, **kwargs)
            new_rec["answer"] = actual_answer
            new_rec["selected_answer"] = i
            new_records.append(new_rec)

    return new_records


def process(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    expand_choices: bool = True,
):
    records = dataset.to_list()
    records = expand_mcq_records(
        records,
        tokenizer=tokenizer,
        max_length=max_length,
        expand_choices=expand_choices,
    )
    return records
