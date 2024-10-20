from pathlib import Path
from datasets import Dataset
from typing import List
import torch
import json
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer
from .utils import create_prompt_letter_answer, process_batch
from functools import partial
from copy import deepcopy


def load_dataset(data_dir: Path, files: List[Path]):
    data = []
    for file in files:
        with open(data_dir / file, "r") as f:
            data.extend(f.readlines())
    data = [json.loads(d) for d in data]
    return Dataset.from_list(data)


def collate_fn(batch, shuffle_labels=False, make_prompt=False):
    if shuffle_labels:
        new_batch = deepcopy(batch)
        for point in new_batch:
            n_choices = len(point["choices"])
            point["answer"] = torch.randint(0, n_choices, (1,)).item()

        batch = new_batch

    text = [
        create_prompt_letter_answer(point) if make_prompt else point for point in batch
    ]
    return text


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    shuffle_labels: bool = False,
    make_prompt: bool = True,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(
            collate_fn, shuffle_labels=shuffle_labels, make_prompt=make_prompt
        ),
    )


def _get_dataloader(
    dataset: Dataset,
    batch_size: int,
    tokenizer: LlamaTokenizer,
    label_possibilities: List[int],
    shuffle: bool = False,
    shuffle_labels: bool = False,
    device: torch.device = "cuda",
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(
            process_batch,
            tokenizer=tokenizer,
            label_possibilities=label_possibilities,
            train_on_wrong_answer=shuffle_labels,
            device=device,
        ),
    )


# what do i need
# 1. i need the prompt
# i need the completion mask
# that's all i need

from unlearn_order.utils import create_prompt, create_prompt_letter_answer


def format_single(batch, tokenizer, max_length=512, shuffle_labels=False):
    if shuffle_labels:
        new_batch = batch.copy()
        new_batch["answer"] = torch.randint(0, len(new_batch["choices"]), (1,)).item()
        batch = new_batch
    prompt_str = create_prompt(batch)
    full_answer_str = create_prompt_letter_answer(batch)
    prompt = tokenizer(
        prompt_str,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    full_answer = tokenizer(
        full_answer_str,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    prompt_mask = torch.zeros_like(full_answer["input_ids"])
    prompt_mask[:, : prompt["input_ids"].shape[1]] = 1

    batch["input_ids"] = full_answer["input_ids"]
    batch["attention_mask"] = full_answer["attention_mask"]
    batch["prompt_mask"] = prompt_mask
    batch["prompt_str"] = prompt_str
    batch["full_str"] = full_answer_str
    labels = full_answer["input_ids"].clone()
    labels[prompt_mask.bool()] = -100

    batch["labels"] = labels

    return batch


def collate_batch(batches, tokenizer, shuffle_labels=False):
    max_length = max(
        [len(tokenizer.encode(create_prompt_letter_answer(batch))) for batch in batches]
    )

    batches = [
        format_single(
            batch, tokenizer, max_length=max_length, shuffle_labels=shuffle_labels
        )
        for batch in batches
    ]

    # stack input_ids,
    batch = {
        "input_ids": torch.stack([batch["input_ids"][0] for batch in batches]),
        "labels": torch.stack([batch["labels"][0] for batch in batches]),
        "answers": [batch["answer"] for batch in batches],
        "prompt_str": [batch["prompt_str"] for batch in batches],
        "full_str": [batch["full_str"] for batch in batches],
        "prompt_mask": torch.stack([batch["prompt_mask"][0] for batch in batches]),
    }
    return batch


def collate_eval_batch(batches, tokenizer, shuffle_labels=False):
    max_length = max(
        [len(tokenizer.encode(create_prompt_letter_answer(batch))) for batch in batches]
    )
    new_batches = []
    answers = []
    for batch in batches:
        answers.append(batch["answer"])
        for i in range(len(batch["choices"])):
            new_batch = batch.copy()
            new_batch["answer"] = i
            new_batches.append(
                format_single(
                    new_batch,
                    tokenizer,
                    max_length=max_length,
                    shuffle_labels=shuffle_labels,
                )
            )

    batches = new_batches
    # stack input_ids,
    batch = {
        "input_ids": torch.stack([batch["input_ids"][0] for batch in batches]),
        "labels": torch.stack([batch["labels"][0] for batch in batches]),
        "answers": answers,
        "prompt_str": [batch["prompt_str"] for batch in batches],
        "full_str": [batch["full_str"] for batch in batches],
        "prompt_mask": torch.stack([batch["prompt_mask"][0] for batch in batches]),
    }
    return batch


def get_finetune_dataloader(dataset, tokenizer, batch_size=8, shuffle_labels=False):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(
            collate_batch, shuffle_labels=shuffle_labels, tokenizer=tokenizer
        ),
    )
    return dataloader


def get_eval_dataloader(dataset, tokenizer, batch_size=8):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_eval_batch, tokenizer=tokenizer),
    )
    return dataloader
