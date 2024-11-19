from pathlib import Path
from datasets import Dataset
from datasets.combine import interleave_datasets
from typing import List
import torch
import json
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer
from .utils import create_prompt_letter_answer, create_prompt
from functools import partial
from copy import deepcopy


def load_dataset(data_dir: Path, files: List[Path]):
    data = []
    for file in files:
        # check if ends with .jsonl if not append
        file = file if file.endswith(".jsonl") else file + ".jsonl"
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


def format_single(batch, tokenizer, max_length=512, shuffle_labels=False):
    """
    very sus manual padding
    """
    if shuffle_labels:
        new_batch = batch.copy()
        new_batch["answer"] = torch.randint(0, len(new_batch["choices"]), (1,)).item()
        batch = new_batch

    prompt_str = create_prompt(batch)
    full_answer_str = create_prompt_letter_answer(batch)
    completion_str = full_answer_str[len(prompt_str) :]

    prompt = tokenizer(
        prompt_str,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    completion = tokenizer(
        completion_str,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )

    input_ids = torch.cat([prompt["input_ids"], completion["input_ids"]], dim=-1)
    attention_mask = torch.cat(
        [prompt["attention_mask"], completion["attention_mask"]], dim=-1
    )
    # assert all 1s
    assert (attention_mask == 1).all()
    prompt_mask = torch.zeros_like(input_ids)
    prompt_mask[:, : prompt["input_ids"].shape[1]] = 1
    # pad to max_length at beginning
    pad_id = tokenizer.pad_token_id
    input_ids = torch.cat(
        [
            input_ids,
            torch.full(
                (input_ids.shape[0], max_length - input_ids.shape[1]),
                pad_id,
                dtype=torch.long,
            ),
        ],
        dim=-1,
    )
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.zeros(
                (attention_mask.shape[0], max_length - attention_mask.shape[1]),
                dtype=torch.long,
            ),
        ],
        dim=-1,
    )
    prompt_mask = torch.cat(
        [
            prompt_mask,
            torch.zeros(
                (prompt_mask.shape[0], max_length - prompt_mask.shape[1]),
                dtype=torch.long,
            ),
        ],
        dim=-1,
    )

    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    batch["prompt_mask"] = prompt_mask

    batch["prompt_str"] = prompt_str
    batch["full_str"] = full_answer_str
    labels = input_ids.clone()
    labels[prompt_mask.bool()] = -100

    batch["labels"] = labels
    # utf-8 byte length
    batch["byte_length"] = len(completion_str.encode("utf-8"))

    return batch


def collate_batch(batches, tokenizer, shuffle_labels=False):
    max_length = (
        max(
            [
                len(tokenizer.encode(create_prompt_letter_answer(batch)))
                for batch in batches
            ]
        )
        + 5
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
        "attention_mask": torch.stack(
            [batch["attention_mask"][0] for batch in batches]
        ),
    }
    return batch


def collate_eval_batch(batches, tokenizer, shuffle_labels=False):
    max_length = 0
    for batch in batches:
        for i in range(len(batch["choices"])):
            new_batch = batch.copy()
            new_batch["answer"] = i
            max_length = max(
                max_length,
                len(tokenizer.encode(create_prompt_letter_answer(new_batch))),
            )

    max_length += 5

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
        "attention_mask": torch.stack(
            [batch["attention_mask"][0] for batch in batches]
        ),
        "byte_length": torch.tensor([batch["byte_length"] for batch in batches]),
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
        shuffle=False,
        collate_fn=partial(collate_eval_batch, tokenizer=tokenizer),
    )
    return dataloader


def merge_datasets(datasets: List[Dataset]) -> Dataset:
    data = []
    for dataset in datasets:
        data.extend(list(dataset.data))
    return Dataset.from_list(data)
