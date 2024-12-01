from datasets import Dataset
from transformers import AutoTokenizer
from functools import partial
from typing import Dict, List
import torch


def map_corpus(data: Dict, tokenizer: AutoTokenizer, max_length: int):
    text = data["text"]
    output = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": output["input_ids"].squeeze(),
        "attention_mask": output["attention_mask"].squeeze(),
        "labels": output["input_ids"].clone().squeeze(),
    }


def expand_corpus_records(records: List[Dict]):
    for rec in records:
        rec["input_ids"] = torch.tensor(rec["input_ids"])
        rec["labels"] = torch.tensor(rec["labels"])
        rec["attention_mask"] = torch.tensor(rec["attention_mask"])
        rec["length"] = rec["input_ids"].size(0)
    return records


def process(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    min_length: int = 0,
):
    dataset = dataset.filter(lambda x: len(x["text"]) > min_length)
    dataset = dataset.map(
        partial(map_corpus, tokenizer=tokenizer, max_length=max_length)
    )

    keep_cols = ["input_ids", "attention_mask", "labels"]
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in keep_cols]
    )
    dataset.set_format("torch")

    dataset = dataset.to_list()
    records = expand_corpus_records(dataset)

    return records
