from pathlib import Path
from typing import List
import json
from datasets import Dataset

from enum import Enum, auto
from pathlib import Path


class Datasets(Enum):
    YEARS = auto()
    WMDP = auto()
    RANDOM_BD = auto()


# MMLU categories to use for forget loss
MMLU_CATS_FORGET = ["STEM", "business", "chemistry", "culture", "geography"]
MMLU_CATS_RETAIN = ["health", "history", "law", "philosophy", "social sciences"]


DATASETS_DICT = {
    Datasets.WMDP: {
        "unlearn_files": [f"wmdp-deduped/corpus_split_{i}" for i in range(5)],
        "val_unlearn_files": [f"wmdp-deduped/split_{i}" for i in range(5)],
        "retain_files": [
            "fineweb-edu/corpus_split_0",
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{MMLU_CATS_RETAIN[i]}" for i in range(5)
        ],
    },
    Datasets.RANDOM_BD: {
        "unlearn_files": [f"random_bd/corpus_split_{i}" for i in range(5)],
        "wrong_unlearn_files": [f"random_bd/whp_corpus_split_{i}" for i in range(5)],
        "fixed_wrong_unlearn_files": [
            f"random_bd/fwf_corpus_split_{i}" for i in range(5)
        ],
        "val_files": [f"random_bd/split_{i}" for i in range(5)],
        "retain_files": [f"fineweb_edu_seed-42/split_{i}" for i in range(5)],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{MMLU_CATS_RETAIN[i]}" for i in range(5)
        ],
        "dev_file": "dates-years-trimmed/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
}


def load_dataset(data_dir: Path, files: List[str]):
    data = []
    for file in files:
        # check if ends with .jsonl if not append
        file = file if file.endswith(".jsonl") else file + ".jsonl"
        with open(data_dir / file, "r") as f:
            data.extend(f.readlines())
    data = [json.loads(d) for d in data]
    return Dataset.from_list(data)
