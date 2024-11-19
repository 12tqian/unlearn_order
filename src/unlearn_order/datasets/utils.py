from enum import Enum, auto
from pathlib import Path


class Datasets(Enum):
    YEARS = auto()
    YEARS_TF = auto()
    MMLU = auto()
    WMDP_CORPUS = auto()
    WMDP_CORPUS_FINEWEB = auto()
    WMDP_CORPUS_MMLU = auto()
    WMDP_MCQ_CORPUS = auto()
    WMDP_MCQ_CORPUS_FINEWEB = auto()
    WMDP_MCQ_FINEWEB = auto()
    WMDP_MCQ_WIKITEXT = auto()
    WMDP_MCQ_LETTER_ANSWER = auto()
    BEAVERTAILS = auto()
    RANDOM_BD = auto()
    RANDOM_BD_SAME_RETAIN = auto()
    RANDOM_BD_ALL_SPLITS = auto()
    RANDOM_BD_WITH_MMLU = auto()
    RANDOM_BD_WITH_MMLU_CORPUS = auto()
    YEARS_MMLU_RETAIN = auto()
    DAY_OF_THE_MONTH = auto()
    NOT_SPECIFIED = auto()


# MMLU categories to use for forget loss
MMLU_CATS_FORGET = ["STEM", "business", "chemistry", "culture", "geography"]

MMLU_CATS_RETAIN = ["health", "history", "law", "philosophy", "social sciences"]

DATASETS_DICT = {
    Datasets.MMLU: {
        "unlearn_files": [
            f"mmlu_cats_random_trimmed/corpus_mmlu_{MMLU_CATS_FORGET[i]}"
            for i in range(5)
        ],
        "wrong_unlearn_files": [
            f"mmlu_cats_random_trimmed/whp_corpus_mmlu_{MMLU_CATS_FORGET[i]}"
            for i in range(5)
        ],
        "fixed_wrong_unlearn_files": [
            f"mmlu_cats_random_trimmed/" f"fwf_corpus_mmlu_{MMLU_CATS_FORGET[i]}"
            for i in range(5)
        ],
        "val_files": [
            f"mmlu_cats_random_trimmed/mmlu_{MMLU_CATS_FORGET[i]}" for i in range(5)
        ],
        "retain_files": [
            f"mmlu_cats_random_trimmed/corpus_mmlu_{MMLU_CATS_RETAIN[i]}"
            for i in range(5)
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{MMLU_CATS_RETAIN[i]}" for i in range(5)
        ],
        "dev_file": "mmlu_cats_random_trimmed/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.WMDP_MCQ_CORPUS: {
        "unlearn_files": [f"wmdp-deduped/corpus_split_{i}" for i in range(5)],
        "val_files": [f"wmdp-deduped/split_{i}" for i in range(5)],
        "dev_file": "wmdp-deduped/dev",
        "wrong_unlearn_files": [f"wmdp-deduped/whp_corpus_split_{i}" for i in range(5)],
        "fixed_wrong_unlearn_files": [
            f"wmdp-deduped/fwf_corpus_split_{i}" for i in range(5)
        ],
        "retain_files": [
            "wikitext/wikitext_dataset",
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{MMLU_CATS_RETAIN[i]}" for i in range(5)
        ],
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.WMDP_MCQ_WIKITEXT: {
        "unlearn_files": [f"wmdp-deduped/mcq_split_{i}" for i in range(5)],
        "val_files": [f"wmdp-deduped/split_{i}" for i in range(5)],
        "dev_file": "wmdp-deduped/dev",
        "wrong_unlearn_files": [f"wmdp-deduped/whp_corpus_split_{i}" for i in range(5)],
        "fixed_wrong_unlearn_files": [
            f"wmdp-deduped/fwf_corpus_split_{i}" for i in range(5)
        ],
        "retain_files": [
            "wikitext/wikitext_dataset",
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{MMLU_CATS_RETAIN[i]}" for i in range(5)
        ],
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.WMDP_MCQ_LETTER_ANSWER: {
        "unlearn_files": [f"wmdp-deduped/mcq_split_{i}" for i in range(5)],
        "val_files": [f"wmdp-deduped/split_{i}" for i in range(5)],
        "dev_file": "wmdp-deduped/dev",
        "wrong_unlearn_files": [f"wmdp-deduped/whp_corpus_split_{i}" for i in range(5)],
        "fixed_wrong_unlearn_files": [
            f"wmdp-deduped/fwf_corpus_split_{i}" for i in range(5)
        ],
        "retain_files": [f"fineweb_edu_seed-42/split_{i}" for i in range(5)],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{MMLU_CATS_RETAIN[i]}" for i in range(5)
        ],
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.BEAVERTAILS: {
        "unlearn_files": [
            "beavertails/criminal_activities_dataset",
            "beavertails/social_issues_dataset",
        ],
        "val_files": [
            "beavertails/criminal_activities_dataset",
            "beavertails/social_issues_dataset",
        ],
        "dev_file": "",
        "retain_files": [""],
        "val_retain_files": [""],
        "retain_dev_file": "",
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
    Datasets.RANDOM_BD_SAME_RETAIN: {
        "unlearn_files": [f"random_bd/corpus_split_{i}" for i in range(5)],
        "val_files": [f"random_bd/split_{i}" for i in range(5)],
        "retain_files": [f"random_bd/corpus_split_{i}" for i in range(5, 10)],
        "val_retain_files": [f"random_bd/split_{i}" for i in range(5, 10)],
        "dev_file": "dates-years-trimmed/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.RANDOM_BD_ALL_SPLITS: {
        "unlearn_files": [],
        "val_files": [f"random_bd/split_{i}" for i in range(10)],
        "retain_files": [],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{MMLU_CATS_RETAIN[i]}" for i in range(5)
        ],
        "dev_file": "dates-years-trimmed/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.RANDOM_BD_WITH_MMLU: {
        "unlearn_files": [],
        "val_files": [
            *[f"random_bd/split_{i}" for i in range(10)],
            *[f"mmlu_cats_random_trimmed/mmlu_{MMLU_CATS_RETAIN[i]}" for i in range(5)],
        ],
        "retain_files": [],
        "val_retain_files": [
            *[f"random_bd/split_{i}" for i in range(10)],
        ],
        "dev_file": "dates-years-trimmed/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.RANDOM_BD_WITH_MMLU_CORPUS: {
        "unlearn_files": [],
        "wrong_unlearn_files": [f"random_bd/corpus_split_{i}" for i in range(5)],
        "val_files": [*[f"random_bd/split_{i}" for i in range(5)]],
        "retain_files": [
            f"mmlu_cats_random_trimmed/corpus_mmlu_{MMLU_CATS_RETAIN[i]}"
            for i in range(5)
        ],
        "val_retain_files": [
            f"mmlu_cats_random_trimmed/mmlu_{MMLU_CATS_RETAIN[i]}" for i in range(5)
        ],
        "dev_file": "dates-years-trimmed/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
    Datasets.DAY_OF_THE_MONTH: {
        "unlearn_files": [],
        "val_files": [f"day_of_the_month/split_{i}" for i in range(1, 5)],
        "retain_files": [],
        "val_retain_files": [f"day_of_the_month/split_{i}" for i in range(1)],
        "dev_file": "day_of_the_month/dev",
        "retain_dev_file": "mmlu_cats_random_trimmed/dev",
    },
}
