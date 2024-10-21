from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from research_tools.config import BaseConfig


class TaskType(Enum):
    FINETUNE = "finetune"
    UNLEARN = "unlearn"
    EVAL = "eval"


class DatasetType(Enum):
    TRAIN = "train"
    VAL = "val"
    COMBINED = "combined"


@dataclass
class Task:
    task_type: TaskType
    dataset_type: DatasetType


@dataclass
class ExpConfig(BaseConfig):
    hf_access_token: Optional[str] = None
    exp_name: str = "test"
    seed = 42
    results_dir: str = "./results/random_bd"
    data_dir: str = "./data/random_bd"
    env_dir: str = ".env"
    python_path: str = "/mnt/align1_drive/tcqian/unlearning_order/venv/bin/python"
    script_path: str = "scripts/main.py"

    model_name: str = "meta-llama/Meta-Llama-3-8B"
    batch_size: int = 4
    lr: float = 3e-6

    use_lora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 8

    tolerance: float = 0.01
    max_epochs: int = 100

    splits: List[int] = field(default_factory=lambda: list(range(10)))

    n_train: int = 1
    n_val: int = 1

    eval_every: int = 5

    model_dtype: str = "bfloat16"
    task_order: List[Task] = field(default_factory=list)
