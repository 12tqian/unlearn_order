import sys

sys.path.append("./src")

import os
import dotenv
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import get_peft_model, LoraConfig
from unlearn_order.dataset import load_dataset
from unlearn_order.pipeline import run_pipeline
import fire
from dataclasses import dataclass
from omegaconf import OmegaConf
from typing import List, Tuple
from enum import Enum
from unlearn_order.common import TaskType, DatasetType, Task, ExpConfig
import torch.multiprocessing as mp
from queue import Empty
from research_tools.config import setup
from research_tools.utils import set_seed
import time


def get_default_cfg():
    return ExpConfig()


def get_cfg(cfg_path: Path) -> ExpConfig:
    if cfg_path is None:
        return get_default_cfg()

    if not cfg_path.exists():
        print(f"Config file {cfg_path} not found.")
        raise FileNotFoundError(f"Config file {cfg_path} not found.")

    schema = OmegaConf.structured(ExpConfig)
    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(schema, cfg)

    return cfg


def run_experiment(cfg: ExpConfig):
    cfg = setup(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, token=cfg.hf_access_token, torch_dtype=cfg.model_dtype
    )
    model = model.to(device)

    tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, token=cfg.hf_access_token
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "v_proj"],
        )

        model = get_peft_model(model, lora_config)

    results = run_pipeline(model, tokenizer, cfg)


def gen_cfgs() -> List[ExpConfig]:
    cfg_list = []

    def get_tasks(task_type: TaskType, test_idx: int):
        assert test_idx in list(range(4)), "test_idx must be in [0, 1, 2, 3]"
        if test_idx == 0:
            return [
                Task(task_type=task_type, dataset_type=DatasetType.TRAIN),
                Task(task_type=task_type, dataset_type=DatasetType.VAL),
            ]
        elif test_idx == 1:
            return [
                Task(task_type=task_type, dataset_type=DatasetType.VAL),
                Task(task_type=task_type, dataset_type=DatasetType.TRAIN),
            ]
        elif test_idx == 2:
            return [Task(task_type=task_type, dataset_type=DatasetType.COMBINED)]
        elif test_idx == 3:
            return [
                Task(task_type=task_type, dataset_type=DatasetType.TRAIN),
                Task(task_type=task_type, dataset_type=DatasetType.VAL),
                Task(task_type=task_type, dataset_type=DatasetType.COMBINED),
            ]

    for finetune_idx in range(3):
        for unlearn_idx in range(3):
            task_list = []
            task_list.extend(get_tasks(TaskType.FINETUNE, finetune_idx))
            task_list.extend(get_tasks(TaskType.EVAL, 0))
            task_list.extend(get_tasks(TaskType.UNLEARN, unlearn_idx))
            task_list.extend(get_tasks(TaskType.EVAL, 0))
            task_list.append(
                Task(task_type=TaskType.FINETUNE, dataset_type=DatasetType.TRAIN)
            )
            task_list.extend(get_tasks(TaskType.EVAL, 0))
            cfg = ExpConfig(
                task_order=task_list,
                exp_name=f"finetune_{finetune_idx}_unlearn_{unlearn_idx}",
            )

            cfg_list.append(cfg)

    return cfg_list


def main(cfg_path: str = None):
    cfg_path = Path(cfg_path)
    cfg = get_cfg(cfg_path)
    set_seed(cfg.seed)

    run_experiment(cfg)


if __name__ == "__main__":
    fire.Fire(main)
