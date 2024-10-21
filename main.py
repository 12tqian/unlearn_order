import sys

sys.path.append("./src")

import os
from pathlib import Path
from research_tools import get_gpus_available
import fire
from omegaconf import OmegaConf
from typing import List, Dict
from unlearn_order.common import TaskType, DatasetType, Task, ExpConfig
from queue import Empty
import time
import multiprocessing as mp
import logging
from research_tools.launcher import launch


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


def main():
    launch(gen_cfgs=gen_cfgs)


if __name__ == "__main__":
    fire.Fire(main)
