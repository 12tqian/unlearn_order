import sys

sys.path.append("./src")

import os
import dotenv
from pathlib import Path
from research_tools import get_gpus_available
import fire
from dataclasses import dataclass
from omegaconf import OmegaConf
from typing import List, Tuple, Dict
from enum import Enum
from unlearn_order.common import TaskType, DatasetType, Task, Experiment
from queue import Empty
import time
import multiprocessing as mp

PYTHON = Path("/mnt/align1_drive/tcqian/unlearning_order/venv/bin/python")


def gen_cfgs() -> List[Experiment]:
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
            cfg = Experiment(
                task_order=task_list,
                exp_name=f"finetune_{finetune_idx}_unlearn_{unlearn_idx}",
            )

            cfg_list.append(cfg)

    return cfg_list


def run_python_script(env_variables: Dict[str, str], args: Dict[str, str]):
    cmd = ""
    for k, v in env_variables.items():
        cmd += f"{k}={v} "
    cmd += f"{PYTHON} scripts/main.py "
    for k, v in args.items():
        cmd += f"--{k} {v} "

    print(f"Running command: {cmd}")
    os.system(cmd)


def run_on_gpu(cfg: Experiment, gpu_id: int):
    # Set the visible GPU for this process

    result_dir = Path(cfg.results_dir) / cfg.exp_name
    os.makedirs(result_dir, exist_ok=True)

    cfg_path = result_dir / "config.yaml"
    OmegaConf.save(cfg, cfg_path)

    results = run_python_script(
        {"CUDA_VISIBLE_DEVICES": str(gpu_id)}, {"cfg_path": cfg_path}
    )

    print(f"Result on GPU {gpu_id}: {cfg.exp_name}")


def gpu_monitor(task_queue: mp.Queue[Experiment], max_gpus: int = 3):
    active_processes: Dict[int, mp.Process] = {}

    while True:
        # Check for available GPUs
        available_gpus = get_gpus_available()
        print(f"Available GPUs: {available_gpus}")
        for gpu_id in available_gpus:
            if (
                gpu_id not in active_processes
                or not active_processes[gpu_id].is_alive()
            ):
                # check if we are using too many GPUs, if keep waiting
                if len(active_processes) >= max_gpus:
                    print("Too many active processes. Waiting for one to finish.")
                    break
                try:
                    # Get a new task from the queue if available
                    cfg = task_queue.get_nowait()
                    # Assign the task to the available GPU
                    print(f"Assigning task {cfg.exp_name} to GPU {gpu_id}")
                    p = mp.Process(target=run_on_gpu, args=(cfg, gpu_id))
                    p.start()
                    active_processes[gpu_id] = p
                except Empty:
                    print("No tasks left in the queue")
                    return  # Exit the monitor when all tasks are completed

        # Sleep for a bit before polling again
        time.sleep(20)


def main():
    cfg_list = gen_cfgs()
    # Check if any GPU is available

    task_queue = mp.Queue()

    # Add tasks to the queue
    for cfg in cfg_list:
        task_queue.put(cfg)

    # Start the GPU monitor to dynamically assign tasks
    monitor_process = mp.Process(target=gpu_monitor, args=(task_queue,))
    monitor_process.start()

    # Wait for the monitor to complete all tasks
    monitor_process.join()


if __name__ == "__main__":
    fire.Fire(main)
