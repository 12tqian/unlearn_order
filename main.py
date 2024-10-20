import sys

sys.path.append("./src")

import os
import dotenv
from pathlib import Path
import torch
from research_tools import get_gpus_available
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
from unlearn_order.common import TaskType, DatasetType, Task, Experiment
import torch.multiprocessing as mp
from queue import Empty
import time


def get_default_cfg():
    return Experiment()


def get_cfg(cfg_path: Path) -> Experiment:
    if cfg_path is None:
        return get_default_cfg()

    if not cfg_path.exists():
        print(f"Config file {cfg_path} not found.")
        raise FileNotFoundError(f"Config file {cfg_path} not found.")

    schema = OmegaConf.structured(Experiment)
    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(schema, cfg)

    return cfg


hf_access_token = None


def setup(cfg: Experiment) -> Experiment:
    env_path = Path(cfg.env_dir)
    if os.path.exists(env_path):
        dotenv.load_dotenv(env_path, verbose=True)
        print("Loaded environment variables from .env file.")

    hf_access_token = os.getenv("HUGGINGFACE_API_KEY")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        [str(i) for i in get_gpus_available()]
    )
    cfg.hf_access_token = hf_access_token
    return cfg


def run_experiment(cfg: Experiment):
    cfg = setup(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "No GPU available."

    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, token=cfg.hf_access_token, torch_dtype=cfg.model_dtype
    )
    model = model.to(device)

    tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, token=hf_access_token
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


def run_on_gpu(cfg: Experiment, gpu_id: int):
    # Set the visible GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Running on GPU: {gpu_id}")

    # Now, GPU 0 (the only visible one to this process) is actually the assigned one
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "No GPU available."

    results = run_experiment(cfg)

    print(f"Result on GPU {gpu_id}: {cfg.exp_name}")


def gpu_monitor(task_queue, max_gpus: int = 3):
    active_processes = {}

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
                    time.sleep(1)
                    continue
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
        time.sleep(1)


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
