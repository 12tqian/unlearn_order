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


def main(cfg_path: Path = None):
    cfg = get_cfg(cfg_path)
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
    breakpoint()

    run_pipeline(model, tokenizer, cfg)


if __name__ == "__main__":
    fire.Fire(main)
