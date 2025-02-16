from pathlib import Path
import dotenv
import sys
import os

env_file = "../.env"

if os.path.exists(env_file):
    dotenv.load_dotenv(env_file, verbose=True)
    print("Loaded environment variables from .env file.")


import wandb
from research_tools.utils import set_seed
import torch
import os
from pathlib import Path
import pickle
from relearn.unlearn.rmu.folds import super_rmu
from relearn.datasets import Datasets
import fire
from typing import Optional
from research_tools.logging import ColoredLogger
import logging


logger = ColoredLogger("relearn", logging.INFO)


def objective():
    wandb.init()
    config = wandb.config

    set_seed(42)

    hf_access_token = os.getenv("HUGGINGFACE_API_KEY")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == torch.device("cuda")
    model_id = "HuggingFaceH4/zephyr-7b-beta"

    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=hf_access_token,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    data_dir = Path("data")
    cache_path = data_dir / "full.pickle"

    assert cache_path.exists(), "Cache file does not exist"
    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    def trim_data(data, n):
        return {k: v[:n] for k, v in data.items()}

    forget_set = data[Datasets.WMDP]
    retain_set = data["retain"]

    # n_use = 20
    # forget_set = trim_data(data[Datasets.WMDP], n_use)
    # retain_set = trim_data(data["retain"], n_use)

    model, res = super_rmu(
        model,
        tokenizer,
        forget_set,
        retain_set,
        None,
        config.k_folds,
        magnitude=6.5,
        forget_alpha=config.forget_alpha,
        retain_alpha=config.retain_alpha,
        epochs_per_fold=config.epochs_per_fold,
        lr=config.lr,
        lr_end=config.lr * config.lr_decay,
        joint_train=True,
        prefix_forget=False,
        sweeping=True,
    )

    forget_acc = res["forget/acc"]
    retain_acc = res["retain/acc"]

    best_acc = forget_acc if retain_acc >= 0.54 else 1
    wandb.log({"score": best_acc})
    return best_acc


def initialize_sweep():
    # Example sweep configuration
    sweep_configuration = {
        "method": "bayes",
        "name": "super_rmu",
        "metric": {"goal": "minimize", "name": "score"},
        "parameters": {
            "k_folds": {
                # "values": [2]
                "values": [3, 4, 5]
            },
            "epochs_per_fold": {
                # "values": [1]
                "values": [1, 2, 3]
            },
            "lr": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-3},
            "lr_decay": {"distribution": "log_uniform_values", "min": 1e-2, "max": 1},
            "forget_alpha": {
                "distribution": "log_uniform_values",
                "min": 0.25,
                "max": 8,
            },
            "retain_alpha": {
                "distribution": "log_uniform_values",
                "min": 0.5,
                "max": 32,
            },
        },
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="relearn", entity="12tqian"
    )

    return sweep_id


def main(sweep_id: Optional[str] = None, n_trials: Optional[int] = 50):
    if sweep_id is None:
        sweep_id = initialize_sweep()

    logger.info(f"Starting sweep with id: {sweep_id}")
    wandb.agent(
        sweep_id, objective, count=n_trials, project="relearn", entity="12tqian"
    )


if __name__ == "__main__":
    fire.Fire(main)
