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

    data_dir = Path("../data")
    cache_path = data_dir / "full.pickle"

    assert cache_path.exists(), "Cache file does not exist"
    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    model, res = super_rmu(
        model,
        tokenizer,
        data[Datasets.WMDP],
        data["retain"],
        None,
        config.k_folds,
        forget_alpha=config.forget_alpha,
        retain_alpha=config.retain_alpha,
        epochs_per_fold=config.epochs_per_fold,
        lr=config.lr,
        lr_end=config.lr * config.lr_decay,
        joint_train=True,
        prefix_forget=False,
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
        "name": "rmu_retain",
        "metric": {"goal": "minimize", "name": "score"},
        "parameters": {
            "k_folds": {"distribution": "int_uniform_values", "min": 3, "max": 5},
            "epochs_per_fold": {
                "distribution": "int_uniform_values",
                "min": 1,
                "max": 3,
            },
            "lr": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-3},
            "lr_decay": {"distribution": "log_uniform_values", "min": 0.1, "max": 1},
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


def main(sweep_id: Optional[str] = None):
    if sweep_id is None:
        sweep_id = initialize_sweep()

    wandb.agent(sweep_id, objective, count=10, project="relearn", entity="12tqian")


if __name__ == "__main__":
    fire.Fire(main)
