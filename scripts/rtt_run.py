import os
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Union

import fire
import torch
import wandb
from dotenv import load_dotenv
from research_tools.utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer

from relearn.attacks.rtt import train_rtt
from relearn.datasets.utils import VALID_DATASETS, Datasets
from relearn.unlearn.rmu import train_rmu

VALID_MODELS = ["HuggingFaceH4/zephyr-7b-beta"]
VALID_UNLEARN_METHODS = ["rmu"]
BASE_DIR = Path("/mnt/align4_drive/tcqian/unlearn_order")

CACHE_PATH = BASE_DIR / "data" / "data.pickle"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

UNLEARN_CONFIG_DICT = {
    "rmu": {
        "magnitude": 6.5,
        "lr": 1e-5,
        "n_epochs": 1,
        "forget_alphas": {"A": 0.39422, "B": 0.39422},
        "retain_alphas": {"B": 13.51609, "retain": 1},
        "max_batches": 100,
    }
}


def main(
    model_id: str = "HuggingFaceH4/zephyr-7b-beta",
    ds_A_name: str = "MMLU",
    ds_B_name: str = "YEARS",
    ds_retain_name: str = "YEARS",
    unlearn_method: str = "rmu",
    use_wandb: bool = True,
    seed: int = 42,
    device: torch.device = "cuda",
    save: bool = False,
):
    load_dotenv()
    set_seed(seed)
    hf_access_token = os.getenv("HUGGINGFACE_API_KEY")

    assert use_wandb, "Only wandb is supported for now"
    assert ds_A_name in VALID_DATASETS, f"{ds_A_name} not in {VALID_DATASETS}"
    assert ds_B_name in VALID_DATASETS, f"{ds_B_name} not in {VALID_DATASETS}"
    assert ds_retain_name in VALID_DATASETS, f"{ds_retain_name} not in {VALID_DATASETS}"
    assert model_id in VALID_MODELS, f"{model_id} not in {VALID_MODELS}"
    assert (
        unlearn_method in VALID_UNLEARN_METHODS
    ), f"{unlearn_method} not in {VALID_UNLEARN_METHODS}"
    assert CACHE_PATH.exists(), "Cache file does not exist"
    assert CHECKPOINT_DIR.exists(), "Checkpoint directory does not exist"

    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)

    unlearn_config = UNLEARN_CONFIG_DICT[unlearn_method]
    config = {
        "model_id": model_id,
        "unlearn_method": unlearn_method,
        "ds_A_name": ds_A_name,
        "ds_B_name": ds_B_name,
        "ds_retain_name": ds_retain_name,
        "unlearn_config": unlearn_config,
    }

    group_id = f"{model_id}-{ds_A_name}-{ds_B_name}-{ds_retain_name}-{unlearn_method}-{wandb.util.generate_id()}"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=hf_access_token,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    store = {
        "A": data[Datasets[ds_A_name]]["A"],
        "B": data[Datasets[ds_B_name]]["B"],
        "retain": data[Datasets[ds_retain_name]]["retain"],
    }

    eval_dict = {k: v["val"] for k, v in store.items()}

    # forget A
    run = wandb.init(
        project="relearn",
        config=config,
        tags=["debug", "unlearn_A"],
        entity="12tqian",
        group=group_id,
    )

    model = train_rmu(
        model,
        {"A": store["A"]["corpus"]},
        {"B": store["B"]["corpus"], "retain": store["retain"]["corpus"]},
        eval_records_dict=eval_dict,
        n_epochs=unlearn_config["n_epochs"],
        magnitude=unlearn_config["magnitude"],
        lr=unlearn_config["lr"],
        forget_alphas=unlearn_config["forget_alphas"],
        retain_alphas=unlearn_config["retain_alphas"],
        eval_at_start=True,
        max_batches=None,
        use_wandb=True,
        debug=False,
        tokenizer=tokenizer,
    )

    if save:
        model.save_pretrained(CHECKPOINT_DIR / f"{group_id}-forget_A")
    run.finish()

    # forget B
    run = wandb.init(
        project="relearn",
        config=config,
        tags=["debug", "unlearn_B"],
        entity="12tqian",
        group=group_id,
    )

    model = train_rmu(
        model,
        {"B": store["B"]["corpus"]},
        {"retain": store["retain"]["corpus"]},
        eval_records_dict=eval_dict,
        n_epochs=unlearn_config["n_epochs"],
        magnitude=unlearn_config["magnitude"],
        lr=unlearn_config["lr"],
        forget_alphas=unlearn_config["forget_alphas"],
        retain_alphas=unlearn_config["retain_alphas"],
        eval_at_start=True,
        max_batches=None,
        use_wandb=True,
        debug=False,
        tokenizer=tokenizer,
        max_batches=unlearn_config["max_batches"],
    )
    if save:
        model.save_pretrained(CHECKPOINT_DIR / f"{group_id}-unlearn_B")

    run.finish()

    # relearn only A
    run = wandb.init(
        project="relearn",
        config=config,
        tags=["debug", "rtt"],
        entity="12tqian",
        group=group_id,
    )

    new_eval_dict = {k: v for k, v in eval_dict.items() if k != "retain"}

    model = train_rtt(
        model,
        tokenizer,
        10,
        store["A"]["mcq"],
        new_eval_dict,
        batch_size=2,
        lr=1e-6,
        eval_at_start=False,
        grad_accum_steps=2,
        use_wandb=True,
    )

    if save:
        model.save_pretrained(CHECKPOINT_DIR / f"{group_id}-relearn_A")

    run.finish()


if __name__ == "__main__":
    main()
    fire.Fire(main)
