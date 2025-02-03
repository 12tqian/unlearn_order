import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Union

import fire
import torch
from dotenv import load_dotenv
from research_tools.logging import ColoredLogger
from research_tools.utils import set_seed
from slugify import slugify
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
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
        "unlearn_A": {
            "magnitude": 6.5,
            "lr": 1e-5,
            "n_epochs": 12,
            "forget_alphas": {"A": 0.39422, "B": 0.39422},
            "retain_alphas": {"B": 13.51609, "retain": 1},
        },
        "unlearn_B": {
            "magnitude": 6.5,
            "lr": 1e-5,
            "n_epochs": 1,
            "forget_alphas": {"A": 0.39422, "B": 0.39422},
            "retain_alphas": {"B": 13.51609, "retain": 1},
            "max_batches": 100,
        },
        "rtt": {
            "lr": 1e-6,
            "n_epochs": 10,
            "batch_size": 2,
            "grad_accum_steps": 2,
        },
    }
}


def main(
    model_id: str = "HuggingFaceH4/zephyr-7b-beta",
    ds_A_name: str = "WMDP",
    ds_B_name: str = "WMDP",
    unlearn_method: str = "rmu",
    use_wandb: bool = True,
    seed: int = 42,
    device: torch.device = "cuda",
    save: bool = False,
    load_A_from: str = None,
    load_B_from: str = None,
):
    load_dotenv()
    set_seed(seed)
    hf_access_token = os.getenv("HUGGINGFACE_API_KEY")

    assert use_wandb, "Only wandb is supported for now"
    assert ds_A_name in VALID_DATASETS, f"{ds_A_name} not in {VALID_DATASETS}"
    assert ds_B_name in VALID_DATASETS, f"{ds_B_name} not in {VALID_DATASETS}"
    assert model_id in VALID_MODELS, f"{model_id} not in {VALID_MODELS}"
    assert (
        unlearn_method in VALID_UNLEARN_METHODS
    ), f"{unlearn_method} not in {VALID_UNLEARN_METHODS}"
    assert CACHE_PATH.exists(), "Cache file does not exist"
    assert CHECKPOINT_DIR.exists(), "Checkpoint directory does not exist"

    group_id = f"{model_id}-{ds_A_name}-{ds_B_name}-{unlearn_method}-{wandb.util.generate_id()}"
    group_id = slugify(group_id)

    save_dir = CHECKPOINT_DIR / group_id
    save_dir.mkdir(exist_ok=True)

    logger = ColoredLogger("relearn", logging.INFO, save_dir / "log.txt")

    logger.info(f"Loading data from {CACHE_PATH}")

    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)

    logger.info(f"Loaded data from {CACHE_PATH}")

    unlearn_config = UNLEARN_CONFIG_DICT[unlearn_method]

    logger.info(f"Starting experiment with group_id {group_id}")

    store = {
        "A": data[Datasets[ds_A_name]]["A"],
        "B": data[Datasets[ds_B_name]]["B"],
        "retain": data["retain"],
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=hf_access_token,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    eval_dict = {k: v["val"] for k, v in store.items()}

    if load_B_from and not load_A_from:
        logger.warning("load_B_from is set but load_A_from is not.")

    # forget A
    if load_A_from:
        if load_B_from is None:
            logger.info(f"Loading model from {load_A_from}, loading A only")
            model = AutoModelForCausalLM.from_pretrained(load_A_from).to(device)
    else:
        logger.info("Starting forget A")

        run_config = unlearn_config["unlearn_A"]

        run = wandb.init(
            project="relearn",
            config=run_config,
            tags=["debug", "unlearn_A"],
            entity="12tqian",
            group=group_id,
        )

        model = train_rmu(
            model,
            {"A": store["A"]["corpus"]},
            {"B": store["B"]["corpus"], "retain": store["retain"]["corpus"]},
            eval_records_dict=eval_dict,
            n_epochs=run_config["n_epochs"],
            magnitude=run_config["magnitude"],
            lr=run_config["lr"],
            forget_alphas=run_config["forget_alphas"],
            retain_alphas=run_config["retain_alphas"],
            eval_at_start=True,
            max_batches=None,
            use_wandb=True,
            debug=False,
            tokenizer=tokenizer,
        )

        if save:
            logger.info(f"Saving model to {save_dir / 'forget_A'}")
            model.save_pretrained(save_dir / "forget_A")

        run.finish()

    # forget B
    if load_B_from:
        logger.info(f"Loading model from {load_B_from}")
        model = AutoModelForCausalLM.from_pretrained(load_B_from).to(device)
    else:
        run_config = unlearn_config["unlearn_B"]
        logger.info("Starting forget B")

        run = wandb.init(
            project="relearn",
            config=run_config,
            tags=["debug", "unlearn_B"],
            entity="12tqian",
            group=group_id,
        )

        model = train_rmu(
            model,
            {"B": store["B"]["corpus"]},
            {"retain": store["retain"]["corpus"]},
            eval_records_dict=eval_dict,
            n_epochs=run_config["n_epochs"],
            magnitude=run_config["magnitude"],
            lr=run_config["lr"],
            forget_alphas=run_config["forget_alphas"],
            retain_alphas=run_config["retain_alphas"],
            eval_at_start=False,
            use_wandb=True,
            debug=False,
            tokenizer=tokenizer,
            max_batches=run_config["max_batches"],
        )

        if save:
            logger.info(f"Saving model to {save_dir / 'forget_B'}")
            model.save_pretrained(save_dir / "forget_B")

        run.finish()

    # relearn only A
    logger.info("Starting relearn A")
    run_config = unlearn_config["rtt"]
    run = wandb.init(
        project="relearn",
        config=run_config,
        tags=["debug", "rtt"],
        entity="12tqian",
        group=group_id,
    )

    new_eval_dict = {k: v for k, v in eval_dict.items() if k != "retain"}

    model = train_rtt(
        model,
        tokenizer,
        run_config["n_epochs"],
        store["A"]["mcq"],
        new_eval_dict,
        batch_size=run_config["batch_size"],
        lr=run_config["lr"],
        eval_at_start=False,
        grad_accum_steps=run_config["grad_accum_steps"],
        use_wandb=True,
    )

    if save:
        logger.info(f"Saving model to {save_dir / 'relearn_A'}")
        model.save_pretrained(save_dir / "relearn_A")

    run.finish()

    logger.info("Finished experiment")


if __name__ == "__main__":
    main()
    fire.Fire(main)
