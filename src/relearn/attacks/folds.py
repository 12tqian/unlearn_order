from relearn.attacks.rtt import train_rtt
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from relearn.datasets.folds import get_folds_shuffled, fold_name
import itertools
import torch


def super_rtt(
    model_path: str,
    tokenizer: AutoTokenizer,
    records: Dict[str, Dict],
    k_folds: int,
    lr: float = 5e-5,
    n_epochs: int = 24,
    use_wandb: bool = True,
    device: str = "cuda",
):
    folds = get_folds_shuffled(records, k_folds)

    for i in range(k_folds):

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(device)

        train_fold_inds = range(i + 1)

        # combine train folds into one list
        train_list = []
        for j in train_fold_inds:
            train_list.extend(folds[j]["mcq"])

        in_set = "".join([fold_name(j) for j in train_fold_inds])

        eval_dict = {
            f"{in_set}/{fold_name(j)}": folds[j]["val"] for j in range(k_folds)
        }

        model = train_rtt(
            model,
            tokenizer,
            n_epochs,
            train_list,
            eval_dict,
            lr=lr,
            use_wandb=use_wandb,
            eval_at_start=True if i == 0 else False,
            # eval_at_start=False
        )

    return model


def exponential_rtt(
    model_path: str,
    tokenizer: AutoTokenizer,
    folds: List[Dict],
    n_epochs: int = 24,
    lr: float = 5e-5,
    use_wandb: bool = True,
    device: str = "cuda",
):
    # iterate over all subsets
    subsets = []
    k_folds = len(folds)
    for i in range(1, k_folds):
        subsets.extend(itertools.combinations(range(k_folds), i))

    for subset in subsets:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(device)

        train_fold_inds = subset
        # eval_fold_inds = list(set(range(k_folds)) - set(subset))
        in_set = "".join([fold_name(i) for i in subset])

        # combine train folds into one list
        train_list = []
        for j in train_fold_inds:
            train_list += folds[j]["mcq"]

        eval_dict = {
            f"{in_set}/{fold_name(j)}": folds[j]["val"] for j in range(k_folds)
        }

        model = train_rtt(
            model,
            tokenizer,
            n_epochs,
            train_list,
            eval_dict,
            lr=lr,
            use_wandb=use_wandb,
            eval_at_start=False,
        )

    return model
