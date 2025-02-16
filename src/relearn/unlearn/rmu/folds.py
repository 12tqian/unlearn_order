from relearn.unlearn.rmu import train_rmu
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional
import itertools
import wandb
from relearn.evaluate import run_eval
import numpy as np


import torch


def group_shuffle(data: List, group_size: int = 1, perm: Optional[torch.Tensor] = None):
    if perm is None:
        n = len(data) // group_size
        perm = torch.randperm(n)

    res = []
    for i in perm:
        res += data[i * group_size : (i + 1) * group_size]
    return res


def create_k_folds(data: List, k: int, group_size: int = 1):
    n = len(data) // group_size
    fold_size = n // k
    folds = [fold_size] * k
    for i in range(n % k):
        folds[i] += 1

    assert sum(folds) == n

    res = []
    start = 0

    for fold_size in folds:
        start_idx = start * group_size
        end_idx = (start + fold_size) * group_size

        res.append(data[start_idx:end_idx])

        start += fold_size

    return res


def get_folds_shuffled(records: Dict[str, List], k: int):

    perm = torch.randperm(len(records["mcq"]))

    store = [
        {"corpus": c, "mcq": m, "val": v}
        for c, m, v in zip(
            create_k_folds(group_shuffle(records["corpus"], 3, perm=perm), k, 3),
            create_k_folds(group_shuffle(records["mcq"], perm=perm), k),
            create_k_folds(
                group_shuffle(
                    records["val"],
                    4,
                    perm=perm,
                ),
                k,
                4,
            ),
        )
    ]

    return store


# informationally separate by their standards?
# no only need to eval on that
# for now same epochs for everything
# hopefully learning rate is good enough

# prior linear decay??


def super_rmu(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    forget_records_dict: Dict[str, Dict],
    retain_records_dict: Dict[str, Dict],
    eval_records_dict: Dict[str, Dict],
    k_folds: int,
    forget_alpha: float,
    retain_alpha: float,
    epochs_per_fold: int,
    lr: float = 1e-5,
    lr_end: Optional[float] = None,
    magnitude: float = 6.5,
    joint_train: bool = False,
    prefix_forget: bool = True,
    epsilon: float = 1e-6,
    use_wandb: bool = True,
    sweeping: bool = False,
):
    assert k_folds <= 26, "k_folds must be less than 26"

    def fold_name(i: int):
        return chr(ord("A") + i)

    folds = get_folds_shuffled(forget_records_dict, k_folds)

    def get_data(fold_inds: List[int]):
        if joint_train:
            return {fold_name(i): folds[i]["corpus"] for i in fold_inds}
        else:
            return {
                fold_name(-1): list(
                    itertools.chain(
                        *[f[i]["corpus"] for i, f in enumerate(folds) if i in fold_inds]
                    )
                )
            }

    if eval_records_dict is None:
        eval_records_dict = {fold_name(i): folds[i]["val"] for i in range(k_folds)}
        eval_records_dict["retain"] = retain_records_dict["val"]

    base_epoch = 0
    control_vecs = {}

    # forget alpha for one fold, two fold
    for i in range(k_folds):
        print(f"Unlearning fold {fold_name(i)}")

        if prefix_forget:
            forget_fold_inds = list(range(i + 1))
        else:
            forget_fold_inds = [i]

        cur_lr = (
            np.exp(
                (np.log(lr) * (k_folds - 1 - i) + np.log(lr_end) * i) / (k_folds - 1)
            )
            if lr_end is not None
            else lr
        )

        retain_fold_inds = list(range(i + 1, k_folds))

        forget_dict = get_data(forget_fold_inds)
        retain_dict = get_data(retain_fold_inds)
        retain_dict["retain"] = retain_records_dict["corpus"]

        # weird retain coef cause i finetuned for 2/2
        model, control_vecs_next = train_rmu(
            model,
            forget_dict,
            retain_dict,
            eval_records_dict,
            magnitude=magnitude,
            forget_alphas={
                k: (
                    forget_alpha / (i + 1 + epsilon)
                    if prefix_forget
                    else forget_alpha / (1 + epsilon)
                )
                for k in forget_dict.keys()
            },
            retain_alphas={
                **{
                    k: retain_alpha / (k_folds - i - 1 + epsilon)
                    for k in retain_dict.keys()
                },
                **{
                    "retain": 1.0,
                },
            },
            lr=cur_lr,
            tokenizer=tokenizer,
            use_wandb=use_wandb,
            eval_at_start=True if i == 0 else False,
            n_epochs=epochs_per_fold,
            max_batches=None,
            base_epoch=base_epoch,
            return_control_vecs=True,
            control_vecs_init=control_vecs,
            print_evals=True,
        )
        control_vecs.update(control_vecs_next)
        base_epoch += epochs_per_fold
    if sweeping:
        full_eval = {
            "forget": forget_records_dict["val"],
            "retain": retain_records_dict["val"],
        }
        res = run_eval(model, tokenizer, full_eval, -1)
        if use_wandb:
            wandb.log(res)

        return model, res

    return model
