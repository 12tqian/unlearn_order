from relearn.unlearn.rmu import train_rmu
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional
import itertools
import wandb
from relearn.evaluate import run_eval
import numpy as np
from relearn.datasets.folds import get_folds_shuffled, fold_name

import torch
from torch.optim import AdamW
from .utils import get_params
from torch.optim.lr_scheduler import ExponentialLR

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
    batch_size: int = 4,
    lr_decay: float = 1.0,
    magnitude: float = 6.5,
    joint_train: bool = False,
    prefix_forget: bool = True,
    epsilon: float = 1e-6,
    use_wandb: bool = True,
    sweeping: bool = False,
    same_optimizer: bool = False,
    activation_layer: int = 7,
    train_layers: List[int] = [5, 6, 7],
    param_names: List[str] = ["down_proj"],
    actual_retain_alpha: float = 1.0,
):
    assert k_folds <= 26, "k_folds must be less than 26"

    folds = get_folds_shuffled(forget_records_dict, k_folds)

    optimizer = AdamW(
        get_params(model, train_layers, param_names),
        lr=lr,
    )
    scheduler = ExponentialLR(optimizer, gamma=lr_decay)

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

    penalty = 0

    # forget alpha for one fold, two fold
    for i in range(k_folds):
        print(f"Unlearning fold {fold_name(i)}")

        if prefix_forget:
            forget_fold_inds = list(range(i + 1))
        else:
            forget_fold_inds = [i]

        cur_lr = lr * (lr_decay**i) if not same_optimizer else lr

        retain_fold_inds = list(range(i + 1, k_folds))

        forget_dict = get_data(forget_fold_inds)
        retain_dict = get_data(retain_fold_inds)
        retain_dict["retain"] = retain_records_dict["corpus"]

        # weird retain coef cause i finetuned for 2/2
        model, control_vecs_next, eval_dict = train_rmu(
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
                    # unsure if below is necessary?
                    "retain": actual_retain_alpha
                    + (retain_alpha if i == k_folds - 1 else 0)
                },
            },
            lr=cur_lr,
            tokenizer=tokenizer,
            use_wandb=use_wandb,
            eval_at_start=True if i == 0 else False,
            n_epochs=epochs_per_fold,
            batch_size=batch_size,
            max_batches=None,
            base_epoch=base_epoch,
            return_control_vecs=True,
            control_vecs_init=control_vecs,
            print_evals=True,
            monitor_name=f"{fold_name(i)}/acc",
            activation_layer=activation_layer,
            train_layers=train_layers,
            param_names=param_names,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        penalty += 1

        if eval_dict["retain/acc"] < 0.54:
            break

        control_vecs.update(control_vecs_next)
        base_epoch += epochs_per_fold
    if sweeping:
        full_eval = {
            "forget": forget_records_dict["val"],
            "retain": retain_records_dict["val"],
        }
        res = run_eval(model, tokenizer, full_eval, -1)

        if res["retain/acc"] < 0.54:
            # i hope this gives a better signal
            res["retain/acc"] = res["retain/acc"] - penalty

        if use_wandb:
            wandb.log(res)

        return model, res

    return model
