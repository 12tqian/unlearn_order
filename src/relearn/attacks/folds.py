from relearn.attacks.rtt import train_rtt
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from relearn.datasets.folds import get_folds_shuffled, fold_name


def super_rtt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: Dict[str, Dict],
    k_folds: int,
):

    folds = get_folds_shuffled(records, k_folds)

    for i in range(k_folds):

        train_fold_inds = range(i + 1)
        eval_fold_inds = range(i + 1, k_folds)

        train_dict = {fold_name(i): folds[i]["corpus"] for i in train_fold_inds}

        eval_dict = {fold_name(i): folds[i]["val"] for i in eval_fold_inds}

        model = train_rtt(
            model,
            train_dict,
            eval_dict,
            lr=5e-5,
            tokenizer=tokenizer,
            use_wandb=True,
            eval_at_start=True if i == 0 else False,
            n_epochs=2,
            max_batches=None,
            base_epoch=0,
        )
