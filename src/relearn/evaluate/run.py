import wandb
from relearn.evaluate.mcq import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List


def run_eval(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_records_dict: Dict[str, List[Dict]],
    epoch: int,
    log_samples: bool = True,
    use_wandb: bool = True,
):
    log_dict = {"epoch": epoch}

    for eval_name, eval_records in eval_records_dict.items():
        acc = evaluate(model, eval_records, batch_size=8, normalize_loss=False)

        log_dict[f"{eval_name}/acc"] = acc
    
    if use_wandb:
        wandb.log(log_dict)
    
    return log_dict