from typing import Dict, List
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from relearn.utils import get_token_loss, fix_seq_len
from relearn.evaluate.mcq import evaluate


def train_step_gd(
    model: AutoModelForCausalLM,
    forget_batch: Dict,
    retain_batch: Dict,
    forget_alpha: float = 0.1,
    use_log_1_minus_p: bool = True,
    retain_both: bool = False,
):
    forget_batch = fix_seq_len(
        forget_batch, ["input_ids", "attention_mask", "labels", "completion_mask"]
    )
    retain_batch = fix_seq_len(
        retain_batch, ["input_ids", "attention_mask", "labels", "completion_mask"]
    )

    forget_input_ids = forget_batch["input_ids"].to(model.device)
    forget_attention_mask = forget_batch["attention_mask"].to(model.device)
    forget_labels = forget_batch["labels"].to(model.device)
    forget_completion_mask = forget_batch["completion_mask"].to(model.device)

    retain_input_ids = retain_batch["input_ids"].to(model.device)
    retain_attention_mask = retain_batch["attention_mask"].to(model.device)
    retain_labels = retain_batch["labels"].to(model.device)
    retain_completion_mask = retain_batch["completion_mask"].to(model.device)

    forget_labels[~forget_completion_mask.bool()] = -100
    retain_labels[~retain_completion_mask.bool()] = -100

    forget_is_away = use_log_1_minus_p
    forget_sign = 1 if use_log_1_minus_p else -1

    if retain_both:
        forget_is_away = False
        forget_sign = 1

    forget_loss = (
        get_token_loss(
            model,
            is_away=forget_is_away,
            input_ids=forget_input_ids,
            attention_mask=forget_attention_mask,
            labels=forget_labels,
        )
        * forget_sign
    )

    forget_loss = forget_loss * forget_alpha
    forget_loss.backward()

    retain_loss = get_token_loss(
        model,
        is_away=False,
        input_ids=retain_input_ids,
        attention_mask=retain_attention_mask,
        labels=retain_labels,
    )

    retain_loss.backward()

    loss = forget_loss.detach() + retain_loss.detach()
    loss = loss.detach()
    forget_loss = forget_loss.detach()
    retain_loss = retain_loss.detach()

    loss_dict = {
        "loss": loss,
        "forget_loss": forget_loss,
        "retain_loss": retain_loss,
    }

    return loss_dict


def train_epoch_gd(
    model: AutoModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    forget_train_records: List[Dict],
    retain_train_records: List[Dict],
    retain_both: bool = False,
    batch_size: int = 4,
    forget_alpha: float = 0.1,
    log_steps: int = 50,
    grad_accum_steps: int = 1,
    use_log_1_minus_p: bool = True,
):
    model.train()
    forget_dataloader = DataLoader(
        forget_train_records, batch_size=batch_size, shuffle=True
    )
    retain_dataloader = DataLoader(
        retain_train_records, batch_size=batch_size, shuffle=True
    )
    loss_traj = []
    for step, (forget_batch, retain_batch) in (
        pbar := tqdm(enumerate(zip(forget_dataloader, retain_dataloader)))
    ):
        loss_dict = train_step_gd(
            model,
            forget_batch,
            retain_batch,
            forget_alpha,
            use_log_1_minus_p,
            retain_both=retain_both,
        )

        for k, v in loss_dict.items():
            if torch.isnan(v):
                print(f"{step}: Loss {k} is NaN")
                continue

        for k, v in loss_dict.items():
            if torch.isinf(v):
                print(f"{step}: Loss {k} is Inf")
                print(forget_batch["labels"])
                continue

        pbar.set_postfix(
            {
                "Loss": loss_dict["loss"].item(),
                "Forget Loss": loss_dict["forget_loss"].item(),
                "Retain Loss": loss_dict["retain_loss"].item(),
            }
        )

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        loss_traj.append(loss_dict)
        if (step + 1) % log_steps == 0:
            print(
                f"Epoch {epoch}, Step {step}, Loss {loss_dict['loss']}, Forget Loss {loss_dict['forget_loss']}, Retain Loss {loss_dict['retain_loss']}"
            )

    return loss_traj


def train_gd(
    model: AutoModelForCausalLM,
    n_epochs: int,
    forget_train_records: List[Dict],
    retain_train_records: List[Dict],
    eval_records_dict: Dict[str, List[Dict]],
    retain_both: bool = False,
    batch_size: int = 4,
    forget_alpha: float = 0.1,
    lr: float = 3e-5,
    log_steps: int = 50,
    eval_at_start: bool = True,
    grad_accum_steps: int = 1,
    use_log_1_minus_p: bool = True,
    eval: bool = True,
    params: List = None,
):
    def run_eval(prefix: str):
        if eval:
            for eval_name, eval_records in eval_records_dict.items():
                acc = evaluate(model, eval_records, batch_size=8, normalize_loss=False)
                print(f"{prefix} {eval_name} Accuracy: {acc}")

    if eval_at_start:
        run_eval("Start")

    optimizer = torch.optim.Adam(
        params=model.parameters() if params is None else params, lr=lr
    )

    for epoch in range(n_epochs):
        train_epoch_gd(
            model,
            optimizer,
            epoch,
            forget_train_records,
            retain_train_records,
            batch_size=batch_size,
            forget_alpha=forget_alpha,
            retain_both=retain_both,
            log_steps=log_steps,
            grad_accum_steps=grad_accum_steps,
            use_log_1_minus_p=use_log_1_minus_p,
        )
        run_eval(f"Epoch {epoch}")
    return model
