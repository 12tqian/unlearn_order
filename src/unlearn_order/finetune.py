import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import DataLoader
from functools import partial
from .utils import (
    get_loss_question_letter_answer,
    get_loss_and_acc,
    doc_to_choice,
    get_token_loss,
    get_loss,
)
from .dataset import get_finetune_dataloader
from .eval import eval_dataset
from tqdm import tqdm
from datasets import Dataset
from typing import List, Dict
from torch.utils.data import DataLoader


def finetune_model(
    model,
    tokenizer,
    dataset,
    batch_size=8,
    shuffle_labels=False,
    max_epochs=100,
    tolerance=0.01,
    eval_every=10,
    lr=3e-5,
    grad_accum_steps=1,
):
    random_chance = 1 / len(doc_to_choice)

    accuracy_cut = (
        [random_chance - tolerance, random_chance + tolerance]
        if shuffle_labels
        else [1 - tolerance, 1]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_traj = []
    acc_traj = []

    if eval_every is None:
        eval_every = max_epochs

    eval_every = min(eval_every, max_epochs)

    # for epoch in range(max_epochs):
    for epoch in tqdm(range(max_epochs)):
        model.train()
        dataloader = get_finetune_dataloader(
            dataset, tokenizer, batch_size=batch_size, shuffle_labels=shuffle_labels
        )

        # initialize the dataloader again so stuff gets randomized each time
        total_loss = 0
        n_samples = 0
        # for step, batch in enumerate(tqdm(dataloader)):

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model.device)
            # how much memory is this taking up?
            labels = batch["labels"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            output = model(input_ids=input_ids, labels=labels, return_dict=True)
            loss = output.loss
            loss = loss / grad_accum_steps

            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.detach().item() * grad_accum_steps
            n_samples += input_ids.shape[0]

        loss_traj.append(total_loss / n_samples)
        if eval_every is not None and (epoch + 1) % eval_every == 0:
            acc = eval_dataset(model, tokenizer, dataset, batch_size=batch_size)
            acc_traj.append(acc)
            print(f"Epoch {epoch + 1} loss: {total_loss / n_samples} acc: {acc}")

            if accuracy_cut[0] <= acc <= accuracy_cut[1]:
                break

    return model, loss_traj, acc_traj


def gradient_ascent(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    forget_dataset: Dataset,
    forget_collate_fn: partial,
    retain_dataset: Dataset,
    retain_collate_fn: partial,
    forget_coef: float,
    max_epochs: int = 100,
    eval_every: int = 10,
    lr: float = 3e-5,
    batch_size: int = 8,
    grad_accum_steps: int = 1,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    forget_dataloader = DataLoader(
        forget_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=forget_collate_fn,
    )
    retain_dataloader = DataLoader(
        retain_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=retain_collate_fn,
    )

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        n_samples = 0

        for step, (forget_batch, retain_batch) in enumerate(
            zip(forget_dataloader, retain_dataloader)
        ):
            forget_input_ids = forget_batch["input_ids"].to(model.device)
            forget_labels = forget_batch["labels"].to(model.device)

            retain_input_ids = retain_batch["input_ids"].to(model.device)
            retain_labels = retain_batch["labels"].to(model.device)
            forget_loss = get_loss(
                model,
                forget_input_ids,
                forget_labels,
                forget_labels != -100,
                is_away=True,
            )
            retain_loss = get_loss(
                model,
                retain_input_ids,
                retain_labels,
                retain_labels != -100,
                is_away=False,
            )

            loss = retain_loss + forget_coef * forget_loss
            loss = loss / grad_accum_steps

            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.detach().item() * grad_accum_steps
            n_samples += forget_input_ids.shape[0]
            if (step + 1) % (grad_accum_steps * 5) == 0:
                print(
                    f"Epoch {epoch} step {step + 1} forget loss: {forget_loss.item()} retain loss: {retain_loss.item()}"
                )
                print(
                    (forget_labels != -100).sum(),
                    "HI THERE",
                    forget_labels.shape,
                    forget_input_ids.shape,
                )


def finetune_model_multiple_datasets(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    datasets: Dict[str, Dataset],
    loss_coefs: Dict[str, float],
    shuffle_labels: Dict[str, bool] = None,
    batch_size: int = 8,
    max_epochs: int = 100,
    tolerance: float = 0.01,
    eval_every: int = 10,
    lr: float = 3e-5,
    grad_accum_steps: int = 1,
    collate_fns: Dict[str, partial] = None,
):
    random_chance = 1 / len(doc_to_choice)
    accuracy_cut = (
        [random_chance - tolerance, random_chance + tolerance]
        if shuffle_labels
        else [1 - tolerance, 1]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_traj = {dataset_name: [] for dataset_name in datasets}
    acc_traj = {dataset_name: [] for dataset_name in datasets}

    if eval_every is None:
        eval_every = max_epochs

    eval_every = min(eval_every, max_epochs)

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        dataloaders = {
            dataset_name: DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fns[dataset_name],
            )
            for dataset_name, dataset in datasets.items()
        }

        n_samples = 0

        min_length = min(len(dataloader) for dataloader in dataloaders.values())
        for step in tqdm(range(min_length)):
            total_loss_dict = {dataset_name: 0 for dataset_name in datasets}

            optimizer.zero_grad()

            for dataset_name, dataloader in dataloaders.items():
                batch = next(iter(dataloader))

                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)

                output = model(input_ids=input_ids, labels=labels, return_dict=True)
                loss = output.loss * loss_coefs[dataset_name]
                loss = loss / grad_accum_steps
                loss.backward()
                if torch.isnan(loss).any():
                    print(f"Found nan loss in {dataset_name}")
                    print(loss, input_ids)
                    print(batch)
                    print(grad_accum_steps)
                assert not torch.isnan(loss).any()

                total_loss_dict[dataset_name] += loss.detach().item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                # assert no gradients are nan
                for name, param in model.named_parameters():
                    if torch.isnan(param.grad).any():
                        print(f"Found nan gradient in {name}")
                optimizer.step()
                optimizer.zero_grad()
                if (step + 1) % (grad_accum_steps * 5) == 0:
                    print(f"Step {step + 1} loss: {total_loss_dict}")

            total_loss = sum(total_loss_dict.values())

            n_samples += input_ids.shape[0]

        for dataset_name in datasets:
            loss_traj[dataset_name].append(total_loss_dict[dataset_name] / n_samples)

        if eval_every is not None and (epoch + 1) % eval_every == 0:
            # acc_dict = {
            #     dataset_name: eval_dataset(
            #         model, tokenizer, dataset, batch_size=batch_size
            #     )
            #     for dataset_name, dataset in datasets.items()
            # }
            acc_dict = {}
            for dataset_name, dataset in datasets.items():
                try:
                    acc_dict[dataset_name] = eval_dataset(
                        model, tokenizer, dataset, batch_size=batch_size
                    )
                except:
                    acc_dict[dataset_name] = 0

            for dataset_name in datasets:
                acc_traj[dataset_name].append(acc_dict[dataset_name])
                print(
                    f"Dataset {dataset_name} epoch {epoch + 1} loss: {total_loss_dict[dataset_name] / n_samples} acc: {acc_dict[dataset_name]}"
                )

            if all(
                accuracy_cut[0] <= acc <= accuracy_cut[1] for acc in acc_dict.values()
            ):
                break

    return model, loss_traj, acc_traj
