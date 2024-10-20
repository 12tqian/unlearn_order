import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import DataLoader
from functools import partial
from .utils import get_loss_question_letter_answer, get_loss_and_acc, doc_to_choice
from .dataset import get_finetune_dataloader
from .eval import eval_dataset
from tqdm import tqdm


def finetune_model(
    model,
    tokenizer,
    dataset,
    batch_size=8,
    shuffle_labels=False,
    max_epochs=100,
    tolerance=0.01,
    print_every=10,
    lr=3e-5,
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
            labels = batch["labels"].to(model.device)
            output = model(input_ids=input_ids, labels=labels, return_dict=True)
            loss = output.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.detach().item()
            n_samples += input_ids.shape[0]

        acc = eval_dataset(model, tokenizer, dataset, batch_size=batch_size)

        loss_traj.append(total_loss / n_samples)
        acc_traj.append(acc)
        if print_every is not None and (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1} loss: {total_loss / n_samples} acc: {acc}")

        if accuracy_cut[0] <= acc <= accuracy_cut[1]:
            break

    return model, loss_traj, acc_traj
