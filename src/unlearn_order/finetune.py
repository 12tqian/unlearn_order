import torch

from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import DataLoader


def finetune_model(model: LlamaForCausalLM, dataloader: DataLoader, lr: float = 1e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for batch in dataloader:
        inputs, labels = batch
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model
