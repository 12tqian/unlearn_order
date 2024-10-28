from abc import ABC, abstractmethod
from datasets import Dataset
from torch import nn
from transformers import LlamaForCausalLM, LlamaTokenizer


class BaseAlgo(ABC, nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        alpha: float,
    ):
        self.model = model
        self.alpha = alpha

    @abstractmethod
    def get_forget_loss(self):
        pass

    @abstractmethod
    def get_retain_loss(self):
        pass

    def get_loss(self, forget_batch, retain_batch):
        forget_loss = self.get_forget_loss(forget_batch)
        retain_loss = self.get_retain_loss(retain_batch)
        return self.alpha * forget_loss + (1 - self.alpha) * retain_loss

    def forward(self, forget_batch, retain_batch):
        return self.get_loss(forget_batch, retain_batch)

    




