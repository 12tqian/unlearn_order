from abc import ABC, abstractmethod
from datasets import Dataset
from torch import nn
from transformers import LlamaForCausalLM, LlamaTokenizer


class BaseAlgo(ABC, nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        tokenizer: LlamaTokenizer,
        forget_dataset: Dataset,
        retain_dataset: Dataset,
        alpha: float,
    ):
        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha

    @abstractmethod
    def unlearn(self):
        pass
