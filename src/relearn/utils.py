import torch
from typing import Dict, List
from transformers import LlamaTokenizer, LlamaForCausalLM
import random
from torch.nn import functional as F
from torch.utils.data import Dataset
from pathlib import Path
import json


def log_1_minus_p_loss(
    logits: torch.Tensor, labels: torch.Tensor, threshold: float = -5.0
):
    """
    Copied from HarmBench repository
    Computes log(1-P(x)) in a numerically stable manner
    """
    # Compute the log(sum(exp(logits))) for each token position
    log_sum_exp_all = torch.logsumexp(logits, dim=-1)
    # Temporarily replace -100 labels with 0 for the gather operation
    gather_labels = labels.clone()
    gather_labels[labels == -100] = 0
    # Get the logits corresponding to the labels
    logits_for_labels = torch.gather(logits, -1, gather_labels.unsqueeze(-1)).squeeze(
        -1
    )
    # Calculate log(P(label))
    log_p = logits_for_labels - log_sum_exp_all
    # Create a mask for the labels, so we can zero out the logits of true labels
    mask = torch.zeros_like(logits).scatter_(-1, gather_labels.unsqueeze(-1), 1.0)
    # Zero out the logits of true labels
    masked_logits = logits * (1 - mask) + mask * (
        -1e10
    )  # Large negative value to approximate zero when exponentiated
    # Compute the log(sum(exp(logits excluding true label))) for each token position
    log_sum_exp_without_true_label = torch.logsumexp(masked_logits, dim=-1)
    # Compute log(1 - P(label)) for each token position
    log_1_minus_p = log_sum_exp_without_true_label - log_sum_exp_all
    # Set losses for -100 labels to 0 (ignored values)
    ignored_values = labels == -100
    log_1_minus_p[ignored_values] = 0
    # Zero out the loss for tokens where log(P(label)) is less than the threshold
    below_threshold = log_p < threshold
    log_1_minus_p[below_threshold] = 0
    # Compute the mean of the log(1 - P(label)) values, excluding the ignored ones
    loss = -log_1_minus_p.sum() / (~ignored_values).sum().float()
    return loss


def get_token_loss(
    model: LlamaForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    is_away: bool = False,
):
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True,
    )
    logits = output.logits

    # Shift logits and labels for causal LM
    shifted_logits = logits[:, :-1, :].contiguous()  # Exclude last token for prediction
    shifted_labels = labels[:, 1:].contiguous()  # Exclude first token for labels

    # Flatten tensors for loss computation
    shifted_logits = shifted_logits.view(
        -1, shifted_logits.size(-1)
    )  # [batch_size*seq_len, vocab_size]
    shifted_labels = shifted_labels.view(-1)  # [batch_size*seq_len]

    return (
        F.cross_entropy(shifted_logits, shifted_labels)
        if not is_away
        else log_1_minus_p_loss(shifted_logits, shifted_labels)
    )


def fix_seq_len(batch: Dict, keys: List[str]):
    max_seq_len = max(batch["length"])
    for key in keys:
        batch[key] = batch[key][:, :max_seq_len]
    return batch
