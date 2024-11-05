import torch
from torch import nn
from peft import PeftModel
from transformers import LlamaForCausalLM
from typing import Union, Dict, List


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


def get_layer_module(model: Union[PeftModel, LlamaForCausalLM], layer: int):
    if isinstance(model, PeftModel):
        return model.model.model.layers[layer]
    else:
        return model.model.layers[layer]


def set_params_grads(params: List[nn.Parameter], requires_grad: bool):
    for param in params:
        param.requires_grad = requires_grad


def set_model_grads(model: nn.Module, requires_grad: bool):
    for param in model.parameters():
        param.requires_grad = requires_grad


def combine_losses(losses: Dict[str, torch.Tensor], loss_coefs: Dict[str, float]):
    return {
        loss_name: (loss_coefs[loss_name] * loss_value).mean()
        for loss_name, loss_value in losses.items()
    }
