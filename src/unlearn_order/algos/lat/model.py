from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from torch import nn

# import autotokenizer
from transformers import AutoTokenizer
from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import DataLoader
from dataclasses import dataclass

from omegaconf import DictConfig
from .hooks import add_hooks
from .utils import log_1_minus_p_loss, get_layer_module, set_model_grads, combine_losses
from peft import PeftModel
from peft.tuners.lora import LoraModel
from jaxtyping import Float


class Adversary(nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        attack_mask: torch.Tensor,
        epsilon: float,
    ):
        super(Adversary, self).__init__()
        self.hidden_size = model.config.hidden_size
        self.attack_mask = attack_mask
        self.epsilon = epsilon

        self.batch_size = attack_mask.size(0)
        self.seq_len = attack_mask.size(1)
        self.attack = nn.Parameter(
            torch.zeros(
                self.batch_size,
                self.seq_len,
                self.hidden_size,
                device=model.device,
                dtype=model.dtype,
            ),
        )

    def set_attack_grads(self, requires_grad: bool):
        self.attack.requires_grad = requires_grad

    def forward(self, x: torch.Tensor):
        x[self.attack_mask] += self.attack[self.attack_mask]
        return x

    def project(self):
        with torch.no_grad():
            norms = torch.norm(self.attack, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.attack.div_(scale)


def get_adversary_hook(
    adversary: Adversary,
):
    def hook_fn(module, input, output):
        nonlocal adversary

        if isinstance(output, tuple):
            activation: Float[torch.Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[torch.Tensor, "batch_size seq_len d_model"] = output

        perturbed_activation = adversary.forward(activation)

        if isinstance(output, tuple):
            return (perturbed_activation, *output[1:])
        else:
            return perturbed_activation

    return hook_fn


def get_token_loss(
    model: LlamaForCausalLM,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    is_away: bool = False,
):
    # push towards completions that are correct (or away)
    # mask ensures you only look at completions

    logits = model(input_ids=tokens).logits
    final_logits = logits[:, :-1][mask[:, 1:]]
    labels = tokens[:, 1:][mask[:, 1:]]
    return (
        F.cross_entropy(final_logits, labels)
        if not is_away
        else log_1_minus_p_loss(final_logits, labels)
    )


def get_towards_away_loss(
    model: LlamaForCausalLM, batch: Dict[str, torch.Tensor], is_adv: bool = True
):
    # i need coefficients?
    towards_tokens = batch["adv_tokens"].to(model.device)
    away_tokens = batch["def_tokens"].to(model.device)
    towards_labels_mask = batch["adv_labels_mask"].to(model.device)
    away_labels_mask = batch["def_labels_mask"].to(model.device)

    if not is_adv:
        towards_tokens, away_tokens = away_tokens, towards_tokens
        towards_labels_mask, away_labels_mask = (
            away_labels_mask,
            towards_labels_mask,
        )

    towards_loss = get_token_loss(model, towards_tokens, towards_labels_mask)
    away_loss = get_token_loss(model, away_tokens, away_labels_mask, is_away=True)
    losses = {
        "towards": towards_loss,
        "away": away_loss,
    }
    # for k, v in losses.items():
    #     v.backward(retain_graph=True)
    return losses


def projected_gradient_descent(
    model: LlamaForCausalLM,
    batch: Dict[str, torch.Tensor],
    layer: int,
    attack_completions: bool = False,
    epsilon: float = 20.0,
    n_pgd_steps: int = 16,
    lr_adv: float = 1e-3,
    loss_coefs: Dict[str, float] = {
        "towards": 1.0,
        "away": 1.0,
        "kl": 1.0,
        "sft": 1.0,
    },
):
    set_model_grads(model, False)

    attack_mask = batch["prompt_mask"]

    if attack_completions:
        adv_mask = batch["adv_labels_mask"]
        def_mask = batch["def_labels_mask"]
        attack_mask = torch.any(torch.stack([attack_mask, adv_mask, def_mask]), dim=0)

    adversary = Adversary(model, attack_mask, epsilon)

    optimizer = torch.optim.Adam(adversary.parameters(), lr=lr_adv)

    for _ in range(n_pgd_steps):
        optimizer.zero_grad()

        with add_hooks(
            module_forward_hooks=[
                (get_layer_module(model, layer), get_adversary_hook(adversary))
            ],
            module_forward_pre_hooks=[],
        ):
            losses = get_towards_away_loss(model, batch)

        # TODO(tcqian): investigate gradient clipping and nans
        total_loss = sum(combine_losses(losses, loss_coefs).values())
        total_loss.backward()
        optimizer.step()

        # assumption is you always step/project in that order
        adversary.project()

    return adversary


class LATModel(nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        tokenizer: LlamaTokenizer,
        n_def_per_adv_steps: int = 4,
        loss_coefs: Dict[str, float] = {
            "towards": 1.0,
            "away": 1.0,
            "kl": 1.0,
            "sft": 1.0,
        },
        layer: int = 12,
        attack_completions: bool = True,
        epsilon: float = 20.0,
        n_pgd_steps: int = 16,
        lr_adv: float = 1e-3,
    ):
        super(LATModel, self).__init__()

        self.model = model
        self.epsilon = epsilon
        self.tokenizer = tokenizer
        self.attack_completions = attack_completions
        self.n_pgd_steps = n_pgd_steps
        self.layer = layer
        self.loss_coefs = loss_coefs
        self.lr_adv = lr_adv

    def train_adversary(self, batch: Dict[str, torch.Tensor]):
        """
        Run projected gradient descent. You shouldn't add to completion. Let's say you add in the 12th layer. Then just look at last token position to find the limit.
        """
        return projected_gradient_descent(
            self.model,
            batch,
            attack_completions=self.attack_completions,
            layer=self.layer,
            epsilon=self.epsilon,
            n_pgd_steps=self.n_pgd_steps,
            lr_adv=self.lr_adv,
            loss_coefs=self.loss_coefs,
        )

    def train_defense(
        self,
        batch: Dict[str, torch.Tensor],
        sft_batch: Dict[str, torch.Tensor],
        adversary: Adversary,
    ):
        set_model_grads(self.model, True)
        set_model_grads(adversary, False)

        with add_hooks(
            module_forward_hooks=[
                (
                    get_layer_module(self.model, self.layer),
                    get_adversary_hook(adversary),
                )
            ],
            module_forward_pre_hooks=[],
        ):
            losses = get_towards_away_loss(self.model, batch, is_adv=False)

        if "sft" in self.loss_coefs and self.loss_coefs["sft"] > 0:
            sft_tokens = sft_batch["def_tokens"].to(self.model.device)
            sft_labels_mask = sft_batch["def_labels_mask"].to(self.model.device)

            losses["sft"] = get_token_loss(self.model, sft_tokens, sft_labels_mask)

        if "kl" in self.loss_coefs and self.loss_coefs["kl"] > 0:
            sft_tokens = sft_batch["def_tokens"].to(self.model.device)
            sft_labels_mask = sft_batch["def_labels_mask"].to(self.model.device)
            assert isinstance(
                self.model, PeftModel
            ), "The model must be a peft_model to run KL-penalty"

            with torch.no_grad():
                self.model.disable_adapter_layers()
                base_logits = self.model(input_ids=sft_tokens).logits
                base_logits = torch.log_softmax(base_logits[sft_labels_mask], dim=-1)
                self.model.enable_adapter_layers()

            new_logits = self.model(input_ids=sft_tokens).logits
            new_logits = torch.softmax(new_logits[sft_labels_mask], dim=-1)
            losses["kl"] = F.kl_div(base_logits, new_logits)

        losses = combine_losses(losses, self.loss_coefs)

        return losses

    def forward(
        self, batch: Dict[str, torch.Tensor], sft_batch: Dict[str, torch.Tensor]
    ):
        adversary = self.train_adversary(batch)
        losses = self.train_defense(batch, sft_batch, adversary)

        return losses
