import torch
from typing import TypedDict
from transformers import LlamaTokenizer, LlamaForCausalLM
import random
from torch.nn import functional as F

MAX_SEQ_LEN = 512
doc_to_choice = ["A", "B", "C", "D"]
test_prompts = ["Hi, my name is", "Once upon a time,", "The capital of France"]


class Point(TypedDict):
    question: str
    choices: list[str]
    answer: int
    is_false: bool


# def create_prompt_letter_answer(point: Point) -> str:
#     return "\n".join(
#         [point["question"]]
#         + [f"{doc_to_choice[i]}. {c}" for i, c in enumerate(point["choices"])]
#         + [
#             f"Answer: {doc_to_choice[i]}. {c}"
#             for i, c in enumerate(point["choices"])
#             if i == point["answer"]
#         ]
#     )


def create_answer_letter_answer(point: Point) -> str:
    return "\n".join(
        [
            f"{doc_to_choice[i]}. {c}"
            for i, c in enumerate(point["choices"])
            if i == point["answer"]
        ]
    )


def get_log_probs(logits, tokens):
    log_probs = logits.log_softmax(dim=-1)
    log_probs_for_tokens = (
        log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    )
    return log_probs_for_tokens


def create_prompt_letter_answer(point: Point) -> str:
    return "\n".join(
        [point["question"]]
        + [f"{doc_to_choice[i]}. {c}" for i, c in enumerate(point["choices"])]
        + [
            f"Answer: {doc_to_choice[i]}. {c}"
            for i, c in enumerate(point["choices"])
            if i == point["answer"]
        ]
    )
    # return "\n".join(
    #     [point["question"]]
    #     + [f"{doc_to_choice[i]}. {c}" for i, c in enumerate(point["choices"])]
    #     + [
    #         f"Answer: {doc_to_choice[i]}"
    #         for i, c in enumerate(point["choices"])
    #         if i == point["answer"]
    #     ]
    # )


def create_prompt_question_answer(point: Point) -> str:
    return " ".join(
        [point["question"]]
        + [f"{c}" for i, c in enumerate(point["choices"]) if i == point["answer"]]
    )


def create_answer_letter_answer(point: Point) -> str:
    return "\n".join(
        [
            f"{doc_to_choice[i]}. {c}"
            for i, c in enumerate(point["choices"])
            if i == point["answer"]
        ]
    )


def get_loss_question_letter_answer(
    model: LlamaForCausalLM,
    batch,
    device: torch.device,
    tokenizer: LlamaTokenizer,
):
    prompts = batch

    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding=True,
    ).to(device)
    logits = model(**model.prepare_inputs_for_generation(**tokens)).logits
    print(logits.shape)
    return -get_log_probs(logits, tokens["input_ids"]).mean()


def sample_tokens(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    device,
    prompts=test_prompts,
    max_length=15,
):
    model.eval()
    generated_texts = []

    for prompt_text in prompts:
        encoded_input = tokenizer(
            prompt_text, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = encoded_input["input_ids"].to(device)
        attention_mask = encoded_input["attention_mask"].to(device)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        texts = [
            tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]
        generated_texts.extend(texts)

    return generated_texts


def create_prompt(point) -> str:
    return "\n".join(
        [point["question"]]
        + [f"{doc_to_choice[i]}. {c}" for i, c in enumerate(point["choices"])]
        + ["Answer:"]
    )


def get_loss_and_acc(
    model: LlamaForCausalLM,
    tokens: torch.Tensor,
    last_pos_label_ids: torch.Tensor,
    label_possibilities: list[int],
) -> tuple[torch.Tensor, float]:
    logits = model(**model.prepare_inputs_for_generation(**tokens)).logits[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, last_pos_label_ids)
    label_impossibilities = list(set(range(logits.shape[1])) - set(label_possibilities))
    logits[:, label_impossibilities] = -float("inf")
    acc = (logits.argmax(dim=-1) == last_pos_label_ids).float().sum().item()
    return loss, acc, logits[:, label_possibilities].detach().cpu().numpy()


def process_batch(
    batch: list[Point],
    device: torch.device,
    tokenizer: LlamaTokenizer,
    label_possibilities: list[int],
    train_on_wrong_answer: bool = False,
    print_a_prompt: bool = False,
    print_prefix: str = "prompts",
):
    prompts = [create_prompt(point) for point in batch]
    if print_a_prompt:
        print(f"{print_prefix}: {prompts}")
    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding=True,
    ).to(device)

    def get_answer(point):
        if train_on_wrong_answer:
            return random.Random(point["question"]).choice(
                [i for i in range(len(doc_to_choice)) if i != point["answer"]]
            )
        else:
            return point["answer"]

    last_pos_label_ids = torch.tensor(
        [label_possibilities[get_answer(point)] for point in batch], device=device
    )
    return tokens, last_pos_label_ids


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


def get_loss(
    model: LlamaForCausalLM,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    is_away: bool = False,
):
    # push towards completions that are correct (or away)
    # mask ensures you only look at completions
    logits = model(input_ids=tokens).logits
    final_logits = logits[:, :-1][mask[:, 1:]]
    labels = labels[:, 1:][mask[:, 1:]]
    return (
        F.cross_entropy(final_logits, labels)
        if not is_away
        else log_1_minus_p_loss(final_logits, labels)
    )


def my_loss(
    model: LlamaForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    is_away: bool = False,
):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

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
