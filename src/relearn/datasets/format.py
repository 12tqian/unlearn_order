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


def create_answer_letter_answer(point: Point) -> str:
    return "\n".join(
        [
            f"{doc_to_choice[i]}. {c}"
            for i, c in enumerate(point["choices"])
            if i == point["answer"]
        ]
    )


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


def create_prompt_answer_only(point: Point) -> str:
    return "\n".join(
        [point["question"]]
        + [f"{doc_to_choice[i]}. {c}" for i, c in enumerate(point["choices"])]
        + [
            f"Answer: {doc_to_choice[i]}"
            for i, c in enumerate(point["choices"])
            if i == point["answer"]
        ]
    )


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
