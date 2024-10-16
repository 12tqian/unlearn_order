import torch

MAX_SEQ_LEN = 512
doc_to_choice = ["A", "B", "C", "D"]
test_prompts = ["Hi, my name is", "Once upon a time,", "The capital of France"]


doc_to_choice = ["A", "B", "C", "D"]


def create_prompt_letter_answer(point) -> str:
    return "\n".join(
        [point["question"]]
        + [f"{doc_to_choice[i]}. {c}" for i, c in enumerate(point["choices"])]
        + [
            f"Answer: {doc_to_choice[i]}. {c}"
            for i, c in enumerate(point["choices"])
            if i == point["answer"]
        ]
    )


def create_answer_letter_answer(point) -> str:
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


def get_loss_question_letter_answer(
    model,
    batch,
    device: torch.device,
    tokenizer: AutoTokenizer,
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
    return -get_log_probs(logits, tokens["input_ids"]).mean()


def sample_tokens(model, tokenizer, device, prompts=test_prompts, max_length=15):
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
    model, tokens, last_pos_label_ids, label_possibilities
) -> tuple[torch.Tensor, float]:
    logits = model(**model.prepare_inputs_for_generation(**tokens)).logits[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, last_pos_label_ids)
    label_impossibilities = list(set(range(logits.shape[1])) - set(label_possibilities))
    logits[:, label_impossibilities] = -float("inf")
    acc = (logits.argmax(dim=-1) == last_pos_label_ids).float().sum().item()
    return loss, acc, logits[:, label_possibilities].detach().cpu().numpy()
