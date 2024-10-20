from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import DataLoader
import torch
from .utils import doc_to_choice
from .dataset import get_eval_dataloader


def eval_dataset(
    model: LlamaForCausalLM, tokenizer: LlamaTokenizer, dataset, batch_size=8
):
    torch.cuda.empty_cache()

    n_choices = len(doc_to_choice)
    new_batch_size = batch_size // n_choices
    dataloader = get_eval_dataloader(dataset, tokenizer, batch_size=new_batch_size)
    model.eval()

    n_correct = 0
    n_total = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(model.device)
        labels = batch["labels"].to(model.device)
        output = model(input_ids=input_ids, labels=labels, return_dict=True)

        # for each, do byte length normalized completion probability
        # then do the averag
        logits = output.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # get look ahead
        log_probs = (
            log_probs[:, :-1]
            .gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )

        prompt_mask = batch["prompt_mask"].to(model.device)
        # get things that are not in prompt, set them to 0
        log_probs[prompt_mask[:, :-1].bool()] = 0
        completion_log_probs = log_probs.sum(dim=-1)

        full_str = batch["full_str"]
        prompt_str = batch["prompt_str"]

        answer_str = [full_str[i][len(prompt_str[i]) :] for i in range(len(full_str))]

        byte_lengths = [len(s.encode("utf-8")) for s in answer_str]
        byte_lengths = torch.tensor(byte_lengths, device=model.device)

        completion_normalized_log_probs = completion_log_probs / byte_lengths

        n_choices = len(doc_to_choice)
        # n_choices x batch_size
        completion_normalized_log_probs = completion_normalized_log_probs.view(
            -1, n_choices
        )
        completion_choice = completion_normalized_log_probs.argmax(dim=-1)

        answers = torch.tensor(batch["answers"], device=model.device)
        n_correct += (answers == completion_choice).sum().item()
        n_total += answers.shape[0]

    accuracy = n_correct / n_total

    return accuracy
