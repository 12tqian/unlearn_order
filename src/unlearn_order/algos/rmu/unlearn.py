import os
import datetime

import numpy as np
import torch
from transformers import AdamW
import tqdm as tqdm
from typing import List
import argparse
from .utils import load_model, get_params, forward_with_cache, get_data
from transformers import LlamaForCausalLM


def run_rmu(
    updated_model,
    frozen_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
):
    rmu_config = vars(args)
    print("====rmu Config====")
    print("\n".join(f"{k}={v}" for k, v in rmu_config.items()))
    print("=====")

    updated_model = updated_model.train()
    params = get_params(updated_model, args.layer_ids, args.param_ids)
    optimizer = AdamW(params, lr=args.lr)
    frozen_module = eval(
        args.module_str.format(model_name="frozen_model", layer_id=args.layer_id)
    )
    updated_module = eval(
        args.module_str.format(model_name="updated_model", layer_id=args.layer_id)
    )

    control_vectors_list = []
    for i in range(len(forget_data_list)):
        random_vector = torch.rand(
            1,
            1,
            updated_model.config.hidden_size,
            dtype=updated_model.dtype,
            device=updated_model.device,
        )
        control_vec = (
            random_vector / torch.norm(random_vector) * args.steering_coeff_list[i]
        )
        control_vectors_list.append(control_vec)

    num_batches = min(
        args.max_num_batches,
        min([len(f) for f in forget_data_list]),
        min([len(r) for r in retain_data_list]),
    )

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    for epoch in range(1):
        print(f"======= Epoch {epoch} =======")
        with tqdm.tqdm(total=num_batches) as pbar:
            for idx in range(num_batches):
                topic_idx = idx % len(forget_data_list)
                batch_idx = idx // len(forget_data_list)
                control_vec = control_vectors_list[topic_idx]
                unlearn_batch = forget_data_list[topic_idx][batch_idx]
                retain_batch = retain_data_list[topic_idx][batch_idx]

                # Unlearning loss
                max_length = 512 if topic_idx == 0 else 768
                unlearn_inputs = tokenizer(
                    unlearn_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                updated_forget_activations = forward_with_cache(
                    updated_model, unlearn_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)

                unlearn_loss = torch.nn.functional.mse_loss(
                    updated_forget_activations, control_vec
                )

                # Retain loss
                retain_inputs = tokenizer(
                    retain_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(updated_model.device)
                updated_retain_activations = forward_with_cache(
                    updated_model, retain_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)
                frozen_retain_activations = forward_with_cache(
                    frozen_model, retain_inputs, module=frozen_module, no_grad=True
                ).to(updated_model.device)

                retain_loss = torch.nn.functional.mse_loss(
                    updated_retain_activations, frozen_retain_activations
                )
                retain_loss *= args.alpha[topic_idx]

                # Update model
                loss = unlearn_loss + retain_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(
                    f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}"
                )

                # ======= Logging ======
                if args.verbose:
                    frozen_forget_activations = forward_with_cache(
                        frozen_model, unlearn_inputs, module=frozen_module, no_grad=True
                    ).to(updated_model.device)
                    unlearn_cosine = torch.nn.functional.cosine_similarity(
                        updated_forget_activations, frozen_forget_activations, dim=-1
                    ).mean()
                    retain_cosine = torch.nn.functional.cosine_similarity(
                        updated_retain_activations, frozen_retain_activations, dim=-1
                    ).mean()

                    print(f"unlearn_cosine_sim={unlearn_cosine.item()}")
                    print(f"retain_cosine_sim={retain_cosine.item()}")
                    print(
                        f"Topic {topic_idx} updated_forget_activations.norm=",
                        torch.mean(
                            updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0
                        ).item(),
                    )
                    print(
                        f"Topic {topic_idx} frozen_forget_activations.norm=",
                        torch.mean(
                            frozen_forget_activations.norm(dim=-1).mean(dim=1), dim=0
                        ).item(),
                    )
                    print(
                        f"Topic {topic_idx} updated_retain_activations.norm=",
                        torch.mean(
                            updated_retain_activations.norm(dim=-1).mean(dim=1), dim=0
                        ).item(),
                    )
                    print(
                        f"Topic {topic_idx} frozen_retain_activations.norm=",
                        torch.mean(
                            frozen_retain_activations.norm(dim=-1).mean(dim=1), dim=0
                        ).item(),
                    )

                pbar.update(1)

    tokenizer.truncation_side = truncation_side
    # Save model
    if args.output_dir:
        path = args.output_dir
    else:
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = f"models/{args.model_name_or_path}_alpha-{args.alpha}_batches-{num_batches}_layer-{args.layer_id}_{date}"
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved model to {path}")


def run_rmu_wrapper(
    model_name_or_path: str = "HuggingFaceH4/zephyr-7b-beta",
    module_str: str = "{model_name}.model.layers[{layer_id}]",
    output_dir: str = None,
    retain_corpora: List[str] = ["wikitext", "wikitext"],
    forget_corpora: List[str] = ["bio-forget-corpus", "cyber-forget-corpus"],
    alpha: List[int] = [100, 100],
    steering_coeffs: List[int] = [20, 20],
    lr: float = 5e-5,
    min_len: int = 0,
    max_len: int = 2000,
    batch_size: int = 4,
    max_num_batches: int = 80,
    layer_id: int = 7,
    layer_ids: List[int] = [5, 6, 7],
    param_ids: List[int] = [6],
    seed: int = 42,
    verbose: bool = False,
):

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=model_name_or_path)
    parser.add_argument("--module_str", type=str, default=module_str)
    parser.add_argument("--output_dir", type=str, default=output_dir)
    parser.add_argument("--retain_corpora", type=str, default=",".join(retain_corpora))
    parser.add_argument("--forget_corpora", type=str, default=",".join(forget_corpora))
    parser.add_argument("--alpha", type=str, default=",".join(map(str, alpha)))
    parser.add_argument(
        "--steering_coeffs", type=str, default=",".join(map(str, steering_coeffs))
    )
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--min_len", type=int, default=min_len)
    parser.add_argument("--max_len", type=int, default=max_len)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--max_num_batches", type=int, default=max_num_batches)
    parser.add_argument("--layer_id", type=int, default=layer_id)
    parser.add_argument("--layer_ids", type=str, default=",".join(map(str, layer_ids)))
    parser.add_argument("--param_ids", type=str, default=",".join(map(str, param_ids)))
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--verbose", action="store_true", default=verbose)

    args = parser.parse_args([])

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    frozen_model, tokenizer = load_model(args.model_name_or_path)
    print("HI")
    updated_model, tokenizer = load_model(args.model_name_or_path)
    forget_data_list, retain_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        args.min_len,
        args.max_len,
        args.batch_size,
    )
    run_rmu(
        updated_model,
        frozen_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        args,
    )
