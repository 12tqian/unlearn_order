from pathlib import Path
from .finetune import finetune_model
from .eval import eval_dataset
from .common import TaskType, DatasetType, Task, ExpConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from .dataset import load_dataset
import pandas as pd


def run_pipeline(model: LlamaForCausalLM, tokenizer: LlamaTokenizer, cfg: ExpConfig):
    data_dir = Path(cfg.data_dir)

    splits = cfg.splits
    n_train = cfg.n_train
    n_val = cfg.n_val

    train_files = [f"split_{splits[i]}.jsonl" for i in range(n_train)]
    val_files = [f"split_{splits[i]}.jsonl" for i in range(n_train, n_train + n_val)]
    combined_files = train_files + val_files

    train_dataset = load_dataset(data_dir, train_files)
    val_dataset = load_dataset(data_dir, val_files)
    combined_dataset = load_dataset(data_dir, combined_files)

    dataset_dict = {
        DatasetType.TRAIN: train_dataset,
        DatasetType.VAL: val_dataset,
        DatasetType.COMBINED: combined_dataset,
    }

    task_results = []
    for task in cfg.task_order:
        dataset_type = task.dataset_type
        task_type = task.task_type

        if task_type == TaskType.FINETUNE or task_type == TaskType.UNLEARN:
            shuffle_labels = task_type == TaskType.UNLEARN

            model, loss_traj, acc_traj = finetune_model(
                model,
                tokenizer,
                dataset_dict[dataset_type],
                batch_size=cfg.batch_size,
                shuffle_labels=shuffle_labels,
                tolerance=cfg.tolerance,
                max_epochs=cfg.max_epochs,
                lr=cfg.lr,
                eval_every=cfg.eval_every,
            )
            acc = acc_traj[-1]
        elif task_type == TaskType.EVAL:
            acc = eval_dataset(
                model, tokenizer, dataset_dict[dataset_type], batch_size=cfg.batch_size
            )
        else:
            raise ValueError(f"Unknown task {task_type}")

        print(f"Dataset type: {dataset_type}, task type: {task_type}, acc: {acc}")
        task_results.append(acc)

    # make a pandas dataframe with results, include task_order info
    results_df = pd.DataFrame(task_results, columns=["accuracy"])
    results_df["dataset_type"] = [task.dataset_type.value for task in cfg.task_order]
    results_df["task_type"] = [task.task_type.value for task in cfg.task_order]
    save_path = Path(cfg.results_dir) / cfg.exp_name
    save_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(Path(cfg.results_dir) / cfg.exp_name / "results.csv", index=True)

    return results_df
