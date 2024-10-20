from .finetune import finetune_model
from .eval import eval_dataset


def run_pipeline(
    model, tokenizer, experiments, batch_size=16, tolerance=0.05, max_epochs=100
):
    for t, name, dataset in experiments:
        shuffle_labels = False
        if t == "f" or t == "u":
            shuffle_labels = t == "u"
            model, loss_traj, acc_tracj = finetune_model(
                model,
                tokenizer,
                dataset,
                batch_size=batch_size,
                shuffle_labels=shuffle_labels,
                tolerance=tolerance,
                max_epochs=max_epochs,
            )
            print(f"task {t}.{name} done with accuracy {acc_tracj[-1]}")
        elif t == "e":
            acc = eval_dataset(model, tokenizer, dataset, batch_size=batch_size)
            print(f"task {t}.{name} done with accuracy {acc}")
        else:
            raise ValueError(f"Unknown task {t}")
