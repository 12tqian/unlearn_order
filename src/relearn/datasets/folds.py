from typing import List, Optional, Dict
import torch


def fold_name(i: int):
    return chr(ord("A") + i)


def group_shuffle(data: List, group_size: int = 1, perm: Optional[torch.Tensor] = None):
    if perm is None:
        n = len(data) // group_size
        perm = torch.randperm(n)

    res = []
    for i in perm:
        res += data[i * group_size : (i + 1) * group_size]
    return res


def create_k_folds(data: List, k: int, group_size: int = 1):
    n = len(data) // group_size
    fold_size = n // k
    folds = [fold_size] * k
    for i in range(n % k):
        folds[i] += 1

    assert sum(folds) == n

    res = []
    start = 0

    for fold_size in folds:
        start_idx = start * group_size
        end_idx = (start + fold_size) * group_size

        res.append(data[start_idx:end_idx])

        start += fold_size

    return res


def get_folds_shuffled(records: Dict[str, List], k: int):

    perm = torch.randperm(len(records["mcq"]))

    store = [
        {"corpus": c, "mcq": m, "val": v}
        for c, m, v in zip(
            create_k_folds(group_shuffle(records["corpus"], 3, perm=perm), k, 3),
            create_k_folds(group_shuffle(records["mcq"], perm=perm), k),
            create_k_folds(
                group_shuffle(
                    records["val"],
                    4,
                    perm=perm,
                ),
                k,
                4,
            ),
        )
    ]

    return store
