from typing import Dict, List

def fix_seq_len(batch: Dict, keys: List[str]):
    max_seq_len = max(batch["length"])
    for key in keys:
        batch[key] = batch[key][:, :max_seq_len]
    return batch