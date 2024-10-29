from ..base import BaseAlgo
from .unlearn import run_rmu
from copy import deepcopy

# context mask
# evaluation will always be on multiple choice cause i'm lazy af
# we will check completions


class GA(BaseAlgo):

    def __init__(
        self,
        model,
        tokenizer,
        forget_dataset,
        retain_dataset,
        alpha="100,100",
        module_str="{model_name}.model.layers[{layer_id}]",
        steering_coeffs="20,20",
        lr=5e-5,
        min_len=0,
        batch_size=4,
        max_num_batches=80,
        layer_id=7,
        layer_ids="5,6,7",
        param_ids="6",
        seed=42,
    ):
        super().__init__(model, tokenizer, forget_dataset, retain_dataset, alpha)
        pass

    def unlearn(self):

        frozen_model = deepcopy(self.model)

        updated_model, updated_tokenizer = run_rmu(
            self.model,
            frozen_model,
            self.tokenizer,
            self.forget_dataset,
            self.retain_dataset,
            self.alpha,
        )

    def evaluate(self, dataset):
        pass
