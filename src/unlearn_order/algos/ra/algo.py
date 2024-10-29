from ..base import BaseAlgo
from copy import deepcopy

class RA(BaseAlgo):
    def __init__(
        self,
        model,
        tokenizer,
    ):
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
