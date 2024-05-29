from typing import Any, Dict, List

import torch
from transformers import AutoModelForTokenClassification

from jovis_model.models.base import BaseModel
from jovis_model.config import Config


class NERTransformer(BaseModel):
    def __init__(self, config: Config, metrics: Dict[str, Any] = None):
        super().__init__(
            config,
            use_hf_model=True,
            model_type=AutoModelForTokenClassification,
            metrics=metrics,
        )

    def forward(self, **inputs: torch.Tensor):
        return self.model(**inputs)

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> dict:
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        outputs = self.forward(**inputs)
        loss = outputs[0]

        return {"loss": loss}

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, data_type: str = "valid"
    ):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        outputs = self.forward(**inputs)
        loss, logits = outputs[:2]

        return {"logits": logits, "labels": inputs["labels"]}

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], data_type: str = "valid"
    ):
        pass
