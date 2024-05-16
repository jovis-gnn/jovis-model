from typing import Any, Dict, List

import torch
from transformers import AutoModelForSequenceClassification

from jovis_model.models.klue.base import BaseTransformer
from jovis_model.configs.base import BaseConfig


class SCTransformer(BaseTransformer):
    def __init__(self, config: BaseConfig, metrics: Dict[str, Any]):
        super().__init__(
            config,
            num_labels=config.params.num_labels,
            model_type=AutoModelForSequenceClassification,
            metrics=metrics,
        )

    def forward(self, **inputs: torch.Tensor):
        return self.model(**inputs)

    def training_step(self, batch: List[torch.Tensor], batch_idx: int):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if self.is_use_token_type():
            inputs["token_type_ids"] = batch[2]

        outputs = self.forward(**inputs)
        loss = outputs[0]

        return {"loss": loss}

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, data_type: str = "valid"
    ):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if self.is_use_token_type():
            inputs["token_type_ids"] = batch[2]

        outputs = self.forward(**inputs)
        loss, logits = outputs[:2]

        return {"logits": logits, "labels": inputs["labels"]}

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], data_type: str = "valid"
    ):
        labels = torch.cat([output["labels"] for output in outputs], dim=0)
        preds = self._convert_outputs_to_preds(outputs)

        return self.metric(preds, labels)

    def _convert_outputs_to_preds(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        logits = torch.cat([output["logits"] for output in outputs], dim=0)
        return torch.argmax(logits, dim=1)
