from typing import Dict, List

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification

from jovis_model.config import Config
from jovis_model.models.base import BaseModel
from jovis_model.utils.metric import macro_f1


class SCTransformer(BaseModel):
    def __init__(self, config: Config):
        super().__init__(
            config,
            use_hf_model=True,
            model_type=AutoModelForSequenceClassification,
            **{"config_kwargs": {"num_labels": config.params.num_labels}},
        )

        extra_model_params = (
            "encoder_layerdrop",
            "decoder_layerdrop",
            "dropout",
            "attention_dropout",
        )
        for p in extra_model_params:
            if getattr(self.config.params, p, None):
                assert hasattr(
                    self.p_config, p
                ), f"model config doesn't have a `{p}` attribute"
                setattr(self.p_config, p, getattr(self.config.params, p))

        self.metric = {"f1": macro_f1}

    def forward(self, **inputs: torch.Tensor):
        return self.model(**inputs)

    def training_step(self, batch: List[torch.Tensor]):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        outputs = self.forward(**inputs)
        loss = outputs[0]

        return {"loss": loss}

    def validation_step(self, batch: List[torch.Tensor]):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

        outputs = self.forward(**inputs)
        loss, logits = outputs[:2]

        return {"loss": loss, "logits": logits, "labels": inputs["labels"]}

    def convert_outputs(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        preds = torch.argmax(outputs["logits"], dim=1)
        labels = outputs["labels"]
        return preds, labels

    def get_metric(self, preds: np.ndarray, labels: np.ndarray) -> dict:
        res = {}
        for k, func in self.metric.items():
            res[k] = func(preds, labels)

        return res
