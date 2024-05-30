from typing import Any, Dict, List

import torch
from transformers import AutoModelForSequenceClassification

from jovis_model.models.base import BaseModel
from jovis_model.config import Config


class SCTransformer(BaseModel):
    def __init__(self, config: Config, metrics: Dict[str, Any] = None):
        super().__init__(
            config,
            use_hf_model=True,
            model_type=AutoModelForSequenceClassification,
            metrics=metrics,
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

    def _convert_outputs_to_preds(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        logits = torch.cat([output["logits"] for output in outputs], dim=0)
        return torch.argmax(logits, dim=1)

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], data_type: str = "valid"
    ):
        labels = torch.cat([output["labels"] for output in outputs], dim=0)
        preds = self._convert_outputs_to_preds(outputs)

        return self.metric(preds, labels)
