import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from jovis_model.config import Config


class BaseTransformer:

    USE_TOKEN_TYPE_MODELS = ["bert", "xlnet", "electra"]

    def __init__(
        self,
        config: Config,
        num_labels: Optional[int] = None,
        mode: str = "base",
        config_: Optional[PretrainedConfig] = None,
        model_type: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        metrics: Dict[str, Any] = {},
        **config_kwargs: Dict[str, Any],
    ):
        self.config = config
        cache_dir = (
            self.config.params.cache_dir if self.config.params.cache_dir else None
        )
        if config_ is None:
            self.config_ = AutoConfig.from_pretrained(
                (
                    self.config.params.config_name
                    if self.config.params.config_name
                    else self.config.params.model_name_or_path
                ),
                **({"num_labels": num_labels} if num_labels else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )
        else:
            self.config_: PretrainedConfig = config_

        extra_model_params = (
            "encoder_layerdrop",
            "decoder_layerdrop",
            "dropout",
            "attention_dropout",
        )
        for p in extra_model_params:
            if getattr(self.config.params, p, None):
                assert hasattr(
                    self.config_, p
                ), f"model config doesn't have a `{p}` attribute"
                setattr(self.config_, p, getattr(self.config.params, p))

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                (
                    self.config.params.tokenizer_name
                    if self.config.params.tokenizer_name
                    else self.config.params.model_name_or_path
                ),
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer = tokenizer

        self.model = model_type.from_pretrained(
            self.config.params.hf_name,
            from_tf=bool(".ckpt" in self.config.params.hf_name),
            config=self.config_,
            cache_dir=cache_dir,
        )
        self.metrics = nn.ModuleDict(metrics)
        self.eval_dataset_type = "valid"

    def is_use_token_type(self) -> bool:
        if self.config_.model_type in set(self.USE_TOKEN_TYPE_MODELS):
            return True
        else:
            return False

    def total_steps(self) -> Any:
        num_devices = max(1, self.config.params.num_gpus)
        effective_batch_size = (
            self.config.params.train_batch_size
            * self.config.params.gradient_accumulation_steps
            * num_devices
        )
        return (
            self.config.params.dataset_size / effective_batch_size
        ) * self.config.params.num_train_epochs

    def num_warmup_steps(self) -> Any:
        num_warmup_steps = self.config.params.warmup_steps
        if num_warmup_steps is None and self.config.params.warmup_ratio is not None:
            num_warmup_steps = self.total_steps() * self.config.params.warmup_ratio
            num_warmup_steps = math.ceil(num_warmup_steps)

        if num_warmup_steps is None:
            num_warmup_steps = 0
        return num_warmup_steps

    def get_lr_scheduler(self) -> Any:
        arg_to_scheduler = {
            "linear": get_linear_schedule_with_warmup,
            "cosine": get_cosine_schedule_with_warmup,
            "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
            "polynomial": get_polynomial_decay_schedule_with_warmup,
        }
        get_schedule_func = arg_to_scheduler[self.config.params.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt,
            num_warmup_steps=self.num_warmup_steps(),
            num_training_steps=self.total_steps(),
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.params.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.config.params.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.config.params.learning_rate,
                scale_parameter=False,
                relative_step=False,
            )
        else:
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.config.params.learning_rate,
                eps=self.config.params.adam_epsilon,
            )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()
        return optimizer, scheduler

    def training_step(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def training_step_end(
        self, training_step_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {"loss": training_step_outputs["loss"].mean()}

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, data_type: str
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _convert_outputs_to_preds(self, outputs: List[Dict[str, torch.Tensor]]) -> Any:
        # outputs is output (dict, return object from validation_step) of list
        raise NotImplementedError

    def _set_metrics_device(self) -> None:
        device = next(self.parameters()).device
        for _, metric in self.metrics.items():
            if metric.device is None:
                metric.device = device

    def validation_epoch_end(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        data_type: str = "valid",
        write_predictions: bool = False,
    ) -> None:
        preds = self._convert_outputs_to_preds(outputs)
        labels = torch.cat([output["labels"] for output in outputs], dim=0)

        if write_predictions is True:
            self.predictions = preds

        self._set_metrics_device()
        for k, metric in self.metrics.itmes():
            metric(preds, labels)
