from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)

from jovis_model.config import Config


class BaseModel:

    def __init__(
        self,
        config: Config,
        use_hf_model: bool,
        model_type: Optional[str] = None,
        **kwargs: Dict[str, Dict[str, Any]],
    ):
        self.config = config
        if use_hf_model:
            cache_dir = (
                self.config.params.cache_dir if self.config.params.cache_dir else None
            )

            config_kwargs = kwargs.get("config_kwargs", {})
            self.p_config: PretrainedConfig = AutoConfig.from_pretrained(
                (
                    self.config.params.config_name
                    if self.config.params.config_name
                    else self.config.params.hf_name
                ),
                cache_dir=cache_dir,
                **config_kwargs,
            )

            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                (
                    self.config.params.tokenizer_name
                    if self.config.params.tokenizer_name
                    else self.config.params.hf_name
                ),
                cache_dir=cache_dir,
                **config_kwargs,
            )

            model_kwargs = kwargs.get("model_kwargs", {})
            self.model = model_type.from_pretrained(
                self.config.params.hf_name,
                from_tf=bool(".ckpt" in self.config.params.hf_name),
                config=self.p_config,
                cache_dir=cache_dir,
                **model_kwargs,
            )
        else:
            self.model = model_type(self.config)

    def forward(self, **inputs: Dict[str, List[torch.Tensor]]):
        raise NotImplementedError

    def training_step(self, batch: List[torch.Tensor]):
        raise NotImplementedError

    def validation_step(self, batch: List[torch.Tensor]):
        raise NotImplementedError

    def convert_outputs(self, outputs: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def get_metric(self):
        raise NotImplementedError

    def inference(self, sample_inputs: torch.Tensor):
        raise NotImplementedError
