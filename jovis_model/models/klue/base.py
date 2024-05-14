from typing import Any, Dict, Optional

import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PretrainedTokenizer,
)


class BaseTransformer:

    USE_TOKEN_TYPE_MODELS = ["bert", "xlnet", "electra"]

    def __init__(
        self,
        params: Dict[str, Any],
        num_labels: Optional[int] = None,
        mode: str = "base",
        config: Optional[PretrainedConfig] = None,
        model_type: Optional[str] = None,
        tokenizer: Optional[PretrainedTokenizer] = None,
        metrics: Dict[str, Any] = {},
        **config_kwargs: Dict[str, Any],
    ):
        self.params = params
        cache_dir = self.params.cache_dir if self.params.cache_dir else None
        if config is None:
            self.config = AutoConfig.from_pretrained(
                (
                    self.params.config_name
                    if self.params.config_name
                    else self.params.model_name_or_path
                ),
                **({"num_labels": num_labels} if num_labels else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )
        else:
            self.config: PretrainedConfig = config

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                (
                    self.params.tokenizer_name
                    if self.params.tokenizer_name
                    else self.params.model_name_or_path
                ),
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer = tokenizer

        self.model = model_type.from_pretrained(
            self.params.model_name_or_path,
            config=self.config,
            cache_dir=cache_dir,
        )
        self.metrics = nn.ModuleDict(metrics)
        self.eval_dataset_type = "valid"

    def is_use_token_type(self) -> bool:
        if self.config.model_type in set(self.USE_TOKEN_TYPE_MODELS):
            return True
        else:
            return False
