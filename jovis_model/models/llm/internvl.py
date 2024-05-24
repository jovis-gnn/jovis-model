from typing import Any, Dict

import torch
from transformers import AutoModel

from jovis_model.models.base import BaseModel
from jovis_model.config import Config


class InternVLModel(BaseModel):
    def __init__(self, config: Config, metrics: Dict[str, Any] = None):
        self.config = config
        super().__init__(
            self.config,
            use_hf_model=True,
            model_type=AutoModel,
            **{
                "config_kwargs": {
                    "trust_remote_code": True,
                },
                "model_kwargs": {
                    "torch_dtype": torch.bfloat16,
                    "trust_remote_code": True,
                },
            }
        )

    def inference(self, sample_inputs: torch.Tensor):
        generation_config = {
            "num_beams": 1,
            "max_new_tokens": self.config.params.max_new_tokens,
        }
        pixel_values, prompt = sample_inputs
        outputs = self.model.chat(
            self.tokenizer, pixel_values, prompt, generation_config
        )
        return outputs
