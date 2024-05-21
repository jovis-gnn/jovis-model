from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM

from jovis_model.models.base import BaseModel
from jovis_model.config import Config


class ChatModel(BaseModel):
    def __init__(self, config: Config, metrics: Dict[str, Any] = None):
        self.config = config
        super().__init__(
            self.config,
            use_hf_model=True,
            model_type=AutoModelForCausalLM,
            **{"model_kwargs": {"torch_dtype": torch.bfloat16}}
        )

    def inference(self, sample_inputs: torch.Tensor):
        outputs = self.model.generate(
            input_ids=sample_inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=True,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            repetition_penalty=1.0,
            length_penalty=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output_text = self.tokenizer.decode(outputs[0])
        return output_text
