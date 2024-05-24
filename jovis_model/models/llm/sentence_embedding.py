from typing import Any, Dict

import torch
from transformers import AutoModel

from jovis_model.models.base import BaseModel
from jovis_model.config import Config


class SentenceEmbedding(BaseModel):
    def __init__(self, config: Config, metrics: Dict[str, Any] = None):
        self.config = config
        super().__init__(
            self.config,
            use_hf_model=True,
            model_type=AutoModel,
        )

    def inference(self, sample_inputs: torch.Tensor):
        outputs = self.model(**sample_inputs)
        token_embeddings = outputs[0]
        input_mask_expanded = (
            sample_inputs["attention_mask"]
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        outputs = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        return outputs
