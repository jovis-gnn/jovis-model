from transformers import AutoTokenizer

import torch

from jovis_model.config import Config
from jovis_model.datasets.base import BaseDataProcessor


class ChatProcessor(BaseDataProcessor):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config
        cache_dir = (
            self.config.params.cache_dir if self.config.params.cache_dir else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            (
                config.params.tokenizer_name
                if config.params.tokenizer_name
                else config.params.hf_name
            ),
            cache_dir=cache_dir,
        )

    def _convert_features(self, dialogs):
        chat = self.tokenizer.apply_chat_template([dialogs])[0]
        tokens = torch.tensor(chat).long()
        tokens = tokens.unsqueeze(0)

        return tokens
