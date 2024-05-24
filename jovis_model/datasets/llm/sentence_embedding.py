from transformers import AutoTokenizer

from jovis_model.config import Config
from jovis_model.datasets.base import BaseDataProcessor


class SentenceProcessor(BaseDataProcessor):
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

    def _convert_features(self, sentences):
        tokens = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )

        return tokens
