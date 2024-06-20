import json

from jovis_model.config import Config
from jovis_model.datasets.base import BaseDataProcessor


class BedrockProcessor(BaseDataProcessor):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = config

    def convert_features(self, dialogs):
        args = {
            "anthropic_version": self.config.params.anthropic_version,
            "max_tokens": self.config.params.max_new_tokens,
        }
        args["messages"] = dialogs
        body = json.dumps(args)
        return body
