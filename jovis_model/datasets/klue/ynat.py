import os
import json
from typing import Any, Dict, List

import torch
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizer

from jovis_model.datasets.common import DataProcessor
from jovis_model.datasets.inputs import InputExample, InputFeatures
from jovis_model.datasets.klue.utils import convert_examples_to_features


class YNATProcessor(DataProcessor):
    def __init__(self, params: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(params, tokenizer)

    def get_labels(self) -> List[str]:
        return ["정치", "경제", "사회", "생활문화", "세계", "IT과학", "스포츠"]

    def _create_examples(self, file_path: str) -> List[InputExample]:
        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            data_lst = json.load(f)

        for data in data_lst:
            guid, title, label = data["guid"], data["title"], data["label"]
            examples.append(InputExample(guid=guid, text_a=title, label=label))

        return examples

    def _convert_features(self, examples: List[InputExample]) -> List[InputFeatures]:
        return convert_examples_to_features(
            examples,
            self.tokenizer,
            label_list=self.get_labels(),
            max_length=self.hparams.max_seq_length,
            task_mode="classification",
        )

    def _create_dataset(self, file_path: str):
        examples = self._create_examples(file_path)
        features = self._convert_features(examples)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long
        )
        all_token_type_ids = torch.tensor(
            [0 if f.token_type_ids is None else f.token_type_ids for f in features],
            dtype=torch.long,
        )
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels
        )
        dataset.examples = examples
        return dataset

    def get_dataset(self, data_dir: str, file_name: str):
        file_path = os.path.join(data_dir, file_name)

        return self._create_dataset(file_path)
