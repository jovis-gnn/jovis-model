import os
import re
import unicodedata
from typing import List

import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

from jovis_model.config import Config
from jovis_model.datasets.base import BaseDataProcessor
from jovis_model.datasets.input import TextToLabel, TextToLabelFeature
from jovis_model.datasets.klue.utils import convert_examples_to_features


class NERProcessor(BaseDataProcessor):
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
        self.labels_set = set()

    def _is_punctuation(char: str) -> bool:
        cp = ord(char)

        if (
            (cp >= 33 and cp <= 47)
            or (cp >= 58 and cp <= 64)
            or (cp >= 91 and cp <= 96)
            or (cp >= 123 and cp <= 126)
        ):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def _create_examples(self, file_path: str) -> List[TextToLabel]:
        docs = pd.read_csv(file_path)
        examples = []
        for doc_id, doc in docs.values:
            strip_char = "##"
            new_doc = doc
            labels = ["O"] * len(doc)

            tag_pattern = r"<[ㄱ-ㅎ|가-힣|a-z|0-9|\s]+:[A-Z]+>"
            while True:
                match = re.search(tag_pattern, new_doc)
                if match is None:
                    break
                start, end = match.start(), match.end()
                tag = new_doc[start:end][1:-1]
                target, label = tag.split(":")

                new_doc = new_doc[:start] + target + new_doc[end:]
                label = [
                    f"B-{l}" if idx == 0 else f"I-{l}"
                    for idx, l in enumerate([label] * len(target))
                ]
                labels = labels[:start] + label + labels[end:]
                self.labels_set.update(labels)

            assert len(new_doc) == len(labels), "mismatched label exists in document."

            sent_words = new_doc.split(" ")
            labels_without_space = []
            for char, l in zip(new_doc, labels):
                if char != " ":
                    labels_without_space.append(l)
            new_labels = []
            char_idx = 0
            for word in sent_words:
                correct_syllable_num = len(word)
                tokenized_word = self.tokenizer.tokenize(word)
                contain_unk = (
                    True if self.tokenizer.unk_token in tokenized_word else False
                )
                for i, token in enumerate(tokenized_word):
                    token = token.replace(strip_char, "")
                    if not token:
                        new_labels.append("O")
                        continue
                    new_labels.append(labels_without_space[char_idx])
                    if not contain_unk:
                        char_idx += len(token)
                if contain_unk:
                    char_idx += correct_syllable_num

            examples.append(TextToLabel(iid=doc_id, text_a=new_doc, label=new_labels))
        return examples

    def _convert_features(
        self, examples: List[TextToLabel]
    ) -> List[TextToLabelFeature]:
        return convert_examples_to_features(
            examples,
            self.tokenizer,
            label_list=sorted(list(self.labels_set)),
            max_length=self.config.params.max_seq_length,
            task_mode="tagging",
        )

    def _create_dataset(self, file_path: str) -> TensorDataset:
        examples = self._create_examples(file_path)
        features = self._convert_features(examples)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long
        )
        # Some model does not make use of token type ids (e.g. RoBERTa)
        all_token_type_ids = torch.tensor(
            [0 if f.token_type_ids is None else f.token_type_ids for f in features],
            dtype=torch.long,
        )
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        return TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels
        )

    def get_dataset(self, data_dir: str, file_name: str):
        file_path = os.path.join(data_dir, file_name)

        return self._create_dataset(file_path)
