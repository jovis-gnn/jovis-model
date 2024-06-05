import os
from typing import Union, Dict, List

import torch
from torch.utils.data import TensorDataset
import pandas as pd
from transformers import AutoTokenizer

from jovis_model.config import Config
from jovis_model.datasets.input import DialogToLabel, TextToLabelFeature
from jovis_model.datasets.base import BaseDataProcessor


class ChatFormat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.convert_tokens_to_ids("<|start_header_id|>"))
        tokens.extend(self.tokenizer.encode(message["role"], add_special_tokens=False))
        tokens.append(self.tokenizer.convert_tokens_to_ids("<|end_header_id|>"))
        tokens.extend(self.tokenizer.encode("\n\n", add_special_tokens=False))
        return tokens

    def encode_message(self, message) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), add_special_tokens=False)
        )
        tokens.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        return tokens

    def encode_dialog_prompt(self, dialog: List[Dict]) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.convert_tokens_to_ids("<|begin_of_text|>"))
        for message in dialog:
            tokens.extend(self.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens


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
        if not self.tokenizer.pad_token_id:
            pad_token_id = self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")
            self.tokenizer.pad_token = "<|end_of_text|>"
            self.tokenizer.pad_token_id = pad_token_id
        self.chat_formatter = ChatFormat(self.tokenizer)

    def create_examples(self, file_path: str) -> List[DialogToLabel]:
        examples = []
        assert (
            os.path.basename(file_path).split(".")[-1] == "csv"
        ), "ChatProcessor only support csv file format."
        data = pd.read_csv(file_path)
        data = data.dropna()
        assert set(data.columns) == set(
            ["dialog_id", "seq_id", "role", "content"]
        ), "csv file should have explicit columns : ['dialog_id', 'seq_id', 'role', 'content]"
        data = (
            data.groupby("dialog_id")
            .apply(lambda x: x.to_dict(orient="records"), include_groups=False)
            .to_dict()
        )

        examples = []
        for eid, dialog in data.items():
            dialog = sorted(dialog, key=lambda x: x["seq_id"])
            if not (len(dialog) > 1 and dialog[-1]["role"] == "assistant"):
                continue
            label = dialog[-1]["content"]
            dialog = dialog[:-1]
            examples.append(DialogToLabel(eid=eid, dialog=dialog, label=label))

        return examples

    def convert_features(
        self, examples: Union[List[Dict], List[List[Dict]], List[DialogToLabel]]
    ) -> List[TextToLabelFeature]:
        if isinstance(examples[0], DialogToLabel):
            features = []
            for example in examples:
                dialog_ids = self.chat_formatter.encode_dialog_prompt(example.dialog)
                output_ids = self.tokenizer.encode(
                    example.label, add_special_tokens=False
                ) + [self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                input_ids = dialog_ids + output_ids
                label = [-100] * len(dialog_ids) + output_ids
                attention_mask = torch.tensor(input_ids).ge(0).tolist()

                feature = TextToLabelFeature(
                    input_ids=input_ids, attention_mask=attention_mask, label=label
                )
                features.append(feature)
            return features
        else:
            chat = self.tokenizer.apply_chat_template(examples)
            tokens = torch.tensor(chat).long()
            tokens = tokens.unsqueeze(0)

            return tokens

    def create_dataset(self, file_path: str):
        examples = self.create_examples(file_path)
        features = self.convert_features(examples)

        chunk_size = self.config.params.context_length
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "label": [],
        }
        packed_features = {
            "input_ids": [],
            "attention_mask": [],
            "label": [],
        }
        for feature in features:
            buffer = {k: v + getattr(feature, k) for k, v in buffer.items()}

            while len(next(iter(buffer.values()))) > chunk_size:
                for k, v in buffer.items():
                    packed_features[k].append(buffer[k][:chunk_size])
                buffer = {k: v[chunk_size:] for k, v in buffer.items()}

        all_input_ids = torch.tensor(packed_features["input_ids"], dtype=torch.long)
        all_attention_mask = torch.tensor(
            packed_features["attention_mask"], dtype=torch.long
        )
        all_labels = torch.tensor(packed_features["label"], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
        return dataset

    def get_dataset(self, data_dir: str, file_name: str):
        file_path = os.path.join(data_dir, file_name)

        return self.create_dataset(file_path)
