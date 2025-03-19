import os
from typing import Optional, List
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers import AutoTokenizer


class InputProcessor:
    def __init__(self, config):
        self.config = config
        self.image_transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.Resize(
                    size=(self.config["img_size"], self.config["img_size"]),
                    antialias=True,
                ),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config["huggingface_model"])

    def process_image(self, product_ids: List[str]) -> List[torch.Tensor]:
        image_paths = [
            os.path.join(self.config["image_dir"], f"{product_id}.jpg")
            for product_id in product_ids
        ][: self.config["outfit_max_len"]]

        images = [Image.open(image_path) for image_path in image_paths]
        outputs = [self.image_transform(image) for image in images]

        return outputs

    def process_text(self, texts: List[str]) -> List[torch.Tensor]:
        texts = texts[: self.config["outfit_max_len"]]
        outputs = self.tokenizer(
            texts,
            max_length=self.config["text_max_len"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return list(outputs["input_ids"]), list(outputs["attention_mask"])

    def process_inputs(
        self, product_ids: List[str], texts: Optional[List[str]], pad=True
    ):
        inputs = defaultdict(list)
        padding_mask = [torch.BoolTensor([False])] * min(
            len(product_ids), self.config["outfit_max_len"]
        )
        image_arrays = self.process_image(product_ids)
        inputs["padding_mask"] = padding_mask
        inputs["image_arrays"] = image_arrays
        if self.config["use_text"]:
            text_token_ids, attention_masks = self.process_text(texts)
            inputs["text_token_ids"] = text_token_ids
            inputs["attention_masks"] = attention_masks
        if pad:
            for k, v in inputs.items():
                if k == "padding_mask":
                    pad_item = torch.BoolTensor([True])
                else:
                    pad_item = torch.zeros_like(v[-1])
                v_with_pad = v + [pad_item] * (self.config["outfit_max_len"] - len(v))
                inputs[k] = v_with_pad

        for k, v in inputs.items():
            if k == "padding_mask":
                inputs[k] = torch.stack(v)
                inputs[k] = inputs[k].squeeze(-1)
            else:
                inputs[k] = torch.stack(v)

        return inputs


class OutfitTransformerDataset(Dataset):
    def __init__(self, config, data_dict, type):
        self.task = config["task"]
        self.input_processor = InputProcessor(config)
        self.data = []
        for outfit_dict in list(data_dict[f"{type}_outfit2product"].values()):
            self.data.append(
                [
                    outfit_dict["label"],
                    outfit_dict["product_ids"],
                    outfit_dict["texts"],
                ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.task == "cp":
            label, product_ids, texts = self.data[idx]
            inputs = self.input_processor.process_inputs(product_ids, texts)
            res = {"labels": torch.FloatTensor([int(label)]), "inputs": inputs}
        else:
            _, product_ids, texts = self.data[idx]
            product_ids, texts = np.array(product_ids), np.array(texts)
            indices = list(range(len(product_ids)))
            np.random.shuffle(indices)
            product_ids, texts = product_ids[indices].tolist(), texts[indices].tolist()

            anchor_product_ids, anchor_texts = product_ids[1:], texts[1:]
            positive_product_ids, positive_texts = [product_ids[0]], [texts[0]]
            anchor_inputs = self.input_processor.process_inputs(
                anchor_product_ids, anchor_texts
            )
            positive_inputs = self.input_processor.process_inputs(
                positive_product_ids, positive_texts, pad=False
            )
            res = {"anchor_inputs": anchor_inputs, "positive_inputs": positive_inputs}

        return res
