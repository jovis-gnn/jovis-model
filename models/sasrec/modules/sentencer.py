import json
import os
import pickle
import random
import subprocess
from typing import Union

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from transformers import ElectraModel, ElectraPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from models.sasrec.string_matcher import StringMatchingModel


class ProdNameModel(nn.Module):
    model_args = {
        "temp": 0.05,
        "pooler_type": "cls",
    }

    def __init__(self, device):
        super().__init__()

        self.device = device

        current_path = os.path.realpath(__file__)
        self.data_dir = os.path.join(current_path, "..", "..", "dvcdata")
        self.data_dir = os.path.normpath(self.data_dir)

        self.item2fine_ctgr_file = os.path.join(
            self.data_dir,
            "item2category_20230306_entity2label.json",
        )

        self.fine_ctgr2prodname_file = os.path.join(
            self.data_dir,
            "fine_ctgr2prodname.json",
        )
        self.fine_ctgr2prodname = self.load_data_from_dvc(
            file=self.fine_ctgr2prodname_file,
        )

        self.prodname2emb_file = os.path.join(
            self.data_dir,
            "prodname2embeddings.pickle",
        )
        self.prodname2emb = self.load_data_from_dvc(
            file=self.prodname2emb_file,
        )
        self.all_prod_names = list(self.prodname2emb)

        self.model_ckpt_dir = os.path.join(
            self.data_dir,
            "model",
        )
        self.load_data_from_dvc(
            file=self.model_ckpt_dir,
        )

        self.load_model(model_name_or_path=self.model_ckpt_dir)
        self.model = self.model.to(self.device)

        self.string_matching_model = StringMatchingModel()
        self.string_matching_model.load(fp=self.item2fine_ctgr_file)

    def load_model(self, model_name_or_path: str):
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.model = ElectraForCL.from_pretrained(
            model_name_or_path,
            config=self.config,
            **ProdNameModel.model_args,
        )

    def load_data_from_dvc(self, file: str):
        subprocess.run(f"dvc pull {file}.dvc", shell=True)

        if file.endswith(".json"):
            with open(file=file) as f:
                data = json.load(fp=f)
        elif file.endswith(".pickle"):
            with open(file=file, mode="rb") as f:
                data = pickle.load(file=f)
        else:
            return None

        return data

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs

    def __call__(self, product_names: Union[str, list[str]]) -> torch.Tensor:
        tokenizer_outputs = self.tokenizer(
            product_names, return_tensors="pt", padding=True
        )

        inputs = {
            "input_ids": tokenizer_outputs["input_ids"].to(self.device),
            "attention_mask": tokenizer_outputs["attention_mask"].to(self.device),
        }
        model_outputs = self.forward(**inputs)
        cls_embeddings = model_outputs.pooler_output

        final_embs = []

        if isinstance(product_names, list):
            for product_name in product_names:
                fine_ctgr = self.string_matching_model(product_name)

                if fine_ctgr:
                    prod_names = self.fine_ctgr2prodname[fine_ctgr]
                else:
                    prod_names = random.sample(population=self.all_prod_names, k=1000)

                embs = []
                prod_name_embeddings = torch.stack(
                    [
                        torch.tensor(self.prodname2emb[prod_name]["embedding"])
                        for prod_name in prod_names
                    ]
                )
                prod_name_embeddings = prod_name_embeddings.to(self.device)

                sim_scores = self.model.sim(
                    x=cls_embeddings.unsqueeze(1),
                    y=prod_name_embeddings.unsqueeze(0),
                )
                sim_scores = sim_scores.squeeze(1)

                sim_idxs = torch.argmax(sim_scores, dim=1).detach().cpu().tolist()
                for idx in sim_idxs:
                    prod_name = prod_names[idx]
                    emb = torch.tensor(self.prodname2emb[prod_name]["embedding"])
                    embs.append(emb)

            final_embs.extend(embs)
        elif isinstance(product_names, str):
            fine_ctgr = self.string_matching_model(product_names)

            if fine_ctgr:
                prod_names = self.fine_ctgr2prodname[fine_ctgr]
            else:
                prod_names = random.sample(population=self.all_prod_names, k=1000)

            embs = []
            prod_name_embeddings = torch.stack(
                [
                    torch.tensor(self.prodname2emb[prod_name]["embedding"])
                    for prod_name in prod_names
                ]
            )
            prod_name_embeddings = prod_name_embeddings.to(self.device)

            sim_scores = self.model.sim(
                x=cls_embeddings.unsqueeze(1),
                y=prod_name_embeddings.unsqueeze(0),
            )
            sim_scores = sim_scores.squeeze(1)

            sim_idxs = torch.argmax(sim_scores, dim=1).detach().cpu().tolist()
            for idx in sim_idxs:
                prod_name = prod_names[idx]
                emb = torch.tensor(self.prodname2emb[prod_name]["embedding"])
                embs.append(emb)

            final_embs.extend(embs)

        final_embs = torch.stack(final_embs)

        return final_embs


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(
            in_features=config.hidden_size,
            out_features=256,
        )
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls",
            "cls_before_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
        ], f"unrecognized pooling type {self.pooler_type}"

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type in ["cls_before_pooler", "cls"]:
            return last_hidden[:, 0]

        if self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)

        if self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]

            pooled_result = (
                (first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

            return pooled_result

        if self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]

            pooled_result = (
                (last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

            return pooled_result

        raise NotImplementedError


class ElectraForCL(ElectraPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, **model_args):
        super().__init__(config)

        self.model_args = model_args
        self.electra = ElectraModel(config)

        self.pooler_type = self.model_args["pooler_type"]
        self.pooler = Pooler(self.pooler_type)

        if self.pooler_type == "cls":
            self.mlp = MLPLayer(config)

        self.sim = Similarity(temp=self.model_args["temp"])

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooler_output = self.pooler(
            attention_mask=attention_mask,
            outputs=outputs,
        )

        if self.pooler_type == "cls":
            pooler_output = self.mlp(pooler_output)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )
