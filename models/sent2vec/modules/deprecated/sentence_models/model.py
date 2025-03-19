import json
import logging
import os
import pickle
import random
import subprocess
from typing import Union

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer

from .sent_emb_model import ElectraForCL
from .string_matching_model import StringMatchingModel


class ProdNameModel(nn.Module):
    model_args = {
        "temp": 0.05,
        "pooler_type": "cls",
    }

    def __init__(
        self,
        logger: logging.Logger = None,
        model_data_dir: str = "data",
        mode: str = "embedding",
        k: int = 10,
        use_pooler_output: bool = True,
        item2fine_ctgr_file: str = "",
        fine_ctgr2prodname_file: str = "",
        prodname2emb_file: str = "",
        prodname2prodid_file: str = "",
        model_ckpt_dir: str = "",
    ):
        super().__init__()

        if not logger:
            self.logger = logging.getLogger()
        else:
            self.logger = logger

        self.mode = mode
        self.k = k
        self.use_pooler_output = use_pooler_output

        self.item2fine_ctgr_file = item2fine_ctgr_file
        self.fine_ctgr2prodname_file = fine_ctgr2prodname_file
        self.prodname2emb_file = prodname2emb_file
        self.prodname2prodid_file = prodname2prodid_file
        self.model_ckpt_dir = model_ckpt_dir

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model_data_dir = model_data_dir

        self.item2fine_ctgr_file = os.path.join(
            self.model_data_dir,
            self.item2fine_ctgr_file,
        )
        self.item2fine_ctgr_file = os.path.normpath(self.item2fine_ctgr_file)
        self.logger.info("Loading item2fine_ctgr from %s", self.item2fine_ctgr_file)
        self.item2fine_ctgr = self.load_data_from_dvc(
            file=self.item2fine_ctgr_file,
        )

        self.fine_ctgr2prodname_file = os.path.join(
            self.model_data_dir,
            self.fine_ctgr2prodname_file,
        )
        self.fine_ctgr2prodname_file = os.path.normpath(self.fine_ctgr2prodname_file)
        self.logger.info(
            "Loading fine_ctgr2prodname from %s", self.fine_ctgr2prodname_file
        )
        self.fine_ctgr2prodname = self.load_data_from_dvc(
            file=self.fine_ctgr2prodname_file,
        )

        self.prodname2emb_file = os.path.join(
            self.model_data_dir,
            self.prodname2emb_file,
        )
        self.prodname2emb_file = os.path.normpath(self.prodname2emb_file)
        self.logger.info("Loading prodname2emb from %s", self.prodname2emb_file)
        self.prodname2emb = self.load_data_from_dvc(
            file=self.prodname2emb_file,
        )
        self.all_prod_names = list(self.prodname2emb)

        self.prodname2prodid_file = os.path.join(
            self.model_data_dir,
            self.prodname2prodid_file,
        )
        self.prodname2prodid_file = os.path.normpath(self.prodname2prodid_file)
        self.logger.info("Loading prodname2prodid from %s", self.prodname2prodid_file)
        self.prodname2prodid = self.load_data_from_dvc(
            file=self.prodname2prodid_file,
        )

        self.model_ckpt_dir = os.path.join(
            self.model_data_dir,
            self.model_ckpt_dir,
        )
        self.model_ckpt_dir = os.path.normpath(self.model_ckpt_dir)
        self.logger.info("Loading model_ckpt from %s", self.model_ckpt_dir)
        self.load_data_from_dvc(
            file=self.model_ckpt_dir,
        )

        self.load_model(model_name_or_path=self.model_ckpt_dir)
        self.model = self.model.to(self.device)

        self.string_matching_model = StringMatchingModel()
        self.string_matching_model.load(fp=self.item2fine_ctgr_file)

    def load_model(self, model_name_or_path: str):
        self.logger.info("Loading model(s) from %s", model_name_or_path)

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.model = ElectraForCL.from_pretrained(
            model_name_or_path,
            config=self.config,
            **ProdNameModel.model_args,
        )

    def load_data_from_dvc(self, file: str):
        if not os.path.exists(file):
            self.logger.info("Pulling DVC file from %s", file)
            subprocess.run(f"dvc pull {file}.dvc", shell=True)

        if file.endswith("model"):
            return file

        if file.endswith(".json"):
            with open(file=file) as f:
                data = json.load(fp=f)
        elif file.endswith(".pickle"):
            with open(file=file, mode="rb") as f:
                data = pickle.load(file=f)
        else:
            raise ValueError(f"Invalid file: {file}")

        return data

    def get_similarity(
        self,
        prod_name: str,
    ) -> tuple[torch.Tensor, list[str]]:
        tokenizer_outputs = self.tokenizer(
            prod_name,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            "input_ids": tokenizer_outputs["input_ids"].to(self.device),
            "attention_mask": tokenizer_outputs["attention_mask"].to(self.device),
        }
        model_outputs = self.model(**inputs)

        if self.use_pooler_output:
            cls_embeddings = model_outputs.pooler_output  # [batch_size, 256]
        else:
            cls_embeddings = model_outputs.last_hidden_state[:, 0]  # [batch_size, 768]

        try:
            fine_ctgr = self.string_matching_model(prod_name)
        except KeyError as err:
            self.logger("Got %s. Setting fine_ctgr to None", err)
            fine_ctgr = None

        if fine_ctgr:
            candidate_prod_names = self.fine_ctgr2prodname[fine_ctgr]
        else:
            candidate_prod_names = random.sample(population=self.all_prod_names, k=1000)

        prod_name_embeddings = torch.stack(
            [
                self.prodname2emb[candidate_prod_name].clone().detach()
                for candidate_prod_name in candidate_prod_names
            ]
        )
        prod_name_embeddings = prod_name_embeddings.to(self.device)

        similarity_scores = self.model.sim(
            x=cls_embeddings.unsqueeze(1),
            y=prod_name_embeddings.unsqueeze(0),
        )
        similarity_scores = similarity_scores.squeeze(1)

        return similarity_scores, candidate_prod_names

    def forward(self, prod_names: Union[str, list[str]]) -> torch.Tensor:
        if self.mode == "embedding":
            return self.forward_get_emb(prod_names)
        elif self.mode == "topk":
            return self.forward_get_topk_ids(prod_names)
        elif self.mode == "create_embeddings":
            return self.forward_create_embs(prod_names)
        else:
            return NotImplementedError

    def forward_get_topk_ids(self, prod_names: Union[str, list[str]]) -> torch.Tensor:
        final_pred_ids = []

        if isinstance(prod_names, list):
            for prod_name in prod_names:
                similarity_scores, _ = self.get_similarity(prod_name=prod_name)
                topk_scores, topk_ids = torch.topk(similarity_scores, k=self.k, dim=-1)
                final_pred_ids.append(topk_ids)
        elif isinstance(prod_names, str):
            similarity_scores, _ = self.get_similarity(prod_name=prod_names)
            topk_scores, topk_ids = torch.topk(similarity_scores, k=self.k, dim=-1)
            final_pred_ids.extend(topk_ids)

        final_pred_ids = torch.cat(final_pred_ids)  # [batch_size, self.k]

        return final_pred_ids

    def forward_get_emb(self, prod_names: Union[str, list[str]]) -> torch.Tensor:
        final_embs = []

        if isinstance(prod_names, list):
            for prod_name in prod_names:
                similarity_scores, candidate_prod_names = self.get_similarity(
                    prod_name=prod_name
                )

                similarity_idxs = (
                    torch.argmax(similarity_scores, dim=1).detach().cpu().tolist()
                )
                for idx in similarity_idxs:
                    candidate_prod_name = candidate_prod_names[idx]
                    emb = torch.tensor(self.prodname2emb[candidate_prod_name])
                    final_embs.append(emb)
        elif isinstance(prod_names, str):
            similarity_scores, candidate_prod_names = self.get_similarity(
                prod_name=prod_names
            )

            similarity_idxs = (
                torch.argmax(similarity_scores, dim=1).detach().cpu().tolist()
            )
            for idx in similarity_idxs:
                candidate_prod_name = candidate_prod_names[idx]
                emb = torch.tensor(self.prodname2emb[candidate_prod_name])
                final_embs.append(emb)

        final_embs = torch.stack(final_embs)  # [batch_size, emb_dim]

        return final_embs

    def forward_create_embs(self, prod_names: Union[str, list[str]]) -> torch.Tensor:
        tokenizer_outputs = self.tokenizer(
            prod_names,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            "input_ids": tokenizer_outputs["input_ids"].to(self.device),
            "attention_mask": tokenizer_outputs["attention_mask"].to(self.device),
        }

        model_outputs = self.model(**inputs)
        if self.use_pooler_output:
            cls_embeddings = model_outputs.pooler_output  # [batch_size, 256]
        else:
            cls_embeddings = model_outputs.last_hidden_state[:, 0]  # [batch_size, 768]

        return cls_embeddings

    def __call__(self, prod_names: Union[str, list[str]]) -> torch.Tensor:
        return self.forward(prod_names)
