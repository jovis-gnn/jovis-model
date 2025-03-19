import json
import os
import pickle

import boto3
import faiss
import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from models.helper import init_logger
from models.sent2vec.modules.dataset import Sent2VecData

# from models.metric import eval_metric
from models.sent2vec.modules.model import ElectraForCL
from models.sent2vec.utils.helper import get_filtered_dataset_from_s3


class Sent2VecTrainer:
    def __init__(self, **config):
        self.logger = init_logger("trainer")
        self.config = config
        self.logger.info(json.dumps(config, indent=4))
        self.logger.info("Load dataset & preprocessing")
        self.data_dict = get_filtered_dataset_from_s3(
            bucket_name=self.config["etl_s3_bucket_name"],
            etl_version=self.config["etl_version"],
            company_id=self.config["company_id"],
            etl_id=self.config["etl_id"],
            dataset_checkpoint=self.config["dataset_checkpoint"],
        )
        self.save_mapping()
        embed_ds = Sent2VecData(self.data_dict)
        self.embed_dl = DataLoader(
            embed_ds,
            batch_size=self.config["batch_size_test"],
            shuffle=False,
            num_workers=8,
        )
        self.device = (
            torch.device("cuda:{}".format(self.config["device_num"]))
            if self.config["use_gpu"]
            else torch.device("cpu")
        )

        curr_file = os.path.abspath(__file__)
        curr_dir = os.path.dirname(curr_file)

        model_name_or_path = os.path.join(
            curr_dir,
            "..",
            configs["model_name_or_path"],
        )
        self.model_name_or_path = os.path.normpath(model_name_or_path)

        if not os.path.exists(model_name_or_path):
            model_name_or_path = "monologg/koelectra-base-discriminator"
            self.logger.warning("Loading default model: %s", model_name_or_path)
        else:
            self.logger.info("Loading trained model: %s", model_name_or_path)

        self.model_config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_args = {"temp": 0.05, "pooler_type": "cls"}
        self.model = ElectraForCL.from_pretrained(
            model_name_or_path,
            config=self.model_config,
            **self.model_args,
        )
        self.model = self.model.to(self.device)

        # We initialize these as None but instantiate them in train.
        self.optimizer = None
        self.scheduler = None

        self.num_epochs = self.config["num_epochs"]

        self.logger.info("Starting experiment at %s", self.config["remote_server_uri"])
        self.logger.info("Experiment name: %s", self.config["experiment_name"])

        mlflow.set_tracking_uri(self.config["remote_server_uri"])
        mlflow.set_experiment(self.config["experiment_name"])

    def save_mapping(self):
        bucket = boto3.resource("s3").Bucket(self.config["model_s3_bucket_name"])
        mapping = {
            "product2idx": self.data_dict["product2idx"],
            "idx2product": self.data_dict["idx2product"],
        }
        mapping_b = pickle.dumps(mapping)
        bucket.Object(
            "{}/{}/{}/{}/mapping.pkl".format(
                self.config["company_id"],
                "sent2vec",
                self.config["etl_id"],
                self.config["model_id"],
            )
        ).put(Body=mapping_b)

    def create_optimizer_and_scheduler(
        self,
        optimizer_name: str = "AdamW",
        lr_scheduler: str = "linear",
    ) -> None:
        if optimizer_name == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )
        elif optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            raise NotImplementedError

        if lr_scheduler == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer)
        elif lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["num_epochs"],
            )
        else:
            raise NotImplementedError

    def get_item_embeddings(self):
        self.model.eval()
        embeddings = []
        pids = []
        for productid_model_indices, prod_names in tqdm(self.embed_dl):
            pids += productid_model_indices.cpu().numpy().tolist()
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
            if self.config["use_pooler_output"]:
                item_embeddings = model_outputs.pooler_output  # [batch_size, 256]
            else:
                item_embeddings = model_outputs.last_hidden_state[
                    :, 0
                ]  # [batch_size, 768]
            embeddings.append(item_embeddings.detach().cpu().numpy())
        embeddings = np.vstack(embeddings)
        return pids, embeddings

    def build_faiss_index(self):
        productid_model_indices, item_embeddings = self.get_item_embeddings()
        faiss_index = faiss.IndexFlatIP(item_embeddings.shape[1])
        faiss_index = faiss.IndexIDMap2(faiss_index)
        faiss_index.add_with_ids(item_embeddings, np.array(productid_model_indices))
        faiss.write_index(faiss_index, "/tmp/faiss.index")

        bucket = boto3.resource("s3").Bucket(self.config["model_s3_bucket_name"])
        bucket.Object(
            "{}/{}/{}/{}/faiss.index".format(
                self.config["company_id"],
                "sent2vec",
                self.config["etl_id"],
                self.config["model_id"],
            )
        ).put(Body=open("/tmp/faiss.index", "rb"))

    def save_checkpoint(self):
        pass

    # def record_loss(self, e, res=None, train_loss_=None):
    #     if train_loss_ is not None:
    #         line1 = "[{} epoch] BPRLoss: {:.4f}".format(e, train_loss_)
    #         self.logger.info(line1)
    #     if res is not None:
    #         line2 = "[{} epoch] general USER: {}, general HIT: {}, genearl HR: {:.4f}, general MAP: {:.4f}".format(
    #             e,
    #             res["general_NUM"],
    #             res["general_HIT"],
    #             res["general_HR"],
    #             res["general_MAP"],
    #         )
    #         self.logger.info(line2)

    # def record_loss_mlflow(self, e, train_loss_=None, reg_term_=None):
    #     if train_loss_ is not None:
    #         mlflow.log_metric("bpr loss", train_loss_, e)
    #     if reg_term_ is not None:
    #         mlflow.log_metric("reg term", reg_term_, e)

    # def record_metric_mlflow(self, e, res, cur_lr=None):
    #     if cur_lr is not None:
    #         mlflow.log_metric("learning rate", cur_lr, e)

    #     mlflow.log_metric("general_user", res["general_NUM"], e)
    #     mlflow.log_metric("general_hit", res["general_HIT"], e)
    #     mlflow.log_metric("general_HR", res["general_HR"], e)
    #     mlflow.log_metric("general_MAP", res["general_MAP"], e)

    #     mlflow.log_metric("cold_user", res["cold_NUM"], e)
    #     mlflow.log_metric("cold_hit", res["cold_HIT"], e)
    #     mlflow.log_metric("cold_HR", res["cold_HR"], e)
    #     mlflow.log_metric("cold_MAP", res["cold_MAP"], e)

    def whole_process_batch(self, cur_epoch):
        pass

    def train_one_epoch(
        self,
        best_loss: float = float("inf"),
    ):
        best_model = None
        best_optimizer = None
        best_scheduler = None

        pbar = tqdm(
            iterable=self.embed_dl,
            desc=f"Training ElectraCL model from {self.model_name_or_path}",
            total=len(self.embed_dl),
        )
        for idx, batch in enumerate(pbar):
            self.model.zero_grad()

            product_names = batch[1]

            tokenizer_output = self.tokenizer(
                product_names,
                return_tensors="pt",
                padding=True,
            )

            input_ids = tokenizer_output["input_ids"].unsqueeze(1)
            attention_mask = tokenizer_output["attention_mask"].unsqueeze(1)

            # For training contrastive sentence embedding models (e.g., SimCSE)
            #   we have to pass in the same sentence twice.
            two_sent_input_ids = torch.cat(
                [input_ids, input_ids], dim=1
            )  # [batch_size, 2, seq_len]
            two_sent_attention_mask = torch.cat(
                [attention_mask, attention_mask], dim=1
            )  # [batch_size, 2, seq_len]

            inputs = {
                "input_ids": two_sent_input_ids.to(self.device),
                "attention_mask": two_sent_attention_mask.to(self.device),
                "train": True,
            }

            model_output = self.model(**inputs)

            loss = model_output.loss
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            mlflow.log_metric(
                key="train_loss",
                value=loss.item(),
                step=idx,
            )

            if idx % 100 == 0:
                self.logger.info("Loss: %.6f", loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = self.model.state_dict()
                best_optimizer = self.optimizer.state_dict()
                best_scheduler = self.scheduler.state_dict()

        return (
            best_loss,
            best_model,
            best_optimizer,
            best_scheduler,
        )

    def train(self):
        self.create_optimizer_and_scheduler(
            optimizer_name=self.config["optimizer"],
            lr_scheduler=self.config["lr_scheduler"],
        )

        best_loss = float("inf")
        for epoch in range(self.num_epochs):
            (
                loss,
                best_model,
                best_optimizer,
                best_scheduler,
            ) = self.train_one_epoch(best_loss=best_loss)

            try:
                self.model.load_state_dict(best_model)
                self.optimizer.load_state_dict(best_optimizer)
                self.scheduler.load_state_dict(best_scheduler)
            except TypeError:
                pass

            if loss < best_loss:
                best_loss = loss

            mlflow.log_metric(
                key="epoch_loss",
                value=best_loss,
                step=epoch,
            )

            # self.save_checkpoint()

            # self.whole_process_batch(epoch)


if __name__ == "__main__":
    curr_file = os.path.abspath(__file__)
    curr_dir = os.path.dirname(curr_file)

    config_file = os.path.join(
        curr_dir,
        "..",
        "configs",
        "config.json",
    )

    with open(file=config_file) as f:
        configs = json.load(f)

    trainer = Sent2VecTrainer(**configs)
    trainer.train()

    trainer.build_faiss_index()
