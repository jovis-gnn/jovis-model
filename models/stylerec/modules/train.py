import os
import json

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.helper import init_logger
from models.stylerec.modules.model import OutfitTransformer
from models.stylerec.utils.loss import focal_loss, info_nce
from models.stylerec.utils.helper import get_outfit_dataset_from_db
from models.stylerec.modules.dataset import OutfitTransformerDataset


class StyleRecTrainer:
    def __init__(self, **config):
        self.logger = init_logger("trainer")
        self.config = config

        self.logger.info(
            f"Start to initialize for training {self.config['task'].upper()} task"
        )
        self.data_dict = get_outfit_dataset_from_db(self.config)
        train_ds = OutfitTransformerDataset(self.config, self.data_dict, type="train")
        self.train_dl = DataLoader(
            dataset=train_ds,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
        )
        valid_ds = OutfitTransformerDataset(self.config, self.data_dict, type="valid")
        self.valid_dl = DataLoader(
            dataset=valid_ds,
            batch_size=self.config["batch_size_test"],
            num_workers=self.config["num_workers"],
        )

        self.device = (
            torch.device("cuda") if self.config["use_gpu"] else torch.device("cpu")
        )
        self.model = OutfitTransformer(self.config).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    def save_checkpoint(self, e):
        local_save_path = os.path.join(
            "../checkpoints",
            self.config["model_id"],
        )
        os.makedirs(local_save_path, exist_ok=True)
        local_save_path = os.path.join(
            local_save_path, f"{self.config['task']}_checkpoint.pth"
        )
        checkpoint = {
            "epoch": e,
            "state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, local_save_path)

    def load_checkpoint(self, e):
        pass

    def record_loss(self, e: int, res: dict = None, loss: float = None):
        line = "[{} epoch]".format(e + 1)
        if loss:
            line += " loss: {:.4f}".format(loss)
        if res:
            for k, v in res.items():
                line += " {}: {:.4f}".format(k, v)
        self.logger.info(line)

    def iterate(self, e, dataloader, is_train):
        if is_train:
            colour, desc = "blue", f"{self.config['task'].upper()} Training"
        else:
            colour, desc = "green", f"{self.config['task'].upper()} Evaluating"

        pbar = tqdm(
            colour=colour,
            desc=f"{desc} Epoch: {e + 1}",
            total=len(dataloader),
            dynamic_ncols=True,
        )
        batch_loss = []
        batch_labels = []
        batch_preds = []
        for idx, batch in enumerate(dataloader):
            if self.config["task"] == "cp":
                labels = batch["labels"].to(self.device)
                inputs = {k: v.to(self.device) for k, v in batch["inputs"].items()}

                embeddings = self.model.encode(inputs)
                preds = self.model(embeddings, inputs["padding_mask"])
                loss = focal_loss(preds, labels)
                batch_loss.append(loss.item())

                if is_train:
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                batch_labels.append(labels.clone().flatten().detach().cpu().bool())
                batch_preds.append(preds.flatten().detach().cpu())
                is_correct = batch_labels[-1] == (batch_preds[-1] > 0.5)
                acc = torch.sum(is_correct).item() / torch.numel(is_correct)
            else:
                anchor = {
                    k: v.to(self.device) for k, v in batch["anchor_inputs"].items()
                }
                positive = {
                    k: v.to(self.device) for k, v in batch["positive_inputs"].items()
                }
                anchor_embeddings = self.model.encode(anchor)
                anchor_embeddings = self.model(
                    anchor_embeddings, anchor["padding_mask"]
                )
                positive_embeddings = self.model.encode(positive)
                positive_embeddings = self.model(
                    positive_embeddings, positive["padding_mask"]
                )
                loss = info_nce(
                    query=anchor_embeddings, positive_key=positive_embeddings
                )
                batch_loss.append(loss.item())

                if is_train:
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                with torch.no_grad():
                    logits = anchor_embeddings @ positive_embeddings.transpose(-2, -1)
                    labels = torch.arange(
                        len(anchor_embeddings), device=anchor_embeddings.device
                    )
                    batch_labels.append(labels.cpu())
                    batch_preds.append(torch.argmax(logits, dim=-1).cpu())
                    is_correct = batch_labels[-1] == batch_preds[-1]
                    acc = torch.sum(is_correct).item() / torch.numel(is_correct)

            pbar.update(1)
            pbar.set_description(
                f"{desc} Epoch: {e + 1}/{self.config['train_epoch']} loss: {loss.item():.4f} acc: {acc:.4f}"
            )
        pbar.close()

        batch_loss = np.array(batch_loss).mean()
        batch_labels = torch.cat(batch_labels)
        batch_preds = torch.cat(batch_preds)
        if self.config["task"] == "cp":
            is_correct = batch_labels == (batch_preds > 0.5)
        else:
            is_correct = batch_labels == batch_preds
        train_acc = torch.sum(is_correct).item() / torch.numel(is_correct)
        res = {"acc": train_acc}
        self.record_loss(e, res=res, loss=batch_loss)

        return res, batch_loss

    def train(self):
        valid_patience, best_epoch, best_valid_acc = 0, 0, 0
        for e in range(self.config["train_epoch"]):
            self.model.train()
            train_res, train_loss = self.iterate(e, self.train_dl, is_train=True)
            self.model.eval()
            with torch.no_grad():
                valid_res, valid_loss = self.iterate(e, self.valid_dl, is_train=False)

            if valid_res["acc"] < best_valid_acc:
                valid_patience += 1
            else:
                best_epoch = e + 1
                best_valid_acc = valid_res["acc"]
                self.save_checkpoint(e)
                valid_patience = 0

            if valid_patience == self.config["patience"]:
                self.logger.info(f"Early stopping. Best epoch : {best_epoch}")
                break

    def run(self):
        self.train()


if __name__ == "__main__":
    with open("../configs/config.json", "rb") as f:
        configs = json.load(f)
    trainer = StyleRecTrainer(**configs)
    trainer.run()
