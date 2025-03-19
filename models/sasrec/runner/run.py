import os
import boto3
import pickle
import json
import mlflow
import faiss
import torch
import numpy as np
from torch.utils.data import DataLoader

from models.sasrec.runner.initializer import build_model
from models.sasrec.utils.helper import init_logger, get_filtered_dataset_from_s3
from models.sasrec.utils.dataset import UnifiedDataset
from models.sasrec.utils.scheduler import get_scheduler
from models.metric import eval_metric

# import pdb


class Trainer:
    def __init__(self, **config):
        self.logger = init_logger("trainer")
        self.config = config
        self.data_dict = get_filtered_dataset_from_s3(
            self.config["etl_s3_bucket_name"],
            "common_dataset/v{}/companyid={}/etlid={}/checkpoint={}".format(
                self.config["etl_version"],
                self.config["company_id"],
                self.config["etl_id"],
                self.config["dataset_checkpoint"],
            ),
        )
        self.save_mapping()
        train_ds = UnifiedDataset(
            data_dict=self.data_dict, seq_len=self.config["seq_len"], usage="train"
        )
        self.train_dl = DataLoader(
            train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=0,
        )
        test_ds = UnifiedDataset(
            data_dict=self.data_dict, seq_len=self.config["seq_len"], usage="test"
        )
        self.test_dl = DataLoader(
            test_ds,
            batch_size=self.config["batch_size_test"],
            shuffle=False,
            num_workers=0,
        )
        self.device = (
            torch.device("cuda:{}".format(self.config["device_num"]))
            if self.config["use_gpu"]
            else torch.device("cpu")
        )
        self.model = build_model(self.config["model_type"])(
            self.config,
            num_user=self.data_dict["num_device"],
            num_item=self.data_dict["num_product"],
            device=self.device,
            num_meta=self.data_dict["num_meta"],
            meta_information=self.data_dict["meta_table"],
        ).to(self.device)

        if self.config["weight_decay_opt"]:
            self.opt = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.config["lr"])

        total_steps = int(len(self.train_dl))
        warmup_steps = total_steps // 10

        if self.config["scheduler_type"] != "original":
            self.scheduler = get_scheduler(
                self.opt, self.config["scheduler_type"], total_steps, warmup_steps
            )

        mlflow.set_tracking_uri(self.config["remote_server_uri"])
        mlflow.set_experiment(self.config["experiment_name"])

    def save_mapping(self):
        session = boto3.Session()
        client = session.client("s3")
        mapping = {
            "product2idx": self.data_dict["product2idx"],
            "idx2product": self.data_dict["idx2product"],
        }
        mapping_b = pickle.dumps(mapping)
        client.put_object(
            Body=mapping_b,
            Bucket=self.config["model_s3_bucket_name"],
            Key="{}/{}/{}/{}/mapping.pkl".format(
                self.config["company_id"],
                "sasrec",
                self.config["etl_id"],
                self.config["model_id"],
            ),
        )

    def build_faiss_index(self):
        item_embeddings = (
            self.model.get_item_embedding(
                torch.arange(1, self.model.num_item).to(self.device)
            )
            .detach()
            .cpu()
            .numpy()
        )

        faiss_index = faiss.IndexFlatIP(item_embeddings.shape[1])
        faiss_index = faiss.IndexIDMap2(faiss_index)
        faiss_index.add_with_ids(item_embeddings, np.arange(1, self.model.num_item))
        faiss.write_index(faiss_index, "/tmp/faiss.index")

        session = boto3.Session()
        client = session.client("s3")
        client.put_object(
            Body=open("/tmp/faiss.index", "rb"),
            Bucket=self.config["model_s3_bucket_name"],
            Key="{}/{}/{}/{}/faiss.index".format(
                self.config["company_id"],
                "sasrec",
                self.config["etl_id"],
                self.config["model_id"],
            ),
        )

    def save_checkpoint(self):
        _history, _history_mask = [
            torch.zeros(1, self.config["seq_len"]).type(torch.int64).to(self.device)
            for _ in range(2)
        ]
        orig_forw_ptr = self.model.forward
        self.model.forward = self.model.get_history_embedding
        torch.onnx.export(
            self.model,
            (_history, _history_mask),
            "/tmp/sasrec.onnx",
            verbose=False,
            input_names=["history", "history_mask"],
            output_names=["history_embedding"],
            opset_version=14,
        )
        self.model.forward = orig_forw_ptr

        session = boto3.Session()
        client = session.client("s3")
        client.put_object(
            Body=open("/tmp/sasrec.onnx", "rb"),
            Bucket=self.config["model_s3_bucket_name"],
            Key="{}/{}/{}/{}/sasrec.onnx".format(
                self.config["company_id"],
                "sasrec",
                self.config["etl_id"],
                self.config["model_id"],
            ),
        )

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_device": self.data_dict["num_device"],
                "num_product": self.data_dict["num_product"],
                "config": self.config,
            },
            "/tmp/sasrec.pth",
        )
        client.put_object(
            Body=open("/tmp/sasrec.pth", "rb"),
            Bucket=self.config["model_s3_bucket_name"],
            Key="{}/{}/{}/{}/sasrec.pth".format(
                self.config["company_id"],
                "sasrec",
                self.config["etl_id"],
                self.config["model_id"],
            ),
        )

    def record_loss(self, e, res=None, train_loss_=None):
        if train_loss_ is not None:
            line1 = "[{} epoch] BPRLoss: {:.4f}".format(e, train_loss_)
            self.logger.info(line1)
        if res is not None:
            line2 = "[{} epoch] general USER: {}, general HIT: {}, genearl HR: {:.4f}, general MAP: {:.4f}".format(
                e,
                res["general_NUM"],
                res["general_HIT"],
                res["general_HR"],
                res["general_MAP"],
            )
            self.logger.info(line2)

    def record_loss_mlflow(self, e, train_loss_=None, reg_term_=None):
        if train_loss_ is not None:
            mlflow.log_metric("bpr loss", train_loss_, e)
        if reg_term_ is not None:
            mlflow.log_metric("reg term", reg_term_, e)

    def record_metric_mlflow(self, e, res, cur_lr=None):
        if cur_lr is not None:
            mlflow.log_metric("learning rate", cur_lr, e)

        mlflow.log_metric("general_user", res["general_NUM"], e)
        mlflow.log_metric("general_hit", res["general_HIT"], e)
        mlflow.log_metric("general_HR", res["general_HR"], e)
        mlflow.log_metric("general_MAP", res["general_MAP"], e)

        mlflow.log_metric("cold_user", res["cold_NUM"], e)
        mlflow.log_metric("cold_hit", res["cold_HIT"], e)
        mlflow.log_metric("cold_HR", res["cold_HR"], e)
        mlflow.log_metric("cold_MAP", res["cold_MAP"], e)

    def whole_process_batch(self, cur_epoch):
        self.model.eval()
        filtered_pool_on_device = None
        if self.config["filter_item_pool"]:
            filtered_pool = self.data_dict["train_df"]["productid_model_index"].unique()

            filtered_pool_on_device = torch.LongTensor(filtered_pool).to(self.device)

        user_gt_dict, user_rec_dict = {}, {}
        for outer_idx, (
            user_pos,
            history,
            history_mask,
            occ,
            purchase_mask,
        ) in enumerate(self.test_dl):
            user = user_pos[:, 0].type(torch.LongTensor).to(self.device)
            history = history.type(torch.LongTensor).to(self.device)
            history_mask = history_mask.type(torch.LongTensor).to(self.device)
            predictions = self.model.recommend(
                user, history, history_mask, item=filtered_pool_on_device
            )
            # predictions = predictions.detach()
            predictions = predictions.detach() + purchase_mask.type(
                predictions.dtype
            ).to(self.device)
            _, recommends = torch.topk(predictions, self.config["top_k"])
            recommends = recommends.detach().cpu().numpy()

            if self.config["filter_item_pool"]:
                recommends = filtered_pool[recommends]

            user_occ = list(zip(user.cpu().numpy().astype(np.int32), occ.numpy()))
            pos = user_pos[:, 1].cpu().numpy().astype(np.int32)
            for inner_idx, per_user_occ in enumerate(user_occ):
                user_gt_dict[per_user_occ] = [pos[inner_idx]]
                user_rec_dict[per_user_occ] = recommends[inner_idx].tolist()

        res = eval_metric(
            user_gt_dict,
            user_rec_dict,
            user_train_dict=self.data_dict["device_test_dict"],
            topk=self.config["top_k"],
            get_result=True,
        )

        return res

    def train(self, eval_only_last=True, save_every_epoch=True):
        start_epoch = 0

        if not eval_only_last:
            start_test_res = self.whole_process_batch(cur_epoch=0)
            self.record_loss(start_test_res, "start")
            self.record_loss_mlflow(start_test_res, 0, 0)

        for e in range(start_epoch, start_epoch + self.config["train_epoch"]):
            self.model.train()
            orig_loss_batch, reg_batch = [], []
            for idx, per_input_batch in enumerate(self.train_dl):
                per_input_batch = [
                    (
                        per_input.type(torch.LongTensor).to(self.device)
                        if len(per_input) > 0
                        else None
                    )
                    for per_input in per_input_batch
                ]
                user_pos, history, history_mask, neg, random_neg = per_input_batch
                user, pos = user_pos[:, 0], user_pos[:, -1]

                self.opt.zero_grad()
                total_loss, orig_loss, reg = self.model(
                    user, pos, neg, history, history_mask, random_neg
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["clip_grad_ratio"]
                )
                self.opt.step()

                if self.config["scheduler_type"] != "original":
                    self.scheduler.step()

                orig_loss_batch.append(orig_loss.item())
                if reg is not None:
                    reg_batch.append(reg.item())
                else:
                    reg_batch.append(0)

            train_loss_, reg_term_ = list(
                map(lambda x: np.array(x).mean(), (orig_loss_batch, reg_batch))
            )

            if not eval_only_last or e == start_epoch + self.config["train_epoch"] - 1:
                test_res = self.whole_process_batch(cur_epoch=e + 1)
                cur_lr = [self.opt.param_groups[0]["lr"]]

                self.record_loss(e, test_res, train_loss_)
                self.record_metric_mlflow(e, test_res, cur_lr[0])
                self.record_loss_mlflow(e, train_loss_, reg_term_)

            if save_every_epoch:
                self.record_loss(e, train_loss_=train_loss_)
                self.record_loss_mlflow(e, train_loss_, reg_term_)

                self.save_checkpoint()
                self.build_faiss_index()

    def local_process_batch(self):
        self.model.eval()

        user_gt_dict, user_rec_dict = {}, {}
        for idx, (user, history, history_mask) in enumerate(self.test_dl):
            history = history.type(torch.LongTensor).to(self.device)
            history_mask = history_mask.type(torch.LongTensor).to(self.device)
            history_embeddings = self.model.get_history_embedding(history, history_mask)
            history_embeddings = history_embeddings.detach().cpu().numpy()

            item_embeddings = (
                self.model.item_embedding.weight.detach().cpu().numpy()[1:, :]
            )

            faiss_index = faiss.IndexFlatIP(item_embeddings.shape[1])
            faiss_index = faiss.IndexIDMap2(faiss_index)
            faiss_index.add_with_ids(item_embeddings, np.arange(1, self.model.num_item))

            _, recommends = faiss_index.search(history_embeddings, self.config["top_k"])

            users = user[:, 0].type(torch.LongTensor).numpy()
            for idx, per_user in enumerate(users):
                user_gt_dict[per_user] = self.data_dict["device_test_dict"][per_user]
                user_rec_dict[per_user] = recommends[idx].tolist()

        res = eval_metric(
            user_gt_dict,
            user_rec_dict,
            user_train_dict=self.data_dict["device_train_dict"],
            topk=self.config["top_k"],
            get_result=True,
        )

        return res

    def run(self):
        with mlflow.start_run(run_name=self.config["run_name"]):
            mlflow.log_params(self.config)

            if len(os.getcwd().split("/")) > 6:
                dir_of_interest = ["modules", "runner"]

                _ = [
                    mlflow.log_artifacts(
                        os.path.join(os.path.dirname(os.getcwd()), cur_dir)
                    )
                    for cur_dir in dir_of_interest
                ]
                # mlflow.log_artifacts(os.getcwd())

            self.train()


if __name__ == "__main__":
    with open(
        "/home/omnious/workspace/hyunsoochung/prcmd-model/models/sasrec/configs/nova_config.json",
        "rb",
    ) as f:
        configs = json.load(f)
    trainer = Trainer(**configs)
    trainer.run()
