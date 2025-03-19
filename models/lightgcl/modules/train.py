import json
import pickle

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.helper import BotoSession, init_logger
from models.lightgcl.modules.dataset import LightGCLData, LightGCLTest, NegativeSampler
from models.lightgcl.modules.model import LightGCL
from models.lightgcl.utils.helper import get_filtered_dataset_from_s3
from models.metric import eval_metric


class LightGCLTrainer:
    def __init__(self, **config):
        self.logger = init_logger("trainer")
        self.config = config
        self.session = BotoSession().refreshable_session()
        self.logger.info(json.dumps(config, indent=4))
        self.logger.info(self.config["memo"])
        self.logger.info("Load dataset & preprocessing")
        self.data_dict = get_filtered_dataset_from_s3(
            bucket_name=self.config["etl_s3_bucket_name"],
            etl_version=self.config["etl_version"],
            company_id=self.config["company_id"],
            etl_id=self.config["etl_id"],
            dataset_checkpoint=self.config["dataset_checkpoint"],
        )
        self.save_mapping()
        train_ds = LightGCLData(self.data_dict)
        test_ds = LightGCLTest(self.data_dict)
        ng_sampler = NegativeSampler(self.data_dict)
        self.train_dl = DataLoader(
            train_ds,
            collate_fn=ng_sampler.sampling,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=8,
        )
        self.test_dl = DataLoader(
            test_ds,
            batch_size=self.config["batch_size_test"],
            shuffle=False,
            num_workers=8,
        )
        self.device = (
            torch.device("cuda:{}".format(self.config["device_num"]))
            if self.config["use_gpu"]
            else torch.device("cpu")
        )

        self.logger.info("Load graph matrix")
        device_product_norm, product_device_norm = train_ds.get_graph_matrix()
        device_product_norm = device_product_norm.coalesce().to(self.device)
        product_device_norm = product_device_norm.coalesce().to(self.device)

        self.logger.info("Load Model")
        self.model = LightGCL(
            self.config,
            num_device=self.data_dict["num_device"],
            num_product=self.data_dict["num_product"],
            adj_matrix=device_product_norm,
        ).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

        mlflow.set_tracking_uri(self.config["remote_server_uri"])
        mlflow.set_experiment(self.config["experiment_name"])

    def save_mapping(self):
        client = self.session.client("s3")
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
                "lightgcl",
                self.config["etl_id"],
                self.config["model_id"],
            ),
        )

    def save_checkpoint(self, e):
        client = self.session.client("s3")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_device": self.data_dict["num_device"],
                "num_product": self.data_dict["num_product"],
                "config": self.config,
            },
            "/tmp/lightgcl.pth",
        )
        client.put_object(
            Body=open("/tmp/lightgcl.pth", "rb"),
            Bucket=self.config["model_s3_bucket_name"],
            Key="{}/{}/{}/{}/lightgcl_{}.pth".format(
                self.config["company_id"],
                "lightgcl",
                self.config["etl_id"],
                self.config["model_id"],
                e,
            ),
        )

    def record_loss(self, e, res=None, train_loss_=None):
        if train_loss_ is not None:
            line1 = "[{} epoch] BPRLoss: {:.4f}".format(e, train_loss_)
            self.logger.info(line1)
        if res is not None:
            line2 = "[{} epoch] general USER: {}, general HIT: {}, general HR: {:.4f}, general MAP: {:.4f}".format(
                e,
                res["general_NUM"],
                res["general_HIT"],
                res["general_HR"],
                res["general_MAP"],
            )
            self.logger.info(line2)

    def record_loss_mlflow(
        self,
        e: int,
        train_loss_: float = None,
        reg_term_: float = None,
        log_name: str = "bpr loss",
    ):
        if train_loss_ is not None:
            mlflow.log_metric(log_name, train_loss_, e)

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

    def whole_process_batch(self):
        self.model.eval()
        user_gt_dict, user_rec_dict = {}, {}
        for u, mask in tqdm(self.test_dl):
            w_cold = u[:, 0].detach().cpu().numpy()
            wo_cold = []
            for u_ in w_cold:
                if self.data_dict["device_train_dict"].get(u_, -1) == -1:
                    wo_cold.append(0)
                else:
                    wo_cold.append(int(u_))
            wo_cold = torch.LongTensor(wo_cold).to(self.device)
            mask = mask.type(torch.FloatTensor).to(self.device)

            predictions = self.model.recommend(wo_cold)
            predictions = predictions + mask
            scores, recommends = torch.topk(predictions, self.config["top_k"])
            recommends = recommends.detach().cpu().numpy()

            for idx, u_ in enumerate(w_cold):
                user_gt_dict[u_] = self.data_dict["device_test_dict"][u_]
                user_rec_dict[u_] = recommends[idx]

        res = eval_metric(
            user_gt_dict,
            user_rec_dict,
            self.data_dict["device_train_dict"],
            topk=self.config["top_k"],
        )

        return res

    def train(self):
        start_epoch = 0
        best_map = 0

        start_test_res = self.whole_process_batch()
        self.record_loss("start", res=start_test_res)
        self.record_loss_mlflow(0, 0)
        self.record_metric_mlflow(0, start_test_res)
        self.save_checkpoint("start")

        global_step = 0

        validation_patience, best_perf, cur_perf = 0, 0, 0
        for e in range(start_epoch, start_epoch + self.config["train_epoch"]):
            if validation_patience > self.config["patience"]:
                break
            self.model.train()

            train_batch_loss = []
            for pos_neg_pair in tqdm(self.train_dl):
                user, pos, neg = (
                    pos_neg_pair[:, 0],
                    pos_neg_pair[:, 1],
                    pos_neg_pair[:, -1],
                )
                user = torch.LongTensor(user).to(self.device)
                pos = torch.LongTensor(pos).to(self.device)
                neg = torch.LongTensor(neg).to(self.device)

                inputs = {
                    "user": user,
                    "item": pos,
                    "pos": pos,
                    "neg": neg,
                }

                self.model.zero_grad()

                (
                    total_loss,
                    contrastive_loss,
                    bpr_loss,
                    regularization_loss,
                ) = self.model(**inputs)

                total_loss.backward()
                self.opt.step()

                train_batch_loss.append(total_loss.item())

                mlflow.log_metrics(
                    {
                        "train_total_loss": total_loss.item(),
                        "train_contrastive_loss": contrastive_loss.item(),
                        "train_bpr_loss": bpr_loss.item(),
                        "train_regularization_loss": regularization_loss.item(),
                    },
                    step=global_step,
                )

                global_step += 1

            train_loss_ = np.array(train_batch_loss).mean()
            test_res = self.whole_process_batch()

            cur_perf = test_res["general_MAP"]
            if best_perf >= cur_perf:
                validation_patience += 1
            else:
                validation_patience = 0
                best_perf = cur_perf
                self.save_checkpoint(e)

            self.record_loss(e, test_res, train_loss_)
            self.record_loss_mlflow(e, train_loss_, log_name="epoch_train_loss")
            self.record_metric_mlflow(e, test_res)

            if test_res["general_MAP"] > best_map:
                best_map = test_res["general_MAP"]

    def run(self):
        with mlflow.start_run(run_name=self.config["run_name"]):
            mlflow.log_params(self.config)
            self.train()


if __name__ == "__main__":
    with open("../configs/config.json", "rb") as f:
        configs = json.load(f)
    trainer = LightGCLTrainer(**configs)
    trainer.run()
