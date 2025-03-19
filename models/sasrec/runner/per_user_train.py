import os
import boto3
import pickle
import json
import mlflow
import faiss
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.sasrec.runner.initializer import build_model
from models.sasrec.utils.helper import init_logger, get_dataset_from_s3, get_ranker_dataset
from models.sasrec.utils.dataset import BasicDataset
from models.sasrec.utils.scheduler import get_scheduler
from models.sasrec.utils.metric import eval_metric


class Trainer:
    def __init__(self, **config):
        self.logger = init_logger("trainer")
        self.config = config
        self.data_dict = get_dataset_from_s3(
            self.config["etl_s3_bucket_name"],
            "sasrec_dataset/v{}/companyid={}/etlid={}/checkpoint={}".format(
                self.config["etl_version"],
                self.config["company_id"],
                self.config["etl_id"],
                self.config["dataset_checkpoint"],
            ),
        )

        product_conv_dict = self.data_dict["df"][["productid_index", "productid_model_index"]].drop_duplicates().set_index("productid_index").to_dict()["productid_model_index"]
        device_conv_dict = self.data_dict["df"][["deviceid_index", "deviceid_model_index"]].drop_duplicates().set_index("deviceid_index").to_dict()["deviceid_model_index"]

        self.ranker_data_dict = get_ranker_dataset()
        bestseller100 = self.ranker_data_dict[self.ranker_data_dict["strategy_index"] == "1"]["productid_index"].unique()
        self.bestseller100 = np.array([product_conv_dict[idx] for idx in bestseller100])

        visual = self.ranker_data_dict[self.ranker_data_dict["strategy_index"] == "2"].groupby("deviceid_index")["productid_index"].apply(list).to_frame().reset_index()
        visual["deviceid_index"] = visual["deviceid_index"].apply(lambda x: device_conv_dict[x])
        visual["productid_index"] = visual["productid_index"].apply(lambda x: [val for val in [product_conv_dict.get(idx) for idx in x] if val])
        self.visual_dict = visual.set_index('deviceid_index')['productid_index'].to_dict()

        self.save_mapping()
        train_ds = BasicDataset(
            data_dict=self.data_dict, seq_len=self.config["seq_len"], usage="train"
        )
        self.train_dl = DataLoader(
            train_ds,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=4,
        )
        self.test_ds = BasicDataset(
            data_dict=self.data_dict, seq_len=self.config["seq_len"], usage="test"
        )
        self.test_dl = DataLoader(
            self.test_ds,
            batch_size=self.config["batch_size_test"],
            shuffle=False,
            num_workers=4,
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
            num_product=None,
            meta_information=None,
        ).to(self.device)

        if self.config["weight_decay_opt"]:
            self.opt = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

        total_steps = (
            int(len(self.train_dl) / self.config["batch_size"])
            * self.config["train_epoch"]
        )
        warmup_steps = total_steps // 10

        if self.config["scheduler_type"] != "original":
            self.scheduler = get_scheduler(
                self.opt, self.config["scheduler_type"], total_steps, warmup_steps
            )

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
                "sasrec",
                self.config["etl_id"],
                self.config["model_id"],
            )
        ).put(Body=mapping_b)

    def build_faiss_index(self):
        item_embeddings = (
            self.model.item_embedding(torch.arange(1, self.model.num_item).to(self.device))
            .detach()
            .cpu()
            .numpy()
        )

        faiss_index = faiss.IndexFlatIP(item_embeddings.shape[1])
        faiss_index = faiss.IndexIDMap2(faiss_index)
        faiss_index.add_with_ids(item_embeddings, np.arange(1, self.model.num_item))
        faiss.write_index(faiss_index, "/tmp/faiss.index")

        bucket = boto3.resource("s3").Bucket(self.config["model_s3_bucket_name"])
        bucket.Object(
            "{}/{}/{}/{}/faiss.index".format(
                self.config["company_id"],
                "sasrec",
                self.config["etl_id"],
                self.config["model_id"],
            )
        ).put(Body=open("/tmp/faiss.index", "rb"))

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
        )
        self.model.forward = orig_forw_ptr
        bucket = boto3.resource("s3").Bucket(self.config["model_s3_bucket_name"])
        bucket.Object(
            "{}/{}/{}/{}/sasrec.onnx".format(
                self.config["company_id"],
                "sasrec",
                self.config["etl_id"],
                self.config["model_id"],
            )
        ).put(Body=open("/tmp/sasrec.onnx", "rb"))

    def record_loss(self, res, e, train_loss_=None):
        if train_loss_ is not None:
            line1 = "[{} epoch] BPRLoss: {:.4f}".format(e, train_loss_)
            self.logger.info(line1)
        line2 = "[{} epoch] global USER: {}, global HIT: {}, global HR: {:.4f}, global MAP: {:.4f}".format(
            e,
            res["global_NUM"],
            res["global_HIT"],
            res["global_HR"],
            res["global_MAP"],
        )
        self.logger.info(line2)

    def record_loss_mlflow(
        self,
        res,
        e,
        cur_lr,
        train_loss_=None,
        reg_term_=None,
        link_left=None,
        link_right=None,
    ):
        if train_loss_ is not None:
            mlflow.log_metric("bpr loss", train_loss_, e)
        if reg_term_ is not None:
            mlflow.log_metric("reg term", reg_term_, e)
        if link_left is not None:
            mlflow.log_metric("left link", link_left, e)
        if link_right is not None:
            mlflow.log_metric("right link", link_right, e)
        mlflow.log_metric("learning rate", cur_lr, e)

        mlflow.log_metric("recent user", res["recent_NUM"], e)
        mlflow.log_metric("recent hit", res["recent_HIT"], e)
        mlflow.log_metric("recent HR", res["recent_HR"], e)
        mlflow.log_metric("recent MAP", res["recent_MAP"], e)

        mlflow.log_metric("inter user", res["inter_NUM"], e)
        mlflow.log_metric("inter hit", res["inter_HIT"], e)
        mlflow.log_metric("inter HR", res["inter_HR"], e)
        mlflow.log_metric("inter MAP", res["inter_MAP"], e)

        mlflow.log_metric("new user", res["new_NUM"], e)
        mlflow.log_metric("new hit", res["new_HIT"], e)
        mlflow.log_metric("new HR", res["new_HR"], e)
        mlflow.log_metric("new MAP", res["new_MAP"], e)

        mlflow.log_metric("global user", res["global_NUM"], e)
        mlflow.log_metric("global hit", res["global_HIT"], e)
        mlflow.log_metric("global HR", res["global_HR"], e)
        mlflow.log_metric("global MAP", res["global_MAP"], e)

    def whole_process_batch(self, cur_epoch, recent_user_interval=8):
        self.model.eval()

        test_num_common_recs = []
        user_gt_dict = {}
        user_rec_dict = {}
        # for idx, (user, history, history_mask) in enumerate(self.test_dl):
        for idx, (user, history, history_mask) in enumerate(tqdm(self.test_dl)):
            user = user[:, 0].type(torch.LongTensor).to(self.device)
            history = history.type(torch.LongTensor).to(self.device)
            history_mask = history_mask.type(torch.LongTensor).to(self.device)

            # filtered_pool = [np.array(self.visual_dict[per_user]) for per_user in user.cpu().numpy()]
            filtered_pool = [np.unique(np.concatenate([np.array(self.visual_dict[per_user]), self.bestseller100])) for per_user in user.cpu().numpy()]

            filtered_pool_on_device = [torch.LongTensor(per_pool).to(self.device) for per_pool in filtered_pool]

            for bidx in range(user.shape[0]):
                predictions = self.model.recommend(
                    user[bidx:bidx + 1], history[bidx:bidx + 1], history_mask[bidx:bidx + 1], item=filtered_pool_on_device[bidx]
                )
                predictions = predictions.detach()
                _, recommends = torch.topk(predictions, self.config["top_k"])
                recommends = recommends.detach().cpu().numpy()

                recommends = filtered_pool[bidx][recommends][0]

                user_bidx = user[bidx:bidx + 1].cpu().numpy()[0]

                user_gt_dict[user_bidx] = self.data_dict["device_test_dict"][user_bidx]
                user_rec_dict[user_bidx] = recommends.tolist()

            if self.config["overlap"]:
                row2row_comp = np.concatenate(
                    [
                        np.tile(
                            np.expand_dims(recommends, axis=1),
                            (1, recommends.shape[0], 1),
                        ),
                        np.tile(
                            np.expand_dims(recommends, axis=0),
                            (recommends.shape[0], 1, 1),
                        ),
                    ],
                    axis=-1,
                )
                common_recs = np.apply_along_axis(
                    lambda x: self.config["top_k"] * 2 - len(np.unique(x)),
                    axis=-1,
                    arr=row2row_comp,
                )
                num_common_recs = np.sum(np.triu(common_recs, k=1)) / (
                    np.sum(np.triu(np.ones_like(common_recs), k=1)) + 1e-8
                )

                test_num_common_recs.append(num_common_recs)

        if self.config["overlap"]:
            mlflow.log_metric(
                "test_num_common_recs", np.mean(test_num_common_recs), cur_epoch
            )

        res = eval_metric(
            user_gt_dict,
            user_rec_dict,
            self.data_dict["device_type_dict"],
            topk=self.config["top_k"],
            get_result=True,
        )

        return res

    def train(self):
        start_epoch = 0
        best_map = 0
        start_test_res = self.whole_process_batch(cur_epoch=0)
        self.record_loss(start_test_res, "start")
        self.record_loss_mlflow(start_test_res, 0, 0)

        validation_patience, best_perf, cur_perf = 0, 0, 0
        for e in range(start_epoch, start_epoch + self.config["train_epoch"]):
            if validation_patience > self.config["patience"]:
                break
            self.model.train()
            train_batch_loss, reg_batch = [], []
            link_left_batch, link_right_batch = [], []
            # for idx, per_input_batch in enumerate(self.train_dl):
            for idx, per_input_batch in enumerate(tqdm(self.train_dl)):
                per_input_batch = [
                    per_input.type(torch.LongTensor).to(self.device)
                    if len(per_input) > 0
                    else None
                    for per_input in per_input_batch
                ]
                user_pos, history, history_mask, neg, random_neg = per_input_batch
                user, pos = user_pos[:, 0], user_pos[:, 1]

                if sum(history_mask[:, -1]) < user_pos.shape[0]:
                    pass
                self.opt.zero_grad()
                total_loss, loss, link_left, link_right, reg = self.model(
                    user, pos, neg, history, history_mask, random_neg
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["clip_grad_ratio"]
                )
                self.opt.step()

                train_batch_loss.append(loss.item())
                if reg is not None:
                    reg_batch.append(reg.item())
                else:
                    reg_batch.append(0)
                link_left_batch.append(link_left.item())
                link_right_batch.append(link_right.item())

            train_loss_, reg_term_, link_left_loss, link_right_loss = list(
                map(
                    lambda x: np.array(x).mean(),
                    (train_batch_loss, reg_batch, link_left_batch, link_right_batch),
                )
            )
            test_res = self.whole_process_batch(cur_epoch=e + 1)
            cur_lr = [self.opt.param_groups[0]["lr"]]
            cur_perf = test_res["global_MAP"]

            if best_perf >= cur_perf:
                validation_patience += 1
            else:
                validation_patience = 0
                best_perf = cur_perf

            self.record_loss(test_res, e, train_loss_)
            self.record_loss_mlflow(
                test_res,
                e,
                cur_lr[0],
                train_loss_,
                reg_term_,
                link_left_loss,
                link_right_loss,
            )

            if test_res["global_MAP"] > best_map:
                best_map = test_res["global_MAP"]
                # self.save_checkpoint()
                # self.build_faiss_index()

    def run(self):
        with mlflow.start_run(run_name=self.config["run_name"]):
            mlflow.log_params(self.config)

            if len(os.getcwd().split("/")) > 6:
                mlflow.log_artifacts(os.path.dirname(os.getcwd()))

            self.train()


if __name__ == "__main__":
    with open(
        "/home/omnious/workspace/hyunsoochung/prcmd-model/models/sasrec/configs/sasrec_config.json",
        "rb",
    ) as f:
        configs = json.load(f)
    trainer = Trainer(**configs)
    trainer.run()
