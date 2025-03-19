import os
import boto3
import pickle
import json
import mlflow
import faiss
import torch
import numpy as np
from tqdm import tqdm

from models.sasrec.runner.initializer import build_model
from models.sasrec.utils.helper import (
    init_logger,
    get_graphs_from_s3,
    get_ranker_dataset,
)
from models.sasrec.utils.scheduler import get_scheduler
from models.metric import eval_metric
from torch_geometric.data import Data

import torch_geometric


class Trainer:
    def __init__(self, **config):
        self.logger = init_logger("trainer")
        self.config = config
        self.data_dict = get_graphs_from_s3(
            self.config["etl_s3_bucket_name"],
            "sasrec_dataset/v{}/companyid={}/etlid={}/checkpoint={}".format(
                self.config["etl_version"],
                self.config["company_id"],
                self.config["etl_id"],
                self.config["dataset_checkpoint"],
            ),
        )

        if False:
            product_conv_dict = (
                self.data_dict["df"][["productid_index", "productid_model_index"]]
                .drop_duplicates()
                .set_index("productid_index")
                .to_dict()["productid_model_index"]
            )
            # device_conv_dict = self.data_dict["df"][["deviceid_index", "deviceid_model_index"]].drop_duplicates().set_index("deviceid_index").to_dict()["deviceid_model_index"]

            self.ranker_data_dict = get_ranker_dataset()
            bestseller100 = self.ranker_data_dict[
                self.ranker_data_dict["strategy_index"] == "1"
            ]["productid_index"].unique()
            self.bestseller100 = np.array(
                [product_conv_dict[idx] for idx in bestseller100]
            )

        self.save_mapping()

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

        col_of_int = ['deviceid_model_index', 'productid_model_index', 'session', 'occurence']
        self.train_df_ext, self.test_df_ext = self.data_dict['train_df'][col_of_int].to_numpy(), self.data_dict['test_df'][col_of_int].to_numpy()

        train_data_list, test_data_list = self.convert_dataset(self.train_df_ext), self.convert_dataset(self.test_df_ext)

        self.train_dl = torch_geometric.loader.DataLoader(train_data_list, batch_size=self.config["batch_size"], shuffle=True)
        self.test_dl = torch_geometric.loader.DataLoader(test_data_list, batch_size=self.config["batch_size_test"], shuffle=False)

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
            self.model.item_embedding(
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
            opset_version=14,
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

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_device": self.data_dict["num_device"],
                "num_product": self.data_dict["num_product"],
                "config": self.config,
            },
            "/tmp/sasrec.pth",
        )

        bucket.Object(
            "{}/{}/{}/{}/sasrec.pth".format(
                self.config["company_id"],
                "sasrec",
                self.config["etl_id"],
                self.config["model_id"],
            )
        ).put(Body=open("/tmp/sasrec.pth", "rb"))

    def record_loss(self, res, e, train_loss_=None):
        if train_loss_ is not None:
            line1 = "[{} epoch] BPRLoss: {:.4f}".format(e, train_loss_)
            self.logger.info(line1)
        line2 = "[{} epoch] general USER: {}, general HIT: {}, genearl HR: {:.4f}, general MAP: {:.4f}".format(
            e,
            res["general_NUM"],
            res["general_HIT"],
            res["general_HR"],
            res["general_MAP"],
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

        mlflow.log_metric("general_user", res["general_NUM"], e)
        mlflow.log_metric("general_hit", res["general_HIT"], e)
        mlflow.log_metric("general_HR", res["general_HR"], e)
        mlflow.log_metric("general_MAP", res["general_MAP"], e)

        mlflow.log_metric("cold_user", res["cold_NUM"], e)
        mlflow.log_metric("cold_hit", res["cold_HIT"], e)
        mlflow.log_metric("cold_HR", res["cold_HR"], e)
        mlflow.log_metric("cold_MAP", res["cold_MAP"], e)

    def convert_dataset(self, dataset):
        data_list = []
        for dataset_idx in tqdm(range(len(dataset))):
            cur_user, cur_pos, cur_session, cur_occ = dataset[dataset_idx]

            history = self.data_dict['device_train_dict'][(cur_user, cur_session)][:cur_occ - 1]

            idx, nodes, senders, x = 0, {}, [], []
            for node in history:
                if node not in nodes:
                    nodes[node] = idx
                    x.append([node])
                    idx += 1
                senders.append(nodes[node])
            receivers = senders[:]
            del senders[-1]    # the last item is a receiver
            del receivers[0]    # the first item is a sender
            edge_index = torch.LongTensor([senders, receivers]).to(self.device)
            x, y = torch.LongTensor(x).to(self.device), torch.LongTensor([cur_pos]).to(self.device)
            # user, session = torch.LongTensor([cur_user]).to(self.device), torch.LongTensor([cur_session]).to(self.device)
            user, session = torch.tensor([cur_user], dtype=torch.long), torch.tensor([cur_session], dtype=torch.long)

            data_list.append(Data(x=x, edge_index=edge_index, y=y, user=user, session=session))

        return data_list

    def whole_process_batch(self, cur_epoch, recent_user_interval=8):
        self.model.eval()
        filtered_pool_on_device = None
        if self.config["filter_item_pool"]:
            # popularity = np.array(
            #     [
            #         len(self.data_dict["product_train_dict"].get(x, []))
            #         for x in np.arange(self.data_dict["num_product"])
            #     ]
            # )
            # popularity_idx = np.array(
            #     sorted(range(len(popularity)), key=lambda k: popularity[k])
            # )

            # filtered_pool = self.data_dict["train_df"][self.data_dict['train_df']["product_count"] > 5]["productid_model_index"].unique()
            # filtered_pool = self.bestseller100

            filtered_pool = self.data_dict['train_df']['productid_model_index'].unique()
            # filtered_pool = popularity_idx[popularity[np.isin(popularity, np.arange(5))].shape[0]:]

            filtered_pool_on_device = torch.LongTensor(filtered_pool).to(self.device)

        test_num_common_recs = []
        user_gt_dict = {}
        user_rec_dict = {}
        for idx, batch in enumerate(tqdm(self.test_dl)):
            user, session = batch.user.to(self.device), batch.session.to(self.device)
            predictions = self.model.recommend(user, session, batch, item=filtered_pool_on_device)
            predictions = predictions.detach()
            # predictions = predictions.detach() + purchase_mask.type(predictions.dtype).to(self.device)
            _, recommends = torch.topk(predictions, self.config["top_k"])
            recommends = recommends.detach().cpu().numpy()

            if self.config["filter_item_pool"]:
                recommends = filtered_pool[recommends]

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

            user_np, session_np = user.cpu().numpy(), session.cpu().numpy()
            for idx, per_user_session in enumerate(zip(user_np, session_np)):
                user_gt_dict[per_user_session] = self.data_dict["device_test_dict"][per_user_session]
                user_rec_dict[per_user_session] = recommends[idx].tolist()

        if self.config["overlap"]:
            mlflow.log_metric(
                "test_num_common_recs", np.mean(test_num_common_recs), cur_epoch
            )

        res = eval_metric(
            user_gt_dict,
            user_rec_dict,
            user_train_dict=self.data_dict["device_train_dict"],
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
            for idx, batch in enumerate(tqdm(self.train_dl)):
                user = batch.user.to(self.device)
                pos, pos_np = batch.y, batch.y.cpu().numpy()

                neg, random_neg = [], []
                for rand_idx in range(len(batch.y)):
                    neg_idx = np.random.randint(self.data_dict['train_df'].shape[0])
                    while self.data_dict['train_df'].iloc[neg_idx]['productid_model_index'] == pos_np[rand_idx]:
                        neg_idx = np.random.randint(self.data_dict['train_df'].shape[0])
                    per_neg = self.data_dict['train_df'].iloc[neg_idx]['productid_model_index']

                    per_random_neg = np.random.randint(self.data_dict['num_product'])
                    while per_random_neg == neg or per_random_neg == pos_np[rand_idx]:
                        per_random_neg = np.random.randint(self.data_dict['num_product'])

                    _ = list(map(lambda arr, x : arr.append(x), (neg, random_neg), (per_neg, per_random_neg)))

                neg, random_neg = torch.LongTensor(neg).to(self.device), torch.LongTensor(random_neg).to(self.device)

                self.opt.zero_grad()
                total_loss, loss, link_left, link_right, reg = self.model(
                    user, pos, neg, batch, random_neg
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["clip_grad_ratio"]
                )
                self.opt.step()

                if self.config["scheduler_type"] != "original":
                    self.scheduler.step()

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
            cur_perf = test_res["general_MAP"]

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

            if test_res["general_MAP"] > best_map:
                best_map = test_res["general_MAP"]
                # self.save_checkpoint()
                # self.build_faiss_index()

    def local_process_batch(self):
        self.model.eval()

        user_gt_dict, user_rec_dict = {}, {}
        for idx, (user, history, history_mask) in enumerate(tqdm(self.test_dl)):
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
                mlflow.log_artifacts(os.path.dirname(os.getcwd()))

            self.train()


if __name__ == "__main__":
    with open(
        "/home/omnious/workspace/hyunsoochung/prcmd-model/models/sasrec/configs/srgnn_config.json",
        "rb",
    ) as f:
        configs = json.load(f)
    trainer = Trainer(**configs)
    trainer.run()
