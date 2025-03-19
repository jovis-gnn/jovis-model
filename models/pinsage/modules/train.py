import os
import json
import boto3
import pickle
from os.path import join, dirname


import faiss
import torch
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from models.helper import init_logger
from models.metric import eval_metric
from models.pinsage.utils.helper import get_filtered_dataset_from_s3
from models.pinsage.modules.model import PinSAGE
from models.pinsage.modules.dataset import (
    PinSAGEData,
    PinSAGETestData,
    NeighborSampler,
    PinSAGECollator,
)


class PinSAGETrainer:
    def __init__(self, **config):
        self.logger = init_logger("trainer")
        self.config = config

        if not self.config.get("aws_access_key_id", None) or not self.config.get(
            "aws_secret_access_key", None
        ):
            dotenv_path = join(dirname(dirname(dirname(dirname(__file__)))), ".env")
            load_dotenv(dotenv_path)
            self.config["aws_access_key_id"] = os.environ["AWS_ACCESS_KEY_ID"]
            self.config["aws_secret_access_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]

        self.skip_test = self.config.get("skip_test", True)
        self.data_dict = get_filtered_dataset_from_s3(self.config)

        self.save_mapping()

        train_ds = PinSAGEData(self.data_dict, self.config["batch_size"])
        neighbor_sampler = NeighborSampler(
            graph=train_ds.graph,
            random_walk_length=self.config["random_walk_length"],
            random_walk_restart_prob=self.config["random_walk_restart_prob"],
            num_random_walks=self.config["num_random_walks"],
            num_neighbors=self.config["num_neighbors"],
            num_layers=self.config["num_layers"],
        )
        collator = PinSAGECollator(graph=train_ds.graph, sampler=neighbor_sampler)
        self.train_dl = DataLoader(
            train_ds, collate_fn=collator.collate_train, num_workers=8
        )
        self.train_dl = iter(self.train_dl)
        self.embed_dl = DataLoader(
            torch.arange(train_ds.graph.number_of_nodes("productid")),
            batch_size=self.config["batch_size_test"],
            collate_fn=collator.collate_test,
            num_workers=8,
        )
        test_ds = PinSAGETestData(data_dict=self.data_dict)
        self.test_dl = DataLoader(
            test_ds,
            batch_size=self.config["batch_size_test"],
            shuffle=False,
            num_workers=8,
        )
        self.device = (
            torch.device("cuda") if self.config["use_gpu"] else torch.device("cpu")
        )
        self.model = PinSAGE(self.config, train_ds.graph).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    def save_mapping(self):
        session = boto3.Session()
        client = session.client(
            "s3",
            aws_access_key_id=self.config["aws_access_key_id"],
            aws_secret_access_key=self.config["aws_secret_access_key"],
        )
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
                "pinsage",
                self.config["etl_id"],
                self.config["model_id"],
            ),
        )

    def build_item_matrix(self):
        self.model.eval()
        if not self.skip_test:
            pbar = tqdm(
                colour="green",
                desc="Build item embeddings",
                total=len(self.embed_dl),
                dynamic_ncols=True,
            )
        with torch.no_grad():
            h_item_batches = []
            for blocks in self.embed_dl:
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(self.device)
                h_item_batches.append(self.model.get_embedding(blocks))
                if not self.skip_test:
                    pbar.update(1)
            if not self.skip_test:
                pbar.close()
            h_item = torch.cat(h_item_batches, 0)
        item_embeddings = h_item.detach().cpu().numpy()
        if not self.skip_test:
            h_item = h_item.matmul(h_item.T).detach().cpu().numpy()
            torch.cuda.empty_cache()
        else:
            h_item = None
        return item_embeddings, h_item

    def build_faiss_index(self, item_embeddings):
        faiss_index = faiss.IndexFlatIP(item_embeddings.shape[1])
        faiss_index = faiss.IndexIDMap2(faiss_index)
        faiss_index.add_with_ids(item_embeddings, np.arange(len(item_embeddings)))
        faiss.write_index(faiss_index, f"/tmp/{self.config['model_id']}.index")

        session = boto3.Session()
        client = session.client(
            "s3",
            aws_access_key_id=self.config["aws_access_key_id"],
            aws_secret_access_key=self.config["aws_secret_access_key"],
        )
        client.put_object(
            Body=open(f"/tmp/{self.config['model_id']}.index", "rb"),
            Bucket=self.config["model_s3_bucket_name"],
            Key="{}/{}/{}/{}/faiss.index".format(
                self.config["company_id"],
                "pinsage",
                self.config["etl_id"],
                self.config["model_id"],
            ),
        )

    def save_checkpoint(self, e):
        session = boto3.Session()
        client = session.client(
            "s3",
            aws_access_key_id=self.config["aws_access_key_id"],
            aws_secret_access_key=self.config["aws_secret_access_key"],
        )
        torch.save(
            self.model.state_dict(),
            f"/tmp/{self.config['model_id']}.pth",
        )
        client.put_object(
            Body=open(f"/tmp/{self.config['model_id']}.pth", "rb"),
            Bucket=self.config["model_s3_bucket_name"],
            Key="{}/{}/{}/{}/pinsage.pth".format(
                self.config["company_id"],
                "pinsage",
                self.config["etl_id"],
                self.config["model_id"],
            ),
        )

    def record_loss(self, e: int, res: dict = None, train_loss_: float = None):
        if train_loss_:
            line1 = "[{} epoch] BPRLoss: {:.4f}".format(e, train_loss_)
            self.logger.info(line1)
        if res:
            for k, v in res.items():
                line2 = (
                    "[{} epoch][{}] USER: {}, HIT: {}, HR: {:.4f}, MAP: {:.4f}".format(
                        e,
                        k,
                        v["general_NUM"],
                        v["general_HIT"],
                        v["general_HR"],
                        v["general_MAP"],
                    )
                )
                self.logger.info(line2)

    def recommend(self, e, item_sim_matrix):
        user_gt_dict_gt1, user_rec_dict_gt1 = {}, {}
        user_gt_dict_gtN, user_rec_dict_gtN = {}, {}

        if not self.skip_test:
            pbar = tqdm(
                colour="green",
                desc=f"Evaluating Epoch: {e + 1}",
                total=len(self.test_dl),
                dynamic_ncols=True,
            )
        for user, query, gt, mask, occ in self.test_dl:
            user = user.cpu().numpy()
            query = query.cpu().numpy()
            gt = gt.cpu().numpy()
            occ = occ.cpu().numpy()
            predictions = torch.FloatTensor(item_sim_matrix[query])
            predictions = predictions + mask
            scores, recommends = torch.topk(predictions, self.config["top_k"])
            recommends = recommends.cpu().numpy()

            for user_, occ_, query_, gt_, rec_ in zip(user, occ, query, gt, recommends):
                if query_ == 0:
                    continue

                user_last_product_lst = self.data_dict["device_train_dict_"].get(
                    user_, []
                )
                if user_last_product_lst:
                    user_last_product = user_last_product_lst[-1]
                else:
                    user_last_product = None
                if (
                    user_last_product
                    and query_ == user_last_product
                    and user_gt_dict_gtN.get(user_, -1) == -1
                ):
                    user_gt_dict_gtN[user_] = self.data_dict["device_test_dict"][user_]
                    user_rec_dict_gtN[user_] = rec_.tolist()

                u_key = f"{user_}_{occ_}"
                user_gt_dict_gt1[u_key] = [gt_]
                user_rec_dict_gt1[u_key] = rec_.tolist()

            if not self.skip_test:
                pbar.update(1)
        if not self.skip_test:
            pbar.close()

        res_gt1 = eval_metric(
            user_gt_dict_gt1,
            user_rec_dict_gt1,
            topk=self.config["top_k"],
        )
        res_gtN = eval_metric(
            user_gt_dict_gtN,
            user_rec_dict_gtN,
            topk=self.config["top_k"],
        )

        return {"gt1": res_gt1, "gtN": res_gtN}

    def train(self):
        start_epoch = 0

        if not self.skip_test:
            _, item_sim_matrix = self.build_item_matrix()
            start_test_res = self.recommend(-1, item_sim_matrix)
            self.record_loss("start", res=start_test_res)
            self.record_loss_mlflow(0, 0)
            self.record_metric_mlflow(0, start_test_res)

        validation_patience, best_perf, cur_perf = 0, 0, 0
        for e in range(start_epoch, start_epoch + self.config["train_epoch"]):
            if not self.skip_test and validation_patience > self.config["patience"]:
                break
            self.model.train()

            train_batch_loss = []
            if not self.skip_test:
                pbar = tqdm(
                    colour="blue",
                    desc=f"Training Epoch: {e + 1}",
                    total=self.config["batches_per_epoch"],
                    dynamic_ncols=True,
                )
            for _ in range(self.config["batches_per_epoch"]):
                pos_graph, neg_graph, blocks = next(self.train_dl)
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(self.device)
                pos_graph = pos_graph.to(self.device)
                neg_graph = neg_graph.to(self.device)

                self.opt.zero_grad()
                loss = self.model(pos_graph, neg_graph, blocks).mean()
                loss.backward()
                self.opt.step()

                train_batch_loss.append(loss.item())
                if not self.skip_test:
                    pbar.update(1)
                    pbar.set_description(
                        f"Training Epoch: {e + 1}/{self.config['train_epoch']} loss: {loss.item():.4f}"
                    )
            if not self.skip_test:
                pbar.close()
            train_loss_ = np.array(train_batch_loss).mean()

            item_embeddings, item_sim_matrix = self.build_item_matrix()
            if not self.skip_test:
                test_res = self.recommend(e, item_sim_matrix)
                cur_perf = test_res["gt1"]["general_MAP"]
                if best_perf >= cur_perf:
                    validation_patience += 1
                else:
                    validation_patience = 0
                    best_perf = cur_perf
                    self.build_faiss_index(item_embeddings)
                    self.save_checkpoint(e)
            else:
                test_res = None
                self.build_faiss_index(item_embeddings)
                self.save_checkpoint(e)

            self.record_loss(e, res=test_res, train_loss_=train_loss_)

    def run(self):
        self.train()


if __name__ == "__main__":
    with open("../configs/config.json", "rb") as f:
        configs = json.load(f)
    trainer = PinSAGETrainer(**configs)
    trainer.run()
