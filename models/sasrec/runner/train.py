import os
import boto3
import pickle
import json
from os.path import join, dirname

import faiss
import torch
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from models.sasrec.runner.initializer import build_model
from models.sasrec.utils.helper import init_logger, get_filtered_dataset_from_s3
from models.sasrec.utils.dataset import SND, MND
from models.sasrec.utils.scheduler import get_scheduler
from models.metric import eval_metric

from models.sasrec.modules.sentencer import ProdNameModel


class Trainer:
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
        self.full_train = self.config.get("full_train", True)

        self.data_dict = get_filtered_dataset_from_s3(self.config)

        dataset_getter = {
            **dict.fromkeys(["BPR", "BCE"], SND),
            **dict.fromkeys(["SCE", "LSCE", "LBPR_max"], MND),
        }
        sample_getter = {
            **dict.fromkeys(["BPR", "BCE"], "weak"),
            **dict.fromkeys(["SCE"], "orig"),
            **dict.fromkeys(["LSCE", "LBPR_max"], "multi"),
        }

        dataset_func, sample_method = (
            dataset_getter[self.config["loss_type"]],
            sample_getter[self.config["loss_type"]],
        )
        seq_len, batchwise_sample = tuple(
            map(self.config.get, ["seq_len", "batchwise_sample"])
        )

        self.num_popular_negs, self.num_total_negs = 50, 100
        kwargs_dict = {
            "num_popular_negs": self.num_popular_negs,
            "num_total_negs": self.num_total_negs,
            "full_train": self.full_train,
        }

        self.save_mapping()

        train_ds = dataset_func(
            self.data_dict,
            seq_len,
            usage="train",
            batchwise_sample=batchwise_sample,
            sample_method=sample_method,
            **kwargs_dict,
        )
        test_ds = dataset_func(
            self.data_dict,
            seq_len,
            usage="test",
            batchwise_sample=batchwise_sample,
            sample_method=sample_method,
            **kwargs_dict,
        )

        self.popularity = np.array(
            [
                len(self.data_dict["product_train_dict"].get(x, []))
                for x in np.arange(self.data_dict["num_product"])
            ]
        )

        self.train_dl = DataLoader(
            train_ds, batch_size=self.config["batch_size"], shuffle=True, num_workers=8
        )
        self.test_dl = DataLoader(
            test_ds,
            batch_size=self.config["batch_size_test"],
            shuffle=False,
            num_workers=4,
        )

        self.device = (
            torch.device("cuda") if self.config["use_gpu"] else torch.device("cpu")
        )

        if False:
            productname_model = ProdNameModel(self.device)

            name_embed_list = []
            for idx in tqdm(range(self.data_dict["num_product"])):
                name_embed_list.append(
                    productname_model(self.data_dict["productname_table"][idx])
                    .detach()
                    .cpu()
                    .numpy()
                )

            productname_embedding = np.concatenate(name_embed_list, axis=0)

            np.save("productname_embedding.npy", productname_embedding)

        self.model = build_model(self.config["model_type"])(
            self.config,
            num_user=self.data_dict["num_device"],
            num_item=self.data_dict["num_product"],
            device=self.device,
            num_meta=self.data_dict["num_meta"],
            meta_information=self.data_dict["meta_table"],
            productname_table=None,
        ).to(self.device)

        opt_mapping = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}
        weight_decay = (
            self.config["weight_decay"] if self.config["weight_decay_opt"] else 0
        )

        self.opt = opt_mapping[self.config["optimizer"]](
            self.model.parameters(), lr=self.config["lr"], weight_decay=weight_decay
        )

        total_steps = int(len(self.train_dl))
        warmup_steps = total_steps // 10

        if self.config["scheduler_type"] != "original":
            self.scheduler = get_scheduler(
                self.opt, self.config["scheduler_type"], total_steps, warmup_steps
            )

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
            f"/tmp/{self.config['model_id']}.onnx",
            verbose=False,
            input_names=["history", "history_mask"],
            output_names=["history_embedding"],
            opset_version=14,
        )
        self.model.forward = orig_forw_ptr

        session = boto3.Session()
        client = session.client(
            "s3",
            aws_access_key_id=self.config["aws_access_key_id"],
            aws_secret_access_key=self.config["aws_secret_access_key"],
        )
        client.put_object(
            Body=open(f"/tmp/{self.config['model_id']}.onnx", "rb"),
            Bucket=self.config["model_s3_bucket_name"],
            Key="{}/{}/{}/{}/sasrec.onnx".format(
                self.config["company_id"],
                "sasrec",
                self.config["etl_id"],
                self.config["model_id"],
            ),
        )

        torch.save(self.model.state_dict(), f"/tmp/{self.config['model_id']}.pth")
        client.put_object(
            Body=open(f"/tmp/{self.config['model_id']}.pth", "rb"),
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

    def whole_process_batch(self, cur_epoch):
        self.model.eval()

        if not self.skip_test:
            pbar = tqdm(
                colour="green",
                desc="Evaluating Epoch: {}".format(cur_epoch),
                total=len(self.test_dl),
                dynamic_ncols=True,
            )

        filtered_pool_on_device = None
        if self.config["filter_item_pool"]:
            filtered_pool = self.data_dict["train_df"]["productid_model_index"].unique()
            filtered_pool_on_device = torch.LongTensor(filtered_pool).to(self.device)

        user_gt_dict_1, user_rec_dict_1, user_gt_dict_N, user_rec_dict_N = (
            {},
            {},
            {},
            {},
        )
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
                user,
                history,
                history_mask,
                item=filtered_pool_on_device,
                return_out_only=True,
            )
            predictions = predictions.detach() + purchase_mask.type(
                predictions.dtype
            ).to(self.device)
            scores, recommends = torch.topk(predictions, self.config["top_k"])
            recommends = recommends.detach().cpu().numpy()

            if self.config["filter_item_pool"]:
                recommends = filtered_pool[recommends]

            user_occ = list(zip(user.cpu().numpy().astype(np.int32), occ.numpy()))
            pos = user_pos[:, 1].cpu().numpy().astype(np.int32)
            for inner_idx, per_user_occ in enumerate(user_occ):
                user_gt_dict_1[per_user_occ] = [pos[inner_idx]]
                user_rec_dict_1[per_user_occ] = recommends[inner_idx].tolist()

                if per_user_occ[-1] == 1:
                    user_gt_dict_N[per_user_occ] = self.data_dict["device_test_dict"][
                        per_user_occ[0]
                    ]
                    user_rec_dict_N[per_user_occ] = recommends[inner_idx].tolist()

            if not self.skip_test:
                pbar.update(1)

        if not self.skip_test:
            pbar.close()

        res_gt1 = eval_metric(
            user_gt_dict_1,
            user_rec_dict_1,
            topk=self.config["top_k"],
        )

        res_gtN = eval_metric(
            user_gt_dict_N,
            user_rec_dict_N,
            topk=self.config["top_k"],
        )

        return {"gt1": res_gt1, "gtN": res_gtN}

    def train(self, skip_first=True):
        validation_patience, best_perf, cur_perf = 0, 0, 0

        if not self.skip_test and not skip_first:
            start_test_res = self.whole_process_batch(cur_epoch=0)

            self.record_loss(0, start_test_res)

        for epoch in range(self.config["train_epoch"]):
            if not self.skip_test and validation_patience > self.config["patience"]:
                break

            if not self.skip_test:
                pbar = tqdm(
                    colour="blue",
                    desc="Training Epoch: {}".format(epoch),
                    total=len(self.train_dl),
                    dynamic_ncols=True,
                )

            self.model.train()
            total_loss_batch = []
            for idx, per_input_batch in enumerate(self.train_dl):
                per_input_batch = [
                    (
                        per_input.type(torch.LongTensor).to(self.device)
                        if len(per_input) > 0
                        else None
                    )
                    for per_input in per_input_batch
                ]
                if self.config["batchwise_sample"]:
                    user_pos, history, history_mask = per_input_batch
                else:
                    user_pos, history, history_mask, neg, random_neg = per_input_batch
                user, pos = user_pos[:, 0], user_pos[:, -1]

                if self.config["batchwise_sample"]:
                    pos_np = pos.detach().cpu().numpy()

                    candidates = np.delete(
                        np.arange(self.data_dict["num_product"]), pos_np
                    )

                    if self.config["loss_type"] in ["LSCE", "LBPR_max"]:
                        candidates_prob = np.delete(self.popularity, pos_np)

                        bw_neg = np.random.choice(
                            candidates,
                            (
                                user.shape[0],
                                self.num_popular_negs,
                            ),
                            p=candidates_prob / sum(candidates_prob),
                            replace=True,
                        )

                        bw_ran = np.random.choice(
                            candidates,
                            (
                                user.shape[0],
                                self.num_total_negs - self.num_popular_negs,
                            ),
                            replace=True,
                        )
                    elif self.config["loss_type"] in ["SCE"]:
                        bw_ran = np.random.choice(
                            candidates,
                            (
                                user.shape[0],
                                self.num_total_negs,
                            ),
                            replace=True,
                        )

                        bw_neg = bw_ran

                    neg, random_neg = tuple(
                        map(
                            lambda x: torch.LongTensor(x).to(self.device),
                            (bw_neg, bw_ran),
                        )
                    )

                self.opt.zero_grad()
                total_loss, _, _ = self.model(
                    user, pos, neg, history, history_mask, random_neg
                )
                total_loss.backward()

                if self.config["clip_grad_ratio"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["clip_grad_ratio"]
                    )
                self.opt.step()

                if self.config["scheduler_type"] != "original":
                    self.scheduler.step()

                _ = list(
                    map(
                        lambda x, arr: arr.append(x),
                        [total_loss.item()],
                        [
                            total_loss_batch,
                        ],
                    )
                )

                if not self.skip_test:
                    pbar.update(1)
                    pbar.set_description(
                        "Training Epoch: {} Loss: {:.4f}".format(
                            epoch + 1, total_loss.item()
                        )
                    )

            if not self.skip_test:
                pbar.close()

            train_loss_ = np.array(total_loss_batch).mean()

            if not self.skip_test:
                test_res = self.whole_process_batch(cur_epoch=epoch + 1)
                cur_perf = test_res["gt1"]["general_MAP"]

                if best_perf >= cur_perf:
                    validation_patience += 1
                else:
                    validation_patience = 0
                    best_perf = cur_perf
            else:
                test_res = None

            self.record_loss(epoch + 1, test_res, train_loss_)

            if self.skip_test:
                self.build_faiss_index()
                self.save_checkpoint()

    def local_process_batch(self, cur_epoch):
        self.model.eval()

        user_gt_dict, user_rec_dict = {}, {}
        for user_pos, history, history_mask, occ, purchase_mask in tqdm(self.test_dl):
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

            user_occ = list(
                zip(user_pos[:, 0].cpu().numpy().astype(np.int32), occ.numpy())
            )
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

    def run(self):
        self.train(skip_first=True)


if __name__ == "__main__":
    with open("../configs/sasrec_config.json", "rb") as f:
        configs = json.load(f)

    _seed = 42

    np.random.seed(_seed)
    torch.manual_seed(_seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # np.set_printoptions(suppress=True)
    # torch.set_printoptions(sci_mode=False)

    trainer = Trainer(**configs)
    trainer.run()
