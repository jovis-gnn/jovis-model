import io
import faiss
import json
import boto3
import pickle
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from models.sent2vec.utils.helper import get_filtered_dataset_from_s3
from models.sent2vec.modules.model import ElectraForCL
from models.sent2vec.modules.dataset import Sent2VecTest
from models.helper import make_report

# from models.sent2vec.modules.string_matching_model import StringMatchingModel


class Sent2VecInferencer:
    def __init__(
        self,
        etl_s3_bucket_name: str,
        model_s3_bucket_name: str,
        company_id: str,
        etl_id: str,
        etl_version: str,
        model_id: str,
        dataset_checkpoint: str,
        use_gpu: bool,
    ):
        self.etl_s3_bucket_name = etl_s3_bucket_name
        self.model_s3_bucket_name = model_s3_bucket_name
        self.company_id = company_id
        self.etl_id = etl_id
        self.etl_version = etl_version
        self.model_id = model_id
        self.dataset_checkpoint = dataset_checkpoint
        self.device = torch.device("cuda:0") if use_gpu else torch.device("cpu")

        self.get_dataset()
        self.load_from_s3()

    def load_from_s3(self):
        bucket = boto3.resource("s3").Bucket(self.model_s3_bucket_name)
        objects = bucket.objects.filter(
            Prefix=f"{self.company_id}/sent2vec/{self.etl_id}/{self.model_id}"
        )
        obj_list = list(objects.all())

        for obj in obj_list:
            stream_ = io.BytesIO(obj.get()["Body"].read())
            if "faiss.index" in obj.key:
                faiss_raw_reader = faiss.BufferedIOReader(
                    faiss.PyCallbackIOReader(stream_.read)
                )
                self.faiss_index = faiss.read_index(faiss_raw_reader)
            elif "mapping" in obj.key:
                mapping = pickle.load(stream_)
                self.product2idx, self.idx2product = (
                    mapping["product2idx"],
                    mapping["idx2product"],
                )

        with open("models/sent2vec/configs/config.json", "rb") as f:
            self.config = json.load(f)

        model_config = AutoConfig.from_pretrained("models/sent2vec/data/model")
        self.tokenizer = AutoTokenizer.from_pretrained("models/sent2vec/data/model")
        model_args = {"temp": 0.05, "pooler_type": "cls"}
        self.model = ElectraForCL.from_pretrained(
            "models/sent2vec/data/model", config=model_config, **model_args
        )
        self.model = self.model.to(self.device)

        # with open(
        #     "models/sent2vec/data/item2category_20230306_entity2label.json", "rb"
        # ) as f:
        #     item2fine_ctgr_file = json.load(f)
        # self.string_matching_model = StringMatchingModel()
        # self.string_matching_model.load(fp=item2fine_ctgr_file)

    def get_dataset(self, test_batch_size=1):
        self.data_dict = get_filtered_dataset_from_s3(
            bucket_name=self.etl_s3_bucket_name,
            etl_version=self.etl_version,
            company_id=self.company_id,
            etl_id=self.etl_id,
            dataset_checkpoint=self.dataset_checkpoint,
        )
        self.test_ds = Sent2VecTest(self.data_dict)
        self.test_dl = DataLoader(
            self.test_ds,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=8,
        )

    def process_torch_batch(self, num_batch=1, top_k=12):
        device_rec_dict = {}
        for idx, (user, query, gt, mask, occ, prod_name) in enumerate(
            tqdm(self.test_dl)
        ):
            user = user.cpu().numpy()
            query = query.cpu().numpy()
            gt = gt.cpu().numpy()
            occ = occ.cpu().numpy()
            mask = mask.cpu().numpy()
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
            if self.config["use_pooler_output"]:
                query_embeddings = model_outputs.pooler_output
            else:
                query_embeddings = model_outputs.last_hidden_state[:, 0]
            query_embeddings = query_embeddings.detach().cpu().numpy()
            scores, recommends = self.faiss_index.search(
                query_embeddings, self.faiss_index.ntotal
            )
            all_scores = np.ones(
                shape=(len(recommends), self.data_dict["num_product"])
            ) * (-10000)
            np.put_along_axis(all_scores, recommends, scores, axis=1)

            all_scores = all_scores + mask
            scores = np.sort(all_scores)[:, ::-1][:, :top_k]
            recommends = np.argsort(all_scores)[:, ::-1][:, :top_k]

            for user_, occ_, query_, rec_, score_ in zip(
                user, occ, query, recommends, scores
            ):
                if query_ == 0:
                    continue
                history = [
                    self.data_dict["idx2product"][p]
                    for p in self.data_dict["device_train_dict"].get(user_, [])[
                        : occ_ - 1
                    ][-top_k:][::-1]
                ][:1]
                rec_ = [self.data_dict["idx2product"][p] for p in rec_]
                user_ = self.data_dict["idx2device"][user_]
                u_key = f"{user_}_{occ_}"
                device_rec_dict[u_key] = {}
                device_rec_dict[u_key]["history"] = history
                device_rec_dict[u_key]["recommend"] = rec_
                device_rec_dict[u_key]["recommend_score"] = score_
            if idx == (num_batch - 1):
                break

        product_meta_dict = {}
        for pid, pname in self.test_ds.pid2pname.items():
            product_meta_dict[self.data_dict["idx2product"][pid]] = pname

        return device_rec_dict, product_meta_dict


if __name__ == "__main__":
    si = Sent2VecInferencer(
        etl_s3_bucket_name="prcmd-offline-store-dev",
        model_s3_bucket_name="prcmd-candidate-model-dev",
        company_id="01H2YP5B7M6R22A2ZEAACQQQRQ",
        etl_id="01HMVEW920VP9G1WNBJ4F98HT3",
        etl_version=4,
        model_id="test",
        dataset_checkpoint="4weeks",
        use_gpu=True,
    )
    device_rec_dict, product_meta_dict = si.process_torch_batch(num_batch=100, top_k=12)
    make_report(
        model_name="sent2vec",
        model_id="test_sent2vec",
        result_dict=device_rec_dict,
        meta_dict=product_meta_dict,
        image_path="/home/omnious/workspace/jovis/babathe_images_dev",
        save_path="/home/omnious/workspace/jovis/prcmd-model",
    )
