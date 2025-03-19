import io

import boto3
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lightgcl.modules.dataset import LightGCLData, LightGCLTest
from models.lightgcl.modules.model import LightGCL
from models.lightgcl.utils.helper import get_filtered_dataset_from_s3


class LightGCLInferencer:
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
            Prefix=f"{self.company_id}/lightgcl/{self.etl_id}/{self.model_id}"
        )
        obj_list = list(objects.all())

        for obj in obj_list:
            stream_ = io.BytesIO(obj.get()["Body"].read())
            if "lightgcl_start.pth" in obj.key:
                self.checkpoint = torch.load(stream_)
                self.torch_model = LightGCL(
                    self.checkpoint["config"],
                    num_device=self.checkpoint["num_device"],
                    num_product=self.checkpoint["num_product"],
                    graph_matrix=[self.device_product_norm, self.product_device_norm],
                ).to(self.device)
                self.torch_model.load_state_dict(
                    self.checkpoint["model_state_dict"], strict=False
                )

    def get_dataset(self, test_batch_size=1):
        self.data_dict = get_filtered_dataset_from_s3(
            self.etl_s3_bucket_name,
            "sasrec_dataset/v{}/companyid={}/etlid={}/checkpoint={}".format(
                self.etl_version,
                self.company_id,
                self.etl_id,
                self.dataset_checkpoint,
            ),
        )
        train_ds = LightGCLData(self.data_dict)
        self.device_product_norm, self.product_device_norm = train_ds.get_graph_matrix()
        self.device_product_norm = self.device_product_norm.coalesce().to(self.device)
        self.product_device_norm = self.product_device_norm.coalesce().to(self.device)
        self.test_ds = LightGCLTest(self.data_dict)
        self.test_dl = DataLoader(
            self.test_ds,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=8,
        )

    def process_torch_batch(self, num_batch=1, top_k=100):
        self.torch_model.eval()

        device_rec_dict = {}
        for idx, (u, mask) in enumerate(tqdm(self.test_dl)):
            w_cold = u[:, 0].detach().cpu().numpy()
            wo_cold = []
            for u_ in w_cold:
                if self.data_dict["device_train_dict"].get(u_, -1) == -1:
                    wo_cold.append(0)
                else:
                    wo_cold.append(int(u_))
            wo_cold = torch.LongTensor(wo_cold).to(self.device)
            mask = mask.type(torch.FloatTensor).to(self.device)

            predictions = self.torch_model.recommend(wo_cold)
            predictions = predictions + mask
            scores, recommends = torch.topk(predictions, top_k)
            recommends = recommends.detach().cpu().numpy()
            scores = scores[0].detach().cpu().numpy()

            for idx_, u_ in enumerate(w_cold):
                user = self.data_dict["idx2device"][u_]
                device_rec_dict[user] = {}
                history = [
                    self.data_dict["idx2product"][p]
                    for p in self.data_dict["device_train_dict"].get(u_, [])[-top_k:][
                        ::-1
                    ]
                ]
                rec = [self.data_dict["idx2product"][p] for p in recommends[idx_]]
                device_rec_dict[user]["history"] = history
                device_rec_dict[user]["recommend"] = rec
                device_rec_dict[user]["recommend_score"] = scores
            if idx == (num_batch - 1):
                break

        return device_rec_dict
