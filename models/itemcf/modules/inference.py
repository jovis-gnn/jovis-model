import io
import boto3
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import scipy

from models.itemcf.modules.dataset import ItemCFData
from models.itemcf.utils.helper import get_filtered_dataset_from_s3


class ItemCFInferencer:
    def __init__(
        self,
        etl_s3_bucket_name: str,
        model_s3_bucket_name: str,
        company_id: str,
        etl_id: str,
        etl_version: str,
        model_id: str,
        dataset_checkpoint: str,
    ):
        self.etl_s3_bucket_name = etl_s3_bucket_name
        self.model_s3_bucket_name = model_s3_bucket_name
        self.company_id = company_id
        self.etl_id = etl_id
        self.etl_version = etl_version
        self.model_id = model_id
        self.dataset_checkpoint = dataset_checkpoint

        self.get_dataset()
        self.load_from_s3()

    def load_from_s3(self):
        bucket = boto3.resource("s3").Bucket
        objects = bucket.objects.filter(
            Prefix=f"{self.company_id}/itemcf/{self.etl_id}/{self.model_id}"
        )
        obj_list = list(objects.all())

        for obj in obj_list:
            stream_ = io.BytesIO(obj.get()["Body"].read())
            if "itemcf.npz" in obj.key:
                self.model = scipy.sparse.load_npz(stream_)

    def get_dataset(self, test_batch_size=1):
        self.data_dict = get_filtered_dataset_from_s3(
            self.config["etl_s3_bucket_name"],
            "sasrec_dataset/v{}/companyid={}/etlid={}/checkpoint={}".format(
                self.config["etl_version"],
                self.config["company_id"],
                self.config["etl_id"],
                self.config["dataset_checkpoint"],
            ),
        )
        self.test_ds = ItemCFData(data_dict=self.data_dict)
        self.test_dl = DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=8,
        )

    def process_torch_batch(self, num_batch=1, num_target=5, top_k=100):
        cnt = 0
        device_rec_dict = {}
        for idx, (user, mask) in enumerate(tqdm(self.test_dl)):
            user = user.cpu().numpy()[0]

            device_train_data = self.data_dict["device_train_dict"].get(int(user), [])[
                -(num_target):
            ][::-1]
            if len(device_train_data) == 0:
                continue
            predictions = torch.FloatTensor(self.model[device_train_data].sum(axis=0))
            predictions *= mask
            scores, recommends = torch.topk(predictions, top_k)

            user = self.data_dict["idx2device"][int(user)]
            history = [self.data_dict["idx2product"][h] for h in device_train_data]
            recommends = [
                self.data_dict["idx2product"][r] for r in recommends.cpu().numpy()[0]
            ]
            scores = scores.cpu().numpy()[0]
            device_rec_dict[user] = {}
            device_rec_dict[user]["history"] = history
            device_rec_dict[user]["recommend"] = recommends
            device_rec_dict[user]["recommend_score"] = scores
            cnt += 1
            if cnt == num_batch:
                break
        return device_rec_dict
