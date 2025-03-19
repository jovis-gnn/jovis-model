import io
import boto3
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models.pinsage.utils.helper import get_filtered_dataset_from_s3
from models.pinsage.modules.model import PinSAGE
from models.pinsage.modules.dataset import (
    PinSAGEData,
    PinSAGETestData,
    NeighborSampler,
    PinSAGECollator,
)
from models.helper import make_report


class PinSAGEInferencer:
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

    def load_from_s3(self):
        bucket = boto3.resource("s3").Bucket(self.model_s3_bucket_name)
        objects = bucket.objects.filter(
            Prefix=f"{self.company_id}/pinsage/{self.etl_id}/{self.model_id}"
        )
        obj_list = list(objects.all())

        for obj in obj_list:
            stream_ = io.BytesIO(obj.get()["Body"].read())
            if "pinsage.pth" in obj.key:
                self.checkpoint = torch.load(stream_)
                self.model = PinSAGE(self.checkpoint["config"], self.train_ds.graph).to(
                    self.device
                )
                self.model.load_state_dict(
                    self.checkpoint["model_state_dict"], strict=False
                )

    def get_dataset(self, test_batch_size=1):
        self.data_dict = get_filtered_dataset_from_s3(
            bucket_name=self.etl_s3_bucket_name,
            etl_version=self.etl_version,
            company_id=self.company_id,
            etl_id=self.etl_id,
            dataset_checkpoint=self.dataset_checkpoint,
        )
        self.train_ds = PinSAGEData(self.data_dict, 1)
        self.load_from_s3()
        neighbor_sampler = NeighborSampler(
            graph=self.train_ds.graph,
            random_walk_length=self.checkpoint["config"]["random_walk_length"],
            random_walk_restart_prob=self.checkpoint["config"][
                "random_walk_restart_prob"
            ],
            num_random_walks=self.checkpoint["config"]["num_random_walks"],
            num_neighbors=self.checkpoint["config"]["num_neighbors"],
            num_layers=self.checkpoint["config"]["num_layers"],
        )
        collator = PinSAGECollator(graph=self.train_ds.graph, sampler=neighbor_sampler)
        embed_dl = DataLoader(
            torch.arange(self.train_ds.graph.number_of_nodes("productid")),
            batch_size=1024,
            collate_fn=collator.collate_test,
            num_workers=8,
        )
        test_ds = PinSAGETestData(data_dict=self.data_dict)
        self.test_dl = DataLoader(
            test_ds,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=8,
        )
        self.model.eval()
        with torch.no_grad():
            h_item_batches = []
            for blocks in tqdm(embed_dl):
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(self.device)
                h_item_batches.append(self.model.get_embedding(blocks))
            h_item = torch.cat(h_item_batches, 0)
            self.h_item = h_item
        self.item_sim_matrix = h_item.matmul(h_item.T).detach().cpu().numpy()

    def process_torch_batch(self, num_batch=1, top_k=100):
        device_rec_dict = {}
        for idx, (user, query, gt, mask, occ) in enumerate(tqdm(self.test_dl)):
            user = user.cpu().numpy()
            query = query.cpu().numpy()
            gt = gt.cpu().numpy()
            occ = occ.cpu().numpy()
            predictions = torch.FloatTensor(self.item_sim_matrix[query])
            predictions = predictions + mask
            scores, recommends = torch.topk(predictions, top_k)
            scores = scores.cpu().numpy()
            recommends = recommends.cpu().numpy()

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

        return device_rec_dict, self.data_dict["product2pname"]


if __name__ == "__main__":
    pi = PinSAGEInferencer(
        etl_s3_bucket_name="prcmd-offline-store-dev",
        model_s3_bucket_name="prcmd-candidate-model-dev",
        company_id="01H2YP5B7M6R22A2ZEAACQQQRQ",
        etl_id="01HMVEW920VP9G1WNBJ4F98HT3",
        etl_version=4,
        model_id="pinsage_tuning_2",
        dataset_checkpoint="4weeks",
        use_gpu=True,
    )
    device_rec_dict, product_meta_dict = pi.process_torch_batch(num_batch=10, top_k=12)
    make_report(
        model_name="pinsage",
        model_id="pinsage_tuning_2",
        result_dict=device_rec_dict,
        meta_dict=product_meta_dict,
        image_path="/home/omnious/workspace/jovis/babathe_images_dev",
        save_path="/home/omnious/workspace/jovis/prcmd-model",
    )
