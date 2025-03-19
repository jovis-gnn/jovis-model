import pickle
from tqdm import tqdm

import json
import mlflow

import scipy
import torch
from torch.utils.data import DataLoader

from models.helper import init_logger, BotoSession
from models.metric import eval_metric
from models.itemcf.utils.helper import get_filtered_dataset_from_s3
from models.itemcf.modules.model import ItemCF
from models.itemcf.modules.dataset import ItemCFData


class ItemCFTrainer:
    def __init__(self, **config):
        self.logger = init_logger("trainer")
        self.config = config
        self.session = BotoSession().refreshable_session()
        self.logger.info(json.dumps(config, indent=4))
        self.data_dict = get_filtered_dataset_from_s3(
            self.config["etl_s3_bucket_name"],
            "sasrec_dataset/v{}/companyid={}/etlid={}/checkpoint={}".format(
                self.config["etl_version"],
                self.config["company_id"],
                self.config["etl_id"],
                self.config["dataset_checkpoint"],
            ),
        )
        self.save_mapping()
        self.test_ds = ItemCFData(data_dict=self.data_dict)
        self.test_dl = DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )
        self.model = ItemCF(self.data_dict)
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
                "itemcf",
                self.config["etl_id"],
                self.config["model_id"],
            ),
        )

    def save_checkpoint(self, item_sim_matrix):
        client = self.session.client("s3")
        scipy.sparse.save_npz("/tmp/itemcf.npz", item_sim_matrix)
        client.put_object(
            Body=open("/tmp/itemcf.npz", "rb"),
            Bucket=self.config["model_s3_bucket_name"],
            Key="{}/{}/{}/{}/itemcf.npz".format(
                self.config["company_id"],
                "itemcf",
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

    def whole_process_batch(self, item_sim_matrix, num_target=5):
        user_gt_dict, user_rec_dict = {}, {}
        for idx, (user, mask) in enumerate(tqdm(self.test_dl)):
            user = user.cpu().numpy()[0]

            device_train_data = self.data_dict["device_train_dict"].get(int(user), [])[
                -(num_target):
            ]
            if len(device_train_data) == 0:
                continue
            predictions = torch.FloatTensor(
                item_sim_matrix[device_train_data].sum(axis=0)
            )
            predictions = predictions + mask
            scores, recommends = torch.topk(predictions, self.config["top_k"])
            recommends = recommends.detach().cpu().numpy()

            user_gt_dict[user] = self.data_dict["device_test_dict"][user]
            user_rec_dict[user] = recommends[0].tolist()

        res = eval_metric(
            user_gt_dict,
            user_rec_dict,
            self.data_dict["device_train_dict"],
            topk=self.config["top_k"],
        )

        return res

    def train(self):
        item_sim_matrix = self.model.get_item_sim_matrix()
        self.save_checkpoint(item_sim_matrix)
        res = self.whole_process_batch(item_sim_matrix)
        self.record_loss(0, res)
        self.record_metric_mlflow(0, res)

    def run(self):
        with mlflow.start_run(run_name=self.config["run_name"]):
            mlflow.log_params(self.config)
            self.train()


if __name__ == "__main__":
    with open("../configs/config.json", "rb") as f:
        configs = json.load(f)
    trainer = ItemCFTrainer(**configs)
    trainer.run()
