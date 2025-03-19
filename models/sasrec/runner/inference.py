import io
import faiss
import boto3
import torch
import onnx
import onnxruntime
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.sasrec.utils.helper import get_filtered_dataset_from_s3
from models.sasrec.utils.dataset import UnifiedDataset
from models.sasrec.runner.initializer import build_model


class SASRecInferencer:
    def __init__(
        self,
        etl_s3_bucket_name: str,
        model_s3_bucket_name: str,
        company_id: str,
        etl_id: str,
        etl_version: str,
        model_id: str,
        dataset_checkpoint: str,
        window_size: int,
        use_gpu: bool
    ):
        self.etl_s3_bucket_name = etl_s3_bucket_name
        self.model_s3_bucket_name = model_s3_bucket_name
        self.company_id = company_id
        self.etl_id = etl_id
        self.etl_version = etl_version
        self.model_id = model_id
        self.dataset_checkpoint = dataset_checkpoint
        self.window_size = window_size
        self.device = (
            torch.device("cuda:0") if use_gpu else torch.device("cpu")
        )

        self.load_from_s3()
        self.get_dataset()

    def load_from_s3(self):
        bucket = boto3.resource("s3").Bucket(self.model_s3_bucket_name)
        objects = bucket.objects.filter(
            Prefix=f"{self.company_id}/sasrec/{self.etl_id}/{self.model_id}"
        )
        obj_list = list(objects.all())

        for obj in obj_list:
            stream_ = io.BytesIO(obj.get()['Body'].read())
            if 'faiss.index' in obj.key:
                faiss_raw_reader = faiss.BufferedIOReader(faiss.PyCallbackIOReader(stream_.read))
                self.faiss_index = faiss.read_index(faiss_raw_reader)
            elif 'mapping' in obj.key:
                mapping = pickle.load(stream_)
                self.product2idx, self.idx2product = mapping['product2idx'], mapping['idx2product']
            elif 'sasrec.onnx' in obj.key:
                self.onnx_model = onnx.load(stream_)
            elif 'sasrec.pth' in obj.key:
                self.checkpoint = torch.load(stream_)
                self.torch_model = build_model(self.checkpoint['config']['model_type'])(
                    self.checkpoint["config"],
                    num_user=self.checkpoint["num_device"],
                    num_item=self.checkpoint["num_product"],
                    device=self.device,
                ).to(self.device)
                self.torch_model.load_state_dict(self.checkpoint["model_state_dict"], strict=False)

    def get_dataset(self, test_batch_size=1):
        self.data_dict = get_filtered_dataset_from_s3(
            self.etl_s3_bucket_name,
            f"sasrec_dataset/v{self.etl_version}/companyid={self.company_id}/etlid={self.etl_id}/checkpoint={self.dataset_checkpoint}",
            filter_test_df=True
        )
        self.product_conv_dict = self.data_dict["df"][["productid_model_index", "productid"]].drop_duplicates().set_index("productid_model_index").to_dict()["productid"]
        self.device_conv_dict = self.data_dict["df"][["deviceid_model_index", "deviceid"]].drop_duplicates().set_index("deviceid_model_index").to_dict()["deviceid"]
        self.product_conv_arr = self.data_dict["df"][["productid_model_index", "productid"]].drop_duplicates().set_index("productid_model_index").sort_index().to_numpy()[:, 0]

        self.test_ds = UnifiedDataset(
            data_dict=self.data_dict, seq_len=self.checkpoint["config"]["seq_len"], window_size=self.window_size, usage="test"
        )
        self.test_dl = DataLoader(
            self.test_ds,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=4,
        )

    def process_onnx_batch(self, num_batch=1, top_k=100):
        session = onnxruntime.InferenceSession(self.onnx_model.SerializeToString())

        # for idx, input_batch in enumerate(tqdm(self.test_dl)):
        #     user, history, history_mask, purchase_mask = [per_input.numpy() if per_input.shape[-1] > 50 else per_input.numpy().astype(np.int64) for per_input in input_batch]

        device_rec_dict = {}
        for idx, (user, history, history_mask, _, purchase_mask) in enumerate(tqdm(self.test_dl)):
            history = history.numpy().astype(np.int64)
            history_mask = history_mask.numpy().astype(np.int64)

            user, purchase_mask = user.cpu().numpy(), purchase_mask.cpu().numpy()

            his_emb = session.run(['history_embedding'], {'history': history, 'history_mask': history_mask})[0]

            prev_scores, prev_recommends = self.faiss_index.search(his_emb, top_k)

            raw_scores, raw_recommends = self.faiss_index.search(his_emb, purchase_mask.shape[-1])
            raw_scores += np.take_along_axis(purchase_mask, raw_recommends, axis=1)

            unsorted_idx = np.argpartition(raw_scores, -top_k, 1)[:, -top_k:]
            sorted_idx = np.take_along_axis(unsorted_idx, np.argsort(np.take_along_axis(raw_scores, unsorted_idx, 1), 1), 1)[:, ::-1]

            recommends, scores = np.take_along_axis(raw_recommends, sorted_idx, 1), np.take_along_axis(raw_scores, sorted_idx, 1)

            conv_recommends = [[self.product_conv_dict[idx] for idx in recommends[0]]]
            conv_users = [self.device_conv_dict[int(user[0][0])]]
            history = history[:, -top_k:]
            conv_history = []
            for h in history:
                conv_history.append([self.product_conv_dict.get(idx, -1) for idx in h][::-1])
            for user, rec, score, history in zip(conv_users, conv_recommends, scores, conv_history):
                device_rec_dict[user] = {}
                device_rec_dict[user]["history"] = history
                device_rec_dict[user]["recommend"] = rec
                device_rec_dict[user]["recommend_score"] = score
            if idx == (num_batch - 1):
                break
        return device_rec_dict

    def process_torch_batch(self, num_batch=1, top_k=100):
        self.torch_model.eval()

        device_rec_dict = {}
        for idx, (user, history, history_mask, _, purchase_mask) in enumerate(tqdm(self.test_dl)):
            history = history.type(torch.LongTensor).to(self.device)
            history_mask = history_mask.type(torch.LongTensor).to(self.device)

            his_emb = self.torch_model.get_history_embedding(history, history_mask).detach().cpu().numpy()

            prev_scores, prev_recommends = self.faiss_index.search(his_emb, top_k)

            raw_scores, raw_recommends = self.faiss_index.search(his_emb, purchase_mask.shape[-1])
            raw_scores += np.take_along_axis(purchase_mask.cpu().numpy(), raw_recommends, axis=1)

            unsorted_idx = np.argpartition(raw_scores, -top_k, 1)[:, -top_k:]
            sorted_idx = np.take_along_axis(unsorted_idx, np.argsort(np.take_along_axis(raw_scores, unsorted_idx, 1), 1), 1)[:, ::-1]

            recommends, scores = np.take_along_axis(raw_recommends, sorted_idx, 1), np.take_along_axis(raw_scores, sorted_idx, 1)

            conv_recommends = self.product_conv_arr[recommends - 1]
            conv_users = [self.device_conv_dict[idx] for idx in user.numpy().astype(np.int64)[:, 0]]
            history = history.detach().cpu().numpy().astype(np.int64)[:, -top_k:]
            conv_history = []
            for h in history:
                conv_history.append([self.product_conv_dict.get(idx, -1) for idx in h][::-1])
            for user, rec, score, history in zip(conv_users, conv_recommends, scores, conv_history):
                device_rec_dict[user] = {}
                device_rec_dict[user]["history"] = history
                device_rec_dict[user]["recommend"] = rec
                device_rec_dict[user]["recommend_score"] = score
            if idx == (num_batch - 1):
                break
        return device_rec_dict


if __name__ == "__main__":
    # si = SASRecInferencer(
    #     etl_s3_bucket_name="prcmd-offline-store-dev",
    #     model_s3_bucket_name="prcmd-candidate-model-dev",
    #     company_id="01H2YP5B7M6R22A2ZEAACQQQRQ",
    #     etl_id="01HF90H0DTMA3ZQKC4C17P32D1",
    #     etl_version="1",
    #     model_id="test_sasrec",
    #     dataset_checkpoint="4week",
    #     window_size=0,
    #     use_gpu=True
    # )

    si = SASRecInferencer(
        etl_s3_bucket_name="prcmd-offline-store-dev",
        model_s3_bucket_name="prcmd-candidate-model-dev",
        company_id="01H2YP5B7M6R22A2ZEAACQQQRQ",
        etl_id="01HF90H0DTMA3ZQKC4C17P32D1",
        etl_version="1",
        model_id="01HFRJBHEME0GDWRFT51WPYVK8",
        dataset_checkpoint="4week",
        window_size=0,
        use_gpu=True
    )
    # device_rec_dict = si.process_onnx_batch(num_batch=80, top_k=12)

    device_rec_dict = si.process_torch_batch(num_batch=5, top_k=12)

    # bucket = boto3.resource("s3").Bucket('prcmd-candidate-model-dev')

    # obj = bucket.Object(
    #     "{}/{}/{}/{}/sasrec.onnx".format(
    #         "01H2YP5B7M6R22A2ZEAACQQQRQ",
    #         "sasrec",
    #         "01HBAGB20408KKG9PZ3Z44W244",
    #         "test_sasrec",
    #     )
    # )

    # model = onnx.load(io.BytesIO(obj.get()['Body'].read()))

    # session = onnxruntime.InferenceSession(model.SerializeToString())

    # history, history_mask = np.arange(50).reshape(1, 50).astype(np.int64), np.ones(50).reshape(1, 50).astype(np.int64)

    # out = session.run(['history_embedding'], {'history': history, 'history_mask': history_mask})[0]
