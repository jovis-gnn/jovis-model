import io
import boto3
from boto3.session import Session

import numpy as np
import lightgbm as lgb

from models.helper import get_data_from_s3


class RankerInferencer:
    def __init__(
        self,
        etl_s3_bucket_name: str,
        ranker_s3_bucket_name: str,
        company_id: str,
        strategy_id: str,
        etl_id: str,
        etl_version: str,
        model_id: str,
        dataset_checkpoint: str
    ):
        self.etl_s3_bucket_name = etl_s3_bucket_name
        self.ranker_s3_bucket_name = ranker_s3_bucket_name
        self.company_id = company_id
        self.strategy_id = strategy_id
        self.etl_id = etl_id
        self.etl_version = etl_version
        self.model_id = model_id
        self.dataset_checkpoint = dataset_checkpoint
        self.model = self.get_model_from_string()
        self.filtering_string = [
            "device_id", "product_id",
            "deviceid_index", "productid_index", "timestamp", "event_timestamp", "label",
            "companyid", "etlid", "checkpoint", "validation", "group_id", "finecategorygroupid_index",
            "heelheightname_index", "toetypename_index", "heelshapename_index", "soletypename_index",
            "strapname_index", "sizename_index", "mainmaterialname_index", "submaterialname_index"
        ]

    def merge_id(self, df):
        prefix = f"index_features/v{self.etl_version}/companyid={self.company_id}/feature=deviceid"
        device_idx = get_data_from_s3(self.etl_s3_bucket_name, prefix)
        prefix = f"index_features/v{self.etl_version}/companyid={self.company_id}/feature=productid"
        product_idx = get_data_from_s3(self.etl_s3_bucket_name, prefix)
        device_idx = device_idx.rename(columns={"feature_name": "device_id", "feature_index": "deviceid_index"})[["device_id", "deviceid_index"]]
        product_idx = product_idx.rename(columns={"feature_name": "product_id", "feature_index": "productid_index"})[["product_id", "productid_index"]]

        df = df.merge(device_idx, on="deviceid_index", how="left")
        df = df.merge(product_idx, on="productid_index", how="left")
        return df

    def get_device_train_dict(self, topk):
        prefix = f"device_train_history/v{self.etl_version}/companyid={self.company_id}/etlid={self.etl_id}"
        train_df = get_data_from_s3(self.etl_s3_bucket_name, prefix)
        train_df = train_df[["deviceid_index", "productid_index", "timestamp"]]
        train_df = (
            train_df[["deviceid_index", "productid_index", "timestamp"]]
            .sort_values(by="timestamp", ascending=False)
            .groupby(["deviceid_index"])
            .head(topk)
        )
        train_df = self.merge_id(train_df)
        device_train_dict = (
            train_df.groupby("device_id")["product_id"]
            .apply(list)
            .to_dict()
        )
        return device_train_dict

    def get_test_dataset(self):
        prefix = f"testset/v{self.etl_version}/companyid={self.company_id}/etlid={self.etl_id}/checkpoint={self.dataset_checkpoint}"
        test_df = get_data_from_s3(self.etl_s3_bucket_name, prefix)
        test_df = self.merge_id(test_df)
        return test_df

    def get_model_from_string(self):
        session = Session()
        bucket = session.resource("s3").Bucket(
            self.ranker_s3_bucket_name
        )
        objects = bucket.objects.filter(
            Prefix="{}/{}/{}/{}".format(
                self.company_id, self.strategy_id, self.etl_id, self.model_id
            )
        )
        bs = b""
        for obj in objects:
            bs += obj.get()["Body"].read()
        buf = io.BytesIO(bs)
        ranker = lgb.Booster(model_str=buf.read().decode("UTF-8"))
        return ranker

    def filter_columns(self, cols):
        feature_columns = []
        for col in cols:
            valid = True
            for fs in self.filtering_string:
                if fs in col:
                    valid = False
            if valid:
                feature_columns.append(col)
        return feature_columns

    def get_product_meta_columns(self):
        client = boto3.client("glue", region_name="ap-northeast-2")
        tb = client.get_table(
            DatabaseName=self.etl_s3_bucket_name,
            Name="product_meta_v{}".format(self.etl_version),
        )["Table"]
        meta_columns = ["strategy_index"] + [
            c["Name"] for c in tb["StorageDescriptor"]["Columns"]
        ]
        return meta_columns

    def inference(self, df, topk):
        feature_columns = self.filter_columns(df.columns)
        meta_columns = self.get_product_meta_columns()
        categoricals = [
            idx for idx, c in enumerate(feature_columns) if c in meta_columns
        ]
        df[np.array(feature_columns)[categoricals]] = df[
            np.array(feature_columns)[categoricals]
        ].astype("float32")
        df["prediction"] = self.model.predict(df[feature_columns])
        df = (
            df[["device_id", "product_id", "prediction"]]
            .sort_values(by="prediction", ascending=False)
            .groupby(["device_id"])
            .head(topk)
        )
        user_rec_dict = (
            df.groupby("device_id")["product_id"]
            .apply(list)
            .to_dict()
        )
        return user_rec_dict
