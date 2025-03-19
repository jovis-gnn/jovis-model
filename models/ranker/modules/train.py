import os
import json
import boto3
import pickle
from io import BytesIO
from os.path import join, dirname

import numpy as np
import pandas as pd
import lightgbm as lgb
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMRanker
from dotenv import load_dotenv


from models.helper import init_logger
from models.ranker.utils.helper import get_filtered_dataset_from_s3


class RankerTrainer:
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
        self.ranker_dataset = get_filtered_dataset_from_s3(self.config)

        self.model = LGBMRanker(
            objective=self.config["objective"],
            num_leaves=self.config["num_leaves"],
            n_estimators=self.config["num_iterations"],
            boosting_type=self.config["boosting_type"],
            verbosity=self.config["verbosity"],
            metric=self.config["metric"],
            early_stopping_round=self.config["early_stopping_round"],
            importance_type="gain",
            device="cpu",
        )
        session = boto3.Session()
        self.client = session.client(
            "s3",
            aws_access_key_id=self.config["aws_access_key_id"],
            aws_secret_access_key=self.config["aws_secret_access_key"],
        )

    def save_mapping(self, mappers):
        mapping_b = pickle.dumps(mappers)
        self.client.put_object(
            Body=mapping_b,
            Bucket=self.config["model_s3_bucket_name"],
            Key="{}/{}/{}/{}/mapping.pkl".format(
                self.config["company_id"],
                "ranker",
                self.config["etl_id"],
                self.config["model_id"],
            ),
        )

    def save_checkpoint(self):
        self.model.booster_.save_model("/tmp/ranker.txt")
        self.client.put_object(
            Body=open("/tmp/ranker.txt", "rb"),
            Bucket=self.config["model_s3_bucket_name"],
            Key="{}/{}/{}/{}/ranker.txt".format(
                self.config["company_id"],
                "ranker",
                self.config["etl_id"],
                self.config["model_id"],
            ),
        )

    def save_importance(self, feature_importance, feature_columns):
        df = pd.DataFrame(
            zip(feature_columns, feature_importance), columns=["feature", "importance"]
        )
        df = df.sort_values(by="importance", ascending=False)

        plt.style.use("ggplot")
        fig, ax = plt.subplots(1)
        fig.set_size_inches(10, 0.25 * len(feature_importance))
        ax.set_title("LightGBM feature importance")
        ax.set_xlabel("importance")
        ax.set_ylabel("feature")
        ax.set_yticklabels(df["feature"])
        sns.barplot(
            data=df, x="importance", y="feature", ax=ax, palette="blend:#7AB,#EDA"
        )
        for i in ax.containers:
            ax.bar_label(
                i,
            )
        plt.close(fig)

        fig.savefig("/tmp/tmp.png", bbox_inches="tight", dpi=100)
        img = Image.open("/tmp/tmp.png")

        buffer = BytesIO()
        img.save(buffer, format="WEBP")
        buffer.seek(0)
        self.client.upload_fileobj(
            Fileobj=buffer,
            Bucket=self.config["model_s3_bucket_name"],
            Key="{}/{}/{}/{}/importance.webp".format(
                self.config["company_id"],
                "ranker",
                self.config["etl_id"],
                self.config["model_id"],
            ),
        )

        return img

    def train(self):
        filter_columns = [
            "deviceid",
            "productid",
            "timestamp",
            "event_timestamp",
            "label",
            "group_id",
            "validation",
        ]
        categoricals = self.config["categoricals"]

        mappers = {}
        for c in self.config["categoricals"]:
            self.ranker_dataset[f"{c}_index"] = (
                self.ranker_dataset[c].astype("category").cat.codes.values
            )
            self.ranker_dataset[f"{c}_index"] = self.ranker_dataset[
                f"{c}_index"
            ].replace(-1, np.nan)

            mapper = (
                self.ranker_dataset[[c, f"{c}_index"]]
                .drop_duplicates()
                .set_index(c)
                .to_dict()
            )
            mappers.update(mapper)

        self.save_mapping(mappers)

        feature_columns = [
            c
            for c in self.ranker_dataset.columns
            if c not in filter_columns and c not in self.config["categoricals"]
        ]
        categoricals = [f"{c}_index" for c in self.config["categoricals"]]
        self.ranker_dataset[categoricals] = self.ranker_dataset[categoricals].astype(
            "float32"
        )

        trainset = self.ranker_dataset.query("validation == False").sort_values(
            by="group_id"
        )
        train_X, train_y, train_basket = (
            trainset[feature_columns],
            trainset["label"],
            trainset.groupby("group_id")["group_id"].count(),
        )
        testset = self.ranker_dataset.query("validation == True").sort_values(
            by="group_id"
        )
        test_X, test_y, test_basket = (
            testset[feature_columns],
            testset["label"],
            testset.groupby("group_id")["group_id"].count(),
        )

        self.model.fit(
            X=train_X,
            y=train_y,
            group=train_basket,
            eval_set=[(test_X, test_y)],
            eval_group=[test_basket],
            eval_at=12,
            feature_name=list(train_X.columns),
            categorical_feature=categoricals,
            callbacks=[lgb.log_evaluation(1)],
        )
        self.save_checkpoint()

        self.logger.info(
            f"Train finished. best iteration : {self.model.best_iteration_}"
        )
        self.save_importance(
            feature_importance=self.model.feature_importances_,
            feature_columns=feature_columns,
        )

    def run(self, topk=12):
        self.train()


if __name__ == "__main__":
    with open("../configs/config.json", "rb") as f:
        configs = json.load(f)
    trainer = RankerTrainer(**configs)
    trainer.run()
