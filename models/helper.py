import io
import os
import pickle
import logging
import multiprocessing
from functools import partial
from datetime import datetime, timedelta

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm


def init_logger(logger_name, level="INFO"):
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s: %(message)s",
        datefmt="%I:%M:%S:%p",
        level=logging.INFO,
    )
    logger = logging.getLogger(logger_name)
    if level != "INFO":
        logger.setLevel(logging.ERROR)
    return logger


def check_s3_prefix_exists(client, bucket, prefix):
    res = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = []
    if "Contents" in res:
        for r in res["Contents"]:
            keys.append(r["Key"])
    return keys


def get_index_features(client, config, feature_name):
    prefix_format = "index_features/v{}/companyid={}/feature={}"
    res = client.list_objects_v2(
        Bucket=config["etl_s3_bucket_name"],
        Prefix=prefix_format.format(
            config["etl_version"], config["company_id"], feature_name
        ),
    )
    target_keys = []
    if "Contents" in res:
        for r in res["Contents"]:
            target_keys.append(r["Key"])

    dfs = []
    for key in target_keys:
        res = client.get_object(Bucket=config["etl_s3_bucket_name"], Key=key)
        tmp_df = pd.read_parquet(io.BytesIO(res["Body"].read()))
        dfs.append(tmp_df)

    df = pd.concat(dfs, ignore_index=True)
    return df


def get_target_interactions(client, config, target_days=14):
    cur = datetime.utcnow()
    bound = cur - timedelta(days=target_days)

    target_keys = []
    while cur >= bound:
        params = [
            config["etl_s3_bucket_name"],
            config["etl_version"],
            config["company_id"],
        ] + list(map(int, datetime.strftime(cur, "%Y-%m-%d").split("-")))
        check_prefix = f"interactions/v{config['etl_version']}/companyid={config['company_id']}/year={params[3]}/month={params[4]}/day={params[5]}"
        tmp_keys = check_s3_prefix_exists(
            client, config["etl_s3_bucket_name"], check_prefix
        )
        target_keys += tmp_keys
        cur -= timedelta(days=1)

    dfs = []
    for key in target_keys:
        res = client.get_object(Bucket=config["etl_s3_bucket_name"], Key=key)
        tmp_df = pd.read_parquet(io.BytesIO(res["Body"].read()))
        dfs.append(tmp_df)

    df = pd.concat(dfs, ignore_index=True)

    return df


def get_product_meta(client, config):
    prefix = f"product_meta/v{config['etl_version']}/companyid={config['company_id']}"
    res = client.list_objects_v2(Bucket=config["etl_s3_bucket_name"], Prefix=prefix)
    keys = []
    if "Contents" in res:
        for r in res["Contents"]:
            keys.append(r["Key"])

    dfs = []
    for k in keys:
        res = client.get_object(Bucket=config["etl_s3_bucket_name"], Key=k)
        tmp_df = pd.read_parquet(io.BytesIO(res["Body"].read()))
        dfs.append(tmp_df)

    df = pd.concat(dfs, ignore_index=True)

    return df


def argument_parser(kwargs):
    res = {}
    target_arguments = [
        "interactions_columns",
        "product_meta_columns",
        "train_strategies",
        "test_strategies",
        "filtering_string",
    ]
    for k, v in kwargs.items():
        if k in target_arguments:
            if len(v.split(",")[0].split(":")) > 1:
                tmp = {}
                for i in v.split(","):
                    k_, v_ = i.split(":")
                    tmp[k_] = v_
                res[k] = tmp
            else:
                res[k] = v.split(",")
        else:
            res[k] = v
    return res


def download_(data, save_path):
    session = boto3.Session()
    client = session.client("s3", config=Config(signature_version=UNSIGNED))

    product_id, url = data
    bucket, *key = url.split("/")[2:]
    key = "/".join(key)
    target_path = os.path.join(save_path, str(product_id) + ".jpg")
    if os.path.isfile(target_path):
        url = url
    else:
        img = client.get_object(Bucket=bucket, Key=key)
        img = img["Body"].read()
        with open(target_path, "wb") as f:
            f.write(img)


def download_product_images(image_df, save_path):
    load_dotenv("../.env")
    image_df = image_df.dropna()
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

    data = image_df.values
    pool = multiprocessing.Pool(processes=8)
    for _ in tqdm(pool.imap_unordered(partial(download_, save_path=save_path), data)):
        pass
    pool.close()
    pool.join()


if __name__ == "__main__":
    with open("/data/shared2/musinsa_snap/snap_list.pkl", "rb") as f:
        tmp_data = pickle.load(f)
    data = []
    for snap in tmp_data:
        row = [snap["snap_id"], snap["style_name"], snap["tpo_name"]]
        for p in snap["products"]:
            p_row = row + [
                p["product_id"],
                p["product_name"],
                p["category_name"],
                p["s3_image_url"],
            ]
            data.append(p_row)

    outfit_df = pd.DataFrame(
        data,
        columns=[
            "outfit_id",
            "style_name",
            "tpo_name",
            "product_id",
            "product_name",
            "product_category",
            "product_s3_image_url",
        ],
    )
    image_df = (
        outfit_df[["product_id", "product_s3_image_url"]].dropna().drop_duplicates()
    )
    print("image_df loaded")
    download_product_images(
        image_df=image_df, save_path="/home/omnious/workspace/jovis/musinsa_snap_images"
    )
