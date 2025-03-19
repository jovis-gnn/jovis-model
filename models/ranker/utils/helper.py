import io

import boto3
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt


from models.helper import init_logger


def get_filtered_dataset_from_s3(config):
    logger = init_logger("preprocessor")
    session = boto3.Session()
    client = session.client(
        "s3",
        aws_access_key_id=config["aws_access_key_id"],
        aws_secret_access_key=config["aws_secret_access_key"],
    )
    prefix = f"ranker_dataset/v{config['etl_version']}/companyid={config['company_id']}/etlid={config['etl_id']}"
    res = client.list_objects_v2(
        Bucket=config["etl_s3_bucket_name"],
        Prefix=prefix,
    )
    keys = []
    if "Contents" in res:
        for r in res["Contents"]:
            keys.append(r["Key"])
    else:
        logger.info(
            f"There's no object keys for bucket : {config['etl_s3_bucket_name']}, prefix : {prefix}"
        )
    dfs = []
    for key in keys:
        res = client.get_object(Bucket=config["etl_s3_bucket_name"], Key=key)
        tmp_df = pd.read_parquet(io.BytesIO(res["Body"].read()))
        dfs.append(tmp_df)
    df = pd.concat(dfs, ignore_index=True)

    # generate group id
    df["group_id"] = df.groupby(["deviceid", "event_timestamp"]).ngroup()

    # train / val split
    df.loc[df["event_timestamp"] == df["event_timestamp"].max(), "validation"] = True
    df.loc[df["event_timestamp"] != df["event_timestamp"].max(), "validation"] = False

    return df


def save_importance(feature_importance, feature_columns):
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
    sns.barplot(data=df, x="importance", y="feature", ax=ax, palette="blend:#7AB,#EDA")
    for i in ax.containers:
        ax.bar_label(
            i,
        )
    plt.close(fig)

    fig.savefig("/tmp/tmp.png", bbox_inches="tight", dpi=100)
    img = Image.open("/tmp/tmp.png")
    return img
