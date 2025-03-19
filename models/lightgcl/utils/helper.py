import numpy as np
import torch
from torch.nn import functional as F

from models.helper import get_data_from_s3


def sparse_dropout(matrix, dropout_rate):
    if dropout_rate == 0.0:
        return matrix

    indices = matrix.indices()
    values = F.dropout(matrix.values(), p=dropout_rate)
    size = matrix.size()

    return torch.sparse.FloatTensor(indices, values, size)


def get_filtered_dataset_from_s3(
    bucket_name,
    etl_version,
    company_id,
    etl_id,
    dataset_checkpoint,
    filter_test_df=False,
    import_meta=True,
):
    df = get_data_from_s3(
        bucket_name=bucket_name,
        prefix="common_dataset/v{}/companyid={}/etlid={}/checkpoint={}".format(
            etl_version, company_id, etl_id, dataset_checkpoint
        ),
    )
    df = df[df["event_index"] == "3"]
    df = df.assign(timestamp=df.timestamp.dt.round("H"))
    df["prev_device"] = df["deviceid_model_index"].shift().fillna(0).astype(np.int32)
    df["prev_product"] = df["productid_model_index"].shift().fillna(0).astype(np.int32)
    df = df[
        ~(
            (df["deviceid_model_index"] == df["prev_device"])
            & (df["productid_model_index"] == df["prev_product"])
        )
    ]
    df = df.drop(columns=["prev_device", "prev_product"])

    raw_interacted_users = df.groupby("productid_model_index")[
        "deviceid_model_index"
    ].nunique()
    interacted_users = raw_interacted_users.reset_index().rename(
        columns={"deviceid_model_index": "interacted_users"}
    )
    df = df.merge(interacted_users, on="productid_model_index", how="left")
    df = df[df["interacted_users"] > 5]

    df["productid_model_index"] = (
        df["productid_model_index"].astype("category").cat.codes.values + 1
    )
    df["deviceid_model_index"] = (
        df["deviceid_model_index"].astype("category").cat.codes.values + 1
    )

    df = df.sort_values(
        by=["deviceid_model_index", "occurence"], ascending=[True, True]
    ).reset_index(drop=True)
    tmp_occ = df.groupby("deviceid_model_index")["productid_model_index"].count()
    df_fixed_occ = (
        tmp_occ.apply(lambda x: np.arange(x) + 1)
        .explode("productid_model_index")
        .rename("df_fixed_occ")
    ).to_numpy()
    df["occurence"] = df_fixed_occ

    df = df.merge(
        df.groupby("productid_model_index")["deviceid_model_index"]
        .nunique()
        .reset_index()
        .rename(columns={"deviceid_model_index": "product_device_cnt"}),
        on="productid_model_index",
        how="left",
    )
    df = df.merge(
        df.groupby("deviceid_model_index")["productid_model_index"]
        .nunique()
        .reset_index()
        .rename(columns={"productid_model_index": "device_product_cnt"}),
        on="deviceid_model_index",
        how="left",
    )
    df["norm"] = 1 / (
        df["device_product_cnt"].pow(1.0 / 2) * df["product_device_cnt"].pow(1.0 / 2)
    )

    meta_df = get_data_from_s3(
        bucket_name=bucket_name,
        prefix="product_meta/v{}/companyid={}".format(etl_version, company_id),
    )
    meta_df = (
        meta_df.sort_values("timestamp", ascending=False)
        .drop_duplicates(subset=["productid_index"])
        .sort_index()[["productid_index", "productname"]]
        .dropna()
    )
    tmp_df = df[["productid", "productid_index"]].drop_duplicates()
    product2pname = meta_df.merge(tmp_df, on=["productid_index"], how="left").dropna()
    product2pname = (
        product2pname[["productid", "productname"]]
        .set_index("productid")
        .to_dict()["productname"]
    )

    device2idx = (
        df[["deviceid", "deviceid_model_index"]]
        .drop_duplicates()
        .set_index("deviceid")
        .to_dict()["deviceid_model_index"]
    )
    idx2device = {idx: deviceid for deviceid, idx in device2idx.items()}
    product2idx = (
        df[["productid", "productid_model_index"]]
        .drop_duplicates()
        .set_index("productid")
        .to_dict()["productid_model_index"]
    )
    idx2product = {idx: productid for productid, idx in product2idx.items()}
    num_device, num_product = (
        df["deviceid_model_index"].nunique() + 1,
        df["productid_model_index"].nunique() + 1,
    )
    train_df = df.query("validation == False")
    test_df = df.query("validation == True")
    device_train_dict = (
        df.sort_values(by="occurence")
        .groupby("deviceid_model_index")["productid_model_index"]
        .apply(list)
        .to_dict()
    )
    device_test_dict = (
        test_df.groupby("deviceid_model_index")["productid_model_index"]
        .apply(list)
        .to_dict()
    )
    # product_train_dict = (
    #     train_df.groupby("productid_model_index")["deviceid_model_index"]
    #     .apply(list)
    #     .to_dict()
    # )

    # test_df["coldness"] = test_df.apply(lambda row: (row["occurence"] == 1) * -1, axis=1)
    # device_test_dict = test_df.set_index(["deviceid_model_index", "occurence"])["coldness"].to_dict()

    data_dict = {}
    data_dict["df"] = df
    data_dict["train_df"] = train_df
    data_dict["test_df"] = test_df
    data_dict["num_device"] = num_device
    data_dict["num_product"] = num_product
    data_dict["device2idx"] = device2idx
    data_dict["idx2device"] = idx2device
    data_dict["product2idx"] = product2idx
    data_dict["product2pname"] = product2pname
    data_dict["idx2product"] = idx2product
    data_dict["device_train_dict"] = device_train_dict
    # data_dict["product_train_dict"] = product_train_dict
    data_dict["device_test_dict"] = device_test_dict

    return data_dict
