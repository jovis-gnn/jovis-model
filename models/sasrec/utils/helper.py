import boto3
import numpy as np

from models.helper import (
    init_logger,
    get_target_interactions,
)


def get_filtered_dataset_from_s3(config):
    logger = init_logger("preprocessor")

    target_days = config["target_days"]
    company_mapper = {
        "default": {
            "user_limit": 0,
            "item_limit": 1,
            "events": [
                "detail_page_view",
                "purchase_click",
                "recommend_click",
                "add_to_wishlist",
            ],
        },
        "01HZZT74A7XKQP9G69CHQH69HK": {
            "user_limit": 5,
            "item_limit": 1,
            "events": ["detail_page_view"],
        },
        "01J4P16VSA9JB9RW010VXA05C2": {
            "user_limit": 0,
            "item_limit": 1,
            "events": [
                "detail_page_view",
                "purchase_click",
                "recommend_click",
                "add_to_wishlist",
            ],
        },
    }
    session = boto3.Session()
    client = session.client(
        "s3",
        aws_access_key_id=config["aws_access_key_id"],
        aws_secret_access_key=config["aws_secret_access_key"],
    )

    # load target interactions
    logger.info("load target interactions")
    df = get_target_interactions(client, config, target_days)

    if company_mapper.get(config["company_id"]):
        target_events = company_mapper[config["company_id"]]["events"]
        target_user_limit = company_mapper[config["company_id"]]["user_limit"]
        target_item_limit = company_mapper[config["company_id"]]["item_limit"]
    else:
        target_events = company_mapper["default"]["events"]
        target_user_limit = company_mapper["default"]["user_limit"]
        target_item_limit = company_mapper["default"]["item_limit"]

    # event index load & filtering
    logger.info("event filtering")
    df = df[df["event"].isin(target_events)]
    df = df.drop(["suggestqueryid", "suggestproductids"], axis=1)

    # filter dataset by user / item counts
    logger.info("filter dataset by user / item counts")
    raw_interacted_users = df.groupby("productid")["deviceid"].nunique()
    interacted_users = raw_interacted_users.reset_index().rename(
        columns={"deviceid": "interacted_users"}
    )
    df = df.merge(interacted_users, on="productid", how="left")
    df = df[df["interacted_users"] > target_user_limit]

    raw_interacted_prods = df.groupby("deviceid")["productid"].nunique()
    interacted_prods = raw_interacted_prods.reset_index().rename(
        columns={"productid": "interacted_prods"}
    )
    df = df.merge(interacted_prods, on="deviceid", how="left")
    df = df[df["interacted_prods"] > target_item_limit]

    # filter out consecutive items.
    logger.info("filter out consecutive items")
    df = df.sort_values(
        by=["deviceid", "timestamp"], ascending=[True, True]
    ).reset_index(drop=True)
    df["prev_device"] = df["deviceid"].shift().fillna("None")
    df["prev_product"] = df["productid"].shift().fillna("None")
    df = df[
        ~(
            (df["deviceid"] == df["prev_device"])
            & (df["productid"] == df["prev_product"])
        )
    ]
    print(len(df))
    df = df.drop(columns=["prev_device", "prev_product"], axis=1)

    # create index for model
    logger.info("create index for model")
    df["productid_model_index"] = (
        df["productid"].astype("category").cat.codes.values + 1
    )
    df["deviceid_model_index"] = df["deviceid"].astype("category").cat.codes.values + 1

    # create occurence
    logger.info("create occurence")
    tmp_occ = df.groupby("deviceid_model_index")["productid_model_index"].count()
    df["occurence"] = (
        tmp_occ.apply(lambda x: np.arange(x) + 1)
        .explode("productid_model_index")
        .rename("df_fixed_occ")
        .to_numpy()
    )

    # merge original device, product ids
    # logger.info("merge original device, product ids")
    # deviceid_df = get_index_features(client, config, "deviceid")
    # deviceid_df = deviceid_df.rename(
    #     columns={"feature_name": "deviceid", "feature_index": "deviceid_index"}
    # )
    # df = pd.merge(df, deviceid_df, on=["deviceid_index"], how="left")
    # productid_df = get_index_features(client, config, "productid")
    # productid_df = productid_df.rename(
    #     columns={"feature_name": "productid", "feature_index": "productid_index"}
    # )
    # df = pd.merge(df, productid_df, on=["productid_index"], how="left")
    df = df[
        [
            "deviceid",
            "deviceid_model_index",
            "productid",
            "productid_model_index",
            "occurence",
            "timestamp",
        ]
    ]

    # split train / val by ratio (0.1)
    logger.info("split train / val by ratio (0.1)")
    df = df.sort_values(by="timestamp", ascending=False)
    top_10_percent_count = int(len(df) * 0.1)
    df["validation"] = 0
    df.loc[df.head(top_10_percent_count).index, "validation"] = 1
    train_df = df.query("validation == 0").reset_index(drop=True)
    test_df = df.query("validation == 1").reset_index(drop=True)

    # create test occurence
    logger.info("create test occurence")
    test_occ = test_df.groupby("deviceid_model_index")["productid_model_index"].count()
    test_df["test_occurence"] = (
        test_occ.apply(lambda x: np.arange(x) + 1)
        .explode("productid_model_index")
        .rename("df_fixed_occ")
        .to_numpy()
    )

    # create mappings
    logger.info("create mappings")
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
    device_train_dict = (
        df.sort_values(by="occurence")
        .groupby("deviceid_model_index")["productid_model_index"]
        .apply(list)
        .to_dict()
    )
    product_train_dict = (
        train_df.groupby("productid_model_index")["deviceid_model_index"]
        .apply(list)
        .to_dict()
    )
    device_test_dict = (
        test_df.groupby("deviceid_model_index")["productid_model_index"]
        .apply(list)
        .to_dict()
    )

    # filter reclick
    test_df["reclick"] = test_df.apply(
        lambda row: int(
            row["productid_model_index"]
            in device_train_dict[row["deviceid_model_index"]][: row["occurence"] - 1][
                -5:
            ]
        ),
        axis=1,
    )
    test_df = test_df[test_df["reclick"] == 0]

    data_dict = {}
    data_dict["df"] = df
    data_dict["train_df"] = train_df
    data_dict["test_df"] = test_df
    data_dict["num_device"] = num_device
    data_dict["num_product"] = num_product
    data_dict["device2idx"] = device2idx
    data_dict["idx2device"] = idx2device
    data_dict["product2idx"] = product2idx
    data_dict["idx2product"] = idx2product

    data_dict["device_train_dict"] = device_train_dict
    data_dict["product_train_dict"] = product_train_dict
    data_dict["device_test_dict"] = device_test_dict

    data_dict["num_meta"] = []
    data_dict["meta_table"] = None
    data_dict["productname_table"] = None
    data_dict["tagger_table"] = None

    return data_dict
