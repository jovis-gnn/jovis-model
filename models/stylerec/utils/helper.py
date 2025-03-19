import json
from collections import defaultdict

import torch

# from models.helper import init_logger


def get_polyvore_dataset(split):
    outfit_data = json.load(
        open(f"/home/omnious/workspace/jovis/polyvore_outfits/nondisjoint/{split}.json")
    )
    meta_data = json.load(
        open(
            "/home/omnious/workspace/jovis/polyvore_outfits/polyvore_item_metadata.json"
        )
    )
    item_ids = set()
    categories = set()
    item_id2category = {}
    item_id2desc = {}
    category2item_ids = {}
    outfit_id2item_id = {}
    for outfit in outfit_data:
        outfit_id = outfit["set_id"]
        for item in outfit["items"]:
            item_id = item["item_id"]
            item_ids.add(item_id)
            category = "<" + meta_data[item_id]["semantic_category"] + ">"
            categories.add(category)
            item_id2category[item_id] = category
            if category not in category2item_ids:
                category2item_ids[category] = set()
            category2item_ids[category].add(item_id)
            desc = meta_data[item_id]["title"]
            if not desc:
                desc = meta_data[item_id]["url_name"]
            item_id2desc[item_id] = desc.replace("\n", "").strip().lower()
            outfit_id2item_id[f"{outfit['set_id']}_{item['index']}"] = item_id

    categories = list(categories)

    cp_inputs = []
    with open(
        f"/home/omnious/workspace/jovis/polyvore_outfits/nondisjoint/compatibility_{split}.txt",
        "r",
    ) as f:
        cp_data = f.readlines()
        for d in cp_data:
            cp_inputs.append(d.split())

    outfit2product = defaultdict(dict)
    for inputs in cp_inputs:
        label, *product_ids = inputs
        outfit_id = product_ids[0].split("_")[0]
        product_ids = [outfit_id2item_id[i] for i in product_ids]
        product_categories = [item_id2category.get(i, "") for i in product_ids]
        product_descs = [item_id2desc.get(i, "") for i in product_ids]
        outfit2product[outfit_id]["label"] = label
        outfit2product[outfit_id]["product_ids"] = product_ids
        outfit2product[outfit_id]["texts"] = product_categories
        outfit2product[outfit_id]["desc"] = product_descs

    return outfit2product


def get_outfit_dataset_from_db(config):
    # logger = init_logger("preprocessor")

    data_dict = {}
    if config["dataset"] == "polyvore":
        train_outfit2product = get_polyvore_dataset("train")
        valid_outfit2product = get_polyvore_dataset("valid")

        data_dict["train_outfit2product"] = train_outfit2product
        data_dict["valid_outfit2product"] = valid_outfit2product
    else:
        pass

    return data_dict


def stack_tensors(mask, tensor):
    B, S = mask.shape
    mask = mask.reshape(-1)
    s = list(tensor.shape)
    tensor = tensor.reshape([s[0] * s[1]] + s[2:])
    tensor = tensor[~mask]
    return tensor


def unstack_tensors(mask, tensor):
    B, S = mask.shape
    mask = mask.reshape(-1)
    new_shape = [B * S] + list(tensor.shape)[1:]
    device = tensor.device if tensor.device.type == "cuda" else torch.device("cpu")
    new_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=device)
    new_tensor[~mask] = tensor
    new_tensor = new_tensor.reshape([B, S] + list(tensor.shape)[1:])
    return new_tensor


def stack_dict(batch):
    mask = batch.get("padding_mask", None)
    stacked_batch = {
        key: stack_tensors(mask, value) if key != "padding_mask" else value
        for key, value in batch.items()
    }
    return stacked_batch


def unstack_dict(batch):
    mask = batch.get("padding_mask", None)
    stacked_batch = {
        key: unstack_tensors(mask, value) if key != "padding_mask" else value
        for key, value in batch.items()
    }
    return stacked_batch
