import argparse
import json
import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sentence_models import ProdNameModel


def inference(
    model,
    prod_names: str | list[str],
    batch_size: int,
) -> torch.Tensor:
    dataloader = DataLoader(prod_names, batch_size=batch_size, shuffle=False)

    all_preds = []
    for batch in tqdm(
        iterable=dataloader,
        desc="Performing inference",
        total=len(dataloader),
    ):
        topk_preds = model(prod_names=prod_names)
        topk_preds = topk_preds.detach().cpu()
        all_preds.extend(topk_preds)

    all_preds = torch.stack(all_preds)

    return all_preds


if __name__ == "__main__":
    logger = logging.getLogger()
    msg_format = "[%(asctime)s - %(levelname)s - %(filename)s: %(lineno)d (%(funcName)s)] %(message)s"
    logging.basicConfig(
        format=msg_format,
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
        ],
    )

    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(current_dir, "..", "configs")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        default="topk",
        type=str,
        choices=["embedding", "topk"],
        help="Whether or not we want to extract embeddings or do top-k inference.",
    )

    args = parser.parse_args()

    logger.info("Loading config for %s mode", args.mode)
    if args.mode == "embedding":
        config_file = os.path.join(config_dir, "prod_name_embedding_config.json")
    elif args.mode == "topk":
        config_file = os.path.join(config_dir, "prod_name_topk_config.json")
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    with open(file=config_file) as f:
        config = json.load(fp=f)

    model_data_dir = os.path.join(current_dir, "..", "data")
    assert os.path.exists(model_data_dir)

    if config["use_pooler_output"]:
        prodname2emb_file = config["prodname2emb_file"].format(256)
    else:
        prodname2emb_file = config["prodname2emb_file"].format(768)

    model_args = {
        "model_data_dir": model_data_dir,
        "mode": args.mode,
        "use_pooler_output": config["use_pooler_output"],
        "item2fine_ctgr_file": config["item2fine_ctgr_file"],
        "fine_ctgr2prodname_file": config["fine_ctgr2prodname_file"],
        "prodname2emb_file": prodname2emb_file,
        "prodname2prodid_file": config["prodname2prodid_file"],
        "model_ckpt_dir": config["model_ckpt_dir"],
    }
    model = ProdNameModel(
        logger=logger,
        **model_args,
    )

    topk_preds = inference(
        model=model,
        prod_names=[
            "two-way embroidered padding jumper_black",
            "black sweater",
            "blue summer jeans",
        ],
        batch_size=config["batch_size"],
    )
