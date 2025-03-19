import json
import logging
import os
import pickle

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from sentence_models import ProdNameModel


def create_embeddings(
    model,
    data_file: str,
    batch_size: int,
) -> list[int]:
    if data_file.endswith(".csv"):
        data = pd.read_csv(data_file)
        prod_names = data["productname"].tolist()
    else:
        raise NotImplementedError

    all_embs = {}

    dataloader = DataLoader(prod_names, batch_size=batch_size, shuffle=False)
    for batch in tqdm(
        iterable=dataloader,
        desc="Getting embeddings",
        total=len(dataloader),
    ):
        embs = model(prod_names=batch)
        embs = embs.detach().cpu()
        prodname2emb = {name: embs[idx] for idx, name in enumerate(batch)}
        all_embs.update(prodname2emb)

    return all_embs


if __name__ == "__main__":
    logger = logging.getLogger()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(current_dir, "..", "configs")

    logger.info("Loading config for create_embeddings mode.")
    config_file = os.path.join(config_dir, "prod_name_create_embeddings_config.json")

    with open(file=config_file) as f:
        config = json.load(fp=f)

    model_args = {
        "model_data_dir": config["model_data_dir"],
        "mode": "create_embeddings",
        "use_pooler_output": config["use_pooler_output"],
    }
    model = ProdNameModel(
        logger=logger,
        **model_args,
    )

    embs = create_embeddings(
        model=model,
        data_file=config["source_data_file"],
        batch_size=config["batch_size"],
    )

    if config["use_pooler_output"]:
        save_file = os.path.join(
            config["model_data_dir"],
            "prodname2embeddings_256.pickle",
        )
    else:
        save_file = os.path.join(
            config["model_data_dir"],
            "prodname2embeddings_768.pickle",
        )

    logger.info("Saving embeddings in %s", save_file)

    with open(file=save_file, mode="wb") as f:
        pickle.dump(obj=embs, file=f)
