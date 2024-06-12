import os
import json
from urllib import request
import logging
import multiprocessing
from tqdm import tqdm
from typing import List
from functools import partial

import numpy as np
import faiss


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


def download_(data, save_path):
    productid, url = data
    target_path = os.path.join(save_path, str(productid) + ".JPEG")
    if os.path.isfile(target_path):
        url = url
    else:
        try:
            request.urlretrieve(url, target_path)
        except Exception:
            try:
                request.urlretrieve(url, target_path)
            except Exception as e:
                print(f"productid {productid} got error : {e}")


def download_images_from_url(data: List[List[str]], save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

    pool = multiprocessing.Pool(processes=24)
    for _ in tqdm(pool.imap_unordered(partial(download_, save_path=save_path), data)):
        pass
    pool.close()
    pool.join()


def build_faiss_index(
    embeddings: List[List[float]],
    save_path: str,
    save_name: str,
    pids: List[str] = None,
):
    embeddings = np.array(embeddings).astype(np.float32)
    faiss.normalize_L2(embeddings)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index = faiss.IndexIDMap2(faiss_index)
    faiss_index.add_with_ids(embeddings, np.arange(len(embeddings)))
    faiss.write_index(faiss_index, os.path.join(save_path, f"{save_name}.index"))

    if pids:
        with open(os.path.join(save_path, f"{save_name}_map.json"), "w") as f:
            json.dump({idx: pid for idx, pid in enumerate(pids)}, f)


if __name__ == "__main__":
    file_path = "/home/omnious/workspace/jovis/jovis-model/jovis_model/_db/llm/multimodal_test/export_data_skb.json"
    save_path = "/home/omnious/workspace/jovis/jovis-model/jovis_model/_db/llm/multimodal_test/images/export_data_skb"
    with open(file_path, "r") as f:
        json_loaded = json.load(f)
    data = []
    for pid, meta in json_loaded.items():
        data.append([pid, meta["s3_url"]])
    download_images_from_url(data, save_path)
