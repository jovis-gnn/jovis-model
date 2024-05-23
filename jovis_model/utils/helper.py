import os
import json
from urllib import request
import logging
import multiprocessing
from tqdm import tqdm
from typing import List
from functools import partial


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


if __name__ == "__main__":
    file_path = "/home/omnious/workspace/jovis/jovis-model/jovis_model/_db/llm/multimodal_test/export_data_skb.json"
    save_path = "/home/omnious/workspace/jovis/jovis-model/jovis_model/_db/llm/multimodal_test/images/export_data_skb"
    with open(file_path, "r") as f:
        json_loaded = json.load(f)
    data = []
    for pid, meta in json_loaded.items():
        data.append([pid, meta["s3_url"]])
    download_images_from_url(data, save_path)
