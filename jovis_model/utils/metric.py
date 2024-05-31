from typing import List

from sklearn.metrics import f1_score
import numpy as np


def macro_f1(
    targets: np.ndarray,
    preds: np.ndarray,
    label_list: List[str] = None,
    zero_division="warn",
):
    if label_list is not None:
        label_list = list(range(len(label_list)))
    return f1_score(targets, preds, labels=label_list, average="macro")
