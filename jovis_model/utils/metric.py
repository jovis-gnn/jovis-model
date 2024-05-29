from typing import List

import sklearn
import numpy as np


def macro_f1(
    targets: np.ndarray,
    preds: np.ndarray,
    label_list: List[str] = None,
    zero_division="warn",
):
    if label_list is not None:
        label_list = list(range(len(label_list)))
    return sklearn.metrics.f1_score(targets, preds, labels=label_list, average="macro")
