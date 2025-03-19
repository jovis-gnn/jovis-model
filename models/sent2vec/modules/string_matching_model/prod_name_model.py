import json
import re

import pandas as pd

from .config import SPACE_CHARS
from .prod_name_utils import (
    find_latest_entity,
    is_end_of_word,
    is_korean,
    is_singular_form,
    is_subword,
    remove_black_watch,
    remove_brackets_and_contents,
    remove_tie_dye,
)


class StringMatchingModel:
    def __init__(self) -> None:
        self.item2ctgr: dict[str, str] = {}

    def load(self, fp: str) -> None:
        """Load appropriate info for model.

        In the case of string matching, this will be a dict.
        Perform lowercasing and whitespace stripping on items.
        Sort item2ctgr according to entity length in ascending order.
        """
        print(f"Load string matching dict from {fp}")
        with open(file=fp) as f:
            item2ctgr = json.load(fp=f)

        item2ctgr_pairs = [(i.lower(), c) for i, c in item2ctgr.items()]
        item2ctgr_pairs = sorted(item2ctgr_pairs, key=lambda x: len(x[0]), reverse=True)
        item2ctgr = {i: c for i, c in item2ctgr_pairs}

        self.item2ctgr = item2ctgr

    def preprocess_prod_name(self, prod_name: str) -> str:
        """
        Performs basic pre-processing on string.

        Pre-processing:
            1. Lowercase and strip whitespace.
            2. Remove brackets and their contents.
            3. Remove `tie-dye` samples.
        """
        if pd.isnull(prod_name):
            return ""

        prod_name = prod_name.lower().strip()
        prod_name = remove_brackets_and_contents(prod_name)
        prod_name = remove_tie_dye(prod_name)
        return prod_name

    def match_name(self, prod_name: str):
        if not prod_name:
            return ""

        prod_name = self.preprocess_prod_name(prod_name)

        label2starts = {}
        for item, ctgr in self.item2ctgr.items():
            if not item:
                continue

            if ctgr == "WATCHES":
                prod_name = remove_black_watch(prod_name)

            if ctgr == "SOCKS":
                if (
                    "레드삭스" in prod_name
                    or "삭스 블루" in prod_name
                    or "삭스블루" in prod_name
                    or "사은품 양말" in prod_name
                    or "사은품양말" in prod_name
                    or "부츠삭스" in prod_name
                ):
                    continue

            matches = re.finditer(item, prod_name)
            for match in matches:
                start = match.start()
                end = match.end()

                if item == "티":
                    if prod_name[start + 1 :].startswith("셔츠"):  # ex) '티셔츠'.
                        pass
                    elif start != 0:
                        if end == len(prod_name):  # ex) '긴 소매티'
                            pass
                        elif prod_name[end] == " ":  # ex) '긴 소매티 (2022)'
                            pass
                    else:
                        continue
                elif item == "t":
                    if start == 0:
                        try:
                            if prod_name[start + 1] not in SPACE_CHARS:
                                continue
                        except (
                            IndexError
                        ) as err:  # Cases like `타이다이T` are processed to `t` which were missed previously.
                            print(f"Got {err} for {prod_name}.")
                            pass
                    elif start != 0:
                        if end == len(prod_name):
                            pass
                        elif (
                            prod_name[end] not in SPACE_CHARS
                        ):  # The "t" has to be on its own.
                            continue

                        if prod_name[start - 1] not in SPACE_CHARS:
                            continue
                else:
                    if is_korean(item):
                        if not is_end_of_word(prod_name, end):
                            continue
                    else:
                        if is_subword(prod_name, [start, end]):
                            if not is_singular_form(prod_name, end):
                                continue

                try:
                    label2starts[ctgr].add(start)
                except KeyError:
                    label2starts[ctgr] = set([start])

        if label2starts:
            result, label_idx = find_latest_entity(label2starts)

            if len(label2starts) != 1:
                if "with" in prod_name[:label_idx]:
                    label2starts_filt = {
                        label: starts
                        for label, starts in label2starts.items()
                        if label != result
                    }
                    result, _ = find_latest_entity(label2starts_filt)
        else:
            result = ""

        return result

    def forward(self, s: str):
        return self.match_name(s)

    def __call__(self, s: str) -> (str, float):
        return self.forward(s)


class SpanModel:
    def __init__(self) -> None:
        self.item2ctgr: dict[str, str] = {}

    def load(self, fp: str) -> None:
        """Load appropriate info for model.

        In the case of string matching, this will be a dict.
        Perform lowercasing and whitespace stripping on items.
        Sort item2ctgr according to entity length in ascending order.
        """
        print(f"Load string matching dict from {fp}")
        with open(file=fp) as f:
            item2ctgr = json.load(fp=f)

        item2ctgr_pairs = [(i.lower(), c) for i, c in item2ctgr.items()]
        item2ctgr_pairs = sorted(item2ctgr_pairs, key=lambda x: len(x[0]), reverse=True)
        item2ctgr = {i: c for i, c in item2ctgr_pairs}

        self.item2ctgr = item2ctgr

    def preprocess_prod_name(self, prod_name: str) -> str:
        """
        Performs basic pre-processing on string.

        Pre-processing:
            1. Lowercase and strip whitespace.
            2. Remove brackets and their contents.
            3. Remove `tie-dye` samples.
        """
        if pd.isnull(prod_name):
            return ""

        prod_name = prod_name.lower().strip()
        prod_name = remove_brackets_and_contents(prod_name)
        prod_name = remove_tie_dye(prod_name)
        return prod_name

    def match_name(self, prod_name: str):
        if not prod_name:
            return ""

        prod_name = self.preprocess_prod_name(prod_name)

        item2names = {}
        label2starts = {}
        for item, ctgr in self.item2ctgr.items():
            if not item:
                continue

            if ctgr == "WATCHES":
                prod_name = remove_black_watch(prod_name)

            if ctgr == "SOCKS":
                if (
                    "레드삭스" in prod_name
                    or "삭스 블루" in prod_name
                    or "삭스블루" in prod_name
                    or "사은품 양말" in prod_name
                    or "사은품양말" in prod_name
                    or "부츠삭스" in prod_name
                ):
                    continue

            matches = re.finditer(item, prod_name)
            for match in matches:
                start = match.start()
                end = match.end()

                if item == "티":
                    if prod_name[start + 1 :].startswith("셔츠"):  # ex) '티셔츠'.
                        pass
                    elif start != 0:
                        if end == len(prod_name):  # ex) '긴 소매티'
                            pass
                        elif prod_name[end] == " ":  # ex) '긴 소매티 (2022)'
                            pass
                    else:
                        continue
                elif item == "t":
                    if start == 0:
                        try:
                            if prod_name[start + 1] not in SPACE_CHARS:
                                continue
                        except (
                            IndexError
                        ) as err:  # Cases like `타이다이T` are processed to `t` which were missed previously.
                            print(f"Got {err} for {prod_name}.")
                            pass
                    elif start != 0:
                        if end == len(prod_name):
                            pass
                        elif (
                            prod_name[end] not in SPACE_CHARS
                        ):  # The "t" has to be on its own.
                            continue

                        if prod_name[start - 1] not in SPACE_CHARS:
                            continue
                else:
                    if is_korean(item):
                        if not is_end_of_word(prod_name, end):
                            continue
                    else:
                        if is_subword(prod_name, [start, end]):
                            if not is_singular_form(prod_name, end):
                                continue

                try:
                    item2names[item].add(prod_name)
                    label2starts[item].add(start)
                except KeyError:
                    item2names[item] = set([prod_name])
                    label2starts[item] = set([start])

        if label2starts:
            result, label_idx = find_latest_entity(label2starts)

            if len(label2starts) != 1:
                if "with" in prod_name[:label_idx]:
                    label2starts_filt = {
                        label: starts
                        for label, starts in label2starts.items()
                        if label != result
                    }
                    result, _ = find_latest_entity(label2starts_filt)
        else:
            result = ""

        item2names_filt = {k: v for k, v in item2names.items() if k == result}

        return item2names_filt

    def forward(self, s: str):
        return self.match_name(s)

    def __call__(self, s: str) -> (str, float):
        return self.forward(s)
