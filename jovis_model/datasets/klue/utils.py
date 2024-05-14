from typing import List, Optional, Union

from transformers import PreTrainedTokenizer

from jovis_model.datasets.inputs import InputExample, InputFeatures


def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    label_list: List[str],
    max_length: Optional[int] = None,
    task_mode: Optional[str] = None,
) -> List[InputFeatures]:
    if max_length is None:
        max_length = tokenizer.max_len

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None, List[int]]:
        if example.label is None:
            return None
        if task_mode == "classification":
            return label_map[example.label]
        elif task_mode == "regression":
            return float(example.label)
        elif task_mode == "tagging":
            token_label = [label_map["O"]] * (max_length)
            for i, label in enumerate(example.label[: max_length - 2]):
                token_label[i + 1] = label_map[label]
            return token_label
        raise KeyError(task_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    # for i, example in enumerate(examples[:5]):
    #     logger.info("*** Example ***")
    #     logger.info("guid: %s" % (example.guid))
    #     logger.info("features: %s" % features[i])

    return features
