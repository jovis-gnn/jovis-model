import json
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from transformers import PreTrainedTokenizer


@dataclass
class InputExample:
    """A single example of data.utils

    This is for YNAT, KLUE-NLI, KLUE-NER, and KLUE-RE.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)

    def to_json_string(self) -> None:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    """A single set of features of data to feed into the pretrained models.

    This is for YNAT, KLUE-STS, KLUE-NLI, KLUE-NER, and KLUE-RE. Property names
    are same with the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self) -> None:
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


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
