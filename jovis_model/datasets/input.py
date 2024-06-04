import json
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class TextToLabel:
    iid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)

    def to_json_string(self) -> None:
        return json.dumps(self.to_dict(), indent=4)


@dataclass(frozen=True)
class TextToLabelFeature:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self) -> None:
        return json.dumps(dataclasses.asdict(self))


@dataclass
class DialogToLabel:
    eid: str
    dialog: List[dict]
    label: str

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)

    def to_json_string(self) -> None:
        return json.dumps(self.to_dict(), indent=4)
