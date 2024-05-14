from typing import Optional

from pydantic import BaseModel, Field, model_validator

from .klue import KLUEConfig


class BaseConfig(BaseModel):
    pkg: str
    task: str
    data_dir: str
    train_file_name: str
    dev_file_name: Optional[str] = Field(default=None)
    test_file_name: Optional[str] = Field(default=None)
    output_dir: str
    gpus: int = Field(default=1)
    fp16: bool = Field(default=True)
    params: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def load_train_config(self):
        if self.pkg == "klue" and self.task == "ynat":
            self.params = KLUEConfig(**self.params)
        return self
