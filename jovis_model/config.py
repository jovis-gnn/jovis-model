from typing import Optional

from pydantic import BaseModel, Field, model_validator

from jovis_model.configs.klue import KLUEConfig
from jovis_model.configs.llm import LLMConfig


class Config(BaseModel):
    pkg: str
    task: str
    use_hf_model: bool
    data_dir: Optional[str] = Field(default=None)
    train_file_name: Optional[str] = Field(default=None)
    dev_file_name: Optional[str] = Field(default=None)
    test_file_name: Optional[str] = Field(default=None)
    output_dir: Optional[str] = Field(default=None)
    gpus: Optional[int] = Field(default=1)
    fp16: Optional[bool] = Field(default=True)
    params: Optional[dict] = Field(default_factory=dict)

    @model_validator(mode="after")
    def load_train_config(self):
        if self.pkg == "klue" and self.task == "ynat":
            self.params = KLUEConfig(**self.params)
        elif self.pkg == "llm" and self.task == "chat":
            self.params = LLMConfig(**self.params)
        return self
