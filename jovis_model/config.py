from pydantic import BaseModel, Field, model_validator

from jovis_model.configs.klue import KLUEConfig
from jovis_model.configs.llm import LLMConfig
from jovis_model.configs.mclip import MCLIPConfig
from jovis_model.configs.bedrock import BedrockConfig


class Config(BaseModel):
    pkg: str
    task: str
    use_hf_model: bool
    data_dir: str = Field(default=None)
    output_dir: str = Field(default=None)
    train_file_name: str = Field(default=None)
    eval_file_name: str = Field(default=None)
    test_file_name: str = Field(default=None)
    params: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def load_train_config(self):
        if self.pkg == "klue":
            self.params = KLUEConfig(**self.params)
        elif self.pkg == "llm" and self.task in [
            "chat",
            "internvl",
            "sentence_embedding",
        ]:
            self.params = LLMConfig(**self.params)
        elif self.pkg == "llm" and self.task == "bedrock":
            self.params = BedrockConfig(**self.params)
        elif self.pkg == "llm" and self.task == "mclip":
            self.params = MCLIPConfig(**self.params)
        return self
