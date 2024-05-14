from pydantic import BaseModel, Field, model_validator

from .klue import TrainConfig


class Config(BaseModel):
    command: str
    task: str
    output_dir: str
    gpus: int = Field(default=None)
    fp16: bool = Field(default=None)
    num_sanity_val_steps: int = Field(default=2)
    max_grad_norm: float = Field(default=1.0)
    gradient_accumulation_steps: int = Field(default=1)
    seed: int = Field(default=42)
    metric_key: str = Field(default="loss")
    patience: int = Field(default=5)
    early_stopping_mode: str = Field(default="max")
    train_config: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def load_train_config(self):
        print(self)
        if self.task == "ynat":
            self.train_config = TrainConfig(**self.train_config)
        return self
