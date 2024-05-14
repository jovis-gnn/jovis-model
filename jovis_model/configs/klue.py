from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    m_name: str
    config_name: str
    tokenizer_name: str
    cache_dir: str
    encoder_layerdrop: float
    decode_layerdrop: float
    dropout: float
    attention_dropout: float
    learning_rate: float = Field(default=5e-5)
    weight_decay: float = Field(default=0.0)
    adam_epsilon: float = Field(default=1e-8)
    warmup_steps: int = Field(default=None)
    warmup_ratio: float = Field(default=None)
    num_train_epochs: int = Field(default=4)
    adafactor: bool
    verbose_step_count: int = Field(default=100)
