from typing import Optional
from pydantic import BaseModel, Field, Extra


class KLUEConfig(BaseModel, extra=Extra.allow):
    hf_name: str = Field(
        default="klue/roberta-base", description="hugging face model name"
    )
    config_name: str = Field(
        default="klue/roberta-base", description="hugging face model config name"
    )
    tokenizer_name: str = Field(
        default="klue/roberta-base", description="hugging face tokenizer name"
    )
    cache_dir: str = Field(default=None, description="huggingface hub cache directory")
    num_workers: int = Field(default=4)
    train_batch_size: int = Field(default=32)
    eval_batch_size: int = Field(default=64)
    max_seq_length: int = Field(default=128)
    num_sanity_val_steps: int = Field(default=2)
    max_grad_norm: float = Field(default=1.0)
    gradient_accumulation_steps: int = Field(default=1)
    seed: int = Field(default=42)
    metric: str = Field(default="loss")
    patience: int = Field(default=5)
    early_stopping_mode: str = Field(default="max")
    encoder_layerdrop: Optional[float] = Field(default=None)
    decode_layerdrop: Optional[float] = Field(default=None)
    dropout: Optional[float] = Field(default=None)
    attention_dropout: Optional[float] = Field(default=None)
    learning_rate: float = Field(default=5e-5)
    lr_scheduler: str = Field(default="linear")
    weight_decay: float = Field(default=0.0)
    adam_epsilon: float = Field(default=1e-8)
    warmup_steps: int = Field(default=None)
    warmup_ratio: float = Field(default=None)
    num_train_epochs: int = Field(default=4)
    adafactor: bool = Field(default=True)
    verbose_step_count: int = Field(default=100)
