from pydantic import BaseModel, Field, Extra


class LLMConfig(BaseModel, extra=Extra.allow):
    hf_name: str = Field(
        default="klue/roberta-base", description="hugging face model name"
    )
    config_name: str = Field(default=None, description="hugging face model config name")
    tokenizer_name: str = Field(default=None, description="hugging face tokenizer name")
    cache_dir: str = Field(
        default="/home/omnious/.cache/huggingface/hub",
        description="huggingface hub cache directory",
    )

    enable_fsdp: bool = Field(default=False, description="train params")
    use_peft: bool = Field(default=False, description="train params")
    quantization: bool = Field(default=False, description="train params")
    mixed_precision: bool = Field(default=False, description="train params")
    use_fp16: bool = Field(default=False, description="train params")
    context_length: int = Field(default=4096, description="train params")

    max_new_tokens: int = Field(default=200)
    top_p: float = Field(default=1.0)
    top_k: int = Field(default=50)
    temperature: float = Field(default=1.0)
