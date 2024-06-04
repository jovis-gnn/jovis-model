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

    enable_fsdp: bool = Field(default=False)
    use_peft: bool = Field(default=False)
    quantization: bool = Field(default=False)
    mixed_precision: bool = Field(default=False)
    use_fp16: bool = Field(default=False)

    max_new_tokens: int = Field(default=200)
    top_p: float = Field(default=1.0)
    top_k: int = Field(default=50)
    temperature: float = Field(default=1.0)
