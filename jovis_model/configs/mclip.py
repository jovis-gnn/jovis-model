from pydantic import BaseModel, Field, Extra


class MCLIPConfig(BaseModel, extra=Extra.allow):
    hf_name: str = Field(
        default="M-CLIP/XLM-Roberta-Large-Vit-B-32",
        description="hugging face model name",
    )
    config_name: str = Field(default=None, description="hugging face model config name")
    tokenizer_name: str = Field(default=None, description="hugging face tokenizer name")
    cache_dir: str = Field(
        default="/home/omnious/.cache/huggingface/hub",
        description="huggingface hub cache directory",
    )
    enable_fsdp: bool = Field(default=False)
