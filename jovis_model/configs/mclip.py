from pydantic import BaseModel, Field, Extra


class MCLIPConfig(BaseModel, extra=Extra.allow):
    hf_name: str = Field(
        default="M-CLIP/XLM-Roberta-Large-Vit-B-32",
        description="hugging face model name",
    )
