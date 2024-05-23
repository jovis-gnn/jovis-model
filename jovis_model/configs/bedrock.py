from pydantic import BaseModel, Field, Extra


class BedrockConfig(BaseModel, extra=Extra.allow):
    m_type_id: str = Field(
        default="anthropic.claude-3-haiku-20240307-v1:0", description="bedrock model id"
    )
    kb_id: str = Field(default="ZFRR7GGQWG")
    region_name: str = Field(default="us-east-1")
    anthropic_version: str = Field(default="bedrock-2023-05-31")

    max_new_tokens: int = Field(default=200)
