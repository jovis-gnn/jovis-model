from pydantic import BaseModel, Field, Extra


class BedrockConfig(BaseModel, extra=Extra.allow):
    role_arn: str = Field(
        default="arn:aws:iam::851725209478:role/mediacommerce-skb-bedrock"
    )
    m_type_id: str = Field(
        default="anthropic.claude-3-haiku-20240307-v1:0", description="bedrock model id"
    )
    kb_id: str = Field(default="ZFRR7GGQWG")
    region_name: str = Field(default="us-east-1")
    anthropic_version: str = Field(default="bedrock-2023-05-31")

    enable_fsdp: bool = Field(default=False)
    max_new_tokens: int = Field(default=200)
    seed: int = Field(default=42)
