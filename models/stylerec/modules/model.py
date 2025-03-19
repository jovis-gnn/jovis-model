from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoModel

from models.stylerec.utils.helper import stack_dict, unstack_dict


class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim, bias=False)

    def forward(self, images):
        embed = self.model(images)
        # embed = F.normalize(embed, p=2, dim=1)
        return embed


class TextEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        huggingface_model: str = "sentence-transformers/all-MiniLM-L12-v2",
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(huggingface_model)
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc_embed = nn.Parameter(
            torch.empty(self.model.config.hidden_size, embedding_dim)
        )
        nn.init.normal_(self.fc_embed, std=self.model.config.hidden_size**-0.5)
        # self.fc_embed = nn.Sequential(
        #     nn.LeakyReLU(), nn.Dropout(0.3), nn.Linear(768, embedding_dim)
        # )

    def _mean_pooling(self, model_output, attention_mask):
        token_embeds = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        )

        return torch.sum(token_embeds * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids, attention_mask):
        embed = self.model(input_ids, attention_mask)
        embed = self._mean_pooling(embed, attention_mask) @ self.fc_embed

        return embed


class EmbeddingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config["embedding_dim"]
        self.normalize_embedding = config["normalize_embedding"]

        if config["use_text"]:
            self.txt_encoder = TextEncoder(embedding_dim=self.embedding_dim // 2)
            self.img_encoder = ImageEncoder(embedding_dim=self.embedding_dim // 2)
        else:
            self.img_encoder = ImageEncoder(embedding_dim=self.embedding_dim)

    def forward(
        self,
        image_arrays: torch.Tensor,
        text_token_ids: torch.Tensor = None,
        attention_masks: torch.Tensor = None,
    ):
        embed = self.img_encoder(image_arrays)
        if text_token_ids is not None:
            txt_embed = self.txt_encoder(text_token_ids, attention_masks)
            embed = torch.concat([embed, txt_embed], dim=1)

        if self.normalize_embedding:
            embed = F.normalize(embed, p=2, dim=1)
        return embed


class OutfitTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_model = EmbeddingModel(config)
        self.embedding_dim = config["embedding_dim"]
        self.ffn_dim = config["ffn_dim"]
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            dim_feedforward=self.ffn_dim,
            nhead=self.n_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=self.n_layers,
            norm=nn.LayerNorm(self.embedding_dim),
            enable_nested_tensor=False,
        )

        self.task = config["task"]
        self.task2id = {"cp": 0, "cir": 1}
        self.task_embeddings = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.embedding_dim,
            max_norm=1 if self.embedding_model.normalize_embedding else None,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(self.embedding_dim, 1), nn.Sigmoid()
        )

    def encode(self, batch_inputs: Dict[str, torch.Tensor]):
        batch_inputs = stack_dict(batch_inputs)
        outputs = {"padding_mask": batch_inputs["padding_mask"]}
        del batch_inputs["padding_mask"]
        batch_embeds = self.embedding_model(**batch_inputs)
        outputs["batch_embeds"] = batch_embeds
        outputs = unstack_dict(outputs)

        return outputs["batch_embeds"]

    def forward(
        self, embeddings, padding_mask, query_inputs=None, normalize: bool = True
    ):
        batch_size, *_ = embeddings.shape

        task_id = torch.LongTensor(
            [self.task2id[self.task] for _ in range(batch_size)]
        ).to(embeddings.device)
        prefix_embed = self.task_embeddings(task_id).unsqueeze(1)
        prefix_mask = torch.zeros((batch_size, 1), dtype=torch.bool).to(
            embeddings.device
        )

        embeddings = torch.cat([prefix_embed, embeddings], dim=1)
        padding_mask = torch.cat([prefix_mask, padding_mask], dim=1)

        outputs = self.transformer(embeddings, src_key_padding_mask=padding_mask)[
            :, 0, :
        ]
        if self.task == "cp":
            outputs = self.classifier(outputs)
        elif self.task == "cir":
            if normalize:
                outputs = F.normalize(outputs, p=2, dim=-1)

        return outputs
