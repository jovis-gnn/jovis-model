from typing import List

# from PIL import Image
import torch
import clip
from transformers import AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedModel

from jovis_model.config import Config


class MCLIPConfig(PretrainedConfig):
    model_type = "M-CLIP"

    def __init__(
        self,
        modelBase="xlm-roberta-large",
        transformerDimSize=1024,
        imageDimSize=768,
        **kwargs
    ):
        self.transformerDimensions = transformerDimSize
        self.numDims = imageDimSize
        self.modelBase = modelBase
        super().__init__(**kwargs)


class MultilingualCLIP(PreTrainedModel):
    config_class = MCLIPConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.transformer = AutoModel.from_pretrained(
            config.modelBase, cache_dir=kwargs.get("cache_dir")
        )
        self.LinearTransformation = torch.nn.Linear(
            in_features=config.transformerDimensions, out_features=config.numDims
        )

    def forward(self, txt, tokenizer):
        txt_tok = tokenizer(txt, padding=True, return_tensors="pt")
        embs = self.transformer(**txt_tok)[0]
        att = txt_tok["attention_mask"]
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.LinearTransformation(embs)

    @classmethod
    def _load_state_dict_into_model(
        cls, model, state_dict, pretrained_model_name_or_path, _fast_init=True
    ):
        model.load_state_dict(state_dict)
        return model, [], [], []


class CLIPModel:
    def __init__(self, config: Config):
        self.device = "cpu"
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.params.hf_name)
        self.text_model = MultilingualCLIP.from_pretrained(self.config.params.hf_name)
        self.text_model = self.text_model.to(self.device)
        self.image_model, self.image_processor = clip.load(
            "ViT-B/32", device=self.device
        )

    def inference(self, sample_inputs: List[str], dtype="text"):
        if dtype == "text":
            outputs = self.text_model.forward(sample_inputs, self.tokenizer)
        elif dtype == "image":
            outputs = self.image_processor(sample_inputs).unsqueeze(0).to(self.device)

        return outputs
