from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM

from jovis_model.models.base import BaseModel
from jovis_model.config import Config


class ChatModel(BaseModel):
    def __init__(self, config: Config):
        self.config = config
        super().__init__(
            self.config,
            use_hf_model=self.config.use_hf_model,
            model_type=AutoModelForCausalLM,
            config_kwargs={
                "trust_remote_code": True,
            },
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "load_in_8bit": True if self.config.params.quantization else None,
                "trust_remote_code": True,
            },
        )
        if not self.tokenizer.pad_token_id:
            pad_token_id = self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")
            self.tokenizer.pad_token = "<|end_of_text|>"
            self.tokenizer.pad_token_id = pad_token_id
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if len(self.tokenizer) > self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, **inputs: Dict[str, List[torch.Tensor]]):
        return self.model(**inputs)

    def training_step(self, batch: List[torch.Tensor]):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
        outputs = self.forward(**inputs)
        loss = outputs[0]

        return {"loss": loss}

    def validation_step(self, batch: List[torch.Tensor]):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
        outputs = self.forward(**inputs)
        loss, logits = outputs[:2]

        return {"loss": loss, "logits": logits, "labels": inputs["labels"]}

    def convert_outputs(self, outputs: Dict[str, torch.Tensor]):
        preds = torch.argmax(outputs["logits"], dim=1)
        labels = outputs["labels"]
        return preds, labels

    def inference(self, sample_inputs: torch.Tensor):
        outputs = self.model.generate(
            input_ids=sample_inputs,
            max_new_tokens=self.config.params.max_new_tokens,
            do_sample=True,
            top_p=self.config.params.top_p,
            temperature=self.config.params.temperature,
            top_k=self.config.params.top_k,
            repetition_penalty=1.0,
            length_penalty=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output_text = self.tokenizer.decode(outputs[0])
        stop = "<|start_header_id|>assistant<|end_header_id|>"
        pos = output_text.rfind(stop)
        output_text = output_text[pos + len(stop) :]
        if "<|eot_id|>" in output_text:
            output_text = output_text.replace("<|eot_id|>", "")
        output_text = output_text.lstrip().rstrip()
        return output_text
