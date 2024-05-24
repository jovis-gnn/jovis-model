import importlib

from torch.utils.data import DataLoader, Dataset

from jovis_model.config import Config


DATA_MODULE_LIST = {
    "klue_ynat": "jovis_model.datasets.klue.ynat.YNATProcessor",
    "llm_chat": "jovis_model.datasets.llm.chat.ChatProcessor",
    "llm_bedrock": "jovis_model.datasets.llm.bedrock.BedrockProcessor",
    "llm_internvl": "jovis_model.datasets.llm.internvl.InternVLProcessor",
    "llm_sentence_embedding": "jovis_model.datasets.llm.sentence_embedding.SentenceProcessor",
}

MODEL_MODULE_LIST = {
    "klue_ynat": "jovis_model.models.klue.sequence_classification.SCTransformer",
    "llm_chat": "jovis_model.models.llm.chat.ChatModel",
    "llm_bedrock": "jovis_model.models.llm.bedrock.Bedrock",
    "llm_internvl": "jovis_model.models.llm.internvl.InternVLModel",
    "llm_sentence_embedding": "jovis_model.models.llm.sentence_embedding.SentenceEmbedding",
}


class ModelModule:
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        module_name = MODEL_MODULE_LIST[f"{self.config.pkg}_{self.config.task}"]
        module, class_ = module_name.rsplit(".", 1)
        module = importlib.import_module(module)
        self.processor = getattr(module, class_)(self.config)


class DataModule:
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        module_name = DATA_MODULE_LIST[f"{self.config.pkg}_{self.config.task}"]
        module, class_ = module_name.rsplit(".", 1)
        module = importlib.import_module(module)
        self.processor = getattr(module, class_)(self.config)

    def prepare_dataset(self, dataset_type: str) -> Dataset:
        if dataset_type == "train":
            assert self.config.train_file_name is not None, "There's no train dataset."
            dataset = self.processor.get_dataset(
                self.config.data_dir, self.config.train_file_name
            )
        elif dataset_type == "dev":
            assert self.config.dev_file_name is not None, "There's no eval dataset."
            dataset = self.processor.get_dataset(
                self.config.data_dir, self.config.dev_file_name
            )
        elif dataset_type == "test":
            assert self.config.test_file_name is not None, "There's no test dataset."
            dataset = self.processor.get_dataset(
                self.config.data_dir, self.config.test_file_name
            )
        else:
            raise ValueError(f"{dataset_type} do not support. [train|dev|test]")

        return dataset

    def get_dataloader(self, dataset_type: str, batch_size: int, shuffle: bool = False):
        dataset = self.prepare_dataset(dataset_type)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.params.num_workers,
        )
        return dataloader
