import importlib

from torch.utils.data import DataLoader, Dataset

from jovis_model.configs.base import BaseConfig


MODULES = {"klue_ynat": "jovis_model.datasets.klue.ynat.YNATProcessor"}


class DataProcessor:
    def __init__(self, config: BaseConfig) -> None:
        self.config = config

    def get_dataset(self, data_dir: str, file_name: str, dataset_type: str) -> Dataset:
        raise NotImplementedError()


class CommonDataModule:
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        module_name = MODULES[f"{self.config.pkg}_{self.config.task}"]
        module, class_ = module_name.rsplit(".", 1)
        module = importlib.import_module(module)
        self.processor = getattr(module, class_)(self.config)

    def prepare_dataset(self, dataset_type: str) -> Dataset:
        if dataset_type == "train":
            dataset = self.processor.get_dataset(
                self.config.data_dir, self.config.train_file_name, dataset_type
            )
        elif dataset_type == "dev":
            dataset = self.processor.get_dataset(
                self.config.data_dir, self.config.dev_file_name, dataset_type
            )
        elif dataset_type == "test":
            dataset = self.processor.get_dataset(
                self.config.data_dir, self.config.test_file_name, dataset_type
            )
        else:
            raise ValueError(f"{dataset_type} do not support. [train|dev|test]")

        return dataset

    def get_dataloader(self, dataset_type: str, batch_size: int, shuffle: bool = False):
        return DataLoader(
            self.prepare_dataset(dataset_type),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.params.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            "train", self.config.params.train_batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            "dev", self.config.params.eval_batch_size, shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            "test", self.config.params.eval_batch_size, shuffle=False
        )
