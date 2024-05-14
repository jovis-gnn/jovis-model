from typing import Any, Dict

from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader, Dataset


class DataProcessor:
    def __init__(self, params: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> None:
        self.params = params
        self.tokenizer = tokenizer

    def get_dataset(self, data_dir: str, file_name: str, dataset_type: str) -> Dataset:
        raise NotImplementedError()


class CommonDataModule:
    def __init__(self, params: Dict[str, Any], processor: DataProcessor):
        super().__init__()
        self.params = params
        self.processor = processor

    def prepare_dataset(self, dataset_type: str) -> Dataset:
        if dataset_type == "train":
            dataset = self.processor.get_dataset(
                self.params.data_dir, self.params.train_file_name, dataset_type
            )
        elif dataset_type == "dev":
            dataset = self.processor.get_dataset(
                self.params.data_dir, self.params.dev_file_name, dataset_type
            )
        elif dataset_type == "test":
            dataset = self.processor.get_dataset(
                self.params.data_dir, self.params.test_file_name, dataset_type
            )
        else:
            raise ValueError(f"{dataset_type} do not support. [train|dev|test]")

        return dataset

    def get_dataloader(self, dataset_type: str, batch_size: int, shuffle: bool = False):
        return DataLoader(
            self.prepare_dataset(dataset_type),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.params.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train", self.params.train_batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("dev", self.params.eval_batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", self.params.eval_batch_size, shuffle=False)
