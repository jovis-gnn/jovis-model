from torch.utils.data import Dataset

from jovis_model.config import Config


class BaseDataProcessor:
    def __init__(self, config: Config) -> None:
        self.config = config

    def get_dataset(self, data_dir: str, file_name: str, dataset_type: str) -> Dataset:
        raise NotImplementedError()
