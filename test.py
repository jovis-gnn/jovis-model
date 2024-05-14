from jovis_model.configs.base import BaseConfig
from jovis_model.datasets.common import CommonDataModule

params = {
    "pkg": "klue",
    "task": "ynat",
    "data_dir": "/home/omnious/workspace/jovis/jovis-model/jovis_model/_db/klue/ynat-v1.1",
    "train_file_name": "ynat-v1.1_train.json",
    "dev_file_name": "ynat-v1.1_dev.json",
    "output_dir": "/home/omnious/workspace/jovis/jovis-model/outputs",
}
config = BaseConfig(**params)
dm = CommonDataModule(config)
dl = dm.train_dataloader()
