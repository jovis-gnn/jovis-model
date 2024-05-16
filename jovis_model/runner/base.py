import importlib

from jovis_model.utils.module import ModelModules
from jovis_model.datasets.common import CommonDataModule


class ModelRunner:
    def __init__(self, config):
        self.config = config
        self.get_dataloader()
        self.get_model()

    def get_dataloader(self):
        cdm = CommonDataModule(self.config)
        self.train_dataloader = cdm.train_dataloader()
        self.config.params.num_labels = len(cdm.processor.get_labels())
        self.config.params.dataset_size = len(self.train_dataloader)

    def get_model(self):
        module_name = ModelModules[f"{self.config.pkg}_{self.config.task}"]
        module, class_ = module_name.rsplit(".", 1)
        module = importlib.import_module(module)
        self.model = getattr(module, class_)(self.config, {})

    def train(self):
        for idx, batch in enumerate(self.train_dataloader):
            loss = self.model.training_step(batch, idx)
            print(loss)
