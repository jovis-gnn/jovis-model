import numpy as np
from torch.utils.data import Dataset


class ItemCFData(Dataset):
    def __init__(self, data_dict):
        super(ItemCFData, self).__init__()
        self.num_product = data_dict["num_product"]
        self.test_users = list(set(data_dict["test_df"]["deviceid_model_index"].values))
        self.train_dict = data_dict["device_train_dict"]

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, idx):
        u = self.test_users[idx]
        device_train_data = self.train_dict.get(u, [])
        mask = np.ones(self.num_product)
        mask[device_train_data] = -10000

        return u, mask
