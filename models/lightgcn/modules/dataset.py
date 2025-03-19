import torch
import numpy as np
from torch.utils.data import Dataset


class NegativeSampler:
    def __init__(self, data_dict):
        self.product_cand = {}
        self.num_product = data_dict["num_product"]
        self.device_train_dict = data_dict["device_train_dict"]

    def sampling(self, batch):
        triplets = []
        for pos_pair in batch:
            neg = np.random.randint(self.num_product)
            while neg in self.device_train_dict[pos_pair[0]]:
                neg = np.random.randint(self.num_product)
            triplet = np.expand_dims(np.concatenate([pos_pair, [neg]], axis=0), 0)
            triplets.append(triplet)
        triplets = np.concatenate(triplets, axis=0)
        return triplets


class LightGCNData(Dataset):
    def __init__(self, data_dict):
        super(LightGCNData, self).__init__()
        self.num_device = data_dict["num_device"]
        self.num_product = data_dict["num_product"]
        self.pos_pairs = data_dict["train_df"][
            ["deviceid_model_index", "productid_model_index", "norm"]
        ].values.astype(np.float32)

    def get_graph_matrix(self):
        device_product_norm = torch.sparse.FloatTensor(
            torch.stack(
                [
                    torch.LongTensor(self.pos_pairs[:, 0]),
                    torch.LongTensor(self.pos_pairs[:, 1]),
                ]
            ),
            torch.FloatTensor(self.pos_pairs[:, 2]),
            (self.num_device, self.num_product),
        )
        product_device_norm = torch.sparse.FloatTensor(
            torch.stack(
                [
                    torch.LongTensor(self.pos_pairs[:, 1]),
                    torch.LongTensor(self.pos_pairs[:, 0]),
                ]
            ),
            torch.FloatTensor(self.pos_pairs[:, 2]),
            (self.num_product, self.num_device),
        )
        return [device_product_norm, product_device_norm]

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, idx):
        return self.pos_pairs[:, :2][idx]


class LightGCNTest(Dataset):
    def __init__(self, data_dict):
        super(LightGCNTest, self).__init__()
        self.test_data = (
            data_dict["test_df"][["deviceid_model_index"]]
            .drop_duplicates()
            .values.astype(np.float32)
        )
        self.train_dict = data_dict["device_train_dict"]
        self.num_product = data_dict["num_product"]

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        u = self.test_data[idx]
        device_train_data = self.train_dict.get(u[0], [])
        mask = np.zeros(self.num_product)
        mask[device_train_data] = -10000

        return u, mask
