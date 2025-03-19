import numpy as np
from torch.utils.data import Dataset


class Sent2VecData(Dataset):
    def __init__(self, data_dict):
        super(Sent2VecData, self).__init__()
        self.train_data = (
            data_dict["train_df"][["productid_model_index", "productname"]]
            .drop_duplicates()
            .values
        )
        self.productid_model_indices, self.prod_names = (
            self.train_data[:, 0],
            self.train_data[:, 1],
        )

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.productid_model_indices[idx], self.prod_names[idx]


class Sent2VecTest(Dataset):
    def __init__(self, data_dict):
        super(Sent2VecTest, self).__init__()
        self.test_data = data_dict["test_df"][
            [
                "deviceid_model_index",
                "productid_model_index",
                "occurence",
            ]
        ].values.astype(np.int32)
        self.pid2pname = (
            data_dict["df"][["productid_model_index", "productname"]]
            .drop_duplicates()
            .set_index("productid_model_index")
            .to_dict()["productname"]
        )
        self.train_dict = data_dict["device_train_dict"]
        self.num_product = data_dict["num_product"]

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        u, gt, occurence = self.test_data[idx]
        user_interaction = self.train_dict.get(u, [])
        valid_history = []
        if len(user_interaction) > 0:
            valid_history = user_interaction[: occurence - 1]
            if len(valid_history) > 0:
                query = int(valid_history[-1])
                pname = self.pid2pname[query]
            else:
                query = 0
                pname = "cold start"
        else:
            query = 0
            pname = "cold start"
        mask = np.zeros(self.num_product)
        mask[valid_history] = -10000

        return u, query, gt, mask, occurence, pname
