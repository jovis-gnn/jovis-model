import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


class ItemCF:
    def __init__(self, data_dict):
        self.train_df = data_dict["train_df"]
        self.num_product = data_dict["num_product"]
        self.num_device = data_dict["num_device"]

    def get_item_sim_matrix(self):
        item_indices = self.train_df["productid_model_index"].values
        user_indices = self.train_df["deviceid_model_index"].values
        data = np.ones(len(item_indices)).tolist()
        shape = [self.num_product, self.num_device]
        item_m = csr_matrix(
            (data, (item_indices, user_indices)), shape=tuple(shape), dtype=np.float32
        )
        item_m_norm = csr_matrix(normalize(item_m, norm="l2", axis=0))

        item_sim_matrix = item_m_norm.dot(item_m_norm.T)
        # item_sim_matrix = item_m.dot(item_m.T)
        item_sim_matrix.setdiag(np.zeros(self.num_product))
        return item_sim_matrix
