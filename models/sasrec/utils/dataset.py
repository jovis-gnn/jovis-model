import numpy as np
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, data_dict, seq_len, usage: str = "train"):
        super(BasicDataset, self).__init__()
        self.num_product = data_dict["num_product"]
        self.device_train_dict = data_dict["device_train_dict"]
        self.seq_len = int(seq_len)
        self.usage = usage

        temp_data = data_dict["{}".format(usage + "_df")][
            ["deviceid_model_index", "productid_model_index"]
            if usage == "train"
            else ["deviceid_model_index"]
        ]
        if usage != "train":
            temp_data = temp_data.drop_duplicates()
        self.data = temp_data.values.astype(np.float32)
        self.occurences = data_dict["{}".format(usage + "_df")][
            ["occurence"]
        ].values.astype(np.int32)

        self.used_products = np.unique(self.data[:, -1].astype(int))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur_data = self.data[idx]
        user_interaction = self.device_train_dict.get(int(cur_data[0]), [])
        # user_interaction = self.device_train_dict.get((int(cur_data[0]), int(cur_data[1])), [])
        user_history, user_history_mask = np.zeros(self.seq_len), np.zeros(self.seq_len)

        if len(user_interaction) > 0:
            occurence = self.occurences[idx][0]
            valid_history = user_interaction[:occurence - 1]
            valid_history = valid_history[-1 * self.seq_len :]
            starting_idx = (
                -1 * len(valid_history) if len(valid_history) > 0 else len(user_history)
            )
            user_history[starting_idx:] = valid_history
            user_history_mask[starting_idx:] = np.ones_like(valid_history)

        if self.usage == "test":
            purchase_mask = np.concatenate([[-1e9], np.zeros(self.num_product - 1)])
            purchase_mask[np.unique(user_history).astype(int)] = -1e9

            return (cur_data, user_history, user_history_mask, purchase_mask)

        return (cur_data, user_history, user_history_mask, *self._sample(cur_data))

    def _sample(self, pos_pair):
        random_idx = np.random.randint(self.data.shape[0])
        while int(self.data[random_idx][1]) == int(pos_pair[1]):
            random_idx = np.random.randint(self.data.shape[0])
        popular_neg = int(self.data[random_idx][1])

        # random_neg = np.random.randint(self.num_product)
        random_neg = np.random.choice(self.used_products)
        while random_neg == int(pos_pair[1]) or random_neg == popular_neg:
            # random_neg = np.random.randint(self.num_product)
            random_neg = np.random.choice(self.used_products)

        return (popular_neg, random_neg)


class SRGNNDataset(Dataset):
    def __init__(self, data_dict, seq_len, usage: str = "train"):
        super(SRGNNDataset, self).__init__()
        self.num_product = data_dict["num_product"]
        self.device_train_dict = data_dict["device_train_dict"]
        self.seq_len = int(seq_len)
        self.usage = usage

        cur_dict = data_dict["{}".format(usage + "_df")]

        temp_data = cur_dict[
            ["deviceid_model_index", "session", "productid_model_index"]
            if usage == "train"
            else ["deviceid_model_index", "session"]
        ]
        if usage != "train":
            temp_data = temp_data.drop_duplicates()
        self.data = temp_data.values.astype(np.float32)

        self.occurences = cur_dict[["occurence"]].values.astype(np.int32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur_data = self.data[idx]
        user_interaction = self.device_train_dict.get((int(cur_data[0]), int(cur_data[1])), [])
        user_history = np.zeros(self.seq_len)

        if len(user_interaction) > 0:
            occurence = self.occurences[idx][0]
            valid_history = user_interaction[:occurence - 1]
            valid_history = valid_history[-1 * self.seq_len :]
            starting_idx = (
                -1 * len(valid_history) if len(valid_history) > 0 else len(user_history)
            )
            user_history[starting_idx:] = valid_history

        if self.usage == "test":
            purchase_mask = np.zeros(self.num_product)
            purchase_mask[np.unique(user_history).astype(int)] = -1e9

            return (cur_data, user_history, starting_idx, purchase_mask)

        return (cur_data, user_history, starting_idx, *self._sample(cur_data))

    def _sample(self, pos_pair):
        random_idx = np.random.randint(self.data.shape[0])
        while int(self.data[random_idx][1]) == int(pos_pair[2]):
            random_idx = np.random.randint(self.data.shape[0])
        popular_neg = int(self.data[random_idx][1])

        random_neg = np.random.randint(self.num_product)
        while random_neg == int(pos_pair[2]) or random_neg == popular_neg:
            random_neg = np.random.randint(self.num_product)

        return (popular_neg, random_neg)


class UnifiedDataset(Dataset):
    def __init__(self, data_dict, seq_len, window_size=0, usage: str = "train"):
        super(UnifiedDataset, self).__init__()
        self.num_product = data_dict["num_product"]
        self.device_train_dict = data_dict["device_train_dict"]
        self.seq_len = int(seq_len)
        self.usage = usage

        self.window_size = window_size

        self.data = data_dict["{}".format(usage + "_df")][
            ["deviceid_model_index", "productid_model_index"]
        ].values.astype(np.float32)
        self.occurences = data_dict["{}".format(usage + "_df")][
            ["occurence"]
        ].values.astype(np.int32)

        self.used_products = np.unique(self.data[:, -1].astype(int))

        if usage == "test":
            idxes = np.random.choice(self.data.shape[0], int(self.data.shape[0] * 0.5), replace=False)
            self.data, self.occurences = tuple(map(lambda x: x[idxes], (self.data, self.occurences)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur_data = self.data[idx]
        user_interaction = self.device_train_dict.get(int(cur_data[0]), [])
        user_history, user_history_mask = np.zeros(self.seq_len), np.zeros(self.seq_len)

        occurence = self.occurences[idx][0]

        if len(user_interaction) > 0:
            valid_history = user_interaction[:occurence - 1]
            valid_history = valid_history[(-1 * self.seq_len):]

            # if self.usage == 'test' and self.window_size > 0:
            if self.window_size > 0:
                valid_history = valid_history[(-1 * self.window_size):]

            starting_idx = (
                -1 * len(valid_history) if len(valid_history) > 0 else len(user_history)
            )
            user_history[starting_idx:] = valid_history
            user_history_mask[starting_idx:] = np.ones_like(valid_history)

        if self.usage == "test":
            purchase_mask = np.zeros(self.num_product)
            # purchase_mask[np.unique(user_history).astype(int)] = -1e9
            purchase_mask[np.concatenate([[0], user_interaction[:occurence - 1][-5:]], axis=-1).astype(int)] = -1e9

            return (cur_data, user_history, user_history_mask, occurence, purchase_mask)

        return (cur_data, user_history, user_history_mask, *self._sample(cur_data))

    def _sample(self, pos_pair):
        random_idx = np.random.randint(self.data.shape[0])
        while int(self.data[random_idx][1]) == int(pos_pair[1]):
            random_idx = np.random.randint(self.data.shape[0])
        popular_neg = int(self.data[random_idx][1])

        random_neg = np.random.randint(self.num_product)
        # random_neg = np.random.choice(self.used_products)
        while random_neg == int(pos_pair[1]) or random_neg == popular_neg:
            random_neg = np.random.randint(self.num_product)
            # random_neg = np.random.choice(self.used_products)

        return (popular_neg, random_neg)


class SND(Dataset):
    def __init__(self, data_dict, seq_len, window_size=0, usage="train", batchwise_sample=False, sample_method="weak", **kwargs):
        super(SND, self).__init__()
        self.num_product = data_dict["num_product"]
        self.device_train_dict = data_dict["device_train_dict"]

        self.seq_len = int(seq_len)
        self.usage = usage
        self.window_size = window_size
        self.batchwise_sample = batchwise_sample

        self.full_train = kwargs["full_train"]

        if not self.full_train:
            df_name = "{}".format(usage + "_df")
        else:
            df_name = "df"

        self.data = data_dict[df_name][["deviceid_model_index", "productid_model_index"]].values.astype(np.float32)
        self.occurences = data_dict[df_name][["occurence"]].values.astype(np.int32)

        self.popularity = np.array([len(data_dict['product_train_dict'].get(x, [])) for x in np.arange(data_dict['num_product'])])
        self.popularity_idx = np.array(sorted(range(len(self.popularity)), key=lambda k : self.popularity[k]))

        self.sample_method = sample_method
        self.sample_map = {"weak" : self._sample, }

        if usage == "test" and not self.full_train:
            self.test_occurences = data_dict["test_df"][["test_occurence"]].values.astype(np.int32)

            idxes = np.random.choice(self.data.shape[0], int(self.data.shape[0] * 0.4), replace=False)
            self.data, self.occurences, self.test_occurences = tuple(map(lambda x: x[idxes], (self.data, self.occurences, self.test_occurences)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur_data = self.data[idx]
        user_history, user_history_mask = np.zeros(self.seq_len), np.zeros(self.seq_len)
        user_interaction = self.device_train_dict.get(int(cur_data[0]), [])

        occurence = self.occurences[idx][0]

        if len(user_interaction) > 0:
            valid_history = user_interaction[:occurence - 1]
            valid_history = valid_history[(-1 * self.seq_len):]

            if self.window_size > 0:
                valid_history = valid_history[(-1 * self.window_size):]

            starting_idx = (-1 * len(valid_history) if len(valid_history) > 0 else len(user_history))

            user_history[starting_idx:] = valid_history
            user_history_mask[starting_idx:] = np.ones_like(valid_history)

        if self.usage == "test" and not self.full_train:
            purchase_mask = np.zeros(self.num_product)

            # purchase_mask[np.unique(user_history).astype(int)] = -1e9
            purchase_mask[np.concatenate([[0], user_interaction[:occurence - 1][-5:]], axis=-1).astype(int)] = -1e9

            test_occurence = self.test_occurences[idx][0]

            return (cur_data, user_history, user_history_mask, test_occurence, purchase_mask)

        if self.batchwise_sample:
            return (cur_data, user_history, user_history_mask)
        return (cur_data, user_history, user_history_mask, *self.sample_map[self.sample_method](cur_data))

    def _sample(self, pos_pair):
        pop_idx = np.random.randint(self.data.shape[0])
        while int(self.data[pop_idx][1]) in [0, int(pos_pair[1])]:
            pop_idx = np.random.randint(self.data.shape[0])
        popular_neg = int(self.data[pop_idx][1])

        random_neg = np.random.randint(self.num_product)
        while random_neg in [0, int(pos_pair[1]), popular_neg]:
            random_neg = np.random.randint(self.num_product)

        return popular_neg, random_neg


class MND(Dataset):
    def __init__(self, data_dict, seq_len, window_size=0, usage="train", batchwise_sample=False, sample_method="orig", **kwargs):
        super(MND, self).__init__()
        self.num_product = data_dict["num_product"]
        self.device_train_dict = data_dict["device_train_dict"]

        self.seq_len = int(seq_len)
        self.usage = usage
        self.window_size = window_size
        self.batchwise_sample = batchwise_sample

        self.full_train = kwargs["full_train"]

        if not self.full_train:
            df_name = "{}".format(usage + "_df")
        else:
            df_name = "df"

        self.data = data_dict[df_name][["deviceid_model_index", "productid_model_index"]].values.astype(np.float32)
        self.occurences = data_dict[df_name][["occurence"]].values.astype(np.int32)

        self.num_total_negs = kwargs['num_total_negs']
        self.num_popular_negs = kwargs['num_popular_negs']

        self.popularity = np.array([len(data_dict['product_train_dict'].get(x, [])) for x in np.arange(data_dict['num_product'])])
        self.popularity_idx = np.array(sorted(range(len(self.popularity)), key=lambda k : self.popularity[k]))

        self.sample_method = sample_method
        self.sample_map = {"orig": self._sample, "multi": self._multi_sample}

        if usage == "test" and not self.full_train:
            self.test_occurences = data_dict["test_df"][["test_occurence"]].values.astype(np.int32)

            idxes = np.random.choice(self.data.shape[0], int(self.data.shape[0] * 0.4), replace=False)
            self.data, self.occurences, self.test_occurences = tuple(map(lambda x: x[idxes], (self.data, self.occurences, self.test_occurences)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur_data = self.data[idx]
        user_history, user_history_mask = np.zeros(self.seq_len), np.zeros(self.seq_len)
        user_interaction = self.device_train_dict.get(int(cur_data[0]), [])

        occurence = self.occurences[idx][0]

        if len(user_interaction) > 0:
            valid_history = user_interaction[:occurence - 1]
            valid_history = valid_history[-1 * self.seq_len :]

            if self.window_size > 0:
                valid_history = valid_history[(-1 * self.window_size):]

            starting_idx = (-1 * len(valid_history) if len(valid_history) > 0 else len(user_history))

            user_history[starting_idx:] = valid_history
            user_history_mask[starting_idx:] = np.ones_like(valid_history)

        if self.usage == "test" and not self.full_train:
            purchase_mask = np.zeros(self.num_product)

            # purchase_mask[np.unique(user_history).astype(int)] = -1e9
            purchase_mask[np.concatenate([[0], user_interaction[:occurence - 1][-5:]], axis=-1).astype(int)] = -1e9

            test_occurence = self.test_occurences[idx][0]

            return (cur_data, user_history, user_history_mask, test_occurence, purchase_mask)

        if self.batchwise_sample:
            return (cur_data, user_history, user_history_mask)
        return (cur_data, user_history, user_history_mask, *self.sample_map[self.sample_method](cur_data))

    def _sample(self, pos_pair):
        pop_idx = np.random.randint(self.data.shape[0])
        while int(self.data[pop_idx][1]) == int(pos_pair[1]):
            pop_idx = np.random.randint(self.data.shape[0])
        popular_neg = int(self.data[pop_idx][1])

        items_total = np.arange(self.num_product)
        # sampled_negatives = np.random.choice(np.delete(items_total, items_total == int(pos_pair[1])), (self.num_total_negs, ), replace=False)
        sampled_negatives = np.random.choice(items_total, (self.num_total_negs, ), replace=False)

        return popular_neg, sampled_negatives

    def _multi_sample(self, pos_pair):
        items_total = np.arange(self.num_product)
        candidates = np.delete(items_total, [int(pos_pair[1]),])

        mod_popularity = np.delete(self.popularity, int(pos_pair[1]))

        bin_size, bin_selection = int(self.num_product // 20), np.random.choice(20)
        bin_candidates, bin_popularity = list(map(lambda x: x[bin_size * bin_selection: bin_size * (bin_selection + 1)], [candidates, mod_popularity]))

        # popular_negs = np.random.choice(candidates, (self.num_popular_negs, ), p = mod_popularity / sum(mod_popularity), replace=False)
        popular_negs = np.random.choice(bin_candidates, (self.num_popular_negs, ), p=bin_popularity / sum(bin_popularity), replace=False)

        num_random_negs = self.num_total_negs - self.num_popular_negs
        # random_negs = np.random.choice(np.delete(items_total, popular_negs.tolist() + [int(pos_pair[1])]), (num_random_negs, ), replace=False)
        random_negs = np.random.choice(candidates, (num_random_negs, ), replace=False)

        return popular_negs, random_negs
