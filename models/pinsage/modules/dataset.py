import dgl
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset


def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    for col in g.nodes[ntype].data.keys():
        if not assign_id and col == dgl.NID:
            continue
        induced_nodes = ndata[dgl.NID]
        ndata[col] = g.nodes[ntype].data[col][induced_nodes]


def assign_textual_node_features(ndata, textset, ntype):
    node_ids = ndata[dgl.NID].numpy()

    for field_name, field in textset.fields.items():
        examples = [getattr(textset[i], field_name) for i in node_ids]

        tokens, lengths = field.process(examples)

        if not field.batch_first:
            tokens = tokens.t()

        ndata[field_name] = tokens
        ndata[field_name + "__len"] = lengths


def assign_features_to_blocks(blocks, g, textset, ntype):
    assign_simple_node_features(blocks[0].srcdata, g, ntype)
    assign_textual_node_features(blocks[0].srcdata, textset, ntype)
    assign_simple_node_features(blocks[-1].dstdata, g, ntype)
    assign_textual_node_features(blocks[-1].dstdata, textset, ntype)


class PinSAGEData(IterableDataset):
    def __init__(self, data_dict, batch_size):
        self.train_df = data_dict["train_df"]
        self.num_device = data_dict["num_device"]
        self.num_product = data_dict["num_product"]

        self.device_meta = data_dict.get("device_meta", None)
        self.product_meta = data_dict.get("product_meta", None)

        self.batch_size = batch_size
        self.graph = self.build()

    def add_node_meta(self, graph):
        for col in self.device_meta.columns:
            if col == "deviceid_model_index":
                continue
            graph.nodes["deviceid"].data[col] = torch.LongTensor(
                self.device_meta[col].cat.codes.values
            )

        for col in self.product_meta.columns:
            if col == "productid_model_index":
                continue
            graph.nodes["productid"].data[col] = torch.LongTensor(
                self.product_meta[col].cat.codes.values
            )

        return graph

    def add_edge_meta(self, graph):
        graph.edges["device2product"].data["timestamp"] = torch.LongTensor(
            self.train_df["timestamp"].values
        )
        graph.edges["product2device"].data["timestamp"] = torch.LongTensor(
            self.train_df["timestamp"].values
        )

        return graph

    def build(self):
        num_nodes = {"deviceid": self.num_device, "productid": self.num_product}
        edges = {}
        edges[("deviceid", "device2product", "productid")] = (
            self.train_df["deviceid_model_index"].values.astype("int64"),
            self.train_df["productid_model_index"].values.astype("int64"),
        )
        edges[("productid", "product2device", "deviceid")] = (
            self.train_df["productid_model_index"].values.astype("int64"),
            self.train_df["deviceid_model_index"].values.astype("int64"),
        )
        graph = dgl.heterograph(edges, num_nodes)
        # graph = self.add_edge_meta(graph)
        return graph

    def __iter__(self):
        while True:
            anc_nodes = torch.randint(
                1, self.graph.number_of_nodes("productid"), (self.batch_size,)
            )
            pos_nodes = dgl.sampling.random_walk(
                self.graph, anc_nodes, metapath=["product2device", "device2product"]
            )[0][:, 2]
            neg_nodes = torch.randint(
                1, self.graph.number_of_nodes("productid"), (self.batch_size,)
            )

            mask = pos_nodes != -1
            yield anc_nodes[mask], pos_nodes[mask], neg_nodes[mask]


class PinSAGETestData(Dataset):
    def __init__(self, data_dict):
        super(PinSAGETestData, self).__init__()
        self.test_data = data_dict["test_df"][
            ["deviceid_model_index", "productid_model_index", "occurence"]
        ].values.astype(np.int32)
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
            else:
                query = 0
        else:
            query = 0
        mask = np.zeros(self.num_product)
        mask[valid_history] = -10000

        return u, query, gt, mask, occurence


class NeighborSampler:
    def __init__(
        self,
        graph,
        random_walk_length,
        random_walk_restart_prob,
        num_random_walks,
        num_neighbors,
        num_layers,
    ):
        self.graph = graph
        self.samplers = [
            dgl.sampling.PinSAGESampler(
                graph,
                "productid",
                "deviceid",
                random_walk_length,
                random_walk_restart_prob,
                num_random_walks,
                num_neighbors,
            )
            for _ in range(num_layers)
        ]

    def compact_and_copy(self, frontier, seeds):
        """
        Convert graph information for message passing. src node -> dst node.
        """
        block = dgl.to_block(frontier, seeds)
        for col, data in frontier.edata.items():
            if col == dgl.EID:
                continue
            block.edata[col] = data[block.edata[dgl.EID]]
        return block

    def sample_blocks(self, seeds, anc_nodes=None, pos_nodes=None, neg_nodes=None):
        blocks = []
        for sampler in self.samplers:
            frontier = sampler(seeds)
            if anc_nodes is not None:
                eids = frontier.edge_ids(
                    torch.cat([anc_nodes, anc_nodes, seeds]),
                    torch.cat([pos_nodes, neg_nodes, seeds]),
                    return_uv=True,
                )[2]
                if len(eids) > 0:
                    frontier = dgl.remove_edges(frontier, eids)
            block = self.compact_and_copy(frontier, seeds)
            blocks.insert(0, block)
            seeds = block.srcdata[dgl.NID]
        return blocks

    def sample_from_item_pairs(self, anc_nodes, pos_nodes, neg_nodes):
        pos_graph = dgl.graph(
            (anc_nodes, pos_nodes), num_nodes=self.graph.number_of_nodes("productid")
        )
        neg_graph = dgl.graph(
            (anc_nodes, neg_nodes), num_nodes=self.graph.number_of_nodes("productid")
        )
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]

        blocks = self.sample_blocks(seeds, anc_nodes, pos_nodes, neg_nodes)
        return pos_graph, neg_graph, blocks


class PinSAGECollator:
    def __init__(self, sampler, graph, textset=None):
        self.sampler = sampler
        self.graph = graph
        self.textset = textset

    def collate_train(self, batches):
        anc_nodes, pos_nodes, neg_nodes = batches[0]
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(
            anc_nodes, pos_nodes, neg_nodes
        )

        if self.textset is not None:
            assign_features_to_blocks(blocks, self.graph, self.textset, "productid")

        return pos_graph, neg_graph, blocks

    def collate_test(self, samples):
        batch = torch.LongTensor(samples)
        blocks = self.sampler.sample_blocks(batch)

        if self.textset is not None:
            assign_features_to_blocks(blocks, self.graph, self.textset, "productid")
        return blocks
