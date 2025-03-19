import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn


class BagOfWordsPretrained(nn.Module):
    def __init__(self, field, hidden_dims):
        super().__init__()

        input_dims = field.vocab.vectors.shape[1]
        self.emb = nn.Embedding(
            len(field.vocab.itos),
            input_dims,
            padding_idx=field.vocab.stoi[field.pad_token],
        )
        self.emb.weight[:] = field.vocab.vectors
        self.proj = nn.Linear(input_dims, hidden_dims)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)

        for param in self.emb.parameters():
            param.requires_grad = False

    def forward(self, x, length):
        x = self.emb(x).sum(1) / length.unsqueeze(1).float()
        return self.proj(x)


class BagOfWords(nn.Module):
    def __init__(self, field, hidden_dims):
        super().__init__()

        self.emb = nn.Embedding(
            len(field.vocab.itos),
            hidden_dims,
            padding_idx=field.vocab.stoi[field.pad_token],
        )
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x, length):
        return self.emb(x).sum(1) / length.unsqueeze(1).float()


class LinearProjector(nn.Module):
    """
    self.inputs : module dictionary
        that includes module for each feature.
    forward : projects each input feature of the graph linearly and sums them up.
        It represents feature embeddings of nodes.
    """

    def __init__(self, g, textset, hidden_dims):
        super().__init__()

        self.module_dict = self.init_modules(g, textset, hidden_dims)

    def init_modules(self, g, textset, hidden_dims):
        """We initialize the linear projections of each input feature ``x`` as
        follows:
        * If ``x`` is a scalar integral feature, we assume that ``x`` is a categorical
        feature, and assume the range of ``x`` is 0..max(x).
        * If ``x`` is a float one-dimentional feature, we assume that ``x`` is a
        numeric vector.
        * If ``x`` is a field of a textset, we process it as bag of words.
        """
        module_dict = nn.ModuleDict()

        for column, data in g.nodes["item_id"].data.items():
            if column == dgl.NID:
                continue
            if data.dtype == torch.float32:
                assert data.ndim == 2
                m = nn.Linear(data.shape[1], hidden_dims)
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                module_dict[column] = m
            elif data.dtype == torch.int64:
                assert data.ndim == 1
                m = nn.Embedding(data.max() + 2, hidden_dims, padding_idx=-1)
                nn.init.xavier_uniform_(m.weight)
                module_dict[column] = m

        if textset is not None:
            for column, field in textset.fields.items():
                if field.vocab.vectors:
                    module_dict[column] = BagOfWordsPretrained(field, hidden_dims)
                else:
                    module_dict[column] = BagOfWords(field, hidden_dims)

        return module_dict

    def forward(self, ndata):
        projections = []
        for feature, data in ndata.items():
            if feature == dgl.NID or feature.endswith("__len"):
                # This is an additional feature indicating the length of the ``feature``
                # column; we shouldn't process this.
                continue

            module = self.module_dict[feature]
            if isinstance(module, (BagOfWords, BagOfWordsPretrained)):
                # Textual feature; find the length and pass it to the textual module.
                length = ndata[feature + "__len"]
                result = module(data, length)
            else:
                result = module(data)
            projections.append(result)

        return torch.stack(projections, 1).sum(1)


class WeightedSAGEConv(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()

        self.act = F.relu
        self.Q = nn.Linear(input_dims, hidden_dims)
        self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        self.reset_parameters()
        self.dropout = nn.Dropout(0.5)

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, g, h, weights):
        """
        g : graph
        h : node features
        weights : scalar edge weights
        """
        h_src, h_dst = h
        with g.local_scope():
            g.srcdata["n"] = self.act(self.Q(self.dropout(h_src)))
            g.edata["w"] = weights.float()
            g.update_all(fn.u_mul_e("n", "w", "m"), fn.sum("m", "n"))
            g.update_all(fn.copy_e("w", "m"), fn.sum("m", "ws"))
            n = g.dstdata["n"]
            ws = g.dstdata["ws"].unsqueeze(1).clamp(min=1)
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0, torch.tensor(1.0).to(z_norm), z_norm)
            z = z / z_norm
            return z


class SAGENet(nn.Module):
    def __init__(self, hidden_dims, n_layers):
        """
        g : DGLHeteroGraph
            The user-item interaction graph.
            This is only for finding the range of categorical variables.
        item_textsets : torchtext.data.Dataset
            The textual features of each item node.
        """
        super().__init__()

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(WeightedSAGEConv(hidden_dims, hidden_dims, hidden_dims))

    def forward(self, blocks, h):
        for layer, block in zip(self.convs, blocks):
            h_dst = h[: block.number_of_nodes("DST/" + block.ntypes[0])]
            h = layer(block, (h, h_dst), block.edata["weights"])
        return h


class ItemToItemScorer(nn.Module):
    def __init__(self, full_graph):
        super().__init__()

        n_nodes = full_graph.number_of_nodes("productid")
        self.bias = nn.Parameter(torch.zeros(n_nodes, 1))

    def _add_bias(self, edges):
        bias_src = self.bias[edges.src[dgl.NID]]
        bias_dst = self.bias[edges.dst[dgl.NID]]
        return {"s": edges.data["s"] + bias_src + bias_dst}

    def forward(self, item_item_graph, h):
        """
        item_item_graph : graph consists of edges connecting the pairs
        h : hidden state of every node
        """
        with item_item_graph.local_scope():
            item_item_graph.ndata["h"] = h
            item_item_graph.apply_edges(fn.u_dot_v("h", "h", "s"))
            item_item_graph.apply_edges(self._add_bias)
            pair_score = item_item_graph.edata["s"]
        return pair_score


class PinSAGE(nn.Module):
    def __init__(self, config, graph, textset=None):
        super(PinSAGE, self).__init__()
        self.hidden_dims = config["hidden_dims"]
        self.num_layers = config["num_layers"]
        self.use_feature_embedding = config["use_feature_embedding"]

        if self.use_feature_embedding:
            assert (
                textset is not None
            ), "You have to put item text features to the model."
            self.proj = LinearProjector(graph, textset, self.hidden_dims)
        else:
            self.item_embedding = nn.Embedding(
                graph.num_nodes("productid"), self.hidden_dims
            )
            nn.init.xavier_uniform_(self.item_embedding.weight)

        self.sage = SAGENet(self.hidden_dims, self.num_layers)
        self.scorer = ItemToItemScorer(graph)

    def get_embedding(self, blocks):
        if self.use_feature_embedding:
            h_item = self.proj(blocks[0].srcdata)
            h_item_dst = self.proj(blocks[-1].dstdata)
            return h_item_dst + self.sage(blocks, h_item)
        else:
            h_item = self.item_embedding(blocks[0].srcdata[dgl.NID])
            h_item_dst = self.item_embedding(blocks[-1].dstdata[dgl.NID])
            return h_item_dst + self.sage(blocks, h_item)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_embedding(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)
