import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, config, num_device, num_product, graph_matrix):
        super(LightGCN, self).__init__()
        self.num_device = num_device
        self.num_product = num_product
        self.weight_decay = config["weight_decay"]
        self.embed_size = config["embed_size"]
        self.num_gc = config["num_gc"]

        self.item_embedding = nn.Embedding(self.num_product, self.embed_size)
        self.user_embedding = nn.Embedding(self.num_device, self.embed_size)
        self.user_item_norm, self.item_user_norm = graph_matrix
        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, user, pos, neg):
        user_embed = self.user_embedding(user)
        pos_embed = self.item_embedding(pos)
        neg_embed = self.item_embedding(neg)

        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        user_outputs = [user_embed]
        pos_outputs = [pos_embed]
        neg_outputs = [neg_embed]
        for _ in range(self.num_gc):
            user_embeddings = torch.sparse.mm(self.user_item_norm, item_embeddings)
            item_embeddings = torch.sparse.mm(self.item_user_norm, user_embeddings)
            user_outputs.append(user_embeddings[user])
            pos_outputs.append(item_embeddings[pos])
            neg_outputs.append(item_embeddings[neg])
        user_outputs = torch.stack(user_outputs, dim=1).mean(dim=1)
        pos_outputs = torch.stack(pos_outputs, dim=1).mean(dim=1)
        neg_outputs = torch.stack(neg_outputs, dim=1).mean(dim=1)

        pos_out = torch.mul(user_outputs, pos_outputs).sum(dim=1)
        neg_out = torch.mul(user_outputs, neg_outputs).sum(dim=1)

        # out = neg_out - pos_out
        # loss = torch.mean(F.softplus(out))

        out = pos_out - neg_out
        loss = F.logsigmoid(out).sum()
        reg_loss = self.weight_decay * (
            0.5
            * (
                user_embed.norm().pow(2)
                + pos_embed.norm().pow(2)
                + neg_embed.norm().pow(2)
            )
            / float(self.num_device)
        )

        return -loss + reg_loss

    def recommend(self, user):
        user_embed = self.user_embedding(user)

        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        user_outputs = [user_embed]
        item_outputs = [item_embeddings]
        for _ in range(self.num_gc):
            user_embeddings = torch.sparse.mm(self.user_item_norm, item_embeddings)
            item_embeddings = torch.sparse.mm(self.item_user_norm, user_embeddings)
            user_outputs.append(user_embeddings[user])
            item_outputs.append(item_embeddings)
        user_outputs = torch.stack(user_outputs, dim=1).mean(dim=1).unsqueeze(1)
        item_outputs = torch.stack(item_outputs, dim=1).mean(dim=1)

        out = torch.mul(user_outputs, item_outputs).sum(dim=-1)
        return out
