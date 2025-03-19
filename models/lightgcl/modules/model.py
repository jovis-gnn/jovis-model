import torch
import torch.nn as nn

from ..utils.helper import sparse_dropout


class LightGCL(nn.Module):
    def __init__(self, config, num_device, num_product, adj_matrix):
        super().__init__()

        self.num_device = num_device
        self.num_product = num_product

        self.weight_decay = config["weight_decay"]
        self.embed_size = config["embed_size"]
        self.num_gc = config["num_gc"]
        self.cl_loss_weight = config["cl_loss_weight"]
        self.l2_reg_weight = config["l2_reg_weight"]
        self.dropout_rate = config["dropout_rate"]
        self.temperature = config["temperature"]
        self.svd_q = config["svd_q"]
        self.num_layers = config["num_layers"]

        self.adj_matrix = adj_matrix

        self.prepare_embeddings_and_svd()

    def prepare_embeddings_and_svd(self):
        user_data = torch.empty(self.num_device, self.embed_size)
        user_data = nn.init.xavier_uniform_(user_data)
        self.user_embedding_0 = nn.Parameter(user_data)

        item_data = torch.empty(self.num_product, self.embed_size)
        item_data = nn.init.xavier_uniform_(item_data)
        self.item_embedding_0 = nn.Parameter(item_data)

        self.user_embedding_list = [None] * (self.num_layers + 1)
        self.item_embedding_list = [None] * (self.num_layers + 1)
        self.user_embedding_list[0] = self.user_embedding_0
        self.item_embedding_list[0] = self.item_embedding_0

        self.user_local_agg_repr_list = [None] * (self.num_layers + 1)
        self.item_local_agg_repr_list = [None] * (self.num_layers + 1)

        self.user_svd_agg_repr_list = [None] * (self.num_layers + 1)
        self.item_svd_agg_repr_list = [None] * (self.num_layers + 1)
        self.user_svd_agg_repr_list[0] = self.user_embedding_0
        self.item_svd_agg_repr_list[0] = self.item_embedding_0

        self.user_embedding = None
        self.item_embedding = None

        u_matrix, sigma_matrix, v_matrix = torch.svd_lowrank(
            self.adj_matrix,
            q=self.svd_q,
        )
        self.u_matrix = u_matrix
        self.sigma_matrix = sigma_matrix
        self.v_matrix = v_matrix

        # Pre-compute matrix multiplications for efficiency.
        self.u_times_sigma = u_matrix @ torch.diag(sigma_matrix)
        self.v_times_sigma = v_matrix @ torch.diag(sigma_matrix)

    def _init_weight(self):
        nn.init.normal_(self.user_embedding_0.weight, std=0.1)
        nn.init.normal_(self.item_embedding_0.weight, std=0.1)

    def forward(self, user, item, pos, neg):
        # Get representations across layers.
        for layer in range(1, self.num_layers + 1):
            adj_matrix_dropout = sparse_dropout(self.adj_matrix, self.dropout_rate)

            self.user_local_agg_repr_list[layer] = torch.spmm(
                adj_matrix_dropout,
                self.item_embedding_list[layer - 1],
            )
            self.item_local_agg_repr_list[layer] = torch.spmm(
                adj_matrix_dropout.T,
                self.user_embedding_list[layer - 1],
            )

            v_times_item_embs = self.v_matrix.T @ self.item_embedding_list[layer - 1]
            self.user_svd_agg_repr_list[layer] = self.u_times_sigma @ v_times_item_embs

            u_times_user_embs = self.u_matrix.T @ self.user_embedding_list[layer - 1]
            self.item_svd_agg_repr_list[layer] = self.v_times_sigma @ u_times_user_embs

            self.user_embedding_list[layer] = self.user_local_agg_repr_list[layer]
            self.item_embedding_list[layer] = self.item_local_agg_repr_list[layer]

        self.user_svd_agg_repr = sum(self.user_svd_agg_repr_list)
        self.item_svd_agg_repr = sum(self.item_svd_agg_repr_list)

        self.user_embedding = sum(self.user_embedding_list)
        self.item_embedding = sum(self.item_embedding_list)

        # Contrastive loss.
        user_product = (
            self.user_svd_agg_repr[user] @ self.user_embedding.T
        ) / self.temperature
        max_user_value = torch.max(user_product)
        user_neg_score = (
            torch.log(torch.exp(user_product - max_user_value).sum(1) + 1e-8)
            + max_user_value
        )
        user_neg_score = user_neg_score.mean()

        item_product = (
            self.item_svd_agg_repr[item] @ self.item_embedding.T
        ) / self.temperature
        max_item_value = torch.max(item_product)
        item_neg_score = (
            torch.log(torch.exp(item_product - max_item_value).sum(1) + 1e-8)
            + max_item_value
        )
        item_neg_score = item_neg_score.mean()

        neg_score = user_neg_score + item_neg_score

        user_pos_score = torch.clamp(
            (self.user_svd_agg_repr[user] * self.user_embedding[user]).sum(1)
            / self.temperature,
            min=-5.0,
            max=5.0,
        ).mean()
        item_pos_score = torch.clamp(
            (self.item_svd_agg_repr[item] * self.item_embedding[item]).sum(1)
            / self.temperature,
            min=-5.0,
            max=5.0,
        ).mean()
        pos_score = user_pos_score + item_pos_score

        contrastive_loss = -pos_score + neg_score

        # BPR loss.
        user_embs = self.user_embedding[user]
        pos_embs = self.item_embedding[pos]
        neg_embs = self.item_embedding[neg]

        pos_scores = (user_embs * pos_embs).sum(-1)
        neg_scores = (user_embs * neg_embs).sum(-1)

        bpr_loss = -(pos_scores - neg_scores).sigmoid().log().mean()

        # Regularization loss.
        regularization_loss = 0
        for param in self.parameters():
            regularization_loss += param.norm(2).square()

        total_loss = (
            bpr_loss
            + (contrastive_loss * self.cl_loss_weight)
            + (regularization_loss * self.l2_reg_weight)
        )

        return total_loss, contrastive_loss, bpr_loss, regularization_loss

    def recommend(self, user: torch.Tensor) -> torch.Tensor:
        if self.user_embedding is not None:
            user_embed = self.user_embedding[user]
            preds = user_embed @ self.item_embedding.T
        else:
            user_embed = self.user_embedding_0[user]
            preds = user_embed @ self.item_embedding_0.T

        return preds
