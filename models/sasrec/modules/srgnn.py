import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GatedGraphConv


class Session2Embedding(nn.Module):
    def __init__(self, hidden_size):
        super(Session2Embedding, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, session_embedding, batch):
        sections = torch.bincount(batch)
        v_i = torch.split(session_embedding, tuple(sections.cpu().numpy()))    # split whole x back into graphs G_i
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in v_i)    # repeat |V|_i times for the last node embedding

        # Eq(6)
        alpha = self.q(torch.sigmoid(self.W_1(torch.cat(v_n_repeat, dim=0)) + self.W_2(session_embedding)))    # |V|_i * 1
        s_g_whole = alpha * session_embedding    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(sections.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        # Eq(7)
        v_n = tuple(nodes[-1].view(1, -1) for nodes in v_i)
        s_h = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))

        return s_h


class SRGNN(nn.Module):
    def __init__(self, config, num_user, num_item, device):
        super(SRGNN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.weight_decay = config["weight_decay"]
        self.embed_size = config["embed_size"]

        self.device = device

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.drop_out = config["drop_out"]
        self.seq_len = config["seq_len"]

        self.loss_type = config["loss_type"]
        self.init_scheme = config["init_scheme"]
        self.link_loss = config["link_loss"]
        self.forward_link = config["forward_link"]

        self.item_embedding = nn.Embedding(self.num_item, self.embed_size)
        self.gated = GatedGraphConv(self.embed_size, num_layers=2)
        self.ses2emb = Session2Embedding(self.embed_size)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                if self.init_scheme in ["Kaiming", "kaiming"]:
                    torch.nn.init.kaiming_normal_(m.weight)
                elif self.init_scheme in ["Xavier", "xavier"]:
                    torch.nn.init.xavier_normal_(m.weight)
                elif self.init_scheme in ["Normal", "normal"]:
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
            elif isinstance(m, nn.ReLU):
                m = nn.ReLU(inplace=False)

    def forward(self, user, pos, neg, data, random_neg):
        pos_init_embed, neg_init_embed = self.get_item_embedding(pos), self.get_item_embedding(random_neg)

        history_embed = self.get_history_embedding(data)

        total_loss, orig_loss, norm_reg, unif_reg = self._get_loss(history_embed, pos_init_embed, neg_init_embed)

        if self.loss_type != "BCE":
            # norm_reg = self.get_norm_reg(pos, random_neg, history, history_mask)
            total_loss = total_loss + norm_reg

        return (total_loss, orig_loss, unif_reg, unif_reg, norm_reg)

    def get_item_embedding(self, item):
        item_embedding = self.item_embedding(item)

        return item_embedding

    def get_history_embedding(self, data):
        x, edge_index, batch = data.x - 1, data.edge_index, data.batch

        embedding = self.item_embedding(x).squeeze()
        hidden = self.gated(embedding, edge_index)
        hidden2 = F.relu(hidden)

        return self.ses2emb(hidden2, batch)

    def _get_loss(self, history_embed, pos_item_embed, neg_item_embed):
        loss_func = {"BPR": self._compute_BPR, "BCE" : self._compute_BCE}

        his_pos_ran = loss_func[self.loss_type](history_embed, pos_item_embed, neg_item_embed)

        total_loss = his_pos_ran

        raw_norm_reg = history_embed.norm(dim=1).pow(2).sum() + pos_item_embed.norm(dim=1).pow(2).sum() + neg_item_embed.norm(dim=1).pow(2).sum()
        norm_reg = self.weight_decay * raw_norm_reg

        # unif_reg = self.beta_ * sum(list(map(lambda x: torch.pdist(x, p=2).pow(2).mul(-2.).exp().sum(), (history_embed, pos_item_embed, neg_item_embed))))

        # unif_reg += self.beta_ * torch.cdist(history_embed, neg_item_embed, p=2).pow(2).mul(-2.).exp().sum()

        return total_loss, his_pos_ran, norm_reg, norm_reg

    @torch.inference_mode()
    def recommend(self, user, sess, data, item=None):
        history_embed = self.get_history_embedding(data).unsqueeze(1)

        if item is None:
            item_init_embed = self.item_embedding.weight
        else:
            item_init_embed = self.get_item_embedding(item)

        item_init_embed = item_init_embed.unsqueeze(0).expand(user.shape[0], -1, -1)

        out = torch.mul(history_embed.expand(-1, item_init_embed.shape[1], -1), item_init_embed).sum(dim=-1)

        return out

    def _compute_BPR(self, history_embed, pos_embed, neg_embed):
        pos_preds = torch.mul(history_embed, pos_embed).sum(dim=-1, keepdims=True)
        neg_preds = torch.mul(history_embed, neg_embed).sum(dim=-1, keepdims=True)

        out = pos_preds - neg_preds

        log_prob = F.logsigmoid(out).sum()
        # log_prob = F.logsigmoid(out).mean()
        loss = -log_prob

        return loss

    def _compute_BCE(self, history_embed, pos_embed, neg_embed):
        pos_preds = torch.mul(history_embed, pos_embed).sum(dim=-1, keepdims=True)
        neg_preds = torch.mul(history_embed, neg_embed).sum(dim=-1, keepdims=True)

        preds = torch.cat([pos_preds, neg_preds], dim=-1)
        labels = torch.cat([torch.ones_like(pos_preds), torch.zeros_like(neg_preds)], dim=-1).to(self.device)

        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        loss = criterion(preds, labels)

        return loss

    def get_norm_reg(self, pos, neg, history, history_mask):
        pos_embed, neg_embed = self.item_embedding(pos), self.item_embedding(neg)

        history_embed = self.item_embedding(history) + self.positional_encoding.unsqueeze(0).expand(history.shape[0], -1, -1)

        attention_mask = self._get_sequence_mask(self.n_heads, history_mask, self.device)

        for layer in self.layers:
            history_embed = layer(history_embed, mask=attention_mask)

        history_out = history_embed[:, -1, :]

        raw_norm_reg = history_out.norm(dim=1).pow(2).sum() + pos_embed.norm(dim=1).pow(2).sum() + neg_embed.norm(dim=1).pow(2).sum()

        norm_reg = self.weight_decay * raw_norm_reg

        return norm_reg
