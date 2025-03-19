import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sasrec.utils.layers import Basic_Block


class L2Rec(nn.Module):
    def __init__(self, config, num_user, num_item, device, normalize=True, beta_=0.05):
        super(L2Rec, self).__init__()
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

        self.normalize = normalize
        self.beta_ = beta_

        self.item_embedding = nn.Embedding(self.num_item, self.embed_size)

        self.positional_encoding = nn.Parameter(torch.empty(self.seq_len, self.embed_size))
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.1)

        self.layers = nn.ModuleList([Basic_Block(self.embed_size, self.n_heads, self.drop_out) for _ in range(self.n_layers)])

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

    def forward(self, user, pos, neg, history, history_mask, random_neg):
        pos_init_embed, neg_init_embed, ran_init_embed = tuple(map(lambda x: self.get_item_embedding(x), (pos, neg, random_neg)))

        history_embed = self.get_history_embedding(history, history_mask)

        total_loss, orig_loss = self._get_linked_loss(history_embed, pos_init_embed, neg_init_embed, ran_init_embed)
        unif_reg = torch.tensor(0).to(self.device)

        if self.normalize:
            unif_reg = self._get_unif_reg(history_embed, pos_init_embed, neg_init_embed, ran_init_embed)
            total_loss = total_loss + unif_reg

        if self.loss_type == "BPR":
            total_loss += self._get_raw_norm_reg(pos, neg, random_neg, history, history_mask)

        return (total_loss, orig_loss, unif_reg)

    def get_item_embedding(self, item):
        item_embedding = self.item_embedding(item)

        if self.normalize:
            item_embedding = F.normalize(item_embedding, p=2, dim=-1)

        return item_embedding

    def get_history_embedding(self, history, history_mask, return_att=False):
        history_embed = self._add_positional_embedding(self.get_item_embedding(history))

        if self.normalize:
            history_embed = F.normalize(history_embed, p=2, dim=-1)

        # attention_mask = self._get_sequence_mask(self.n_heads, history_mask, self.device)
        attention_mask = self._get_ones_mask(self.n_heads, history_mask, self.device)

        history_embed *= (history_mask.unsqueeze(-1))

        attentions_list = []
        for layer in self.layers:
            history_embed, attentions = layer(history_embed, mask=attention_mask)

            attentions_list.append((attentions * history_mask.unsqueeze(-1) * history_mask.unsqueeze(1)).unsqueeze(0))

            history_embed *= (history_mask.unsqueeze(-1))

        history_out = history_embed[:, -1, :]

        batch_attentions = torch.cat(attentions_list, dim=0)

        if self.normalize:
            history_out = F.normalize(history_out, p=2, dim=-1)

        if return_att:
            return history_out, batch_attentions
        return history_out

    def _get_linked_loss(self, history_embed, pos_item_embed, neg_item_embed, ran_item_embed):
        loss_func = {"BPR": self._compute_BPR, "BCE" : self._compute_BCE}

        left_args = {True: (history_embed, pos_item_embed, neg_item_embed), False: (history_embed, pos_item_embed, ran_item_embed)}
        right_args = {True: (history_embed, neg_item_embed, ran_item_embed), False: (history_embed, ran_item_embed, neg_item_embed)}

        total_loss = loss_func[self.loss_type](*left_args[self.forward_link])

        if self.link_loss:
            total_loss += loss_func[self.loss_type](*right_args[self.forward_link])

        bpr_term = loss_func[self.loss_type](history_embed, pos_item_embed, ran_item_embed)

        return total_loss, bpr_term

    @torch.inference_mode()
    def recommend(self, user, history, history_mask, item=None):
        history_embed = self.get_history_embedding(history, history_mask)

        if item is not None:
            item_init_embed = self.get_item_embedding(item)
        else:
            item_init_embed = self.item_embedding.weight

            if self.normalize:
                item_init_embed = F.normalize(item_init_embed, p=2, dim=-1)

        item_init_embed = item_init_embed.unsqueeze(0).expand(history.shape[0], -1, -1)

        out = torch.mul(history_embed.unsqueeze(1).expand(-1, item_init_embed.shape[1], -1), item_init_embed).sum(dim=-1)

        return out

    def _add_positional_embedding(self, history_embed):
        if self.normalize:
            pos_embed = F.normalize(self.positional_encoding, p=2, dim=-1).unsqueeze(0).expand(history_embed.shape[0], -1, -1)
        else:
            pos_embed = self.positional_encoding.unsqueeze(0).expand(history_embed.shape[0], -1, -1)

        history_embed += pos_embed

        return history_embed

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

    def _get_sequence_mask(self, n_heads, mask, device):
        batch_size, sequence_len = mask.shape[:2]

        mask_1 = mask.unsqueeze(1).expand(-1, sequence_len, -1)
        mask_2 = mask.unsqueeze(2).expand(-1, -1, sequence_len)

        sequence_mask = (mask_1 * mask_2).to(device)

        # sequence_mask = torch.tril(sequence_mask)

        attention_mask = torch.cat([sequence_mask for _ in range(n_heads)], dim=0)

        attention_mask.requires_grad = False

        return attention_mask

    def _get_ones_mask(self, n_heads, mask, device):
        batch_size, sequence_len = mask.shape[:2]

        ones_mask = torch.ones(sequence_len, sequence_len).unsqueeze(0).expand(batch_size, -1, -1).to(device)

        # ones_mask = torch.tril(ones_mask)

        attention_mask = torch.cat([ones_mask for _ in range(n_heads)], dim=0)

        attention_mask.requires_grad = False

        return attention_mask

    def _get_unif_reg(self, history_embed, pos_item_embed, neg_item_embed, ran_item_embed):
        item_args = (history_embed, pos_item_embed, ran_item_embed) + self.link_loss * (neg_item_embed,)
        unif_reg = self.beta_ * sum(list(map(lambda x: torch.pdist(x, p=2).pow(2).mul(-2.).exp().sum(), item_args)))

        unif_reg += self.beta_ * torch.cdist(history_embed, ran_item_embed, p=2).pow(2).mul(-2.).exp().sum()

        return unif_reg

    def _get_raw_norm_reg(self, pos, neg, random_neg, history, history_mask):
        pos_init_embed, neg_init_embed, ran_init_embed = tuple(map(lambda x: self.item_embedding(x), (pos, neg, random_neg)))

        history_embed = self._add_positional_embedding(self.get_item_embedding(history))
        attention_mask = self._get_sequence_mask(self.n_heads, history_mask, self.device)

        for layer in self.layers:
            history_embed = layer(history_embed, mask=attention_mask)

        history_out = history_embed[:, -1, :]

        raw_norm_reg = history_out.norm(dim=1).pow(2).sum() + pos_init_embed.norm(dim=1).pow(2).sum() + ran_init_embed.norm(dim=1).pow(2).sum()

        if self.link_loss:
            raw_norm_reg += neg_init_embed.norm(dim=1).pow(2).sum()

        norm_reg = self.weight_decay * raw_norm_reg

        return norm_reg
