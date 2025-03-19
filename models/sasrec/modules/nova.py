import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sasrec.utils.layers import Nova_Block


class Novafuser(nn.Module):
    def __init__(self, config, num_user, num_item, device, num_meta, meta_information, fusion_method='gating'):
        super(Novafuser, self).__init__()
        self.num_user = num_user
        self.num_item = num_item

        self.weight_decay = config["weight_decay"]
        self.init_scale = config["init_scale"]
        self.embed_size = config["embed_size"]

        self.device = device

        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.dropout_ratio = config["dropout_ratio"]
        self.attn_dropout_ratio = config["attn_dropout_ratio"]
        self.seq_len = config["seq_len"]

        self.loss_type = config["loss_type"]
        self.init_scheme = config["init_scheme"]
        self.link_loss = config["link_loss"]
        self.forward_link = config["forward_link"]
        self.num_sides = config["num_sides"]

        self.gamma = config["gamma"]
        self.batchwise_sample = config["batchwise_sample"]

        self.normalize = config["normalize"]
        self.beta = config["beta"]
        self.embed_regularize = config["embed_regularize"]

        self.item_embedding = nn.Embedding(self.num_item, self.embed_size)

        self.meta_information = None
        if meta_information is not None and len(num_meta) > 0:
            self.meta_embedding = nn.ModuleList([nn.Embedding(per_, self.embed_size) for per_ in num_meta])
            self.meta_information = torch.LongTensor(meta_information).to(device)

        self.positional_encoding = nn.Parameter(torch.empty(self.seq_len, self.embed_size))
        nn.init.normal_(self.positional_encoding, mean=0.0, std=self.init_scale)

        layer_basic_args = (self.embed_size, self.num_sides, self.num_heads, self.dropout_ratio, self.attn_dropout_ratio)
        self.layers = nn.ModuleList([Nova_Block(*layer_basic_args, config["pre_ln"], fusion_method) for layer_idx in range(self.num_layers)])

        if not config["pre_ln"]:
            self.init_layers = nn.ModuleList([nn.Dropout(config["dropout_ratio"]), nn.LayerNorm(self.embed_size),])
        else:
            # self.init_layers = nn.ModuleList([nn.Dropout(config["dropout_ratio"]),])
            self.init_layers = nn.ModuleList([nn.Identity(),])

        # self.aap_classifier = nn.Linear(self.embed_size, num_meta[0])

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                if self.init_scheme in ["Kaiming", "kaiming"]:
                    torch.nn.init.kaiming_normal_(m.weight)
                elif self.init_scheme in ["Xavier", "xavier"]:
                    torch.nn.init.xavier_normal_(m.weight)
                elif self.init_scheme in ["Normal", "normal"]:
                    torch.nn.init.normal_(m.weight, mean=0.0, std=self.init_scale)
            elif isinstance(m, nn.ReLU):
                m = nn.ReLU(inplace=False)
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def get_loss_func(self, loss_name):
        return (loss_name, getattr(self, "_compute_" + loss_name))

    def forward(self, user, pos, neg, history, history_mask, random_neg=None):
        pos_embed, neg_embed, ran_embed = tuple(map(lambda x: self.get_item_embedding(x), (pos, neg, random_neg)))

        history_embed = self.get_history_embedding(history, history_mask)

        embeds = (pos_embed, neg_embed, ran_embed)

        loss_getter = {**dict.fromkeys(["BPR", "BCE"], self._compute_TR), **dict(list(map(self.get_loss_func, ["SCE", "LSCE", "LBPR_max"])))}
        loss_args = {**dict.fromkeys(["BPR", "BCE", "LSCE", "LBPR_max"], embeds), **dict.fromkeys(["SCE"], embeds[::2])}

        if self.loss_type in ["BPR", "BCE"]:
            total_loss, left, right = loss_getter[self.loss_type](history_embed, *loss_args[self.loss_type])
        else:
            total_loss = loss_getter[self.loss_type](history_embed, *loss_args[self.loss_type])
            left, right = (total_loss, total_loss)

        # aap_term = self._get_aap_loss(self.aap_classifier(ran_item_embed), self.meta_information[random_neg].squeeze())

        if self.normalize:
            unif_reg = self._get_unif_reg(history_embed, pos_embed, neg_embed, ran_embed)
            total_loss = total_loss + unif_reg

        if self.embed_regularize:
            norm_reg = self._get_norm_reg(history_embed, pos_embed, neg_embed, ran_embed)
            total_loss = total_loss + norm_reg

        return total_loss, left, right

    def get_item_embedding(self, item):
        item_embedding = self.item_embedding(item)

        if self.normalize:
            item_embedding = F.normalize(item_embedding, p=2, dim=-1)

        return item_embedding

    def get_history_embedding(self, history, history_mask, return_att=False):
        history_embed = self.get_item_embedding(history)
        positional_encoding = self.positional_encoding.unsqueeze(0).expand(history_embed.shape[0], -1, -1)

        attention_mask = self._get_sequence_mask(self.num_heads, history_mask, self.device)

        side_embed = positional_encoding.unsqueeze(2)
        if self.meta_information is not None:
            meta_history = self.meta_information[history]

            meta_embed = torch.cat([self.meta_embedding[idx](meta_history[..., idx:idx + 1]) for idx in range(len(self.meta_embedding))], dim=2)
            side_embed = torch.cat([meta_embed, positional_encoding.unsqueeze(2)], dim=2)

        if self.normalize:
            history_embed, side_embed = tuple(map(lambda x: F.normalize(x, p=2, dim=-1), (history_embed, side_embed)))

        for layer in self.layers:
            history_embed = layer(history_embed, side_embed, mask=attention_mask)

        history_out = history_embed[:, -1, :]

        if self.normalize:
            history_out = F.normalize(history_out, p=2, dim=-1)

        return history_out

    def _get_aap_loss(self, ran_item_pred, ran_item_meta):
        criterion = nn.CrossEntropyLoss()

        aap_term = self.beta * criterion(ran_item_pred, ran_item_meta)

        return aap_term

    @torch.inference_mode()
    def recommend(self, user, history, history_mask, item=None, return_out_only=False):
        history_embed = self.get_history_embedding(history, history_mask)

        if item is not None:
            item_embed = self.get_item_embedding(item)
        else:
            item_embed = self.item_embedding.weight

            if self.normalize:
                item_embed = F.normalize(item_embed, p=2, dim=-1)

        item_embed = item_embed.unsqueeze(0).expand(history.shape[0], -1, -1)

        out = torch.mul(history_embed.unsqueeze(1).expand(-1, item_embed.shape[1], -1), item_embed).sum(dim=-1)

        if return_out_only:
            return out
        return out, history_embed, item_embed

    def _get_preds(self, left_term, right_term):
        if len(right_term.shape) == 2:
            score = torch.mul(left_term, right_term).sum(dim=-1, keepdims=True)
        elif len(right_term.shape) == 3:
            # score = torch.mean(torch.mul(left_term.unsqueeze(1), right_term).sum(dim=-1, keepdims=True), dim=1)
            score = torch.mul(left_term.unsqueeze(1), right_term).sum(dim=-1)

        return score

    def _compute_TR(self, history_embed, pos_embed, neg_embed, ran_embed):
        loss_func = {"BPR": self._compute_BPR, "BCE" : self._compute_BCE}

        left_args = {True: (history_embed, pos_embed, neg_embed), False: (history_embed, pos_embed, ran_embed)}
        right_args = {True: (history_embed, neg_embed, ran_embed), False: (history_embed, ran_embed, neg_embed)}

        left = loss_func[self.loss_type](*left_args[self.forward_link])

        right = 0
        if self.link_loss:
            right = self.gamma * loss_func[self.loss_type](*right_args[self.forward_link])

        total_loss = left + right

        return total_loss, left, right

    def _compute_BPR(self, history_embed, pos_embed, neg_embed):
        pos_preds, neg_preds = tuple(map(self._get_preds, (history_embed,) * 2, (pos_embed, neg_embed)))

        out = pos_preds - neg_preds

        log_prob = F.logsigmoid(out).sum()
        loss = -log_prob

        return loss

    def _compute_BCE(self, history_embed, pos_embed, neg_embed):
        pos_preds, neg_preds = tuple(map(self._get_preds, (history_embed,) * 2, (pos_embed, neg_embed)))

        preds = torch.cat([pos_preds, neg_preds], dim=-1)
        labels = torch.cat([torch.ones_like(pos_preds), torch.zeros_like(neg_preds)], dim=-1).to(self.device)

        criterion = nn.BCEWithLogitsLoss(reduction='sum')
        loss = criterion(preds, labels)

        return loss

    def _compute_SCE(self, history_embed, pos_embed, neg_embed):
        item_embed = torch.cat([pos_embed.unsqueeze(1), neg_embed], dim=1)
        preds = torch.mul(history_embed.unsqueeze(1).expand(-1, item_embed.shape[1], -1), item_embed).sum(dim=-1)
        labels = torch.zeros(history_embed.shape[0]).long().to(self.device)

        pos_preds, neg_preds = tuple(map(self._get_preds, (history_embed,) * 2, (pos_embed.unsqueeze(1), neg_embed)))

        criterion = nn.CrossEntropyLoss(reduction='sum')
        loss = criterion(preds, labels)

        return loss

    def _compute_LSCE(self, history_embed, pos_embed, neg_embed, ran_embed):
        pos_preds, neg_preds, ran_preds = tuple(map(self._get_preds, (history_embed,) * 3, (pos_embed.unsqueeze(1), neg_embed, ran_embed)))

        larg, rarg = (neg_preds, ran_preds) if self.forward_link else (ran_preds, neg_preds)

        left_preds = torch.cat([pos_preds, larg, rarg], dim=-1)
        # left_preds = torch.cat([pos_preds, larg], dim=-1)
        left_labels = torch.zeros(history_embed.shape[0]).long().to(self.device)

        criterion = nn.CrossEntropyLoss(reduction='sum')
        left_loss = criterion(left_preds, left_labels)

        right_preds = torch.cat([larg.unsqueeze(-1), rarg.unsqueeze(1).expand(-1, larg.shape[-1], -1)], dim=-1)
        mp_loss = torch.sum(torch.mean(torch.logsumexp(right_preds, dim=-1) - torch.log(torch.exp(larg)), dim=-1))

        # mil_loss = torch.sum(torch.logsumexp(torch.cat([neg_preds, ran_preds], dim=-1), dim=-1) - torch.logsumexp(neg_preds, dim=-1))

        loss = left_loss + self.gamma * mp_loss

        return loss

    def _compute_LBPR_max(self, history_embed, pos_embed, neg_embed, ran_embed):
        pos_preds, neg_preds, ran_preds = tuple(map(self._get_preds, (history_embed,) * 3, (pos_embed.unsqueeze(1), neg_embed, ran_embed)))

        larg, rarg = (neg_preds, ran_preds) if self.forward_link else (ran_preds, neg_preds)
        arg = torch.cat([neg_preds, ran_preds], dim=-1)

        left_score_weights = F.softmax(arg, dim=-1)

        left_out = pos_preds.expand(-1, arg.shape[-1]) - arg
        left_loss = -torch.log(torch.sum(left_score_weights * torch.sigmoid(left_out), dim=-1, keepdims=True)).sum()

        right_score_weights = F.softmax(rarg, dim=-1).unsqueeze(1).expand(-1, larg.shape[1], -1)

        right_out = larg.unsqueeze(-1).expand(-1, -1, rarg.shape[-1]) - rarg.unsqueeze(1).expand(-1, larg.shape[-1], -1)
        right_loss = -torch.log(torch.mean(torch.sum(right_score_weights * torch.sigmoid(right_out), dim=-1), dim=-1)).sum()

        loss = left_loss + right_loss

        return loss

    def _get_sequence_mask(self, num_heads, mask, device):
        batch_size, sequence_len = mask.shape[:2]

        mask_1 = mask.unsqueeze(1).expand(-1, sequence_len, -1)
        mask_2 = mask.unsqueeze(2).expand(-1, -1, sequence_len)

        sequence_mask = (mask_1 * mask_2).to(device)

        # sequence_mask = torch.tril(sequence_mask)

        attention_mask = torch.cat([sequence_mask for _ in range(num_heads)], dim=0)

        attention_mask.requires_grad = False

        return attention_mask

    def _get_unif_reg(self, history_embed, pos_item_embed, neg_item_embed, ran_item_embed):
        item_args = (history_embed, pos_item_embed, ran_item_embed) + self.link_loss * (neg_item_embed,)
        unif_reg = self.beta * sum(list(map(lambda x: torch.pdist(x, p=2).pow(2).mul(-2.).exp().sum(), item_args)))

        unif_reg += self.beta * torch.cdist(history_embed, ran_item_embed, p=2).pow(2).mul(-2.).exp().sum()

        return unif_reg

    def _get_norm_reg(self, history_embed, pos_item_embed, neg_item_embed, ran_item_embed):
        raw_norm_reg = history_embed.norm(dim=1).pow(2).sum() + pos_item_embed.norm(dim=1).pow(2).sum() + ran_item_embed.norm(dim=1).pow(2).sum()

        if self.link_loss:
            raw_norm_reg += neg_item_embed.norm(dim=1).pow(2).sum()

        norm_reg = self.weight_decay * raw_norm_reg

        return norm_reg
