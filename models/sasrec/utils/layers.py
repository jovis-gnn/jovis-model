import torch
import torch.nn as nn
# import torch.nn.functional as F

import math


class Decoupled_Block(nn.Module):
    def __init__(self, dim_H, num_sides=1, num_heads=4, dropout_ratio=0.0, attn_dropout_ratio=0.0, pre_ln=False):
        super(Decoupled_Block, self).__init__()

        self.mha = Decoupled_MHA(dim_H, num_sides, num_heads, dropout_ratio, attn_dropout_ratio, pre_ln)
        self.ff = FF(dim_H, dropout_ratio, pre_ln)

        if pre_ln:
            self.ln_mha = nn.Identity()
            self.ln_ff = nn.Identity()
        else:
            self.ln_mha = nn.LayerNorm(dim_H)
            self.ln_ff = nn.LayerNorm(dim_H)

    def forward(self, id, side, mask=None):
        out = self.mha(id, side, mask)
        out = self.ln_mha(out)

        out = self.ff(out)
        out = self.ln_ff(out)

        return out


class Decoupled_MHA(nn.Module):
    def __init__(self, dim_H, num_sides=1, num_heads=4, dropout_ratio=0.0, attn_dropout_ratio=0.0, pre_ln=False):
        super(Decoupled_MHA, self).__init__()
        self.dim_H = dim_H
        self.num_heads = num_heads
        self.num_sides = num_sides

        if pre_ln:
            ln_func = nn.LayerNorm(dim_H)
        else:
            ln_func = nn.Identity()

        self.id_layers = nn.ModuleDict(
            {
                "fc_q": nn.Linear(dim_H, dim_H),
                "fc_k": nn.Linear(dim_H, dim_H),
                "fc_v": nn.Linear(dim_H, dim_H),
                "ln": ln_func(dim_H),
            }
        )

        def per_side_layer(idx):
            return nn.ModuleDict(
                {
                    f"{idx}_fc_q": nn.Linear(dim_H, dim_H),
                    f"{idx}_fc_k": nn.Linear(dim_H, dim_H),
                    f"{idx}_ln": ln_func(dim_H),
                }
            )

        self.side_layers = nn.ModuleList(
            [per_side_layer(idx) for idx in range(num_sides)]
        )

        self.fc_o = nn.Linear(dim_H, dim_H)
        self.dropout = nn.Dropout(dropout_ratio)
        self.attn_dropout = nn.Dropout(attn_dropout_ratio)

    """ id : (batch_size, seq_len, dim_H) / side : (batch_size, seq_len, num_sides, dim_H) """

    def forward(self, id, side, mask=None):
        batch, seq_len, _ = id.size()
        _, _, num_sides, _ = side.size()

        residual = id
        id = self.id_layers["ln"](id)
        side = torch.cat(
            [
                self.side_layers[idx][f"{idx}_ln"](side[:, :, idx, :]).unsqueeze(2)
                for idx in range(num_sides)
            ],
            2,
        )

        id_Q, id_K, id_V = (
            self.id_layers["fc_q"](id),
            self.id_layers["fc_k"](id),
            self.id_layers["fc_v"](id),
        )
        side_Q = torch.cat(
            [
                self.side_layers[idx][f"{idx}_fc_q"](side[:, :, idx, :]).unsqueeze(2)
                for idx in range(num_sides)
            ],
            2,
        )
        side_K = torch.cat(
            [
                self.side_layers[idx][f"{idx}_fc_k"](side[:, :, idx, :]).unsqueeze(2)
                for idx in range(num_sides)
            ],
            2,
        )

        split_H = self.dim_H // self.num_heads

        id_Q_, id_K_, id_V_ = map(
            lambda target: torch.cat(target.split(split_H, -1), 0), (id_Q, id_K, id_V)
        )
        side_Q_ = torch.cat(
            [
                torch.cat(side_Q[:, :, idx, :].split(split_H, -1), 0).unsqueeze(2)
                for idx in range(num_sides)
            ],
            2,
        )
        side_K_ = torch.cat(
            [
                torch.cat(side_K[:, :, idx, :].split(split_H, -1), 0).unsqueeze(2)
                for idx in range(num_sides)
            ],
            2,
        )

        id_A_ = self._get_attn(id_Q_, id_K_, mask)
        side_A_ = torch.cat(
            [
                self._get_attn(side_Q_[:, :, idx, :], side_K_[:, :, idx, :], mask)
                for idx in range(num_sides)
            ],
            2,
        )

        A_ = torch.cat([id_A_, side_A_], 2)
        fused_A_ = torch.max(A_, 2)

        O_ = torch.cat(fused_A_.bmm(id_V_).split(id.size(0), 0), 2)
        # O = residual + F.relu(self.fc_o(O_))
        O__ = residual + self.dropout(self.fc_o(O_))

        return O__

    def _get_attn(self, Q_, K_, mask=None):
        raw_A_ = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_H)

        if mask is not None:
            raw_A_ = raw_A_.masked_fill(mask == 0, -1e8)

        # A_ = torch.softmax(raw_A_, -1).unsqueeze(2)
        A_ = self.attn_dropout(torch.softmax(raw_A_, -1)).unsqueeze(2)

        return A_


class Nova_Block(nn.Module):
    def __init__(self, dim_H, num_sides=1, num_heads=4, dropout_ratio=0.0, attn_dropout_ratio=0.0, pre_ln=False, fusion_method='mean'):
        super(Nova_Block, self).__init__()

        self.mha = Nova_MHA(dim_H, num_sides, num_heads, dropout_ratio, attn_dropout_ratio, pre_ln, fusion_method)
        self.ff = FF(dim_H, dropout_ratio, pre_ln)

        if pre_ln:
            self.ln_mha = nn.Identity()
            self.ln_ff = nn.Identity()
        else:
            self.ln_mha = nn.LayerNorm(dim_H)
            self.ln_ff = nn.LayerNorm(dim_H)

    def forward(self, id, side, mask=None):
        out, _ = self.mha(id, side, mask)
        out = self.ln_mha(out)

        out = self.ff(out)
        out = self.ln_ff(out)

        return out


class Nova_MHA(nn.Module):
    def __init__(self, dim_H, num_sides=1, num_heads=4, dropout_ratio=0.0, attn_dropout_ratio=0.0, pre_ln=False, fusion_method='mean'):
        super(Nova_MHA, self).__init__()
        self.dim_H = dim_H
        self.num_heads = num_heads
        self.num_sides = num_sides

        self.fc_q = nn.Linear(dim_H, dim_H, bias=False)
        self.fc_k = nn.Linear(dim_H, dim_H, bias=False)
        self.fc_v = nn.Linear(dim_H, dim_H, bias=False)

        self.fc_o = nn.Linear(dim_H, dim_H)
        self.dropout = nn.Dropout(dropout_ratio)
        self.attn_dropout = nn.Dropout(attn_dropout_ratio)

        if pre_ln:
            self.id_ln = nn.LayerNorm(dim_H)
            self.side_lns = nn.ModuleList(
                [
                    nn.ModuleDict({f"{idx}_ln": nn.LayerNorm(dim_H)})
                    for idx in range(num_sides)
                ]
            )
        else:
            self.id_ln = nn.Identity()
            self.side_lns = nn.Identity()

        self.fusion_method = fusion_method

        if fusion_method == 'linear':
            self.fc_f = nn.Linear(dim_H * (num_sides + 1), dim_H, bias=False)
        elif fusion_method == 'gating':
            # self.fc_g = nn.Linear(dim_H, 1, bias=False)

            self.act = nn.GELU()
            self.fc_f = nn.Linear(dim_H, dim_H * 2, bias=False)

    """ id : (batch_size, seq_len, dim_H) / side : (batch_size, seq_len, num_sides, dim_H) """

    def forward(self, id, side, mask=None):
        batch, seq_len, _ = id.size()
        _, _, num_sides, _ = side.size()

        residual = id
        id = self.id_ln(id)
        side = torch.cat(
            [
                self.side_lns[idx][f"{idx}_ln"](side[:, :, idx, :]).unsqueeze(2)
                for idx in range(num_sides)
            ],
            2,
        )

        if self.fusion_method == 'mean':
            fused_id = torch.mean(torch.cat([id.unsqueeze(2), side], dim=2), dim=2)
        elif self.fusion_method == 'max':
            fused_id = torch.max(torch.cat([id.unsqueeze(2), side], dim=2), dim=2)[0]
        elif self.fusion_method == 'linear':
            fused_id = self.fc_f(torch.reshape(torch.cat([id.unsqueeze(2), side], 2), (batch, seq_len, -1)))
        elif self.fusion_method == 'gating':
            # gating = torch.sigmoid(self.fc_f(torch.cat([id.unsqueeze(2), side], 2)))
            # fused_id = torch.sum(gating * torch.cat([id.unsqueeze(2), side], 2), 2)

            gating, value = self.fc_f(torch.cat([id.unsqueeze(2), side], 2)).chunk(2, dim=-1)
            fused_id = torch.sum(self.act(gating) * value, 2)

        fused_Q, fused_K = self.fc_q(fused_id), self.fc_k(fused_id)
        id_V = self.fc_v(id)

        split_H = self.dim_H // self.num_heads
        Q_ = torch.cat(fused_Q.split(split_H, 2), 0)
        K_ = torch.cat(fused_K.split(split_H, 2), 0)
        V_ = torch.cat(id_V.split(split_H, 2), 0)

        A_ = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_H)

        if mask is not None:
            A_ = A_.masked_fill(mask == 0, -1e8)

        # A = torch.softmax(A_, -1)
        A = self.attn_dropout(torch.softmax(A_, -1))

        O_ = torch.cat(A.bmm(V_).split(id.size(0), 0), 2)
        # O = residual + F.relu(self.fc_o(O_))
        O__ = residual + self.dropout(self.fc_o(O_))

        # return O__
        return O__, A


class ResiDual_Block(nn.Module):
    def __init__(self, dim_H, num_heads=8, dropout_ratio=0.0, attn_dropout_ratio=0.0, pre_ln=False):
        super(ResiDual_Block, self).__init__()
        self.mha = ResiDual_MHA(dim_H, num_heads, dropout_ratio, attn_dropout_ratio)
        self.ff = FF(dim_H, dropout_ratio, pre_ln)

        if pre_ln:
            self.ln_ff = nn.Identity()
        else:
            self.ln_ff = nn.LayerNorm(dim_H)

    def forward(self, x, mask=None):
        out, _ = self.mha(x, mask)

        out = self.ff(out)
        out = self.ln_ff(out)

        return out


class ResiDual_MHA(nn.Module):
    def __init__(self, dim_H, num_heads=8, dropout_ratio=0.0, attn_dropout_ratio=0.0):
        super(ResiDual_MHA, self).__init__()
        self.dim_H = dim_H
        self.num_heads = num_heads

        self.fc_q = nn.Linear(dim_H, dim_H, bias=False)
        self.fc_k = nn.Linear(dim_H, dim_H, bias=False)
        self.fc_v = nn.Linear(dim_H, dim_H, bias=False)

        self.fc_o = nn.Linear(dim_H, dim_H, bias=False)
        self.dropout = nn.Dropout(dropout_ratio)
        self.attn_dropout = nn.Dropout(attn_dropout_ratio)

        self.ln_start = nn.LayerNorm(dim_H)
        self.ln_add_left = nn.LayerNorm(dim_H)
        self.ln_add_right = nn.LayerNorm(dim_H)

    def forward(self, id, mask=None):
        batch, seq_len, _ = id.size()
        residual = id
        id_ln = self.ln_start(id)

        Q, K, V = self.fc_q(id_ln), self.fc_k(id_ln), self.fc_v(id_ln)

        split_H = self.dim_H // self.num_heads
        Q_ = torch.cat(Q.split(split_H, 2), 0)
        K_ = torch.cat(K.split(split_H, 2), 0)
        V_ = torch.cat(V.split(split_H, 2), 0)

        raw_A_ = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(split_H)

        if mask is not None:
            raw_A_ = raw_A_.masked_fill(mask == 0, -1e8)
        A = self.attn_dropout(torch.softmax(raw_A_, -1))

        O_ = A.bmm(V_)
        O_ = torch.cat(O_.split(batch, 0), 2)

        id_ln = self.ln_add_left(id_ln + O_)
        residual = self.ln_add_right(residual + O_)

        O__ = id_ln + residual

        return O__, A


class Basic_Block(nn.Module):
    def __init__(self, dim_H, num_heads=8, dropout_ratio=0.0, attn_dropout_ratio=0.0, pre_ln=False, last_layer_flag=True):
        super(Basic_Block, self).__init__()
        self.mha = MHA(dim_H, num_heads, dropout_ratio, attn_dropout_ratio, pre_ln)
        self.ff = FF(dim_H, dropout_ratio, pre_ln)

        if pre_ln:
            self.ln_mha = nn.Identity()
            self.ln_ff = nn.Identity()
        else:
            self.ln_mha = nn.LayerNorm(dim_H)
            self.ln_ff = nn.LayerNorm(dim_H)

    def forward(self, x, mask=None):
        out, _ = self.mha(x, mask)
        out = self.ln_mha(out)

        out = self.ff(out)
        out = self.ln_ff(out)

        return out


class MHA(nn.Module):
    def __init__(self, dim_H, num_heads=8, dropout_ratio=0.0, attn_dropout_ratio=0.0, pre_ln=False):
        super(MHA, self).__init__()
        self.dim_H = dim_H
        self.num_heads = num_heads

        self.fc_q = nn.Linear(dim_H, dim_H, bias=False)
        self.fc_k = nn.Linear(dim_H, dim_H, bias=False)
        self.fc_v = nn.Linear(dim_H, dim_H, bias=False)

        self.fc_o = nn.Linear(dim_H, dim_H, bias=False)
        self.dropout = nn.Dropout(dropout_ratio)
        self.attn_dropout = nn.Dropout(attn_dropout_ratio)

        if pre_ln:
            self.ln = nn.LayerNorm(dim_H)
        else:
            self.ln = nn.Identity()

    def forward(self, id, mask=None):
        batch, seq_len, _ = id.size()
        residual = id
        id = self.ln(id)

        Q, K, V = self.fc_q(id), self.fc_k(id), self.fc_v(id)

        split_H = self.dim_H // self.num_heads
        Q_ = torch.cat(Q.split(split_H, 2), 0)
        K_ = torch.cat(K.split(split_H, 2), 0)
        V_ = torch.cat(V.split(split_H, 2), 0)

        raw_A_ = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(split_H)

        if mask is not None:
            raw_A_ = raw_A_.masked_fill(mask == 0, -1e8)
        A = self.attn_dropout(torch.softmax(raw_A_, -1))

        O_ = A.bmm(V_)

        O_ = torch.cat(O_.split(batch, 0), 2)
        O__ = residual + self.dropout(self.fc_o(O_))

        # return O__
        return O__, A


class FF(nn.Module):
    def __init__(self, dim_H, dropout_ratio=0.0, pre_ln=False):
        super(FF, self).__init__()
        self.fc1 = nn.Linear(dim_H, dim_H)
        self.fc2 = nn.Linear(dim_H, dim_H)

        self.dropout = nn.Dropout(dropout_ratio)

        if pre_ln:
            self.ln = nn.LayerNorm(dim_H, eps=1e-6)
        else:
            self.ln = nn.Identity()

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        residual = x
        x = self.ln(x)

        out = self.fc1(x)
        # out = F.relu(out)
        out = self.gelu(out)
        out = self.fc2(out)
        out = self.dropout(out)

        out += residual

        return out


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        out = torch.nn.functional.silu(gate) * x
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=512, device=torch.device("cpu")):
        super().__init__()
        encoding = self._get_sinusoidal_embedding(hidden_dim, max_len)

        self.encoding = encoding.unsqueeze(0).to(device)
        self.encoding.requires_grad = False

    def _get_sinusoidal_embedding(self, hidden_dim, max_len):
        encoding = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        sinusoidal = torch.exp(
            torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim)
        )
        encoding[:, 0::2] = torch.sin(position * sinusoidal)
        encoding[:, 1::2] = torch.cos(position * sinusoidal)

        return encoding

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
