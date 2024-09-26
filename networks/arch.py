import copy
import time
import torch
import torch.nn as nn
from torch.nn.functional import softmax

from networks.layers import clones, LayerNorm
from networks.layers import MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, Embeddings, MLP
from networks.transformer_layers import TransformerLayer
from networks.embeddings import get_embeddings

# edited from AnnotatedTransformer.ipynb


class FuseNode(nn.Module):
    def __init__(self, d_model, num_attr):
        super(FuseNode, self).__init__()
        self.d_model = d_model
        self.num_attr = num_attr
        self.fuse_layer = nn.Linear(2 * d_model * num_attr, d_model)

    def forward(self, x, init_x):       # B x (N x A) x D
        x = x.reshape(x.shape[0], -1, self.d_model * self.num_attr)     # B x N x (A x D)
        init_x = init_x.reshape(x.shape[0], -1, self.d_model * self.num_attr)
        return self.fuse_layer(torch.cat((x, init_x), dim=-1))          # B x N x D


class MidGenerator(nn.Module):

    def __init__(self, d_model, mid_vocabs, mid_q_emb, mid_v_emb):
        super(MidGenerator, self).__init__()
        self.proj_q = nn.Linear(d_model * 4, len(mid_vocabs['q']))
        self.proj_v = nn.Linear(d_model * 4, len(mid_vocabs['v']))

        self.proj_s = nn.Linear(d_model * 6, len(mid_vocabs['s']))
        self.proj_o = nn.Linear(d_model * 6, len(mid_vocabs['o']))

        self.mid_q_emb = mid_q_emb
        self.mid_v_emb = mid_v_emb

    def forward(self, x):
        phy_x, soc_x, qry_x, eff_x = x
        interactions = torch.cat((phy_x[:, 0, :], soc_x[:, 0, :], qry_x[:, 0, :], eff_x[:, 0, :]), dim=-1)

        q_pred = self.proj_q(interactions)
        q_argmax = self.mid_q_emb(q_pred.argmax(dim=-1))    # B x D

        v_pred = self.proj_v(interactions)
        v_argmax = self.mid_v_emb(v_pred.argmax(dim=-1))    # B x D

        int_qv = torch.cat((phy_x[:, 0, :], soc_x[:, 0, :], qry_x[:, 0, :], eff_x[:, 0, :], q_argmax, v_argmax), dim=-1)

        s_pred = self.proj_s(int_qv)   # B x S
        o_pred = self.proj_o(int_qv)   # B x O

        return q_pred, s_pred, v_pred, o_pred


class Generator(nn.Module):

    def __init__(self, d_model, action_types, tgt_embed):
        super(Generator, self).__init__()
        self.proj_type = nn.Linear(d_model * 4, action_types)
        self.proj_arg1 = nn.Linear(d_model * 2, 1)
        self.proj_arg2 = nn.Linear(d_model * 2, 1)
        self.tgt_embed = tgt_embed

    def forward(self, x):
        phy_x, soc_x, qry_x, eff_x = x
        nodes = phy_x   # B x N x D

        interactions = torch.cat((phy_x[:, 0, :], soc_x[:, 0, :], qry_x[:, 0, :], eff_x[:, 0, :]), dim=-1)
        action = self.proj_type(interactions)   # B x T

        type_argmax = self.tgt_embed(action.argmax(dim=-1))     # B x D
        nodes_given_type = torch.concat((nodes, type_argmax.unsqueeze(1).repeat(1, nodes.shape[1], 1)), dim=-1)
        arg1 = self.proj_arg1(nodes_given_type).squeeze(-1)  # B x N
        arg2 = self.proj_arg2(nodes_given_type).squeeze(-1)  # B x N
        return action, arg1, arg2       # B x T, B x N, B x N


class ObjGenerator(nn.Module):

    def __init__(self, d_model):
        super(ObjGenerator, self).__init__()
        self.proj_obj = nn.Linear(d_model, 1)

    def forward(self, x, x_mask):
        nodes = x[0]        # B x N x D
        mask = x_mask       # B x N
        pred_obj = self.proj_obj(nodes).squeeze(-1)     # B x N
        return pred_obj * mask - (1 - mask) * 100


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, phy_x, soc_x, qry_x, eff_x, x_masks=None, use_LN=True):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            phy_x, soc_x, qry_x, eff_x = layer(phy_x, soc_x, qry_x, eff_x, x_masks)
        if use_LN:
            return self.norm(phy_x), self.norm(soc_x), self.norm(qry_x), self.norm(eff_x)
        else:
            return phy_x, soc_x, qry_x, eff_x


class EncoderLayer(nn.Module):
    """
    Encoder is made up of three channels.
    For social and query channel, each layer is made up of self-attn and feed forward.
    For physical channel, we apply Edge Transformer.
    """

    def __init__(self, d_model, d_ff, num_heads, self_attn, feed_forward, modality_interaction, dropout, device='cpu'):
        super(EncoderLayer, self).__init__()
        self.phy_layer = TransformerLayer(d_model, self_attn, feed_forward, dropout)
        self.soc_layer = TransformerLayer(d_model, self_attn, feed_forward, dropout)
        self.qry_layer = TransformerLayer(d_model, self_attn, feed_forward, dropout)
        self.eff_layer = TransformerLayer(d_model, self_attn, feed_forward, dropout)
        self.modality_interaction = modality_interaction
        self.d_model = d_model

    def forward(self, phy_x, soc_x, qry_x, eff_x, x_masks=None):
        """Separately calculate each channel and perform modality interaction."""
        phy_x = self.phy_layer(phy_x, x_masks[0])
        soc_x = self.soc_layer(soc_x, x_masks[1])
        qry_x = self.qry_layer(qry_x, x_masks[2])
        eff_x = self.eff_layer(eff_x, x_masks[3])
        phy_x, soc_x, qry_x, eff_x = self.modality_interaction(phy_x, soc_x, qry_x, eff_x)
        return phy_x, soc_x, qry_x, eff_x


class ModalityInteraction(nn.Module):
    """
    Exchange information across four channels.
    """

    def __init__(self, size, d_mint, dropout, use_effect=True):
        super(ModalityInteraction, self).__init__()
        self.size = size
        self.use_effect = use_effect
        if use_effect:
            self.mlp = MLP(size * 4, d_mint, size * 4, 1, dropout)
        else:
            self.mlp = MLP(size * 3, d_mint, size * 3, 1, dropout)

    def forward(self, phy_x, soc_x, qry_x, eff_x):
        phy_int = phy_x[:, 0, :]
        soc_int = soc_x[:, 0, :]
        qry_int = qry_x[:, 0, :]
        eff_int = eff_x[:, 0, :]
        if self.use_effect:
            ints = torch.cat((phy_int, soc_int, qry_int, eff_int), dim=-1)
        else:
            ints = torch.cat((phy_int, soc_int, qry_int), dim=-1)
        ints = self.mlp(ints)
        if self.use_effect:
            phy_int, soc_int, qry_int, eff_int = torch.split(ints, [self.size] * 4, dim=-1)
        else:
            phy_int, soc_int, qry_int = torch.split(ints, [self.size] * 3, dim=-1)
        phy_x[:, 0] = phy_int
        soc_x[:, 0] = soc_int
        qry_x[:, 0] = qry_int
        eff_x[:, 0] = eff_int

        return phy_x, soc_x, qry_x, eff_x
