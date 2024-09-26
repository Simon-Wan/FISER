import torch
import torch.nn as nn
import math


def get_embeddings(src_vocab, tgt_vocab, mid_vocabs, d_model, num_attr, device='cuda'):
    """
    Get embedding layers
    :param src_vocab: source vocabulary
    :param tgt_vocab: target vocabulary
    :param mid_vocabs: middle output vocabulary
    :param d_model: dimension of model
    :param num_attr: number of attributes
    :param device: cuda device
    :return: physical embedding, social embedding, query embedding, action embedding, effect embedding,
             Q embedding, S embedding, V embedding, O embedding
    """

    src_embedding = TextEmbedding(d_model, src_vocab).to(device)
    tgt_embedding = TextEmbedding(d_model, tgt_vocab).to(device)
    mid_q_embedding = TextEmbedding(d_model, mid_vocabs['q']).to(device)
    mid_s_embedding = TextEmbedding(d_model, mid_vocabs['s']).to(device)
    mid_v_embedding = TextEmbedding(d_model, mid_vocabs['v']).to(device)
    mid_o_embedding = TextEmbedding(d_model, mid_vocabs['o']).to(device)
    return src_embedding, src_embedding, src_embedding, tgt_embedding, src_embedding, \
               mid_q_embedding, mid_s_embedding, mid_v_embedding, mid_o_embedding


class TextEmbedding(nn.Module):
    def __init__(self, d_model, text_vocab):
        super(TextEmbedding, self).__init__()
        self.d_model = d_model
        self.text_emb = nn.Embedding(len(text_vocab)+1, self.d_model, padding_idx=0)

    def forward(self, tokens):
        """
        L includes the interaction [INT] token (at position 0)
        :param tokens: Text input (B x L)
        :return: text_embed (B x L x D)
        """
        text_embed = self.text_emb(tokens)
        factor = math.sqrt(self.d_model)
        return text_embed * factor
