from networks.arch import *


def make_e2e_model(
        mid, src_vocab, tgt_vocab, mid_vocabs, device,
        num_attr=16, d_model=64, h=8, d_ff=1024, dropout=0.1,
        N_qsvo=3, N_obj=3, N_action=3
):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, device=device)
    mint = ModalityInteraction(d_model, d_ff, dropout, use_effect=True)
    fuse_node = FuseNode(d_model, num_attr=num_attr)
    if mid == 'none':
        encoder_qsvo = None
        encoder_obj = None
        encoder_action = Encoder(EncoderLayer(d_model, d_ff, h, c(attn), c(ff), mint, dropout, device=device),
                                 N_qsvo + N_obj + N_action)
    elif mid == 'qsvo':
        encoder_qsvo = Encoder(EncoderLayer(d_model, d_ff, h, c(attn), c(ff), mint, dropout, device=device), N_qsvo)
        encoder_obj = None
        encoder_action = Encoder(EncoderLayer(d_model, d_ff, h, c(attn), c(ff), mint, dropout, device=device),
                                 N_obj + N_action)
    elif mid == 'obj':
        encoder_qsvo = None
        encoder_obj = Encoder(EncoderLayer(d_model, d_ff, h, c(attn), c(ff), mint, dropout, device=device),
                              N_qsvo + N_obj)
        encoder_action = Encoder(EncoderLayer(d_model, d_ff, h, c(attn), c(ff), mint, dropout, device=device), N_action)
    else:
        encoder_qsvo = Encoder(EncoderLayer(d_model, d_ff, h, c(attn), c(ff), mint, dropout, device=device), N_qsvo)
        encoder_obj = Encoder(EncoderLayer(d_model, d_ff, h, c(attn), c(ff), mint, dropout, device=device), N_obj)
        encoder_action = Encoder(EncoderLayer(d_model, d_ff, h, c(attn), c(ff), mint, dropout, device=device), N_action)

    physical_emb, social_emb, query_emb, action_emb, effect_emb, mid_q_emb, mid_s_emb, mid_v_emb, mid_o_emb \
        = get_embeddings(src_vocab, tgt_vocab, mid_vocabs, d_model, num_attr, device)

    social_emb = nn.Sequential(social_emb, c(position))
    query_emb = nn.Sequential(query_emb, c(position))
    effect_emb = nn.Sequential(effect_emb, c(position))

    mid_generator = MidGenerator(d_model, mid_vocabs, mid_q_emb, mid_v_emb)
    obj_generator = ObjGenerator(d_model)
    generator = Generator(d_model, len(tgt_vocab) + 1, action_emb)

    model = E2EModel(
        mid,
        encoder_qsvo,
        encoder_obj,
        encoder_action,
        fuse_node,
        {'physical': physical_emb, 'social': social_emb, 'query': query_emb, 'effect': effect_emb},
        {'action': action_emb},
        {'q': mid_q_emb, 's': mid_s_emb, 'v': mid_v_emb, 'o': mid_o_emb},
        mid_generator,
        obj_generator,
        generator,
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class E2EModel(nn.Module):
    """
    Edited from a standard Encoder-Decoder architecture.
    """

    def __init__(self, mid, encoder_qsvo, encoder_obj, encoder_action, fuse_node,
                 src_embed, tgt_embed, mid_embed, mid_generator, obj_generator, generator):
        super(E2EModel, self).__init__()
        self.mid = mid
        self.encoder_qsvo = encoder_qsvo
        self.encoder_obj = encoder_obj
        self.encoder_action = encoder_action
        self.fuse_node = fuse_node
        self.src_embed = src_embed  # dict of physical/social/query embedding layer
        self.phy_embed = src_embed['physical']
        self.soc_embed = src_embed['social']
        self.qry_embed = src_embed['query']
        self.eff_embed = src_embed['effect']
        self.tgt_embed = tgt_embed['action']  # dict of action type embedding layer
        self.mid_q_embed = mid_embed['q']
        self.mid_s_embed = mid_embed['s']
        self.mid_v_embed = mid_embed['v']
        self.mid_o_embed = mid_embed['o']
        self.mid_generator = mid_generator  # generating simplified subgoal
        self.obj_generator = obj_generator  # generating target object
        self.generator = generator  # generating action type and argument(s)

    def forward(self, src, src_mask=None, qsvo_supervision=None, obj_supervision=None,
                replace_indices=None, object_indices=None):
        phy_input, soc_input, qry_input, eff_input, init_phy_input = src

        phy_x = self.fuse_node(self.phy_embed(phy_input), self.phy_embed(init_phy_input))
        soc_x = self.soc_embed(soc_input)
        if replace_indices is not None and object_indices is not None:
            for batch_idx in range(phy_x.shape[0]):
                source = phy_x[batch_idx][object_indices[batch_idx]]
                index = replace_indices[batch_idx].unsqueeze(-1).expand((-1, phy_x.shape[2]))
                soc_x[batch_idx].scatter_(0, index, source)
        qry_x = self.qry_embed(qry_input)
        eff_x = self.eff_embed(eff_input)

        if self.mid == 'none':
            hidden = self.encoder_action(phy_x, soc_x, qry_x, eff_x, src_mask)
            return hidden, None, None
        if self.mid == 'qsvo' or self.mid == 'qsvo+obj':
            qsvo_hidden = self.encoder_qsvo(phy_x, soc_x, qry_x, eff_x, src_mask)
            qsvo_output = self.mid_generator(qsvo_hidden)
            mid_q, mid_s, mid_v, mid_o = qsvo_output
            if self.training:
                qsvo_input = qsvo_supervision
            else:
                qsvo_input = (
                    mid_q.argmax(dim=-1),  # B x 1
                    mid_s.argmax(dim=-1),  # B x 1
                    mid_v.argmax(dim=-1),  # B x 1
                    mid_o.argmax(dim=-1),  # B x 1
                )
            qsvo_x = (
                self.mid_q_embed(qsvo_input[0]),
                self.mid_s_embed(qsvo_input[1]),
                self.mid_v_embed(qsvo_input[2]),
                self.mid_o_embed(qsvo_input[3]),
            )
            qsvo_x = torch.cat((qsvo_hidden[1][:, 0:1], qsvo_x[0].unsqueeze(dim=1),qsvo_x[1].unsqueeze(dim=1),
                               qsvo_x[2].unsqueeze(dim=1), qsvo_x[3].unsqueeze(dim=1)), dim=1)
            qsvo_mask = (src_mask[0], torch.ones([qsvo_x.shape[0], qsvo_x.shape[1]]).to(qsvo_x.device),
                         src_mask[2], src_mask[3])
        if self.mid == 'qsvo':
            hidden = self.encoder_action(qsvo_hidden[0], qsvo_x, qsvo_hidden[2], qsvo_hidden[3], qsvo_mask)
            return hidden, qsvo_output, None
        if self.mid == 'qsvo+obj':
            obj_hidden = self.encoder_obj(qsvo_hidden[0], qsvo_x, qsvo_hidden[2], qsvo_hidden[3], qsvo_mask)
        if self.mid == 'obj':
            obj_hidden = self.encoder_obj(phy_x, soc_x, qry_x, eff_x, src_mask)
        if self.mid == 'obj' or self.mid == 'qsvo+obj':
            obj_output = self.obj_generator(obj_hidden, src_mask[0])
            if self.training:
                obj_input = obj_supervision
            else:
                obj_input = obj_output.argmax(dim=-1)
            target_obj_token = [phy_x[idx][obj_id] for idx, obj_id in enumerate(obj_input)]
            target_obj_token = torch.stack(target_obj_token, dim=0)

            obj_x = torch.cat((obj_hidden[2][:, 0:1], target_obj_token.unsqueeze(dim=1)), dim=1)
        if self.mid == 'obj':
            obj_mask = (src_mask[0], src_mask[1],
                        torch.ones([obj_x.shape[0], obj_x.shape[1]]).to(obj_x.device), src_mask[3])
            hidden = self.encoder_action(obj_hidden[0], obj_hidden[1], obj_x, obj_hidden[3], obj_mask)
            return hidden, None, obj_output
        if self.mid == 'qsvo+obj':
            qsvo_obj_mask = (src_mask[0], torch.ones([qsvo_x.shape[0], qsvo_x.shape[1]]).to(qsvo_x.device),
                             torch.ones([obj_x.shape[0], obj_x.shape[1]]).to(obj_x.device), src_mask[3])
            hidden = self.encoder_action(obj_hidden[0], obj_hidden[1], obj_x, obj_hidden[3], qsvo_obj_mask)
            return hidden, qsvo_output, obj_output
