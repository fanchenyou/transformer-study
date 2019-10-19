# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.attention import MultiHeadedAttention, PositionwiseFeedForward

class ALBERTEmbeddings(nn.Module):
    """The embedding module from word, position and token_type embeddings.
        Compare this with utils.embedding.BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, hidden_size, max_len, n_segments):
        super(ALBERTEmbeddings).__init__()
        # Original BERT Embedding
        # self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.hidden) # token embedding

        # factorized embedding
        self.tok_embed1 = nn.Embedding(vocab_size, embed_size)
        self.tok_embed2 = nn.Linear(embed_size, hidden_size)

        self.pos_embed = nn.Embedding(max_len, hidden_size)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, hidden_size)  # segment(token type) embedding

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)  # (S,) -> (B, S)

        # factorized embedding
        e = self.tok_embed1(x)
        e = self.tok_embed2(e)
        e = e + self.pos_embed(pos) + self.seg_embed(seg)
        # return self.drop(self.norm(e))
        # return self.norm(e)
        return e


class Transformer_ALBERT(nn.Module):
    """ Transformer with Self-Attentive Blocks"""

    def __init__(self, cfg):
        super(Transformer_ALBERT).__init__()
        self.embed = ALBERTEmbeddings(cfg.vocab_size, cfg.embedding,
                                      cfg.hidden, cfg.max_len, cfg.n_segments)
        self.norm0 = nn.LayerNorm(cfg.hidden)

        # To use parameter-sharing strategies
        self.n_layers = cfg.n_layers
        self.attn = MultiHeadedAttention(1, cfg.hidden, dropout=0.0)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = nn.LayerNorm(cfg.hidden)

        # not use dropout
        self.pwff = PositionwiseFeedForward(cfg.hidden, cfg.hidden_ff, dropout=0.0)
        self.norm2 = nn.LayerNorm(cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg, mask):
        h = self.norm0(self.embed(x, seg))

        for _ in range(self.n_layers):
            # share all attn params across layers
            h = self.attn(h, mask)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))

        return h


class ALBERT(nn.Module):
    """ALBERT Model for Pretrain :
    Masked LM and sentence-order prediction(SOP)"""

    def __init__(self, cfg):
        super(ALBERT).__init__()
        self.transformer = Transformer_ALBERT(cfg)
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ2 = F.gelu
        self.norm = nn.LayerNorm(cfg.hidden)
        self.classifier = nn.Linear(cfg.hidden, 2) # whether in order or not

        # decoder is shared with embedding layer
        ## project hidden layer to embedding layer
        embed_weight2 = self.transformer.embed.tok_embed2.weight
        n_hidden, n_embedding = embed_weight2.size()
        self.decoder1 = nn.Linear(n_hidden, n_embedding, bias=False)
        self.decoder1.weight.data = embed_weight2.data.t()

        ## project embedding layer to vocabulary layer
        embed_weight1 = self.transformer.embed.tok_embed1.weight
        n_vocab, n_embedding = embed_weight1.size()
        self.decoder2 = nn.Linear(n_embedding, n_vocab, bias=False)
        self.decoder2.weight = embed_weight1
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc(h[:, 0]))
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))

        logits_lm = self.decoder2(self.decoder1(h_masked)) + self.decoder_bias
        logits_clsf = self.classifier(pooled_h)

        return logits_lm, logits_clsf

