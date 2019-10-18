# coding=utf-8

""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa).

step1: download WikiText 2 Dataset https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
put into data/wikitext-2

step2: start training

python transformer_albert_7.py \
--data_file './data/wikitext-2/wiki.train.tokens' \
--vocab './data/vocab_albert.txt' \
--train_cfg './config/pretrain.json' \
--model_cfg './config/albert_unittest.json' \
--max_pred 75 --mask_prob 0.15 \
--mask_alpha 4 --mask_beta 1 --max_gram 3 \
--save_dir './output/albert' \
--log_dir './runs/albert'


step3: check tensorboard
tensorboard --logdir=runs

"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from utils.tokenization import FullTokenizer
from utils.attention import MultiHeadedAttention, PositionwiseFeedForward
from utils.data_iters import Preprocess4Pretrain, SentPairDataLoader


class Config():
    """Configuration for BERT model"""
    vocab_size = None  # Size of Vocabulary
    hidden = 768  # Dimension of Hidden Layer in Transformer Encoder
    hidden_ff = 768 * 4  # Dimension of Intermediate Layers in Positionwise Feedforward Net
    embedding = 128  # Factorized embedding parameterization

    n_layers = 12  # Numher of Hidden Layers
    n_heads = 768 // 64  # Numher of Heads in Multi-Headed Attention Layers
    # activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    max_len = 512  # Maximum Length for Positional Embeddings
    n_segments = 2  # Number of Sentence Segments

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


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


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):

    cfg = train.Config.from_json(args.train_cfg)
    model_cfg = models.Config.from_json(args.model_cfg)

    set_seeds(cfg.seed)

    tokenizer = FullTokenizer(vocab_file=args.vocab, do_lower_case=True)
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

    pipeline = [Preprocess4Pretrain(args.max_pred,
                                    args.mask_prob,
                                    list(tokenizer.vocab.keys()),
                                    tokenizer.convert_tokens_to_ids,
                                    model_cfg.max_len,
                                    args.mask_alpha,
                                    args.mask_beta,
                                    args.max_gram)]
    data_iter = SentPairDataLoader(args.data_file,
                                   cfg.batch_size,
                                   tokenize,
                                   model_cfg.max_len,
                                   pipeline=pipeline)

    model = ALBERT(model_cfg)
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss()

    optimizer = optim.optim4GPU(cfg, model)
    trainer = train.Trainer(cfg, model, data_iter, optimizer, args.save_dir, get_device())

    writer = SummaryWriter(log_dir=args.log_dir) # for tensorboardX

    def get_loss(model, batch, global_step): # make sure loss is tensor
        input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch

        logits_lm, logits_clsf = model(input_ids, segment_ids, input_mask, masked_pos)
        loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids) # for masked LM
        loss_lm = (loss_lm*masked_weights.float()).mean()
        loss_sop = criterion2(logits_clsf, is_next) # for sentence classification
        writer.add_scalars('data/scalar_group',
                           {'loss_lm': loss_lm.item(),
                            'loss_sop': loss_sop.item(),
                            'loss_total': (loss_lm + loss_sop).item(),
                            'lr': optimizer.get_lr()[0],
                           },
                           global_step)
        return loss_lm + loss_sop

    trainer.train(get_loss, model_file=None, data_parallel=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ALBERT Language Model')
    parser.add_argument('--data_file', type=str, default='./data/wiki.train.tokens')
    parser.add_argument('--vocab', type=str, default='./data/vocab.txt')
    parser.add_argument('--train_cfg', type=str, default='./config/pretrain.json')
    parser.add_argument('--model_cfg', type=str, default='./config/albert_unittest.json')

    # official google-reacher/bert is use 20, but 20/512(=seq_len)*100 make only 3% Mask
    # So, using 76(=0.15*512) as `max_pred`
    parser.add_argument('--max_pred', type=int, default=76, help='max tokens of prediction')
    parser.add_argument('--mask_prob', type=float, default=0.15, help='masking probability')

    # try to n-gram masking SpanBERT(Joshi et al., 2019)
    parser.add_argument('--mask_alpha', type=int,
                        default=4, help="How many tokens to form a group.")
    parser.add_argument('--mask_beta', type=int,
                        default=1, help="How many tokens to mask within each group.")
    parser.add_argument('--max_gram', type=int,
                        default=3, help="number of max n-gram to masking")

    parser.add_argument('--save_dir', type=str, default='./saved')
    parser.add_argument('--log_dir', type=str, default='./log')

    args = parser.parse_args()
    main(args)
