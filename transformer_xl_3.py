"""
src:https://github.com/kimiyoung/transformer-xl/tree/master/pytorch

The task is two-fold, see paper section 3.1
1) to predict the second part of a sentence (Next Sentence Prediction)
2) to predict the masked words of a sentence (Masked LM)

step 0 (optional): Let's prepare the dataset first.
cd data/
bash getdata.sh

python ./utils/vocab.py -c data/qa_pair.txt -o data/vocab.small

step 2: train the network
python transformer_bert_from_scratch_5.py \
-c data/qa_pair.txt -v data/vocab.small -o output/bert.model

"""
import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer


sys.path.insert(0, './')
from utils import data_utils


class XLNet(nn.Module):
    """
        Defines a Transformer-XL computation graph with additional
        support for XLNet.

        Doc url:
        Args:

        inp_k: input. int32 Tensor in shape [len, bsz], the input token IDs.
        seg_id: int32 Tensor in shape [len, bsz], the input segment IDs.
        input_mask: float32 Tensor in shape [len, bsz], the input mask.
          0 for real tokens and 1 for padding.
        mems: a list of float32 Tensors in shape [mem_len, bsz, d_model], memory
          from previous batches. The length of the list equals n_layer.
          If None, no memory is used.
        perm_mask: float32 Tensor in shape [len, len, bsz].
          If perm_mask[i, j, k] = 0, i attend to j in batch k;
          if perm_mask[i, j, k] = 1, i does not attend to j in batch k.
          If None, each position attends to all the others.
        target_mapping: float32 Tensor in shape [num_predict, len, bsz].
          If target_mapping[i, j, k] = 1, the i-th predict in batch k is
          on the j-th token.
          Only used during pretraining for partial prediction.
          Set to None during finetuning.
        inp_q: target_mask, float32 Tensor in shape [len, bsz].
          1 for tokens with losses and 0 for tokens without losses.
          Only used during pretraining for two-stream attention.
          Set to None during finetuning.

        n_layer: int, the number of layers.
        d_model: int, the hidden size.
        n_head: int, the number of attention heads.
        d_head: int, the dimension size of each attention head.
        d_inner: int, the hidden size in feed-forward layers.
        ff_activation: str, "relu" or "gelu".
        n_token: int, the vocab size.

        dropout: float, dropout rate.
        dropatt: float, dropout rate on attention probabilities.

        mem_len: int, the number of tokens to cache.
        reuse_len: int, the number of tokens in the currect batch to be cached
          and reused in the future.
        bi_data: bool, whether to use bidirectional input pipeline.
          Usually set to True during pretraining and False during finetuning.
        clamp_len: int, clamp all relative distances larger than clamp_len.
          -1 means no clamping.

      """

    def __init__(self, n_token, n_layer, n_head, d_head, d_inner, d_model, dropout, dropatt,
                 attn_type, bi_data, clamp_len, same_length, reuse_len, mem_len):
        super(XLNet, self).__init__()

        self.n_token = n_token
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.d_model = d_model
        self.dropout = dropout
        self.dropatt = dropatt
        self.bi_data = bi_data
        self.clamp_len = clamp_len
        self.same_length = same_length
        self.reuse_len = reuse_len
        self.mem_len = mem_len
        self.attn_type = attn_type

        self.embedding = nn.Embedding(n_token, d_model)
        self.Dropout = nn.Dropout(p=dropout)
        self.DropAttn = nn.Dropout(p=dropatt)

        self.r_w_bias = nn.Parameter(torch.randn(self.n_layer, self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.randn(self.n_layer, self.n_head, self.d_head))

        ##### Segment embedding
        self.r_s_bias = nn.Parameter(torch.randn(self.n_layer, self.n_head, self.d_head))

        self.seg_embed = nn.Parameter(torch.randn(self.n_layer, 2, self.n_head, self.d_head))

        self.mask_emb = nn.Parameter(torch.randn(1, 1, d_model))

        # post-attention projection (back to `d_model`)
        self.proj_o = nn.Parameter(torch.randn(self.d_model, self.n_head, self.d_head))

        #### Project hidden states to a specific head with a 4D-shape.
        self.q_proj_weight = nn.Parameter(torch.randn(self.d_model,
                                                      self.n_head, self.d_head))
        self.k_proj_weight = nn.Parameter(torch.randn(self.d_model,
                                                      self.n_head, self.d_head))
        self.v_proj_weight = nn.Parameter(torch.randn(self.d_model,
                                                      self.n_head, self.d_head))
        self.r_proj_weight = nn.Parameter(torch.randn(self.d_model,
                                                      self.n_head, self.d_head))

        self.layer_norm = nn.LayerNorm(d_model)

        self.conv1 = nn.Linear(d_model, d_inner)
        self.conv2 = nn.Linear(d_inner, d_model)
        self.relu = nn.ReLU(inplace=True)

        self.softmax_b = nn.Parameter(torch.zeros(self.n_token))


    def rel_shift(self, x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = x.shape

        x = torch.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
        #x = x[1:, 0:, 0:, 0:]  # tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        x = x[1:, ...]
        x = torch.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
        #x = x[0:, 0:klen, 0:, 0:]  # tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])
        x = torch.index_select(x, 1, torch.arange(klen, device=x.device, dtype=torch.long))

        return x

    def positionwise_ffn(self, inp, activation_type='gelu'):

        """Position-wise Feed-forward Network."""
        output = self.conv1(inp)
        output = self.Dropout(output)
        if activation_type == 'relu':
            output = self.relu(output)
        elif activation_type == 'gelu':
            #output = self.gelu(output)
            output = F.gelu(output)
        else:
            raise ValueError('Unsupported activation type {}'.format(activation_type))

        output = self.layer_norm(output + inp)
        return output

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""

        # post-attention projection (back to `d_model`)
        attn_out = torch.einsum('ibnd,hnd->ibh', attn_vec, self.proj_o)

        attn_out = self.Dropout(attn_out)
        if residual:
            output = self.layer_norm(attn_out + h)
        else:
            output = self.layer_norm(attn_out)

        return output

    def head_projection(self, h, name):
        """Project hidden states to a specific head with a 4D-shape."""

        if name == 'q':
            proj_weight = self.q_proj_weight
        elif name == 'k':
            proj_weight = self.k_proj_weight
        elif name == 'v':
            proj_weight = self.v_proj_weight
        elif name == 'r':
            proj_weight = self.r_proj_weight
        else:
            raise ValueError('Unknown `name` {}.'.format(name))

        head = torch.einsum('ibh,hnd->ibnd', h, proj_weight)

        return head

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                      r_w_bias, r_r_bias, r_s_bias, attn_mask, scale):

        """Core relative positional attention operations."""
        # https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/modeling_xlnet.py#L242
        # more details https://mlexplained.com/2019/07/04/building-the-transformer-xl-from-scratch/

        # content based attention score
        ac = torch.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h)

        # position based attention score
        bd = torch.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r)
        bd = self.rel_shift(bd, klen=ac.shape[1])

        # segment based attention score
        if seg_mat is None:
            ef = 0
        else:
            ef = torch.einsum('ibnd,snd->ibns', q_head + r_s_bias, seg_embed)
            ef = torch.einsum('ijbs,ibns->ijbn', seg_mat, ef)

        # merge attention scores and perform masking
        attn_score = (ac + bd + ef) * scale
        if attn_mask is not None:
            # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
            attn_score = attn_score - 1e30 * attn_mask

        # attention probability
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.DropAttn(attn_prob)

        # attention output
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)

        return attn_vec

    def rel_multihead_attn(self, h, r, r_w_bias, r_r_bias, seg_mat, r_s_bias, seg_embed,
                           attn_mask, mems, d_model, n_head, d_head, dropout, dropatt):
        """Multi-head attention with relative positional encoding."""

        scale = 1 / (d_head ** 0.5)
        if mems is not None and len(mems.size()) > 1:
            cat = torch.cat([mems, h], dim=0)
        else:
            cat = h

        # content heads
        q_head_h = self.head_projection(h, 'q')
        k_head_h = self.head_projection(cat, 'k')
        v_head_h = self.head_projection(cat, 'v')

        # positional heads
        k_head_r = self.head_projection(r, 'r')

        # core attention ops
        attn_vec = self.rel_attn_core(
            q_head_h, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, r_w_bias,
            r_r_bias, r_s_bias, attn_mask, scale)

        # post processing
        output = self.post_attention(h, attn_vec)

        return output

    def two_stream_rel_attn(self, h, g, r, mems, r_w_bias, r_r_bias, seg_mat, r_s_bias,
                            seg_embed, attn_mask_h, attn_mask_g, target_mapping):
        '''
        Call in Line 528, for each layer_i
        output_h, output_g = self.two_stream_rel_attn(
            h=output_h,
            g=output_g,
            r=pos_emb,
            r_w_bias=self.r_w_bias[i],
            r_r_bias=self.r_r_bias[i],
            seg_mat=seg_mat,
            r_s_bias=r_s_bias_i,
            seg_embed=seg_embed_i,
            attn_mask_h=non_tgt_mask,
            attn_mask_g=attn_mask,
            mems=mems[i],
            target_mapping=target_mapping)
        '''


        scale = 1 / (self.d_head ** 0.5)

        # content based attention score
        if mems is not None and len(mems.size()) > 1:
            cat = torch.cat([mems, h], dim=0)
        else:
            cat = h

        # content-based key head
        k_head_h = self.head_projection(cat, 'k')

        # content-based value head
        v_head_h = self.head_projection(cat, 'v')

        # position-based key head
        k_head_r = self.head_projection(r, 'r')

        ##### h-stream
        # content-stream query head
        q_head_h = self.head_projection(h, 'q')

        # core attention ops
        # h^(m)_zt = LayerNorm(h^(m-1)_zt + RelAttn(h^(m-1)_zt + [h~^(m-1), hT(m-1)_z<=t]))
        attn_vec_h = self.rel_attn_core(
            q_head_h, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, r_w_bias,
            r_r_bias, r_s_bias, attn_mask_h, scale)

        # post processing
        output_h = self.post_attention(h, attn_vec_h)

        ##### g-stream
        # query-stream query head
        q_head_g = self.head_projection(g, 'q')

        # core attention ops
        # g^(m)_zt = LayerNorm(g^(m-1)_zt + RelAttn(g^(m-1)_zt + [h~^(m-1), hT(m-1)_z<=t]))
        if target_mapping is not None:
            q_head_g = torch.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
            attn_vec_g = self.rel_attn_core(
                q_head_g, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, r_w_bias,
                r_r_bias, r_s_bias, attn_mask_g, scale)
            attn_vec_g = torch.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
        else:
            attn_vec_g = self.rel_attn_core(
                q_head_g, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, r_w_bias,
                r_r_bias, r_s_bias, attn_mask_g, scale)

        # post processing
        output_g = self.post_attention(g, attn_vec_g)

        return output_h, output_g

    def _create_mask(self, qlen, mlen, dtype, same_length=False):
        """create causal attention mask."""
        # [[0,1,1],
        #  [0,0,1],
        #  [0,0,0]]
        """
        https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/modeling_xlnet.py#L606
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.
        
                  same_length=False:      same_length=True:
                  <mlen > <  qlen >       <mlen > <  qlen >
               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]
        """

        attn_mask = torch.ones([qlen, qlen], dtype=dtype)
        mask_u = torch.triu(attn_mask)  # Upper triangular part.
        mask_dia = torch.tril(attn_mask) & torch.triu(attn_mask)  # Diagonal. Figure 2(c)

        attn_mask_pad = torch.zeros([qlen, mlen], dtype=dtype)
        ret = torch.cat([attn_mask_pad, mask_u - mask_dia], dim=1)  # [qlen, mlen]
        if same_length:
            # [[0,1,1],
            #  [1,0,1],
            #  [1,1,0]]
            mask_l = torch.tril(attn_mask)  # Lower triangular part.
            ret = torch.cat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], dim=1)

        return ret.type(dtype=torch.float32)  # [qlen, qlen]

    def positional_embedding(self, pos_seq, inv_freq):
        sinusoid_inp = torch.einsum('i,d->id', pos_seq, inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb[:, None, :]

        return pos_emb

    def _cache_mem(self, curr_out, prev_mem, mem_len, reuse_len=None):
        """cache hidden states into memory."""

        with torch.no_grad():
            if mem_len is None or mem_len == 0:
                return None
            else:
                if reuse_len is not None and reuse_len > 0:
                    curr_out = curr_out[:reuse_len]

                if prev_mem is None:
                    new_mem = curr_out[-mem_len:]
                else:
                    new_mem = torch.cat([prev_mem, curr_out], dim=0)[-mem_len:]

            return new_mem

    def relative_positional_encoding(self, qlen, klen, d_model, clamp_len, attn_type,
                                     bi_data, bsz=None, dtype=None):
        """create relative positional encoding."""

        freq_seq = torch.arange(0, d_model, 2.0)
        if dtype is not None and dtype != torch.float32:
            freq_seq = freq_seq.type(dtype)
        inv_freq = 1 / (10000 ** (freq_seq / d_model))

        assert attn_type == 'bi'  # always set to XLNet
        beg, end = klen, -qlen


        if bi_data:
            fwd_pos_seq = torch.arange(beg, end, -1.0)
            bwd_pos_seq = torch.arange(-beg, -end, 1.0)

            if dtype is not None and dtype != torch.float32:
                fwd_pos_seq = fwd_pos_seq.type(dtype=dtype)
                bwd_pos_seq = bwd_pos_seq.type(dtype=dtype)

            if clamp_len > 0:
                fwd_pos_seq = torch.clamp(fwd_pos_seq, -clamp_len, clamp_len)
                bwd_pos_seq = torch.clamp(bwd_pos_seq, -clamp_len, clamp_len)

            fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
            bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
        else:
            fwd_pos_seq = torch.arange(beg, end, -1.0)
            if dtype is not None and dtype != torch.float32:
                fwd_pos_seq = fwd_pos_seq.type(dtype=dtype)
            if clamp_len > 0:
                fwd_pos_seq = torch.clamp(fwd_pos_seq, -clamp_len, clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)

        return pos_emb

    def forward(self, inp_k, seg_id, input_mask, mems, perm_mask, target_mapping, inp_q):
        new_mems = []

        bsz = inp_k.shape[1]
        qlen = inp_k.shape[0]
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        ##### Attention mask
        # causal attention mask
        assert self.attn_type == 'bi'
        attn_mask = None

        # data mask: input mask & perm mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz],
                                    dtype=torch.float32)
            data_mask = torch.cat([mems_mask, data_mask], dim=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = attn_mask.gt(0).type(torch.float32)

        if attn_mask is not None:
            non_tgt_mask = -torch.eye(qlen, dtype=torch.float32)  # [qlen, qlen]
            non_tgt_mask = torch.cat([torch.zeros([qlen, mlen], dtype=torch.float32),  # [qlen, klen]
                                      non_tgt_mask],
                                     dim=-1)
            # attention mask is cancelled by non_tgt_mask (?)
            non_tgt_mask = (attn_mask +
                            non_tgt_mask[:, :, None, None]).gt(0).type(dtype=torch.float32)
        else:
            non_tgt_mask = None

        ##### Word embedding
        lookup_table = self.embedding
        word_emb_k = lookup_table(inp_k)

        if inp_q is not None:
            if target_mapping is not None:
                word_emb_q = self.mask_emb.repeat(target_mapping.shape[0], bsz, 1)
            else:
                inp_q_ext = inp_q[:, :, None]
                word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k

        #### Figure 2(a), Content Stream(Original Attention), h^(0)_t = e(x_i) = e(inp_k)
        output_h = self.Dropout(word_emb_k)
        if inp_q is not None:
            #### Query Stream, g^(0)_t = w
            #### the first layer query stream is initialized with a trainable vector
            output_g = self.Dropout(word_emb_q)

        ##### Segment embedding
        # paper
        # Given a pair of positions i and j in the sequence, if
        # i and j are from the same segment
        if seg_id is not None:
            # Convert `seg_id` to one-hot `seg_mat`
            mem_pad = torch.zeros([mlen, bsz], dtype=torch.int32)
            cat_ids = torch.cat([mem_pad, seg_id], dim=0)

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = (~torch.eq(seg_id[:, None], cat_ids[None, :])).type(torch.long)
            seg_mat = torch.eye(2, dtype=torch.float32)[seg_mat]
        else:
            seg_mat = None

        ##### Positional encoding
        pos_emb = self.relative_positional_encoding(
            qlen, klen, self.d_model, self.clamp_len, self.attn_type, self.bi_data,
            bsz=bsz, dtype=torch.float32)
        pos_emb = self.Dropout(pos_emb)

        ##### Attention layers
        if mems is None:
            mems = [None] * self.n_layer

        for i in range(self.n_layer):
            # cache new mems
            new_mems.append(self._cache_mem(output_h, mems[i], self.mem_len, self.reuse_len))

            # segment bias
            if seg_id is None:
                r_s_bias_i = None
                seg_embed_i = None
            else:
                r_s_bias_i = self.r_s_bias[i]
                seg_embed_i = self.seg_embed[i]

            if inp_q is not None:
                output_h, output_g = self.two_stream_rel_attn(
                    h=output_h,
                    g=output_g,
                    r=pos_emb,
                    r_w_bias=self.r_w_bias[i],
                    r_r_bias=self.r_r_bias[i],
                    seg_mat=seg_mat,
                    r_s_bias=r_s_bias_i,
                    seg_embed=seg_embed_i,
                    attn_mask_h=non_tgt_mask,
                    attn_mask_g=attn_mask,
                    mems=mems[i],
                    target_mapping=target_mapping)
            else:
                output_h = self.rel_multihead_attn(
                    h=output_h,
                    r=pos_emb,
                    r_w_bias=self.r_w_bias[i],
                    r_r_bias=self.r_r_bias[i],
                    seg_mat=seg_mat,
                    r_s_bias=r_s_bias_i,
                    seg_embed=seg_embed_i,
                    attn_mask=non_tgt_mask,
                    mems=mems[i])

            if inp_q is not None:
                output_g = self.positionwise_ffn(inp=output_g)

            output_h = self.positionwise_ffn(inp=output_h)

        if inp_q is not None:
            output = self.Dropout(output_g)
        else:
            output = self.Dropout(output_h)

        logits = torch.einsum('ibd,nd->ibn', output, lookup_table.weight) + self.softmax_b

        return logits, new_mems


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/shakespeare/hamlet.txt')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased',
                        help='Path to the sentence piece model from pytorch-pretrained-BERT')
    parser.add_argument('--seq_len', type=int, default=512, help="Sequence length.")
    parser.add_argument('--reuse_len', type=int, default=256,
                        help="Number of token that can be reused as memory. "
                             "Could be half of `seq_len`.")
    parser.add_argument('--perm_size', type=int,
                        default=256,
                        help="the length of longest permutation. Could be set to be reuse_len.")
    parser.add_argument('--bi_data', type=bool, default=False,
                        help="whether to create bidirectional data")
    parser.add_argument('--mask_alpha', type=int,
                        default=6, help="How many tokens to form a group.")
    parser.add_argument('--mask_beta', type=int,
                        default=1, help="How many tokens to mask within each group.")
    parser.add_argument('--num_predict', type=int,
                        default=85, help="Num of tokens to predict.")
    parser.add_argument('--mem_len', type=int,
                        default=384, help="Number of steps to cache")
    parser.add_argument('--num_epoch', type=int,
                        default=100, help="Number of epochs")

    args = parser.parse_args()

    sp = BertTokenizer.from_pretrained(args.tokenizer)
    model = XLNet(n_token=len(sp.vocab), n_layer=6, n_head=4, d_head=8,
                  d_inner=32, d_model=32,
                  dropout=0.1, dropatt=0.1,
                  attn_type="bi", bi_data=args.bi_data,
                  clamp_len=-1, same_length=False,
                  reuse_len=args.reuse_len, mem_len=args.mem_len)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    for num_epoch in range(args.num_epoch):
        mems = None

        '''
        feature: 
            input: int32 array
            is_masked: bool array
            target: int32 array, one word shift away from input
            seg_id: 0...1...
            label: 0
        '''
        features = data_utils._create_data(sp=sp,
                                           input_paths=args.data,
                                           seq_len=args.seq_len,
                                           reuse_len=args.reuse_len,
                                           bi_data=args.bi_data,
                                           num_predict=args.num_predict,
                                           mask_alpha=args.mask_alpha,
                                           mask_beta=args.mask_beta)


        num_step = 0
        for feature in features:

            '''
            Various mask types:
                        
                # Set the permutation indices of non-masked (& non-functional) tokens to the
                # smallest index (-1):
                # (1) they can be seen by all other positions
                # (2) they cannot see masked positions, so there won't be information leak
            
                # Create `target_mask`: non-functional and masked tokens
                # 1: use mask as input and have loss
                # 0: use token (or [SEP], [CLS]) as input and do not have loss
            
                # Create `perm_mask`
                # `target_tokens` cannot see themselves
                # put `rev_index` if real mask(not cls or sep) else `rev_index + 1`
                self_rev_index = torch.where(target_tokens, rev_index, rev_index + 1)
            
                # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
                # 0: can attend if i > j or j is non-masked
                perm_mask = (self_rev_index[:, None] <= rev_index[None, :]) & masked_or_func_tokens.bool()
                perm_mask = perm_mask.type(torch.float32)
            
                # new target: [next token] for LM and [curr token] (self) for PLM
                new_targets = torch.cat([inputs[0: 1], targets[: -1]], dim=0)
            
                # construct inputs_k
                inputs_k = inputs
            
                # construct inputs_q
                inputs_q = target_mask
            
                return perm_mask, new_targets, target_mask, inputs_k, inputs_q
            '''

            permutation = data_utils.make_permute(feature,
                                                  reuse_len=args.reuse_len,
                                                  seq_len=args.seq_len,
                                                  perm_size=args.perm_size,
                                                  num_predict=args.num_predict)

            # batch size is 1
            inp_k = permutation['input_k'].unsqueeze(-1)  # [seq_len, 1(=bsz)]
            seg_id = permutation['seg_id'].unsqueeze(-1)  # [seq_len, 1(=bsz)]
            target = permutation['target'].unsqueeze(-1)  # [num_predict, 1(=bsz)]
            perm_mask = permutation['perm_mask'].unsqueeze(-1)  # [seq_len, seq_len, 1(=bsz)]
            target_mapping = \
                permutation['target_mapping'].unsqueeze(-1)  # [num_predict, seq_len, 1(=bsz)]
            inp_q = permutation['input_q'].unsqueeze(-1)  # [seq_len, 1(=bsz)]
            tgt_mask = permutation['target_mask'].unsqueeze(-1)  # [num_predict, 1(=bsz)]

            # logits size [seq_len, 1, voc_size]
            logits, new_mems = model(inp_k=inp_k, seg_id=seg_id, input_mask=None,
                                     mems=mems, perm_mask=perm_mask,
                                     target_mapping=target_mapping, inp_q=inp_q)

            #print(logits.size())

            # crossentropy loss accumulated on targeted predictions
            lm_loss = criterion(logits.transpose(1, 2), target).type(torch.float32)
            tgt_mask_sum = tgt_mask.reshape(-1).sum()
            lm_loss_sum = (lm_loss * tgt_mask).reshape(-1).sum()

            optimizer.zero_grad()
            total_loss = lm_loss_sum / tgt_mask_sum
            print('Number of Epoch: %04d in %04d Step' % ((num_epoch + 1), (num_step + 1)),
                  'cost =', '{:.6f}'.format(total_loss))
            num_step += 1

            total_loss.backward()
            optimizer.step()

            mems = new_mems
