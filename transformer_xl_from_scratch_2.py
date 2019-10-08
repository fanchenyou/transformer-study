"""
This is a tutorial on how to define a simple XLNet which has a single
attention head from scratch.


tutorial: https://mlexplained.com/2019/07/04/building-the-transformer-xl-from-scratch/
code: https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/deep_dives/transformer_xl_from_scratch.ipynb

"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")


# define params
seq, batch_size, embedding_dim = 7, 3, 32
prev_seq = 6  # previous seq len store in memory
inner_dim = 17  # internal dimension

# word embeddings
word_embs = torch.rand(seq, batch_size, embedding_dim)

# hidden states from the previous sequence
memory = torch.rand(prev_seq, batch_size, embedding_dim)

# linear transformation to each of the keys, queries, and values
linear_k = nn.Linear(embedding_dim, inner_dim)
linear_v = nn.Linear(embedding_dim, inner_dim)
linear_q = nn.Linear(embedding_dim, inner_dim)

# memory is concatenated across the sequence dimension and fed as keys/values.
word_embs_w_memory = torch.cat([memory, word_embs], dim=0)
k_tfmd = linear_k(word_embs_w_memory)
v_tfmd = linear_v(word_embs_w_memory)
q_tfmd = linear_q(word_embs)  # No memory for the queries

# content attention: compute scaled dot product attention as usual Transformer.
content_attn = torch.einsum("ibd,jbd->ijb", q_tfmd, k_tfmd) / (embedding_dim ** 0.5)


# Relative positional encodings, which represents the distance between any two tokens.
u = torch.rand(17).expand_as(q_tfmd)
content_attn = content_attn + torch.einsum("ibd,jbd->ijb", u, k_tfmd) / (embedding_dim ** 0.5)
# [12., 11., 10.,  9.,  8.,  7.,  6.,  5.,  4.,  3.,  2.,  1.,  0.])
pos_idxs = torch.arange(seq + prev_seq - 1, -1, -1.0, dtype=torch.float)
inv_freq = 1 / (10000 ** (torch.arange(0.0, embedding_dim, 2.0) / embedding_dim))
sinusoid_inp = torch.einsum("i,j->ij", pos_idxs, inv_freq)
relative_positional_embeddings = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)[:, None, :]


# apply transformations to the positional embeddings
linear_p = nn.Linear(embedding_dim, inner_dim)
pos_tfmd = linear_p(relative_positional_embeddings)

# add the positional bias during attention computation.
v = torch.rand(17)  # positional bias
pos_attn = torch.einsum("ibd,jd->ijb", q_tfmd + v, pos_tfmd[:, 0, :]) / (embedding_dim ** 0.5)  # scale
print(pos_attn.shape)


# The attention is computed as the sum of content and positional attention.
raw_attn = content_attn + pos_attn

'''When we do language modeling, we need to prevent the model from peeping words to be predicting. 
In the Transformer, we achieve this by setting the attention score to zero. 
This masks out words that we don't want the model to be able to see.'''
mask = torch.triu(
    torch.ones((seq, seq + prev_seq)),
    diagonal=1 + prev_seq,
)
print(mask)
print(mask.shape)
mask = mask.bool()[..., None]
print(mask.shape)
raw_attn = raw_attn.masked_fill(mask, -float('inf'))  # if is one, then fill -inf attention value, after softmax is zero
print(raw_attn[:, :, 0])

# the outputs is the weighted sum of the value vectors using the attention scores.
attn = torch.softmax(raw_attn, dim=1)
attn_weighted_sum = torch.einsum("ijb,jbd->ibd", attn, v_tfmd)

'''Finally, we project the attention weighted sums back to their original dimension and 
apply a residual connection and layer normalization. 
We apply layer normalization after the residual connection.'''

linear_out = nn.Linear(inner_dim, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)
output = layer_norm(word_embs + linear_out(attn_weighted_sum))
print(output.shape)
