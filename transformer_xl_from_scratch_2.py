# https://mlexplained.com/2019/07/04/building-the-transformer-xl-from-scratch/
# https://github.com/keitakurita/Practical_NLP_in_PyTorch/blob/master/deep_dives/transformer_xl_from_scratch.ipynb

from typing import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda:1")

'''
A Single Attention Head
We'll start off by implementing a single attention head in a MultiHeadAttention layer. 
To make things concrete, let's consider the first layer and assume we receive an input of word embeddings
of shape (seq=7, batch_size=3, embedding_dim=32). 
Note that the Transformer XL does not add positional embeddings to the input.
'''
seq, batch_size, embedding_dim = 7, 3, 32
word_embs = torch.rand(seq, batch_size, embedding_dim)

prev_seq = 6
memory = torch.rand(prev_seq, batch_size, embedding_dim) # hidden states from the previous sequence


'''
Each attention head takes keys, queries, and values as input. The processing goes like this:

Apply a separate linear transformation to each of the keys, queries, and values.
Compute attention scores for each of the values.
For each query, compute an attention-weighted sum of the values.
Apply a residual connection and layer normalization.
We'll start off with the linear transformation.
'''
inner_dim = 17 # this will be the internal dimension
linear_k = nn.Linear(embedding_dim, inner_dim)
linear_v = nn.Linear(embedding_dim, inner_dim)
linear_q = nn.Linear(embedding_dim, inner_dim)

'''

The memory is concatenated across the sequence dimension and fed as keys/values. 
Be careful, as it's not concatenated with the queries. 
This is because each query represents one word we want to predict, 
so we can't modify the number of queries.
'''
word_embs_w_memory = torch.cat([memory, word_embs], dim=0)
k_tfmd = linear_k(word_embs_w_memory)
v_tfmd = linear_v(word_embs_w_memory)
q_tfmd = linear_q(word_embs) # No memory for the queries

'''
Now, we compute scaled dot product attention as per the usual Transformer. 
'''
content_attn = torch.einsum("ibd,jbd->ijb", q_tfmd, k_tfmd) / (embedding_dim ** 0.5) # scale

'''
Relative positional encodings
Transformer XL computes an embedding that represents the distance between any two tokens. This is used to compute the attention between the two words.
'''
u = torch.rand(17).expand_as(q_tfmd)
content_attn = content_attn + torch.einsum("ibd,jbd->ijb", u, k_tfmd) / (embedding_dim ** 0.5)
pos_idxs = torch.arange(seq + prev_seq - 1, -1, -1.0, dtype=torch.float)
print(pos_idxs)

inv_freq = 1 / (10000 ** (torch.arange(0.0, embedding_dim, 2.0) / embedding_dim))
sinusoid_inp = torch.einsum("i,j->ij", pos_idxs, inv_freq)
plt.plot(sinusoid_inp[0, :].detach().numpy())
plt.plot(sinusoid_inp[6, :].detach().numpy())

relative_positional_embeddings = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)[:,None,:]
print(relative_positional_embeddings.shape)

'''
We also apply transformations to the positional embeddings separate from the values/keys.
'''
linear_p = nn.Linear(embedding_dim, inner_dim)
pos_tfmd = linear_p(relative_positional_embeddings)

'''
This time, we'll be adding the positional bias during attention computation.
'''

## inefficient implementation
v = torch.rand(17) # positional bias
pos_attn = torch.einsum("ibd,jd->ijb", q_tfmd + v, pos_tfmd[:,0,:]) / (embedding_dim ** 0.5) # scale
print(pos_attn.shape)

# Use padding + shifting to efficiently compute the attention for all
zero_pad = torch.zeros((seq, 1, batch_size), dtype=torch.float)
pos_attn = (torch.cat([zero_pad, pos_attn], dim=1)
                    .view(seq + prev_seq + 1, seq, batch_size)[1:]
                    .view_as(pos_attn))
print(pos_attn.shape)


'''The attention is computed as the sum of content and positional attention.
'''
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
mask = mask.bool()[...,None]
print(mask.shape)
raw_attn = raw_attn.masked_fill(mask, -float('inf'))   # if is one, then fill -inf attention value, after softmax is zero
print(raw_attn[:,:,0])


'''We can now compute the outputs as the weighted sum of the value vectors using the attention scores.
'''
attn = torch.softmax(raw_attn, dim=1)
attn_weighted_sum = torch.einsum("ijb,jbd->ibd", attn, v_tfmd)
print(attn_weighted_sum.shape)

'''Finally, we project the attention weighted sums back to their original dimension and 
apply a residual connection and layer normalization. 
We apply layer normalization after the residual connection.'''

linear_out = nn.Linear(inner_dim, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)
output = layer_norm(word_embs + linear_out(attn_weighted_sum))
print(output.shape)