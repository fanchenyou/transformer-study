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