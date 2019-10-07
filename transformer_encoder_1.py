"""
Sequence-to-Sequence Modeling with nn.Transformer and TorchText
src: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

This is a tutorial on how to train a sequence-to-sequence model
that uses the nn.Transformer module.

This code shows how to predict a word which follows a sequence of words.
The ``nn.TransformerEncoder`` consists of multiple layers of
``nn.TransformerEncoderLayer``.

Along with the input sequence, a square attention mask is required
because the self-attention layers in ``nn.TransformerEncoder`` are
only allowed to attend the earlier positions in the sequence.

For the language modeling task, any tokens on the future
positions should be masked. To have the actual words, the output
of ``nn.TransformerEncoder`` model is sent to the final Linear
layer, which is followed by a log-Softmax function.
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.embedding import PositionalEmbedding

import torchtext
from torchtext.data.utils import get_tokenizer


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEmbedding(ninp)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        # refer to https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)  # multi-layer attention layers
        self.word_encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        """
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py

        These masks ensure that predictions for position i depend only on the unmasked positions
        j and are applied identically for each sequence in a batch.

        Since transformer is one-directional, so word_i cannot attend to later words
        So the attention weight is set to expt(-inf) = 0

        tensor([[0., -inf, -inf,  ..., -inf, -inf, -inf],
                [0., 0., -inf,  ..., -inf, -inf, -inf],
                [0., 0., 0.,  ..., -inf, -inf, -inf],
                ...,
                [0., 0., 0.,  ..., 0., -inf, -inf],
                [0., 0., 0.,  ..., 0., 0., -inf],
                [0., 0., 0.,  ..., 0., 0., 0.]])
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)  # upper triangle mask
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        self.word_encoder.weight.data.uniform_(-0.1, 0.1)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src)).to(src.device)
            self.src_mask = mask

        src = self.word_encoder(src) * math.sqrt(self.ninp)
        src = src + self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)


######################################################################
# The training process uses Wikitext-2 dataset from ``torchtext``. The
# vocab object is built based on the train dataset and is used to numericalize
# tokens into tensors. Starting from sequential data, the ``batchify()``
# function arranges the dataset into columns, trimming off any tokens remaining
# after the data has been divided into batches of size ``batch_size``.
# For instance, with the alphabet as the sequence (total length of 26)
# and a batch size of 4, we would divide the alphabet into 4 sequences of
# length 6.


TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

######################################################################
# ``get_batch()`` function generates the input and target sequence for
# the transformer-study model. It subdivides the source data into chunks of
# length ``bptt``. For the language modeling task, the model needs the
# following words as ``Target``. For example, with a ``bptt`` value of 2,
# we will get the following two Variables for ``i`` = 0:


bptt = 35


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


ntokens = len(TEXT.vocab.stoi)  # the size of vocabulary
emsize = 200  # embedding dimension
nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train():
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)

    # simply segment train_data to every segment of size bptt(=35)
    # for each segment, starts at every 1...bptt point and predict following words
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # train_data (104335, 20)
        data, targets = get_batch(train_data, i)  # ((35, 20), (700,))
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


######################################################################
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.

best_val_loss = float("inf")
epochs = 3  # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

######################################################################
# Evaluate the model with the test dataset

test_loss = evaluate(best_model, test_data)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
