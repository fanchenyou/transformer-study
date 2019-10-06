## Several Transformer network variants tutorials

#### 1. transformer_encoder, [src](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
    * Use Pytorch nn.transformer package to build an encoder for language prediction
   
#### 2. transformer_xl_from_scratch, [src](https://mlexplained.com/2019/07/04/building-the-transformer-xl-from-scratch/)
    * Build Transformer + XLNet
    
#### 3. transformer_xl_multihead, [src](https://mlexplained.com/2019/07/04/building-the-transformer-xl-from-scratch/)
    * Build Transformer + XLNet + MultiAttention heads

#### 4. xlnet, [paper](https://arxiv.org/pdf/1906.08237.pdf), [src](https://github.com/graykode/xlnet-Pytorch), [tutorial](https://towardsdatascience.com/what-is-xlnet-and-why-it-outperforms-bert-8d8fce710335)
    * An excellent tutorial version of XLNet from above link
    * Add more comments for understanding
    * Requirements: Python 3 + Pytorch v1.2 
    * TODO: Add GPU support

#### 5. Bert from scratch, [paper](https://arxiv.org/abs/1810.04805), [src](https://github.com/codertimo/BERT-pytorch/tree/master/bert_pytorch/model)
    * Build Bert - Bidirectional Transformer
    * The task is two-fold, see paper section 3.1
        1) to predict the second part of a sentence (Next Sentence Prediction)
        2) to predict the masked words of a sentence (Masked LM)
    * step 1: generate vocabulary file "vocab.small" in ./data
    * step 2: train the network
    * See transformer_bert_from_scratch_5.py for more details.

#### 6. Bert from Pytorch Official Implementation, [paper](https://arxiv.org/abs/1810.04805), [src](https://github.com/huggingface/transformers)
    * Build Bert - Bidirectional Transformer
    * Utilize official Pytorch API to implement the interface of using existing code and pre-trained model
    * pip install transformers tb-nightly 


#### TODO. ALBERT, [paper](https://arxiv.org/abs/1909.11942v1)
    * Lite BERT



### Requirements
Python = 3.6+
 
PyTorch = 1.2+ [[here]](https://pytorch.org/)

GPU training with 4G+ memory, testing with 1G+ memory.
