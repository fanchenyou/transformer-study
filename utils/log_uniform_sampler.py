'''
source:
https://github.com/huggingface/transformers/blob/master/transformers/modeling_transfo_xl_utilities.py
'''

import torch

class LogUniformSampler(object):
    def __init__(self, range_max, n_sample):
        """
        Reference : https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/candidate_sampling_ops.py
            `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`
        expected count can be approximated by 1 - (1 - p)^n
        and we use a numerically stable version -expm1(num_tries * log1p(-p))
        Our implementation fixes num_tries at 2 * n_sample, and the actual #samples will vary from run to run
        """
        with torch.no_grad():
            self.range_max = range_max
            log_indices = torch.arange(1., range_max + 2., 1.).log_()
            self.dist = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]

            self.log_q = (- (-self.dist.double().log1p_() * 2 * n_sample).expm1_()).log_().float()

        self.n_sample = n_sample

    def sample(self, labels):
        """
            labels: [b1, b2]
        Return
            true_log_probs: [b1, b2]
            samp_log_probs: [n_sample]
            neg_samples: [n_sample]
        """

        # neg_samples = torch.empty(0).long()
        n_sample = self.n_sample
        n_tries = 2 * n_sample

        with torch.no_grad():
            neg_samples = torch.multinomial(self.dist, n_tries, replacement=True).unique()
            device = labels.device
            neg_samples = neg_samples.to(device)
            true_log_probs = self.log_q[labels].to(device)
            samp_log_probs = self.log_q[neg_samples].to(device)
            return true_log_probs, samp_log_probs, neg_samples


def sample_logits(embedding, bias, labels, inputs, sampler):
    """
        embedding: an nn.Embedding layer
        bias: [n_vocab]
        labels: [b1, b2]
        inputs: [b1, b2, n_emb]
        sampler: you may use a LogUniformSampler
    Return
        logits: [b1, b2, 1 + n_sample]
    """
    true_log_probs, samp_log_probs, neg_samples = sampler.sample(labels)
    n_sample = neg_samples.size(0)
    b1, b2 = labels.size(0), labels.size(1)
    all_ids = torch.cat([labels.view(-1), neg_samples])
    all_w = embedding(all_ids)
    true_w = all_w[: -n_sample].view(b1, b2, -1)
    sample_w = all_w[- n_sample:].view(n_sample, -1)

    all_b = bias[all_ids]
    true_b = all_b[: -n_sample].view(b1, b2)
    sample_b = all_b[- n_sample:]

    hit = (labels[:, :, None] == neg_samples).detach()

    true_logits = torch.einsum('ijk,ijk->ij',
                               [true_w, inputs]) + true_b - true_log_probs
    sample_logits = torch.einsum('lk,ijk->ijl',
                                 [sample_w, inputs]) + sample_b - samp_log_probs
    sample_logits.masked_fill_(hit, -1e30)
    logits = torch.cat([true_logits[:, :, None], sample_logits], -1)

    return logits
