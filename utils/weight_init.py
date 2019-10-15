import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.uniform_(m.weight, -0.1, 0.1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, 0.01)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            nn.init.uniform_(m.weight, -0.1, 0.1)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            nn.init.uniform_(m.cluster_weight, -0.1, 0.1)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            nn.init.constant_(m.bias, 0.0)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, 0.01)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            nn.init.uniform_(m.r_emb, -0.1, 0.1)
        if hasattr(m, 'r_w_bias'):
            nn.init.uniform_(m.r_w_bias, -0.1, 0.1)
        if hasattr(m, 'r_r_bias'):
            nn.init.uniform_(m.r_r_bias, -0.1, 0.1)
        if hasattr(m, 'r_bias'):
            nn.init.constant_(m.r_bias, 0.0)

