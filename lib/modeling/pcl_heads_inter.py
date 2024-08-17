import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from core.config import cfg
import nn as mynn
import utils.net as net_utils
from collections import OrderedDict
import numpy as np




class MIL(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.mil_score0 = nn.Linear(dim_in, dim_out)
        self.mil_score1 = nn.Linear(dim_in, dim_out)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.mil_score0.weight, std=0.01)
        init.constant_(self.mil_score0.bias, 0)
        init.normal_(self.mil_score1.weight, std=0.01)
        init.constant_(self.mil_score1.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'mil_score0.weight': 'mil_score0_w',
            'mil_score0.bias': 'mil_score0_b',
            'mil_score1.weight': 'mil_score1_w',
            'mil_score1.bias': 'mil_score1_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        mil_score0 = self.mil_score0(x)
        mil_score1 = self.mil_score1(x)
        attention_map = F.softmax(mil_score0, dim=0)
        cls_score = F.softmax(mil_score1, dim=1)  # num_box * 20
        mil_score = attention_map * cls_score

        # import pdb; pdb.set_trace()

        return mil_score




class Embedding(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.embedding = nn.Linear(dim_in, dim_out)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.embedding.weight, std=0.01)
        init.constant_(self.embedding.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'embedding.weight': 'embedding_w',
            'embedding.bias': 'embedding_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        embedding = self.embedding(x)

        return embedding




class Recover_feat(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.recover_feat = nn.Linear(dim_in, dim_out)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.recover_feat.weight, std=0.01)
        init.constant_(self.recover_feat.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'recover_feat.weight': 'recover_feat_w',
            'recover_feat.bias': 'recover_feat_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        recover_feat = self.recover_feat(x)

        return recover_feat




class refine_outputs(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.refine_score = [] 
        for i_refine in range(cfg.REFINE_TIMES):
            self.refine_score.append(nn.Linear(dim_in, dim_out))
        self.refine_score = nn.ModuleList(self.refine_score)

        self._init_weights()

    def _init_weights(self):
        for i_refine in range(cfg.REFINE_TIMES):
            init.normal_(self.refine_score[i_refine].weight, std=0.01)
            init.constant_(self.refine_score[i_refine].bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        for i_refine in range(cfg.REFINE_TIMES):
            detectron_weight_mapping.update({
                'refine_score.%d.weight' % i_refine: 'refine_score%d_w' % i_refine,
                'refine_score.%d.bias' % i_refine: 'refine_score%d_b' % i_refine
            })
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        refine_score = [F.softmax(refine(x), dim=1) for refine in self.refine_score]

        return refine_score


def mil_losses(cls_score, labels):
    cls_score = cls_score.clamp(1e-6, 1 - 1e-6)
    labels = labels.clamp(0, 1)
    loss = -labels * torch.log(cls_score) - (1 - labels) * torch.log(1 - cls_score)

    return loss.mean()

