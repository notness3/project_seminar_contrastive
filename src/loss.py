import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self,
                 emb_size: int,
                 num_classes: int,
                 device: str = 'cuda',
                 s: float = 64.0,
                 m: float = 0.5,
                 eps: float = 1e-6,
                 **kwargs
                 ):
        super(ArcFaceLoss, self).__init__()

        self.in_features = emb_size
        self.out_features = num_classes

        self.s = s
        self.m = m

        self.threshold = math.pi - m
        self.eps = eps

        self.device = device

        self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, target: torch.LongTensor = None) -> torch.Tensor:
        cos_theta = F.linear(F.normalize(x), F.normalize(self.weight.to(self.device)))

        if not target.numel():
            return cos_theta

        theta = torch.acos(torch.clamp(cos_theta, -1 + self.eps, 1.0 - self.eps))

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        mask = torch.where(theta > self.threshold, torch.zeros_like(one_hot), one_hot)

        logits = torch.cos(torch.where(mask.bool(), theta + self.m, theta))

        logits *= self.s

        return logits


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin

    @staticmethod
    def calc_euclidean(x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

