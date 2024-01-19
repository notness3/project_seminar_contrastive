import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


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
    def calc_distance(x1, x2):
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)

        cos_sim = (x1 * x2).sum(dim=1)

        return 1 - cos_sim


    @torch.no_grad()
    def log_stuff(self, pos_scores, neg_scores, prefix):
        pos_mean = pos_scores.mean().item()
        neg_mean = neg_scores.mean().item()
        difference = pos_mean - neg_mean

        wandb.log({
            f"{prefix}_pos_mean": pos_mean,
            f"{prefix}_neg_mean": neg_mean,
            f"{prefix}_difference": difference
        })

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, prefix: str) -> torch.Tensor:
        distance_positive = self.calc_distance(anchor, positive)
        distance_negative = self.calc_distance(anchor, negative)

        self.log_stuff(distance_positive, distance_negative, prefix)

        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class ContrastiveCrossEntropy(nn.Module):
    def __init__(self, margin=1.0, **kwargs):
        super().__init__()
        self.margin = margin

    @torch.no_grad()
    def log_stuff(self, pos_scores, neg_scores, prefix):
        pos_mean = pos_scores.mean().item()
        neg_mean = neg_scores.mean().item()
        difference = pos_mean - neg_mean

        wandb.log({
            f"{prefix}_pos_mean": pos_mean,
            f"{prefix}_neg_mean": neg_mean,
            f"{prefix}_difference": difference
        })

    def forward(self, vac_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor, prefix: str) -> torch.Tensor:
        pos_scores = (vac_emb * pos_emb).sum(dim=1)
        neg_scores = (vac_emb * neg_emb).sum(dim=1)

        self.log_stuff(pos_scores, neg_scores, prefix)

        loss_val = torch.exp(neg_scores + self.margin) - pos_scores

        # loss_val = torch.clamp(loss_val, min=1.001, max=2**16)
        loss_val[loss_val < 1] = 1

        return loss_val.log().mean()
