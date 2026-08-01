import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class NTXentloss(nn.Module):
    def __init__(self, temperature=0.07, alpha=2, hidden_norm=True):
        super(NTXentloss, self).__init__()
        self.temperature = float(temperature)
        self.alpha = alpha
        self.hidden_norm = hidden_norm

    def forward(self, zis, zjs):
        assert zis.shape == zjs.shape, f"Shape mismatch: {zis.shape} vs {zjs.shape}"
        assert zis.dtype == zjs.dtype, f"Type mismatch: {zis.dtype} vs {zjs.dtype}"

        batch_size = zis.shape[0]
        device = zis.device

        if self.hidden_norm:
            zis = F.normalize(zis, p=2, dim=-1)
            zjs = F.normalize(zjs, p=2, dim=-1)

        features = torch.cat([zis, zjs], dim=0)

        similarity = torch.matmul(features, features.T)
        if self.hidden_norm:
            similarity = similarity.clamp(-1, 1)  

        mask = torch.eye(2 * batch_size, device=device)
        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), device=device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size)

        similarity = similarity / self.temperature

        pos_sim = torch.sum(similarity * pos_mask, dim=1)

        neg_mask = 1 - pos_mask - mask
        neg_sim = similarity * neg_mask

        max_neg_sim = torch.max(neg_sim, dim=1, keepdim=True)[0]
        max_neg_sim = torch.max(max_neg_sim, torch.zeros_like(max_neg_sim))  

        exp_neg = torch.exp(neg_sim - max_neg_sim)
        exp_neg = exp_neg * neg_mask

        neg_sum = torch.sum(exp_neg, dim=1)

        exp_pos = torch.exp(pos_sim - max_neg_sim.squeeze())

        denominator = exp_pos + neg_sum + 1e-8
        assert torch.all(denominator > 0), "Denominator contains non-positive values"

        consistency_loss = torch.mean(-torch.log(exp_pos / denominator))


        loss = self.alpha * consistency_loss

        return loss