import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])
        
        n_repeats = length // len(y)
        epsilon = length % len(y)
        
        y = np.concatenate([y]*n_repeats + [y[:epsilon]])
        
    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)

        y = y[start:start + length]

    return y


class SoftAUCLoss(nn.Module):
    def __init__(self, margin=1.0, pos_weight=1.0, neg_weight=1.0):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, preds, labels, sample_weights=None):
        labels = labels.to(dtype=preds.dtype)
        if sample_weights is not None:
            sample_weights = sample_weights.to(device=preds.device, dtype=preds.dtype)
        losses = []
        for class_idx in range(labels.shape[1]):
            class_labels = labels[:, class_idx]
            pos_mask = class_labels > 0.5
            neg_mask = class_labels < 0.5
            if not torch.any(pos_mask) or not torch.any(neg_mask):
                continue

            pos_preds = preds[pos_mask, class_idx]
            neg_preds = preds[neg_mask, class_idx]
            pos_weights = self.pos_weight * (class_labels[pos_mask] - 0.5)
            neg_weights = self.neg_weight * (0.5 - class_labels[neg_mask])
            if sample_weights is not None:
                pos_weights = pos_weights * sample_weights[pos_mask]
                neg_weights = neg_weights * sample_weights[neg_mask]

            diff = pos_preds[:, None] - neg_preds[None, :]
            pair_weights = pos_weights[:, None] * neg_weights[None, :]
            weighted_loss = F.softplus(-diff * self.margin) * pair_weights
            losses.append(weighted_loss.sum() / pair_weights.sum().clamp_min(1e-6))

        if not losses:
            return preds.sum() * 0.0
        return torch.stack(losses).mean()
