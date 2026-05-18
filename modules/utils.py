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
    def __init__(self, margin=1.0, min_label_gap=1e-6):
        super().__init__()
        self.margin = margin
        self.min_label_gap = min_label_gap

    def forward(self, preds, labels, sample_weights=None):
        labels = labels.to(dtype=preds.dtype)
        if sample_weights is not None:
            sample_weights = sample_weights.to(device=preds.device, dtype=preds.dtype)
        losses = []
        for class_idx in range(labels.shape[1]):
            class_labels = labels[:, class_idx]
            label_diff = class_labels[:, None] - class_labels[None, :]
            pair_mask = label_diff > self.min_label_gap
            if not torch.any(pair_mask):
                continue

            pred_diff = preds[:, class_idx][:, None] - preds[:, class_idx][None, :]
            pair_weights = label_diff.clamp_min(0.0)
            if sample_weights is not None:
                pair_weights = pair_weights * sample_weights[:, None] * sample_weights[None, :]

            weighted_loss = F.softplus(-pred_diff * self.margin) * pair_weights
            losses.append(
                weighted_loss[pair_mask].sum()
                / pair_weights[pair_mask].sum().clamp_min(1e-6)
            )

        if not losses:
            return preds.sum() * 0.0
        return torch.stack(losses).mean()
