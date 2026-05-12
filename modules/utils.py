import numpy as np

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
        pos_preds = preds[labels>0.5]
        neg_preds = preds[labels<0.5]
        pos_labels = labels[labels>0.5]
        neg_labels = labels[labels<0.5]

        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return torch.tensor(0.0, device=preds.device)

        pos_weights = torch.ones_like(pos_preds) * self.pos_weight * (pos_labels-0.5)
        neg_weights = torch.ones_like(neg_preds) * self.neg_weight * (0.5-neg_labels)
        if sample_weights is not None:
            sample_weights = torch.stack([sample_weights]*labels.shape[1], dim=1)
            pos_weights = pos_weights * sample_weights
            neg_weights = neg_weights * sample_weights

        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)  # [N_pos, N_neg]
        loss_matrix = torch.log(1 + torch.exp(-diff * self.margin))  # [N_pos, N_neg]

        weighted_loss = loss_matrix * pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)

        return weighted_loss.mean()