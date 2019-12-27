import torch
import numpy as np

"""
Example usage:

# Regular segmentation loss:
ys_pred_oai = self.models['segm'](xs_oai)
loss_segm = self.losses['segm'](input_=ys_pred_oai,
                                target=ys_true_arg_oai)

# Mixup
xs_mixup, ys_mixup_a, ys_mixup_b, lambda_mixup = mixup_data(
    x=xs_oai, y=ys_true_arg_oai,
    alpha=self.config['mixup_alpha'], device=maybe_gpu)
ys_pred_oai = self.models['segm'](xs_mixup)
loss_segm = mixup_criterion(criterion=self.losses['segm'],
                            pred=ys_pred_oai,
                            y_a=ys_mixup_a,
                            y_b=ys_mixup_b,
                            lam=lambda_mixup)
"""


def mixup_data(x, y, alpha=1.0, device='cpu'):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
