import torch
import random
import numpy as np


def mixup(lq, gt, alpha=1.2):
    if random.random() < 0.5:
        return lq, gt

    v = np.random.beta(alpha, alpha)
    r_index = torch.randperm(lq.size(0)).to(gt.device)

    lq = v * lq + (1 - v) * lq[r_index, :]
    gt = v * gt + (1 - v) * gt[r_index, :]
    return lq, gt
