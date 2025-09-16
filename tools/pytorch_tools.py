import torch
from torch.utils.data import random_split
import numpy as np
from typing import Sequence

def determine_device():
    """Return GPU if available, otherwise CPU"""
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def split_dataset(dataset, ptrain: float, pval: float):
    """Wrapper for random_split. Given dataset, return train-val-test subsets"""
    train_size = int(ptrain * len(dataset))
    val_size = int(pval * len(dataset))
    test_size = int(len(dataset) - train_size - val_size)

    return random_split(dataset, [train_size, val_size, test_size])

def events_to_voxel(events, num_bins=5, image_size=(240, 320)):
    """
    Convert events (t, x, y, p) to a voxel grid tensor of shape [C, H, W],
    where C=num_bins, H=img_size[0], W=img_size[1].
    Polarity is split across bins and summed.
    """
    C = num_bins
    H, W = image_size
    voxel_grid = np.zeros((C, H, W), dtype=np.float32)

    if len(events) == 0:
        return voxel_grid

    t, x, y, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]

    # normalize timestamps to [0, num_bins-1]
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-6)  # avoid div0
    bin_idx = np.floor(t_norm * (C - 1)).astype(np.int64)

    # clamp coordinates
    x = np.clip(x.astype(int), 0, W - 1)
    y = np.clip(y.astype(int), 0, H - 1)

    for b, xi, yi, pi in zip(bin_idx, x, y, p):
        voxel_grid[b, yi, xi] += pi  # sum polarity

    return voxel_grid

def _lpips_scale(im: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    return 2 * im - 1