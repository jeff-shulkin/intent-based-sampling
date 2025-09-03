import torch
import torch.nn as nn
import torch.nn.functional as F

class EventEmbed(nn.Module):
    def __init__(self, num_voxels, d_model):
        super().__init__()
        self.num_voxels = num_voxels
        self.proj = nn.Linear(num_voxels * 4, d_model)

    def forward(self, event_voxel_tensor):
        batch_size = event_voxel_tensor.size()
        flattened_tensor = event_voxel_tensor.view(batch_size, -1)
        return self.proj(flattened_tensor)