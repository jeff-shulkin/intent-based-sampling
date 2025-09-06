import torch
import torch.nn as nn
import torch.nn.functional as F

class EventEmbed(nn.Module):
    def __init__(self, num_voxels, d_model, image_size=(224, 224), patch_size=16):
        super().__init__()
        self.num_voxels = num_voxels
        self.image_size=image_size
        self.patch_size = patch_size

        # Calculate the correct number of patches
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        # Input dimension: num_voxels * patch_size^2
        input_dim = num_voxels * (patch_size ** 2)
        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, event_voxel_tensor):
        batch_size = event_voxel_tensor.size(0)
        height, width = event_voxel_tensor.size(2), event_voxel_tensor.size(3)
        
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            print(f"Image dimensions ({height}x{width}) must be divisible by patch size {self.patch_size}")
        
        # Split into patches
        patches = event_voxel_tensor.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, self.num_voxels, self.num_patches, -1)
        patches = patches.view(batch_size, self.num_patches, -1)
        return self.proj(patches)