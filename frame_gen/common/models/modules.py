import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class EventEmbed(nn.Module):
    def __init__(self, num_voxels, d_model, image_size=(224, 224), patch_size=16, flatten=True):
        super().__init__()
        self.num_voxels = num_voxels
        self.image_size=image_size
        self.patch_size = patch_size
        self.flatten = flatten

        # Calculate the correct number of patches
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        # create projection layer
        self.proj = nn.Conv2d(in_channels=num_voxels, out_channels=d_model, kernel_size=patch_size, stride=patch_size, bias=True)

        # create normalization layer
        self.norm = nn.LayerNorm(d_model)

        # initialize weights
        nn.init.xavier_uniform_(self.proj.weight, gain=0.02)
        if self.proj.bias != None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, event_voxel_tensor):
        batch_size, channels, width, height = event_voxel_tensor.shape

        if event_voxel_tensor.dim() != 4:
            print(f"Expected 4D input [B, C, W, H], got shape {tuple(event_voxel_tensor.shape)}")
            sys.exit(1)

        if channels != self.num_voxels:
            print(f"Expected {self.num_voxels} voxels, received {channels} voxels instead.")
            sys.exit(1)
        
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            print(f"Image dimensions ({height}x{width}) must be divisible by patch size {self.patch_size}")
            sys.exit(1)

        x = self.proj(event_voxel_tensor)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)

        return self.norm(x)