import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from torchvision.utils import _log_api_usage_once
from tools.pytorch_tools import events_to_voxel

class Event_ToTensor:
    def __init__(self, image_size=(224, 224), num_bins=5) -> None:
        self.image_size = image_size
        self.num_bins = num_bins
        _log_api_usage_once(self)

    def __call__(self, event_array):
        voxel_array = events_to_voxel(event_array, self.num_bins, self.image_size)
        voxel_tensor = torch.from_numpy(voxel_array).float()
        if voxel_tensor.numel():
            voxel_tensor /= (voxel_tensor.max() + 1e-6)
        return voxel_tensor
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    

class EventVoxelEmbed(nn.Module):
    def __init__(self, num_voxels, d_model=1024, image_size=(224, 224), patch_size=16, flatten=True):
        super(EventVoxelEmbed, self).__init__()

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
        _, channels, height, width = event_voxel_tensor.shape

        if event_voxel_tensor.dim() != 4:
            print(f"Expected 4D input [B, C, H, W], got shape {tuple(event_voxel_tensor.shape)}")
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

class EventVoxelEncoder(nn.Module):
    def __init__(self, num_voxels=5, embed_dim=1024, patch_size=16, image_size=(224,224)):
        super(EventVoxelEncoder, self).__init__()

        self.num_voxels = num_voxels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        
        # Define the event embedding layer
        self.embed = EventVoxelEmbed(
            num_voxels=num_voxels,
            d_model=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
            flatten=True)
        
        # Define the internal encoder
        trans_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=(embed_dim * 4),
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.internal_transformer = nn.TransformerEncoder(trans_layer, num_layers=4)
    
    def forward(self, event_voxels):
        x = self.embed(event_voxels)
        x = self.internal_transformer(x)

        return x