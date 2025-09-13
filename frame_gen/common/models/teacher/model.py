import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import timm
from frame_gen.common.models.modules import EventVoxelEncoder
import math


class FusionFrameGen(nn.Module):
    """
    This model is the Event-RGB fusion model model, whose goal is to predict the next frame given a previous RGB image and current event stream.
    The general goal of this model is to obtain max accuracy across a number of different datasets. 
    """

    def __init__(
            self,
            image_size=(224, 224),
            patch_size=16,
            embed_dim=512,
            nhid=2048,
            nhead=8,
            nlayers=8,
            dropout=0.1,
            num_voxels=1024,
            device=None
            ):
        super(FusionFrameGen, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.nhid = nhid
        self.nhead = nhead
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_voxels = num_voxels
        self.src_mask = None
        self.device = device

        self.image_height, self.image_width = self._pair(image_size)

        # Define RGB and event encoders
        self.rgb_encoder = timm.create_model("swinv2_large_window12to16_192to256.ms_in22k_ft_in1k", pretrained=True, features_only=True)
        self._freeze_layer(self.rgb_encoder)

        self.event_encoder = EventVoxelEncoder(
            num_voxels=5,
            embed_dim=1024,
            patch_size=16,
            image_size=self.image_size)
        
        # Define RGB and event projection layers
        rgb_feature_dim = self.rgb_encoder.feature_info[-1]["num_chs"]
        event_feature_dim = self.event_encoder.embed_dim

        self.rgb_proj = nn.Linear(rgb_feature_dim, self.embed_dim)
        self.event_proj = nn.Linear(event_feature_dim, self.embed_dim)

        # Define fusion decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first = True
        )
        self.fusion_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=nlayers
        )
        
        # Define head projection
        self.head = nn.Linear(self.embed_dim, 3 * patch_size * patch_size)

    @staticmethod
    def _pair(t):
        return t if isinstance(t, tuple) else (t, t)

    @staticmethod
    def _freeze_layer(layer):
        for param in layer.parameters():
            param.requires_grad = False

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz, device=self.device) * float('-inf'), diagonal=1)

    def forward(self, rgb_frame, event_voxels):
        # Encode RGB and event voxels
        rgb_feats = self.rgb_encoder(rgb_frame)[-1].flatten(2).transpose(1, 2)
        event_feats = self.event_encoder(event_voxels)

        # Project RGB and event encodings to embed_dim
        batch_size, _, _ = rgb_feats.shape
        rgb_tokens = self.rgb_proj(rgb_feats)
        event_tokens = self.event_proj(event_feats)

        # Cross-attention decoder
        decoded_tokens = self.fusion_decoder(tgt=rgb_tokens, memory=event_tokens)
        
        # Predict RGB patches
        rgb_patches = self.head(decoded_tokens)

        # Reconstruct image from patches
        H_patch = self.image_size[0] // self.patch_size
        W_patch = self.image_size[1] // self.patch_size
        
        # Reshape to final image
        img = rgb_patches.reshape(batch_size, H_patch, W_patch, 3, self.patch_size, self.patch_size)
        img = img.permute(0, 3, 1, 4, 2, 5)  # [B, 3, H_patch, patch_size, W_patch, patch_size]
        img = img.reshape(batch_size, 3, H_patch * self.patch_size, W_patch * self.patch_size)

        return img