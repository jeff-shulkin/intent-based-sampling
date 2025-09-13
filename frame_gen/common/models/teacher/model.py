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
        self.rgb_encoder = timm.create_model(
            "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
            pretrained=True,
            features_only=True
        )
        self._freeze_layer(self.rgb_encoder)

        self.event_encoder = EventVoxelEncoder(
            num_voxels=5,
            embed_dim=1024,
            patch_size=16,
            image_size=self.image_size)
        
        # Define RGB and event projection layers
        rgb_feature_info = self.rgb_encoder.feature_info[-1]
        rgb_feature_dim = rgb_feature_info["num_chs"]
        reduction_factor = rgb_feature_info["reduction"]
        if isinstance(reduction_factor, int):
            self.rgb_feature_size = (
                image_size[0] // reduction_factor,
                image_size[1] // reduction_factor
            )
        else:  # it's a tuple
            self.rgb_feature_size = (
                image_size[0] // reduction_factor[0],
                image_size[1] // reduction_factor[1]
            )

        self.rgb_proj = nn.Linear(rgb_feature_dim, self.embed_dim)
        self.event_proj = nn.Linear(self.event_encoder.embed_dim, self.embed_dim)

        # Define position encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, self.rgb_feature_size[0] * self.rgb_feature_size[1], embed_dim))

        # Define fusion decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.nhead,
            dim_feedforward=nhid,
            dropout=self.dropout,
            batch_first=True,
            bias=True,
            device=self.device
        )
        self.fusion_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=nlayers
        )
        
        # Define head projection
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(4 * self.embed_dim, 3 * (patch_size ** 2))
        )

        # Initialize layer weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias != None:
                nn.init.constant_(module.bias, 0)

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
        batch_size = rgb_frame.size(0)
        
        # Encode RGB and event voxels
        rgb_feats = self.rgb_encoder(rgb_frame)[-1]
        rgb_tokens = rgb_feats.flatten(2).transpose(1, 2)

        event_tokens = self.event_encoder(event_voxels)

        # Project RGB and event encodings to embed_dim
        rgb_tokens = self.rgb_proj(rgb_tokens)
        event_tokens = self.event_proj(event_tokens)

        # Encode positions onto RGB tokens
        rgb_tokens += self.pos_encoder[:, :rgb_tokens.size(1), :]

        # Cross-attention decoder
        fused_tokens = self.fusion_decoder(tgt=rgb_tokens, memory=event_tokens)
        
        # Predict RGB patches
        rgb_patches = self.head(fused_tokens)

        # Reconstruct image from patches
        H, W = self.rgb_feature_size
        patches = rgb_patches.view(batch_size, H, W, 3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5)  # [B, 3, H, patch_size, W, patch_size]
        img = patches.contiguous().view(
            batch_size, 3, H * self.patch_size, W * self.patch_size
        )

        return img