import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
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
        # RGB encoder: remove avgpool and fc layers to get spatial map
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-2]
        self.rgb_encoder = nn.Sequential(*modules)
        self._freeze_layer(self.rgb_encoder)

        # Run through encoder once to get feature dimensions and size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *self.image_size)
            rgb_feats = self.rgb_encoder(dummy)
            self.rgb_feature_dim = rgb_feats.shape[1]
            self.rgb_feature_size = rgb_feats.shape[2], rgb_feats.shape[3]

        self.event_encoder = EventVoxelEncoder(
            num_voxels=5,
            embed_dim=1024,
            patch_size=16,
            image_size=self.image_size)
        
        # Define RGB and event projection layers
        self.combined_proj = nn.Linear(self.rgb_feature_dim + self.event_encoder.embed_dim, self.embed_dim)
        self.gt_proj = nn.Linear(self.rgb_feature_dim, self.embed_dim)

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
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 8, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
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

    def forward(self, rgb_frame, event_voxels, gt_rgb_frame=None):
        # Encode and embed RGB frame
        rgb_feats = self.rgb_encoder(rgb_frame)
        batch_size, channels, height, width = rgb_feats.shape

        # Flatten rgb tokens into [B, H*W, C]
        rgb_emb = rgb_feats.flatten(2).transpose(1,2).mean(dim=1)

        # Encode and embed event stream voxels
        event_emb = self.event_encoder(event_voxels).mean(dim=1)

        # Concatenate RGB and Event tokens into fused embedding
        combined_tokens = torch.cat([rgb_emb, event_emb], dim=1)
        fused_emb = self.combined_proj(combined_tokens)

        # Encode positions onto RGB tokens
        fused_emb = fused_emb.unsqueeze(1)
        fused_emb += self.pos_encoder[:, :fused_emb.size(1), :]

        # Optional: Pass ground truth RGB frame through encoder -> embedding -> FC pipeline
        gt_rgb_emb = None
        if gt_rgb_frame != None:
            gt_rgb_enc = self.rgb_encoder(gt_rgb_frame)
            gt_rgb_tokens = gt_rgb_enc.flatten(2).transpose(1, 2)
            gt_rgb_emb = self.gt_proj(gt_rgb_tokens).mean(dim=1)

        # Cross-attention decoder
        rgb_tokens = rgb_feats.flatten(2).transpose(1, 2)
        fused_tokens = self.fusion_decoder(tgt=rgb_tokens, memory=fused_emb)

        # Reshape tokens into 2D spatial map
        _, num_pixels, channels = fused_tokens.shape
        H, W = self.rgb_feature_size
        if (H * W) != num_pixels:
            H = W = int(num_pixels ** 0.5)
        spatial_map = fused_tokens.transpose(1, 2).view(batch_size, channels, H, W)
        
        # Convolve spatial map
        img = self.head(spatial_map)

        # Interpolate generated image to input resolution
        predicted_frame = F.interpolate(img, size=self.image_size, mode="bilinear", align_corners=False)
        
        # Training mode: return predicted frame alongside Event+Voxel FC output and GT RGB FC output
        return predicted_frame, fused_emb.squeeze(1), gt_rgb_emb