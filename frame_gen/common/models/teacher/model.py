import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from timm.layers import PatchEmbed
from frame_gen.common.models.modules import EventEmbed
import math


class NextFrameTransformerTeacher(nn.Module):
    """
    This model is the "Teacher" transformer model, whose goal is to predict any N number of frames.
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
        super(NextFrameTransformerTeacher, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.nhid = nhid
        self.nhead = nhead
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_voxels = num_voxels
        self.src_mask = None

        image_height, image_width = self._pair(image_size)

        # Define patching and positional embedding
        self.rgb_emb = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim
        )

        self.event_emb = EventEmbed(
            num_voxels=num_voxels,
            d_model=embed_dim,
            image_size=image_size
        )

        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.rgb_pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )
        self.event_voxel_pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )
        self.query_pos = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # Define transformer early fusion encoder
        fusion_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_encoder = nn.TransformerEncoder(
            encoder_layer=fusion_encoder_layer,
            num_layers=nlayers,
            enable_nested_tensor=True,
        )

        # Define transformer decoder
        rgb_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first = True
        )
        self.rgb_decoder = nn.TransformerDecoder(
            decoder_layer=rgb_decoder_layer,
            num_layers=nlayers
        )

        # Define output projection layer
        ntoken = 3 * (image_height // patch_size) * (image_width // patch_size)
        self.output_proj = nn.Linear(embed_dim, (3 * patch_size ** 2))

        # Define patch -> original resolution upscaling
        self.upscale = nn.PixelShuffle(self.patch_size)

        # Initialize transformer weights
        self.init_weights()

    @staticmethod
    def _pair(t):
        return t if isinstance(t, tuple) else (t, t)

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz, device=self.device) * float('-inf'), diagonal=1)

    def init_weights(self):
        initrange = 0.1

        # PatchEmbed (RGB) convolution
        nn.init.xavier_uniform_(self.rgb_emb.proj.weight)
        if self.rgb_emb.proj.bias is not None:
            nn.init.zeros_(self.rgb_emb.proj.bias)

        # EventEmbed weights (if it has a linear layer)
        if hasattr(self.event_emb, 'proj'):
            nn.init.xavier_uniform_(self.event_emb.proj.weight)
            if self.event_emb.proj.bias is not None:
                nn.init.zeros_(self.event_emb.proj.bias)

        # Output projection layer
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, rgb_frame, events):
        # Generate tokens for both RGB and event frames
        rgb_tokens = self.rgb_emb(rgb_frame)
        event_tokens = self.event_emb(events)

        # Add positional embeddings for both rgb and event tokens
        rgb_tokens += self.rgb_pos_embedding[:, :rgb_tokens.size(1), :]
        event_tokens += self.event_voxel_pos_embedding[:, :event_tokens.size(1), :]

        # Concatenate tokens
        fused_tokens = torch.cat([rgb_tokens, event_tokens], dim=1)

        # Encode fused tokens
        encoded_tokens = self.fusion_encoder(fused_tokens, mask=self.src_mask)

        # Decode for next RGB frame generation
        batch_size = rgb_tokens.size(0)
        num_queries = rgb_tokens.size(1)
        query_pos = self.query_pos.expand(batch_size, -1, -1)
        decoded_tokens = self.rgb_decoder(query_pos, encoded_tokens)
        predicted_patches = self.output_proj(decoded_tokens)

        # Reshape patches back to image
        # predicted_patches shape: [batch_size, num_patches, 3 * patch_size * patch_size]
        batch_size = predicted_patches.size(0)
        patch_size = self.patch_size
        h_patches = self.image_size[0] // patch_size
        w_patches = self.image_size[1] // patch_size

        # Reshape to [batch_size, h_patches, w_patches, 3, patch_size, patch_size]
        predicted_patches = predicted_patches.view(batch_size, h_patches, w_patches, 3, patch_size, patch_size)

        # Rearrange to [batch_size, 3, height, width]
        predicted_patches = predicted_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        predicted_patches = predicted_patches.view(batch_size, 3, self.image_size[0], self.image_size[1])

        # Upscale patches into full image_size resolution frame
        predicted_frame = self.upscale(self.predicted_patches)
        
        return predicted_frame
