import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from timm.models.layers import PatchEmbed
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
            num_voxels=1024
            ):
        super(NextFrameTransformerTeacher, self).__init__(d_model=embed_dim, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)
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
            embed_dim=embed_dim
        )

        num_patches = (image_size // patch_size) ** 2
        self.rgb_pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )
        self.event_voxel_pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )
        
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
            num_layers=8,
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
            decoder_layer=rgb_decoder_layer
        )

        # Define output projection layer
        ntoken = 3 * (image_height // patch_size) * (image_width // patch_size)  
        self.output_proj = nn.Linear(embed_dim, ntoken)

        # Initialize transformer weights
        self.init_weights()

    def _pair(t):
        return t if isinstance(t, tuple) else (t, t)

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))
    
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
        decoded_tokens = self.rgb_decoder(encoded_tokens)
        predicted_frame = self.output_proj(decoded_tokens)
        return predicted_frame