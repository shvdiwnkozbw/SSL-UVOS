import torch
import einops
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .attention import Block
from .vision_transformer import vit_small, vit_base

class AttEncoder(nn.Module):
    """Spatio-temporal attention encoder for object discovery."""
    def __init__(self, resolution, 
                       num_t=1,
                       path_drop=0.1,
                       attn_drop_t=0.4,
                       num_frames=3,
                       dino_path=None
                ):
        """
        Args:
            resolution: Tuple of integers specifying width and height of input image
            num_frames: Frame length used in training
            num_t: Number of spatio-temporal transformer layers
        """
        super(AttEncoder, self).__init__()

        self.resolution = resolution
        self.T = num_frames        
        self.encoder_dims = 384
        self.hid_dims = 128
        
        self.encoder = vit_small(8)
        if dino_path is not None:
            self.encoder.load_state_dict(torch.load(dino_path), strict=False)
        self.down_time = 8
        self.end_size = (resolution[0] // self.down_time, resolution[1] // self.down_time)
        self.num_t = num_t
        self.temporal_transformer = nn.ModuleList([Block(
                                                        dim=self.encoder_dims, 
                                                        num_heads=8,
                                                        n_token=num_frames, 
                                                        drop=path_drop,
                                                        attn_drop=attn_drop_t)
                                                        for j in range(self.num_t)])

    def forward(self, image, training=False):
        ## input: 'image' has shape B, T, C, H, W  
        ## output: 'attn_dino' has shape BT, HW, HW, 'attn_temporal' has shape B, THW, THW
        
        # ViT encoder
        bs = image.shape[0]
        image_t = einops.rearrange(image, 'b t c h w -> (b t) c h w')

        x, cls_token, attn, k = self.encoder(image_t)  # DINO backbone
        x = einops.rearrange(x, '(b t) c h w -> b (t h w) c', t=self.T) ##spatial_flatten of output feature
        k = einops.rearrange(k, '(b t) c h w -> b t (h w) c', t=self.T) ##spatial_flatten of key feature
        cls_token = einops.rearrange(cls_token, '(b t) c -> b t c', t=self.T) ##per-frame cls token
        k = torch.cat([cls_token.unsqueeze(2), k], dim=2)
        k = einops.rearrange(k, 'b t hw c -> b (t hw) c', t=self.T)
        
        ##### spatio-temporal attention calculation
        if self.num_t > 0:
            for block in self.temporal_transformer:
                k, attn_temporal = block(k)
        
        x = einops.rearrange(x, 'b (t hw) c -> b t hw c', t=self.T) ##spatial-temporal_map
        k = einops.rearrange(k, 'b (t hw) c -> b t hw c', t=self.T)[:, :, 1:]

        attn_dino = attn.mean(dim=1) # average multi-head attention
        attn_dino = attn_dino[:, 1:, 1:] # exclude cls token
        attn_temporal = attn_temporal.mean(dim=1) # average multi-head attention
        attn_temporal = einops.rearrange(attn_temporal, 'b (t hw) (p n) -> b t hw p n', t=self.T, p=self.T)
        attn_temporal = attn_temporal[:, :, 1:, :, 1:] # exclude cls token of each timestamp
        attn_temporal = einops.rearrange(attn_temporal, 'b t hw p n -> b (t hw) (p n)') # per-clip spatio-temporal attention

        if not training:
            return attn_dino, attn_temporal, attn_dino, self.end_size[0]
        else:
            return attn_dino, x, attn_temporal, k, attn_dino
    