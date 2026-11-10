import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log, sqrt
import torch.nn.init as init

def sinusoidal_embeddings(t, dim, max_period=10000):
    """
    Convert a timestep of dim [B] to a sinusoidal embedding of dimension [B, D]  
    """
    # Frequencies
    frequencies = torch.exp(-log(max_period) * torch.arange(start=0, end=(dim // 2)) / (dim // 2)).to(t.device)

    # Calculate sinusoidal embeddings
    # converts timesteps to be of dim [B//2, 1]
    cos_embed = torch.cos(frequencies[None] * t[:, None]) 
    sin_embed = torch.sin(frequencies[None] * t[:, None])

    # Concatenate cosine and sine embeddings
    t_embed = torch.cat([cos_embed, sin_embed], dim=-1)
    return t_embed

class Downsample(nn.Module):
    def __init__(self, num_channels, stride=2):
        super().__init__()
        self.num_channels = num_channels
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        x = self.avg_pool(x)
        return x

class Upsample(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        # Use nearest neighbors to double spatial resolution by factor of x2
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        return x

class ResBlock(nn.Module):
    def __init__(self, num_channels, embed_channels, out_channels, dropout):
        super().__init__()
        # Define terms
        self.out_channels = out_channels
        self.embed_channels = embed_channels
        self.num_channels = num_channels

        # Define main layers 
        self.conv2d_1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(num_channels, out_channels, kernel_size=3, padding=1)
        self.SiLU = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)

        # Normalization
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=num_channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=num_channels)
        self.norm3 = nn.GroupNorm(num_groups=8, num_channels=num_channels)

        # Timestep Embedding projection
        self.timestep_proj = nn.Linear(embed_channels, num_channels * 2)

        # Skip connection
        if num_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(num_channels, out_channels, kernel_size=1)

    def forward(self, x, t):
        h = self.conv2d_1(self.SiLU(self.norm1(x)))

        # Timestep is of shape [B, 2*C], project to [B, 2C, 1, 1]
        e_t = self.timestep_proj(t).view(h.shape[0], 2 * self.num_channels, 1, 1)

        # Gamma and beta are of shapes [B, C]
        gamma, beta = torch.chunk(e_t, chunks=2, dim=1)

        # Shift and scale
        h = self.norm3(h) * (1 + gamma) + beta

        h = self.skip(x) + self.conv2d_2(self.dropout(self.SiLU(self.norm2(h))))

        return h

class SpatialAttentionBlock(nn.Module):
    def __init__(self, num_channels, num_heads=1):
        super().__init__()

        # Define hyperparameters
        self.num_channels = num_channels
        self.num_heads = num_heads

        # Initialize layers
        self.norm = nn.GroupNorm(num_groups=32, num_channels=self.num_channels)
        self.softmax = nn.Softmax(dim=-1)
        
        # 1D Convolution with kernel size of 1, apply equivalent linear transformation to every spatial token.
        self.qkv_proj = nn.Conv1d(num_channels, 3 * num_channels, kernel_size=1, stride=1, padding=0) # QKV projection is a 1x1 convolution C -> 3C
        self.out_proj = nn.Conv1d(num_channels, num_channels, kernel_size=1, stride=1, padding=0) # Output projection is 1x1 convolution C --> C
        
        ## Output projection should be zero initialized
        init.zeros_(self.out_proj.weight)
        init.zeros_(self.out_proj.bias)

    def forward(self, x):
        # Flatten input feature map from [B, C, H, W]
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        x = x.reshape((B, self.num_channels, H * W))

        # Apply GroupNorm and Linear Q, K, V projections
        qkv = self.qkv_proj(self.norm(x))

        # Chunk into Q, K, V
        Q, K, V = torch.chunk(qkv, chunks=3, dim=1)

        # Split accross attention head
        # Reshape [B, C, N] to [B, h, d_k, N] and permute to [B, h, N, d_k]
        Q = Q.view((Q.shape[0], self.num_heads, self.num_channels // self.num_heads, Q.shape[2])).permute((0, 1, 3, 2))
        K = K.view((K.shape[0], self.num_heads, self.num_channels // self.num_heads, K.shape[2])).permute((0, 1, 3, 2))
        V = V.view((V.shape[0], self.num_heads, self.num_channels // self.num_heads, V.shape[2])).permute((0, 1, 3, 2))

        ## Attention is bidirectional (no causal masking)
        # Scaled dot-product self-attention
        d_k = self.num_channels // self.num_heads

        attn_weights = self.softmax((Q @ K.transpose(-1, -2)) / sqrt(d_k))
        z = attn_weights @ V # [B, h, N, d_k]

        ## Output projection + residual (skip connections)
        # Concatenate across head dimensions: Rearrange from [B, h, N, d_k] to [B, h, d_k, N] and restore spatial dim to [B, C, N]
        z = torch.permute(z, (0, 1, 3, 2)).reshape((B, self.num_channels, H * W))

        y = self.out_proj(z) + x
        return y.reshape(B, self.num_channels, H, W)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_channels=128, num_encoder=3, num_decoder=3, out_channels=3, num_res_encoder=4, num_res_decoder=5, sin_embed_dim=128, timestep_dim=512, level_channels=256, attention_dims=16, attention_heads=1, dropout=0.3, channel_mult=[2, 2, 2], beta=0.999):
        super().__init__()

        self.in_channels = in_channels
        self.num_channels = num_channels
        self.out_channels = out_channels
        self.num_res_encoder = num_res_encoder
        self.num_res_decoder = num_res_decoder
        self.sin_embed_dim = sin_embed_dim
        self.timestep_dim = timestep_dim
        self.level_channels = level_channels
        self.num_encoder = num_encoder
        self.num_decoder = num_decoder

        # Input Convolution
        self.conv2d_in = nn.Conv2d(self.in_channels, self.num_channels * channel_mult[0], kernel_size=3, padding=1)

        # Timestep Embedding
        self.timestep_embed = nn.Sequential(
            nn.Linear(self.sin_embed_dim, self.timestep_dim),
            nn.SiLU(), 
            nn.Linear(self.timestep_dim, self.timestep_dim))

        # Encoder
        self.Encoders = nn.ModuleList()

        for i in range(self.num_encoder):    
            Encoder = nn.ModuleList([])
            current_channels = self.num_channels * channel_mult[i]
            for _ in range(self.num_res_encoder):
                Encoder.append(ResBlock(current_channels, self.timestep_dim, self.num_channels * channel_mult[i], dropout=0))
                if i % 2 != 0:
                    Encoder.append(SpatialAttentionBlock(current_channels, attention_heads))

            if i != self.num_encoder - 1: 
                Encoder.append(Downsample(self.num_channels * channel_mult[i]))

            self.Encoders.append(Encoder)

        # Bottleneck
        self.bottleneck = nn.ModuleList([])
        self.bottleneck.extend([ResBlock(current_channels, self.timestep_dim, current_channels, dropout=0),
                                SpatialAttentionBlock(current_channels),
                                ResBlock(current_channels, self.timestep_dim, current_channels, dropout=0)])

        # Decoder
        self.Decoders = nn.ModuleList()

        decoder_channel_mult = channel_mult[::-1] # reverses the list.

        for i in range(self.num_decoder):
            Decoder = nn.ModuleList([])
            current_channels = self.num_channels * decoder_channel_mult[i]
            for _ in range(self.num_res_decoder):
                Decoder.append(ResBlock(current_channels + channel_mult[i] * num_channels, self.timestep_dim, current_channels, dropout=0))
                if i % 2 != 0:
                    Decoder.append(SpatialAttentionBlock(current_channels, attention_heads))

            if i != self.num_decoder - 1:
                Decoder.append(Upsample(current_channels))
            self.Decoders.append(Decoder)
            
        # Output convolutions
        self.conv2d_out = nn.Conv2d(current_channels, self.out_channels, kernel_size=3, padding=1)

        # Group Norm
        self.norm = nn.GroupNorm(num_groups=32, num_channels=current_channels)

        # Activation functions
        self.silu = nn.SiLU()

        # For EMA, 0.999 default
        self.beta = beta

    def forward(self, xt, t):
        # Input 3x3 convolution 
        # [B, 3, 32, 32] -> [B, 256, 32, 32]
        x = self.conv2d_in(xt)

        # Get Timestep Embeddings
        # []
        t_sin_embed = sinusoidal_embeddings(t, self.num_channels)
        t_embed = self.timestep_embed(t_sin_embed)

        # Save skip connections, start with initial value after input projection. 
        h = [x]

        # Encoder
        # 4 * ResBlock(256 -> 256 and downsample to 16x16
        for i, Encoder in enumerate(self.Encoders):
            for _, layer in enumerate(Encoder):
                # Check if layer has attention, if yes, then save after that. 
                if isinstance(layer, ResBlock):
                    x = layer(x, t_embed)
                    if i % 2 == 0:
                        h.append(x)
                elif isinstance(layer, Downsample):
                    x = layer(x)
                    h.append(x)
                elif isinstance(layer, SpatialAttentionBlock):
                    x = layer(x)
                    h.append(x)

        # Pass through the bottleneck
        for _, layer in enumerate(self.bottleneck):
            if isinstance(layer, ResBlock):
                x = layer(x, t_embed)
            elif isinstance(layer, SpatialAttentionBlock):
                x = layer(x)

        # Decoder
        for Decoder in self.Decoders:
            for _, layer in enumerate(Decoder):
                if isinstance(layer, ResBlock):
                    x = torch.cat((x, h.pop()), dim=1)
                    x = layer(x, t_embed) # applied to resblock
                elif isinstance(layer, Upsample):
                    x = layer(x)
                elif isinstance(layer, SpatialAttentionBlock):
                    x = layer(x)
                
        # Output convolution
        x = self.conv2d_out(self.silu(self.norm(x)))

        return x