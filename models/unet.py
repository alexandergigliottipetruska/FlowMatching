import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_embeddings(t, dim, max_period=10000):
    """
    Convert a timestep of dim [B] to a sinusoidal embedding of dimension [B, D]  
    """
    # Frequencies
    frequencies = torch.exp(-torch.log(max_period) * torch.arange(start=0, end=(dim // 2)) / (dim // 2))

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
        self.conv2d = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        x = self.conv2d(x)
        return x

class Upsample(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.conv2d = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1) # no stride as that is used for downsampling

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        # Use nearest neighbors to double spatial resolution by factor of x2
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv2d(x)

        return x

class ResBlock(nn.Module):
    def __init__(self, num_channels, embed_channels, out_channels):
        super().__init__()

        # Define main layers 
        self.conv2d_1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(num_channels, out_channels, kernel_size=3, padding=1)
        self.SiLU = nn.SiLU()

        # Normalization
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=num_channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        # Timestep Embedding projection
        self.timestep_proj = nn.Linear(embed_channels, num_channels)

        # Skip connection
        if num_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(num_channels, out_channels, kernel_size=1)

    def forward(self, x, t):
        h = self.conv2d_1(self.SiLU(self.norm1(x)))

        # Convert time step embedding to (B, C, 1, 1)
        h = h + self.timestep_proj(t).view(h.shape[0], h.shape[1], 1, 1) 

        h = self.skip(x) + self.conv2d_2(self.SiLU(self.norm2(h)))
        
        return h

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_channels=32, out_channels=3, num_res_blocks=2):
        super().__init__()

        self.in_channels = in_channels
        self.num_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        # Input Convolution
        self.conv2d_in = nn.Conv2d(self.in_channels, self.num_channels, kernel_size=3, padding=1)

        # Timestep Embedding
        self.timestep_dim = 4 * self.num_channels
        self.timestep_embed = nn.Sequential(
            nn.Linear(self.num_channels, self.timestep_dim),
            nn.SiLU(), 
            nn.Linear(self.timestep_dim, self.timestep_dim))

        # Encoder
        self.Encoder = nn.ModuleList([])
        old_channels = self.num_channels
        new_channels = self.num_channels

        for i in range(self.num_res_blocks):
            self.Encoder.append(ResBlock(old_channels, self.timestep_dim, new_channels))
            self.Encoder.append(Downsample(new_channels))

            old_channels = new_channels
            new_channels *= 2

        # Bottleneck
        self.bottleneck = ResBlock(old_channels, self.timestep_dim, new_channels)

        # Skip connection dims
        skip_dims = [64, 32]

        # Decoder
        self.Decoder = nn.ModuleList([])
        old_channels = new_channels
        for i in range(self.num_res_blocks):
            new_channels = new_channels // 2
            self.Decoder.append(Upsample(old_channels))
            self.Decoder.append(ResBlock(old_channels + skip_dims[i], self.timestep_dim, new_channels))

            old_channels = new_channels

        # Output convolutions
        self.conv2d_out = nn.Conv2d(old_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, xt, t):
        # Input 3x3 convolution 
        x = self.conv2d_in(xt)

        # Get Timestep Embeddings
        t_sin_embed = sinusoidal_embeddings(t, self.num_channels)
        t_embed = self.timestep_embed(t_sin_embed)

        # Save skip connections
        h = []

        # Encoder
        for i, layer in enumerate(self.Encoder):
            if i % 2 == 0:
                x = layer(x, t_embed)
                h.append(x)
            else:
                x = layer(x)

        # Pass through the bottleneck
        x = self.bottleneck(x, t_embed)

        # Decoder
        for i, layer in enumerate(self.Decoder):
            if i % 2 != 0:
                x = torch.cat((x, h.pop()), dim=1)
                x = layer(x, t_embed)
            else:
                x = layer(x)

        # Output convolution
        x = self.conv2d_out(x)

        return x