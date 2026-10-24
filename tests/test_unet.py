import torch
from unet import UNet

model = UNet(in_channels=3)

def test_velocity_shape():
    xt = (8, 3, 32, 32)
    t = torch.rand(0)

    velocity = model(xt, t)

    assert velocity.shape == xt.shape