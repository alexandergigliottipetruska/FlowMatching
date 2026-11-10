import torch
from models.unet import UNet, SpatialAttentionBlock, ResBlock, Downsample, Upsample
from torch import optim, nn

model = UNet(in_channels=3)

def test_velocity_shape():
    # Dims xt: [8, 3, 32, 32] and t: [8]
    # Output must be dim [8, 3, 32, 32] with finite values
    
    # Construct targets and source
    x0 =  torch.randn(8, 3, 32, 32)
    x1 = torch.randn(8, 3, 32, 32)

    # Sample timesteps
    t = torch.rand(8)[:, None, None, None] # broadcast to shape (B, 1, 1, 1)

    # Calculate intermediate states 
    xt = (1 - t) * x0 + t * x1

    # Calculate target velocity
    ut = (x1 - x0)

    # Get velocity
    vt = model(xt, t)

    assert vt.shape == ut.shape
    assert torch.isfinite(vt).all()

def test_gradient_flow():
    # Dims xt: [8, 3, 32, 32] and t: [8]
    # Output must be dim [8, 3, 32, 32] with finite values
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Construct targets and source
    x0 =  torch.randn(8, 3, 32, 32)
    x1 = torch.randn(8, 3, 32, 32)

    # Sample timesteps
    t = torch.rand(8)[:, None, None, None] # broadcast to shape (B, 1, 1, 1)

    # Calculate intermediate states 
    xt = (1 - t) * x0 + t * x1

    # Calculate target velocity
    ut = (x1 - x0)

    optimizer.zero_grad()

    # Get velocity
    vt = model(xt, t)

    loss = criterion(vt, ut)
    assert torch.isfinite(loss).all()

    loss.backward()
    optimizer.step()

    for name, param in model.named_parameters():
        print(f"Gradient for {name}: {param.grad}")
        assert torch.isfinite(param.grad).all()


def test_gradient_flow():
    # Dims xt: [8, 3, 32, 32] and t: [8]
    # Output must be dim [8, 3, 32, 32] with finite values
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Construct targets and source
    x0 =  torch.randn(8, 3, 32, 32)
    x1 = torch.randn(8, 3, 32, 32)

    # Sample timesteps
    t = torch.rand(8)[:, None, None, None] # broadcast to shape (B, 1, 1, 1)

    # Calculate intermediate states 
    xt = (1 - t) * x0 + t * x1

    # Calculate target velocity
    ut = (x1 - x0)

    optimizer.zero_grad()

    # Get velocity
    vt = model(xt, t)

    loss = criterion(vt, ut)
    assert torch.isfinite(loss).all()

    loss.backward()
    optimizer.step()

    for name, param in model.named_parameters():
        print(f"Gradient for {name}: {param.grad}")
        assert torch.isfinite(param.grad).all()


def test_attention_placement():
    attn_block_shapes = []

    def hook(module, inputs, output):
        attn_block_shapes.append(output.shape[-2:])

    for module in model.modules():
        if isinstance(module, SpatialAttentionBlock):
            module.register_forward_hook(hook)

    x = torch.randn(1, 3, 32, 32)
    t = torch.tensor([0.5])

    with torch.no_grad():
        model(x, t)

    assert attn_block_shapes.count((16, 16)) == 9
    assert attn_block_shapes.count((8, 8)) == 1
    assert len(attn_block_shapes) == 10

def test_encoder_decoder_placement():
    encoder_shapes = []
    decoder_shapes = []

    def encoder_hook(module, inputs, output):
        encoder_shapes.append(output.shape[-2:])

    def decoder_hook(module, inputs, output):
        decoder_shapes.append(output.shape[-2:])

    for Encoder in model.Encoders:
        for layer in Encoder:
            if isinstance(layer, ResBlock):
                layer.register_forward_hook(encoder_hook)

    for Decoder in model.Decoders:
        for layer in Decoder:
            if isinstance(layer, ResBlock):
                layer.register_forward_hook(decoder_hook)

    x = torch.randn(1, 3, 32, 32)
    t = torch.tensor([0.5])

    with torch.no_grad():
        model(x, t)

    assert encoder_shapes == ([(32, 32)] * 4 + [(16, 16)] * 4 + [(8, 8)] * 4)
    assert decoder_shapes == ([(8, 8)]*5 + [(16, 16)]*5 + [(32, 32)]*5)
