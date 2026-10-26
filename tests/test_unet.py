import torch
from models.unet import UNet
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