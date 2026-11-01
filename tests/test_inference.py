import torch
from data import dataset_builder, dataloader_builder
from models.unet import UNet
from flow_matching import sample_path
from sample import euler_sampler
from torch import nn, optim
from config import TrainingConfig

# Set device
device = "mps"
model = UNet()
model.to(device)
model.eval()

def test_dataset_normalization():
    # images must be between [-1, 1] and inverse transform restores [0, 1]
    pass

def test_constant_velocity():
    batch_size=64
    num_steps=1
    
    # Sample x0 from Gaussian noise with mean 0 and std 1
    x0 = torch.randn(batch_size, 3, 32, 32).to(device)

    # For N equally spaced timesteps
    delta_t = 1 / num_steps 

    xk = x0
    xk = xk.to(device)
    tk = torch.full(size=(batch_size,), fill_value=delta_t).to(device)

    for k in range(num_steps):
        # Predict velocity of intermediate state
        vk = 0.25 # constant velocity

        # Update image
        xk = xk + delta_t * vk

        # Update time step
        tk += delta_t

    x1 = xk

    assert torch.allclose(x0 + 0.25, x1, atol=1e-6, rtol=1e-5)

    num_steps=10
    
    # Sample x0 from Gaussian noise with mean 0 and std 1
    x0 = torch.randn(batch_size, 3, 32, 32).to(device)

    # For N equally spaced timesteps
    delta_t = 1 / num_steps 

    xk = x0
    xk = xk.to(device)
    tk = torch.full(size=(batch_size,), fill_value=delta_t).to(device)

    for k in range(num_steps):
        # Predict velocity of intermediate state
        vk = 0.25 # constant velocity

        # Update image
        xk = xk + delta_t * vk

        # Update time step
        tk += delta_t

    x1 = xk

    assert torch.allclose(x0 + 0.25, x1, atol=1e-6, rtol=1e-5)

def test_time_dependent_velocity():
    #euler_sampler()
    # Sample x0 from Gaussian noise with mean 0 and std 1
    batch_size = 64
    num_steps = 10

    x0 = torch.randn(batch_size, 3, 32, 32).to(device)

    # For N equally spaced timesteps
    delta_t = 1 / num_steps 

    xk = x0
    xk = xk.to(device)
    tk = torch.full(size=(batch_size,), fill_value=delta_t).to(device)

    for k in range(num_steps):
        # Predict velocity of intermediate state
        with torch.no_grad():
            t_k = k * delta_t
            vk =  2 * t_k

        # Update image
        xk = xk + delta_t * vk

        # Update time step
        tk += delta_t

    x1 = xk
    assert torch.allclose(x0 + 0.9, x1, atol=1e-6, rtol=1e-5)