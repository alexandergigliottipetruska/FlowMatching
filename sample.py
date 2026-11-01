# Implement ODE sampler (Euler Integration) for Inference
import torch
import argparse
from torchvision.transforms import v2
from torchvision import transforms
from models.unet import UNet
from torchvision.utils import make_grid, save_image
from config import InferenceConfig
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def inverse_transform(tensor):
    # Undo normalization to convert tensor to image. 
    inverse_transformation = transforms.Compose([v2.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                   v2.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                     std = [ 1., 1., 1. ]),])

    return inverse_transformation(tensor)

def euler_sampler(model, batch_size, num_steps, device):

    # Sample x0 from Gaussian noise with mean 0 and std 1
    x0 = torch.randn(batch_size, 3, 32, 32).to(device)

    # For N equally spaced timesteps
    delta_t = 1 / num_steps 

    xk = x0
    xk = xk.to(device)
    tk = torch.full(size=(batch_size,), fill_value=0.0).to(device)

    for k in range(num_steps):
        # Predict velocity of intermediate state
        with torch.no_grad():
            vk = model(xk, tk)

        # Update image
        xk = xk + delta_t * vk

        # Update time step
        tk += delta_t

    x1 = xk

    # Undo normalization
    image_tensor = inverse_transform(x1)

    # Clamp values between 0 and 1
    image_tensor = torch.clamp(image_tensor, 0, 1)

    # Convert to torchvision grid
    image_grid = make_grid(image_tensor, nrow=4)

    # Save image
    save_image(image_grid, "outputs/samples/cifar10_inference.png")

    # Save noise
    noise_tensor = torch.clamp(inverse_transform(x0), 0, 1)
    noise_grid = make_grid(noise_tensor, nrow=4)
    save_image(noise_grid, "outputs/samples/noise_inference.png")
    
    return noise_tensor, image_tensor

if __name__ == "__main__":
# Create parse object
    parser = argparse.ArgumentParser()

    # Add optional arguments
    parser.add_argument("--batch-size", default=64, type=int, help="batch size for training")
    parser.add_argument("--num-steps", default=10, type=int, help="number of steps N for Euler Integration")
    parser.add_argument("--device", default="mps", type=str, help="CPU or GPU device")
    parser.add_argument("--data-dir", default="data", type=str, help="directory with CIFAR10 dataset")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints", type=str, help="directory that stores checkpoints")

    # Access arguments passed via CLI
    args = parser.parse_args()

    # Set up config
    config = InferenceConfig(
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        device=args.device,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir
    )

    # Checkpoint directory for loading checkpoints
    checkpoint_dir = config.checkpoint_dir

    # Set device
    device = config.device

    # Instanstiate model
    model = UNet()
    model.load_state_dict(torch.load(f"{config.checkpoint_dir}/unet_checkpoint.pt", weights_only=True, map_location="cpu"))
    model.to(device)
    model.eval()

    noise, image = euler_sampler(model, config.batch_size, config.num_steps, config.device)