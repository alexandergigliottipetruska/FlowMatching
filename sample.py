# Implement ODE sampler (Euler Integration) for Inference
import torch
from torchvision.transforms import v2
from torchvision import transforms
from FlowMatching.models.unet import UNet
from torchvision.utils import make_grid, save_image

# Checkpoint directory for loading checkpoints
output_dir = ""

# Set device
device = "cuda"

# Instanstiate model
model = UNet().to(device)
model.load_state_dict(output_dir)
model.eval()

# Implement Euler loop → At each step, pass the current image and timestep to your U-Net. 
# Use its predicted velocity to update the image tensor.

def inverse_transform(tensor):
    # Undo normalization to convert tensor to image. 
    inverse_transformation = transforms.Compose([v2.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.225 ]),
                                   v2.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                     std = [ 1., 1., 1. ]),])

    return inverse_transformation(tensor)

def euler_sampler(model, batch_size, num_steps, device):

    # Sample x0 from Gaussian noise with mean 0 and std 1
    x0 = torch.randn(batch_size, 3, 32, 32)

    # For N equally spaced timesteps
    delta_t = 1 / num_steps 

    xk = x0
    tk = delta_t

    for k in range(num_steps):
        # Predict velocity of intermediate state
        vk = model(xk, tk)

        # Update image
        xk = xk + delta_t * vk

        # Update time step
        tk = k * delta_t

    x1 = xk

    # Undo normalization
    image_tensor = inverse_transform(x1)

    # Clamp values between 0 and 1
    image_tensor = torch.clamp(image_tensor, 0, 1)

    # Convert to torchvision grid
    image_grid = make_grid(image_tensor, nrow=4)

    # Save image
    save_image(image_grid, "cifar10_inference.png")

    return image_tensor