import argparse
import matplotlib.pyplot as plt
import torch
from data import dataset_builder, dataloader_builder
from models.unet import UNet
from flow_matching import sample_path
from config import TrainingConfig
from torch import nn, optim
from sample import euler_sampler, inverse_transform

# Set up config
config = TrainingConfig(
    batch_size=32,
    learning_rate=10e-4,
    epochs=5000,
    seed=1,
    device="mps",
    data_dir="data",
    output_dir="outputs/checkpoints"
)

num_steps = 100

# Set up Datasets and DataLoader for Train and Test data
train_dataset = dataset_builder(
    data_dir=config.data_dir
)

train_dataloader = dataloader_builder(
    train_dataset,
    batch_size=1
)

# Set device
if torch.cuda.is_available():
    device = "cuda"          
elif torch.backends.mps.is_available():
    device = "mps"           
else:
    device = "cpu"

# Initialize model
model = UNet().to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Define loss function
criterion = nn.MSELoss()

# Train dataset
data = next(iter(train_dataloader))
image, _ = data
image = inverse_transform(image)
image = torch.clamp(image, 0, 1)
img = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
plt.imsave("outputs/samples/target.png", img)

optim_steps = 0

# For EMA (Exponential Weighted Average)
ema_model = torch.optim.swa_utils.AveragedModel(model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999), 
            use_buffers=False)

ema_model.update_parameters(model)

optim_steps = 0
warmup_period = 0

# Training loop and checkpointing
while optim_steps <= 1000:
    total_loss = 0
    total_examples = 0
    # set to train
    model.train()

    x1, _ = data # get clean images of dim (B, C, H, W)
    x1 = x1.to(device)

    # Zero gradients
    optimizer.zero_grad()

    # Sample noisy images from a Gaussian distribution with mean 0 and variance 1. 
    x0 = torch.randn_like(x1)
    x0 = x0.to(device)

    # Get t, xt (intermediate samples), and ut (target velocity)
    t, xt, ut = sample_path(x0, x1)

    # Make predictions
    vt = model(xt, t)

    # Compute loss and gradients
    loss = criterion(vt, ut) # between predicted and target velocities
    loss.backward()

    # Update model parameters
    optimizer.step()
    optim_steps += 1
    if optim_steps > warmup_period:
        ema_model.update_parameters(model)

    # Calculate loss
    total_loss += x1.shape[0] * loss.item()
    total_examples += x1.size(0)

    avg_loss = total_loss / total_examples
    print(f"Epochs {optim_steps + 1}/{config.epochs} | Train: {avg_loss:.2f}")

    # Set to evaluation mode
    model.eval() # changes behaviour of model layers

    if optim_steps % 500 == 0:
        noise, image = euler_sampler(model, config.batch_size, num_steps, config.device)

torch.save(model.state_dict(), f"{config.output_dir}/unet_checkpoint_overfit.pt")