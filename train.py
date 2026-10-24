import argparse
import matplotlib.pyplot as plt
import torch
from FlowMatching.data import dataset_builder, dataloader_builder
from FlowMatching.models.unet import UNet
from FlowMatching.flow_matching import sample_path
from FlowMatching.config import TrainingConfig
from torch import nn, optim

# Create parse object
parser = argparse.ArgumentParser()

# Add optional arguments
parser.add_argument("--batch-size", default=64, type=int, help="batch size for training")
parser.add_argument("--learning-rate", default=0.001, type=float, help="learning rate for gradient updates")
parser.add_argument("--epochs", default=10, type=int, help="total number of training epochs")
parser.add_argument("--seed", default=1, type=int, help="seed for training reproducibility")
parser.add_argument("--device", default="cuda", type=str, help="CPU or GPU device")
parser.add_argument("--data_dir", default="data", type=str, help="directory with training dataset")
parser.add_argument("--output_dir", default="output", type=str, help="directory to store outputs and checkpoints")

# Access arguments passed via CLI
args = parser.parse_args()

# Set up config
config = TrainingConfig(
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    epochs=args.epochs,
    seed=args.seed,
    device=args.device,
    data_dir=args.data_dir,
    output_dir=args.output_dir
)

# Set up Datasets and DataLoader for Train and Test data
train_dataset = dataset_builder(
    data_dir=config.data_dir
)

test_dataset = dataset_builder(
    data_dir=config.data_dir,
    train=False
)

train_dataloader = dataloader_builder(
    train_dataset,
    batch_size=config.batch_size
)

test_dataloader = dataloader_builder(
    test_dataset,
    batch_size=config.batch_size
)

model = UNet()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Define loss function
criterion = nn.MSELoss()

# Training loop and checkpointing
for epoch in range(config.epochs):
    total_loss = 0
    total_examples = 0
    # set to train
    model.train()

    for data in train_dataloader:
        x1, _ = data # get clean images of dim (B, C, H, W)

        # Zero gradients
        optimizer.zero_grad()

        # Sample noisy images from a Gaussian distribution with mean 0 and variance 1. 
        x0 = torch.randn_like(x1)

        # Get t, xt (intermediate samples), and ut (target velocity)
        t, xt, ut = sample_path(x0, x1)

        # Make predictions
        vt = model(xt, t)

        # Compute loss and gradients
        loss = criterion(vt, ut) # between predicted and target velocities
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Calculate loss
        total_loss += x1.shape[0] * loss.item()
        total_examples += x1.size(0)

    avg_loss = total_loss / total_examples

    # Set to evaluation mode
    model.eval() # changes behaviour of model layers

    with torch.no_grad():
        total_loss = 0
        total_examples = 0 
        
        for data in test_dataloader():
            x1, _ = data

            # Sample noisy images
            xo = torch.randn_like(x1)

            # Get t, xt (intermediate samples), and ut (target velocity)
            t, xt, ut = sample_path(x0, x1)

            # Make predictions
            vt = model(xt, t)

            # Compute loss
            val_loss = criterion(vt, ut)
            total_loss += x1.shape[0] * val_loss.item()
            total_examples += x1.size(0)

    eval_loss = total_loss / total_examples

    print(f"Epochs {epoch + 1}/{config.epochs} | Train: {avg_loss:.2f} | Eval: {eval_loss:.2f}")

torch.save(model.state_dict(), config.output_dir)