import argparse
import matplotlib.pyplot as plt
import torch
from data import dataset_builder, dataloader_builder
from models.unet import UNet
from flow_matching import sample_path
from config import TrainingConfig
from torch import nn, optim

# Create parse object
parser = argparse.ArgumentParser()

# Add optional arguments
parser.add_argument("--batch-size", default=64, type=int, help="batch size for training")
parser.add_argument("--learning-rate", default=1e-4, type=float, help="learning rate for gradient updates")
parser.add_argument("--epochs", default=10, type=int, help="total number of training epochs")
parser.add_argument("--seed", default=1, type=int, help="seed for training reproducibility")
parser.add_argument("--device", default="cuda", type=str, help="CPU or GPU device")
parser.add_argument("--data-dir", default="data", type=str, help="directory with training dataset")
parser.add_argument("--output-dir", default="outputs/checkpoints", type=str, help="directory to store outputs and checkpoints")

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

# For EMA (Exponential Weighted Average)
ema_model = torch.optim.swa_utils.AveragedModel(model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999), use_buffers=False)

ema_model.update_parameters(model)
# T_max is calls to scheduler step, in this equivalent to optim steps
cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=195000, eta_min=1e-6)
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5000)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine_annealing], milestones=[5000])

optim_steps = 0
warmup_period = 5000

# Checkpointing
def save_checkpoint(model, ema_model, scheduler, optimizer, global_steps):
    torch.save({
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "scheduler": scheduler.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": global_steps,
    
    }, f"{config.output_dir}/research_unet_checkpoint.pt")

# Training loop and checkpointing
for epoch in range(config.epochs):
    total_loss = 0
    total_examples = 0
    # set to train
    model.train()

    for data in train_dataloader:
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

        # Update ema params
        ema_model.update_parameters(model)
        scheduler.step()

        if optim_steps == 1000:
            assert ema_model.n_averaged.item() == 1001

        # Checkpoint every 10000 optim steps
        if optim_steps % 10000 == 0:
            save_checkpoint(model, ema_model, scheduler, optimizer, optim_steps)

        # Calculate loss
        total_loss += x1.shape[0] * loss.item()
        total_examples += x1.size(0)

    avg_loss = total_loss / total_examples

    # Set to evaluation mode
    model.eval() # changes behaviour of model layers

    with torch.no_grad():
        total_loss = 0
        total_examples = 0 
        
        for data in test_dataloader:
            x1, _ = data
            x1 = x1.to(device)

            # Sample noisy images
            x0 = torch.randn_like(x1)
            x0 = x0.to(device)

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

