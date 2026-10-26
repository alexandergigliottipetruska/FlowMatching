import torch
from data import dataset_builder, dataloader_builder
from models.unet import UNet
from flow_matching import sample_path
from torch import nn, optim
from config import TrainingConfig

# Set up config
config = TrainingConfig(
    batch_size=32,
    learning_rate=0.01,
    epochs=5,
    seed=1,
    device="cpu",
    data_dir="data",
    output_dir="outputs"
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

def test_data_flow_targets():
    data_iter = iter(train_dataloader)
    x1, _ = next(data_iter)
    x0 = torch.randn_like(x1)
    t1  = torch.full((x1.shape[0],), fill_value=0.5)
    t2 = torch.ones(x1.shape[0])

    t_1, xt, ut = sample_path(x0, x1, t1)
    t_2, xt2, ut2 = sample_path(x0, x1, t2)

    assert x1.shape == x0.shape
    assert xt.shape == x0.shape
    assert torch.equal(xt2, x1)
    assert ((x1 >= -1) & (x1 <= 1)).all().item()

def test_optimization_step():
    data_iter = iter(train_dataloader)
    old_params = []

    # Save params
    for param in model.parameters():
        old_params.append(param.detach().clone())


    # Get x1, x0, and t
    x1, _ = next(data_iter)
    x0 = torch.randn_like(x1)
    t  = torch.full((x1.shape[0],), fill_value=0.5)

    # get t, xt, and target velocity ut
    t, xt, ut = sample_path(x0, x1, t)

    # Get predicted velocity 
    vt = model(xt, t)

    # Zero gradients
    optimizer.zero_grad()

    # Calculate loss
    loss = criterion(vt, ut)
    assert torch.isfinite(loss).all() # verify loss is finite

    # Backpropagate loss
    loss.backward()
    optimizer.step()

    i = 0 # index

    for name, param in model.named_parameters():
        assert torch.isfinite(param.grad).all() # verify grads are finite
        # Check if parameters changes
        assert not torch.equal(old_params[i], param)
        i += 1

def test_checkpoint_persistence():
    torch.save(model.state_dict(), "outputs/checkpoints/model.pt")    

    data_iter = iter(train_dataloader)

    # Get x1, x0, and t
    x1, _ = next(data_iter)
    x0 = torch.randn_like(x1)
    t  = torch.full((x1.shape[0],), fill_value=0.5)

    # get t, xt, and target velocity ut
    t, xt, ut = sample_path(x0, x1, t)

    # Get predicted velocity 
    vt = model(xt, t)

    model2 = UNet()
    model2.load_state_dict(torch.load("outputs/checkpoints/model.pt", weights_only=True))
    model2.eval()

    vt2 = model2(xt, t)

    assert torch.equal(vt, vt2)

def test_batch_overfitting():
    train_dataloader_mini = dataloader_builder(
        train_dataset,
        batch_size=8
    )

    optimizer2 = optim.Adam(model.parameters(), lr=0.0001)

    data = next(iter(train_dataloader_mini))
    x1, _ = data

    # Check if minibatch size is 8 or not
    assert x1.shape[0] == 8

    x0 = torch.randn_like(x1)

    # get t, xt, and target velocity ut
    t, xt, ut = sample_path(x0, x1)

    loss = 0

    for i in range(200):
        # get preds
        vt = model(xt, t)

        # Zero gradients
        optimizer2.zero_grad()

        # loss
        loss = criterion(vt, ut)
        loss.backward()

        # Update weights
        optimizer2.step()

        if i % 10 == 0:
            print(loss)

        if i == 0:
            initial_loss = loss

    assert not torch.equal(initial_loss, loss)

