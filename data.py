from torchvision import datasets
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

def dataset_builder(data_dir, train=True):
    # note that normalize shifts distribution from [0, 1] -> [-1, 1] after scale=True
    return datasets.CIFAR10(
        root=data_dir,
        train=train,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    )

def dataloader_builder(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )