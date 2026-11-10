from cleanfid import fid
from sample import inverse_transform, euler_sampler
from models.unet import UNet
import torch
from torchvision.utils import make_grid, save_image


generate = True

if generate:
    # Checkpoint directory for loading checkpoints
    checkpoint_dir = "outputs/checkpoints"

    # Set device
    device = "mps"

    # Instanstiate model
    model = UNet()
    model.load_state_dict(torch.load(f"{checkpoint_dir}/unet_checkpoint100.pt", weights_only=True, map_location="cpu"))
    model.to(device)
    model.eval()

    total_images = 0
    batch_size = 64
    num_batches = 500 // batch_size

    for _ in range(0, num_batches):
        noise, image = euler_sampler(model, batch_size, 100, device)
        for j in range(batch_size):
            save_image(image[j, ...], f"outputs/eval/cifar10_500/image{total_images+1}.png")
            total_images += 1

    remaining_images = 500 - batch_size * num_batches
    noise, image = euler_sampler(model, remaining_images, 100, device)
    for i in range(remaining_images):
        save_image(image[i, ...], f"outputs/eval/cifar10_500/image{total_images+1}.png")
        total_images += 1


score = fid.compute_fid(
    "outputs/eval/cifar10_500",
    dataset_name="cifar10",
    dataset_res=32,
    dataset_split="train",
    mode="clean",
    device="cpu",
    num_workers=0)

print("FID: ", score)

