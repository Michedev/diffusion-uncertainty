"""
Generates X_T.pth files for different datasets.

This script generates X_T.pth files for a list of datasets. Each dataset is defined by its name, width, height, and number of channels.
The generated files are saved in the X_T folder.

"""
import sys
from path import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
from diffusion_uncertainty.paths import DIFFUSION_STARTING_POINTS
from dataclasses import dataclass


@dataclass
class Dataset:
    name: str
    width: int
    height: int
    num_channels: int
    num_classes: int

@torch.no_grad()
def main():
    num_samples = 60_000
    device = 'cpu'
    extra_samples = 1_000
    seed = 49394

    datasets: list[Dataset] = [
        Dataset('imagenet64', 64, 64, 3, 1000),  # {dataset_name}_{model_name}
        Dataset('imagenet128', 128, 128, 3, 1000),
        Dataset('imagenet128_uvit', 2**(7-3), 2**(7-3), 4, 1000),
        Dataset('imagenet256', 2**(8-3), 2**(8-3), 4, 1000),
        Dataset('imagenet512', 2**(9-3), 2**(9-3), 4, 1000),
        Dataset('cifar10', 32, 32, 3, 10),
    ]

    for dataset_metadata in datasets:
        print('Generating for', dataset_metadata.name)
        dest_folder = DIFFUSION_STARTING_POINTS / dataset_metadata.name
        if not dest_folder.exists():
            dest_folder.mkdir()
        generator = torch.Generator(device='cpu').manual_seed(seed)
        gen_data = torch.randn(num_samples + extra_samples, dataset_metadata.num_channels, dataset_metadata.height, dataset_metadata.width, device=device, generator=generator)

        y = torch.randint(0, dataset_metadata.num_classes, (num_samples + extra_samples,), device=device, generator=generator)

        print("Stats of gen_data:")  
        print("\tMean:", gen_data.mean().item()) 
        print("\tStandard Deviation:", gen_data.std().item())
        print("\tMin:", gen_data.min().item())
        print("\tMax:", gen_data.max().item())

        
        torch.save(gen_data, dest_folder / 'X_T.pth')
        torch.save(y, dest_folder / 'y.pth')
        print('Using seed:', seed)
        print(f"Saved {dataset_metadata.name} to {dest_folder / 'X_T.pth'}")
        print(f"Saved {dataset_metadata.name} to {dest_folder / 'y.pth'}")

        seed += 1
    print("Done!")


if __name__ == '__main__':
    main()