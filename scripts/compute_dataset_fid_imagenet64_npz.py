from path import Path
import sys
import torch

from diffusion_uncertainty.paths import DATASET_FID
sys.path.append(Path(__file__).absolute().parent.parent)

from torch.utils.data import DataLoader, Dataset
import numpy as np

from scripts import compute_dataset_fid
from torchvision import transforms
from torch import nn

class Imagenet64NPZ(Dataset):

    def __init__(self, path_npz, image_only=False, transform=None):
        self.path_npz = path_npz
        self.image_only = image_only

        with np.load(path_npz) as f:
            self.data = f['data']
            self.labels = f['labels']

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = torch.from_numpy(image)
        image = image.reshape(3, 64, 64)
        if self.transform is not None:
            image = self.transform(image)
        if self.image_only:
            return image
        else:
            return image, self.labels[idx]

def main():
    path_val_npz = Path('data/imagenet64/')

    # Add padding to gen_images
    padding = nn.ZeroPad2d(((224-64)//2, (224-64)//2, (224-64)//2, (224-64)//2))


    datasets = [Imagenet64NPZ(p, image_only=True, transform=padding) for p in path_val_npz.walkfiles('*.npz')]
    for dataset in datasets:
        print(f'{dataset.path_npz=}')
        print(f'{len(dataset)=}')


    dest_folder = DATASET_FID / 'imagenet64'
    if not dest_folder.exists(): dest_folder.mkdir()

    compute_dataset_fid.compute_real_fid_dataset_(*datasets, dataset_name='imagenet64', on_gpu=True, batch_size=128)


if __name__ == '__main__':
    main()    