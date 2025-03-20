import pickle
import sys
import numpy as np
from path import Path
sys.path.insert(0, Path(__file__).parent.parent.absolute())

from diffusion_uncertainty.lsun_churches256 import LSUNChurches256

import json
from typing import List
from path import Path

from tqdm import tqdm
from diffusion_uncertainty.paths import DATA, DATASET_FID, IMAGENET128_TEST, IMAGENET128_TRAIN, IMAGENET128_VAL, IMAGENET256_TEST, IMAGENET256_TRAIN, IMAGENET256_VAL, IMAGENET64_TRAIN, IMAGENET64_VAL, IMAGENET_TEST, IMAGENET_TRAIN, IMAGENET_VAL, LSUN_CHURCHES256_TRAIN, LSUN_CHURCHES256_VAL
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image, ImageFile
import torchvision
import torch
from torch.utils.data import Dataset

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.datasets import CelebA, CIFAR10
from torchvision import transforms
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Compute FID score for real images')
    parser.add_argument('real_imgs_folders', type=Path, nargs='+', help='Folders containing the real images')
    parser.add_argument('--dataset-name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for computing FID score')
    parser.add_argument('--on-gpu', action='store_true', help='Use GPU for computing FID score')
    parser.add_argument('--img-size', type=int, default=256, help='Size of the images')
    # parser.add_argument('--resize', action='store_true', help='Resize images to img_size')
    parser.add_argument('--img-extension', type=str, default='png', help='Extension of the images in the real images folders')
    
    args = parser.parse_args()
    return args


@torch.no_grad()
def compute_real_fid_dataset_(*datasets: Dataset, dataset_name: str, batch_size: int, on_gpu : bool = False):
    """
    Computes the Frechet Inception Distance (FID) between the real images in the given datasets and the
    Inception-v3 features of the generated images. Saves the statistics of the real images in a .pth file
    and the structure of the statistics in a .json file.

    Args:
        *datasets: Variable length argument list of datasets containing the real images.
        batch_size (int): Batch size to use for computing the FID.
        on_gpu (bool): Whether to use the GPU for computing the FID.

    Returns:
        None
    """    
    first_image = datasets[0][0]
    assert first_image.dtype == torch.uint8, f'Expected dtype uint8, got {first_image.dtype}'
    if on_gpu:
        # Use GPU 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        # Use CPU
        device = torch.device('cpu')
    print('Using device:', device)

    dataset_fid_path = DATASET_FID.joinpath(dataset_name)
    
    fid = FrechetInceptionDistance(normalize=False).to(device)

    for dataset in datasets: print('Using device:', device)
    
    fid = FrechetInceptionDistance(normalize=False).to(device)

    for dataset in datasets:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        for batch in tqdm(dataloader):
            if on_gpu:
                batch = batch.cuda()

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        for batch in tqdm(dataloader):
            if on_gpu:
                batch = batch.cuda()
            fid.update(batch, real=True)
    save_fid_real_stats_(dataset_fid_path, fid)

@torch.no_grad()
def compute_real_fid_(*real_imgs_folders: Path, dataset_name: str, img_extension: str, batch_size=64, on_gpu=False, img_size: int = 256, resize: bool = False):
    """
    Computes the Frechet Inception Distance (FID) between the real images in the given folders and the
    Inception-v3 features of the generated images. Saves the statistics of the real images in a .pth file
    and the structure of the statistics in a .json file.

    Args:
        *real_imgs_folders: Variable length argument list of folders containing the real images.
        dataset_name (str): Name of the dataset.
        img_extension (str): Extension of the images in the real images folders.
        batch_size (int): Batch size to use for computing the FID.
        on_gpu (bool): Whether to use the GPU for computing the FID.

    Returns:
        None
    """    
    fid_dataset_path = DATASET_FID.joinpath(dataset_name)
    if not fid_dataset_path.exists():
        fid_dataset_path.mkdir()
    if on_gpu:
        # Use GPU 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        # Use CPU
        device = torch.device('cpu')
    print('Using device:', device)
    
    fid = FrechetInceptionDistance(normalize=True).to(device)  #If argument ``normalize`` is ``True`` images are expected to be dtype ``float`` and have values in the ``[0,1]`` range, else if ``normalize`` is set to ``False`` images are expected to have dtype ``uint8`` and take values in the ``[0, 255]``
    to_tensor = torchvision.transforms.ToTensor()

    batch: torch.Tensor = torch.zeros((batch_size, 3, img_size, img_size), device=device)
    i_batch = 0
    first = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    for real_imgs_folder in real_imgs_folders:
        print('Computing real mu and sigma for', real_imgs_folder)
        if dataset_name == 'imagenet64':
            for chunk in real_imgs_folder.files('*data*'):
                with open(chunk, 'rb') as f:
                    data_dict = pickle.load(f)
                x = data_dict['data']
                del data_dict
                x = x / np.float32(255)
                x = torch.from_numpy(x)
                x = x.view(-1, 64, 64, 3).transpose(1, 3)

                print(f'x.shape: {x.shape}')
                print(f'x.min(): {x.amin()}')
                print(f'x.max(): {x.amax()}')
                for i in range(x.shape[0] // batch_size):
                    x_batch = x[i * batch_size: (i + 1) * batch_size]
                    if on_gpu:
                        x_batch = x_batch.cuda()
                    fid.update(x_batch, real=True)
        else:
            for img_path in tqdm(real_imgs_folder.walkfiles(f'*.{img_extension}')):
                if len(batch) == i_batch:
                    # batches are in [0, 1]
                    fid.update(batch, real=True)
                    i_batch = 0

                img = Image.open(img_path).convert('RGB')
                img = to_tensor(img)
                batch[i_batch] = img

                if first:
                    print(f'img.shape: {img.shape}')
                    print(f'img.min(): {img.amin()}')
                    print(f'img.max(): {img.amax()}')
                    first = False
                
                i_batch += 1

            if i_batch > 0:

                batch = batch[:i_batch]
                fid.update(batch, real=True)
                i_batch = 0
        print('Real mu and sigma computed for', real_imgs_folder)

    save_fid_real_stats_(fid_dataset_path, fid)

def save_fid_real_stats_(fid_dataset_path, fid):
    """
    Save the FID (Fréchet Inception Distance) real features statistics to disk.

    Args:
        fid_dataset_path (str): The path to the FID dataset directory.
        fid (FID): The FID object containing the real features statistics.

    Returns:
        None
    """
    real_features_stats = {'sum': fid.real_features_sum.cpu(), 'cov': fid.real_features_cov_sum.cpu(), 'num_examples': fid.real_features_num_samples} 
    torch.save(real_features_stats, fid_dataset_path / 'real_samples_stats.pth')
    with open(fid_dataset_path / 'stats_structure.json', 'w') as f:
        f.write('// This file is generated by compute_dataset_fid.py\n')
        f.write('// The .pth file contains the statistics of the real images\n') 
        json.dump({k: t.shape if isinstance(t, torch.Tensor) else type(t) for k, t in real_features_stats.items()}, f)
    print('FID state dict saved to', fid_dataset_path / 'fid_state_dict.pth')

if __name__ == '__main__':
    args = get_args()
    define_paths = True
    compute_real_fid_(*args.real_imgs_folders, dataset_name=args.dataset_name, batch_size=args.batch_size, img_extension=args.img_extension, on_gpu=args.on_gpu, img_size=args.img_size, resize=False)

