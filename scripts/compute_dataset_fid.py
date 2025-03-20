import pickle
import sys
from typing import List
import numpy as np
from path import Path
sys.path.insert(0, Path(__file__).parent.parent.absolute())

from diffusion_uncertainty.lsun_churches256 import LSUNChurches256

"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from diffusion_uncertainty.paths import RESULTS
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

from pytorch_fid.inception import InceptionV3
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


class Imagenet64NPZ(torch.utils.data.Dataset):

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


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        with Image.open(path) as img_pil:
            if self.transforms is not None:
                img = self.transforms(img_pil)
        return img


def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)
    dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr

def get_activations_from_dataset(dataset, model, batch_size, dims=2048, device='cpu', num_workers=1):
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(dataset), dims))

    start_idx = 0
    first = True

    for batch in tqdm(dataloader):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if first:
            print(f'{batch.shape=}')
            first = False
            print(f'{batch.dtype=}')
            print(f'{batch.min()=}')
            print(f'{batch.max()=}')
            print(f'{batch.mean()=}')
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

        assert pred.shape[1] == dims, f'Expected {dims}, got {pred.shape[1]}'

    return pred_arr


def get_activations_npz_format(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of npz files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    # if batch_size > len(files):
    #     print(('Warning: batch size is bigger than the data size. '
    #            'Setting batch size to data size'))
    #     batch_size = len(files)
    
    datasets = []
    for file in files:
        dataset = Imagenet64NPZ(file, image_only=True)
        datasets.append(dataset)
    datasets = torch.utils.data.ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(datasets,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)
    
    return get_activations_from_dataloader(dataloader, model, len(datasets), dims, device)

def predict_fid_batch(batch, model, dims, device):
        if batch.dtype == torch.uint8:
            batch = batch.float() / 255.0
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred: torch.Tensor = pred.squeeze(3).squeeze(2).cpu()

        return pred



def get_activations_from_dataloader(dataloader, model, num_files, dims=2048, device='cpu',):
    pred_arr = np.empty((num_files, dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        pred = predict_fid_batch(batch, model, dims, device).numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        f'Training and test covariances have different dimensions, {sigma1.shape=} vs {sigma2.shape=}'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1, npz_format=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    if npz_format:
        act = get_activations_npz_format(files, model, batch_size, dims, device, num_workers)
    else:
        act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_activation_statistics_dataset(dataset, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations_from_dataset(dataset, model, batch_size, dims, device, num_workers)
    print(f'{act.shape=}')
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    print(f'{mu.shape=}')
    print(f'{sigma.shape=}')
    print(f'{sigma=}')
    return mu, sigma

def compute_statistics_of_path(paths: List[Path], model, batch_size, dims, device,
                               num_workers=1, npz_format=False):
    if npz_format:
        files = sorted([file for folder in paths for file in folder.walkfiles('*.npz')])
        print('num npz files:', len(files))
    else:
        files = sorted([file 
                        for path in paths
                        for ext in IMAGE_EXTENSIONS
                        for file in path.walkfiles(f'*.{ext}') ])
    m, s = calculate_activation_statistics(files, model, batch_size,
                                            dims, device, num_workers=num_workers, npz_format=npz_format)

    return m, s

def instantiate_inception_v3(device, dims=2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    return model

def calculate_fid_given_paths(paths, batch_size, device, dims, dataset_name, npz_format, num_workers=1):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(paths, model, batch_size,
                                        dims, device, num_workers=num_workers, npz_format=npz_format)
    # m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
    #                                     dims, device, num_workers)
    # fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    score_dataset_pytorch_fid = RESULTS / 'score_dataset_pytorch_fid'
    if not score_dataset_pytorch_fid.exists():
        score_dataset_pytorch_fid.mkdir()
    dest_folder = score_dataset_pytorch_fid / dataset_name
    if not dest_folder.exists():
        dest_folder.mkdir()

    torch.save(m1, dest_folder / f'm.pt')
    torch.save(s1, dest_folder / f's.pt')
    with open(dest_folder / f'tensor_shape.txt', 'w') as f:
        f.write(f'm: {m1.shape}\n')
        f.write(f's: {s1.shape}\n')

    with open(dest_folder / f'dims.txt', 'w') as f:
        f.write(f'{dims}')


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size to use')
    parser.add_argument('--num-workers', type=int, default=0,
                        help=('Number of processes to use for data loading. '
                            'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                            'By default, uses pool3 features'))
    parser.add_argument('paths', type=str, nargs='+',
                        help=('Paths to the generated images or '
                            'to .npz statistic files'))
    parser.add_argument('--dataset-name', type=str, required=True, help='Name of dataset')
    parser.add_argument('--npz-format', action='store_true', help='Use npz format')

    IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                        'tif', 'tiff', 'webp'}
    
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths]

    calculate_fid_given_paths(paths, args.batch_size, args.device, args.dims, args.dataset_name, npz_format=args.npz_format, num_workers=args.num_workers)
