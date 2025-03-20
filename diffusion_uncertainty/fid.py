from math import ceil
from typing import Literal
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision
from diffusion_uncertainty.paths import DATASET_FID, SCORE_DATASET_FID
from torchvision import transforms
from pytorch_fid.inception import InceptionV3
import numpy as np

from scripts.compute_dataset_fid import calculate_frechet_distance
from diffusion_uncertainty.utils import predict_batch_fid_vector
from typing import List
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusion_uncertainty.paths import DATASET_FID, SCORE_DATASET_FID
from scripts.compute_dataset_fid import calculate_frechet_distance
from diffusion_uncertainty.utils import predict_batch_fid_vector


def get_dims_bayesdiff(dataset_name: str) -> int:
    with open(SCORE_DATASET_FID / dataset_name / 'dims.txt', 'r') as f:
        dims = int(f.read().strip())
    return dims

def load_inception_model_bayesdiff(device: torch.device, dims: int) -> torch.nn.Module:
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx], resize_input=True, normalize_input=True, use_fid_inception=True).to(device)

    return model

def load_inception_model_bayesdiff_with_dims(dataset_name: str, device: torch.device) -> torch.nn.Module:
    dims = get_dims_bayesdiff(dataset_name)
    return load_inception_model_bayesdiff(device, dims)


def load_real_fid_model(dataset_name: str, device: torch.device, normalize=False) -> FrechetInceptionDistance:
    fid_dataset = DATASET_FID / dataset_name

    fid = FrechetInceptionDistance(normalize=normalize, reset_real_features=False).to(device)
    fid.inception = fid.inception.to(device)

    real_stats = torch.load(fid_dataset / 'real_samples_stats.pth')
    fid.real_features_sum = real_stats['sum'].to(device)
    fid.real_features_cov_sum = real_stats['cov'].to(device)
    fid.real_features_num_samples = real_stats['num_examples'].to(device)

    return fid

def compute_fid_score_torchmetrics(gen_images: torch.Tensor, dataset_name: str, device: torch.device, batch_size: int) -> float:
    """
    Compute the FID score for generated images using the TorchMetrics library.

    Args:
        gen_images (torch.Tensor): The generated images.
        dataset_name (str): The name of the dataset.
        device (torch.device): The device to run the computation on.
        batch_size (int): The batch size for computing the FID score.

    Returns:
        float: The FID score.
    """
    fid = load_real_fid_model(dataset_name, device)
    for i in range(ceil(gen_images.shape[0] / batch_size)):
        batch_gen_image = gen_images[i * batch_size: (i + 1) * batch_size]
        batch_gen_image = batch_gen_image.to(device)
        fid.update(batch_gen_image, real=False)

    fid_score = fid.compute().item()
    return fid_score

def compute_fid_score_bayesdiff_from_dataset(gen_images: torch.Tensor, dataset_name: str, device: torch.device, batch_size: int) -> float:
    """
    Compute the FID score for generated images using the BayesDiff method.

    Args:
        gen_images (torch.Tensor): The generated images.
        dataset_name (str): The name of the dataset.
        device (torch.device): The device to run the computation on.
        batch_size (int): The batch size for computing the FID score.

    Returns:
        float: The FID score.
    """
    fid = load_inception_model_bayesdiff(device=device, dims=2048)

    m1 = torch.load(SCORE_DATASET_FID / dataset_name / 'm.pt')
    s1 = torch.load(SCORE_DATASET_FID / dataset_name / 's.pt')

    return compute_fid_score_bayesdiff(gen_images, fid, m1, s1, device, batch_size)


def compute_fid_score_bayesdiff(gen_images: torch.Tensor, inception_module: torch.nn.Module, m1: torch.Tensor, s1: torch.Tensor, device: torch.device, batch_size: int) -> float:
    """
    Compute the FID score for generated images using the BayesDiff approach.

    Args:
        gen_images (torch.Tensor): The generated images.
        inception_module (torch.nn.Module): The Inception model for feature extraction.
        m1 (torch.Tensor): The mean of real features.
        s1 (torch.Tensor): The covariance of real features.
        device (torch.device): The device to run the computation on.
        batch_size (int): The batch size for computing the FID score.

    Returns:
        float: The FID score.
    """
    features = []
    inception_module = inception_module.to(device)
    for i in range(ceil(gen_images.shape[0] / batch_size)):
        batch_gen_image = gen_images[i * batch_size: (i + 1) * batch_size]
        batch_gen_image = batch_gen_image.to(device)
        feature = predict_batch_fid_vector(inception_module, batch_gen_image)
        features.append(feature)
    features = torch.cat(features, dim=0).cpu().numpy()

    m2 = np.mean(features, axis=0)
    s2 = np.cov(features, rowvar=False)

    fid_score = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_score

transforms_map = {
    'lsun-churches': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
        ]),
    'celeba':  transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
        ]),


}

transforms_map_tensor = {k: transforms.Compose([transform for transform in v.transforms if not isinstance(v, transforms.ToTensor)]) for k, v in transforms_map.items()}