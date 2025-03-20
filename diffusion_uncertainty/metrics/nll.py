import torch

from diffusion_uncertainty.init_model import instantiate_model_scheduler
from diffusion_uncertainty.metrics.denoising_diffusion_pytorch import GaussianDiffusion
from diffusion_uncertainty.metrics.iddpm.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from diffusion_uncertainty.utils import load_starting_points
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


@torch.no_grad()
def run_bpd_evaluation(dataset_name: str, start_index: int, num_samples: int, batch_size: int = 64, device='cuda'):
    """
    Runs the evaluation of bits per dimension (BPD) for a given dataset.

    Args:
        dataset_name (str): The name of the dataset. Only 'imagenet64' and 'imagenet128' are supported.
        start_index (int): The starting index of the samples to evaluate.
        num_samples (int): The number of samples to evaluate.
        batch_size (int, optional): The batch size for evaluation. Defaults to 64.

    Returns:
        torch.Tensor: A tensor containing the BPD values for each sample.

    Raises:
        AssertionError: If the dataset_name is not 'imagenet64' or 'imagenet128'.
    """
    
    assert dataset_name in ['imagenet64', 'imagenet128'], 'At the moment, only imagenet64 and imagenet128 are supported. To add support for other datasets, set proper model_mean_type and model_var_type in the GaussianDiffusion constructor.'
    
    # load the model
    model, scheduler = instantiate_model_scheduler(dataset_name)
    print('Loaded model and scheduler for dataset', dataset_name)

    X_T , y = load_starting_points(dataset_name, start_index, num_samples)

    model = model.to(device)


    diffusion = GaussianDiffusion(
                            betas=scheduler.betas,
                            model_mean_type=ModelMeanType.EPSILON,
                            model_var_type=ModelVarType.LEARNED,
                            loss_type=LossType.MSE,
                            rescale_timesteps=False
                            )
    
    starting_point_dataset = TensorDataset(X_T, y)
    loader = DataLoader(starting_point_dataset, batch_size=batch_size, shuffle=False)
    total_bdps = []
    first = True
    print("Calculating BPDs")
    for X_T_batch, y_batch in tqdm(loader):
        X_T_batch = X_T_batch.to(device)
        y_batch = y_batch.to(device)
        bpd_dict = diffusion.calc_bpd_loop(model, X_T_batch, clip_denoised=True, model_kwargs={'y': y_batch})
        total_bpd = bpd_dict['total_bpd']
        if first:
            print(f"Total BPD: {total_bpd}")
            print(f"Total BPD shape: {total_bpd.shape}")
            first = False
        total_bdps.append(total_bpd)
    total_bdps = torch.cat(total_bdps, dim=0)
    return total_bdps
