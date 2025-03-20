"""
functions for managing experiments
"""

from path import Path
import torch
import yaml
from diffusion_uncertainty.paths import RESULTS


def search_uncertainty_run_by(dataset_name: str, start_index: int, num_samples: int, scheduler_type: str | None = None, generation_steps: int | None = None) -> list[Path]:
    runs = []
    for run in (RESULTS / 'score-uncertainty').dirs():
        if not run.joinpath('args.yaml').exists():
            continue
        with open(run / 'args.yaml', 'r') as f:
            config = yaml.safe_load(f)
        if not all(key in config for key in ['dataset', 'start_index', 'num_samples']):
            continue 
        if config['dataset'] != dataset_name:
            continue
        if config['start_index'] != start_index:
            continue
        if config['num_samples'] < num_samples:
            continue
        if scheduler_type is not None:
            if 'scheduler_type' not in config:
                continue
            if config['scheduler_type'] != scheduler_type:
                continue
        if generation_steps is not None:
            if 'generation_steps' not in config:
                continue
            if config['generation_steps'] != generation_steps:
                continue
        runs.append(run)
    return runs

def load_uncertainty_run(run: Path, num_samples: int | None = None) -> torch.Tensor:
    """
    Load uncertainty tensors from a run. The uncertainty tensors are usually splitted into multiple files, this function concatenates them.

    Args:
        run (Path): The path to the directory containing the uncertainty files.
        num_samples (int, optional): The number of samples to load. Defaults to None.

    Returns:
        torch.Tensor: The concatenated uncertainty tensors.
    """
    actual_num_rows = 0
    uncertainties = []
    
    # Get a sorted list of uncertainty files in the run directory
    uncertainty_files = sorted(list(run.files('uncertainty*.pth')))
    
    for pth_file in uncertainty_files:
        # Load the uncertainty tensor from each file
        uncertainty = torch.load(pth_file)
        
        actual_num_rows += uncertainty.shape[0]
        
        uncertainties.append(uncertainty)
        
        # If num_samples is specified and the actual number of rows loaded exceeds or equals num_samples, break the loop
        if num_samples is not None and actual_num_rows >= num_samples:
            break
    
    # If only one uncertainty file is found, return the tensor directly
    if len(uncertainties) == 1:
        uncertainties = uncertainties[0]
    # If no uncertainty files are found, raise an error
    elif len(uncertainties) == 0:
        raise ValueError('No uncertainty files found')
    # If multiple uncertainty files are found, concatenate the tensors along the first dimension
    else:
        uncertainties = torch.cat(uncertainties, dim=0)
    
    # If num_samples is specified and the actual number of rows loaded is less than num_samples, raise an error
    if num_samples is not None and actual_num_rows < num_samples:
        raise ValueError(f'Not enough samples found in {run}')
    
    # If num_samples is specified and the actual number of rows loaded exceeds num_samples, slice the tensor to num_samples
    if num_samples is not None and actual_num_rows > num_samples:
        uncertainties = uncertainties[:num_samples]
    
    return uncertainties