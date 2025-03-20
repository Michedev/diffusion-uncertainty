from typing import overload
import torch
import numpy as np

@overload
def compute_aucs(gt: torch.Tensor, pred: torch.Tensor, uncert: torch.Tensor, intervals: int): ...

@overload
def compute_aucs(gt: np.ndarray, pred: np.ndarray, uncert: np.ndarray, intervals: int): ...

def compute_aucs_from_curve(opt_curve: torch.Tensor, rnd_curve: torch.Tensor, sparse_curve: torch.Tensor, intervals: int) -> tuple[torch.Tensor, torch.Tensor]: ...