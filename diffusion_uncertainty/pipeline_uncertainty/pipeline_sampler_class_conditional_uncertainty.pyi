import torch
from diffusion_uncertainty.uvit.uvit_ae import UViTAE
from typing import List, Dict, Tuple, Union, Optional, overload
from functools import singledispatchmethod

class DiffusionClassConditionalWithUncertainty:

    def __init__(self, model: Union[UViTAE, torch.nn.Module], scheduler, image_size: int, device: torch.device, batch_size: int, init_seed_rng: int, fid_evaluator: Optional[object] = None, return_intermediates: bool = False):
        ...

    
    def generate(self, X_T, y_batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates images using the given inputs.

        Args:
            X_T (torch.Tensor): The input tensor.
            y_batch: The batch of target labels.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the generated images,
            uncertainties, and scores.
        """
        ...

    @overload
    def sample(self, samples: int, classes: int):
        ...
    
    @overload
    def sample(self, X_T: torch.Tensor, y: torch.Tensor):
        ...   
     

    def __call__(self, /, num_samples: Optional[int] = None, num_classes: Optional[int] = None, X_T: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
            """
            Generate samples and compute uncertainties and scores. Either (num_samples, num_classes) or (X_T, y) must be provided.

            Args:
                num_samples (Optional[int]): The number of samples to generate
                num_classes (Optional[int]): The number of classes
                X_T (Optional[torch.Tensor]): The input tensor
                y (Optional[torch.Tensor]): The target tensor

            Returns:
                Dict[str, torch.Tensor]: A dictionary containing the generated samples, uncertainties, scores, and other results.
            """
    def get_y_batch(self, num_classes: int, y: torch.Tensor | None, num_generated_samples, i_batch, generator):
        """
        Get a batch of target labels for the generator. If y is not provided, random labels will be generated based on the number of classes.

        Args:
            num_classes (int): The number of classes.
            y (torch.Tensor | None): The target labels. If None, random labels will be generated.
            num_generated_samples: The number of samples generated so far.
            i_batch: The index of the current batch.
            generator: The random number generator.

        Returns:
            torch.Tensor: A batch of target labels.
        """
        if y is not None:
            y_batch = y[num_generated_samples:num_generated_samples + self.batch_size]
        else:
            y_batch = torch.randint(0, num_classes, (self.batch_size,), device=self.device, generator=generator.manual_seed(self.init_seed_rng + i_batch))
        return y_batch

    def get_X_T_batch(self, X_T: torch.Tensor | None, num_generated_samples, i_batch, generator):
        """
        Returns a batch of input samples X_T. If X_T is not provided, random samples will be generated.

        Args:
            X_T (torch.Tensor | None): Input samples. If not None, a batch of samples is extracted from X_T.
            num_generated_samples (int): Number of generated samples.
            i_batch (int): Batch index.
            generator (torch.Generator): Random number generator.

        Returns:
            torch.Tensor: A batch of input samples X_T.

        """
        ...

        def predict_score(self, input: torch.Tensor, y_batch: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
            """
            Model-agnostic method that predicts the score based on the input, y_batch, and t_tensor.

            Args:
                input (torch.Tensor): The input tensor.
                y_batch (torch.Tensor): The y_batch tensor.
                t_tensor (torch.Tensor): The t_tensor tensor.

            Returns:
                torch.Tensor: The predicted score tensor.
            """
            ...