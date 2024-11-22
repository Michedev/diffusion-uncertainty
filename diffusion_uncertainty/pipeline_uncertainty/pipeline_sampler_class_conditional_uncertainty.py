from click import Option
import torch
from diffusion_uncertainty.uvit.uvit_ae import UViTAE
from typing import List, Dict, Tuple, Union, Optional
from functools import singledispatchmethod

from diffusers.models.unets import UNet2DModel

class DiffusionClassConditionalWithUncertainty:

    def __init__(self, model: Union[UViTAE, torch.nn.Module], scheduler, image_size: int, device: torch.device, batch_size: int, init_seed_rng: int, fid_evaluator: Optional[object] = None, return_intermediates: bool = False):
        self.model = model.to(device)
        self.device = device
        self.image_size = image_size
        self.fid_evaluator = fid_evaluator
        self.batch_size = batch_size
        self.is_uvit = isinstance(model, UViTAE)
        self.is_cifar10 = isinstance(model, UNet2DModel)
        self.init_seed_rng = init_seed_rng
        self.scheduler = scheduler
        self.return_intermediates = return_intermediates


    
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
        X_t: torch.Tensor = X_T.cpu().clone().to(self.device)
        uncertainties = []
        scores = []
        intermediates = []
        self.scheduler.prompt_embeds = y_batch
        with torch.no_grad():
            for t in self.scheduler.timesteps:
                t = t.item()
                t_tensor = torch.full((X_t.shape[0],), t, device=self.device, dtype=torch.long)
                noisy_residual = self.predict_score(X_t, y_batch, t_tensor)
                output = self.scheduler.step(noisy_residual, t, X_t)
                prev_noisy_sample = output.prev_sample
                if self.scheduler.timestep_after_step >= t >= self.scheduler.timestep_end_step:
                    uncertainties.append(output.uncertainty.cpu())
                    scores.append(output.pred_epsilon.cpu())
                if self.return_intermediates:
                    intermediates.append(output.intermediate.cpu())
                X_t = prev_noisy_sample
            if self.is_uvit:
                X_t = self.model.decode(X_t)
            gen_images = (X_t / 2 + 0.5).clamp(0, 1)
            gen_images = gen_images * 255.0
            gen_images = gen_images.round()
            gen_images = gen_images.to(torch.uint8)
        if self.return_intermediates:
            return gen_images, torch.stack(uncertainties, dim=1), torch.stack(scores, dim=1), torch.stack(intermediates, dim=1)
        return gen_images, torch.stack(uncertainties, dim=1), torch.stack(scores, dim=1)
    
    @singledispatchmethod
    def sample(self, samples, classes):
        raise NotImplementedError
    
    @sample.register
    def _(self, samples: int, classes: int):
        return self(num_samples=samples, num_classes=classes)
    
    @sample.register
    def _(self, X_T: torch.Tensor, y: torch.Tensor):
        return self(X_T=X_T, y=y)   
     

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
            
            assert num_samples is not None or X_T is not None, "Either num_samples or X_T must be provided"
            assert num_classes is not None or y is not None, "Either num_classes or y must be provided"
            num_generated_samples = 0
            first = True
            samples_uncertanties: List[torch.Tensor] = []
            samples_scores: List[torch.Tensor] = []
            samples_X_t: List[torch.Tensor] = []
            samples_y: List[torch.Tensor] = []
            samples_gen_images: List[torch.Tensor] = []
            samples_intermediates: List[torch.Tensor] = []
            i_batch = 0
            if num_samples is None:
                num_samples = X_T.shape[0]
            generator = torch.Generator(device=self.device)
            while num_samples > num_generated_samples:
                print(f"Generated samples: {num_generated_samples} / {num_samples}")
                X_T_batch = self.get_X_T_batch(X_T, num_generated_samples, i_batch, generator)
                y_batch = self.get_y_batch(num_classes, y, num_generated_samples, i_batch, generator=generator)
                X_t: torch.Tensor = X_T_batch.cpu().clone()
                samples_X_t.append(X_t)
                samples_y.append(y_batch)

                self.scheduler.prompt_embeds = y_batch
                if self.return_intermediates:
                    gen_images, uncertanties, scores, intermediates = self.generate(X_T=X_T_batch, y_batch=y_batch)
                    samples_intermediates.append(intermediates.cpu())
                else:
                    gen_images, uncertanties, scores = self.generate(X_T=X_T_batch, y_batch=y_batch)
                samples_uncertanties.append(uncertanties)
                samples_scores.append(scores)
                
                num_generated_samples += gen_images.shape[0]

                if first:
                    print(gen_images.shape)
                    print('min:', gen_images.amin())
                    print('max:', gen_images.amax())
                    first = False

                if self.fid_evaluator is not None:
                    self.fid_evaluator.update(gen_images, real=False)

                samples_gen_images.append(gen_images)

                i_batch += 1

            results = {'y': torch.cat(samples_y, dim=0).cpu(), 
                    'x_t': torch.cat(samples_X_t, dim=0).cpu(),
                    'timestep': self.scheduler.timesteps,
                    'gen_images': torch.cat(samples_gen_images, dim=0).cpu()}
            if self.return_intermediates:
                results['intermediates'] = torch.cat(samples_intermediates, dim=0).cpu()
            if self.fid_evaluator is not None:
                results['fid'] = self.fid_evaluator.compute()
            results['uncertainty'] = torch.cat(samples_uncertanties, dim=0).cpu()
            results['score'] = torch.cat(samples_scores, dim=0).cpu()

            return results

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
        if X_T is not None:
            X_T_batch = X_T[num_generated_samples:num_generated_samples + self.batch_size]
        else:
            if self.is_uvit:
                X_T_batch = torch.randn(self.batch_size, 4, self.image_size, self.image_size, device=self.device, dtype=torch.float32, generator=generator.manual_seed(self.init_seed_rng + i_batch))
            else:
                X_T_batch = torch.randn(self.batch_size, 3, self.image_size, self.image_size, device=self.device, dtype=torch.float32, generator=generator.manual_seed(self.init_seed_rng + i_batch))
        return X_T_batch

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
        if self.is_uvit:
            noisy_residual = self.model(input, t_tensor, y_batch)
        elif self.is_cifar10:
            noisy_residual = self.model(input, t_tensor).sample
        else:
            noisy_residual = self.model(input, t_tensor, y=y_batch)[:, :3]
        return noisy_residual