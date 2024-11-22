from re import S
from beartype import beartype
from click import Option
import torch
from diffusion_uncertainty.generate_samples import predict_model
from diffusion_uncertainty.uvit.uvit_ae import UViTAE
from typing import List, Dict, Literal, Union, Optional
from math import pi, sqrt

@beartype
def calculate_threshold_map(threshold: torch.Tensor | float, i: int | None, pixel_wise_uncertainty: torch.Tensor, threshold_type: Literal['higher', 'lower']):
    if isinstance(threshold, float):
        uncertainty_shape: torch.Size = pixel_wise_uncertainty.shape
                        # print('uncertainty_shape:', uncertainty_shape)
        uncertainty_quantile = torch.quantile(pixel_wise_uncertainty.flatten(1).to(torch.float32), threshold, dim=1, keepdim=True).view(uncertainty_shape[0], *([1] * (len(uncertainty_shape) - 1)))
        if threshold_type == 'higher':
            thresholded_map = pixel_wise_uncertainty > uncertainty_quantile
        else:
            thresholded_map = pixel_wise_uncertainty < uncertainty_quantile
        thresholded_map = thresholded_map.float()
    else:
        threshold_i = threshold[i]
        if pixel_wise_uncertainty.dim() == 4:
            threshold_i = threshold_i.unsqueeze(0) 
        if threshold_type == 'higher':
            thresholded_map = pixel_wise_uncertainty > threshold_i
        else:
            thresholded_map = pixel_wise_uncertainty < threshold_i
        thresholded_map: torch.Tensor = thresholded_map.float()
    return thresholded_map

def estimate_score_update_posterior(M: int, model: torch.nn.Module, scheduler, input: torch.Tensor, y_slice: torch.Tensor, t_tensor: torch.Tensor, noisy_residual: torch.Tensor, prev_noisy_sample: torch.Tensor, alpha_hat_t):
    """
    Estimates the score update for a given input sample.

    Args:
        input (torch.Tensor): The input sample.
        y_slice (torch.Tensor): The target output slice.
        i (int): The step number.
        t_tensor (torch.Tensor): The tensor representing the time step.
        noisy_residual (torch.Tensor): The noisy residual.
        prev_noisy_sample (torch.Tensor): The previous noisy sample.
        alpha_hat_t (float): The estimated alpha_hat_t value.

    Returns:
        tuple: A tuple containing the pixel-wise uncertainty and the update scores.
    """
    noisy_residual.requires_grad = False
    input.requires_grad = False
    t_tensor.requires_grad = False
    y_slice.requires_grad = False
    pred_epsilons = []
    pred_x_0 = (input - sqrt(1 - alpha_hat_t) * noisy_residual) / sqrt(alpha_hat_t)
    for _ in range(M):
        x_hat_t = sqrt(alpha_hat_t) * pred_x_0 + sqrt(1 - alpha_hat_t) * torch.randn_like(prev_noisy_sample)
        pred_epsilon = predict_model(model, x_hat_t, t_tensor, y_slice)
        pred_epsilons.append(pred_epsilon)
    pred_epsilons.append(noisy_residual)
    pred_epsilons = torch.stack(tensors=pred_epsilons, dim=0)

    pixel_wise_uncertainty = torch.var(pred_epsilons, dim=0 , unbiased=True)

    uncertainty = pixel_wise_uncertainty.mean(dim=0).sum()
    inv_var: torch.Tensor = 1 / pixel_wise_uncertainty
    post_var_trace = (M * inv_var) + (1 / alpha_hat_t)
    post_precision = 1 / post_var_trace
    new_score = post_precision * (inv_var * pred_epsilon.sum(dim=0))
    return pixel_wise_uncertainty, new_score


class DiffusionClassConditionalGuidedSecondOrder:

    def __init__(self, model: Union[UViTAE, torch.nn.Module], scheduler, threshold: torch.Tensor | float, image_size: int, device: torch.device, batch_size: int, init_seed_rng: int, fid_evaluator: Optional[object] = None, M: int = 5, threshold_type: Literal['higher', 'lower'] = 'higher'):
        assert isinstance(threshold, (torch.Tensor, float)), "Threshold must be a tensor or a float"
        if isinstance(threshold, float):
            assert 0 <= threshold <= 1, "Threshold percentile must be between 0 and 1"
        self.model = model
        self.scheduler = scheduler
        self.threshold = threshold
        self.image_size = image_size
        self.device = device
        self.fid_evaluator = fid_evaluator
        self.batch_size = batch_size
        self.is_uvit = isinstance(model, UViTAE)
        self.init_seed_rng = init_seed_rng
        self.M = M       
        self.lambda_update = 7
        self.threshold_type = threshold_type

    def __call__(self, num_samples: Optional[int] = None, num_classes: Optional[int] = None, X_T: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None, start_step: int = 0, num_steps: int | None = None) -> Dict[str, torch.Tensor]:
        """
        Generate samples using the pipeline sampler with conditional uncertainty guidance.

        Args:
            num_samples (Optional[int]): The number of samples to generate. Either `num_samples` or `X_T` must be provided.
            num_classes (Optional[int]): The number of classes. Either `num_classes` or `y` must be provided.
            X_T (Optional[torch.Tensor]): The input tensor. Shape: (batch_size, 3, image_size, image_size).
            y (Optional[torch.Tensor]): The target tensor. Shape: (batch_size,).
            start_step (int): The starting step for the pipeline sampler.
            num_steps (int | None): The number of steps to run the pipeline sampler. If None, the pipeline sampler will run until `num_samples` samples are generated.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the generated samples and other information.
                - 'y': The target tensor of shape (num_samples,).
                - 'x_t': The input tensor of shape (num_samples, 3, image_size, image_size).
                - 'timestep': The timesteps used in the pipeline sampler.
                - 'gen_images': The generated images tensor of shape (num_samples, 3, image_size, image_size).
                - 'fid' (optional): The FID score if the FID evaluator is provided.
        """
        assert num_samples is not None or X_T is not None, "Either num_samples or X_T must be provided"
        assert num_classes is not None or y is not None, "Either num_classes or y must be provided"

        num_generated_samples = 0
        first = True
        samples_x_t = []
        samples_y = []
        samples_gen_images = []
        if num_steps is None:
            num_steps = self.scheduler.timesteps.shape[0] - start_step
        if num_samples is None:
            num_samples = X_T.shape[0]
        if isinstance(self.threshold, torch.Tensor):
            assert self.threshold.shape[0] == self.scheduler.timesteps.shape[0], f'{self.threshold.shape=} {self.scheduler.timesteps.shape=}'
        self.scheduler.config.after_step = start_step
        self.scheduler.config.num_steps_uc = num_steps
        self.scheduler.set_timesteps(len(self.scheduler.timesteps))
        generator = torch.Generator(device=self.device)
        i_batch = 0
        while num_samples > num_generated_samples:
            print(f"Generated samples: {num_generated_samples} / {num_samples}")
            if X_T is not None:
                input = X_T[num_generated_samples:num_generated_samples + self.batch_size]
                input = input.to(self.device)
            else:
                number_channels = 4 if self.is_uvit else 3
                input: torch.Tensor = torch.randn(self.batch_size, number_channels, self.image_size, self.image_size, device=self.device, dtype=torch.float32, generator=generator.manual_seed(self.init_seed_rng + i_batch))
            x_t = input.cpu().clone()
            samples_x_t.append(x_t)
            if y is not None:
                y_slice = y[num_generated_samples:num_generated_samples + self.batch_size]
                y_slice = y_slice.to(self.device)
            else:
                y_slice = torch.randint(0, num_classes, (self.batch_size,), device=self.device, generator=generator.manual_seed(self.init_seed_rng + i_batch))
            samples_y.append(y_slice)
            self.scheduler.prompt_embeds = y_slice
            momentum_beta = 0.99
            second_order_momentum = torch.zeros_like(input)
            with torch.no_grad():
                for i, t in enumerate(self.scheduler.timesteps):
                    t = t.item()
                    t_1 = self.scheduler.timesteps[i - 1].item() if i > 0 else 0 #t-1
                    t_tensor = torch.full((y_slice.shape[0],), t, device=self.device, dtype=torch.long)
                    noisy_residual = predict_model(self.model, input, t_tensor, y_slice)
                    output = self.scheduler.step(noisy_residual, t, input)
                    prev_noisy_sample = output.prev_sample
                    beta_t_1 = self.scheduler.betas[t_1]
                    alpha_hat_t = self.scheduler.alphas_cumprod[i]

                    if ((start_step + num_steps) > i >= start_step):

                        prev_noisy_sample, second_order_momentum = self.update_with_uncertainty(input, y_slice, momentum_beta, second_order_momentum, i, t, t_tensor, noisy_residual, prev_noisy_sample, alpha_hat_t)
                    input = prev_noisy_sample
                gen_images = input
                if self.is_uvit:
                    gen_images = self.model.decode(gen_images)
                gen_images = (gen_images / 2 + 0.5).clamp(0, 1)
                num_generated_samples += gen_images.shape[0]
                gen_images = gen_images * 255.0
                gen_images = gen_images.round()
                # gen_images = gen_images.clip(0, 255)
                gen_images = gen_images.to(torch.uint8)

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
                'x_t': torch.cat(samples_x_t, dim=0).cpu(),
                'timestep': self.scheduler.timesteps,
                'gen_images': torch.cat(samples_gen_images, dim=0).cpu()}
        if self.fid_evaluator is not None:
            results['fid'] = self.fid_evaluator.compute()

        return results

    def update_with_uncertainty(self, input, y_slice, momentum_beta, second_order_momentum, i, t, t_tensor, noisy_residual, prev_noisy_sample, alpha_hat_t):
        pixel_wise_uncertainty = self.estimate_score_update(input, y_slice, i, t_tensor, noisy_residual, prev_noisy_sample, alpha_hat_t)
        print(f'{self.threshold_type=}')
        print(f'threshold: {self.threshold}')
        thresholded_map = calculate_threshold_map(self.threshold, i, pixel_wise_uncertainty, threshold_type=self.threshold_type)
        # pixel_wise_uncertainty = torch.exp(pixel_wise_uncertainty)
        print('percentage of non zero pixels:', thresholded_map.mean(dim=(1, 2, 3)))

        # batch_mean_uncertainty = pixel_wise_uncertainty.mean(dim=(1, 2, 3), keepdim=True)
        # batch_std_uncertainty = pixel_wise_uncertainty.std(dim=(1, 2, 3), keepdim=True)
        # for i in range(len(pixel_wise_uncertainty)):
        #     mask_i = thresholded_map[i] > 0.001
        #     pixel_wise_uncertainty_i = pixel_wise_uncertainty[i]
        #     pixel_wise_uncertainty_i[mask_i] = (pixel_wise_uncertainty_i[mask_i] - pixel_wise_uncertainty_i[mask_i].mean()) / (pixel_wise_uncertainty_i[mask_i].std() + 1e-6)  
        #     pixel_wise_uncertainty_i[mask_i] = pixel_wise_uncertainty_i[mask_i] * batch_std_uncertainty[i] + batch_mean_uncertainty[i]
        #     pixel_wise_uncertainty[i] = pixel_wise_uncertainty_i

        if second_order_momentum is None:
            second_order_momentum = pixel_wise_uncertainty
        else:
            second_order_momentum = momentum_beta * second_order_momentum + (1 - momentum_beta) * pixel_wise_uncertainty
        corrected_second_order_momentum = second_order_momentum / (1 - momentum_beta ** (i) + 1e-5)

        sqrt_corrected_second_order_momentum = torch.sqrt(corrected_second_order_momentum)
                        
        print('pixel_wise_uncertainty mean:', pixel_wise_uncertainty.mean())
        print('pixel_wise_uncertainty std:', pixel_wise_uncertainty.std())
        print('pixel_wise_uncertainty min:', pixel_wise_uncertainty.amin())
        print('pixel_wise_uncertainty max:', pixel_wise_uncertainty.amax())
                        
        print('second_order_momentum mean:', second_order_momentum.mean())
        print('second_order_momentum std:', second_order_momentum.std())
        print('second_order_momentum min:', second_order_momentum.amin())
        print('second_order_momentum max:', second_order_momentum.amax())

        print('corrected_second_order_momentum mean:', corrected_second_order_momentum.mean())
        print('corrected_second_order_momentum std:', corrected_second_order_momentum.std())
        print('corrected_second_order_momentum min:', corrected_second_order_momentum.amin())
        print('corrected_second_order_momentum max:', corrected_second_order_momentum.amax())

        print('sqrt_corrected_second_order_momentum mean:', sqrt_corrected_second_order_momentum.mean())
        print('sqrt_corrected_second_order_momentum std:', sqrt_corrected_second_order_momentum.std())
        print('sqrt_corrected_second_order_momentum min:', sqrt_corrected_second_order_momentum.amin())
        print('sqrt_corrected_second_order_momentum max:', sqrt_corrected_second_order_momentum.amax())
        
        print('noisy_residual mean:', noisy_residual.mean())
        print('noisy_residual std:', noisy_residual.std())
        print('noisy_residual min:', noisy_residual.amin())
        print('noisy_residual max:', noisy_residual.amax())

        # pixel_wise_uncertainty = (pixel_wise_uncertainty - pixel_wise_uncertainty.mean(dim=(1, 2, 3), keepdim=True)) / (pixel_wise_uncertainty.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
        # pixel_wise_uncertainty = pixel_wise_uncertainty * batch_std_uncertainty + batch_mean_uncertainty
        # min_noisy_residual = noisy_residual.amin()
        # max_noisy_residual = noisy_residual.amax()
        noisy_residual = noisy_residual + pixel_wise_uncertainty *  torch.sign(torch.randn_like(noisy_residual)) * thresholded_map
        # noisy_residual= noisy_residual / (((pixel_wise_uncertainty) * thresholded_map  ) + ((1-thresholded_map) * torch.ones_like(input=sqrt_corrected_second_order_momentum)) + 1e-6) 
        # noisy_residual = (noisy_residual - noisy_residual.amin()) / (noisy_residual.amax() - noisy_residual.amin())
        # noisy_residual = noisy_residual * (max_noisy_residual - min_noisy_residual) + min_noisy_residual


        print('noisy_residual after correction mean:', noisy_residual.mean())
        print('noisy_residual after correction std:', noisy_residual.std())
        print('noisy_residual after correction min:', noisy_residual.amin())
        print('noisy_residual after correction max:', noisy_residual.amax())

        output = self.scheduler.step(noisy_residual, t, input)
        prev_noisy_sample = output.prev_sample
        return prev_noisy_sample, second_order_momentum

    def calculate_threshold_map(self, i, pixel_wise_uncertainty):
        if isinstance(self.threshold, float):
            uncertainty_shape: torch.Size = pixel_wise_uncertainty.shape
                            # print('uncertainty_shape:', uncertainty_shape)
            thresholded_map = pixel_wise_uncertainty > torch.quantile(pixel_wise_uncertainty.flatten(1), self.threshold, dim=1, keepdim=True).view(uncertainty_shape[0], *([1] * (len(uncertainty_shape) - 1)))
            thresholded_map = thresholded_map.float()
        else:
            thresholded_map = pixel_wise_uncertainty > self.threshold[i]
            thresholded_map: torch.Tensor = thresholded_map.float()
        return thresholded_map

    def estimate_score_update(self, input, y_slice, i, t_tensor, noisy_residual, prev_noisy_sample, alpha_hat_t):
        """
        Estimates the score update for a given input sample.

        Args:
            input (torch.Tensor): The input sample.
            y_slice (torch.Tensor): The target output slice.
            i (int): The step number.
            t_tensor (torch.Tensor): The tensor representing the time step.
            noisy_residual (torch.Tensor): The noisy residual.
            prev_noisy_sample (torch.Tensor): The previous noisy sample.
            alpha_hat_t (float): The estimated alpha_hat_t value.

        Returns:
            tuple: A tuple containing the pixel-wise uncertainty and the update scores.
        """
        noisy_residual.requires_grad = False
        input.requires_grad = False
        t_tensor.requires_grad = False
        y_slice.requires_grad = False
        print('Step #:', i)
        pred_epsilons = []
        pred_x_0 = (input - sqrt(1 - alpha_hat_t) * noisy_residual) / sqrt(alpha_hat_t)
        for _ in range(self.M):
            x_hat_t = sqrt(alpha_hat_t) * pred_x_0 + sqrt(1 - alpha_hat_t) * torch.randn_like(prev_noisy_sample)
            pred_epsilon = predict_model(self.model, x_hat_t, t_tensor, y_slice)
            pred_epsilons.append(pred_epsilon)
        # pred_epsilons.append(noisy_residual)
        beta_t = self.scheduler.betas[i]
        pred_epsilons = torch.stack(pred_epsilons, dim=0)

        pixel_wise_uncertainty = (pred_epsilons - noisy_residual.unsqueeze(0)).pow(2).mean(dim=0)
        return pixel_wise_uncertainty
    

    def get_y_batch(self, num_classes, y, num_generated_samples, i_batch, generator, X_t):
        if y is not None:
            y_batch = y[num_generated_samples:num_generated_samples + self.batch_size]
        else:
            y_batch = torch.randint(0, num_classes, (self.batch_size,), device=self.device, generator=generator.manual_seed(self.init_seed_rng + i_batch))
        return y_batch

    def get_X_T_batch(self, X_T, num_generated_samples, i_batch, generator):
        if X_T is not None:
            X_T_batch = X_T[num_generated_samples:num_generated_samples + self.batch_size]
        else:
            if self.is_uvit:
                X_T_batch = torch.randn(self.batch_size, 4, self.image_size, self.image_size, device=self.device, dtype=torch.float32, generator=generator.manual_seed(self.init_seed_rng + i_batch))
            else:
                X_T_batch: torch.Tensor = torch.randn(self.batch_size, 3, self.image_size, self.image_size, device=self.device, dtype=torch.float32, generator=generator.manual_seed(self.init_seed_rng + i_batch))
        return X_T_batch

    def predict_score(self, input: torch.Tensor, y_batch: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
        if self.is_uvit:
            noisy_residual = self.model(input, t_tensor, y_batch)
        elif self.is_cifar10:
            noisy_residual = self.model(input, t_tensor).sample
        else:
            noisy_residual = self.model(input, t_tensor, y=y_batch)[:, :3]
        return noisy_residual