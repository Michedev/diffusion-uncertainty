from click import Option
import torch
from diffusion_uncertainty.generate_samples import predict_model
from diffusion_uncertainty.pipeline_uncertainty.pipeline_sampler_class_conditional_uncertainty_guided_posterior_distribution import calculate_threshold_map
from diffusion_uncertainty.uvit.uvit_ae import UViTAE
from typing import List, Dict, Literal, Union, Optional
from math import sqrt

class DiffusionClassConditionalGuidedGradient:

    def __init__(self, model: Union[UViTAE, torch.nn.Module], scheduler, threshold: torch.Tensor, image_size: int, device: torch.device, batch_size: int, init_seed_rng: int, fid_evaluator: Optional[object] = None, M: int = 5, gradient_wrt: Literal['input', 'score'] = 'input', lambda_update: float = 0.1, threshold_type: Literal['higher', 'lower'] = 'higher', gradient_direction: Literal['ascend', 'descend'] = 'descend'):
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
        self.lambda_update = lambda_update
        self.gradient_wrt = gradient_wrt
        self.threshold_type = threshold_type
        self.gradient_direction = gradient_direction

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
        else:
            assert isinstance(self.threshold, float), f'{self.threshold=}'
            assert self.threshold >= 0 and self.threshold <= 1, f'{self.threshold=}'
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
                input: torch.Tensor = torch.randn(self.batch_size, 3, self.image_size, self.image_size, device=self.device, dtype=torch.float32, generator=generator.manual_seed(self.init_seed_rng + i_batch))
            x_t = input.cpu().clone()
            samples_x_t.append(x_t)
            if y is not None:
                y_slice = y[num_generated_samples:num_generated_samples + self.batch_size]
                y_slice = y_slice.to(self.device)
            else:
                y_slice = torch.randint(0, num_classes, (self.batch_size,), device=self.device, generator=generator.manual_seed(self.init_seed_rng + i_batch))
            samples_y.append(y_slice)
            self.scheduler.prompt_embeds = y_slice
            with torch.no_grad():
                for i, t in enumerate(self.scheduler.timesteps):
                    t = t.item()
                    t_1 = self.scheduler.timesteps[i - 1].item() if i > 0 else 0 #t-1
                    t_tensor = torch.full((y_slice.shape[0],), t, device=self.device, dtype=torch.long)
                    noisy_residual = predict_model(self.model, input, t_tensor, y_slice)
                    output = self.scheduler.step(noisy_residual, t, input)
                    prev_noisy_sample = output.prev_sample
                    beta_t_1 = self.scheduler.betas[t_1]
                    beta_t = self.scheduler.betas[t]
                    alpha_hat_t = self.scheduler.alphas_cumprod[i]

                    if ((start_step + num_steps) > i >= start_step):

                        pixel_wise_uncertainty, update_scores = self.estimate_score_update(input, y_slice, i, t_tensor, noisy_residual, prev_noisy_sample, alpha_hat_t)
                        thresholded_map = calculate_threshold_map(self.threshold, None, pixel_wise_uncertainty, self.threshold_type)
                        thresholded_map = thresholded_map.float()
                        # noisy_residual= noisy_residual * (1 + self.lambda_update) + self.lambda_update * thresholded_map * update_scores
                        with torch.no_grad():
                            # update_scores = torch.exp(input=update_scores)
                            print('gradient mean:', update_scores.mean().item())
                            print('gradient std:', update_scores.std().item())
                            print('gradient min:', update_scores.amin().item())
                            print('gradient max:', update_scores.amax().item())

                            print('noisy_residual mean:', noisy_residual.mean().item())
                            print('noisy_residual std:', noisy_residual.std().item())
                            print('noisy_residual min:', noisy_residual.amin().item())
                            print('noisy_residual max:', noisy_residual.amax().item())
                        gradient_direction: Literal[1] | Literal[-1] = 1 if self.gradient_direction == 'ascend' else -1
                        # post_noisy_residual = noisy_residual * self.lambda_update / update_scores * gradient_direction
                        # print(f'{alpha_hat_t.item()=}')
                        post_noisy_residual = noisy_residual  + self.lambda_update * update_scores
                        noisy_residual = noisy_residual * (1 - thresholded_map) + post_noisy_residual * thresholded_map
                        with torch.no_grad():
                            print('noisy_residual updated mean:', noisy_residual.mean().item())
                            print('noisy_residual updated std:', noisy_residual.std().item())
                            print('noisy_residual updated min:', noisy_residual.amin().item())
                            print('noisy_residual updated max:', noisy_residual.amax().item())
                            
                        output = self.scheduler.step(noisy_residual, t, input)
                        prev_noisy_sample = output.prev_sample
                    input = prev_noisy_sample
                if self.is_uvit:
                    input = self.model.decode(input)
            gen_images = (input / 2 + 0.5).clamp(0, 1)
            num_generated_samples += gen_images.shape[0]

            if first:
                print(gen_images.shape)
                print('min:', gen_images.amin())
                print('max:', gen_images.amax())
                first = False

            gen_images = gen_images * 255.0
            gen_images = gen_images.round()
            gen_images = gen_images.to(torch.uint8)

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
        noisy_residual.requires_grad = self.gradient_wrt == 'score'
        input.requires_grad = self.gradient_wrt == 'input'
        t_tensor.requires_grad = False
        y_slice.requires_grad = False
        print('Step #:', i)
        pred_epsilons = []
        with torch.set_grad_enabled(True):
            pred_epsilon = predict_model(self.model, input, t_tensor, y_slice)
            pred_epsilon.mean(dim=0).sum().backward()
            pred_x_0 = (input - sqrt(1 - alpha_hat_t) * noisy_residual) / sqrt(alpha_hat_t)

            for _ in range(self.M):
                x_hat_t = sqrt(alpha_hat_t) * pred_x_0 + sqrt(1 - alpha_hat_t) * torch.randn_like(prev_noisy_sample)
                pred_epsilon = predict_model(self.model, x_hat_t, t_tensor, y_slice)
                pred_epsilons.append(pred_epsilon)
            pred_epsilons = torch.stack(pred_epsilons, dim=0)

            pixel_wise_uncertainty = (pred_epsilons - noisy_residual.unsqueeze(0)).pow(2).mean(dim=0)
            uncertainty = pixel_wise_uncertainty.mean(dim=0).sum()
            uncertainty.backward()
        # with torch.no_grad():
        #     pred_x_0 = (input - sqrt(1 - alpha_hat_t) * noisy_residual) / sqrt(alpha_hat_t)
        #     for _ in range(self.M):
        #         x_hat_t = sqrt(alpha_hat_t) * pred_x_0 + sqrt(1 - alpha_hat_t) * torch.randn_like(prev_noisy_sample)
        #         pred_epsilon = predict_model(self.model, x_hat_t, t_tensor, y_slice)
        #         pred_epsilons.append(pred_epsilon)
        #     pred_epsilons = torch.stack(pred_epsilons, dim=0)

        #     pixel_wise_uncertainty = (pred_epsilons - noisy_residual.unsqueeze(0)).pow(2).mean(dim=0)

        if self.gradient_wrt == 'input':
            update_scores = input.grad
        else:
            update_scores = noisy_residual.grad
        # update_scores = -7 * uncertainty
        return pixel_wise_uncertainty, update_scores
    

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
                X_T_batch = torch.randn(self.batch_size, 3, self.image_size, self.image_size, device=self.device, dtype=torch.float32, generator=generator.manual_seed(self.init_seed_rng + i_batch))
        return X_T_batch

    def predict_score(self, input: torch.Tensor, y_batch: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
        if self.is_uvit:
            noisy_residual = self.model(input, t_tensor, y_batch)
        elif self.is_cifar10:
            noisy_residual = self.model(input, t_tensor).sample
        else:
            noisy_residual = self.model(input, t_tensor, y=y_batch)[:, :3]
        return noisy_residual