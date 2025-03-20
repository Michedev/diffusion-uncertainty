from diffusion_uncertainty.generate_samples import predict_model
from diffusion_uncertainty.schedulers_uncertainty.mixin import SchedulerUncertaintyMixin


import torch


from math import sqrt


@torch.no_grad()
def generate_samples_model_scheduler_class_conditioned_with_threshold(num_samples, batch_size, image_size, model, scheduler, num_classes: int | torch.Tensor, threshold: torch.Tensor, device=None, fid_evaluator=None, x_T=None, y=None, start_step: int = 0, num_steps: int | None = None, seed: int = 0, is_cifar10: bool = False):
    """
    Generates samples using a model and scheduler, conditioned on a class label, with a threshold for uncertainty.

    Args:
        num_samples (int): The total number of samples to generate.
        batch_size (int): The batch size for generating samples.
        image_size (int): The size of the input image.
        model: The model used for generating samples.
        scheduler: The scheduler used for generating samples.
        num_classes (int or torch.Tensor): The number of classes or a tensor of class labels.
        threshold (torch.Tensor): The threshold for uncertainty.
        device: The device to use for generating samples.
        fid_evaluator: The fid evaluator for computing FID score.

    Returns:
        dict: A dictionary containing the generated samples, class labels, timesteps, and FID score (if fid_evaluator is provided).
    """
    num_generated_samples = 0
    first = True
    print('device:', device)
    samples_x_t = []
    samples_y = []
    samples_gen_images = []
    if num_steps is None:
        num_steps = scheduler.timesteps.shape[0]
    assert threshold.shape[0] == scheduler.timesteps.shape[0], f'{threshold.shape=} {scheduler.timesteps.shape=}'
    scheduler.config.after_step = start_step
    scheduler.config.num_steps_uc = num_steps
    scheduler.set_timesteps(len(scheduler.timesteps))
    generator = torch.Generator(device=device)
    i_batch = 0
    while num_samples > num_generated_samples:
        print(f"Generated samples: {num_generated_samples} / {num_samples}")
        if x_T is not None:
            input = x_T[num_generated_samples:num_generated_samples + batch_size]
            input = input.to(device)
        else:
            input = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=torch.float32, generator=generator.manual_seed(seed + i_batch))
        x_t = input.cpu().clone()
        samples_x_t.append(x_t)
        if y is not None:
            y_slice = y[num_generated_samples:num_generated_samples + batch_size]
            y_slice = y_slice.to(device)
        elif isinstance(num_classes, int):
            y_slice = torch.randint(0, num_classes, (batch_size,), device=device, generator=generator.manual_seed(seed + i_batch))
        else:
            assert num_samples == num_classes.shape[0]
            y_slice = num_classes[num_generated_samples:num_generated_samples + batch_size]
            if y_slice.shape[0] < batch_size:
                input = input[:y_slice.shape[0]]
        samples_y.append(y_slice)
        if isinstance(scheduler, SchedulerUncertaintyMixin):
            scheduler.prompt_embeds = y_slice
        with torch.no_grad():
            for i, t in enumerate(scheduler.timesteps):
                t = t.item()
                t_1 = scheduler.timesteps[i - 1].item() if i > 0 else 0 #t-1
                t_tensor = torch.full((y_slice.shape[0],), t, device=device, dtype=torch.long)
                if is_cifar10:
                    noisy_residual = model(input, t_tensor).sample
                else:
                    noisy_residual = model(input, t_tensor, y=y_slice)[:, :3]
                output = scheduler.step(noisy_residual, t, input)
                prev_noisy_sample = output.prev_sample
                beta_t_1 = scheduler.betas[t_1]
                alpha_hat_t = scheduler.alphas_cumprod[i]

                if isinstance(scheduler, SchedulerUncertaintyMixin):
                    # if scheduler.timestep_after_step >= t >= scheduler.timestep_end_step and (start_step + num_steps >= i >= start_step):
                    if ((start_step + num_steps) >= i >= start_step):
                        # results = []
                        # beta_t_1 = scheduler.betas[t_1]
                        noisy_residual.requires_grad = True
                        input.requires_grad = False
                        t_tensor.requires_grad = False
                        y_slice.requires_grad = False
                        print('Step #:', i)
                        pred_epsilons = []
                        with torch.set_grad_enabled(True):
                            pred_x_0 = (input - sqrt(1 - alpha_hat_t) * noisy_residual) / sqrt(alpha_hat_t)
                            for _ in range(5):
                                x_hat_t = sqrt(alpha_hat_t) * pred_x_0 + sqrt(1 - alpha_hat_t) * torch.randn_like(prev_noisy_sample)
                                pred_epsilon = predict_model(model, x_hat_t, t_tensor, y_slice)
                                pred_epsilons.append(pred_epsilon)
                            pred_epsilons = torch.stack(pred_epsilons, dim=0)
                            # pred_epsilons = (pred_epsilons - pred_epsilons.mean()) / pred_epsilons.std()
                            pixel_wise_uncertainty = (pred_epsilons - noisy_residual.unsqueeze(0)).pow(2).mean(dim=0)
                            # pixel_wise_uncertainty = pred_epsilons.var(dim=0)
                            # pixel_wise_uncertainty = pred_epsilons.pow(2).mean(dim=0)

                            uncertainty = pixel_wise_uncertainty.mean(dim=0).sum()
                            uncertainty.backward()
                        update_scores = noisy_residual.grad * - 1
                        thresholded_map = pixel_wise_uncertainty > threshold[i]
                        thresholded_map = thresholded_map.float()
                        # update_pixels = torch.stack(results, dim=0).mean(dim=0)
                        # print('thresholded_map:', thresholded_map.shape)
                        # print('update_pixels:', update_pixels.shape)
                        # noisy_residual= noisy_residual * (1 - thresholded_map) + noisy_residual * thresholded_map * update_scores
                        noisy_residual= noisy_residual + thresholded_map * update_scores
                        output = scheduler.step(noisy_residual, t, input)
                        prev_noisy_sample = output.prev_sample
                input = prev_noisy_sample

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

        if fid_evaluator is not None:
            fid_evaluator.update(gen_images, real=False)

        samples_gen_images.append(gen_images)

        i_batch += 1

    results = {'y': torch.cat(samples_y, dim=0).cpu(),
               'x_t': torch.cat(samples_x_t, dim=0).cpu(),
               'timestep': scheduler.timesteps,
               'gen_images': torch.cat(samples_gen_images, dim=0).cpu()}
    if fid_evaluator is not None:
        results['fid'] = fid_evaluator.compute()

    return results