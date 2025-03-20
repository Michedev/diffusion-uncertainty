import gc
from math import sqrt
from operator import is_
from random import sample
from typing import Any
from diffusion_uncertainty.guided_diffusion.unet_openai import ResBlock
from diffusion_uncertainty.schedulers_uncertainty.mixin import SchedulerUncertaintyMixin, SchedulerUncertaintyMixin
import torch
import tqdm

from diffusion_uncertainty.uvit.uvit_ae import UViTAE
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unets import UNet2DModel



@torch.no_grad()
def generate_samples_model_scheduler_class_conditioned(num_samples: int, batch_size: int, image_size: int, model: torch.nn.Module, scheduler: SchedulerUncertaintyMixin, num_classes: int | torch.Tensor, device: torch.device = None, fid_evaluator: Any = None, init_seed_rng: int = 0, is_uvit: bool = False, skip_seed: int = 1, is_cifar10: bool = False):
    """
    Generates samples using a model and scheduler, conditioned on a specific class.

    Args:
        num_samples (int): The total number of samples to generate.
        batch_size (int): The batch size for generating samples.
        image_size (int): The size of the input image.
        model (torch.nn.Module): The model used for generating samples.
        scheduler (Scheduler): The scheduler used for generating samples.
        num_classes (int or torch.Tensor): The number of classes or a tensor containing the class labels. If a tensor is given, it must have the same length as the number of samples.
        device (torch.device, optional): The device to use for computation. Defaults to None.

    Returns:
        dict: A dictionary containing the keys 'y', 'score', 'timestep' and 'uncertainty' is scheduler is an uncertainty scheduler
    """

    num_generated_samples = 0
    first = True
    print('device:', device)
    if isinstance(scheduler, SchedulerUncertaintyMixin):
        samples_uncertanties = []
        samples_scores = []
    samples_x_t = []
    samples_y = []
    samples_gen_images = []
    i_batch = 0
    generator = torch.Generator(device=device)
    while num_samples > num_generated_samples:
        print(f"Generated samples: {num_generated_samples} / {num_samples}")
        if is_uvit:
            input = torch.randn(batch_size, 4, image_size, image_size, device=device, dtype=torch.float32, generator=generator.manual_seed(init_seed_rng + i_batch * skip_seed))
        else:
            input = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=torch.float32, generator=generator.manual_seed(init_seed_rng + i_batch * skip_seed))
        x_t = input.cpu().clone()
        samples_x_t.append(x_t)
        if isinstance(num_classes, int):
            y = torch.randint(0, num_classes, (batch_size,), device=device, generator=generator.manual_seed(init_seed_rng + i_batch * skip_seed))
        else:
            assert num_samples == num_classes.shape[0]
            y = num_classes[num_generated_samples:num_generated_samples + batch_size]
            if y.shape[0] < batch_size:
                input = input[:y.shape[0]]
        samples_y.append(y)
        if isinstance(scheduler, SchedulerUncertaintyMixin):
            uncertanties = []
            scores = []
            scheduler.prompt_embeds = y
        with torch.no_grad():
            # print(scheduler.timesteps)
            for t in scheduler.timesteps:
                t = t.item()
                t_tensor = torch.full((input.shape[0],), t, device=device, dtype=torch.long)
                if is_uvit:
                    noisy_residual = model(input, t_tensor, y)
                elif is_cifar10:
                    noisy_residual = model(input, t_tensor).sample
                else:
                    noisy_residual = model(input, t_tensor, y=y)[:, :3]
                output = scheduler.step(noisy_residual, t, input)
                prev_noisy_sample = output.prev_sample
                if isinstance(scheduler, SchedulerUncertaintyMixin):
                    # print('uncertanties:', len(uncertanties))
                    # print('timestep:', t)
                    if scheduler.timestep_after_step >= t >= scheduler.timestep_end_step:
                        uncertanties.append(output.uncertainty.cpu())
                        scores.append(output.pred_epsilon.cpu())
                input = prev_noisy_sample
            if isinstance(scheduler, SchedulerUncertaintyMixin):
                # print('uncertanties:', len(uncertanties))
                samples_uncertanties.append(torch.stack(uncertanties, dim=1))
                samples_scores.append(torch.stack(scores, dim=1))
        
        if is_uvit:
            input = model.decode(input)
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
    if isinstance(scheduler, SchedulerUncertaintyMixin) and samples_uncertanties:
        results['uncertainty'] = torch.cat(samples_uncertanties, dim=0).cpu()
        results['score'] = torch.cat(samples_scores, dim=0).cpu()

    return results

@torch.no_grad()
def generate_samples_model_scheduler_class_conditioned_from_tensor(X_T: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device, model: torch.nn.Module, scheduler: SchedulerUncertaintyMixin, fid_evaluator: Any = None, save_intermediates: bool = False):
    """
    Generates samples using a model and scheduler, conditioned on a tensor.

    Args:
        X_T (torch.Tensor): The input tensor.
        batch_size (int): The batch size.
        y (torch.Tensor): The conditioning tensor.
        device (torch.device): The device to run the model and scheduler on.
        model (torch.nn.Module): The model to generate samples.
        scheduler (SchedulerClassConditionedMixin): The scheduler to control the generation process.
        fid_evaluator (Any, optional): The fid evaluator. Defaults to None.
        save_intermediates (bool, optional): Whether to save intermediate results. Defaults to False.

    Returns:
        dict: A dictionary containing the generated images and other optional results.
    """
    assert X_T.shape[0] == y.shape[0], f'{X_T.shape=} {y.shape=}'

    num_samples = X_T.shape[0]
    image_size = X_T.shape[-1]
    num_generated_samples = 0
    first = True
    print('device:', device)
    if save_intermediates:
        samples_intermediates = []
    if isinstance(scheduler, SchedulerUncertaintyMixin):
        samples_uncertanties = []
        samples_scores = []
    
    samples_gen_images = []
    i_batch = 0
    while num_samples > num_generated_samples:
        print(f"Generated samples: {num_generated_samples} / {num_samples}")
        i_slice = slice(i_batch * batch_size, (i_batch + 1) * batch_size)
        if (i_batch + 1) * batch_size > num_samples:
            i_slice = slice(i_batch * batch_size, num_samples)
        X_T_batch = X_T[i_slice]
        y_batch: torch.Tensor = y[i_slice]

        input = X_T_batch.to(device)
        y_batch = y_batch.to(device)

        if save_intermediates:
            intermediates = []
        if isinstance(scheduler, SchedulerUncertaintyMixin):
            uncertanties = []
            scores = []
            scheduler.prompt_embeds = y_batch
        print('scheduler.timesteps:', scheduler.timesteps)
        scheduler.set_timesteps(len(scheduler.timesteps))
        for t in scheduler.timesteps:
            t = t.item()
            # if isinstance(t, float):
            #     t = round(t)
            t_tensor = torch.full((y_batch.shape[0],), t, device=device, dtype=torch.long)
            input = scheduler.scale_model_input(input, t)
            noisy_residual = model(input, t_tensor, y=y_batch)[:, :3]
            output = scheduler.step(noisy_residual, t, input)
            prev_noisy_sample = output.prev_sample
            if save_intermediates:
                intermediates.append(prev_noisy_sample.cpu())
            if isinstance(scheduler, SchedulerUncertaintyMixin):
                # print('uncertanties:', len(uncertanties))
                # print('timestep:', t)
                if scheduler.timestep_after_step >= t >= scheduler.timestep_end_step:
                    uncertanties.append(output.uncertainty.cpu())
                    scores.append(output.pred_epsilon.cpu())
            input = prev_noisy_sample
        if save_intermediates:
            samples_intermediates.append(torch.stack(intermediates, dim=1))
        if isinstance(scheduler, SchedulerUncertaintyMixin):
            # print('uncertanties:', len(uncertanties))
            samples_uncertanties.append(torch.stack(uncertanties, dim=1))
            samples_scores.append(torch.stack(scores, dim=1))
            
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

    results = {'gen_images': torch.cat(samples_gen_images, dim=0).cpu()}
    if save_intermediates:
        results['intermediates'] = torch.cat(samples_intermediates, dim=0).cpu()
    if fid_evaluator is not None:
        results['fid'] = fid_evaluator.compute()
    if isinstance(scheduler, SchedulerUncertaintyMixin) and samples_uncertanties:
        results['uncertainty'] = torch.cat(samples_uncertanties, dim=0).cpu()
        results['score'] = torch.cat(samples_scores, dim=0).cpu()

    return results

@torch.no_grad()
def generate_samples_model_scheduler_classifier_based_guidance(X_T: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device, model: torch.nn.Module, scheduler: SchedulerUncertaintyMixin, classifier: torch.nn.Module, classifier_scale: float, fid_evaluator: Any = None, save_intermediates: bool = False):
    """
    Generates samples using a model and scheduler, conditioned on a tensor.

    Args:
        X_T (torch.Tensor): The input tensor.
        batch_size (int): The batch size.
        y (torch.Tensor): The conditioning tensor.
        device (torch.device): The device to run the model and scheduler on.
        model (torch.nn.Module): The model to generate samples.
        scheduler (SchedulerClassConditionedMixin): The scheduler to control the generation process.
        fid_evaluator (Any, optional): The fid evaluator. Defaults to None.
        save_intermediates (bool, optional): Whether to save intermediate results. Defaults to False.

    Returns:
        dict: A dictionary containing the generated images and other optional results.
    """
    assert X_T.shape[0] == y.shape[0], f'{X_T.shape=} {y.shape=}'
    
    def cond_fn(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = logits.log_softmax(dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            gradient = torch.autograd.grad(selected.sum(), x_in, create_graph=False, retain_graph=False)[0]
            gradient = gradient.detach()
        x_in.grad = None
        del logits, log_probs, selected
        for param in classifier.parameters():
            if hasattr(param, 'grad'):
                param.grad = None
        torch.cuda.empty_cache()
        return gradient * classifier_scale

    num_samples = X_T.shape[0]
    image_size = X_T.shape[-1]
    num_generated_samples = 0
    first = True
    print('device:', device)
    if save_intermediates:
        samples_intermediates = []
    if isinstance(scheduler, SchedulerUncertaintyMixin):
        samples_uncertanties = []
        samples_scores = []
    
    samples_gen_images = []
    i_batch = 0
    while num_samples > num_generated_samples:
        print(f"Generated samples: {num_generated_samples} / {num_samples}")
        i_slice = slice(i_batch * batch_size, (i_batch + 1) * batch_size)
        if (i_batch + 1) * batch_size > num_samples:
            i_slice = slice(i_batch * batch_size, num_samples)
        X_T_batch = X_T[i_slice]
        y_batch: torch.Tensor = y[i_slice]

        input = X_T_batch.to(device)
        y_batch = y_batch.to(device)

        if save_intermediates:
            intermediates = []
        if isinstance(scheduler, SchedulerUncertaintyMixin):
            uncertanties = []
            scores = []
            scheduler.prompt_embeds = y_batch
        print('scheduler.timesteps:', scheduler.timesteps)
        scheduler.set_timesteps(len(scheduler.timesteps))
        for t in scheduler.timesteps:
            t = t.item()
            # if isinstance(t, float):
            #     t = round(t)
            t_tensor = torch.full((y_batch.shape[0],), t, device=device, dtype=torch.long)
            input = scheduler.scale_model_input(input, t)
            noisy_residual = model(input, t_tensor, y=y_batch)[:, :3]
            classifier_residual = cond_fn(input, t_tensor, y_batch)
            alpha_bar = scheduler.alphas_cumprod[t]
            noisy_residual = noisy_residual  - (1 - alpha_bar).sqrt() * classifier_residual
            output = scheduler.step(noisy_residual, t, input)
            prev_noisy_sample = output.prev_sample
            if save_intermediates:
                intermediates.append(prev_noisy_sample.cpu())
            if isinstance(scheduler, SchedulerUncertaintyMixin):
                # print('uncertanties:', len(uncertanties))
                # print('timestep:', t)
                if scheduler.timestep_after_step >= t >= scheduler.timestep_end_step:
                    uncertanties.append(output.uncertainty.cpu())
                    scores.append(output.pred_epsilon.cpu())
            input = prev_noisy_sample
        if save_intermediates:
            samples_intermediates.append(torch.stack(intermediates, dim=1))
        if isinstance(scheduler, SchedulerUncertaintyMixin):
            # print('uncertanties:', len(uncertanties))
            samples_uncertanties.append(torch.stack(uncertanties, dim=1))
            samples_scores.append(torch.stack(scores, dim=1))
            
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

    results = {'gen_images': torch.cat(samples_gen_images, dim=0).cpu()}
    if save_intermediates:
        results['intermediates'] = torch.cat(samples_intermediates, dim=0).cpu()
    if fid_evaluator is not None:
        results['fid'] = fid_evaluator.compute()
    if isinstance(scheduler, SchedulerUncertaintyMixin) and samples_uncertanties:
        results['uncertainty'] = torch.cat(samples_uncertanties, dim=0).cpu()
        results['score'] = torch.cat(samples_scores, dim=0).cpu()

    return results


@torch.no_grad()
def generate_samples_model_scheduler_unconditioned_from_tensor(X_T: torch.Tensor, batch_size: int, device: torch.device, model: torch.nn.Module, scheduler: SchedulerUncertaintyMixin, fid_evaluator: Any = None, save_intermediates: bool = False):
    """
    Generates samples using a model and scheduler, conditioned on a tensor.

    Args:
        X_T (torch.Tensor): The input tensor.
        batch_size (int): The batch size.
        device (torch.device): The device to run the model and scheduler on.
        model (torch.nn.Module): The model to generate samples.
        scheduler (SchedulerClassConditionedMixin): The scheduler to control the generation process.
        fid_evaluator (Any, optional): The fid evaluator. Defaults to None.
        save_intermediates (bool, optional): Whether to save intermediate results. Defaults to False.

    Returns:
        dict: A dictionary containing the generated images and other optional results.
    """

    num_samples = X_T.shape[0]
    image_size = X_T.shape[-1]
    num_generated_samples = 0
    first = True
    print('device:', device)
    if save_intermediates:
        samples_intermediates = []
    if isinstance(scheduler, SchedulerUncertaintyMixin):
        samples_uncertanties = []
        samples_scores = []
    
    samples_gen_images = []
    i_batch = 0
    while num_samples > num_generated_samples:
        print(f"Generated samples: {num_generated_samples} / {num_samples}")
        i_slice = slice(i_batch * batch_size, (i_batch + 1) * batch_size)
        if (i_batch + 1) * batch_size > num_samples:
            i_slice = slice(i_batch * batch_size, num_samples)
        X_T_batch = X_T[i_slice]

        input = X_T_batch.to(device)

        if save_intermediates:
            intermediates = []
        if isinstance(scheduler, SchedulerUncertaintyMixin):
            uncertanties = []
            scores = []
        # print(scheduler.timesteps)
        for t in scheduler.timesteps:
            t = t.item()
            t_tensor = torch.full((X_T_batch.shape[0],), t, device=device, dtype=torch.long)
            noisy_residual = model(input, t_tensor).sample[:, :3]
            output = scheduler.step(noisy_residual, t, input)
            prev_noisy_sample = output.prev_sample
            if save_intermediates:
                intermediates.append(prev_noisy_sample.cpu())
            if isinstance(scheduler, SchedulerUncertaintyMixin):
                # print('uncertanties:', len(uncertanties))
                # print('timestep:', t)
                if scheduler.timestep_after_step >= t >= scheduler.timestep_end_step:
                    uncertanties.append(output.uncertainty.cpu())
                    scores.append(output.pred_epsilon.cpu())
            input = prev_noisy_sample
        if save_intermediates:
            samples_intermediates.append(torch.stack(intermediates, dim=1))
        if isinstance(scheduler, SchedulerUncertaintyMixin):
            # print('uncertanties:', len(uncertanties))
            samples_uncertanties.append(torch.stack(uncertanties, dim=1))
            samples_scores.append(torch.stack(scores, dim=1))
            
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

    results = {'gen_images': torch.cat(samples_gen_images, dim=0).cpu()}
    if save_intermediates:
        results['intermediates'] = torch.cat(samples_intermediates, dim=0).cpu()
    if fid_evaluator is not None:
        results['fid'] = fid_evaluator.compute()
    if isinstance(scheduler, SchedulerUncertaintyMixin) and samples_uncertanties:
        results['uncertainty'] = torch.cat(samples_uncertanties, dim=0).cpu()
        results['score'] = torch.cat(samples_scores, dim=0).cpu()

    return results




@torch.no_grad()
def generate_samples_model_scheduler_class_conditioned_uvit_from_tensor(X_T: torch.Tensor, y: torch.Tensor, batch_size: int, uvit_ae: torch.nn.Module, scheduler: SchedulerUncertaintyMixin, device: torch.device | str = 'cpu', fid_evaluator: Any = None):
    """
    Generates samples using a model and scheduler conditioned on class labels.

    Args:
        X_T (torch.Tensor): The input tensor.
        y (torch.Tensor): The class labels tensor.
        batch_size (int): The batch size.
        image_size (int): The size of the image.
        uvit_ae (torch.nn.Module): The UVIT autoencoder model.
        scheduler (SchedulerClassConditionedMixin): The scheduler.
        num_classes (int | torch.Tensor): The number of classes or a tensor of class labels.
        device (torch.device | str, optional): The device to use. Defaults to 'cpu'.
        fid_evaluator (Any, optional): The FID evaluator. Defaults to None.

    Returns:
        dict: A dictionary containing the generated samples and other results.
    """
    assert X_T.shape[0] == y.shape[0], f'{X_T.shape=} {y.shape=}'

    num_samples = X_T.shape[0]
    num_generated_samples = 0
    first = True
    print('device:', device)
    if isinstance(scheduler, SchedulerUncertaintyMixin):
        samples_uncertanties = []
        samples_scores = []
    samples_gen_images = []
    i_batch = 0
    while num_samples > num_generated_samples:
        print(f"Generated samples: {num_generated_samples} / {num_samples}")
    
        slice_batch = slice(i_batch * batch_size, (i_batch + 1) * batch_size)
        if (i_batch + 1) * batch_size > num_samples:
            slice_batch = slice(i_batch * batch_size, num_samples)
        X_T_batch = X_T[slice_batch]
        y_batch = y[slice_batch]

        input = X_T_batch.to(device)
        y_batch = y_batch.to(device)

        if isinstance(scheduler, SchedulerUncertaintyMixin):
            uncertanties = []
            scores = []
            scheduler.prompt_embeds = y_batch
        with torch.no_grad():
            # print(scheduler.timesteps)
            scheduler.set_timesteps(len(scheduler.timesteps))
            for t in scheduler.timesteps:
                t = t.item()
                t_tensor = torch.full((y_batch.shape[0],), t, device=device, dtype=torch.long)
                noisy_residual = uvit_ae(input, t_tensor, y_batch)
                output = scheduler.step(noisy_residual, t, input)
                prev_noisy_sample = output.prev_sample
                if isinstance(scheduler, SchedulerUncertaintyMixin):
                    # print('uncertanties:', len(uncertanties))
                    # print('timestep:', t)
                    if scheduler.timestep_after_step >= t >= scheduler.timestep_end_step:
                        uncertanties.append(output.uncertainty.cpu())
                        scores.append(output.pred_epsilon.cpu())
                input = prev_noisy_sample
            if isinstance(scheduler, SchedulerUncertaintyMixin):
                # print('uncertanties:', len(uncertanties))
                samples_uncertanties.append(torch.stack(uncertanties, dim=1))
                samples_scores.append(torch.stack(scores, dim=1))
            input = uvit_ae.decode(input)
        
        gen_images = (input / 2 + 0.5).clamp(0, 1)

        # if isinstance(gen_images, np.ndarray):
        #     gen_images = torch.from_numpy(gen_images).permute(0, 3, 1, 2)
        #     if device is not None:
        #         gen_images = gen_images.to(device)
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

    results = {
               'timestep': scheduler.timesteps,
               'gen_images': torch.cat(samples_gen_images, dim=0).cpu()}
    if fid_evaluator is not None:
        results['fid'] = fid_evaluator.compute()
    if isinstance(scheduler, SchedulerUncertaintyMixin) and samples_uncertanties:
        results['uncertainty'] = torch.cat(samples_uncertanties, dim=0).cpu()
        results['score'] = torch.cat(samples_scores, dim=0).cpu()

    return results


def generate_samples_model_scheduler_class_conditioned_uvit(num_samples: int, batch_size: int, image_size: int, uvit_ae: torch.nn.Module, scheduler: SchedulerUncertaintyMixin, num_classes: int | torch.Tensor, device: torch.device | str = 'cpu', fid_evaluator: Any = None, init_seed_rng: int = 0):
    


    num_generated_samples = 0
    first = True
    print('device:', device)
    if isinstance(scheduler, SchedulerUncertaintyMixin):
        samples_uncertanties = []
        samples_scores = []
    samples_x_t = []
    samples_y = []
    samples_gen_images = []
    i_batch = 0
    generator = torch.Generator(device=device)
    while num_samples > num_generated_samples:
        print(f"Generated samples: {num_generated_samples} / {num_samples}")
        input = torch.randn(batch_size, uvit_ae.in_chans, uvit_ae.img_size, uvit_ae.img_size, device=device, dtype=torch.float32, generator=generator.manual_seed(init_seed_rng + i_batch))
        x_t = input.cpu().clone()
        samples_x_t.append(x_t)
        if isinstance(num_classes, int):
            y = torch.randint(0, num_classes, (batch_size,), device=device, generator=generator.manual_seed(init_seed_rng + i_batch))
        else:
            assert num_samples == num_classes.shape[0]
            y = num_classes[num_generated_samples:num_generated_samples + batch_size]
            if y.shape[0] < batch_size:
                input = input[:y.shape[0]]
        samples_y.append(y)
        if isinstance(scheduler, SchedulerUncertaintyMixin):
            uncertanties = []
            scores = []
            scheduler.prompt_embeds = y
        with torch.no_grad():
            # print(scheduler.timesteps)
            for t in scheduler.timesteps:
                t = t.item()
                t_tensor = torch.full((y.shape[0],), t, device=device, dtype=torch.long)
                noisy_residual = uvit_ae(input, t_tensor, y)
                output = scheduler.step(noisy_residual, t, input)
                prev_noisy_sample = output.prev_sample
                if isinstance(scheduler, SchedulerUncertaintyMixin):
                    # print('uncertanties:', len(uncertanties))
                    # print('timestep:', t)
                    if scheduler.timestep_after_step >= t >= scheduler.timestep_end_step:
                        uncertanties.append(output.uncertainty.cpu())
                        scores.append(output.pred_epsilon.cpu())
                input = prev_noisy_sample
            if isinstance(scheduler, SchedulerUncertaintyMixin):
                # print('uncertanties:', len(uncertanties))
                samples_uncertanties.append(torch.stack(uncertanties, dim=1))
                samples_scores.append(torch.stack(scores, dim=1))
            input = uvit_ae.decode(input)
        
        gen_images = (input / 2 + 0.5).clamp(0, 1)

        # if isinstance(gen_images, np.ndarray):
        #     gen_images = torch.from_numpy(gen_images).permute(0, 3, 1, 2)
        #     if device is not None:
        #         gen_images = gen_images.to(device)
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
    if isinstance(scheduler, SchedulerUncertaintyMixin) and samples_uncertanties:
        results['uncertainty'] = torch.cat(samples_uncertanties, dim=0).cpu()
        results['score'] = torch.cat(samples_scores, dim=0).cpu()

    return results



_first_step_flip_grad = True
_max_width = -1
_max_height = -1
_gradients = []

def predict_model(model, sample, t_tensor, y):
    if isinstance(model, UNet2DModel):
        return model(sample, t_tensor).sample
    elif isinstance(model, UViTAE):
        return model(sample, t_tensor, y)
    else:
        return model(sample, t_tensor, y=y)[:, :3]

def flip_grad_method(model, sample, t_tensor, y):
    global _first_step_flip_grad, _max_width, _max_height, _gradients
    sample_f = torch.flip(sample, dims=[2])
    with torch.set_grad_enabled(True):
        eps = predict_model(model, sample, t_tensor, y)

        eps_f = predict_model(model, sample_f, t_tensor, y)

        eps_f_f = torch.flip(eps_f, dims=[2])

        loss = torch.nn.functional.mse_loss(eps, eps_f_f, reduction='mean')

        loss.backward()

    if _first_step_flip_grad:    
        _first_step_flip_grad = False
        _max_width, _max_height = max([x.shape[2] for x in _gradients]), max([x.shape[3] for x in _gradients])

    # upscale _gradients to max width and height
    # using nearest neighbor interpolation
    _gradients = [torch.nn.functional.interpolate(x, size=(_max_width, _max_height), mode='nearest') for x in _gradients]

    uncertainty = torch.cat(_gradients, dim=1).amax(dim=1, keepdim=True)

    
    _gradients = []
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    

    return eps, uncertainty




def _backward_hook(module, grad_input, grad_output):
    gradient = grad_output[0].amax(dim=1, keepdim=True)
    gradient = (gradient - torch.amin(gradient)) / (torch.amax(gradient) - torch.amin(gradient))
    _gradients.append(gradient)



@torch.no_grad()
def generate_samples_uvit_scheduler_class_conditioned_with_threshold(num_samples, batch_size, image_size, model: UViTAE, scheduler, num_classes: int | torch.Tensor, threshold: torch.Tensor, device=None, fid_evaluator=None, x_T: torch.Tensor | None = None, y: torch.Tensor | None = None, start_step: int = 0, num_steps: int | None = None, seed: int = 0, skip_seed: int = 1):
    """
    Generates samples using a UViTAE model conditioned with a scheduler and threshold.

    Args:
        num_samples (int): The total number of samples to generate.
        batch_size (int): The batch size for generating samples.
        image_size (int): The size of the input image.
        model (UViTAE): The UViTAE model used for generating samples.
        scheduler: The scheduler used for conditioning the generation process.
        num_classes (int | torch.Tensor): The number of classes or a tensor containing the class labels.
        threshold (torch.Tensor): The threshold tensor used for determining the update scores.
        device (str, optional): The device to use for computation. Defaults to None.
        fid_evaluator (object, optional): The fid_evaluator object for computing FID score. Defaults to None.
        x_T (torch.Tensor, optional): The input tensor for conditioning the generation process. Defaults to None.
        y (torch.Tensor, optional): The class labels tensor for conditioning the generation process. Defaults to None.
        start_step (int, optional): The starting step for the scheduler. Defaults to 0.
        num_steps (int | None, optional): The number of steps for the scheduler. Defaults to None.
        seed (int, optional): The seed value for random number generation. Defaults to 0.
        skip_seed (int, optional): The skip seed value for random number generation. Defaults to 1.

    Returns:
        dict: A dictionary containing the generated samples and other relevant information.
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
    if isinstance(scheduler, SchedulerUncertaintyMixin):
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
            input = torch.randn(batch_size, 4, image_size, image_size, device=device, dtype=torch.float32, generator=generator.manual_seed(seed + i_batch * skip_seed))
        x_t = input.cpu().clone()
        samples_x_t.append(x_t)
        if y is not None:
            y_slice = y[num_generated_samples:num_generated_samples + batch_size]
            y_slice = y_slice.to(device)
        elif isinstance(num_classes, int):
            y_slice = torch.randint(0, num_classes, (batch_size,), device=device, generator=generator.manual_seed(seed + i_batch * skip_seed))
        else:
            assert num_samples == num_classes.shape[0]
            y_slice = num_classes[num_generated_samples:num_generated_samples + batch_size]
            if y_slice.shape[0] < batch_size:
                input = input[:y_slice.shape[0]]
        samples_y.append(y_slice)
        if isinstance(scheduler, SchedulerUncertaintyMixin):
            scheduler.prompt_embeds = y_slice
        with torch.no_grad():
            for i_t, t in enumerate(scheduler.timesteps):
                t = t.item()
                t_1 = scheduler.timesteps[i_t - 1].item() if i_t > 0 else 0 #t-1
                t_tensor = torch.full((y_slice.shape[0],), t, device=device, dtype=torch.long)
                noisy_residual = model(input, t_tensor, y_slice)
                output = scheduler.step(noisy_residual, t, input)
                prev_noisy_sample = output.prev_sample
                beta_t_1 = scheduler.betas[t_1]
                alpha_hat_t = scheduler.alphas_cumprod[i_t]

                if isinstance(scheduler, SchedulerUncertaintyMixin):
                    # if scheduler.timestep_after_step >= t >= scheduler.timestep_end_step and (start_step + num_steps >= i >= start_step):
                    if ((start_step + num_steps) > i_t >= start_step):
                        # results = []
                        # beta_t_1 = scheduler.betas[t_1]
                        noisy_residual.requires_grad = True
                        input.requires_grad = False
                        t_tensor.requires_grad = False
                        y_slice.requires_grad = False
                        # print('Step #:', i)
                        pred_epsilons = []
                        with torch.set_grad_enabled(True):
                            pred_x_0 = (input - sqrt(1 - alpha_hat_t) * noisy_residual) / sqrt(alpha_hat_t)
                            for _ in range(5):
                                x_hat_t = sqrt(alpha_hat_t) * pred_x_0 + sqrt(1 - alpha_hat_t) * torch.randn_like(prev_noisy_sample)
                                pred_epsilon = predict_model(model, x_hat_t, t_tensor, y_slice)
                                pred_epsilons.append(pred_epsilon)
                            pred_epsilons = torch.stack(pred_epsilons, dim=0)
                            # print('pred_epsilons:', pred_epsilons.shape)
                            # print('noisy_residual:', noisy_residual.shape)
                            # pixel_wise_uncertainty = (pred_epsilons - noisy_residual.unsqueeze(0)).pow(2).mean(dim=0)
                            pixel_wise_uncertainty = pred_epsilons.var(dim=0)
                            uncertainty = pixel_wise_uncertainty.mean(dim=0).sum()
                            uncertainty.backward()
                        update_scores = noisy_residual.grad * -1
                        thresholded_map = output.uncertainty > threshold[i_t]
                        thresholded_map = thresholded_map.float()
                        # update_pixels = torch.stack(results, dim=0).mean(dim=0)
                        # print('thresholded_map:', thresholded_map.shape)
                        # print('update_pixels:', update_pixels.shape)
                        # noisy_residual= noisy_residual * (1 - thresholded_map) + noisy_residual * thresholded_map * update_scores
                        noisy_residual= noisy_residual + thresholded_map * update_scores
                        output = scheduler.step(noisy_residual, t, input)
                        prev_noisy_sample = output.prev_sample
                input = prev_noisy_sample
        gen_images = model.decode(input)
        gen_images = (gen_images / 2 + 0.5).clamp(0, 1)
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


@torch.no_grad()
def generate_samples_model_scheduler_class_conditioned_with_percentile(num_samples, batch_size, image_size, model, scheduler, num_classes: int | torch.Tensor, percentile: float, device=None, fid_evaluator=None, x_T=None, start_step: int = 0, num_steps: int | None = None, seed: int = 0):
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
    # assert threshold.shape[0] == scheduler.timesteps.shape[0], f'{threshold.shape=} {scheduler.timesteps.shape=}'
    if isinstance(scheduler, SchedulerUncertaintyMixin):
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
        if isinstance(num_classes, int):
            y = torch.randint(0, num_classes, (batch_size,), device=device, generator=generator.manual_seed(seed + i_batch))
        else:
            assert num_samples == num_classes.shape[0]
            y = num_classes[num_generated_samples:num_generated_samples + batch_size]
            if y.shape[0] < batch_size:
                input = input[:y.shape[0]]
        samples_y.append(y)
        if isinstance(scheduler, SchedulerUncertaintyMixin):
            scheduler.prompt_embeds = y
        with torch.no_grad():
            for i, t in enumerate(scheduler.timesteps):
                t = t.item()
                t_1 = scheduler.timesteps[i - 1].item() if i > 0 else 0 #t-1
                t_tensor = torch.full((y.shape[0],), t, device=device, dtype=torch.long)
                noisy_residual = model(input, t_tensor, y=y)[:, :3]
                output = scheduler.step(noisy_residual, t, input)
                prev_noisy_sample = output.prev_sample
                beta_t_1 = scheduler.betas[t_1]
                alpha_hat_t = scheduler.alphas_cumprod[i]

                if isinstance(scheduler, SchedulerUncertaintyMixin):
                    if scheduler.timestep_after_step >= t >= scheduler.timestep_end_step and (start_step + num_steps >= i >= start_step):
                        # results = []
                        # beta_t_1 = scheduler.betas[t_1]
                        noisy_residual.requires_grad = True
                        input.requires_grad = False
                        t_tensor.requires_grad = False
                        y.requires_grad = False
                        print('Step #:', i)
                        pred_epsilons = []
                        with torch.set_grad_enabled(True):
                            pred_x_0 = (input - sqrt(1 - alpha_hat_t) * noisy_residual) / sqrt(alpha_hat_t)
                            for _ in range(5):
                                x_hat_t = sqrt(alpha_hat_t) * pred_x_0 + sqrt(1 - alpha_hat_t) * torch.randn_like(prev_noisy_sample)
                                pred_epsilon = predict_model(model, x_hat_t, t_tensor, y)
                                pred_epsilons.append(pred_epsilon)
                            pred_epsilons = torch.stack(pred_epsilons, dim=0)
                            pixel_wise_uncertainty = pred_epsilons.std(dim=0)
                            uncertainty = pixel_wise_uncertainty.mean(dim=0).sum()
                            uncertainty.backward()
                        update_scores = noisy_residual.grad * 1
                        uncertainty_shape = pixel_wise_uncertainty.shape
                        thresholded_map = pixel_wise_uncertainty > torch.quantile(pixel_wise_uncertainty.flatten(1), percentile, dim=1, keepdim=True).view(uncertainty_shape[0], *([1] * (len(uncertainty_shape) - 1)))

                        thresholded_map = output.uncertainty > thresholded_map
                        thresholded_map = thresholded_map.float()
                        # update_pixels = torch.stack(results, dim=0).mean(dim=0)
                        # print('thresholded_map:', thresholded_map.shape)
                        # print('update_pixels:', update_pixels.shape)
                        noisy_residual= noisy_residual * (1 - thresholded_map) + noisy_residual * thresholded_map * update_scores
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

    results = {'y': torch.cat(tensors=samples_y, dim=0).cpu(), 
               'x_t': torch.cat(samples_x_t, dim=0).cpu(),
               'timestep': scheduler.timesteps,
               'gen_images': torch.cat(samples_gen_images, dim=0).cpu()}
    if fid_evaluator is not None:
        results['fid'] = fid_evaluator.compute()

    return results


@torch.no_grad()
def generate_samples_and_compute_fid_model_scheduler_class_conditioned(num_samples, batch_size, fid_evaluator, image_size, model, scheduler, num_classes: int | torch.Tensor, device=None, return_intermediates=False):
    """
    Generates samples using the given pipeline and computes the Fréchet Inception Distance (FID) score.

    Args:
        num_samples (int): The total number of samples to generate.
        batch_size (int): The batch size to use for generating samples.
        image_size (int): The size of the generated images (assumed to be square).
        fid_evaluator: The FID evaluator object used to compute the FID score.
        pipeline: The pipeline object used for generating samples.
        device: The device to use for generating samples (default: None).

    Returns:
        float: The computed FID score.
    """

    num_generated_samples = 0
    first = True
    print('device:', device)
    while num_samples > num_generated_samples:
        tqdm.tqdm.write(f"Generated samples: {num_generated_samples} / {num_samples}")

        input = torch.randn(batch_size, 3, image_size, image_size, device=device)
        x_t = input.cpu().clone()
        if isinstance(num_classes, int):
            y = torch.randint(0, num_classes, (batch_size,), device=device)
        else:
            assert num_samples == num_classes.shape[0]
            y = num_classes[num_generated_samples:num_generated_samples + batch_size]
            if y.shape[0] < batch_size:
                input = input[:y.shape[0]]

        if isinstance(scheduler, SchedulerUncertaintyMixin):
            scheduler.prompt_embeds = y
            if return_intermediates:
                uncertanties = []
        if return_intermediates:
            scores = []
        with torch.no_grad():
            for t in scheduler.timesteps:
                t = t.item()
                t_tensor = torch.full((y.shape[0],), t, device=device, dtype=torch.long)
                noisy_residual = model(input, t_tensor, y=y)[:, :3]
                output = scheduler.step(noisy_residual, t, input)
                prev_noisy_sample = output.prev_sample
                if isinstance(scheduler, SchedulerUncertaintyMixin) and return_intermediates:
                    uncertanties.append(output.uncertainty.cpu())
                if return_intermediates:
                    scores.append(output.pred_epsilon.cpu())
                input = prev_noisy_sample

        gen_images = (input / 2 + 0.5).clamp(0, 1)

        # if isinstance(gen_images, np.ndarray):
        #     gen_images = torch.from_numpy(gen_images).permute(0, 3, 1, 2)
        #     if device is not None:
        #         gen_images = gen_images.to(device)
        num_generated_samples += gen_images.shape[0]

        if first:
            print(gen_images.shape)
            print('min:', gen_images.amin())
            print('max:', gen_images.amax())
            first = False

        gen_images = gen_images * 255.0
        gen_images = gen_images.round()
        gen_images = gen_images.to(torch.uint8)

        fid_evaluator.update(gen_images, real=False)

        del gen_images

    fid_score = fid_evaluator.compute()
    if not return_intermediates:
        return fid_score
    else:
        intermediates = {'y': y.cpu(), 'x_t': x_t,
                         'score': torch.stack(scores, dim=0).cpu()}
        if isinstance(scheduler, SchedulerUncertaintyMixin):
            intermediates['uncertainty'] = torch.stack(uncertanties, dim=0).cpu()
        return fid_score, intermediates
    


@torch.no_grad()
def generate_samples_and_compute_fid_uvit_scheduler_class_conditioned(num_samples, batch_size, fid_evaluator, image_size, model: UViTAE, scheduler, num_classes: int | torch.Tensor, device=None, return_intermediates=False):
    """
    Generates samples using the given pipeline and computes the Fréchet Inception Distance (FID) score.

    Args:
        num_samples (int): The total number of samples to generate.
        batch_size (int): The batch size to use for generating samples.
        image_size (int): The size of the generated images (assumed to be square).
        fid_evaluator: The FID evaluator object used to compute the FID score.
        pipeline: The pipeline object used for generating samples.
        device: The device to use for generating samples (default: None).

    Returns:
        float: The computed FID score.
    """

    num_generated_samples = 0
    first = True
    print('device:', device)
    while num_samples > num_generated_samples:
        tqdm.tqdm.write(f"Generated samples: {num_generated_samples} / {num_samples}")

        input = torch.randn(batch_size, 4, image_size, image_size, device=device)
        x_t = input.cpu().clone()
        if isinstance(num_classes, int):
            y = torch.randint(0, num_classes, (batch_size,), device=device)
        else:
            assert num_samples == num_classes.shape[0]
            y = num_classes[num_generated_samples:num_generated_samples + batch_size]
            if y.shape[0] < batch_size:
                input = input[:y.shape[0]]

        if isinstance(scheduler, SchedulerUncertaintyMixin):
            scheduler.prompt_embeds = y
            if return_intermediates:
                uncertanties = []
        if return_intermediates:
            scores = []
        with torch.no_grad():
            for t in scheduler.timesteps:
                t = t.item()
                t_tensor = torch.full((y.shape[0],), t, device=device, dtype=torch.long)
                noisy_residual = model(input, t_tensor, y)
                output = scheduler.step(noisy_residual, t, input)
                prev_noisy_sample = output.prev_sample
                if isinstance(scheduler, SchedulerUncertaintyMixin) and return_intermediates:
                    uncertanties.append(output.uncertainty.cpu())
                if return_intermediates:
                    scores.append(output.pred_epsilon.cpu())
                input = prev_noisy_sample

        gen_images = (input / 2 + 0.5).clamp(0, 1)

        # if isinstance(gen_images, np.ndarray):
        #     gen_images = torch.from_numpy(gen_images).permute(0, 3, 1, 2)
        #     if device is not None:
        #         gen_images = gen_images.to(device)
        num_generated_samples += gen_images.shape[0]

        if first:
            print(gen_images.shape)
            print('min:', gen_images.amin())
            print('max:', gen_images.amax())
            first = False

        gen_images = gen_images * 255.0
        gen_images = gen_images.round()
        gen_images = gen_images.to(torch.uint8)

        fid_evaluator.update(gen_images, real=False)

        del gen_images

    fid_score = fid_evaluator.compute()
    if not return_intermediates:
        return fid_score
    else:
        intermediates = {'y': y.cpu(), 'x_t': x_t,
                         'score': torch.stack(scores, dim=0).cpu()}
        if isinstance(scheduler, SchedulerUncertaintyMixin):
            intermediates['uncertainty'] = torch.stack(uncertanties, dim=0).cpu()
        return fid_score, intermediates



@torch.no_grad()
def generate_samples_and_compute_fid_model_scheduler(num_samples, batch_size, fid_evaluator, image_size, model, scheduler, device=None):
    """
    Generates samples using the given pipeline and computes the Fréchet Inception Distance (FID) score.

    Args:
        num_samples (int): The total number of samples to generate.
        batch_size (int): The batch size to use for generating samples.
        image_size (int): The size of the generated images (assumed to be square).
        fid_evaluator: The FID evaluator object used to compute the FID score.
        pipeline: The pipeline object used for generating samples.
        device: The device to use for generating samples (default: None).

    Returns:
        float: The computed FID score.
    """

    num_generated_samples = 0
    first = True
    print('device:', device)
    while num_samples > num_generated_samples:
        tqdm.tqdm.write(f"Generated samples: {num_generated_samples} / {num_samples}")

        input = torch.randn(batch_size, 3, image_size, image_size, device=device)

        with torch.no_grad():
            for t in scheduler.timesteps:
                t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
                noisy_residual = model(input, t_tensor)[:, :3]
                prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
                input = prev_noisy_sample

        gen_images = (input / 2 + 0.5).clamp(0, 1)

        # if isinstance(gen_images, np.ndarray):
        #     gen_images = torch.from_numpy(gen_images).permute(0, 3, 1, 2)
        #     if device is not None:
        #         gen_images = gen_images.to(device)
        num_generated_samples += gen_images.shape[0]

        if first:
            print(gen_images.shape)
            print('min:', gen_images.amin())
            print('max:', gen_images.amax())
            first = False

        gen_images = gen_images * 255.0
        gen_images = gen_images.round()
        gen_images = gen_images.to(torch.uint8)

        fid_evaluator.update(gen_images, real=False)

        del gen_images

    fid_score = fid_evaluator.compute()

    return fid_score