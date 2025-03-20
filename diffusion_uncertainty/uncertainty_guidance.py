import torch
from torch import sqrt
from typing import Literal, Optional
import torch
from torch import Tensor


use_posterior = True


def get_uncertainty_guided_score_with_threshold(pred_epsilon, input, t_tensor, y, model, alpha_hat_t,  threshold, i, output, thresholded_map, update_scores, pred_epsilons, uncertainty, pred_x_0, num_samples=5):
    """
    Calculates the uncertainty-guided score with a threshold for a given set of inputs.

    Args:
        pred_epsilon (torch.Tensor): The predicted epsilon values.
        input (torch.Tensor): The input tensor.
        t_tensor (torch.Tensor): The t tensor.
        y (torch.Tensor): The y tensor.
        model: The model used for prediction.
        alpha_hat_t: The alpha_hat_t value.
        threshold (list): The list of threshold values.
        i (int): The index of the threshold value to use.
        output: The output tensor.
        thresholded_map: The thresholded map tensor.
        update_scores: The update scores tensor.
        pred_epsilons: The predicted epsilon values tensor.
        uncertainty: The uncertainty tensor.
        pred_x_0: The predicted x_0 tensor.
        num_samples (int, optional): The number of samples to use. Defaults to 5.

    Returns:
        torch.Tensor: The updated pred_epsilon tensor.
    """
    
    pred_epsilon.requires_grad = True
    input.requires_grad = False
    t_tensor.requires_grad = False
    print('Step #:', i)
    pred_epsilons = []
    with torch.set_grad_enabled(True):
        pred_x_0 = (input - sqrt(1 - alpha_hat_t) * pred_epsilon) / sqrt(alpha_hat_t)
        for _ in range(num_samples):
            x_hat_t = sqrt(alpha_hat_t) * pred_x_0 + sqrt(1 - alpha_hat_t) * torch.randn_like(pred_epsilon)
            pred_epsilon = predict_model(model, x_hat_t, t_tensor, y)
            pred_epsilons.append(pred_epsilon)
        pred_epsilons = torch.stack(pred_epsilons, dim=0)
        uncertainty = pred_epsilons.pow(2).mean(dim=0)
        uncertainty = uncertainty.mean(dim=0).sum()
        uncertainty.backward()
    update_scores = pred_epsilon.grad * -1
    thresholded_map = output.uncertainty > threshold[i]
    thresholded_map = thresholded_map.float()
    # update_pixels = torch.stack(results, dim=0).mean(dim=0)
    # print('thresholded_map:', thresholded_map.shape)
    # print('update_pixels:', update_pixels.shape)
    pred_epsilon= pred_epsilon * (1 - thresholded_map) + pred_epsilon * thresholded_map * update_scores

    return pred_epsilon

def get_uncertainty_guided_score_with_percentile(pred_epsilon: Tensor, input: Tensor, t_tensor: Tensor, y: Tensor, model, alpha_hat_t, percentile: float, model_type: Literal['unet', 'stable-diffusion', 'stable-diffusion-3', 'flux'], num_uncertainty_samples: int = 5, guidance_scale: Optional[float] = None, lr: float = 1.0, extra_diffusion_kwargs: dict | None = None) -> Tensor:
    """
    Calculates the uncertainty-guided score with percentile.

    Args:
        pred_epsilon (torch.Tensor): The predicted epsilon.
        input (torch.Tensor): The input tensor.
        t_tensor (torch.Tensor): The t tensor.
        y (torch.Tensor): The y tensor.
        model: The model used for prediction.
        alpha_hat_t: The alpha hat t value.
        percentile (float): The percentile value.
        num_uncertainty_samples (int, optional): The number of uncertainty samples. Defaults to 5.

    Returns:
        torch.Tensor: The updated pred_epsilon tensor.
    """
    
    pred_epsilon.requires_grad = not use_posterior
    input.requires_grad = False
    t_tensor.requires_grad = False
    y.requires_grad = False
    pred_epsilons_hat = []
    M = num_uncertainty_samples
    with torch.set_grad_enabled(not use_posterior):
        pred_x_0 = (input - sqrt(1 - alpha_hat_t) * pred_epsilon) / sqrt(alpha_hat_t)
        for _ in range(num_uncertainty_samples):
            x_hat_t = sqrt(alpha_hat_t) * pred_x_0 + sqrt(1 - alpha_hat_t) * torch.randn_like(pred_epsilon)
            if model_type == 'unet':
                pred_epsilon = predict_model(model, x_hat_t, t_tensor, y, extra_diffusion_kwargs=extra_diffusion_kwargs)
            elif model_type == 'stable-diffusion-3':
                t_tensor = t_tensor.reshape((-1,))
                pred_epsilon_hat = predict_model_stable_diffusion_3(model, x_hat_t, t_tensor, y, guidance_scale, extra_diffusion_kwargs=extra_diffusion_kwargs)
            elif model_type == 'flux':
                t_tensor = t_tensor.reshape((-1,)) / 1000.
                pred_epsilon_hat = predict_model_flux(model, x_hat_t, t_tensor, guidance_scale, extra_diffusion_kwargs)
            else:
                assert guidance_scale is not None
                pred_epsilon_hat = predict_model_stable_diffusion(model, x_hat_t, t_tensor, y, guidance_scale, extra_diffusion_kwargs=extra_diffusion_kwargs)
            pred_epsilons_hat.append(pred_epsilon_hat)
        if use_posterior:
            pred_epsilons_hat.append(pred_epsilon)
        pred_epsilons_hat = torch.stack(pred_epsilons_hat, dim=0)
        # pixel_wise_uncertainty = pred_epsilons_hat.pow(2).mean(dim=0)
        # pixel_wise_uncertainty = (pred_epsilons_hat - pred_epsilon.unsqueeze(0)).pow(2).mean(dim=0)        
        pixel_wise_uncertainty = torch.var(pred_epsilons_hat, dim=0)
        uncertainty = pixel_wise_uncertainty.mean(dim=0).sum()
        if not use_posterior: uncertainty.backward()
    uncertainty_shape = pixel_wise_uncertainty.shape
    print(f'{pixel_wise_uncertainty=}')
    
    thresholded_map = pixel_wise_uncertainty > torch.quantile(pixel_wise_uncertainty.to(torch.float32).flatten(1), percentile, dim=1, keepdim=True).view(uncertainty_shape[0], *([1] * (len(uncertainty_shape) - 1)))
    thresholded_map = thresholded_map.float()

    if use_posterior:
        inv_var: torch.Tensor = 1 / pixel_wise_uncertainty
        post_var_trace = (M * inv_var) + (1 / alpha_hat_t)
        post_precision = 1 / post_var_trace
        post_score = post_precision * (inv_var * pred_epsilon.sum(dim=0))
        final_epsilon = (pred_epsilon * (1 - thresholded_map)) + (thresholded_map * post_score)
    else:
        assert pred_epsilon.grad is not None
        update_scores = pred_epsilon.grad * 1

        # print('uncertainty_shape:', uncertainty_shape)
        # thresholded_map = pixel_wise_uncertainty > torch.quantile(pixel_wise_uncertainty.flatten(1), percentile, dim=1, keepdim=True).view(uncertainty_shape[0], *([1] * (len(uncertainty_shape) - 1)))
        # thresholded_map = thresholded_map.float()
        # final_epsilon = (pred_epsilon * (1 - thresholded_map)) + (thresholded_map * (pred_epsilon + update_scores * alpha_hat_t))
        final_epsilon = pred_epsilon + (lr * update_scores * thresholded_map)

    return final_epsilon 



def predict_model_stable_diffusion(model, sample, t_tensor, y, guidance_scale: float, extra_diffusion_kwargs=None):
    if extra_diffusion_kwargs is None:
        extra_diffusion_kwargs = dict()
    pred_noise = model(
        sample=sample,
        timestep=t_tensor,
        encoder_hidden_states=y,
        **extra_diffusion_kwargs,
    )[0]
    noise_pred_uncond, noise_pred_text = pred_noise.chunk(2)
    # return noise_pred_text
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def predict_model_stable_diffusion_3(model, sample, t_tensor, y, guidance_scale: float, extra_diffusion_kwargs=None):
    if extra_diffusion_kwargs is None:
        extra_diffusion_kwargs = dict()
    extra_diffusion_kwargs['return_dict'] = False
    print('t_tensor', t_tensor)
    pred_noise = model(
        hidden_states=sample,
        timestep=t_tensor,
        encoder_hidden_states=y,
        **extra_diffusion_kwargs,
    )[0]
    noise_pred_uncond, noise_pred_text = pred_noise.chunk(2)
    # return noise_pred_text
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def predict_model_flux(model, sample, t_tensor, extra_diffusion_kwargs=None):
    if extra_diffusion_kwargs is None:
        extra_diffusion_kwargs = dict()
    extra_diffusion_kwargs['return_dict'] = False
    print('t_tensor', t_tensor)
    pred_noise = model(
        hidden_states=sample,
        timestep=t_tensor,
        **extra_diffusion_kwargs,
    )[0]
    return pred_noise

def predict_model(model, sample, t_tensor, y, extra_diffusion_kwargs=None):
    if extra_diffusion_kwargs is None:
        extra_diffusion_kwargs = dict()
    return model(sample, t_tensor, y=y, **extra_diffusion_kwargs)[:, :3]
