# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusion_uncertainty.uncertainty_guidance import get_uncertainty_guided_score_with_percentile

from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import EXAMPLE_DOC_STRING, StableDiffusion3Pipeline, retrieve_timesteps
from diffusers_private.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.doc_utils import replace_example_docstring
from diffusers.utils.import_utils import is_torch_xla_available


from ...utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

class StableDiffusion3PipelineUncertainty(StableDiffusion3Pipeline):

    scheduler: FlowMatchEulerDiscreteScheduler # for type hinting

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        start_step_uc: int = 0,
        num_steps_uc: int = 100,
        percentile: float = 0.95,
        lr: float | Callable[[int], float] = 0.001,
        num_uncertainty_samples: int = 5,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if start_step_uc <= i < (start_step_uc + num_steps_uc):
                    assert prompt_embeds is not None
                    if isinstance(lr, Callable):
                        lr_i = lr(i)
                    else:
                        lr_i = lr
                    extra_diffusion_args = dict(pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )
                    t_long = t.long()
                    noise_pred: torch.Tensor = get_uncertainty_guided_score_with_percentile(noise_pred, latent_model_input, t_long, prompt_embeds, self.transformer, alpha_hat_t=self.scheduler.sigmas[i], percentile=percentile, guidance_scale=guidance_scale, model_type='stable-diffusion-3', lr=lr_i, extra_diffusion_kwargs=extra_diffusion_args, num_uncertainty_samples=num_uncertainty_samples)

                    noise_pred = noise_pred.to(torch.float16)


                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)


    # @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    # def __call__(
    #     self,
    #     prompt: Union[str, List[str]] = None,
    #     height: Optional[int] = None,
    #     width: Optional[int] = None,
    #     num_inference_steps: int = 50,
    #     guidance_scale: float = 7.5,
    #     negative_prompt: Optional[Union[str, List[str]]] = None,
    #     num_images_per_prompt: Optional[int] = 1,
    #     eta: float = 0.0,
    #     generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    #     latents: Optional[torch.FloatTensor] = None,
    #     prompt_embeds: Optional[torch.FloatTensor] = None,
    #     negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    #     output_type: Optional[str] = "pil",
    #     return_dict: bool = True,
    #     callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    #     callback_steps: int = 1,
    #     cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    #     guidance_rescale: float = 0.0,
    #     clip_skip: Optional[int] = None,
    #     start_step_uc: int = 0,
    #     num_steps_uc: int = 100,
    #     percentile: float = 0.95,
    #     lr: float | Callable[[int], float] = 0.001,
    # ):
    #     r"""
    #     The call function to the pipeline for generation.

    #     Args:
    #         prompt (`str` or `List[str]`, *optional*):
    #             The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
    #         height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
    #             The height in pixels of the generated image.
    #         width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
    #             The width in pixels of the generated image.
    #         num_inference_steps (`int`, *optional*, defaults to 50):
    #             The number of denoising steps. More denoising steps usually lead to a higher quality image at the
    #             expense of slower inference.
    #         guidance_scale (`float`, *optional*, defaults to 7.5):
    #             A higher guidance scale value encourages the model to generate images closely linked to the text
    #             `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
    #         negative_prompt (`str` or `List[str]`, *optional*):
    #             The prompt or prompts to guide what to not include in image generation. If not defined, you need to
    #             pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
    #         num_images_per_prompt (`int`, *optional*, defaults to 1):
    #             The number of images to generate per prompt.
    #         eta (`float`, *optional*, defaults to 0.0):
    #             Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
    #             to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
    #         generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
    #             A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
    #             generation deterministic.
    #         latents (`torch.FloatTensor`, *optional*):
    #             Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
    #             generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
    #             tensor is generated by sampling using the supplied random `generator`.
    #         prompt_embeds (`torch.FloatTensor`, *optional*):
    #             Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
    #             provided, text embeddings are generated from the `prompt` input argument.
    #         negative_prompt_embeds (`torch.FloatTensor`, *optional*):
    #             Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
    #             not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
    #         output_type (`str`, *optional*, defaults to `"pil"`):
    #             The output format of the generated image. Choose between `PIL.Image` or `np.array`.
    #         return_dict (`bool`, *optional*, defaults to `True`):
    #             Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
    #             plain tuple.
    #         callback (`Callable`, *optional*):
    #             A function that calls every `callback_steps` steps during inference. The function is called with the
    #             following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
    #         callback_steps (`int`, *optional*, defaults to 1):
    #             The frequency at which the `callback` function is called. If not specified, the callback is called at
    #             every step.
    #         cross_attention_kwargs (`dict`, *optional*):
    #             A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
    #             [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
    #         guidance_rescale (`float`, *optional*, defaults to 0.0):
    #             Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
    #             Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
    #             using zero terminal SNR.
    #         clip_skip (`int`, *optional*):
    #             Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
    #             the output of the pre-final layer will be used for computing the prompt embeddings.
    #         start_step_uc (`int`, *optional*, defaults to 0):
    #             The step at which to start the uncertainty calculation. Only applies to the [`~schedulers.DDIMScheduler`].
    #         num_steps_uc (`int`, *optional*, defaults to 100):
    #             The number of steps to calculate the uncertainty. Only applies to the [`~schedulers.DDIMScheduler`].

    #     Examples:

    #     Returns:
    #         [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
    #             If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
    #             otherwise a `tuple` is returned where the first element is a list with the generated images and the
    #             second element is a list of `bool`s indicating whether the corresponding generated image contains
    #             "not-safe-for-work" (nsfw) content.
    #     """
    #     # 0. Default height and width to unet
    #     height = height or self.unet.config.sample_size * self.vae_scale_factor
    #     width = width or self.unet.config.sample_size * self.vae_scale_factor
    #     # to deal with lora scaling and other possible forward hooks

    #     # 1. Check inputs. Raise error if not correct
    #     self.check_inputs(
    #         prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
    #     )

    #     # 2. Define call parameters
    #     if prompt is not None and isinstance(prompt, str):
    #         batch_size = 1
    #     elif prompt is not None and isinstance(prompt, list):
    #         batch_size = len(prompt)
    #     else:
    #         batch_size = prompt_embeds.shape[0]

    #     device = self._execution_device
    #     # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    #     # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    #     # corresponds to doing no classifier free guidance.
    #     do_classifier_free_guidance = guidance_scale > 1.0

    #     # 3. Encode input prompt
    #     lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None

    #     prompt_embeds, negative_prompt_embeds = self.encode_prompt(
    #         prompt,
    #         device,
    #         num_images_per_prompt,
    #         do_classifier_free_guidance,
    #         negative_prompt,
    #         prompt_embeds=prompt_embeds,
    #         negative_prompt_embeds=negative_prompt_embeds,
    #         lora_scale=lora_scale,
    #         clip_skip=clip_skip,
    #     )
    #     # For classifier free guidance, we need to do two forward passes.
    #     # Here we concatenate the unconditional and text embeddings into a single batch
    #     # to avoid doing two forward passes
    #     if do_classifier_free_guidance:
    #         prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    #     # 4. Prepare timesteps
    #     self.scheduler.set_timesteps(num_inference_steps, device=device)
    #     timesteps = self.scheduler.timesteps

    #     # 5. Prepare latent variables
    #     num_channels_latents = self.unet.config.in_channels
    #     latents = self.prepare_latents(
    #         batch_size * num_images_per_prompt,
    #         num_channels_latents,
    #         height,
    #         width,
    #         prompt_embeds.dtype,
    #         device,
    #         generator,
    #         latents,
    #     )

    #     # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    #     extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    #     # 7. Denoising loop
    #     num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    #     with self.progress_bar(total=num_inference_steps) as progress_bar:
    #         for i, t in enumerate(timesteps):
    #             # expand the latents if we are doing classifier free guidance
    #             latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    #             latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

    #             noise_pred = self.unet(
    #                 latent_model_input,
    #                 t,
    #                 encoder_hidden_states=prompt_embeds,
    #                 cross_attention_kwargs=cross_attention_kwargs,
    #                 return_dict=False,
    #             )[0]

    #             # perform guidance
    #             if do_classifier_free_guidance:
    #                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #                 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    #             if do_classifier_free_guidance and guidance_rescale > 0.0:
    #                 # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
    #                 noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

    #             if start_step_uc <= i < (start_step_uc + num_steps_uc):
    #                 assert prompt_embeds is not None
    #                 if isinstance(lr, Callable):
    #                     lr_i = lr(i)
    #                 else:
    #                     lr_i = lr
    #                 noise_pred: torch.Tensor = get_uncertainty_guided_score_with_percentile(noise_pred, latent_model_input, t, prompt_embeds, self.unet, alpha_hat_t=self.scheduler.alphas_cumprod[t], percentile=percentile, guidance_scale=guidance_scale, model_type='stable-diffusion', lr=lr_i)
                    
                
    #             # compute the previous noisy sample x_t -> x_t-1
    #             output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
    #             # print(f'{output.keys()=}')
    #             latents = output["prev_sample"]

    #             # call the callback, if provided
    #             if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
    #                 progress_bar.update()
    #                 if callback is not None and i % callback_steps == 0:
    #                     step_idx = i // getattr(self.scheduler, "order", 1)
    #                     callback(step_idx, t, latents)

    #     if not output_type == "latent":
    #         image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
    #         image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    #     else:
    #         image = latents
    #         has_nsfw_concept = None

    #     if has_nsfw_concept is None:
    #         do_denormalize = [True] * image.shape[0]
    #     else:
    #         do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    #     image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    #     # Offload all models
    #     self.maybe_free_model_hooks()

    #     if not return_dict:
    #         return (image, has_nsfw_concept)

    #     return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
