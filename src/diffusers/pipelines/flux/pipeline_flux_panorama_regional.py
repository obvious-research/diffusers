# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
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
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from .pipeline_flux import calculate_shift, retrieve_timesteps
from ... import FluxPanoramaPipeline, FluxRegionalPipeline
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FluxIPAdapterMixin, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, FluxTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import FluxPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPanoramaRegionalPipeline

        >>> pipe = FluxPanoramaRegionalPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""


class FluxPanoramaRegionalPipeline(FluxPanoramaPipeline, FluxRegionalPipeline):

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        regional_area: Optional[torch.Tensor]=None,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        super().check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length)

        if regional_area is not None and regional_area.shape != (height, width):
            raise ValueError(f"`regional_area` must be of shape {(height, width)} (same as `height` and `width`) but is {regional_area.shape}.")


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        min_overlap_x: Optional[int] = 128,
        min_overlap_y: Optional[int] = 128,
        height_generation: Optional[int] = 1024,
        width_generation: Optional[int] = 1024,
        sigmas: Optional[List[float]] = None,
        regional_area: Optional[torch.FloatTensor] = None,
        regional_prompts: Optional[Union[str, List[str],List[Tuple[str, str]]]] = None,
        regional_base_prompt_trick: bool = True,
        mask_inject_steps: int = 5,
        base_ratio: float = 0.25,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            regional_area=regional_area,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
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
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
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

        latent_image_ids = self._prepare_latent_image_ids(
            batch_size,
            int(height_generation) // (self.vae_scale_factor * 2),
            int(width_generation) // (self.vae_scale_factor * 2),
            device,
            prompt_embeds.dtype
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        latent_views = self.calc_tiles_min_overlap(
            height // self.vae_scale_factor, width // self.vae_scale_factor,
            tile_height=height_generation // self.vae_scale_factor,
            tile_width=width_generation // self.vae_scale_factor,
            min_overlap_x=min_overlap_x // self.vae_scale_factor,
            min_overlap_y=min_overlap_y // self.vae_scale_factor,
        )

        latent_image_ids = self._prepare_latent_image_ids(
            batch_size,
            int(height_generation) // (self.vae_scale_factor * 2),
            int(width_generation) // (self.vae_scale_factor * 2),
            device,
            prompt_embeds.dtype
        )

        count = torch.zeros_like(latents)  # Tracks number of views for each latent position
        value = torch.zeros_like(latents)  # Accumulates the latent values for averaging

        self.set_progress_bar_config(position=0)

        # TODO: split by regional_area  and compute mask for each area
        if regional_area is not None and regional_prompts is not None:

            regional_prompts_views = self.calc_tiles_min_overlap(
                height, width,
                tile_height=height_generation,
                tile_width=width_generation,
                min_overlap_x=min_overlap_x,
                min_overlap_y=min_overlap_y,
                add_weight_matrix=False,
            )

            regional_prompt_embeds = []
            for regional_prompt in regional_prompts:
                regional_prompt_embed, regional_pooled_prompt_embed, regional_text_id = self.encode_prompt(
                     prompt=regional_prompt[1] if regional_base_prompt_trick else regional_prompt,
                     prompt_2=regional_prompt[1] if regional_base_prompt_trick else regional_prompt,
                     prompt_embeds=None,
                     pooled_prompt_embeds=None,
                     device=device,
                     num_images_per_prompt=num_images_per_prompt,
                     max_sequence_length=512,
                     lora_scale=None
                )
                regional_prompt_embeds.append(regional_prompt_embed)

            H, W = height_generation // (self.vae_scale_factor * 2), width_generation // (self.vae_scale_factor * 2)
            hidden_seq_len = H * W

            computed_regional_embeds=[]
            computed_regional_attention_mask=[]

            computed_prompt_embeds = []
            computed_pooled_prompt_embeds = []

            # TODO: reimplement view_batch_size based on the number of regions

            for j, rv in enumerate(regional_prompts_views):
                regional_mask_view = regional_area[rv.coords.top:rv.coords.bottom, rv.coords.left:rv.coords.right]

                interpolated_mask = torch.nn.functional.interpolate(
                    regional_mask_view[None, None, :, :].half(),
                    (H, W),
                    mode='nearest-exact'
                ).flatten().unsqueeze(1).to(dtype=torch.int8)

                # find all unique values in the mask
                unique_values = torch.unique(interpolated_mask)

                local_regional_embeds = []
                masks = []
                for region_id in unique_values:
                    binary_mask = torch.where(interpolated_mask == region_id, 1, 0)
                    binary_mask = binary_mask.repeat(1, regional_prompt_embeds[region_id].size(1))
                    masks.append(binary_mask)
                    local_regional_embeds.append(regional_prompt_embeds[region_id])

                local_regional_embeds = torch.cat(local_regional_embeds, dim=1)
                computed_regional_embeds.append(local_regional_embeds)
                encoder_seq_len = local_regional_embeds.shape[1]


                if regional_base_prompt_trick:
                    selec_prompt = [regional_prompts[i][0] for i in unique_values]
                    regional_base_prompt = prompt + ' with ' + ' and '.join(selec_prompt)

                    regional_base_prompt_embed, regional_pooled_base_prompt_embed, regional_text_id = self.encode_prompt(
                        prompt=regional_base_prompt,
                        prompt_2=regional_base_prompt,
                        prompt_embeds=None,
                        pooled_prompt_embeds=None,
                        device=device,
                        num_images_per_prompt=num_images_per_prompt,
                        max_sequence_length=512,
                        lora_scale=None
                    )
                    computed_prompt_embeds.append(regional_base_prompt_embed)
                    computed_pooled_prompt_embeds.append(regional_pooled_base_prompt_embed)
                else:
                    computed_prompt_embeds.append(prompt_embeds)
                    computed_pooled_prompt_embeds.append(pooled_prompt_embeds)

                # initialize attention mask
                regional_attention_mask = torch.zeros(
                    (encoder_seq_len + hidden_seq_len, encoder_seq_len + hidden_seq_len),
                    device=masks[0].device,
                    dtype=torch.bool
                )
                num_of_regions = len(masks)
                each_prompt_seq_len = encoder_seq_len // num_of_regions

                # initialize self-attended mask
                self_attend_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=masks[0].device, dtype=torch.bool)

                # initialize union mask
                union_masks = torch.zeros((hidden_seq_len, hidden_seq_len), device=masks[0].device, dtype=torch.bool)

                # handle each mask
                for j in range(num_of_regions):
                    # txt attends to itself
                    regional_attention_mask[j * each_prompt_seq_len:(j + 1) * each_prompt_seq_len,
                    j * each_prompt_seq_len:(j + 1) * each_prompt_seq_len] = True

                    # txt attends to corresponding regional img
                    regional_attention_mask[j * each_prompt_seq_len:(j + 1) * each_prompt_seq_len, encoder_seq_len:] = masks[j].transpose(-1, -2)

                    # regional img attends to corresponding txt
                    regional_attention_mask[encoder_seq_len:, j * each_prompt_seq_len:(j + 1) * each_prompt_seq_len] = masks[j]

                    # regional img attends to corresponding regional img
                    img_size_masks = masks[j][:, :1].repeat(1, hidden_seq_len)
                    img_size_masks_transpose = img_size_masks.transpose(-1, -2)
                    self_attend_masks = torch.logical_or(
                        self_attend_masks,
                        torch.logical_and(
                            img_size_masks,
                            img_size_masks_transpose
                        )
                    )

                    # update union
                    union_masks = torch.logical_or(
                        union_masks,
                        torch.logical_or(
                            img_size_masks,
                            img_size_masks_transpose
                        )
                    )

                background_masks = torch.logical_not(union_masks)

                background_and_self_attend_masks = torch.logical_or(background_masks, self_attend_masks)

                regional_attention_mask[encoder_seq_len:, encoder_seq_len:] = background_and_self_attend_masks

                computed_regional_attention_mask.append(regional_attention_mask)

        if 'single_inject_blocks_interval' not in self.joint_attention_kwargs:
            self.joint_attention_kwargs['single_inject_blocks_interval'] = len(self.transformer.single_transformer_blocks)
        if 'double_inject_blocks_interval' not in self.joint_attention_kwargs:
            self.joint_attention_kwargs['double_inject_blocks_interval'] = len(self.transformer.transformer_blocks)


        views_inputs = list(zip(latent_views, computed_regional_embeds, computed_regional_attention_mask, computed_prompt_embeds, computed_pooled_prompt_embeds))

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for j, t in enumerate(timesteps):

                if self.interrupt:
                    continue

                self._current_timestep = t

                # Reset count and value for this timestep
                count.zero_()
                value.zero_()

                # Process each batch of views
                for j, (tile, regional_embeds, attention_mask, prompt_embeds, pooled_prompt_embeds) in tqdm.tqdm(
                     enumerate(views_inputs),
                     total=len(latent_views),
                     position=1,
                     leave=False
                ):
                    if j < mask_inject_steps:
                        chosen_prompt_embeds = regional_embeds
                        self.joint_attention_kwargs['base_ratio'] = base_ratio
                        self.joint_attention_kwargs['regional_attention_mask'] = attention_mask.to(device)
                    else:
                        chosen_prompt_embeds = prompt_embeds
                        self.joint_attention_kwargs['base_ratio'] = None
                        self.joint_attention_kwargs['regional_attention_mask'] = None

                    # Get latents for the current views
                    h_start, h_end = tile.coords.top, tile.coords.bottom
                    w_start, w_end = tile.coords.left, tile.coords.right
                    latents_for_view = latents[:, :, h_start:h_end, w_start:w_end]

                    latents_for_view = self._pack_latents(
                        latents_for_view,
                        latents_for_view.shape[0],
                        num_channels_latents,
                        height_generation // self.vae_scale_factor,
                        width_generation // self.vae_scale_factor
                    )

                    if image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(1).to(latents.dtype) # was t.expand(view_batch_size).to(latents.dtype)

                    with self.transformer.cache_context("cond"):
                        noise_pred = self.transformer(
                            hidden_states=latents_for_view,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=chosen_prompt_embeds,
                            encoder_hidden_states_base=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]

                    if do_true_cfg:
                        if negative_image_embeds is not None:
                            self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds

                        with self.transformer.cache_context("uncond"):
                            neg_noise_pred = self.transformer(
                                hidden_states=latents_for_view,
                                timestep=timestep / 1000,
                                guidance=guidance,
                                pooled_projections=negative_pooled_prompt_embeds,
                                encoder_hidden_states=choosen_negative_prompt_embeds,
                                encoder_hidden_states_base=negative_prompt_embeds,
                                txt_ids=negative_text_ids,
                                img_ids=latent_image_ids,
                                joint_attention_kwargs=self.joint_attention_kwargs,
                                return_dict=False,
                            )[0]
                            noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    noise_pred_unpacked = self._unpack_latents(
                        noise_pred,
                        height_generation,
                        width_generation,
                        self.vae_scale_factor
                    )

                    # Accumulate the denoised result
                    value[:, :, h_start:h_end, w_start:w_end] += noise_pred_unpacked * tile.weight
                    count[:, :, h_start:h_end, w_start:w_end] += tile.weight

                # Average the noise_pred using the accumulated values and counts
                # compute the previous noisy sample x_t -> x_t-1
                noise_pred_global = torch.where(count > 0, value / count, value)
                latents = self.scheduler.step(noise_pred_global, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, j, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            latents = self._pack_latents(latents, batch_size, num_channels_latents,
                                      2 * (int(height) // (self.vae_scale_factor * 2)),
                                      2 * (int(width) // (self.vae_scale_factor * 2)))
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
