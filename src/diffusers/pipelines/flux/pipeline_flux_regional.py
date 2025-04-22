# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
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

import numpy as np
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from ..flux import FluxPipeline
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FluxIPAdapterMixin, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, FluxTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import (
    USE_PEFT_BACKEND,
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
        >>> from diffusers import FluxRegionalPipeline

        >>> pipe = FluxRegionalPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""


class FluxRegionalPipeline(FluxPipeline):

    @torch.inference_mode()
    def __call__(
            self,
            initial_latent: torch.FloatTensor = None,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            num_samples: int = 1,
            width: int = 1024,
            height: int = 1024,
            strength: float = 1.0,
            num_inference_steps: int = 25,
            timesteps: List[int] = None,
            mask_inject_steps: int = 5,
            guidance_scale: float = 5.0,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self._guidance_scale = guidance_scale

        device = self.transformer.device

        # 3. Define call parameters
        batch_size = num_samples if num_samples else prompt_embeds.shape[0]

        import matplotlib.pyplot as plt

        # encode base prompt
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
            max_sequence_length=512,
            lora_scale=None,
        )

        # define base mask and inputs
        base_mask = torch.ones((height, width), device=device,
                               dtype=self.transformer.dtype)  # base mask uses the whole image mask

        # encode regional prompts, define regional inputs
        regional_inputs = []
        if 'regional_prompts' in joint_attention_kwargs and 'regional_masks' in joint_attention_kwargs:
            for regional_prompt, regional_mask in zip(joint_attention_kwargs['regional_prompts'],
                                                      joint_attention_kwargs['regional_masks']):
                regional_prompt_embeds, regional_pooled_prompt_embeds, regional_text_ids = self.encode_prompt(
                    prompt=regional_prompt,
                    prompt_2=regional_prompt,
                    prompt_embeds=None,
                    pooled_prompt_embeds=None,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=512,
                    lora_scale=None,
                )
                regional_inputs.append((regional_mask, regional_prompt_embeds))

        ## prepare masks for regional control
        conds = []
        masks = []
        H, W = height // (self.vae_scale_factor * 2), width // (self.vae_scale_factor * 2)
        hidden_seq_len = H * W
        for mask, cond in regional_inputs:
            if mask is not None:  # resize regional masks to image size, the flatten is to match the seq len
                mask = torch.nn.functional.interpolate(mask[None, None, :, :], (H, W),
                                                       mode='nearest-exact').flatten().unsqueeze(1).repeat(1,
                                                                                                           cond.size(1))
            else:
                mask = torch.ones((H * W, cond.size(1))).to(device=cond.device)
            masks.append(mask)
            conds.append(cond)
        regional_embeds = torch.cat(conds, dim=1)
        encoder_seq_len = regional_embeds.shape[1]

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
        for i in range(num_of_regions):
            # txt attends to itself
            regional_attention_mask[i * each_prompt_seq_len:(i + 1) * each_prompt_seq_len,
            i * each_prompt_seq_len:(i + 1) * each_prompt_seq_len] = True

            # txt attends to corresponding regional img
            regional_attention_mask[i * each_prompt_seq_len:(i + 1) * each_prompt_seq_len, encoder_seq_len:] = masks[
                i].transpose(-1, -2)

            # regional img attends to corresponding txt
            regional_attention_mask[encoder_seq_len:, i * each_prompt_seq_len:(i + 1) * each_prompt_seq_len] = masks[i]

            # regional img attends to corresponding regional img
            img_size_masks = masks[i][:, :1].repeat(1, hidden_seq_len)
            img_size_masks_transpose = img_size_masks.transpose(-1, -2)
            self_attend_masks = torch.logical_or(self_attend_masks,
                                                 torch.logical_and(img_size_masks, img_size_masks_transpose))

            # update union
            union_masks = torch.logical_or(union_masks,
                                           torch.logical_or(img_size_masks, img_size_masks_transpose))

        background_masks = torch.logical_not(union_masks)

        background_and_self_attend_masks = torch.logical_or(background_masks, self_attend_masks)

        regional_attention_mask[encoder_seq_len:, encoder_seq_len:] = background_and_self_attend_masks
        ## done prepare masks for regional control

        print('regional_attention_mask.shape', regional_attention_mask.shape)
        plt.imshow(regional_attention_mask.cpu().numpy())
        plt.savefig('outputs/regional_attention_mask.png')

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            initial_latent,
        )

        # 4.Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        image_seq_len = (int(height) // (2 * self.vae_scale_factor)) * (int(width) // (2 * self.vae_scale_factor))

        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5.handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                if i < mask_inject_steps:
                    chosen_prompt_embeds = regional_embeds
                    base_ratio = joint_attention_kwargs['base_ratio']
                else:
                    chosen_prompt_embeds = prompt_embeds
                    base_ratio = None

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=chosen_prompt_embeds,
                    encoder_hidden_states_base=prompt_embeds,
                    base_ratio=base_ratio,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs={
                        'single_inject_blocks_interval': joint_attention_kwargs[
                            'single_inject_blocks_interval'] if 'single_inject_blocks_interval' in joint_attention_kwargs else len(
                            self.transformer.single_transformer_blocks),
                        'double_inject_blocks_interval': joint_attention_kwargs[
                            'double_inject_blocks_interval'] if 'double_inject_blocks_interval' in joint_attention_kwargs else len(
                            self.transformer.transformer_blocks),
                        'regional_attention_mask': regional_attention_mask if base_ratio is not None else None,
                    },
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)