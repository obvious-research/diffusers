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

from typing import Any, Callable, Dict, List, Optional, Union

import math
import numpy as np
import torch
import tqdm

from diffusers import FluxPipeline


from .pipeline_flux import retrieve_timesteps
from .pipeline_flux_regional import calculate_shift
from ...image_processor import PipelineImageInput
from ...utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
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
        >>> from diffusers import FluxPanoramaPipeline

        >>> pipe = FluxPanoramaPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""

class TBLR:

    def __init__(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def __eq__(self, other):
        return (
            self.top == other.top
            and self.bottom == other.bottom
            and self.left == other.left
            and self.right == other.right
        )


class Tile:

    def __init__(self, coords: TBLR, overlap: TBLR, weight=None):
        self.coords = coords
        self.overlap = overlap
        self.weight = weight

    @property
    def height(self):
        return self.coords.bottom - self.coords.top

    @property
    def width(self):
        return self.coords.right - self.coords.left

    def __eq__(self, other):
        return self.coords == other.coords and self.overlap == other.overlap and self.weight == other.weight


class FluxPanoramaPipeline(FluxPipeline):

    @staticmethod
    def generate_weight_matrix(tile: Tile, device: torch.device):

        weight_matrix = torch.ones((tile.height, tile.width), device=device)

        if tile.overlap.left > 0:
            gradient_left = torch.linspace(0, 1, tile.overlap.left, device=device)
            weight_matrix[:, :tile.overlap.left] *= gradient_left.view(1, -1)
        if tile.overlap.top > 0:
            gradient_top = torch.linspace(0, 1, tile.overlap.top, device=device)
            weight_matrix[:tile.overlap.top, :] *= gradient_top.view(-1, 1)
        if tile.overlap.right > 0:
            gradient_right = torch.linspace(1, 0, tile.overlap.right, device=device)
            weight_matrix[:, -tile.overlap.right:] *= gradient_right.view(1, -1)
        if tile.overlap.bottom > 0:
            gradient_bottom = torch.linspace(1, 0, tile.overlap.bottom, device=device)
            weight_matrix[-tile.overlap.bottom:, :] *= gradient_bottom.view(-1, 1)

        return weight_matrix

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @staticmethod
    def calc_overlap(tiles: list[Tile], num_tiles_x: int, num_tiles_y: int) -> list[Tile]:
        """Calculate and update the overlap of a list of tiles.

        Args:
            tiles (list[Tile]): The list of tiles describing the locations of the respective `tile_images`.
            num_tiles_x: the number of tiles on the x axis.
            num_tiles_y: the number of tiles on the y axis.
        """

        def get_tile_or_none(idx_y: int, idx_x: int) -> Union[Tile, None]:
            if idx_y < 0 or idx_y > num_tiles_y or idx_x < 0 or idx_x > num_tiles_x:
                return None
            return tiles[idx_y * num_tiles_x + idx_x]

        for tile_idx_y in range(num_tiles_y):
            for tile_idx_x in range(num_tiles_x):
                cur_tile = get_tile_or_none(tile_idx_y, tile_idx_x)
                top_neighbor_tile = get_tile_or_none(tile_idx_y - 1, tile_idx_x)
                left_neighbor_tile = get_tile_or_none(tile_idx_y, tile_idx_x - 1)

                assert cur_tile is not None

                # Update cur_tile top-overlap and corresponding top-neighbor bottom-overlap.
                if top_neighbor_tile is not None:
                    cur_tile.overlap.top = max(0, top_neighbor_tile.coords.bottom - cur_tile.coords.top)
                    top_neighbor_tile.overlap.bottom = cur_tile.overlap.top

                # Update cur_tile left-overlap and corresponding left-neighbor right-overlap.
                if left_neighbor_tile is not None:
                    cur_tile.overlap.left = max(0, left_neighbor_tile.coords.right - cur_tile.coords.left)
                    left_neighbor_tile.overlap.right = cur_tile.overlap.left
        return tiles

    def calc_tiles_min_overlap(
            self,
            image_height: int,
            image_width: int,
            tile_height: int,
            tile_width: int,
            min_overlap_x: int = 0,
            min_overlap_y: int = 0,
            add_weight_matrix: bool = True,
    ) -> list[Tile]:
        """Calculate the tile coordinates for a given image shape under a simple tiling scheme with overlaps.

        Args:
            image_height (int): The image height in px.
            image_width (int): The image width in px.
            tile_height (int): The tile height in px. All tiles will have this height.
            tile_width (int): The tile width in px. All tiles will have this width.
            min_overlap_x (int): The target minimum overlap between adjacent tiles. If the tiles do not evenly cover the image
                shape, then the overlap will be spread between the tiles.
            min_overlap_y (int): The target minimum overlap between adjacent tiles. If the tiles do not evenly cover the image
                shape, then the overlap will be spread between the tiles.

        Returns:
            list[Tile]: A list of tiles that cover the image shape. Ordered from left-to-right, top-to-bottom.
        """

        assert min_overlap_y < tile_height
        assert min_overlap_x < tile_width

        # catches the cases when the tile size is larger than the images size and adjusts the tile size
        if image_width < tile_width:
            tile_width = image_width

        if image_height < tile_height:
            tile_height = image_height

        num_tiles_x = math.ceil((image_width - min_overlap_x) / (tile_width - min_overlap_x))
        num_tiles_y = math.ceil((image_height - min_overlap_y) / (tile_height - min_overlap_y))

        # tiles[y * num_tiles_x + x] is the tile for the y'th row, x'th column.
        tiles: list[Tile] = []

        # Calculate tile coordinates. (Ignore overlap values for now.)
        for tile_idx_y in range(num_tiles_y):
            top = (tile_idx_y * (image_height - tile_height)) // (num_tiles_y - 1) if num_tiles_y > 1 else 0
            bottom = top + tile_height

            for tile_idx_x in range(num_tiles_x):
                left = (tile_idx_x * (image_width - tile_width)) // (num_tiles_x - 1) if num_tiles_x > 1 else 0
                right = left + tile_width

                tile = Tile(
                    coords=TBLR(top=top, bottom=bottom, left=left, right=right),
                    overlap=TBLR(top=0, bottom=0, left=0, right=0),
                )

                tiles.append(tile)

        tiles = self.calc_overlap(tiles, num_tiles_x, num_tiles_y)

        # Generate the Cartesian product
        if add_weight_matrix:
            for tile in tiles:
                tile.weight = self.generate_weight_matrix(tile, device=self._execution_device)

        return tiles

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
        view_batch_size: Optional[int] = 1,
        min_overlap_x: Optional[int] = 128,
        min_overlap_y: Optional[int] = 128,
        height_generation: Optional[int] = 1024,
        width_generation: Optional[int] = 1024,
        sigmas: Optional[List[float]] = None,
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
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                True classifier-free guidance (guidance scale) is enabled when `true_cfg_scale` > 1 and
                `negative_prompt` is provided.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
                a model to generate images more aligned with `prompt` at the expense of lower image quality.

                Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to
                the [paper](https://huggingface.co/papers/2210.03142) to learn more.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
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
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

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
            guidance = guidance.expand(view_batch_size)
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
            min_overlap_y=min_overlap_y // self.vae_scale_factor
        )

        print('Number of views per diffusion step: ', len(latent_views))

        latent_image_ids = self._prepare_latent_image_ids(
            batch_size,
            int(height_generation) // (self.vae_scale_factor * 2),
            int(width_generation) // (self.vae_scale_factor * 2),
            device,
            prompt_embeds.dtype)

        views_batch = [latent_views[i: i + view_batch_size] for i in range(0, len(latent_views), view_batch_size)]

        count = torch.zeros_like(latents)  # Tracks number of views for each latent position
        value = torch.zeros_like(latents)  # Accumulates the latent values for averaging

        self.set_progress_bar_config(position=0)

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                # Reset count and value for this timestep
                count.zero_()
                value.zero_()

                # Process each batch of views
                for j, batch_view in tqdm.tqdm(enumerate(views_batch), total=len(views_batch), position=1, leave=False):

                    # Get latents for the current views
                    latents_for_view = torch.cat(
                        [
                            latents[:, :, tile.coords.top:tile.coords.bottom, tile.coords.left:tile.coords.right]
                            for tile in batch_view
                        ]
                    )

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
                    timestep = t.expand(view_batch_size).to(latents.dtype)

                    with self.transformer.cache_context("cond"):
                        noise_pred = self.transformer(
                            hidden_states=latents_for_view,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
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
                                encoder_hidden_states=negative_prompt_embeds,
                                txt_ids=negative_text_ids,
                                img_ids=latent_image_ids,
                                joint_attention_kwargs=self.joint_attention_kwargs,
                                return_dict=False,
                            )[0]
                        noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    noise_pred_unpacked = self._unpack_latents(noise_pred, height_generation, width_generation, self.vae_scale_factor)
                    # Accumulate the denoised results for each view
                    for noise_pred_local, tile in zip(
                            noise_pred_unpacked.chunk(view_batch_size), batch_view
                    ):
                        value[:, :, tile.coords.top:tile.coords.bottom, tile.coords.left:tile.coords.right] += noise_pred_local * tile.weight
                        count[:, :, tile.coords.top:tile.coords.bottom, tile.coords.left:tile.coords.right] += tile.weight

                # Average the noise_pred using the accumulated values and counts
                # compute the previous noisy sample x_t -> x_t-1
                noise_pred_global = torch.where(count > 0, value / count, value)
                latents = self.scheduler.step(noise_pred_global, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            latents = self._pack_latents(latents,  batch_size, num_channels_latents, 2 * (int(height) // (self.vae_scale_factor * 2)), 2 * (int(width) // (self.vae_scale_factor * 2)))
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
