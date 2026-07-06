# Copyright 2025 The HuggingFace Team. All rights reserved.
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


import torch

from ...models import UNet2DModel
from ...schedulers import DDPMScheduler
from ...utils import is_torch_xla_available
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from ...replica_exchange.acceptance import exchange_replicas, compute_tsr_constant


if is_torch_xla_available():
	import torch_xla.core.xla_model as xm

	XLA_AVAILABLE = True
else:
	XLA_AVAILABLE = False


class DDPMPipeline(DiffusionPipeline):
	r"""
	Pipeline for image generation.

	This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
	implemented for all pipelines (downloading, saving, running on a particular device, etc.).

	Parameters:
		unet ([`UNet2DModel`]):
			A `UNet2DModel` to denoise the encoded image latents.
		scheduler ([`SchedulerMixin`]):
			A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
			[`DDPMScheduler`], or [`DDIMScheduler`].
	"""

	model_cpu_offload_seq = "unet"

	def __init__(self, unet: UNet2DModel, scheduler: DDPMScheduler):
		super().__init__()
		self.register_modules(unet=unet, scheduler=scheduler)

	@torch.no_grad()
	def __call__(
		self,
		batch_size: int = 1,
		generator: torch.Generator | list[torch.Generator] | None = None,
		num_inference_steps: int = 1000,
		output_type: str | None = "pil",
		return_dict: bool = True,
		mu: float | None = None,
		tsr_lam: float | None = None, # NEW CODE
		replica_exchange: bool = False,
		n_replicas: int = 1,
	) -> ImagePipelineOutput | tuple:
		r"""
		The call function to the pipeline for generation.

		Args:
			batch_size (`int`, *optional*, defaults to 1):
				The number of images to generate.
			generator (`torch.Generator`, *optional*):
				A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
				generation deterministic.
			num_inference_steps (`int`, *optional*, defaults to 1000):
				The number of denoising steps. More denoising steps usually lead to a higher quality image at the
				expense of slower inference.
			output_type (`str`, *optional*, defaults to `"pil"`):
				The output format of the generated image. Choose between `PIL.Image` or `np.array`.
			return_dict (`bool`, *optional*, defaults to `True`):
				Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

		Example:

		```py
		>>> from diffusers import DDPMPipeline

		>>> # load model and scheduler
		>>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

		>>> # run pipeline in inference (sample random noise and denoise)
		>>> image = pipe().images[0]

		>>> # save image
		>>> image.save("ddpm_generated_image.png")
		```

		Returns:
			[`~pipelines.ImagePipelineOutput`] or `tuple`:
				If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
				returned where the first element is a list with the generated images
		"""
		if replica_exchange:
			batch_size_full=batch_size * n_replicas
			lam_ladder = torch.tensor([tsr_lam, tsr_lam+0.1, tsr_lam+0.2])
			
		else:
			batch_size_full = batch_size
			lam_ladder = torch.tensor([tsr_lam])
		alpha_bars = self.scheduler.alphas_cumprod.to(device=self.device, dtype=self.unet.dtype)

		# Sample gaussian noise to begin loop
		if isinstance(self.unet.config.sample_size, int):
			image_shape = (
				batch_size_full,
				self.unet.config.in_channels,
				self.unet.config.sample_size,
				self.unet.config.sample_size,
			)
		else:
			image_shape = (batch_size_full, self.unet.config.in_channels, *self.unet.config.sample_size)

		if self.device.type == "mps":
			# randn does not work reproducibly on mps
			image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
			image = image.to(self.device)
		else:
			image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

		# set step values
		self.scheduler.set_timesteps(num_inference_steps)

		for t in self.progress_bar(self.scheduler.timesteps):
			a_bar = alpha_bars[t]
			# 1. predict noise model_output
			model_output = self.unet(image, t).sample
			for tsr_idx, lam in enumerate(lam_ladder):
				tsr = compute_tsr_constant(lam, a_bar)
				model_output[batch_size * tsr_idx : batch_size * (tsr_idx+1)] *= tsr

			# 2. compute previous image: x_t -> x_t-1
			image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

			def compute_eps(model, x, t):
				model_output = model(x, t).sample
				return model_output

			image, _ = exchange_replicas(
				model=self.unet,
				xt=image,
				t=t,
				lam_ladder=lam_ladder,
				a_bar=a_bar,
				compute_eps=compute_eps
			)

			if XLA_AVAILABLE:
				xm.mark_step()

		image = (image / 2 + 0.5).clamp(0, 1)
		image = image.cpu().permute(0, 2, 3, 1).numpy()
		if output_type == "pil":
			image = self.numpy_to_pil(image)

		if not return_dict:
			return (image,)

		return ImagePipelineOutput(images=image)
