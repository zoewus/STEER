import torch
import math
from .config_toy import DEVICE, N_DIFFUSION_STEPS, TSR

torch.manual_seed(42)

device = DEVICE
n_diffusion_steps = N_DIFFUSION_STEPS

@torch.no_grad()
def cosine_beta_schedule(timesteps, s=0.008):
	"""Cosine noise schedule, taken from reduce reuse recycle code"""
	steps = timesteps + 1
	t = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
	alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
	alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
	betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
	return torch.clip(betas, 0, 0.999)

betas = cosine_beta_schedule(n_diffusion_steps).to(device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)  # alphā_t

ts_desc = torch.arange(n_diffusion_steps - 1, -1, -1, device=device)



