import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .schedule import betas, alphas, alpha_bars, ts_desc

from .config_toy import DATASETS, DATASETS_IMG, DEVICE

torch.manual_seed(42)
device = DEVICE

class SinusoidalTimeEmbedding(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.dim = dim

	def forward(self, t):
		if t.dim() == 2:
			t = t.squeeze(-1)

		half = self.dim // 2
		freqs = torch.exp(
			-math.log(10000) * torch.arange(half, device=t.device) / half
		)
		args = t[:, None] * freqs[None, :]
		emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

		if self.dim % 2 == 1:
			emb = F.pad(emb, (0, 1))

		return emb


class MLP(nn.Module):
	def __init__(
		self,
		x_dim=1,
		hidden_dim=512,   # 128 -> 512
		time_dim=64,      # 32 -> 64
		n_layers=8,       # 4 -> 8
	):
		super().__init__()

		self.time_embed = SinusoidalTimeEmbedding(time_dim)
		self.input = nn.Linear(x_dim + time_dim, hidden_dim)
		self.layers = nn.ModuleList(
			[nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
		)
		self.output = nn.Linear(hidden_dim, x_dim)

	def forward(self, x, t):
		t_emb = self.time_embed(t)
		h = torch.cat([x, t_emb], dim=-1)
		h = F.silu(self.input(h))
		for layer in self.layers:
			h = h + F.silu(layer(h))
		return self.output(h)



class ResBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_dim=128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        C = base_channels
        # Encoder
        self.conv_in  = nn.Conv2d(in_channels, C, 3, padding=1)
        self.enc1     = ResBlock(C, time_dim)
        self.down1    = nn.Conv2d(C, C*2, 4, stride=2, padding=1)   # 28->14
        self.enc2     = ResBlock(C*2, time_dim)
        self.down2    = nn.Conv2d(C*2, C*4, 4, stride=2, padding=1) # 14->7
        # Bottleneck
        self.mid1     = ResBlock(C*4, time_dim)
        self.mid2     = ResBlock(C*4, time_dim)
        # Decoder
        self.up1      = nn.ConvTranspose2d(C*4, C*2, 4, stride=2, padding=1) # 7->14
        self.dec1     = ResBlock(C*4, time_dim)  # C*4 due to skip
        self.up2      = nn.ConvTranspose2d(C*4, C, 4, stride=2, padding=1)   # 14->28
        self.dec2     = ResBlock(C*2, time_dim)  # C*2 due to skip
        self.norm_out = nn.GroupNorm(8, C*2)
        self.conv_out = nn.Conv2d(C*2, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_embed(t))
        # Encoder
        h0 = F.silu(self.conv_in(x))
        h1 = self.enc1(h0, t_emb)          # (B, C, 28, 28)
        h2 = self.enc2(self.down1(h1), t_emb)  # (B, 2C, 14, 14)
        h  = self.mid2(self.mid1(self.down2(h2), t_emb), t_emb)  # (B, 4C, 7, 7)
        # Decoder with skip connections
        h  = self.dec1(torch.cat([self.up1(h), h2], dim=1), t_emb)
        h  = self.dec2(torch.cat([self.up2(h), h1], dim=1), t_emb)
        return self.conv_out(F.silu(self.norm_out(h)))
	
@torch.no_grad()
def load_model(path, dataset_name):
	"""Load trained model from checkpoint for a specific dataset."""
	model = build_model_for_dataset(dataset_name)
	state = torch.load(path, map_location=device)
	if dataset_name in DATASETS:
		model.load_state_dict(state, strict=True)
	else:
		model.load_state_dict(state, strict=False)

	model.eval()
	return model

@torch.no_grad()
def build_model_for_dataset(dataset_name):
	if dataset_name in DATASETS:
		return MLP().to(device)
	elif dataset_name in DATASETS_IMG:
		cfg = DATASETS_IMG[dataset_name]
		return UNet(in_channels=cfg["sample_shape"][0]).to(device)
	
