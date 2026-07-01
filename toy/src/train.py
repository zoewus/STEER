import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

from .model import MLP, UNet    
from .dataset import generate_gaussian_mixture, build_training_tensor
from .schedule import betas, alphas, alpha_bars, ts_desc
from .config_toy import DEVICE, TRAINING, TRAINING_IMG, DATASETS,DATASETS_IMG, N_DIFFUSION_STEPS, CKPT_DIR
from .model import build_model_for_dataset

device = DEVICE
n_steps = TRAINING["n_steps"]
batch_size = TRAINING["batch_size"]
lr = TRAINING["lr"]
ckpt_dir = CKPT_DIR
n_diffusion_steps = N_DIFFUSION_STEPS


torch.manual_seed(42)     
def train_model(dataset_name, existing_checkpoint=None, save_name=None, k=1.0, log_every=1000, is_ebm=False):
	"""Trains model according to a dataset defined by dataset_config"""
		
	x0_all = build_training_tensor(dataset_name)

	if dataset_name in DATASETS:
		training_setup = TRAINING
		
	elif dataset_name in DATASETS_IMG:
		training_setup = TRAINING_IMG

	n_steps = training_setup["n_steps"]
	batch_size = training_setup["batch_size"]
	lr = training_setup["lr"]

	loader = DataLoader(
		TensorDataset(x0_all),
		batch_size=batch_size,
		shuffle=True,
		drop_last=True,
		num_workers=0,      # keep simple; set >0 if you want
		pin_memory=True,    # helps H2D transfer
	)

	model = build_model_for_dataset(dataset_name).to(device)

	if existing_checkpoint is not None:
		print(f"Loading {existing_checkpoint}")
		checkpoint = torch.load(existing_checkpoint, map_location=device)
		model.load_state_dict(checkpoint)
		model.eval()

	opt = optim.Adam(model.parameters(), lr=lr)
	model.train()

	data_iter = iter(loader)

	for step in range(1, n_steps + 1):
		try:
			(x0,) = next(data_iter)
		except StopIteration:
			data_iter = iter(loader)  # reshuffles because shuffle=True
			(x0,) = next(data_iter)

		x0 = x0.to(device, non_blocking=True)

		t = torch.randint(0, n_diffusion_steps, (batch_size, 1), device=device)
		a_bar = alpha_bars[t]

		if dataset_name in DATASETS_IMG:
			a_bar = a_bar.view(-1, 1, 1, 1)

		noise = torch.randn_like(x0)
		xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise
		xt = xt.detach().requires_grad_(True)

		if is_ebm:
			energy = model(xt, t)          # shape [B, 1] or [B]
			# score = -grad_xt E(xt)
			score = -torch.autograd.grad(
				energy.sum(), xt, create_graph=True
			)[0]                           # shape [B, ...]
			# score should match -noise / sqrt(1 - a_bar)  (denoising score matching)
			target_score = -noise / torch.sqrt(1.0 - a_bar)
			loss = ((score - target_score) ** 2).mean()

		else:
			eps_hat = model(xt, t)
			loss = ((noise - eps_hat) ** 2).mean()

		opt.zero_grad()
		loss.backward()
		opt.step()

		if step % log_every == 0:
			print(f"dataset = {dataset_name} temperature={k} step={step} loss={loss.item():.4f}")

	save_name = dataset_name if save_name is None else save_name
	save_path = f"{ckpt_dir}/{save_name}_{k:.1f}.pt"
	torch.save(model.state_dict(), save_path)
	print(f"Trained model saved to {save_path}")
	return model


if __name__ == "__main__":	

	train_model(dataset_name="composed", is_ebm=False)