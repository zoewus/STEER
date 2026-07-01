import torch 
import numpy as np
from scipy.stats import norm
from .config_toy import DATASETS_IMG, DATASETS

from torchvision import datasets as tv_datasets
from torchvision import transforms

torch.manual_seed(42)
np.random.seed(42)

@torch.no_grad()
def generate_gaussian_mixture(dataset_name, device='cpu'):
	"""Generates mixture of gaussians according to inputted means and standard deviations"""

	dataset_config = DATASETS[dataset_name]
	dataset_shape = dataset_config["dataset_shape"]
	n_samples = dataset_shape[0]

	means = torch.as_tensor(dataset_config['means'], dtype=torch.float32)
	stds = dataset_config['stds']
	
	n_gaussians = len(means)
	
	if isinstance(stds, (int, float)):
		stds = torch.full((n_gaussians,), float(stds))
	else:
		stds = torch.as_tensor(stds, dtype=torch.float32)
		assert len(stds) == n_gaussians, f"stds length {len(stds)} != n_gaussians {n_gaussians}"
		
	component_ids = np.random.choice(n_gaussians, size=n_samples)
	samples = torch.zeros(n_samples, 1, device=device)
	
	for i in range(n_gaussians):
		mask = component_ids == i
		samples[mask] = torch.normal(
			mean=float(means[i]),
			std=float(stds[i]),
			size=(mask.sum(), 1)
		).to(device)
	
	return samples


@torch.no_grad()
def load_mnist_tensor(digit = 4, train=True, normalize_to_minus1_1=True):
	tfms = [transforms.ToTensor()]
	if normalize_to_minus1_1:
		tfms.append(transforms.Lambda(lambda x: (x - 0.1307) / 0.3081))

	ds = tv_datasets.MNIST(
		root="./data",
		train=train,
		download=True,
		transform=transforms.Compose(tfms),
	)
	x = torch.stack([img for img, label in ds if label == digit], dim=0)
	return x


@torch.no_grad()
def build_training_tensor(dataset_name, n_samples=None, train=True):
	if dataset_name in DATASETS:
		return generate_gaussian_mixture(dataset_name)
	elif dataset_name in DATASETS_IMG:
		x = load_mnist_tensor(train=train)
		if n_samples is not None:
			x = x[:n_samples]
		return x
	raise ValueError(f"Unsupported dataset name")


@torch.no_grad()
def compute_mixture_pdf(dataset_name, x_axis, lam=1.0):
	"""Computes analytical pdf of training dataset from dataset config file, used for plotting"""

	dataset_config = DATASETS[dataset_name]
	means = np.array(dataset_config['means'])
	stds = dataset_config['stds']
		
	if isinstance(stds, (int, float)):
		stds = np.full(len(means), stds)
	else:
		stds = np.array(stds)
		
	stds = stds * np.sqrt(lam)
	
	n_gaussians = len(means)
	pdf = np.zeros_like(x_axis)
	
	for mu, sigma in zip(means, stds):
		pdf += norm.pdf(x_axis, loc=mu, scale=sigma)
	
	pdf /= n_gaussians  
	
	return pdf 