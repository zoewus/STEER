import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_DIFFUSION_STEPS = 100

CKPT_DIR = "model_checkpoints"

TRAINING = {
    "n_steps": 500_000,
    "batch_size": 512,
    "lr": 2e-4
}

TRAINING_IMG = {
	"n_steps": 500_000,
	"batch_size": 64,
	"lr": 2e-4
}

DATASETS = {
    "single":   {"dataset_shape": (50_000,1), "means": [0.0],        "stds": 1.0},
    "barrier":  {"dataset_shape": (50_000,1), "means": [-3.0, 3.0],  "stds": 0.5},
    "composed": {
    "dataset_shape": (100_000, 1),
    "means": [-4.0, 0.0, 4.0],
    "stds":  [0.2, 1.5, 0.2]
    }
}

DATASETS_IMG = {
	"mnist": {"sample_shape": (1, 28, 28)}
}

TSR = {
	"sigma":  3.0
}

