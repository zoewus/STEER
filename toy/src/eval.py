import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

from .model import load_model
from .dataset import compute_mixture_pdf
from .config_toy import DEVICE, DATASETS, DATASETS_IMG, CKPT_DIR, N_DIFFUSION_STEPS
from .sample import ddpm_tsr_swapped
from .best_of_n import best_of_n
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
torch.manual_seed(42)
np.random.seed(42)
import argparse

device = DEVICE
ckpt_dir = CKPT_DIR

# ── Presentation style ────────────────────────────────────────────────────────
SAMPLE_COLOR  = "purple"#"#5B9BD5"   # blue
PDF_COLOR     = "royalblue"   # orange
COLD_COLOR    = "#C00000"   # red tint for cold replica label
WARM_COLOR    = "#404040"   # dark grey for warm replica label
HIST_COLOR = "chocolate"          # muted red, CB-safe
PDF_COLOR  = "#4C72B0"          # muted blue, CB-safe

def _temp_label(lam):
	"""Human-readable temperature label from lambda."""
	# if lam >= 1.0:
	# 	return "Standard  (λ=1.0)"
	# elif lam <= 0.15:
	# 	return f"Cold  (λ={lam:.2f})"
	# else:
	# 	return f"Tempered  (λ={lam:.2f})"
	return f"CNS"

def _temp_color(lam):
	return COLD_COLOR if lam < 0.5 else WARM_COLOR


def plot_temperature_triptych(
	dataset_name="composed",
	lam=1.0,
	n_replicas=2,
	replica_swaps=False,
	swap_algorithm=None,
	x_limit=6,
	n_bins=220,
	best_of_n_runs=1,
):
	model = load_model(f"{ckpt_dir}/{dataset_name}_1.0.pt", dataset_name)

	dataset_config = DATASETS[dataset_name]
	dataset_shape  = dataset_config["dataset_shape"]
	x_axis = np.linspace(-x_limit, x_limit, n_bins)
	bins   = np.linspace(-x_limit, x_limit, n_bins)
	
	lam_list = np.round(np.linspace(lam, 2.0 - lam, n_replicas), decimals=2)

	samples_ladder = ddpm_tsr_swapped(
		model, dataset_shape, lam, 1/lam, n_replicas,
		replica_swaps=replica_swaps,
		swap_algorithm=swap_algorithm,
	)

	fig, axes = plt.subplots(
	1, n_replicas,
	figsize=(3.5 * n_replicas, 2.4),           # single-col: 3.5in; double: 7.0in
	layout="constrained",         # replaces tight_layout
	)

	for i in range(n_replicas):
		ax = axes[i]

		col_lam = lam_list[i]

		samples = samples_ladder[dataset_shape[0]*i : dataset_shape[0]*(i+1)]
		samples_np = samples.detach().cpu().numpy().reshape(-1)
		pdf        = compute_mixture_pdf(dataset_name, x_axis, lam=col_lam)

		ax.hist(
			samples_np, bins=80,         # 220 bins is over-binned for papers
			density=True,
			color=HIST_COLOR, alpha=0.35,
			linewidth=0,                  # no bar edges
			label="Samples",
		)
		ax.plot(
			x_axis, pdf,
			color=PDF_COLOR, linewidth=1.5,
			label=r"$p_\lambda(x)$",      # math label looks sharp with usetex
		)
		ax.set_ylim(0, 1.5)
		ax.set_xlim(-x_limit, x_limit)

		ax.set_xlabel(r"$x$")
		ax.set_ylabel("Density")
		ax.yaxis.set_major_formatter(
			ticker.FormatStrFormatter("%.2f")
		)
		ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

		ax.legend(
		handles=[
				mpatches.Patch(
					color=HIST_COLOR, alpha=0.35, label="Samples"
				),
				mlines.Line2D(
					[], [], color=PDF_COLOR, lw=1.5,
					label=r"$p_\lambda(x)$"
				),
			],
			loc="upper right",
			fontsize=8,
			handlelength=1.2,
			handletextpad=0.4,
			borderpad=0,
		)
		if replica_swaps == True:
			ax.set_title(f"STEER {col_lam}", fontsize=9, pad=4)
		else:
			ax.set_title(f"TSR {col_lam}", fontsize=9, pad=4)

	plt.savefig(f"Swap_{replica_swaps}_{col_lam}.png")
	plt.close()


	return fig


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--lam", type=float, default=0.5)
	parser.add_argument("--n_replicas", type=int, default=3)
	parser.add_argument("--replica_exchange", action="store_true", default=False)
	args = parser.parse_args()

	_ = plot_temperature_triptych(
		dataset_name="composed",
		lam=args.lam,
		n_replicas=args.n_replicas,
		replica_swaps=args.replica_exchange,
		swap_algorithm={},
	)