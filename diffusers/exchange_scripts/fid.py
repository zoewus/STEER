from PIL import Image, PngImagePlugin, ImageFile
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import cleanfid.fid as fid
from cleanfid.fid import get_files_features, frechet_distance

ImageFile.LOAD_TRUNCATED_IMAGES = True

REAL_DIR    = "/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/images/laion_5k_real"
PROMPTS_FILE = Path("/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/data_files/laion_5k_prompts.csv")

# ── Setup ─────────────────────────────────────────────────────────────────────

def build_prompt_map(prompts_file=PROMPTS_FILE, index_until=None):
	df = pd.read_csv(prompts_file)
	if index_until: df = df.iloc[:index_until]
	return {i: row["text"] for i, (_, row) in enumerate(df.iterrows())}

def build_clip_model(device):
	clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
	clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
	clip_model.eval()
	return clip_model, clip_processor


def build_feat_model(device):
	return fid.build_feature_extractor("clean", device=torch.device(device))


# ── Feature extraction ────────────────────────────────────────────────────────

def get_features_subset(folder, model, device, n=None, target_indices = None):
	files = sorted(Path(folder).glob("*.png"))[:n]
	return get_files_features(
		[str(f) for f in files],
		model,
		device=device,
	)

# ── FID ───────────────────────────────────────────────────────────────────────

def compute_real_stats(feat_model, device, real_dir=REAL_DIR, n=None, target_indices=None):
	print("Computing real image features...")
	real_feats = get_features_subset(real_dir, feat_model, device=torch.device(device), n=n, target_indices=target_indices)
	mu_real = np.mean(real_feats, axis=0)
	sigma_real = np.cov(real_feats, rowvar=False)
	print(f"Real features computed from {len(real_feats)} images.\n")
	return mu_real, sigma_real


def compute_fid_score(gen_dir, feat_model, mu_real, sigma_real, device, n=None, target_indices=None):
	gen_feats = get_features_subset(gen_dir, feat_model, device=torch.device(device), n=n, target_indices=target_indices)
	mu_gen = np.mean(gen_feats, axis=0)
	sigma_gen = np.cov(gen_feats, rowvar=False)
	return frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


# ── CLIP ──────────────────────────────────────────────────────────────────────

def compute_clip(gen_dir, tsr_lam, prompt_map, clip_model, clip_processor, device, index_until=None):
	scores = []
	for img_path in tqdm(sorted(gen_dir.glob("*.png"))[:index_until], desc=f"CLIP lam={tsr_lam}"):
		idx = int(img_path.stem)
		if idx not in prompt_map:
			continue
		prompt = prompt_map[idx]
		try:
			image = Image.open(img_path).convert("RGB")
		except Exception as e:
			print(f"  Warning, skipping {img_path.name}: {e}")
			continue
		inputs = clip_processor(
			text=[prompt], images=image,
			return_tensors="pt", padding=True,
			truncation=True, max_length=75
		).to(device)
		with torch.no_grad():
			score = clip_model(**inputs).logits_per_image[0, 0].item()
		scores.append(score)
	return np.mean(scores)

# ── Lambda-directory discovery ───────────────────────────────────────────────
# Inverse of lam_dirname(lam) = f"lam{lam:.3f}".replace(".", "p") in the
# generation script, so we can recover the float lambda a folder corresponds
# to without needing the caller to pass in an explicit list of lam values.

def parse_lam_dirname(name):
	if not name.startswith("lam"):
		raise ValueError(f"not a lam dir: {name}")
	return float(name[len("lam"):].replace("p", "."))


def discover_lam_dirs(base_dir):
	"""Return sorted (lam, path) pairs for every lam* subdir found in base_dir."""
	pairs = []
	for d in sorted(Path(base_dir).glob("lam*")):
		if not d.is_dir():
			continue
		try:
			lam = parse_lam_dirname(d.name)
		except ValueError:
			continue
		pairs.append((lam, d))
	return sorted(pairs, key=lambda p: p[0])


def _eval_lam_dirs(lam_dirs, feat_model, mu_real, sigma_real, prompt_map,
                    clip_model, clip_processor, device, index_until, target_indices, tag):
	results = {}
	for lam, samples_dir in lam_dirs:
		png_files = sorted(samples_dir.glob("*.png"))[:index_until]
		if not png_files:
			print(f"[{tag}]  lam={lam:.3f}  SKIPPED (no PNGs in {samples_dir})")
			continue

		fid_val = compute_fid_score(samples_dir, feat_model, mu_real, sigma_real, device,
		                             target_indices=target_indices, n=index_until)
		clip_val = compute_clip(samples_dir, lam, prompt_map, clip_model, clip_processor, device,
		                         index_until=index_until)
		results[lam] = (fid_val, clip_val)
		print(f"[{tag}]  lam={lam:.3f}  FID={fid_val:.4f}  CLIP={clip_val:.4f}")

	return results


# ── Sweep ─────────────────────────────────────────────────────────────────────
def compute_sweep(
	n_particles_list,
	device,
	tsr_dir=None,
	steer_dir=None,
	target_indices=None,
	index_until=None,
):
	"""
	Plots one line per entry in n_particles_list (from steer_dir/steer_{n}/lam*/),
	plus one baseline line from tsr_dir/lam*/ (n_particles-independent).
	"""
	prompt_map                 = build_prompt_map(index_until=index_until)
	clip_model, clip_processor = build_clip_model(device)
	feat_model                 = build_feat_model(device)
	mu_real, sigma_real        = compute_real_stats(feat_model, device, n=index_until)

	fig, ax = plt.subplots(figsize=(8, 6))
	all_lam_counts = 0

	# Baseline TSR line (n_particles-free).
	if tsr_dir is not None:
		tsr_lam_dirs = discover_lam_dirs(tsr_dir)
		tsr_results = _eval_lam_dirs(
			tsr_lam_dirs, feat_model, mu_real, sigma_real, prompt_map,
			clip_model, clip_processor, device, index_until, target_indices, tag="tsr",
		)
		all_lam_counts += len(tsr_results)
		if tsr_results:
			lam_vals = sorted(tsr_results.keys(), reverse=True)
			clip_vals = [tsr_results[lam][1] for lam in lam_vals]
			fid_vals  = [tsr_results[lam][0] for lam in lam_vals]
			ax.plot(clip_vals, fid_vals, marker="o", linewidth=2, label="tsr")
			for lam in lam_vals:
				f, c = tsr_results[lam]
				ax.annotate(f"lam={lam:.2f}", (c, f), textcoords="offset points",
				            xytext=(6, 0), fontsize=8, color="goldenrod")

	# One line per n_particles value.
	if steer_dir is not None:
		for n_particles in n_particles_list:
			base_dir = Path(steer_dir) / f"steer_{n_particles}"
			lam_dirs = discover_lam_dirs(base_dir)
			results = _eval_lam_dirs(
				lam_dirs, feat_model, mu_real, sigma_real, prompt_map,
				clip_model, clip_processor, device, index_until, target_indices,
				tag=f"steer_{n_particles}",
			)
			all_lam_counts += len(results)
			if not results:
				continue

			lam_vals  = sorted(results.keys(), reverse=True)
			clip_vals = [results[lam][1] for lam in lam_vals]
			fid_vals  = [results[lam][0] for lam in lam_vals]

			ax.plot(clip_vals, fid_vals, marker="o", linewidth=2, label=f"n_particles={n_particles}")
			for lam in lam_vals:
				f, c = results[lam]
				ax.annotate(f"lam={lam:.2f}", (c, f), textcoords="offset points",
				            xytext=(6, 0), fontsize=8, color="goldenrod")

	ax.set_xlabel("CLIP", fontsize=12)
	ax.set_ylabel("FID", fontsize=12)
	ax.set_title("FID vs CLIP comparison", fontsize=14)
	ax.legend(fontsize=9)
	ax.grid(True, alpha=0.3)
	plt.tight_layout()
	n_tag = "-".join(str(n) for n in n_particles_list)
	plt.savefig(f"figures/fid_vs_clip_until{index_until}_n{n_tag}_len{all_lam_counts}.png", dpi=150)
	plt.show()