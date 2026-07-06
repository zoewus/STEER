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


# ── Sweep ─────────────────────────────────────────────────────────────────────
def compute_sweep(
	replica_exchanges,
	device,
	target_indices=None,
	index_until=None,
	tsr_dir=None,
	pt_tsr_dir=None,
	lam_values_ptsr=None,
	lam_values_tsr=None
):
	if pt_tsr_dir is not None: PT_TSR_DIR = pt_tsr_dir
	if tsr_dir is not None: TSR_DIR = tsr_dir

	prompt_map                 = build_prompt_map(index_until=index_until)
	clip_model, clip_processor = build_clip_model(device)
	feat_model                 = build_feat_model(device)
	mu_real, sigma_real        = compute_real_stats(feat_model, device, n=index_until)

	tsr_results = {alg: {} for alg in replica_exchanges}

	# map each alg to its (samples_dir, lam_values)
	alg_config = {}
	for alg in replica_exchanges:
		if alg == False:
			alg_config[alg] = (TSR_DIR, lam_values_tsr or [])
		elif alg == True:
			alg_config[alg] = (PT_TSR_DIR, lam_values_ptsr or [])

	for alg in replica_exchanges:
		samples_base, lam_values = alg_config[alg]

		for tsr_lam in lam_values:
			lam_str = f"lam{tsr_lam:.3f}".replace(".", "p")
			samples_dir = samples_base / lam_str

			png_files = sorted(samples_dir.glob("*.png"))[:index_until]
			if not png_files:
				print(f"[{alg}]  lam={tsr_lam:.3f}  SKIPPED (no PNGs in {samples_dir})")
				continue

			fid_val  = compute_fid_score(samples_dir, feat_model, mu_real, sigma_real, device, target_indices=target_indices, n=index_until)
			clip_val = compute_clip(samples_dir, tsr_lam, prompt_map, clip_model, clip_processor, device, index_until=index_until)
			tsr_results[alg][tsr_lam] = (fid_val, clip_val)
			print(f"[{alg}]  lam={tsr_lam:.3f}  FID={fid_val:.4f}  CLIP={clip_val:.4f}")

	fig, ax = plt.subplots(figsize=(8, 6))

	for alg in replica_exchanges:
		lam_vals  = sorted(tsr_results[alg].keys(), reverse=True)
		if not lam_vals:
			continue
		clip_vals = [tsr_results[alg][lam][1] for lam in lam_vals]
		fid_vals  = [tsr_results[alg][lam][0] for lam in lam_vals]

		ax.plot(clip_vals, fid_vals, marker="o", linewidth=2, label=f"replica_exchange={alg}")
		for lam in lam_vals:
			f, c = tsr_results[alg][lam]
			ax.annotate(f"lam={lam:.2f}", (c, f), textcoords="offset points", xytext=(6, 0), fontsize=8, color="goldenrod")

	all_lam_counts = sum(len(v) for v in tsr_results.values())
	ax.set_xlabel("CLIP", fontsize=12)
	ax.set_ylabel("FID", fontsize=12)
	ax.set_title("FID vs CLIP comparison", fontsize=14)
	ax.legend(fontsize=9)
	ax.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(f"figures/fid_vs_clip_until{index_until}_len{all_lam_counts}.png", dpi=150)
	plt.show()

	return tsr_results