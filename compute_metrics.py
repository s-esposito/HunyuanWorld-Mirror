"""Compute PSNR / SSIM / LPIPS from saved rendered + GT images."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from resplat.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim


def load_image_pairs(rendered_dir: Path):
    """Load all gt_XX.png / rendered_XX.png pairs from a directory.

    Returns (gt_bchw, rendered_bchw) tensors in [0, 1].
    """
    gt_files = sorted(rendered_dir.glob("gt_*.png"))
    gt_imgs, rendered_imgs = [], []
    for gt_path in gt_files:
        idx = gt_path.stem.replace("gt_", "")
        render_path = rendered_dir / f"rendered_{idx}.png"
        if not render_path.exists():
            continue
        gt_imgs.append(tf.to_tensor(Image.open(gt_path)))
        rendered_imgs.append(tf.to_tensor(Image.open(render_path)))
    return torch.stack(gt_imgs), torch.stack(rendered_imgs)


def compute_scene_metrics(rendered_dir: Path, device: torch.device):
    """Compute PSNR/SSIM/LPIPS for a single scene's rendered directory."""
    gt_bchw, rendered_bchw = load_image_pairs(rendered_dir)
    if gt_bchw.shape[0] == 0:
        return None

    gt_bchw = gt_bchw.to(device)
    rendered_bchw = rendered_bchw.to(device)

    psnr_vals = compute_psnr(gt_bchw, rendered_bchw)
    ssim_vals = compute_ssim(gt_bchw, rendered_bchw)
    lpips_vals = compute_lpips(gt_bchw, rendered_bchw)

    return {
        "psnr": psnr_vals.mean().item(),
        "ssim": ssim_vals.mean().item(),
        "lpips": lpips_vals.mean().item(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics from saved renders")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Root results dir containing low_res/ and/or high_res/ folders")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    device = torch.device(args.device)

    # Collect all scenes that exist in low_res
    lowres_dir = results_root / "low_res"
    hires_dir = results_root / "high_res"

    scene_names = sorted([
        d.name for d in lowres_dir.iterdir() if d.is_dir()
    ]) if lowres_dir.exists() else []

    print(f"Found {len(scene_names)} scenes")

    agg = {
        "low_res": {"psnr": [], "ssim": [], "lpips": []},
        "high_res": {"psnr": [], "ssim": [], "lpips": []},
    }

    # Evaluate scenes, reusing cached metrics.json where available
    cached, computed = 0, 0
    pbar = tqdm(scene_names, desc="Computing metrics (LPIPS: VGG)")
    for scene_name in pbar:
        for res, res_dir in (("low_res", lowres_dir), ("high_res", hires_dir)):
            scene_dir = res_dir / scene_name
            metrics_path = scene_dir / "metrics.json"
            rendered_dir = scene_dir / "rendered"

            # Try loading cached metrics first
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                for k in ("psnr", "ssim", "lpips"):
                    agg[res][k].append(metrics[k])
                cached += 1
                continue

            # Compute from images
            if not rendered_dir.exists():
                continue
            metrics = compute_scene_metrics(rendered_dir, device)
            if metrics is None:
                continue
            for k in ("psnr", "ssim", "lpips"):
                agg[res][k].append(metrics[k])
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            computed += 1

        # Update progress bar with running averages (PSNR / SSIM / LPIPS_vgg)
        parts = []
        for res, label in (("low_res", "LR"), ("high_res", "HR")):
            if agg[res]["psnr"]:
                p = np.mean(agg[res]["psnr"])
                s = np.mean(agg[res]["ssim"])
                l = np.mean(agg[res]["lpips"])
                parts.append(f"{label} {p:.2f}/{s:.3f}/{l:.3f}")
        pbar.set_postfix_str(" | ".join(parts) if parts else "")

    print(f"\n{cached} cached, {computed} newly computed")

    # Print summary
    for res in ("low_res", "high_res"):
        n = len(agg[res]["psnr"])
        if n == 0:
            continue
        print(f"\n{'='*60}")
        print(f"{res} ({n} scenes):")
        print(f"  PSNR:  {np.mean(agg[res]['psnr']):.3f}")
        print(f"  SSIM:  {np.mean(agg[res]['ssim']):.4f}")
        print(f"  LPIPS: {np.mean(agg[res]['lpips']):.4f}")
        print(f"{'='*60}")

    # Save aggregate results
    summary = {}
    for res in ("low_res", "high_res"):
        if agg[res]["psnr"]:
            summary[res] = {
                "psnr": float(np.mean(agg[res]["psnr"])),
                "ssim": float(np.mean(agg[res]["ssim"])),
                "lpips": float(np.mean(agg[res]["lpips"])),
                "num_scenes": len(agg[res]["psnr"]),
            }
    with open(results_root / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAggregate results saved to {results_root / 'metrics.json'}")
