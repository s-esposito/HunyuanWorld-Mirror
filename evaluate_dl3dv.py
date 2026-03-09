#!/usr/bin/env python
"""
Evaluate HunyuanWorld-Mirror on DL3DV test scenes using ReSplat's 8-view evaluation setup.

Follows the official WorldMirror evaluation protocol:
- Feeds ALL views (context + target) through the model (matching training/wrapper.py)
- Uses is_inference=False so GaussianSplatRenderer handles rendering + scale alignment
- Aligns all poses to first context view (matching MultiViewDataset convention)
- Provides GT camera poses + intrinsics as conditioning (cond_flags=[1,0,1])

CUDA_VISIBLE_DEVICES=0 python evaluate_dl3dv.py --save_splats --render_hires --max_scenes 1 --eval_index dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json
"""

import argparse
import json
import shutil
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from einops import rearrange, repeat
from PIL import Image
from tqdm import tqdm

from resplat.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from resplat.model.ply_export import export_ply
from resplat.visualization.vis_depth import viz_depth_tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ORI_IMAGE_SHAPE = (270, 480)   # Original DL3DV image shape (H, W)
EVAL_IMAGE_SHAPE = (256, 448)  # Evaluation image shape (H, W)


# ---------------------------------------------------------------------------
# 1. Data loading helpers
# ---------------------------------------------------------------------------

def load_eval_index(index_path: str) -> dict:
    """Load ReSplat evaluation index: {scene_key: {context: [...], target: [...]}}."""
    with open(index_path, "r") as f:
        return json.load(f)


def load_image_from_bytes(raw_bytes):
    """Load PIL image from .torch raw bytes."""
    return Image.open(BytesIO(raw_bytes.numpy().tobytes())).convert("RGB")


def load_dl3dv_chunks(data_root):
    """Load all .torch chunks and build scene->chunk mapping."""
    data_root = Path(data_root)
    index_path = data_root / "index.json"
    with open(index_path) as f:
        index = json.load(f)

    scene_to_chunk = {}
    for scene_key, chunk_info in index.items():
        chunk_path = data_root / chunk_info
        scene_to_chunk[scene_key] = chunk_path
    return scene_to_chunk


# ---------------------------------------------------------------------------
# 2. Camera conversion
# ---------------------------------------------------------------------------

def convert_poses_dl3dv(poses):
    """Convert DL3DV .torch poses to c2w (OpenCV) and normalized intrinsics.

    Args:
        poses: Tensor of shape (V, 18) from .torch chunk

    Returns:
        c2w: (V, 4, 4) camera-to-world in OpenCV convention
        intrinsics: (V, 3, 3) normalized intrinsics
    """
    b = poses.shape[0]

    # Intrinsics (normalized by image dimensions)
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    # Extrinsics: stored as OpenCV W2C in .torch file
    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)

    # Invert to get OpenCV C2W
    c2w = torch.inverse(w2c)

    return c2w, intrinsics


def apply_scale_normalization(c2w: torch.Tensor, context_indices: list):
    """Divide all translations by mean context translation norm.

    Matches DL3DV_NVStest._fetch_views() scale normalization.
    """
    ctx_c2w = c2w[context_indices]
    ctx_norms = ctx_c2w[:, :3, 3].norm(dim=1)
    scale = ctx_norms.mean().item()
    if scale < 1e-8:
        scale = 1.0

    c2w_scaled = c2w.clone()
    c2w_scaled[:, :3, 3] /= scale
    return c2w_scaled, scale


def align_poses_to_first_view(c2w: torch.Tensor, first_idx: int):
    """Align all poses to the coordinate frame of the first context view.

    Matches MultiViewDataset.__getitem__: inv(first_c2w) @ c2w.
    """
    first_c2w_inv = torch.linalg.inv(c2w[first_idx])
    return first_c2w_inv.unsqueeze(0) @ c2w


# ---------------------------------------------------------------------------
# 3. Image preprocessing and intrinsics helpers
# ---------------------------------------------------------------------------

def rescale_and_center_crop_pil(pil_img, ori_shape, target_shape):
    """Rescale + center-crop a PIL image (matching ReSplat's rescale_and_crop)."""
    h_in, w_in = ori_shape
    h_out, w_out = target_shape

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)

    pil_img = pil_img.resize((w_scaled, h_scaled), Image.LANCZOS)

    left = (w_scaled - w_out) // 2
    top = (h_scaled - h_out) // 2
    pil_img = pil_img.crop((left, top, left + w_out, top + h_out))

    return pil_img, (h_scaled, w_scaled)


def adjust_intrinsics_for_crop(intrinsics, ori_shape, target_shape):
    """Adjust normalized intrinsics for rescale + center crop."""
    h_in, w_in = ori_shape
    h_out, w_out = target_shape

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)

    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_scaled / w_out
    intrinsics[..., 1, 1] *= h_scaled / h_out
    return intrinsics


def normalized_to_pixel_intrinsics(intrinsics_norm, H, W):
    """Convert normalized intrinsics to pixel-space intrinsics."""
    K = torch.zeros_like(intrinsics_norm)
    K[..., 0, 0] = intrinsics_norm[..., 0, 0] * W
    K[..., 1, 1] = intrinsics_norm[..., 1, 1] * H
    K[..., 0, 2] = intrinsics_norm[..., 0, 2] * W
    K[..., 1, 2] = intrinsics_norm[..., 1, 2] * H
    K[..., 2, 2] = 1.0
    return K


def pil_to_tensor(pil_img):
    """Convert PIL image to torch tensor [0, 1], shape (3, H, W)."""
    return tf.ToTensor()(pil_img)


def write_colmap_reconstruction(out_dir, pred_c2w, pred_intrs, W_wm, H_wm, target_W, target_H):
    """Write COLMAP binary with intrinsics scaled from WorldMirror resolution to target resolution.

    Args:
        out_dir: Path to write sparse/0/ into
        pred_c2w: (N, 4, 4) predicted camera-to-world poses (CPU float)
        pred_intrs: (N, 3, 3) predicted pixel-space intrinsics at WM resolution (CPU float)
        W_wm, H_wm: WorldMirror resolution
        target_W, target_H: target output resolution for COLMAP cameras
    """
    sparse_dir = out_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    scale_x = target_W / W_wm
    scale_y = target_H / H_wm

    reconstruction = pycolmap.Reconstruction()
    n_views = pred_c2w.shape[0]

    for i in range(n_views):
        K = pred_intrs[i].numpy()
        params = np.array([
            K[0, 0] * scale_x, K[1, 1] * scale_y,
            K[0, 2] * scale_x, K[1, 2] * scale_y,
        ])
        cam = pycolmap.Camera(
            model="PINHOLE",
            width=target_W,
            height=target_H,
            params=params,
            camera_id=i + 1,
        )
        reconstruction.add_camera(cam)

        w2c = torch.linalg.inv(pred_c2w[i]).numpy()
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(w2c[:3, :3]), w2c[:3, 3]
        )
        img = pycolmap.Image(
            id=i + 1,
            name=f"frame_{i + 1:04d}.png",
            camera_id=i + 1,
            cam_from_world=cam_from_world,
        )
        img.registered = True
        reconstruction.add_image(img)

    reconstruction.write(str(sparse_dir))


# ---------------------------------------------------------------------------
# 4. Image preprocessing for WorldMirror
# ---------------------------------------------------------------------------

def preprocess_images_for_worldmirror(pil_images: list, target_size: int = 518):
    """Apply WorldMirror's crop preprocessing to PIL images.

    Matches prepare_images_to_tensor from inference_utils.py.

    Returns:
        tensor: [1, S, 3, H_wm, W_wm] in [0,1]
        actual_hw: (H_wm, W_wm) after preprocessing
        crop_offset: vertical crop offset (0 if no crop)
    """
    to_tensor = tf.ToTensor()
    tensors = []

    for img in pil_images:
        orig_w, orig_h = img.size
        new_w = target_size
        new_h = round(orig_h * (new_w / orig_w) / 14) * 14

        img_resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        t = to_tensor(img_resized)

        # Center-crop height if it exceeds target_size
        crop_offset = 0
        if new_h > target_size:
            crop_offset = (new_h - target_size) // 2
            t = t[:, crop_offset:crop_offset + target_size, :]
            new_h = target_size

        tensors.append(t)

    stacked = torch.stack(tensors)  # [S, 3, H, W]
    H_wm, W_wm = stacked.shape[2], stacked.shape[3]
    return stacked.unsqueeze(0), (H_wm, W_wm), crop_offset


# ---------------------------------------------------------------------------
# 5. GT-pose re-rendering
# ---------------------------------------------------------------------------

def rerender_with_gt_poses(predictions, views, num_ctx, H_wm, W_wm, model):
    """Re-render splats using GT poses instead of WorldMirror's predicted poses.

    Steps:
        1. Extract pruned splats and the predicted C2W used for rendering.
        2. Compute a rigid alignment transform (predicted frame → GT frame)
           using the first context view.
        3. Apply the transform to splat means.
        4. Render target views with GT C2W and pixel-space intrinsics.

    Returns:
        rendered_tgt: [N_tgt, H_wm, W_wm, 3] rendered target images in [0, 1].
    """
    splats = predictions["splats"]
    # rendered_extrinsics: scale-aligned predicted C2W used for rendering [1, S+V, 4, 4]
    pred_c2w = predictions["rendered_extrinsics"]
    gt_c2w = views["camera_poses"]   # [1, S+V, 4, 4]
    gt_K = views["camera_intrs"]     # [1, S+V, 3, 3]

    # --- Rigid alignment: predicted frame → GT frame via first context view ---
    T_align = gt_c2w[:, 0] @ torch.linalg.inv(pred_c2w[:, 0])  # [1, 4, 4]

    means = splats["means"][0].unsqueeze(0)  # [1, N, 3]
    means_homo = F.pad(means, (0, 1), value=1.0)  # [1, N, 4]
    means_gt = (T_align[:, :3] @ means_homo.transpose(-1, -2)).transpose(-1, -2)  # [1, N, 3]

    # --- Render target views with GT cameras ---
    gt_tgt_c2w = gt_c2w[:, num_ctx:]   # [1, N_tgt, 4, 4]
    gt_tgt_K = gt_K[:, num_ctx:]       # [1, N_tgt, 3, 3]

    with torch.no_grad():
        rendered_colors, rendered_depths, _ = model.gs_renderer.rasterizer.rasterize_batches(
            means_gt, splats["quats"], splats["scales"], splats["opacities"],
            splats["sh"],
            gt_tgt_c2w.detach(), gt_tgt_K.detach(),
            width=W_wm, height=H_wm,
            sh_degree=0,
        )
    # rendered_colors: [1, N_tgt, H_wm, W_wm, 3]
    # rendered_depths: [1, N_tgt, H_wm, W_wm, 1]
    return rendered_colors[0], rendered_depths[0]


# ---------------------------------------------------------------------------
# 6. Consolidated scene data loader
# ---------------------------------------------------------------------------

_chunk_cache = {}


def load_scene_data(chunk_path, scene_key, context_indices, target_indices,
                    hires_shape=None):
    """Load a single scene's data from a .torch chunk.

    Returns dict with:
        - all_images_pil: list of PIL images for ALL views (ctx then tgt), cropped to eval shape
        - tgt_images_tensor: (N_tgt, 3, H, W) torch tensor [0, 1] for GT metric computation
        - c2w_all: (V, 4, 4) all poses in OpenCV convention
        - intrinsics_all_norm: (V, 3, 3) normalized, crop-adjusted
        - tgt_images_hires: (N_tgt, 3, H_hr, W_hr) if hires_shape is provided
    """
    chunk_key = str(chunk_path)
    if chunk_key not in _chunk_cache:
        _chunk_cache.clear()
        _chunk_cache[chunk_key] = torch.load(chunk_path, weights_only=False)
    chunk = _chunk_cache[chunk_key]

    example = None
    for item in chunk:
        if item["key"] == scene_key:
            example = item
            break
    if example is None:
        return None

    cameras = example["cameras"]  # (V, 18)
    images_raw = example["images"]

    # Convert all poses
    c2w_all, intrinsics_all = convert_poses_dl3dv(cameras)

    # Adjust intrinsics for crop (270x480 -> 256x448)
    intrinsics_all = adjust_intrinsics_for_crop(
        intrinsics_all, ORI_IMAGE_SHAPE, EVAL_IMAGE_SHAPE
    )

    # Load ALL images (context + target) as PIL, cropped to eval shape
    all_indices = context_indices + target_indices
    all_images_pil = []
    for idx in all_indices:
        pil_img = load_image_from_bytes(images_raw[idx])
        pil_img, _ = rescale_and_center_crop_pil(
            pil_img, ORI_IMAGE_SHAPE, EVAL_IMAGE_SHAPE
        )
        all_images_pil.append(pil_img)

    # Also prepare target images as tensors for GT metric computation at eval resolution
    num_ctx = len(context_indices)
    tgt_images_tensor = torch.stack([
        pil_to_tensor(all_images_pil[num_ctx + i])
        for i in range(len(target_indices))
    ])

    result = {
        "all_images_pil": all_images_pil,
        "tgt_images_tensor": tgt_images_tensor,
        "c2w_all": c2w_all,
        "intrinsics_all_norm": intrinsics_all,
    }

    # Load hires images directly from raw originals (no intermediate downscale)
    if hires_shape is not None:
        all_hires_pil = []
        for idx in all_indices:
            pil_img = load_image_from_bytes(images_raw[idx])
            pil_img, _ = rescale_and_center_crop_pil(
                pil_img, ORI_IMAGE_SHAPE, hires_shape
            )
            all_hires_pil.append(pil_img)
        result["all_images_hires_pil"] = all_hires_pil
        result["tgt_images_hires"] = torch.stack([
            pil_to_tensor(all_hires_pil[num_ctx + i])
            for i in range(len(target_indices))
        ])

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate WorldMirror on DL3DV (ReSplat eval setup)")
    parser.add_argument("--dataset_root", type=str, default="/media/stefano/0D91176038319865/data/dl3dv-480p-chunks/test")
    parser.add_argument("--eval_index", type=str,
                        default="dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json")
    parser.add_argument("--worldmirror_path", type=str, default="/home/stefano/Codebase/HunyuanWorld-Mirror")
    parser.add_argument("--output_dir", type=str, default="output/worldmirror_dl3dv_8v")
    parser.add_argument("--target_size", type=int, default=518, help="WorldMirror target image size")
    parser.add_argument("--save_depth", action="store_true", help="Save colorized depth maps for target views")
    parser.add_argument("--save_input_depth", action="store_true", help="Save model-predicted depth maps for input/context views")
    parser.add_argument("--save_splats", action="store_true", help="Save Gaussian splats as PLY file per scene")
    parser.add_argument("--render_hires", action="store_true", help="Re-render Gaussians at high resolution and save images")
    parser.add_argument("--hires_shape", type=int, nargs=2, default=[512, 960], metavar=("H", "W"),
                        help="High-resolution render size (default: 512 960)")
    parser.add_argument("--max_scenes", type=int, default=-1, help="Max scenes to eval (-1 = all)")
    parser.add_argument("--render_mode", type=str, default="pred", choices=["pred", "gt"],
                        help="'pred': use WorldMirror predicted poses (default), "
                             "'gt': re-render with GT poses via rigid alignment")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    H_eval, W_eval = EVAL_IMAGE_SHAPE
    device = torch.device(args.device)

    # # ---- Import WorldMirror ----
    # # WorldMirror also has a `src` package, which collides with ReSplat's `src`.
    # # Temporarily swap sys.modules["src"] so Python resolves the right package.
    # resplat_src = sys.modules.pop("src")
    # sys.path.insert(0, args.worldmirror_path)
    # import src as wm_src  # noqa: this loads WorldMirror's src
    from src.models.models.worldmirror import WorldMirror
    # # Restore ReSplat's src
    # sys.modules["src"] = resplat_src
    # sys.path.remove(args.worldmirror_path)

    # ---- Load model ----
    print("Loading WorldMirror model...")
    model = WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror").to(device)
    model.eval()

    # ---- Count parameters ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {n:,} ({n/1e6:.1f}M)")

    # ---- Load indices ----
    eval_index = load_eval_index(args.eval_index)
    scene_to_chunk = load_dl3dv_chunks(dataset_root)

    available_scenes = [k for k in eval_index if k in scene_to_chunk]
    print(f"Available scenes: {len(available_scenes)}/{len(eval_index)}")

    if args.max_scenes > 0:
        available_scenes = available_scenes[:args.max_scenes]
        print(f"Evaluating first {len(available_scenes)} scenes")

    # ---- Evaluate ----
    all_psnr, all_ssim, all_lpips = [], [], []
    all_psnr_hr, all_ssim_hr, all_lpips_hr = [], [], []
    per_scene_results = {}

    for scene_idx, scene_key in enumerate(tqdm(available_scenes, desc="Evaluating")):
        entry = eval_index[scene_key]
        if entry is None:
            continue
        ctx_indices = entry["context"]
        tgt_indices = entry["target"]
        num_ctx = len(ctx_indices)
        all_indices = ctx_indices + tgt_indices

        # Load scene data (all views, poses in OpenCV convention, images cropped to eval shape)
        chunk_path = scene_to_chunk[scene_key]
        hires_shape = tuple(args.hires_shape) if args.render_hires else None
        scene_data = load_scene_data(chunk_path, scene_key, ctx_indices, tgt_indices,
                                     hires_shape=hires_shape)
        if scene_data is None:
            print(f"  Skipping scene {scene_key}: not found in chunk")
            continue

        all_images_pil = scene_data["all_images_pil"]
        tgt_images_gt = scene_data["tgt_images_tensor"]    # (N_tgt, 3, H_eval, W_eval)
        c2w_all = scene_data["c2w_all"]
        intrinsics_all_norm = scene_data["intrinsics_all_norm"]

        # Scale normalization (matching DL3DV_NVStest._fetch_views)
        c2w_scaled, scale = apply_scale_normalization(c2w_all, ctx_indices)

        # Align all poses to first context view (matching MultiViewDataset.__getitem__)
        c2w_aligned = align_poses_to_first_view(c2w_scaled, ctx_indices[0])

        # ---- Preprocess ALL images for WorldMirror ----
        all_tensor, wm_hw, crop_offset = preprocess_images_for_worldmirror(
            all_images_pil, target_size=args.target_size
        )
        H_wm, W_wm = wm_hw

        # Intrinsics for all views at WorldMirror resolution (pixel-space)
        all_intrinsics_norm = intrinsics_all_norm[all_indices]
        all_K_pixel = normalized_to_pixel_intrinsics(all_intrinsics_norm, H_wm, W_wm)
        if crop_offset > 0:
            all_K_pixel[:, 1, 2] -= crop_offset

        # All C2W (context + target, aligned to first context view)
        all_c2w = c2w_aligned[all_indices]

        # is_target flag
        is_target = torch.tensor(
            [False] * num_ctx + [True] * len(tgt_indices)
        )

        # ---- WorldMirror forward (official eval protocol: is_inference=False) ----
        views = {
            "img": all_tensor.to(device),                         # [1, 16, 3, H_wm, W_wm]
            "camera_poses": all_c2w.unsqueeze(0).to(device),       # [1, 16, 4, 4]
            "camera_intrs": all_K_pixel.unsqueeze(0).to(device),   # [1, 16, 3, 3]
            "is_target": is_target.unsqueeze(0).to(device),        # [1, 16]
        }

        use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16 if use_amp else torch.float32):
                predictions = model(views=views, cond_flags=[1, 0, 1], is_inference=False)

        # ---- Extract rendered target images and depths ----
        if args.render_mode == "pred":
            # Use WorldMirror's predicted-pose renders (existing behavior)
            # GaussianSplatRenderer.render() with is_inference=False handles:
            #   - context-only forward for scale reference
            #   - scale alignment between predicted context and all-view poses
            #   - rendering at predicted target poses
            rendered_all = predictions["rendered_colors"]  # [1, 16, H_wm, W_wm, 3]
            rendered_tgt = rendered_all[0, num_ctx:]       # [N_tgt, H_wm, W_wm, 3]
            rendered_depths_tgt = predictions["rendered_depths"][0, num_ctx:]  # [N_tgt, H_wm, W_wm, 1]
        else:
            # Re-render with GT poses: align splats from predicted→GT frame, render with GT cameras
            rendered_tgt, rendered_depths_tgt = rerender_with_gt_poses(
                predictions, views, num_ctx, H_wm, W_wm, model
            )

        # Resize rendered images to eval resolution (256x448) for fair comparison with ReSplat
        rendered_tgt_bchw = rendered_tgt.permute(0, 3, 1, 2).float()  # [N_tgt, 3, H_wm, W_wm]
        rendered_tgt_eval = F.interpolate(
            rendered_tgt_bchw, size=(H_eval, W_eval), mode='bilinear', align_corners=False
        ).clamp(0, 1)

        # GT target images at eval resolution
        gt_bchw = tgt_images_gt.to(device)  # [N_tgt, 3, H_eval, W_eval]

        # ---- Compute metrics ----
        psnr_vals = compute_psnr(gt_bchw, rendered_tgt_eval)
        ssim_vals = compute_ssim(gt_bchw, rendered_tgt_eval)
        lpips_vals = compute_lpips(gt_bchw, rendered_tgt_eval)

        scene_psnr = psnr_vals.mean().item()
        scene_ssim = ssim_vals.mean().item()
        scene_lpips = lpips_vals.mean().item()

        all_psnr.append(scene_psnr)
        all_ssim.append(scene_ssim)
        all_lpips.append(scene_lpips)
        per_scene_results[scene_key] = {
            "psnr": scene_psnr, "ssim": scene_ssim, "lpips": scene_lpips
        }

        # ---- Save low_res: COLMAP poses + GT images + rendered images ----
        pred_c2w_cpu = predictions["camera_poses"][0].float().cpu()   # [S, 4, 4]
        pred_intrs_cpu = predictions["camera_intrs"][0].float().cpu()  # [S, 3, 3]

        lowres_dir = output_dir / "low_res" / scene_key
        write_colmap_reconstruction(
            lowres_dir, pred_c2w_cpu, pred_intrs_cpu,
            W_wm, H_wm, W_eval, H_eval,
        )
        # Save all GT images at eval resolution
        img_dir = lowres_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for i, pil_img in enumerate(all_images_pil):
            pil_img.save(str(img_dir / f"frame_{i + 1:04d}.png"))
        # Save rendered + GT target images
        rendered_dir = lowres_dir / "rendered"
        rendered_dir.mkdir(parents=True, exist_ok=True)
        for i in range(rendered_tgt_eval.shape[0]):
            tf.ToPILImage()(rendered_tgt_eval[i].cpu()).save(
                str(rendered_dir / f"rendered_{i:02d}.png"))
            tf.ToPILImage()(gt_bchw[i].cpu()).save(
                str(rendered_dir / f"gt_{i:02d}.png"))

        # ---- Save depth maps (optional) ----
        if args.save_depth:
            depth_dir = output_dir / "depth" / scene_key
            depth_dir.mkdir(parents=True, exist_ok=True)
            for i in range(rendered_depths_tgt.shape[0]):
                depth_i = rendered_depths_tgt[i, :, :, 0].float().cpu()  # [H_wm, W_wm]
                depth_viz = viz_depth_tensor(
                    1.0 / depth_i.clamp(min=1e-6), return_numpy=True
                )  # [H_wm, W_wm, 3]
                Image.fromarray(depth_viz).save(str(depth_dir / f"depth_{i:02d}.png"))

        # ---- Save input view predicted depth (optional) ----
        if args.save_input_depth:
            input_depth_dir = output_dir / "input_depth" / scene_key
            input_depth_dir.mkdir(parents=True, exist_ok=True)
            pred_depths = predictions["depth"][0, :num_ctx, :, :, 0]  # [num_ctx, H_wm, W_wm]
            for i in range(pred_depths.shape[0]):
                depth_i = pred_depths[i].float().cpu()  # [H_wm, W_wm]
                depth_viz = viz_depth_tensor(
                    1.0 / depth_i.clamp(min=1e-6), return_numpy=True
                )  # [H_wm, W_wm, 3]
                Image.fromarray(depth_viz).save(str(input_depth_dir / f"input_depth_{i:02d}.png"))

        # ---- Save Gaussian splats (optional) ----
        if args.save_splats:
            splats = predictions["splats"]
            means_0 = splats["means"][0].float()          # [K, 3]
            quats_0 = splats["quats"][0].float()          # [K, 4] wxyz
            scales_0 = splats["scales"][0].float()        # [K, 3] linear
            opacities_0 = splats["opacities"][0].float()  # [K] sigmoid
            sh_0 = splats["sh"][0].float()                # [K, nums_sh, 3]

            # Quaternion: gsplat wxyz -> scipy xyzw (expected by export_ply)
            quats_xyzw = torch.cat([quats_0[:, 1:], quats_0[:, :1]], dim=-1)
            # SH: [K, nums_sh, 3] -> [K, 3, nums_sh] (expected by export_ply)
            harmonics = sh_0.permute(0, 2, 1)

            # Save into low_res scene folder
            ply_path_lr = lowres_dir / "gaussians.ply"
            export_ply(
                extrinsics=torch.eye(4, device=means_0.device),
                means=means_0, scales=scales_0, rotations=quats_xyzw,
                harmonics=harmonics, opacities=opacities_0,
                path=ply_path_lr, align_to_view=False,
            )

        # ---- Render Gaussians at high resolution (optional) ----
        if args.render_hires:
            H_hr, W_hr = args.hires_shape
            hires_dir = output_dir / "high_res" / scene_key

            splats = predictions["splats"]

            # Determine which C2W / K to use for re-rendering all views
            if args.render_mode == "gt":
                render_c2w = views["camera_poses"]  # [1, S, 4, 4]
                render_K = views["camera_intrs"]    # [1, S, 3, 3] pixel-space at W_wm x H_wm

                # Align splat means from predicted frame to GT frame
                pred_c2w_render = predictions["rendered_extrinsics"]
                T_align = render_c2w[:, 0] @ torch.linalg.inv(pred_c2w_render[:, 0])
                means_hr = splats["means"][0].unsqueeze(0)  # [1, N, 3]
                means_homo = F.pad(means_hr, (0, 1), value=1.0)
                means_hr = (T_align[:, :3] @ means_homo.transpose(-1, -2)).transpose(-1, -2)
                splats_means = means_hr
            else:
                render_c2w = predictions["rendered_extrinsics"]  # [1, S, 4, 4]
                render_K = predictions["camera_intrs"]           # [1, S, 3, 3]
                splats_means = splats["means"]

            # Scale intrinsics from WorldMirror resolution to high-res
            scale_x = W_hr / W_wm
            scale_y = H_hr / H_wm
            K_hr = render_K.clone()
            K_hr[..., 0, :] *= scale_x  # fx, cx
            K_hr[..., 1, :] *= scale_y  # fy, cy

            # Render only target views at high resolution
            render_tgt_c2w = render_c2w[:, num_ctx:]  # [1, N_tgt, 4, 4]
            K_hr_tgt = K_hr[:, num_ctx:]              # [1, N_tgt, 3, 3]

            with torch.no_grad():
                hr_colors, _, _ = model.gs_renderer.rasterizer.rasterize_batches(
                    splats_means, splats["quats"], splats["scales"], splats["opacities"],
                    splats["sh"],
                    render_tgt_c2w.detach(), K_hr_tgt.detach(),
                    width=W_hr, height=H_hr,
                    sh_degree=0,
                )
            # hr_colors: [1, N_tgt, H_hr, W_hr, 3]
            rendered_hr_bchw = hr_colors[0].permute(0, 3, 1, 2).float().clamp(0, 1)  # [N_tgt, 3, H_hr, W_hr]

            # GT target images at native high-res (loaded directly from raw source)
            gt_hr_bchw = scene_data["tgt_images_hires"].to(device)

            # Compute hires metrics
            hr_psnr_vals = compute_psnr(gt_hr_bchw, rendered_hr_bchw)
            hr_ssim_vals = compute_ssim(gt_hr_bchw, rendered_hr_bchw)
            hr_lpips_vals = compute_lpips(gt_hr_bchw, rendered_hr_bchw)

            scene_psnr_hr = hr_psnr_vals.mean().item()
            scene_ssim_hr = hr_ssim_vals.mean().item()
            scene_lpips_hr = hr_lpips_vals.mean().item()

            all_psnr_hr.append(scene_psnr_hr)
            all_ssim_hr.append(scene_ssim_hr)
            all_lpips_hr.append(scene_lpips_hr)
            per_scene_results[scene_key]["psnr_hr"] = scene_psnr_hr
            per_scene_results[scene_key]["ssim_hr"] = scene_ssim_hr
            per_scene_results[scene_key]["lpips_hr"] = scene_lpips_hr

            # Save high_res: COLMAP poses + GT images + rendered images
            write_colmap_reconstruction(
                hires_dir, pred_c2w_cpu, pred_intrs_cpu,
                W_wm, H_wm, W_hr, H_hr,
            )
            hr_img_dir = hires_dir / "images"
            hr_img_dir.mkdir(parents=True, exist_ok=True)
            for i, pil_img in enumerate(scene_data["all_images_hires_pil"]):
                pil_img.save(str(hr_img_dir / f"frame_{i + 1:04d}.png"))
            hr_rendered_dir = hires_dir / "rendered"
            hr_rendered_dir.mkdir(parents=True, exist_ok=True)
            for i in range(rendered_hr_bchw.shape[0]):
                tf.ToPILImage()(rendered_hr_bchw[i].cpu()).save(
                    str(hr_rendered_dir / f"rendered_{i:02d}.png"))
                tf.ToPILImage()(gt_hr_bchw[i].cpu()).save(
                    str(hr_rendered_dir / f"gt_{i:02d}.png"))

            # Duplicate splats into high_res scene folder
            if args.save_splats:
                shutil.copy2(str(ply_path_lr), str(hires_dir / "gaussians.ply"))

        if (scene_idx + 1) % 10 == 0 or scene_idx == 0:
            tqdm.write(
                f"  [{scene_idx + 1}/{len(available_scenes)}] "
                f"PSNR: {scene_psnr:.3f} | SSIM: {scene_ssim:.3f} | LPIPS: {scene_lpips:.3f} "
                f"(running avg: PSNR {np.mean(all_psnr):.3f})"
            )

    # ---- Aggregate results ----
    if all_psnr:
        mean_psnr = np.mean(all_psnr)
        mean_ssim = np.mean(all_ssim)
        mean_lpips = np.mean(all_lpips)
    else:
        mean_psnr = mean_ssim = mean_lpips = 0.0

    mode_label = "predicted poses" if args.render_mode == "pred" else "GT poses (aligned)"
    print(f"\n{'='*60}")
    print(f"WorldMirror DL3DV 8-view {H_eval}x{W_eval} [{mode_label}] ({len(all_psnr)} scenes):")
    print(f"  PSNR:  {mean_psnr:.3f}")
    print(f"  SSIM:  {mean_ssim:.4f}")
    print(f"  LPIPS: {mean_lpips:.4f}")
    print(f"{'='*60}")
    print(f"ReSplat reference (init): PSNR=27.365, SSIM=0.877, LPIPS=0.130")

    # Save results
    results = {
        "psnr": mean_psnr, "ssim": mean_ssim, "lpips": mean_lpips,
        "num_scenes": len(all_psnr),
        "config": {
            "eval_shape": list(EVAL_IMAGE_SHAPE),
            "target_size": args.target_size,
            "render_mode": args.render_mode,
            "cond_flags": [1, 0, 1],
            "protocol": "official (all views, is_inference=False, first-view aligned)",
        }
    }

    if args.render_hires and all_psnr_hr:
        H_hr, W_hr = args.hires_shape
        mean_psnr_hr = np.mean(all_psnr_hr)
        mean_ssim_hr = np.mean(all_ssim_hr)
        mean_lpips_hr = np.mean(all_lpips_hr)

        print(f"\nHigh-res {H_hr}x{W_hr} [{mode_label}] ({len(all_psnr_hr)} scenes):")
        print(f"  PSNR:  {mean_psnr_hr:.3f}")
        print(f"  SSIM:  {mean_ssim_hr:.4f}")
        print(f"  LPIPS: {mean_lpips_hr:.4f}")
        print(f"{'='*60}")

        results["hires"] = {
            "psnr": mean_psnr_hr, "ssim": mean_ssim_hr, "lpips": mean_lpips_hr,
            "shape": [H_hr, W_hr],
        }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "per_scene.json", "w") as f:
        json.dump(per_scene_results, f, indent=2)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
