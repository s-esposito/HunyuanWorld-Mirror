#!/usr/bin/env python
"""
Evaluate HunyuanWorld-Mirror on DAVIS video sequences (unposed).

Loads RGB frames from DAVIS JPEGImages and optionally segmentation masks from
Annotations_unsupervised. Runs WorldMirror in fully unposed mode (cond_flags=[0,0,0])
and saves 3D reconstruction outputs (Gaussians, point clouds, depth, COLMAP, video).

Two independent masking options:
  --mask_input      Black out background in input images before feeding to model.
  --mask_gaussians  Filter output Gaussians to keep only foreground, applied at
                    the per-pixel stage before voxel pruning.

Example:
    CUDA_VISIBLE_DEVICES=0 python evaluate_davis.py --scene_name car-turn \
        --mask_input --mask_gaussians --save_splats --save_rendered
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
import torchvision.transforms as tf
from PIL import Image

from src.models.models.worldmirror import WorldMirror
from src.utils.save_utils import save_gs_ply, save_scene_ply, save_depth_png, save_depth_npy
from src.utils.render_utils import render_interpolated_video


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_davis_scene(dataset_root, scene_name, stride=1, max_frames=-1):
    """Load RGB images and segmentation masks from a DAVIS scene.

    Returns:
        pil_images: list of PIL RGB images
        masks: list of numpy bool arrays (H, W), True = foreground
               (None if mask directory does not exist)
        image_names: list of filename stems
    """
    root = Path(dataset_root)
    img_dir = root / "JPEGImages" / "Full-Resolution" / scene_name
    mask_dir = root / "Annotations_unsupervised" / "Full-Resolution" / scene_name

    img_paths = sorted(img_dir.glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No JPEGs found in {img_dir}")

    # Subsample (always keep first and last frame)
    if stride > 1:
        all_paths = img_paths
        img_paths = all_paths[::stride]
        if all_paths[-1] != img_paths[-1]:
            img_paths.append(all_paths[-1])
    if max_frames > 0:
        img_paths = img_paths[:max_frames]

    pil_images = []
    masks = []
    image_names = []
    has_masks = mask_dir.exists()

    for img_path in img_paths:
        pil_images.append(Image.open(img_path).convert("RGB"))
        image_names.append(img_path.stem)

        if has_masks:
            mask_path = mask_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                mask = np.array(Image.open(mask_path))
                masks.append(mask > 0)  # non-zero = foreground
            else:
                # No mask for this frame -> treat everything as foreground
                w, h = pil_images[-1].size
                masks.append(np.ones((h, w), dtype=bool))

    return pil_images, masks if has_masks else None, image_names


def apply_mask_to_images(pil_images, masks):
    """Multiply images by binary masks (background -> black).

    Returns list of masked PIL images.
    """
    masked = []
    for img, mask in zip(pil_images, masks):
        arr = np.array(img)
        arr[~mask] = 0
        masked.append(Image.fromarray(arr))
    return masked


# ---------------------------------------------------------------------------
# 2. Image preprocessing for WorldMirror
# ---------------------------------------------------------------------------

def preprocess_images_for_worldmirror(pil_images, target_size=518):
    """Resize PIL images for WorldMirror input (matching infer.py crop strategy).

    Returns:
        tensor: [1, S, 3, H_wm, W_wm] in [0,1]
        actual_hw: (H_wm, W_wm)
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

        crop_offset = 0
        if new_h > target_size:
            crop_offset = (new_h - target_size) // 2
            t = t[:, crop_offset:crop_offset + target_size, :]
            new_h = target_size

        tensors.append(t)

    stacked = torch.stack(tensors)  # [S, 3, H, W]
    H_wm, W_wm = stacked.shape[2], stacked.shape[3]
    return stacked.unsqueeze(0), (H_wm, W_wm), crop_offset


def resize_masks_to_wm(masks, pil_images, target_size, crop_offset):
    """Resize segmentation masks to match WorldMirror preprocessing.

    Mirrors the same resize + center-crop logic applied to images.

    Returns:
        masks_wm: numpy bool array (S, H_wm, W_wm)
    """
    masks_wm = []
    for mask, img in zip(masks, pil_images):
        orig_w, orig_h = img.size
        new_w = target_size
        new_h = round(orig_h * (new_w / orig_w) / 14) * 14

        # Resize mask with nearest-neighbor to preserve binary labels
        mask_resized = cv2.resize(
            mask.astype(np.uint8), (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        )

        # Center-crop height if needed (same as image preprocessing)
        if new_h > target_size:
            mask_resized = mask_resized[crop_offset:crop_offset + target_size, :]

        masks_wm.append(mask_resized > 0)

    return np.stack(masks_wm, axis=0)  # [S, H_wm, W_wm]


# ---------------------------------------------------------------------------
# 3. COLMAP export (using predicted cameras)
# ---------------------------------------------------------------------------

def write_colmap_reconstruction(out_dir, pred_c2w, pred_intrs, W_wm, H_wm,
                                target_W, target_H, image_names):
    """Write COLMAP binary model from predicted cameras."""
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
            name=f"{image_names[i]}.png",
            camera_id=i + 1,
            cam_from_world=cam_from_world,
        )
        img.registered = True
        reconstruction.add_image(img)

    reconstruction.write(str(sparse_dir))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WorldMirror on DAVIS sequences (unposed)"
    )
    parser.add_argument("--dataset_root", type=str,
                        default="/home/stefano/Data/davis")
    parser.add_argument("--scene_name", type=str, default="car-turn")
    parser.add_argument("--output_dir", type=str,
                        default="output/worldmirror_davis")
    parser.add_argument("--target_size", type=int, default=518)
    parser.add_argument("--stride", type=int, default=1,
                        help="Frame subsampling stride")
    parser.add_argument("--max_frames", type=int, default=-1,
                        help="Max frames to use (-1 = all)")

    # Masking options
    parser.add_argument("--mask_input", action="store_true",
                        help="Apply segmentation mask to input images "
                             "(background -> black)")
    parser.add_argument("--mask_gaussians", action="store_true",
                        help="Filter output Gaussians by foreground mask "
                             "(per-pixel, before voxel pruning)")
    parser.add_argument("--is_video", action="store_true",
                        help="Treat each view independently: split Gaussians "
                             "per view, save/render per-view PLYs")

    # Save options
    parser.add_argument("--save_splats", action="store_true",
                        help="Save Gaussian splats as PLY")
    parser.add_argument("--save_depth", action="store_true",
                        help="Save depth maps (PNG + NPY)")
    parser.add_argument("--save_colmap", action="store_true",
                        help="Save COLMAP sparse reconstruction")
    parser.add_argument("--save_pointmap", action="store_true",
                        help="Save point cloud PLY")
    parser.add_argument("--save_rendered", action="store_true",
                        help="Render interpolated video from Gaussians")

    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir) / args.scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load DAVIS scene ----
    print(f"Loading DAVIS scene: {args.scene_name}")
    pil_images, masks, image_names = load_davis_scene(
        args.dataset_root, args.scene_name,
        stride=args.stride, max_frames=args.max_frames,
    )
    S = len(pil_images)
    print(f"  Loaded {S} frames"
          f"{', with masks' if masks is not None else ', no masks found'}")

    if (args.mask_input or args.mask_gaussians) and masks is None:
        raise RuntimeError(
            "Masking requested but no segmentation masks found at "
            f"{Path(args.dataset_root) / 'Annotations_unsupervised' / 'Full-Resolution' / args.scene_name}"
        )

    # ---- Optionally mask input images ----
    images_for_model = pil_images
    if args.mask_input:
        print("  Applying input masking (background -> black)")
        images_for_model = apply_mask_to_images(pil_images, masks)

    # ---- Preprocess for WorldMirror ----
    all_tensor, (H_wm, W_wm), crop_offset = preprocess_images_for_worldmirror(
        images_for_model, target_size=args.target_size,
    )
    print(f"  WorldMirror input: {all_tensor.shape} (H={H_wm}, W={W_wm})")

    # Resize masks to WM resolution if needed for Gaussian masking
    masks_wm = None
    if args.mask_gaussians:
        masks_wm = resize_masks_to_wm(masks, pil_images, args.target_size, crop_offset)
        fg_count = masks_wm.sum()
        total = masks_wm.size
        print(f"  Foreground pixels at WM res: {fg_count}/{total} "
              f"({100 * fg_count / total:.1f}%)")

    # ---- Load model ----
    print("Loading WorldMirror model...")
    model = WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror").to(device)
    model.eval()

    # ---- Inference (unposed) ----
    views = {"img": all_tensor.to(device)}

    # If masking Gaussians or is_video: temporarily disable pruning/filtering
    # so we get raw per-pixel splats with 1:1 pixel correspondence.
    need_raw_splats = args.mask_gaussians or args.is_video
    if need_raw_splats:
        orig_prune = model.gs_renderer.enable_prune
        orig_conf = model.gs_renderer.enable_conf_filter
        model.gs_renderer.enable_prune = False
        model.gs_renderer.enable_conf_filter = False

    use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    print("Running inference...")
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            predictions = model(views=views, cond_flags=[0, 0, 0])

    # ---- Post-process raw per-pixel Gaussians (masking / per-view split) ----
    per_view_splats = None
    if need_raw_splats:
        splats = predictions["splats"]
        splat_keys = [k for k in ["means", "quats", "scales", "opacities", "sh",
                                   "weights", "residual_sh", "gs_feats"]
                      if k in splats]
        pixels_per_view = H_wm * W_wm

        if args.is_video:
            # Split into per-view chunks, optionally mask, then prune each independently
            per_view_splats = []
            for i in range(S):
                start = i * pixels_per_view
                end = start + pixels_per_view
                vs = {k: splats[k][:, start:end] for k in splat_keys}

                # Apply per-view foreground mask
                if args.mask_gaussians:
                    vm = torch.from_numpy(masks_wm[i].reshape(-1)).bool().to(device)
                    vs = {k: vs[k][:, vm] for k in vs}

                # Prune this view's Gaussians independently
                if orig_prune:
                    pruned = model.gs_renderer.prune_gs(vs)
                    vs = {k: torch.stack(pruned[k]) for k in pruned}

                per_view_splats.append(vs)

            # Merge all views for combined outputs (PLY, interpolated video)
            merged = {}
            for k in per_view_splats[0]:
                merged[k] = torch.cat([vs[k] for vs in per_view_splats], dim=1)
            predictions["splats"] = merged

        else:
            # Flat masking across all views (existing --mask_gaussians behavior)
            fg_mask = torch.from_numpy(
                masks_wm.reshape(-1)
            ).bool().to(device)

            for k in splat_keys:
                splats[k] = splats[k][:, fg_mask]

            if orig_prune:
                pruned = model.gs_renderer.prune_gs(splats)
                for k in pruned:
                    pruned[k] = torch.stack(pruned[k])
                predictions["splats"] = pruned

        # Restore original flags
        model.gs_renderer.enable_prune = orig_prune
        model.gs_renderer.enable_conf_filter = orig_conf

        n_gs = predictions["splats"]["means"][0].shape[0]
        print(f"  Gaussians after processing: {n_gs}")

    # ---- Save outputs ----
    print(f"Saving outputs to {output_dir}")

    # Save input images (at WM resolution)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(S):
        im = (all_tensor[0, i].permute(1, 2, 0).clamp(0, 1) * 255).to(
            torch.uint8
        ).cpu().numpy()
        Image.fromarray(im).save(str(images_dir / f"{image_names[i]}.png"))

    # Save masks (at WM resolution) if masking was used
    if masks_wm is not None:
        mask_out_dir = output_dir / "masks"
        mask_out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(S):
            mask_img = (masks_wm[i].astype(np.uint8)) * 255
            Image.fromarray(mask_img).save(
                str(mask_out_dir / f"{image_names[i]}.png")
            )

    # Save depth maps
    if args.save_depth and "depth" in predictions:
        depth_dir = output_dir / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        for i in range(S):
            depth_i = predictions["depth"][0, i, :, :, 0]
            save_depth_png(depth_dir / f"{image_names[i]}.png", depth_i)
            save_depth_npy(depth_dir / f"{image_names[i]}.npy", depth_i)
        print(f"  Saved {S} depth maps")

    # Save point cloud
    if args.save_pointmap and "pts3d" in predictions:
        pts_list = []
        colors_list = []
        mask_list = []
        for i in range(S):
            pts = predictions["pts3d"][0, i]  # [H, W, 3]
            img_colors = all_tensor[0, i].permute(1, 2, 0).to(device)  # [H, W, 3]
            img_colors = (img_colors * 255).to(torch.uint8)
            pts_list.append(pts.reshape(-1, 3))
            colors_list.append(img_colors.reshape(-1, 3))

            # Apply foreground mask to point cloud if masks available
            if masks_wm is not None:
                mask_list.append(
                    torch.from_numpy(masks_wm[i].reshape(-1)).to(device)
                )

        all_pts = torch.cat(pts_list, dim=0)
        all_colors = torch.cat(colors_list, dim=0)

        if mask_list:
            all_mask = torch.cat(mask_list, dim=0)
            all_pts = all_pts[all_mask]
            all_colors = all_colors[all_mask]

        save_scene_ply(output_dir / "pointmap.ply", all_pts, all_colors)
        print(f"  Saved point cloud ({all_pts.shape[0]} points)")

    # Save Gaussian splats
    if args.save_splats and "splats" in predictions:
        splats = predictions["splats"]
        means = splats["means"][0].float().reshape(-1, 3)
        scales = splats["scales"][0].float().reshape(-1, 3)
        quats = splats["quats"][0].float().reshape(-1, 4)
        opacities = splats["opacities"][0].float().reshape(-1)
        colors = (
            splats["sh"][0] if "sh" in splats else splats["colors"][0]
        ).float().reshape(-1, 3)

        save_gs_ply(
            output_dir / "gaussians.ply",
            means, scales, quats, colors, opacities,
        )
        print(f"  Saved Gaussian splats ({means.shape[0]} Gaussians)")

        # Per-view PLYs
        if args.is_video and per_view_splats is not None:
            pv_dir = output_dir / "per_view_splats"
            pv_dir.mkdir(parents=True, exist_ok=True)
            for i, vs in enumerate(per_view_splats):
                pv_means = vs["means"][0].float().reshape(-1, 3)
                pv_scales = vs["scales"][0].float().reshape(-1, 3)
                pv_quats = vs["quats"][0].float().reshape(-1, 4)
                pv_opacities = vs["opacities"][0].float().reshape(-1)
                pv_colors = (
                    vs["sh"][0] if "sh" in vs else vs["colors"][0]
                ).float().reshape(-1, 3)
                save_gs_ply(
                    pv_dir / f"{image_names[i]}.ply",
                    pv_means, pv_scales, pv_quats, pv_colors, pv_opacities,
                )
            print(f"  Saved {S} per-view PLYs to per_view_splats/")

    # Save COLMAP reconstruction
    if args.save_colmap:
        pred_c2w = predictions["camera_poses"][0].float().cpu()  # [S, 4, 4]
        pred_intrs = predictions["camera_intrs"][0].float().cpu()  # [S, 3, 3]

        # Use original image dimensions for COLMAP cameras
        orig_w, orig_h = pil_images[0].size
        write_colmap_reconstruction(
            output_dir, pred_c2w, pred_intrs,
            W_wm, H_wm, orig_w, orig_h, image_names,
        )
        print("  Saved COLMAP sparse reconstruction")

    # Render from predicted cameras and save individual frames + interpolated video
    if args.save_rendered and "splats" in predictions:
        splats = predictions["splats"]
        cam_poses = predictions["camera_poses"]   # [1, S, 4, 4]
        cam_intrs = predictions["camera_intrs"]   # [1, S, 3, 3]

        # Render from each input camera pose and save as PNGs
        with torch.no_grad():
            rendered_colors, rendered_depths, _ = model.gs_renderer.rasterizer.rasterize_batches(
                splats["means"][:1], splats["quats"][:1],
                splats["scales"][:1], splats["opacities"][:1],
                splats["sh"][:1] if "sh" in splats else splats["colors"][:1],
                cam_poses.float(), cam_intrs.float(),
                width=W_wm, height=H_wm,
                sh_degree=model.gs_renderer.sh_degree if "sh" in splats else None,
            )
        # rendered_colors: [1, S, H, W, 3]
        rgb_dir = output_dir / "rendered_rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        for i in range(S):
            frame = (rendered_colors[0, i].clamp(0, 1) * 255).to(
                torch.uint8
            ).cpu().numpy()
            Image.fromarray(frame).save(str(rgb_dir / f"{image_names[i]}.png"))
        print(f"  Saved {S} rendered frames to rendered_rgb/")

        # Per-view rendered frames (each view's own Gaussians only)
        if args.is_video and per_view_splats is not None:
            pv_rgb_dir = output_dir / "per_view_rendered"
            pv_rgb_dir.mkdir(parents=True, exist_ok=True)
            sh_deg = model.gs_renderer.sh_degree
            with torch.no_grad():
                for i, vs in enumerate(per_view_splats):
                    pv_colors_key = vs["sh"][:1] if "sh" in vs else vs["colors"][:1]
                    pv_rendered, _, _ = model.gs_renderer.rasterizer.rasterize_batches(
                        vs["means"][:1], vs["quats"][:1],
                        vs["scales"][:1], vs["opacities"][:1],
                        pv_colors_key,
                        cam_poses[:, i:i+1].float(), cam_intrs[:, i:i+1].float(),
                        width=W_wm, height=H_wm,
                        sh_degree=sh_deg if "sh" in vs else None,
                    )
                    frame = (pv_rendered[0, 0].clamp(0, 1) * 255).to(
                        torch.uint8
                    ).cpu().numpy()
                    Image.fromarray(frame).save(
                        str(pv_rgb_dir / f"{image_names[i]}.png")
                    )
            print(f"  Saved {S} per-view rendered frames to per_view_rendered/")

        # Interpolated video
        render_interpolated_video(
            model.gs_renderer,
            splats,
            cam_poses, cam_intrs,
            (H_wm, W_wm),
            output_dir / "rendered",
            interp_per_pair=15,
            loop_reverse=(S == 1),
        )
        print("  Saved rendered video")

    print("Done.")


if __name__ == "__main__":
    main()
