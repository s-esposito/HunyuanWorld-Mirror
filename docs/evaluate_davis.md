# DAVIS Evaluation Script

Run HunyuanWorld-Mirror on DAVIS video sequences to reconstruct 3D Gaussians, point clouds, depth maps, and COLMAP models from unposed RGB frames. Optionally use ground truth segmentation masks to isolate foreground objects.

## Prerequisites

- HunyuanWorld-Mirror environment (see main README)
- DAVIS dataset with Full-Resolution images and (optionally) unsupervised annotations

### Expected dataset layout

```
/path/to/davis/
  JPEGImages/Full-Resolution/{scene_name}/*.jpg
  Annotations_unsupervised/Full-Resolution/{scene_name}/*.png   # optional
```

Segmentation masks are indexed PNGs where non-zero pixels indicate foreground.

## Quick start

```bash
# Reconstruct the default scene with Gaussian splats and depth
CUDA_VISIBLE_DEVICES=0 python evaluate_davis.py \
    --scene_name car-turn --save_splats --save_depth

# Foreground-only reconstruction (both masking modes)
CUDA_VISIBLE_DEVICES=0 python evaluate_davis.py \
    --scene_name car-turn --mask_input --mask_gaussians \
    --save_splats --save_pointmap --save_rendered

# Per-view Gaussians from video (every 10th frame, foreground only)
CUDA_VISIBLE_DEVICES=0 python evaluate_davis.py \
    --scene_name car-turn --is_video --mask_gaussians \
    --save_splats --save_rendered --stride 10
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_root` | `/home/stefano/Data/davis` | Path to the DAVIS dataset root |
| `--scene_name` | `car-turn` | Scene directory name |
| `--output_dir` | `output/worldmirror_davis` | Output root (scene name appended automatically) |
| `--target_size` | `518` | WorldMirror input resolution |
| `--stride` | `1` | Frame subsampling stride |
| `--max_frames` | `-1` | Max frames to load (`-1` = all) |
| `--device` | `cuda` | Torch device |

### Masking options

| Flag | Effect |
|---|---|
| `--mask_input` | Black out background pixels in input images before feeding to the model |
| `--mask_gaussians` | Filter output Gaussians to keep only foreground (per-pixel, before voxel pruning) |
| `--is_video` | Treat each view independently: split Gaussians per view, save/render per-view PLYs and frames |

All flags are independent and can be combined. When none are set, the model runs completely unmodified.

### Save options

| Flag | Output |
|---|---|
| `--save_splats` | `gaussians.ply` — 3D Gaussian splat PLY |
| `--save_depth` | `depth/` — per-frame depth maps (PNG + NPY) |
| `--save_colmap` | `sparse/0/` — COLMAP binary model from predicted cameras |
| `--save_pointmap` | `pointmap.ply` — colored point cloud (masked if masks available) |
| `--save_rendered` | `rendered_rgb/` — per-camera rendered PNGs, plus interpolated video |

## Output structure

```
output/worldmirror_davis/{scene_name}/
  images/              # Input frames at WorldMirror resolution
  masks/               # Foreground masks at WM resolution (if masking used)
  depth/               # Depth maps: {frame}.png + {frame}.npy
  gaussians.ply        # 3D Gaussian splats (merged across all views)
  pointmap.ply         # Colored point cloud
  sparse/0/            # COLMAP binary (cameras.bin, images.bin, points3D.bin)
  rendered_rgb/        # Per-camera rendered PNGs (all Gaussians)
  rendered_rgb.mp4     # Interpolated flythrough video
  rendered_depth.mp4   # Interpolated depth video
  per_view_splats/     # Per-view Gaussian PLYs (--is_video only)
  per_view_rendered/   # Per-view rendered PNGs (--is_video only)
```

## How masking works

### `--mask_input`

Multiplies each input image by its binary segmentation mask before preprocessing. Background pixels become black (zero). This causes the model to see only the foreground object, which can improve reconstruction quality for isolated objects but removes all scene context.

### `--mask_gaussians`

Operates on the model's output rather than its input. The model produces one Gaussian per pixel per source view (`S * H * W` total). Before the renderer's voxel pruning step merges and reorders Gaussians (breaking the pixel-to-Gaussian correspondence), this option:

1. Temporarily disables pruning and confidence filtering
2. Runs inference to get raw per-pixel Gaussians
3. Applies the foreground mask to discard background Gaussians
4. Re-runs voxel pruning on the remaining foreground Gaussians
5. Restores the original renderer settings

This preserves the standard pruning behavior on the foreground subset while cleanly removing background geometry.

### `--is_video`

Treats each input view as an independent source of Gaussians. Instead of mixing all views' Gaussians into a single set, the raw per-pixel Gaussians (`H * W` per view) are split into S independent groups. Each group is optionally masked (if `--mask_gaussians`) and pruned independently.

Outputs with `--is_video`:
- `per_view_splats/{frame}.ply` — one Gaussian PLY per view (with `--save_splats`)
- `per_view_rendered/{frame}.png` — each view rendered from its own Gaussians only (with `--save_rendered`)
- The merged `gaussians.ply` and interpolated video are still produced from all views combined.

### Combining flags

| Combination | Effect |
|---|---|
| `--mask_input --mask_gaussians` | Strongest foreground isolation: model sees only foreground, residual background Gaussians filtered out |
| `--is_video --mask_gaussians` | Per-view foreground Gaussians, each view masked and pruned independently |
| `--is_video --mask_input --mask_gaussians` | All three: masked input, per-view split, per-view foreground filtering |

## Inference mode

The script runs in fully unposed mode with `cond_flags=[0, 0, 0]` (no depth, pose, or intrinsics conditioning). The model predicts all camera parameters, depth, normals, point clouds, and Gaussians purely from the RGB frames.
