# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HunyuanWorld-Mirror is a feed-forward model for universal 3D geometric prediction (by Tencent). Given input images, it simultaneously predicts point clouds, depth maps, surface normals, camera parameters, and 3D Gaussians in a single forward pass. It supports optional conditioning on geometric priors (camera poses, intrinsics, depth maps).

## Common Commands

### Environment Setup
```bash
conda create -n hunyuanworld-mirror python=3.10 cmake=3.14.0 -y
conda activate hunyuanworld-mirror
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
```

### Inference
```bash
# CLI inference with full output options
python infer.py --input_path /path/to/images --output_path /path/to/output --save_colmap --save_gs

# Gradio web demo (requires requirements_demo.txt)
python app.py
```

### Training
```bash
# Stage 1: point map, camera, depth, normal heads (no Gaussian splatting)
python training/launch.py train=stage1.yaml

# Stage 2: 3D Gaussian splatting head (freezes stage 1 weights)
python training/launch.py train=stage2.yaml

# Custom config with selectable heads
python training/launch.py train=custom.yaml

# Resume from checkpoint
python training/launch.py train=stage1.yaml ckpt_path=path/to/checkpoint.ckpt

# Restrict GPUs
CUDA_VISIBLE_DEVICES=0,1 python training/launch.py train=stage1.yaml
```

### Evaluation
```bash
python training/launch.py --config-name eval.yaml eval=pointmap.yaml   # Point cloud reconstruction
python training/launch.py --config-name eval.yaml eval=normal.yaml     # Surface normal estimation
python training/launch.py --config-name eval.yaml eval=nvs.yaml       # Novel view synthesis
python training/launch.py --config-name eval.yaml eval=depthmap.yaml  # Depth estimation
python training/launch.py --config-name eval.yaml eval=pose.yaml      # Camera pose estimation

# Evaluate a specific checkpoint
python training/launch.py --config-name eval.yaml eval=pointmap.yaml wrapper.pretrained=path/to/checkpoint.ckpt
```

### Post-inference 3DGS Optimization
```bash
python submodules/gsplat/examples/simple_trainer_worldmirror.py default \
    --data_factor 1 --data_dir /path/to/inference_output --result_dir /path/to/gs_output
```

## Architecture

### Model Pipeline

```
Input Images [B, N, 3, H, W]
    |
    v
PatchEmbed (DINOv2 ViT-L/14) --> tokens [B, N, num_patches, 1024]
    |
    v  (optional prior conditioning: depth/poses/intrinsics tokens)
VisualGeometryTransformer (24 layers, 16 heads, alternating attention)
    |
    +---> CameraHead --> camera poses [N, 4, 4], intrinsics [N, 3, 3]
    +---> DPTHead (pts) --> point clouds [N, H, W, 3] + confidence
    +---> DPTHead (depth) --> depth maps [N, H, W, 1] + confidence
    +---> DPTHead (normal) --> normals [N, H, W, 3] + confidence
    +---> DPTHead (gs) + GaussianSplatRenderer --> 3D Gaussians (means, opacities, scales, quats, sh)
```

### Key Source Modules

- `src/models/models/worldmirror.py` - Main `WorldMirror` model class (inherits `nn.Module` + `PyTorchModelHubMixin`)
- `src/models/models/visual_transformer.py` - `VisualGeometryTransformer` backbone with alternating attention and RoPE
- `src/models/models/rasterization.py` - `GaussianSplatRenderer` for novel view synthesis via 3DGS
- `src/models/heads/camera_head.py` - Camera pose prediction head
- `src/models/heads/dense_head.py` - DPT-based dense prediction heads (shared architecture for pts/depth/normal/gs)
- `src/models/utils/` - Geometry, camera math, rotation, spherical harmonics, prior normalization

### Inference Code

- `infer.py` - Full CLI inference pipeline: image/video loading, model inference, saving, visualization, video rendering
- `app.py` - Gradio web demo with interactive parameter controls
- `src/utils/` - Inference utilities: image preprocessing, output saving, rendering, COLMAP export

### Training System

Built on PyTorch Lightning + Hydra for configuration management.

- `training/launch.py` - Entry point; routes to `train()` or `eval()` based on `--config-name`
- `training/wrapper.py` - `WorldMirrorWrapper(LightningModule)`: training/validation step logic, optimizer setup, evaluation metric computation
- `training/losses/container.py` - `LossContainer` aggregating weighted loss components
- `training/losses/` - Individual losses: camera, point, depth, normal, render (MSE, perceptual, depth)
- `training/data/datamodule.py` - `WorldMirrorDataModule` with dynamic batch sampling
- `training/data/train/hypersim.py` - HyperSim training dataset
- `training/data/eval/` - 15+ evaluation dataset implementations (DTU, NYUv2, ScanNet, RealEstate10K, DL3DV, etc.)
- `training/utils/eval/` - Per-task evaluation metrics

### Configuration (Hydra)

Config root: `training/configs/`

```
configs/
  train.yaml          # Default training config (100K steps, DDP, bf16, val every 1000 steps)
  eval.yaml           # Default evaluation config
  train/              # Training stage overrides (stage1, stage2, custom, all)
  eval/               # Evaluation task configs (pointmap, normal, nvs, depthmap, pose)
  wrapper/worldmirror.yaml  # Model + optimizer + scheduler + loss definitions
  data/default.yaml   # DataModule defaults (24 images/GPU, dynamic resolution)
  paths/default.yaml  # Dataset root paths (must be configured per environment)
```

### Two-Stage Training

- **Stage 1** (`train/stage1.yaml`): Trains camera, point map, depth, and normal heads with conditioning enabled. Backbone is trainable. Loss weights: `[5.0, 1.0, 1.0, 1.0, 0.1, 1.0, 0.05, 0.1]`.
- **Stage 2** (`train/stage2.yaml`): Freezes backbone + stage 1 heads, trains only the GS head and renderer. Loss weights: `[0.1, 1.0, 0.05, 0.1]`.

### Optimizer Configuration

Three learning rate groups: `pretrained` (patch embed, 2e-5), `backbone` (transformer, 1e-4), `new` (prediction heads, 2e-4). All use AdamW with betas `[0.9, 0.95]` and weight decay 0.1. Scheduler: WarmupCosineAnnealing.

## Key Patterns

- Model weights auto-download from HuggingFace: `WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror")`
- Conditioning flags `cond_flags = [depth, poses, intrinsics]` control which priors are fed to the model (0=off, 1=on)
- Image input convention: `[B, N, 3, H, W]` normalized to `[0, 1]`, standard size 518x518
- Camera convention: OpenCV (camera-to-world poses)
- Predictions include confidence maps alongside each output
- `rootutils.setup_root(__file__, indicator="License.txt", pythonpath=True)` in launch.py sets project root on PYTHONPATH
