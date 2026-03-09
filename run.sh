# CUDA_VISIBLE_DEVICES=0 python evaluate_dl3dv.py --save_splats --render_hires
# CUDA_VISIBLE_DEVICES=0 python evaluate_re10k.py --save_splats --render_hires --skip_existing /media/stefano/0D91176038319865/worldmirror/worldmirror_re10k_8v

CUDA_VISIBLE_DEVICES=0 python evaluate_davis.py \
    --scene_name car-turn \
    --mask_gaussians \
    --save_splats \
    --stride 10 \
    --save_rendered \
    --save_depth \
    --save_colmap \
    --is_video