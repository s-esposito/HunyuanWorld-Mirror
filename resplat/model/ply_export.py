from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor
import math


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    extrinsics: Float[Tensor, "4 4"],
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
    align_to_view: bool = True,  # whether to align world space to the view space (camera space) of the extrinsics
    save_gaussian_npz: bool = False,
):
    if align_to_view:
        view_rotation = extrinsics[:3, :3].inverse()
        # Apply the rotation to the means (Gaussian positions).
        means = einsum(view_rotation, means, "i j, ... j -> ... i")

        # Apply the rotation to the Gaussian rotations.
        rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
        rotations = view_rotation.detach().cpu().numpy() @ rotations
        rotations = R.from_matrix(rotations).as_quat()
        x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
        rotations = np.stack((w, x, y, z), axis=-1)
    else:
        rotations = rotations.detach().cpu().numpy()

    num_rest = 3 * (harmonics.shape[-1] - 1)

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(num_rest)]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = (
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        harmonics[..., 0].detach().cpu().contiguous().numpy(),
        harmonics[..., 1:].flatten(start_dim=1).detach().cpu().contiguous().numpy(),
        torch.logit(opacities[..., None]).detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    )

    if save_gaussian_npz:
        gaussian_dict = {
            'mean': attributes[0],
            'log_scale': attributes[-2],
            'rotation': attributes[-1],
            'logit_opacity': attributes[-3],
            'color': attributes[3],
        }

        path.parent.mkdir(exist_ok=True, parents=True)

        npz_path = str(path)[:-3] + 'npz'
        np.savez(npz_path, gaussian_dict)

    
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
    

def save_gaussian_ply(gaussians, visualization_dump, example, save_path, 
    save_all_gaussians=True,  # no trim
    no_align_to_view=False,
    save_gaussian_npz=False,
    ):

    v, _, h, w = example["context"]["image"].shape[1:]

    if gaussians.means.shape[1] != v * h * w:
        # latent gaussians
        scale = v * h * w / gaussians.means.shape[1]
        scale = int(math.sqrt(scale))
        h = h // scale
        w = w // scale

    # Transform means into camera space.
    means = rearrange(
        gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=v, h=h, w=w
    )

    # Create a mask to filter the Gaussians. Throw away Gaussians at the
    # borders, since they're generally of lower quality.
    mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
    GAUSSIAN_TRIM = 2
    mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

    def trim(element):
        element = rearrange(
            element, "() (v h w spp) ... -> h w spp v ...", v=v, h=h, w=w
        )
        return element[mask][None]

    if save_all_gaussians:
        world_rotations = gaussians.rotations[0]
        export_ply(
            example["context"]["extrinsics"][0, v//2],
            gaussians.means[0],
            gaussians.scales[0],
            world_rotations,
            gaussians.harmonics[0],
            gaussians.opacities[0],
            save_path,
            align_to_view=not no_align_to_view,
            save_gaussian_npz=save_gaussian_npz,
        )
    else:
        world_rotations = trim(gaussians.rotations)[0]

        # Align the viewpoint to the middle frame
        export_ply(
            example["context"]["extrinsics"][0, v//2],
            trim(gaussians.means)[0],
            trim(gaussians.scales)[0],
            world_rotations,
            trim(gaussians.harmonics)[0],
            trim(gaussians.opacities)[0],
            save_path,
            align_to_view=not no_align_to_view,
            save_gaussian_npz=save_gaussian_npz,
        )
