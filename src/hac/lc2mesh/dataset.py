"""Dataset utilities for loading asteroid lightcurve data and spherical mesh priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import trimesh

from hac.constants import OBSERVERS
from hac.data import load_data
from hac.lc2mesh.forward_model import observers_to_unit_tensor


@dataclass
class SphericalMeshPrior:
    """Spherical mesh prior (icosphere) for network input."""

    vertices: torch.Tensor  # [n_vertices, 3]
    faces: torch.Tensor  # [n_faces, 3]
    normals: torch.Tensor  # [n_faces, 3]
    centroids: torch.Tensor  # [n_faces, 3]
    areas: torch.Tensor  # [n_faces]
    n_faces: int


@dataclass
class AsteroidData:
    """Container for asteroid lightcurve observations and camera geometry."""

    asteroid_id: int
    camera_names: Sequence[str]
    e_obs: torch.Tensor  # [n_cameras, 3] observer unit vectors
    e_sun: torch.Tensor  # [3] sun unit vector
    phases: torch.Tensor  # [n_phases] rotation phases in radians
    observed: torch.Tensor  # [n_cameras, n_phases] raw observed lightcurves
    observed_norm: torch.Tensor  # [n_cameras, n_phases] normalized observations
    n_cameras: int
    n_phases: int
    device: torch.device
    dtype: torch.dtype


def create_spherical_prior(
    subdivisions: int = 3,
    radius: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> SphericalMeshPrior:
    """Create a unit icosphere mesh as the spherical prior.

    Parameters
    ----------
    subdivisions:
        Number of icosphere subdivision levels (3 → ~320 faces, 4 → ~1280 faces).
    radius:
        Sphere radius.
    device:
        Target PyTorch device.
    dtype:
        Target PyTorch dtype.

    Returns
    -------
    SphericalMeshPrior
        Spherical mesh geometry with vertices, faces, normals, centroids, areas.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    vertices_np = np.asarray(sphere.vertices, dtype=np.float32)
    faces_np = np.asarray(sphere.faces, dtype=np.int64)

    vertices = torch.as_tensor(vertices_np, dtype=dtype, device=device)
    faces = torch.as_tensor(faces_np, dtype=torch.long, device=device)

    # Compute per-face normals, centroids, and areas
    v0 = vertices[faces[:, 0]]  # [n_faces, 3]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    centroids = (v0 + v1 + v2) / 3.0  # [n_faces, 3]

    # Cross product for normals and areas
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = torch.cross(edge1, edge2, dim=1)  # [n_faces, 3]
    areas = 0.5 * torch.norm(cross, dim=1)  # [n_faces]
    normals = cross / torch.norm(cross, dim=1, keepdim=True).clamp_min(1e-12)

    return SphericalMeshPrior(
        vertices=vertices,
        faces=faces,
        normals=normals,
        centroids=centroids,
        areas=areas,
        n_faces=int(faces.shape[0]),
    )


def load_asteroid_data(
    asteroid_id: int,
    curve_type: str = "intensity",
    phase_direction: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> AsteroidData:
    """Load real lightcurve observations for a given asteroid.

    Parameters
    ----------
    asteroid_id:
        Target asteroid model number (1, 2, etc.).
    curve_type:
        Type of lightcurve data ('intensity', 'magnitude', etc.).
    phase_direction:
        Sign for phase angle direction (+1 or -1).
    device:
        Target PyTorch device.
    dtype:
        Target PyTorch dtype.

    Returns
    -------
    AsteroidData
        Asteroid observations with camera geometry, phases, and normalized lightcurves.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load camera geometry
    camera_names, e_obs = observers_to_unit_tensor(OBSERVERS, device=device, dtype=dtype)

    # Sun direction (fixed)
    e_sun = torch.tensor([-1.0, 0.0, 0.0], dtype=dtype, device=device)

    # Load real lightcurve data
    ds_real = load_data(asteroid_id, origin="real", curve_type=curve_type)
    observed_df = ds_real.data.copy()

    observed = torch.as_tensor(
        observed_df[camera_names].to_numpy(dtype=np.float32, copy=True).T,
        dtype=dtype,
        device=device,
    )

    # Normalize per camera
    observed_norm = observed / observed.mean(dim=1, keepdim=True).clamp_min(1e-12)

    n_cameras, n_phases = observed.shape

    # Phase angles
    phases = (
        phase_direction
        * torch.arange(n_phases, dtype=dtype, device=device)
        * (2.0 * torch.pi / n_phases)
    )

    return AsteroidData(
        asteroid_id=asteroid_id,
        camera_names=camera_names,
        e_obs=e_obs,
        e_sun=e_sun,
        phases=phases,
        observed=observed,
        observed_norm=observed_norm,
        n_cameras=n_cameras,
        n_phases=n_phases,
        device=device,
        dtype=dtype,
    )
