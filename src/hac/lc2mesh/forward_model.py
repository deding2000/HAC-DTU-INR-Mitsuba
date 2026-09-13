"""Differentiable asteroid forward photometric model utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from stl import mesh


def normalize_vectors(
    vectors: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize vectors to unit length along ``dim``."""
    return vectors / vectors.norm(dim=dim, keepdim=True).clamp_min(float(eps))


def observer_geometry_to_unit_vector(
    theta_deg: float,
    alpha_deg: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert HAC observer angles to a unit direction vector."""
    theta = torch.deg2rad(
        torch.tensor(float(theta_deg) - 180.0, dtype=dtype, device=device)
    )
    alpha = torch.deg2rad(torch.tensor(float(alpha_deg), dtype=dtype, device=device))

    vec = torch.stack(
        [
            torch.cos(theta) * torch.cos(alpha),
            torch.sin(theta) * torch.cos(alpha),
            torch.sin(alpha),
        ],
        dim=0,
    )
    return normalize_vectors(vec, dim=0)


def observers_to_unit_tensor(
    observer_map: Mapping[str, object],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[list[str], torch.Tensor]:
    """Build ordered observer names and stacked observer vectors."""
    names = list(observer_map.keys())
    e_obs = [
        observer_geometry_to_unit_vector(
            float(observer_map[name].theta),
            float(observer_map[name].alpha),
            device=device,
            dtype=dtype,
        )
        for name in names
    ]
    return names, torch.stack(e_obs, dim=0)


def load_stl_geometry_torch(
    stl_path: str | Path,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Load STL triangles and derived geometric tensors as torch tensors."""
    stl_mesh = mesh.Mesh.from_file(str(stl_path))
    triangles_np = stl_mesh.vectors.astype(np.float32)

    edge1 = triangles_np[:, 1, :] - triangles_np[:, 0, :]
    edge2 = triangles_np[:, 2, :] - triangles_np[:, 0, :]
    cross = np.cross(edge1, edge2)

    twice_area = np.linalg.norm(cross, axis=1)
    normals_np = cross / np.maximum(twice_area[:, None], 1e-12)
    areas_np = 0.5 * twice_area
    centroids_np = triangles_np.mean(axis=1)

    return {
        "triangles": torch.as_tensor(triangles_np, dtype=dtype, device=device),
        "normals": torch.as_tensor(normals_np, dtype=dtype, device=device),
        "areas": torch.as_tensor(areas_np, dtype=dtype, device=device),
        "centroids": torch.as_tensor(centroids_np, dtype=dtype, device=device),
    }


def fit_scale_offset_vectorized(
    model_normalized: torch.Tensor,
    observed: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form per-camera affine calibration coefficients."""
    x = model_normalized
    y = observed

    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    cov = ((x - x_mean) * (y - y_mean)).mean(dim=1)
    var = ((x - x_mean) ** 2).mean(dim=1)

    k = cov / var.clamp_min(float(eps))
    b = y_mean[:, 0] - k * x_mean[:, 0]
    return k, b


def fourier_lowpass_periodic(
    curves: torch.Tensor,
    n_harmonics: int,
) -> torch.Tensor:
    """Periodic Fourier low-pass smoothing of curves ``[..., P]``.

    Keeps the DC component plus the first ``n_harmonics`` harmonics and zeroes
    the rest. Fully differentiable with respect to ``curves``, so it can be used
    inside optimization loops as built-in forward-output smoothing.
    """
    n_phases = int(curves.shape[-1])
    if n_phases < 8:
        return curves

    h = int(max(1, min(int(n_harmonics), n_phases // 2 - 1)))
    coeff = torch.fft.rfft(curves, dim=-1)
    keep = torch.zeros(coeff.shape[-1], dtype=torch.bool, device=curves.device)
    keep[: h + 1] = True
    coeff = torch.where(keep, coeff, torch.zeros_like(coeff))
    return torch.fft.irfft(coeff, n=n_phases, dim=-1)


class AsteroidForwardModel(torch.nn.Module):
    """Differentiable LS+Lambert asteroid forward operator implemented in PyTorch."""

    def __init__(
        self,
        normals: torch.Tensor,
        *,
        lambert_weight: float = 0.1,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        normals_t = normalize_vectors(normals.to(dtype=torch.float32), dim=1, eps=eps)
        self.register_buffer("normals", normals_t)

        self.lambert_weight = float(lambert_weight)
        self.eps = float(eps)

    @property
    def n_facets(self) -> int:
        return int(self.normals.shape[0])

    @staticmethod
    def rotate_vectors_z(vectors: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """Rotate vectors around +z for all phase angles."""
        c = torch.cos(phases)
        s = torch.sin(phases)

        if vectors.ndim == 1:
            x = c * vectors[0] - s * vectors[1]
            y = s * vectors[0] + c * vectors[1]
            z = torch.ones_like(x) * vectors[2]
            return torch.stack([x, y, z], dim=-1)

        if vectors.ndim == 2:
            x = c[:, None] * vectors[None, :, 0] - s[:, None] * vectors[None, :, 1]
            y = s[:, None] * vectors[None, :, 0] + c[:, None] * vectors[None, :, 1]
            z = vectors[None, :, 2].expand_as(x)
            return torch.stack([x, y, z], dim=-1)

        raise ValueError("vectors must have shape [3] or [N, 3]")

    def build_forward_matrix_components(
        self,
        phases: torch.Tensor,
        e_sun: torch.Tensor,
        e_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build LS/Lambert component matrices ``(A_ls, A_prod)`` of shape [C, P, F].

        The full matrix for any Lambert weight ``w`` is ``A(w) = A_ls + w * A_prod``;
        visibility masks and phase factors are computed only once, which makes
        Lambert-weight sweeps cheap.
        """
        phases = phases.to(dtype=self.normals.dtype, device=self.normals.device)
        e_sun = normalize_vectors(
            e_sun.to(dtype=self.normals.dtype, device=self.normals.device),
            dim=-1,
            eps=self.eps,
        )
        e_obs = normalize_vectors(
            e_obs.to(dtype=self.normals.dtype, device=self.normals.device),
            dim=-1,
            eps=self.eps,
        )

        e_sun_rot = self.rotate_vectors_z(e_sun, phases)  # [P, 3]
        e_obs_rot = self.rotate_vectors_z(e_obs, phases)  # [P, C, 3]

        mu0 = torch.einsum("pd,fd->pf", e_sun_rot, self.normals)[:, None, :]  # [P,1,F]
        mu = torch.einsum("pcd,fd->pcf", e_obs_rot, self.normals)  # [P,C,F]

        mu0_pos = torch.clamp(mu0, min=0.0)
        mu_pos = torch.clamp(mu, min=0.0)
        prod = mu_pos * mu0_pos

        ls_term = prod / (mu_pos + mu0_pos + self.eps)

        cos_alpha = torch.einsum("pd,pcd->pc", e_sun_rot, e_obs_rot)
        cos_alpha = torch.clamp(cos_alpha, min=-1.0, max=1.0)
        alpha = torch.acos(cos_alpha)

        visible = (mu0 > 0.0) & (mu > 0.0)
        a_ls = torch.where(
            visible,
            ls_term,
            torch.zeros_like(ls_term),
        )
        a_prod = torch.where(
            visible,
            prod,
            torch.zeros_like(prod),
        )
        return (
            a_ls.permute(1, 0, 2).contiguous(),
            a_prod.permute(1, 0, 2).contiguous(),
        )  # [C,P,F] each

    def build_forward_matrix(
        self,
        phases: torch.Tensor,
        e_sun: torch.Tensor,
        e_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Build LS+Lambert matrix ``A`` with shape [n_cams, n_phases, n_facets]."""
        a_ls, a_prod = self.build_forward_matrix_components(
            phases=phases, e_sun=e_sun, e_obs=e_obs
        )
        if self.lambert_weight == 0.0:
            return a_ls
        return a_ls + self.lambert_weight * a_prod

    def predict_raw_from_matrix(self, g: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Compute raw predictions ``A g`` from a precomputed matrix ``A``.

        Parameters
        ----------
        g:
            Facet weights with shape ``[n_facets]``.
        A:
            Forward matrix with shape ``[n_cameras, n_phases, n_facets]``.
        """
        g_t = g.to(dtype=self.normals.dtype, device=self.normals.device)
        if g_t.ndim != 1 or g_t.shape[0] != self.n_facets:
            raise ValueError(f"g must have shape [{self.n_facets}], got {tuple(g_t.shape)}")

        a_t = A.to(dtype=self.normals.dtype, device=self.normals.device)
        if a_t.ndim != 3:
            raise ValueError(f"A must have 3 dims [C, P, F], got {tuple(a_t.shape)}")
        if a_t.shape[-1] != self.n_facets:
            raise ValueError(
                f"A last dimension must equal n_facets={self.n_facets}, got {a_t.shape[-1]}"
            )

        return torch.einsum("cpf,f->cp", a_t, g_t)

    def predict_normalized_from_matrix(self, g: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Compute per-camera normalized predictions from precomputed matrix ``A``."""
        raw = self.predict_raw_from_matrix(g=g, A=A)
        return raw / raw.mean(dim=1, keepdim=True).clamp_min(self.eps)

    def evaluate_raw(
        self,
        g: torch.Tensor,
        *,
        phases: torch.Tensor,
        e_sun: torch.Tensor,
        e_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(A, A g)`` where ``A g`` has shape [n_cams, n_phases]."""
        A = self.build_forward_matrix(phases=phases, e_sun=e_sun, e_obs=e_obs)
        raw = self.predict_raw_from_matrix(g=g, A=A)
        return A, raw

    def forward(
        self,
        g: torch.Tensor,
        *,
        phases: torch.Tensor,
        e_sun: torch.Tensor,
        e_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-camera normalized predictions from facet weights ``g``."""
        A = self.build_forward_matrix(phases=phases, e_sun=e_sun, e_obs=e_obs)
        return self.predict_normalized_from_matrix(g=g, A=A)