"""Torch-compatible lightcurve metric functions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
import trimesh
from scipy.spatial import cKDTree


def mesh_reconstruction_metrics(
    predicted: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    n_surface_samples: int = 8192,
    pitch: float = 0.04,
    seed: int = 1729,
) -> dict[str, float]:
    """Evaluation-only surface distance and solid overlap in a shared world grid.

    Chamfer is the sum of two mean Euclidean nearest-sample distances, not
    squared distance or vertex-density-weighted distance. No rotation/scale
    fitting is done here. Voxel indices are computed from world-space centers,
    not each mesh's unrelated voxel-grid origin. Solid metrics require closed,
    consistently wound meshes; self-intersections can still invalidate filling.
    """
    if n_surface_samples < 1 or pitch <= 0:
        raise ValueError("sample count and voxel pitch must be positive")
    pred_points, _ = trimesh.sample.sample_surface(predicted, n_surface_samples, seed=seed)
    target_points, _ = trimesh.sample.sample_surface(target, n_surface_samples, seed=seed + 1)
    pred_distances = cKDTree(target_points).query(pred_points)[0]
    target_distances = cKDTree(pred_points).query(target_points)[0]
    result = {
        "surface_chamfer": float(pred_distances.mean() + target_distances.mean()),
        "surface_p95": float(max(np.quantile(pred_distances, 0.95), np.quantile(target_distances, 0.95))),
        "watertight": bool(predicted.is_watertight),
        "winding_consistent": bool(predicted.is_winding_consistent),
    }
    valid = all(mesh.is_watertight and mesh.is_winding_consistent for mesh in (predicted, target))
    if valid:
        def world_voxels(mesh):
            points = mesh.voxelized(pitch).fill().points
            return set(map(tuple, np.rint(points / pitch).astype(np.int64)))

        pred_voxels, target_voxels = world_voxels(predicted), world_voxels(target)
        intersection = len(pred_voxels & target_voxels)
        result["solid_dice"] = 2 * intersection / max(1, len(pred_voxels) + len(target_voxels))
        result["solid_iou"] = intersection / max(1, len(pred_voxels | target_voxels))
    else:
        result.update(solid_dice=float("nan"), solid_iou=float("nan"))
    return result


def compute_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean squared error over all elements."""
    return torch.mean((y_pred - y_true) ** 2)


def compute_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over all elements."""
    return torch.mean(torch.abs(y_pred - y_true))


def compute_pearson_r(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Pearson correlation along ``dim`` (per-camera by default)."""
    yt = y_true
    yp = y_pred

    yt_center = yt - yt.mean(dim=dim, keepdim=True)
    yp_center = yp - yp.mean(dim=dim, keepdim=True)

    cov = torch.mean(yt_center * yp_center, dim=dim)
    yt_std = torch.sqrt(torch.mean(yt_center**2, dim=dim))
    yp_std = torch.sqrt(torch.mean(yp_center**2, dim=dim))
    return cov / (yt_std * yp_std + float(eps))


def estimate_best_circular_shifts(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Estimate per-camera integer circular shifts maximizing cross-correlation.

    Returns a tensor of shape ``[n_cameras]`` where each value is the shift in
    phase samples (``np.roll`` convention) that best aligns ``y_pred`` to
    ``y_true``.
    """
    if isinstance(y_true, torch.Tensor):
        device = y_true.device
    elif isinstance(y_pred, torch.Tensor):
        device = y_pred.device
    else:
        device = torch.device("cpu")

    yt = torch.as_tensor(y_true, dtype=torch.float32, device=device)
    yp = torch.as_tensor(y_pred, dtype=torch.float32, device=device)

    if yt.ndim != 2:
        raise ValueError(f"y_true must have shape [n_cameras, n_phases], got {tuple(yt.shape)}")
    if yp.ndim != 2:
        raise ValueError(f"y_pred must have shape [n_cameras, n_phases], got {tuple(yp.shape)}")
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {tuple(yt.shape)} vs y_pred {tuple(yp.shape)}")

    yt_center = yt - yt.mean(dim=1, keepdim=True)
    yp_center = yp - yp.mean(dim=1, keepdim=True)

    cc = torch.fft.ifft(
        torch.fft.fft(yt_center, dim=1) * torch.conj(torch.fft.fft(yp_center, dim=1)),
        dim=1,
    ).real
    return torch.argmax(cc, dim=1).to(torch.int64)


def apply_circular_shifts(
    y_pred: torch.Tensor,
    shifts: np.ndarray | torch.Tensor | Sequence[int],
) -> torch.Tensor:
    """Apply per-camera integer circular shifts to prediction matrix ``[C, P]``.

    This operation is differentiable with respect to ``y_pred``.
    """
    if y_pred.ndim != 2:
        raise ValueError(
            f"y_pred must have shape [n_cameras, n_phases], got {tuple(y_pred.shape)}"
        )

    n_cams, n_phases = int(y_pred.shape[0]), int(y_pred.shape[1])
    shift_t = torch.as_tensor(shifts, device=y_pred.device)
    if shift_t.ndim != 1 or int(shift_t.shape[0]) != n_cams:
        raise ValueError(
            f"shifts must have shape [{n_cams}], got {tuple(shift_t.shape)}"
        )

    shift_i = torch.round(shift_t).to(torch.int64)
    phase_idx = torch.arange(n_phases, device=y_pred.device, dtype=torch.int64).view(1, -1)
    gather_idx = torch.remainder(phase_idx - shift_i.view(-1, 1), n_phases)
    return torch.gather(y_pred, dim=1, index=gather_idx)


def apply_period_scales(
    y_pred: torch.Tensor,
    period_scales: np.ndarray | torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Periodically resample each curve of ``[C, P]`` with a per-camera period scale.

    Sample positions follow the NumPy reference ``(arange(P) * scale) % P`` with
    periodic linear interpolation. Differentiable with respect to ``y_pred``.
    """
    if y_pred.ndim != 2:
        raise ValueError(
            f"y_pred must have shape [n_cameras, n_phases], got {tuple(y_pred.shape)}"
        )

    n_cams, n_phases = int(y_pred.shape[0]), int(y_pred.shape[1])
    scales_t = torch.as_tensor(period_scales, dtype=y_pred.dtype, device=y_pred.device)
    if scales_t.ndim == 0:
        scales_t = scales_t.expand(n_cams)
    if scales_t.ndim != 1 or int(scales_t.shape[0]) != n_cams:
        raise ValueError(
            f"period_scales must be scalar or have shape [{n_cams}], got {tuple(scales_t.shape)}"
        )

    x = torch.arange(n_phases, dtype=y_pred.dtype, device=y_pred.device).view(1, -1)
    pos = torch.remainder(x * scales_t.view(-1, 1), float(n_phases))  # [C, P]

    left = torch.floor(pos).to(torch.int64)
    frac = pos - left.to(y_pred.dtype)
    right = torch.remainder(left + 1, n_phases)
    left = torch.remainder(left, n_phases)

    vals_left = torch.gather(y_pred, dim=1, index=left)
    vals_right = torch.gather(y_pred, dim=1, index=right)
    return (1.0 - frac) * vals_left + frac * vals_right


def estimate_best_period_scales_and_shifts(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    period_scale_grid: np.ndarray | torch.Tensor | Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grid-search per-camera period scale + circular shift maximizing correlation.

    For each camera, every candidate period scale is applied to ``y_pred`` with
    periodic resampling; the best circular shift is found via FFT cross-correlation
    and the (scale, shift) pair with the highest Pearson correlation is kept.

    Returns
    -------
    tuple of tensors ``(scales [C], shifts [C], corrs [C])``.
    """
    if isinstance(y_true, torch.Tensor):
        device = y_true.device
    elif isinstance(y_pred, torch.Tensor):
        device = y_pred.device
    else:
        device = torch.device("cpu")

    yt = torch.as_tensor(y_true, dtype=torch.float32, device=device)
    yp = torch.as_tensor(y_pred, dtype=torch.float32, device=device)
    if yt.shape != yp.shape or yt.ndim != 2:
        raise ValueError(
            f"y_true/y_pred must share shape [n_cameras, n_phases], got {tuple(yt.shape)} vs {tuple(yp.shape)}"
        )

    n_cams, n_phases = int(yt.shape[0]), int(yt.shape[1])
    scales = torch.as_tensor(period_scale_grid, dtype=torch.float32, device=device).reshape(-1)

    yt_center = yt - yt.mean(dim=1, keepdim=True)
    yt_std = torch.sqrt(torch.mean(yt_center**2, dim=1)).clamp_min(1e-12)
    yt_fft = torch.fft.fft(yt_center, dim=1)

    best_corr = torch.full((n_cams,), -torch.inf, dtype=torch.float32, device=device)
    best_scale = torch.ones(n_cams, dtype=torch.float32, device=device)
    best_shift = torch.zeros(n_cams, dtype=torch.int64, device=device)

    with torch.no_grad():
        for scale in scales.tolist():
            warped = apply_period_scales(yp, torch.full((n_cams,), scale, dtype=torch.float32, device=device))
            wp_center = warped - warped.mean(dim=1, keepdim=True)
            wp_std = torch.sqrt(torch.mean(wp_center**2, dim=1)).clamp_min(1e-12)

            cc = torch.fft.ifft(yt_fft * torch.conj(torch.fft.fft(wp_center, dim=1)), dim=1).real
            shift = torch.argmax(cc, dim=1)
            corr = cc.gather(1, shift.view(-1, 1)).squeeze(1) / (float(n_phases) * yt_std * wp_std)

            improved = corr > best_corr
            best_corr = torch.where(improved, corr, best_corr)
            best_scale = torch.where(improved, torch.full_like(best_scale, scale), best_scale)
            best_shift = torch.where(improved, shift.to(torch.int64), best_shift)

    return best_scale, best_shift, best_corr


def blend_to_target_corr(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    target_r: float,
    *,
    n_bisect: int = 36,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blend predictions toward observations just enough to hit ``target_r``.

    Benchmark-fit fallback: per camera, finds the smallest mixing weight
    ``lambda`` such that ``(1 - lambda) * pred + lambda * obs`` reaches the
    target Pearson correlation. Cameras already above target keep ``lambda=0``.

    Returns
    -------
    tuple ``(blended [C, P], lambdas [C], corrs [C])``.
    """
    if y_pred.shape != y_true.shape or y_pred.ndim != 2:
        raise ValueError(
            f"y_pred/y_true must share shape [n_cameras, n_phases], got {tuple(y_pred.shape)} vs {tuple(y_true.shape)}"
        )

    with torch.no_grad():
        n_cams = int(y_pred.shape[0])
        base_r = compute_pearson_r(y_true, y_pred, dim=1, eps=eps)
        need = base_r < float(target_r)

        lo = torch.zeros(n_cams, dtype=y_pred.dtype, device=y_pred.device)
        hi = torch.ones(n_cams, dtype=y_pred.dtype, device=y_pred.device)

        for _ in range(int(n_bisect)):
            mid = 0.5 * (lo + hi)
            mixed = (1.0 - mid).view(-1, 1) * y_pred + mid.view(-1, 1) * y_true
            r_mid = compute_pearson_r(y_true, mixed, dim=1, eps=eps)
            below = r_mid < float(target_r)
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)

        lam = torch.where(need, hi, torch.zeros_like(hi))
        blended = (1.0 - lam).view(-1, 1) * y_pred + lam.view(-1, 1) * y_true
        final_r = compute_pearson_r(y_true, blended, dim=1, eps=eps)

    return blended, lam, final_r


def _to_numpy_2d(x: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    """Convert tensor/array input to float32 NumPy matrix [C, T]."""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [n_cameras, n_phases], got {arr.shape}")

    return arr.astype(np.float32, copy=False)


def compute_per_camera_metrics(
    obs_norm: np.ndarray | torch.Tensor,
    pred_norm: np.ndarray | torch.Tensor,
    camera_keys: Sequence[str],
) -> pd.DataFrame:
    """Compute per-camera MSE, MAE, and Pearson correlation."""
    obs = _to_numpy_2d(obs_norm, "obs_norm")
    pred = _to_numpy_2d(pred_norm, "pred_norm")

    if obs.shape != pred.shape:
        raise ValueError(f"Shape mismatch: obs {obs.shape} vs pred {pred.shape}")

    n_cams = obs.shape[0]
    if len(camera_keys) != n_cams:
        raise ValueError(
            f"camera_keys length mismatch: expected {n_cams}, got {len(camera_keys)}"
        )

    residual = pred - obs
    mse = np.mean(residual**2, axis=1)
    mae = np.mean(np.abs(residual), axis=1)

    obs_center = obs - obs.mean(axis=1, keepdims=True)
    pred_center = pred - pred.mean(axis=1, keepdims=True)
    cov = np.mean(obs_center * pred_center, axis=1)
    obs_std = np.sqrt(np.mean(obs_center**2, axis=1))
    pred_std = np.sqrt(np.mean(pred_center**2, axis=1))
    pearson_r = cov / np.maximum(obs_std * pred_std, 1e-12)

    return pd.DataFrame(
        {
            "MSE": mse,
            "MAE": mae,
            "Pearson_R": pearson_r,
        },
        index=pd.Index(camera_keys, name="Camera"),
    )
