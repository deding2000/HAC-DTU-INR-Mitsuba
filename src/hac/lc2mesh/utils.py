"""Shared helpers for prior-embedding notebook training/inference."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError


def set_seed(seed: int) -> None:
    """Set deterministic seeds used in notebooks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_device(target_gpu_id: int = 5) -> torch.device:
    """Select requested CUDA device when available, else CPU."""
    if torch.cuda.is_available() and torch.cuda.device_count() > int(target_gpu_id):
        device = torch.device(f"cuda:{int(target_gpu_id)}")
        torch.cuda.set_device(device)
        print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
        print(f"Visible CUDA devices: {torch.cuda.device_count()}")
        return device

    device = torch.device("cpu")
    print("Device: cpu")
    if torch.cuda.is_available():
        print(
            f"Requested GPU id {target_gpu_id} is unavailable; visible GPU count is {torch.cuda.device_count()}."
        )
    return device


def build_unique_edges(faces: np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for i, j, k in faces:
        edges.add(tuple(sorted((int(i), int(j)))))
        edges.add(tuple(sorted((int(j), int(k)))))
        edges.add(tuple(sorted((int(k), int(i)))))
    return np.asarray(sorted(edges), dtype=np.int64)


def make_bidirectional_edges(edges: np.ndarray) -> np.ndarray:
    return np.concatenate([edges, edges[:, ::-1]], axis=0)


def build_vertex_neighbors(num_vertices: int, edges: np.ndarray) -> list[set[int]]:
    neighbors: list[set[int]] = [set() for _ in range(num_vertices)]
    for i, j in edges:
        i = int(i)
        j = int(j)
        neighbors[i].add(j)
        neighbors[j].add(i)
    return neighbors


def laplacian_regularization(
    vertices: torch.Tensor,
    neighbors: list[set[int]],
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for vid, nbrs in enumerate(neighbors):
        if not nbrs:
            continue
        nbr_idx = torch.tensor(list(nbrs), dtype=torch.long, device=vertices.device)
        nbr_mean = vertices[:, nbr_idx, :].mean(dim=1)
        terms.append(((vertices[:, vid, :] - nbr_mean) ** 2).mean())
    if not terms:
        return torch.tensor(0.0, device=vertices.device)
    return torch.stack(terms).mean()


def edge_length_regularization(
    vertices: torch.Tensor,
    ref_vertices: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    curr = torch.linalg.norm(
        vertices[:, edges[:, 0], :] - vertices[:, edges[:, 1], :], dim=-1
    )
    ref = torch.linalg.norm(
        ref_vertices[:, edges[:, 0], :] - ref_vertices[:, edges[:, 1], :], dim=-1
    )
    rel_err = (curr - ref) / ref.clamp(min=1e-6)
    return (rel_err**2).mean()


def make_mesh_tensors(
    vertices_np: np.ndarray,
    faces_np: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    vertices_t = torch.tensor(vertices_np, dtype=torch.float32, device=device)
    faces_t = torch.tensor(faces_np, dtype=torch.long, device=device)
    edges_np = build_unique_edges(faces_np)
    edge_index_np = make_bidirectional_edges(edges_np)
    return {
        "vertices_t": vertices_t,
        "faces_t": faces_t,
        "edges_np": edges_np,
        "edge_index_np": edge_index_np,
        "edges_t": torch.tensor(edges_np, dtype=torch.long, device=device),
        "edge_index_t": torch.tensor(edge_index_np, dtype=torch.long, device=device),
        "vertex_neighbors": build_vertex_neighbors(len(vertices_np), edges_np),
        "vertices_batch": vertices_t.unsqueeze(0),
    }


def differentiable_voxel_measure(
    verts_pred: torch.Tensor,
    verts_ref: torch.Tensor,
    R: float,
    z_half: float = 1.0,
    grid_size: int = 18,
    tau: float = 20.0,
) -> torch.Tensor:
    """Differentiable surrogate of voxel occupancy overlap in [0, 1]."""
    grid_size = int(max(8, min(48, grid_size)))
    x = torch.linspace(-R, R, grid_size, device=verts_pred.device)
    y = torch.linspace(-R, R, grid_size, device=verts_pred.device)
    z = torch.linspace(-z_half, z_half, grid_size, device=verts_pred.device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

    def soft_occ(verts: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(pts.unsqueeze(0), verts)
        min_d = d.min(dim=-1).values
        return torch.sigmoid(-tau * (min_d - (2.0 * R / grid_size)))

    occ_pred = soft_occ(verts_pred)
    occ_ref = soft_occ(verts_ref)
    inter = torch.minimum(occ_pred, occ_ref).sum(dim=1)
    union = torch.maximum(occ_pred, occ_ref).sum(dim=1) + 1e-8
    return (inter / union).mean()


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pairwise = torch.cdist(pred, target)
    pred_to_target = pairwise.min(dim=2).values
    target_to_pred = pairwise.min(dim=1).values
    return pred_to_target.mean() + target_to_pred.mean()


def sample_convex_hull_surface(
    vertices: np.ndarray,
    n_points: int = 4096,
    seed: int | None = None,
) -> np.ndarray:
    """Area-uniform points on the convex-hull *surface* of a vertex set.

    Builds ``scipy.spatial.ConvexHull`` of the input points and samples
    uniformly on the hull triangles (area-weighted facet choice + uniform
    barycentric sampling). Surface sampling is used instead of the hull
    vertices because large hull facets would otherwise be represented by only
    a few corner points, biasing a Chamfer loss.

    Accepts ``[V, 3]`` or batched ``[B, V, 3]``; returns ``[n_points, 3]`` or
    ``[B, n_points, 3]`` (float32 numpy).
    """
    verts = np.asarray(vertices, dtype=np.float32)
    if verts.ndim == 3:
        return np.stack(
            [sample_convex_hull_surface(v, n_points=n_points, seed=seed) for v in verts]
        )
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Expected vertices [V, 3] or [B, V, 3], got {verts.shape}")

    try:
        hull = ConvexHull(verts)
    except QhullError:
        # Degenerate input (coplanar/duplicate points): joggle to recover.
        hull = ConvexHull(verts, qhull_options="QJ")

    tri = hull.points[hull.simplices]  # [F, 3, 3]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    rng = np.random.default_rng(seed)
    face_idx = rng.choice(len(tri), size=int(n_points), p=areas / areas.sum())
    u = rng.random((int(n_points), 1))
    v = rng.random((int(n_points), 1))
    flip = (u + v) > 1.0  # reflect into the triangle for uniform barycentrics
    u = np.where(flip, 1.0 - u, u)
    v = np.where(flip, 1.0 - v, v)
    return tri[face_idx, 0] + u * e1[face_idx] + v * e2[face_idx]


def vertex_normals_from_faces(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute area-weighted differentiable vertex normals for batched triangle meshes."""
    if vertices.ndim != 3:
        raise ValueError(
            f"Expected vertices shape [B, N, 3], got {tuple(vertices.shape)}"
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected faces shape [F, 3], got {tuple(faces.shape)}")

    faces = faces.long()
    v0 = vertices[:, faces[:, 0], :]
    v1 = vertices[:, faces[:, 1], :]
    v2 = vertices[:, faces[:, 2], :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

    vertex_normals = torch.zeros_like(vertices)
    for corner in range(3):
        vertex_normals.index_add_(1, faces[:, corner], face_normals)

    return torch.nn.functional.normalize(vertex_normals, dim=-1, eps=eps)


def soft_correspondence_normal_loss(
    pred_vertices: torch.Tensor,
    pred_normals: torch.Tensor,
    target_vertices: torch.Tensor,
    target_normals: torch.Tensor,
    soft_nn_temperature: float = 30.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine normal consistency with soft nearest-neighbor correspondences."""
    if soft_nn_temperature <= 0:
        raise ValueError("soft_nn_temperature must be > 0")

    if pred_vertices.shape != pred_normals.shape:
        raise ValueError(
            "pred_vertices and pred_normals must have identical [B, N, 3] shape"
        )
    if target_vertices.shape != target_normals.shape:
        raise ValueError(
            "target_vertices and target_normals must have identical [B, M, 3] shape"
        )

    dists = torch.cdist(pred_vertices, target_vertices)
    weights = torch.softmax(-float(soft_nn_temperature) * dists, dim=-1)

    matched_target_normals = weights @ target_normals
    matched_target_normals = torch.nn.functional.normalize(
        matched_target_normals,
        dim=-1,
        eps=eps,
    )

    cosine = (pred_normals * matched_target_normals).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cosine).mean()


def resample_target_to_prior_directions(
    prior_vertices_np: np.ndarray,
    target_vertices_np: np.ndarray,
) -> np.ndarray:
    # Convex-support resampling: for each prior direction u, use max<v,u> over target vertices.
    prior_dirs = prior_vertices_np / np.clip(
        np.linalg.norm(prior_vertices_np, axis=1, keepdims=True),
        1e-9,
        None,
    )
    projections = target_vertices_np @ prior_dirs.T
    matched_r = np.clip(projections.max(axis=0), 1e-6, None)
    return (prior_dirs * matched_r[:, None]).astype(np.float32)


def build_fixed_subset(
    vertices: torch.Tensor,
    max_points: int = 2200,
    seed: int = 7,
) -> torch.Tensor:
    n = vertices.shape[1]
    if n <= max_points:
        return vertices
    g = torch.Generator(device=vertices.device)
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g, device=vertices.device)[:max_points]
    return vertices[:, idx, :]


def deformation_tv_regularization(
    pred_vertices: torch.Tensor,
    base_vertices: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Edge-wise smoothness on deformation field to suppress local spikes."""
    delta = pred_vertices - base_vertices
    di = delta[:, edges[:, 0], :]
    dj = delta[:, edges[:, 1], :]
    edge_ref = torch.linalg.norm(
        base_vertices[:, edges[:, 0], :] - base_vertices[:, edges[:, 1], :],
        dim=-1,
    ).clamp(min=1e-6)
    tv = torch.linalg.norm(di - dj, dim=-1) / edge_ref
    return (tv**2).mean()
