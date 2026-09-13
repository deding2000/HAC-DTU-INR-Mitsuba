"""Mesh/data preparation helpers for lc2mesh notebooks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import trimesh
from stl import mesh as stl_mesh

from hac.constants import CAMERA_ANGLE_ALPHA, MODEL_BASE_RADII
from hac.data import get_model_stl_path
from hac.mesh import normalize_mesh_to_challenge_cylinder
from hac.lc2mesh.utils import make_mesh_tensors


def create_icosphere(subdivisions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Geodesic icosphere with ``4^subdivisions * 20`` faces, all projected to unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    raw = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float32,
    )
    verts = list(raw / np.linalg.norm(raw[0]))

    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    def midpoint(p1: int, p2: int, cache: dict[tuple[int, int], int]) -> int:
        key = (min(p1, p2), max(p1, p2))
        if key not in cache:
            m = (verts[p1] + verts[p2]) / 2
            m = m / np.linalg.norm(m)
            cache[key] = len(verts)
            verts.append(m)
        return cache[key]

    for _ in range(subdivisions):
        new_faces: list[list[int]] = []
        cache: dict[tuple[int, int], int] = {}
        for a, b, c in faces:
            ab = midpoint(a, b, cache)
            bc = midpoint(b, c, cache)
            ca = midpoint(c, a, cache)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces

        V = np.array(verts)
        F = np.array(faces)

    return V, F


def _compute_face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    e2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    cross = np.cross(e1, e2)
    mag = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(mag[:, None], 1e-12)
    areas = 0.5 * mag
    return normals, areas


def _apply_shape_deformation(
    verts: np.ndarray, shape_type: str, xy_scale: float = 1.25
) -> np.ndarray:
    """Deform a unit sphere into various asteroid-like priors with wider XY bounds."""
    v = verts.copy()
    
    if shape_type == "ellipsoid":
        pass  # Remains a base sphere, scaled later

    elif shape_type == "cube":
        # Project unit sphere vertices to a box/cube surface
        # Mapping each vertex vector to the surface of an axis-aligned unit cube [-1, 1]^3
        max_abs = np.max(np.abs(v), axis=1, keepdims=True)
        v = v / np.maximum(max_abs, 1e-12)
        v[:, 2] *= 2
        
    elif shape_type == "peanut":
        # Squeeze the middle to create a contact-binary shape (e.g., Arrokoth)
        squeeze = 0.6 + 1.6 * (v[:, 2] ** 2)
        v[:, 0] *= squeeze
        v[:, 1] *= squeeze
        v[:, 2] *= 1.8  # Elongate along the Z axis
        
    elif shape_type == "diamond":
        # Spinning-top shape with equatorial bulge (e.g., Bennu, Ryugu)
        squeeze = 1.0 - 0.7 * np.abs(v[:, 2])
        v[:, 0] *= squeeze
        v[:, 1] *= squeeze
        v[:, 2] *= 1.2
        
    elif shape_type == "irregular":
        # Bumpy, cratered baseline using low-frequency volumetric noise
        noise = (
            0.15 * np.sin(4 * v[:, 0]) * np.cos(4 * v[:, 1]) +
            0.10 * np.cos(5 * v[:, 2]) +
            0.05 * np.sin(8 * v[:, 0] * v[:, 1])
        )
        v *= (1.0 + noise[:, None])
        
    else:
        raise ValueError(f"Unsupported shape_type: {shape_type}")

    if shape_type not in {"ellipsoid", "cube"}:
        # Boost X and Y dimensions to widen the body
        v[:, 0] *= xy_scale
        v[:, 1] *= xy_scale

        # Scale uniformly based on the largest extent to preserve the new proportion
        # while guaranteeing z remains in [-1, 1] and r <= 1.0.
        r_max = np.max(np.sqrt(v[:, 0]**2 + v[:, 1]**2))
        z_max = np.max(np.abs(v[:, 2]))
        scale = max(r_max, z_max)
        if scale > 0:
            v /= scale
    
    return v


def create_ellipsoidal_prior_mesh(
    *,
    asteroid_id: int,
    subdivisions: int,
    device: torch.device,
    shape_type: str = "ellipsoid",
) -> dict[str, Any]:
    """Create cylinder-constrained prior and tensorized mesh views with selectable shapes."""
    R_cylinder = float(MODEL_BASE_RADII.get(asteroid_id, 1.0))
    sphere_verts, sphere_faces = create_icosphere(subdivisions=subdivisions)

    # Deform the base icosphere to the desired asteroid archetype
    sphere_verts = _apply_shape_deformation(sphere_verts, shape_type)

    scale_xyz = np.array([R_cylinder, R_cylinder, 1.0], dtype=np.float32)
    sphere_verts = sphere_verts * scale_xyz[None, :]

    normals, sigma = _compute_face_geometry(sphere_verts, sphere_faces)

    prior_vertices_np = sphere_verts.astype(np.float32)
    prior_faces_np = sphere_faces.astype(np.int64)
    prior_mesh = make_mesh_tensors(prior_vertices_np, prior_faces_np, device)

    radial_max = np.sqrt(prior_vertices_np[:, 0] ** 2 + prior_vertices_np[:, 1] ** 2).max()
    z_min = float(prior_vertices_np[:, 2].min())
    z_max = float(prior_vertices_np[:, 2].max())

    return {
        "asteroid_id": asteroid_id,
        "R_CYLINDER": R_cylinder,
        "shape_type": shape_type,
        "n_facets": len(normals),
        "normals": normals,
        "sigma": sigma,
        "radial_max": float(radial_max),
        "z_min": z_min,
        "z_max": z_max,
        "prior_vertices_np": prior_vertices_np,
        "prior_faces_np": prior_faces_np,
        "prior_vertices_t": prior_mesh["vertices_t"],
        "prior_faces_t": prior_mesh["faces_t"],
        "prior_edges_np": prior_mesh["edges_np"],
        "prior_edge_index_np": prior_mesh["edge_index_np"],
        "prior_edges_t": prior_mesh["edges_t"],
        "prior_edge_index_t": prior_mesh["edge_index_t"],
        "prior_vertex_neighbors": prior_mesh["vertex_neighbors"],
        "prior_vertices_batch": prior_mesh["vertices_batch"],
    }


def load_training_template_mesh(
    *,
    asteroid_id: int,
    R_cylinder: float,
    device: torch.device,
    mesh_keep_ratio: float = 0.15,
    max_forward_matrix_mb: int = 800,
) -> dict[str, Any]:
    """Load and normalize asteroid template mesh and return tensorized train mesh."""
    model_stl_path = get_model_stl_path(asteroid_id)
    raw_mesh = stl_mesh.Mesh.from_file(str(model_stl_path))
    raw_vertices = raw_mesh.vectors.reshape(-1, 3).astype(np.float32)
    raw_faces = np.arange(len(raw_vertices), dtype=np.int64).reshape(-1, 3)

    template_mesh_full_tm = trimesh.Trimesh(vertices=raw_vertices, faces=raw_faces, process=True)
    template_mesh_full_tm = normalize_mesh_to_challenge_cylinder(
        template_mesh_full_tm,
        target_R=R_cylinder,
        center_xy="bounds",
        copy=True,
        verbose=False,
    )

    forward_rows_est = len(CAMERA_ANGLE_ALPHA) * 360
    max_faces_by_memory = int((max_forward_matrix_mb * 1024 * 1024) / (4 * forward_rows_est))
    requested_faces = int(len(template_mesh_full_tm.faces) * mesh_keep_ratio)
    target_faces = int(np.clip(requested_faces, 6000, max_faces_by_memory))

    if target_faces < len(template_mesh_full_tm.faces):
        try:
            target_reduction = 1.0 - (target_faces / float(len(template_mesh_full_tm.faces)))
            target_reduction = float(np.clip(target_reduction, 0.0, 0.999))
            template_mesh_train_tm = template_mesh_full_tm.simplify_quadric_decimation(
                target_reduction
            )
            reduction_note = (
                f"mild quadric downscaling to {len(template_mesh_train_tm.faces)} faces"
            )
        except Exception as exc:
            step = max(1, len(template_mesh_full_tm.faces) // target_faces)
            keep_idx = np.arange(0, len(template_mesh_full_tm.faces), step, dtype=np.int64)[
                :target_faces
            ]
            template_mesh_train_tm = template_mesh_full_tm.submesh(
                [keep_idx],
                append=True,
                repair=True,
            )
            template_mesh_train_tm.remove_unreferenced_vertices()
            reduction_note = (
                f"mild face downscaling to {len(template_mesh_train_tm.faces)} faces "
                f"(quadric decimation unavailable: {exc})"
            )
    else:
        template_mesh_train_tm = template_mesh_full_tm.copy()
        reduction_note = "no downscaling applied"

    train_template_vertices_np = np.asarray(template_mesh_train_tm.vertices, dtype=np.float32)
    train_template_faces_np = np.asarray(template_mesh_train_tm.faces, dtype=np.int64)
    train_template_mesh = make_mesh_tensors(
        train_template_vertices_np,
        train_template_faces_np,
        device,
    )

    return {
        "model_stl_path": model_stl_path,
        "template_mesh_full_tm": template_mesh_full_tm,
        "template_mesh_train_tm": template_mesh_train_tm,
        "reduction_note": reduction_note,
        "train_template_vertices_np": train_template_vertices_np,
        "train_template_faces_np": train_template_faces_np,
        "train_template_vertices": train_template_mesh["vertices_t"],
        "train_template_faces_t": train_template_mesh["faces_t"],
        "train_edges_np": train_template_mesh["edges_np"],
        "train_edge_index_np": train_template_mesh["edge_index_np"],
        "train_edges_t": train_template_mesh["edges_t"],
        "train_edge_index_t": train_template_mesh["edge_index_t"],
        "train_vertex_neighbors": train_template_mesh["vertex_neighbors"],
        "train_template_vertices_batch": train_template_mesh["vertices_batch"],
    }