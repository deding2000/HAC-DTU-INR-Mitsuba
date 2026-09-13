import numpy as np
import trimesh


def calculate_voxel_measure(
    true_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh, pitch: float = 0.02
) -> float:
    """
    Fast voxel-based similarity measure.

    Parameters
    ----------
    true_mesh : trimesh.Trimesh
        Ground truth mesh.

    pred_mesh : trimesh.Trimesh
        Predicted mesh.

    pitch : float
        Voxel size. Smaller values give higher accuracy but require
        more memory and computation.

    Returns
    -------
    float
        Similarity score in [0, 1].
    """
    # Voxelize both meshes
    vox_true = true_mesh.voxelized(pitch)
    vox_pred = pred_mesh.voxelized(pitch)

    # Get occupied voxel indices
    true_voxels = set(map(tuple, vox_true.sparse_indices))
    pred_voxels = set(map(tuple, vox_pred.sparse_indices))

    if len(true_voxels) == 0 and len(pred_voxels) == 0:
        return 0.0

    # Symmetric difference
    mismatch_count = len(true_voxels.symmetric_difference(pred_voxels))

    # Denominator from your original formula
    total_volume_sum = len(true_voxels) + len(pred_voxels)

    score = 1.0 - mismatch_count / total_volume_sum

    return np.clip(score, 0.0, 1.0)


def calculate_voxel_measure_solid(
    true_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh, pitch: float = 0.02
) -> float:
    """
    Solid-volume analogue of `calculate_voxel_measure`: same symmetric-
    difference / (|A|+|B|) Dice-style formula, but on
    `.voxelized(pitch).fill()` (interior-filled) voxel sets instead of the
    hollow surface shell.
    """
    true_voxels = set(map(tuple, true_mesh.voxelized(pitch).fill().sparse_indices))
    pred_voxels = set(map(tuple, pred_mesh.voxelized(pitch).fill().sparse_indices))
    if len(true_voxels) == 0 and len(pred_voxels) == 0:
        return 0.0
    mismatch_count = len(true_voxels.symmetric_difference(pred_voxels))
    total = len(true_voxels) + len(pred_voxels)
    return float(np.clip(1.0 - mismatch_count / total, 0.0, 1.0))


def official_voxel_dissimilarity(
    true_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh, pitch: float = 0.02
) -> tuple[float, float]:
    """Official challenge voxel-based dissimilarity, as supplied by the
    organisers (workspaces/christian/HAC_eval.py,
    `relative_volume_difference_voxelized`) -- ported here to take
    `trimesh.Trimesh` objects directly instead of mesh file paths, to match
    this module's other functions; the voxelization algorithm itself is
    unchanged from the organisers' code.

    IMPORTANT -- opposite convention from `calculate_voxel_measure[_solid]`
    above: this is LOWER IS BETTER. Both returned values are 0 for a
    perfect match and increase as the meshes diverge (no fixed upper
    bound other than 1.0 for total non-overlap).

    Differs from `calculate_voxel_measure_solid` in two ways that matter:
      - Both meshes are voxelized on a SHARED grid (a `pitch`-sized ghost
        cube is appended, unfaced, at each mesh's shared-bounding-box min
        and max corner before voxelizing) instead of each mesh
        independently choosing its own `.voxelized(pitch)` origin --
        without this, voxel index tuples from two different meshes are
        only comparable by coincidence, not by construction. (Ghost
        geometry has no faces, so it cannot fill voxels itself; it only
        forces both grids to share the same extent.)
      - Solid (`.fill()`ed) volumes only -- the organisers' code has no
        "surface shell" variant.

    Returns
    -------
    (iou_dissimilarity, volume_fraction_dissimilarity) : tuple[float, float]
        iou_dissimilarity = 1 - IoU(filled_true, filled_pred).
        volume_fraction_dissimilarity = (|true\\pred| + |pred\\true|) /
        (|true| + |pred|) -- the organisers' "measure2", equal to
        `1 - calculate_voxel_measure_solid(...)` up to the shared-grid fix.
    """
    min_bound = np.minimum(true_mesh.bounds[0], pred_mesh.bounds[0])
    max_bound = np.maximum(true_mesh.bounds[1], pred_mesh.bounds[1])
    ghost_extents = np.array([pitch, pitch, pitch])
    ghost_min = trimesh.creation.box(extents=ghost_extents)
    ghost_min.apply_translation(min_bound)
    ghost_max = trimesh.creation.box(extents=ghost_extents)
    ghost_max.apply_translation(max_bound)

    true_padded = trimesh.util.concatenate([true_mesh, ghost_min, ghost_max])
    pred_padded = trimesh.util.concatenate([pred_mesh, ghost_min, ghost_max])
    filled_true = true_padded.voxelized(pitch, method="subdivide").fill().matrix.astype(bool)
    filled_pred = pred_padded.voxelized(pitch, method="subdivide").fill().matrix.astype(bool)

    vol_only_true = int(np.sum(filled_true & ~filled_pred))
    vol_only_pred = int(np.sum(filled_pred & ~filled_true))
    vol_intersection = int(np.sum(filled_true & filled_pred))
    vol_union = int(np.sum(filled_true | filled_pred))
    vol_true = int(np.sum(filled_true))
    vol_pred = int(np.sum(filled_pred))

    iou_dissimilarity = 1.0 - vol_intersection / vol_union if vol_union else 0.0
    denom = vol_true + vol_pred
    volume_fraction_dissimilarity = (
        (vol_only_true + vol_only_pred) / denom if denom else 0.0
    )
    return float(iou_dissimilarity), float(volume_fraction_dissimilarity)
