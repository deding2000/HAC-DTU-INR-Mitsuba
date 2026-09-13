"""INR + Mitsuba training loop using the updated `hac.lc2mesh` training
pipeline pulled from the `inr` branch (commit faac378, "Training
improvements", on top of c9aaa31 which is what inr_convex_prior_mitsuba_stage2.py's
model.py already matched). Same warm-start/target/forward-model setup as
that script, but the two-stage optimization itself is now delegated to
`hac.lc2mesh.training.train_stage`, which adds:

  - Gradient clipping (config.clip_grad_norm) and a ReduceLROnPlateau
    scheduler -- both stability fixes independently arrived at by hand in
    this session (see inr_convex_prior_mitsuba_stage2.py's
    GRAD_CLIP_NORM/LR_RESTART_PERIOD), done properly by the model's author.
  - A held-out validation camera split (`split_camera_indices`):
    best-checkpoint selection uses validation-camera MSE, not training
    loss, which is a more principled defense against picking a checkpoint
    that just got lucky on noisy per-step Mitsuba renders than this
    session's plain "best training loss" convention.
  - Displacement-based Stage-2 regularization (residual from the restored
    Stage-1 solution, plus a *displacement* Laplacian rather than an
    absolute-position one), instead of this session's own from-scratch
    Chamfer/Laplacian regularizers.

Stage 1 here ALSO renders through Mitsuba every step (lightcurve MSE + hull
Chamfer), unlike inr_convex_prior_mitsuba_stage2.py's Phase 1 (a cheap,
render-free direct log-radius regression against the convex-inversion
target) -- so Stage 1 step counts must be much smaller here for a
comparable compute budget.
"""

import os

os.environ.setdefault("DSDIR", "/zhome/95/1/155570/HAC-DTU/HAC-collab")

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

torch.set_default_dtype(torch.float32)

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_mono")

from hac.constants import OBSERVERS
from hac.data import get_model_stl_path, load_data
from hac.eval import calculate_voxel_measure, calculate_voxel_measure_solid, official_voxel_dissimilarity
from hac.lc2mesh.data import create_ellipsoidal_prior_mesh
from hac.lc2mesh.metrics import compute_per_camera_metrics
from hac.lc2mesh.model import LightcurveToMesh
from hac.lc2mesh.training import TrainingConfig, split_camera_indices, train_stage
from hac.lc2mesh.utils import sample_convex_hull_surface, set_seed, vertex_normals_from_faces
from hac.mesh import normalize_mesh_to_challenge_cylinder
from hac.mitsuba.forward import ForwardModel
from hac.mitsuba.mesh import MitsubaMesh
from hac.mitsuba.scene import RenderingSettings
from hac.mitsuba.visualize import bitmap_frames, write_movie

ASTEROID_ID = int(os.environ.get("ASTEROID_ID", "3"))
TARGET_ORIGIN = "real"
TARGET_CURVE_TYPE = "intensity"
CONVEX_PRIOR_ORIGIN = "simulated"
CONVEX_PRIOR_CURVE_TYPE = "intensity"
CONVEX_PRIOR_STL_PATH = (
    "/zhome/95/1/155570/HAC-DTU/HAC-collab/reconstructions/convex/intensity/"
    f"asteroid{ASTEROID_ID}_{CONVEX_PRIOR_ORIGIN}_{CONVEX_PRIOR_CURVE_TYPE}.stl"
)

PRIOR_SUBDIVISIONS = 3  # -> 642 vertices / 1280 faces, matches v1
EXPERIMENTS = tuple(OBSERVERS)
N_PHASES = 180
RENDER_WIDTH = 128
RENDER_HEIGHT = 128
RENDER_SPP = 8
HULL_N_SAMPLES = 4096

# Same fitted material as inr_convex_prior_mitsuba_stage2.py -- see that
# file's REAL_INTENSITY_FITTED_BSDF comment for the full derivation.
REAL_INTENSITY_FITTED_BSDF = {
    1: {"alpha": 0.43, "eta": 1.43},
    2: {"alpha": 1.14, "eta": 2.52},
    3: {"alpha": 1.66, "eta": 2.06},
}
if TARGET_ORIGIN == "real" and ASTEROID_ID in REAL_INTENSITY_FITTED_BSDF:
    _fitted = REAL_INTENSITY_FITTED_BSDF[ASTEROID_ID]
    BSDF = {
        "type": "roughplastic",
        "nonlinear": False,
        "alpha": _fitted["alpha"],
        "int_ior": _fitted["eta"],
        "diffuse_reflectance": {"type": "rawconstant", "value": 1.0},
    }
else:
    BSDF = None

STAGE1_STEPS = int(os.environ.get("STAGE1_STEPS", "40"))
STAGE2_STEPS = int(os.environ.get("STAGE2_STEPS", "150"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "10"))
VALIDATION_FRACTION = 0.25
SEED = 42
VOXEL_SIMILARITY_PITCH = 0.02

# Movie rendering (visualization only, post-training) -- same convention
# as inr_convex_prior_mitsuba_stage2.py / mitsuba_shape_optimization_from_convex.py.
MOVIE_EXPERIMENT = "0_hor1"
MOVIE_PHASES = 360
MOVIE_WIDTH = 512
MOVIE_HEIGHT = 512
MOVIE_SPP = 16
MOVIE_FPS = 30
MIN_PROJECTED_PHASE_CENTER_SEPARATION = 5.0

OUTPUT_DIR = Path(__file__).parent / f"inr_convex_prior_mitsuba_stage2_v2_output_asteroid{ASTEROID_ID}_{TARGET_ORIGIN}_{TARGET_CURVE_TYPE}"
RECON_DIR = OUTPUT_DIR / "recon"
MOVIE_DIR = OUTPUT_DIR / "movies"
RECON_DIR.mkdir(parents=True, exist_ok=True)
MOVIE_DIR.mkdir(parents=True, exist_ok=True)
PRIOR_MESH_OBJ_PATH = OUTPUT_DIR / "prior_mesh_full.obj"
NORMALIZED_REFERENCE_MESH_PATH = OUTPUT_DIR / "normalized_reference.obj"
BEST_MESH_PATH = RECON_DIR / f"asteroid{ASTEROID_ID}_best.stl"
STAGE1_MESH_PATH = RECON_DIR / f"asteroid{ASTEROID_ID}_stage1_only.stl"
GT_MESH_PATH = RECON_DIR / f"asteroid{ASTEROID_ID}_ground_truth_normalized.stl"
LOSS_PLOT_PATH = OUTPUT_DIR / "training_loss.png"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
TRUE_MOVIE_PATH = MOVIE_DIR / "true_object.mp4"
STAGE1_MOVIE_PATH = MOVIE_DIR / "stage1_only.mp4"
RECON_MOVIE_PATH = MOVIE_DIR / "best_reconstruction.mp4"
COMPARISON_MOVIE_PATH = MOVIE_DIR / "true_vs_stage1_vs_reconstruction.mp4"
print(f"OUTPUT_DIR = {OUTPUT_DIR}")


def render_movie_frames(mesh_path, *, bsdf):
    forward = ForwardModel(
        MitsubaMesh.from_stl_path(mesh_path)
        if str(mesh_path).endswith(".stl")
        else MitsubaMesh(obj_path=mesh_path),
        n_phases=MOVIE_PHASES,
        observers={MOVIE_EXPERIMENT: OBSERVERS[MOVIE_EXPERIMENT]},
        rendering_settings=RenderingSettings(width=MOVIE_WIDTH, height=MOVIE_HEIGHT, spp=MOVIE_SPP),
        normalize_mesh=False,
        minimum_projected_phase_center_separation=MIN_PROJECTED_PHASE_CENTER_SEPARATION,
        bsdf=bsdf,
    )
    return bitmap_frames(forward.render(seed=1)[0])


def build_vertex_order_permutation(prior_vertices_np, mitsuba_vertices_np):
    """Index array p such that mitsuba_vertices_np == prior_vertices_np[p]
    -- Mitsuba's OBJ loader does not preserve the vertex order of the file
    it reads (same point set, different permutation)."""
    tree = cKDTree(prior_vertices_np)
    dists, indices = tree.query(mitsuba_vertices_np, k=1)
    if dists.max() > 1e-4:
        raise RuntimeError(
            "Mitsuba-loaded vertices don't match the prior mesh's point set "
            f"(max nearest-neighbor distance {dists.max():.6g})."
        )
    return indices


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch device: {device}")

    prior = create_ellipsoidal_prior_mesh(
        asteroid_id=ASTEROID_ID, subdivisions=PRIOR_SUBDIVISIONS, device=device, shape_type="ellipsoid",
    )
    prior_vertices_np = prior["prior_vertices_np"]
    prior_faces_np = prior["prior_faces_np"]
    R_cylinder = float(prior["R_CYLINDER"])
    print(f"Prior mesh: {len(prior_vertices_np)} vertices, {len(prior_faces_np)} faces (R_cylinder={R_cylinder:.4f})")

    convex_inversion_mesh = normalize_mesh_to_challenge_cylinder(
        trimesh.load_mesh(CONVEX_PRIOR_STL_PATH, force="mesh"), target_R=R_cylinder,
    )
    hull_samples = sample_convex_hull_surface(
        np.asarray(convex_inversion_mesh.vertices), n_points=HULL_N_SAMPLES, seed=SEED
    )
    hull_points = torch.as_tensor(hull_samples, dtype=torch.float32, device=device).unsqueeze(0)
    print(f"Hull Chamfer target (convex inversion): {hull_points.shape[1]} samples")

    base_vertices = torch.as_tensor(prior_vertices_np, dtype=torch.float32, device=device).unsqueeze(0)
    faces_t = torch.as_tensor(prior_faces_np, dtype=torch.long, device=device)
    base_normals = vertex_normals_from_faces(base_vertices, faces_t)

    # canonicalize_coarse=True is required: train_stage's stage-2 path
    # asserts config.canonicalize == net.canonicalize_coarse.
    net = LightcurveToMesh(
        faces=faces_t, inr_hidden_dim=128, inr_num_layers=6, R=R_cylinder, canonicalize_coarse=True,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"INR+GCN parameters: {n_params:,}")

    # ================= Mitsuba scene (built once, from the RAW prior --
    # there's no cheap pretrain step here, Stage 1 itself is the first
    # lightcurve-fitting phase) =================
    prior_mesh = trimesh.Trimesh(vertices=prior_vertices_np, faces=prior_faces_np, process=False)
    prior_mesh.export(PRIOR_MESH_OBJ_PATH)

    observers = {experiment: OBSERVERS[experiment] for experiment in EXPERIMENTS}
    forward = ForwardModel(
        MitsubaMesh(obj_path=PRIOR_MESH_OBJ_PATH),
        n_phases=N_PHASES,
        observers=observers,
        rendering_settings=RenderingSettings(width=RENDER_WIDTH, height=RENDER_HEIGHT, spp=RENDER_SPP),
        normalize_mesh=False,
        bsdf=BSDF,
    )
    mitsuba_vertices_np = forward.params[forward.position_key].numpy().reshape(-1, 3).astype(np.float64)
    permutation = build_vertex_order_permutation(prior_vertices_np, mitsuba_vertices_np)
    permutation_t = torch.as_tensor(permutation, dtype=torch.long, device=device)
    print(f"Mitsuba scene built: {len(mitsuba_vertices_np)} vertices, vertex-order permutation verified")

    model_stl_path = get_model_stl_path(model_num=ASTEROID_ID) if ASTEROID_ID <= 3 else None
    reference_mesh = None
    if model_stl_path is not None:
        reference_mesh = normalize_mesh_to_challenge_cylinder(
            trimesh.load_mesh(model_stl_path, force="mesh"), target_R=R_cylinder,
        )
        reference_mesh.export(NORMALIZED_REFERENCE_MESH_PATH)

    @dr.wrap(source="torch", target="drjit")
    def render_curve(positions_flat):
        forward.params[forward.position_key] = mi.Float(dr.ravel(positions_flat))
        forward.params.update()
        return forward.render_lightcurves(seed=0)

    def predict_lightcurves(vertices):
        """train_stage's callable contract: [V, 3] prior-order vertices in,
        normalized [C, P] lightcurves out -- the Mitsuba vertex-order
        permutation is this script's responsibility, not train_stage's."""
        return render_curve(vertices[permutation_t].reshape(-1))

    target_data = load_data(model_num=ASTEROID_ID, origin=TARGET_ORIGIN, curve_type=TARGET_CURVE_TYPE).data
    native_phase_count = len(target_data[EXPERIMENTS[0]])
    phase_indices = np.arange(N_PHASES) * native_phase_count // N_PHASES
    target_np = np.stack(
        [np.asarray(target_data[experiment], dtype=np.float32)[phase_indices] for experiment in EXPERIMENTS]
    )
    target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
    target = target / target.mean(dim=1, keepdim=True).clamp_min(1e-12)

    train_indices, validation_indices = split_camera_indices(
        len(EXPERIMENTS), seed=SEED, validation_fraction=VALIDATION_FRACTION, device=device
    )
    print(
        f"Camera split: {len(train_indices)} train / {len(validation_indices)} validation "
        f"(held out: {[EXPERIMENTS[i] for i in validation_indices.tolist()]})"
    )

    def make_progress(stage):
        def progress(row):
            print(
                f"  [stage{stage}] step {row['step']:4d}: train_mse={row['train_mse']:.6f} "
                f"validation_mse={row['validation_mse']:.6f} score={row['score']:.6f}"
            )
        return progress

    stage1_config = TrainingConfig(
        steps=STAGE1_STEPS, lr_inr=1e-4, hull_weight=0.3, laplacian_weight=0.1,
        displacement_weight=0.0, eval_every=EVAL_EVERY, scheduler_patience=3,
        clip_grad_norm=1.0, canonicalize=True, restore_best=True, selection="validation",
    )
    stage1_start = time.perf_counter()
    stage1_result = train_stage(
        net, base_vertices, base_normals, predict_lightcurves, target, hull_points,
        train_indices, validation_indices, stage1_config, stage=1, progress=make_progress(1),
    )
    stage1_elapsed = time.perf_counter() - stage1_start
    print(
        f"Stage 1 done in {stage1_elapsed:.1f}s -- best validation score="
        f"{stage1_result.best_score:.6f} @ step {stage1_result.best_step}/{STAGE1_STEPS}"
    )

    # Stage 2's `reference` (for its displacement/displacement-Laplacian
    # regularization) is captured from net.inr's state at the START of
    # train_stage's stage=2 call -- i.e. the Stage-1 result restored above.
    stage2_config = TrainingConfig(
        steps=STAGE2_STEPS, lr_inr=1e-6, lr_graph=1e-4, hull_weight=0.1, laplacian_weight=0.1,
        displacement_weight=0.05, eval_every=EVAL_EVERY, scheduler_patience=5,
        clip_grad_norm=1.0, canonicalize=True, restore_best=True, selection="validation",
    )
    stage2_start = time.perf_counter()
    stage2_result = train_stage(
        net, base_vertices, base_normals, predict_lightcurves, target, hull_points,
        train_indices, validation_indices, stage2_config, stage=2, progress=make_progress(2),
    )
    stage2_elapsed = time.perf_counter() - stage2_start
    print(
        f"Stage 2 done in {stage2_elapsed:.1f}s -- best validation score="
        f"{stage2_result.best_score:.6f} @ step {stage2_result.best_step}/{STAGE2_STEPS}"
    )

    # net now holds Stage 2's best (validation-selected) checkpoint (restore_best=True).
    with torch.no_grad():
        inr_best, refined_best = net(base_vertices, base_normals)
        curve_refined = predict_lightcurves(refined_best[0])
        curve_inr_only = predict_lightcurves(inr_best[0])
    per_camera_df = compute_per_camera_metrics(target, curve_refined, EXPERIMENTS)
    per_camera_inr_df = compute_per_camera_metrics(target, curve_inr_only, EXPERIMENTS)
    print(
        f"Refined mesh fit (best checkpoint, ALL cameras): mean_r={per_camera_df['Pearson_R'].mean():.4f}, "
        f"min_r={per_camera_df['Pearson_R'].min():.4f}, mean MSE={per_camera_df['MSE'].mean():.6f}"
    )
    print(
        f"INR-only mesh fit (reference, same checkpoint pre-GCN): "
        f"mean_r={per_camera_inr_df['Pearson_R'].mean():.4f}, "
        f"min_r={per_camera_inr_df['Pearson_R'].min():.4f}, mean MSE={per_camera_inr_df['MSE'].mean():.6f}"
    )

    best_vertices_np = refined_best[0].detach().cpu().numpy()
    best_mesh = trimesh.Trimesh(vertices=best_vertices_np, faces=prior_faces_np, process=False)
    best_mesh.export(BEST_MESH_PATH)
    print(f"Saved best mesh: {BEST_MESH_PATH}")

    stage1_only_vertices_np = inr_best[0].detach().cpu().numpy()
    stage1_only_mesh = trimesh.Trimesh(vertices=stage1_only_vertices_np, faces=prior_faces_np, process=False)
    stage1_only_mesh.export(STAGE1_MESH_PATH)
    print(f"Saved Stage-1-only (pre-GCN) mesh: {STAGE1_MESH_PATH}")

    voxel_similarity_surface = None
    voxel_similarity_solid = None
    official_iou_dissimilarity = None
    official_volfrac_dissimilarity = None
    if reference_mesh is not None:
        voxel_similarity_surface = calculate_voxel_measure(reference_mesh, best_mesh, pitch=VOXEL_SIMILARITY_PITCH)
        voxel_similarity_solid = calculate_voxel_measure_solid(reference_mesh, best_mesh, pitch=VOXEL_SIMILARITY_PITCH)
        # PRIMARY metric: the organisers' own shared-grid, filled-voxel
        # dissimilarity. LOWER IS BETTER (0 = perfect match) -- opposite
        # convention from the two scores above, kept for continuity with
        # this session's earlier comparisons.
        official_iou_dissimilarity, official_volfrac_dissimilarity = official_voxel_dissimilarity(
            reference_mesh, best_mesh, pitch=VOXEL_SIMILARITY_PITCH
        )
        print(
            f"Voxel similarity vs. ground truth -- best checkpoint: "
            f"surface-shell={voxel_similarity_surface:.4f} solid-fill={voxel_similarity_solid:.4f} "
            f"| official (lower=better) 1-IoU={official_iou_dissimilarity:.4f} "
            f"vol-frac-diss={official_volfrac_dissimilarity:.4f}"
        )
    else:
        print(f"No public ground-truth STL for asteroid {ASTEROID_ID}; skipping voxel similarity.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    s2_offset = STAGE1_STEPS
    axes[0].plot(stage1_result.history["step"], stage1_result.history["lc_mse"], label="Stage 1 train lc_mse", color="#d95f02")
    axes[0].plot(
        [s2_offset + s for s in stage2_result.history["step"]],
        stage2_result.history["lc_mse"], label="Stage 2 train lc_mse", color="#e7298a",
    )
    axes[0].plot(
        [row["step"] for row in stage1_result.evaluations],
        [row["validation_mse"] for row in stage1_result.evaluations],
        label="Stage 1 validation_mse", color="#1b9e77", linestyle="--", marker="o",
    )
    axes[0].plot(
        [s2_offset + row["step"] for row in stage2_result.evaluations],
        [row["validation_mse"] for row in stage2_result.evaluations],
        label="Stage 2 validation_mse", color="#66a61e", linestyle="--", marker="o",
    )
    axes[0].axvline(s2_offset, color="k", ls="--", lw=1, alpha=0.5)
    axes[0].axvline(s2_offset + stage2_result.best_step, color="r", ls=":", lw=1, alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Curve MSE")
    axes[0].set_title("v2 (hac.lc2mesh.training): train vs. held-out validation cameras")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(stage1_result.history["step"], stage1_result.history["mean_r"], label="Stage 1 mean_r", color="#d95f02")
    axes[1].plot(stage1_result.history["step"], stage1_result.history["min_r"], label="Stage 1 min_r", color="#1b9e77")
    axes[1].plot(
        [s2_offset + s for s in stage2_result.history["step"]],
        stage2_result.history["mean_r"], label="Stage 2 mean_r", color="#e7298a",
    )
    axes[1].plot(
        [s2_offset + s for s in stage2_result.history["step"]],
        stage2_result.history["min_r"], label="Stage 2 min_r", color="#66a61e",
    )
    axes[1].axvline(s2_offset, color="k", ls="--", lw=1, alpha=0.5)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Pearson r (training cameras)")
    axes[1].set_title("Lightcurve correlation")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(LOSS_PLOT_PATH, dpi=150)
    print(f"Saved loss plot: {LOSS_PLOT_PATH}")

    # --- Movie rendering: GT / Stage-1-only / best reconstruction, plus a
    # side-by-side comparison -- visualization only. ---
    if reference_mesh is not None:
        reference_mesh.export(GT_MESH_PATH)
        true_frames = render_movie_frames(GT_MESH_PATH, bsdf=BSDF)
        stage1_frames = render_movie_frames(STAGE1_MESH_PATH, bsdf=BSDF)
        recon_frames = render_movie_frames(BEST_MESH_PATH, bsdf=BSDF)
        write_movie(TRUE_MOVIE_PATH, true_frames, MOVIE_FPS)
        write_movie(STAGE1_MOVIE_PATH, stage1_frames, MOVIE_FPS)
        write_movie(RECON_MOVIE_PATH, recon_frames, MOVIE_FPS)
        comparison_frames = np.concatenate([true_frames, stage1_frames, recon_frames], axis=2)
        write_movie(COMPARISON_MOVIE_PATH, comparison_frames, MOVIE_FPS)
        print(
            f"Saved movies (GT | stage1-only | best reconstruction) to {MOVIE_DIR} "
            f"(comparison: {COMPARISON_MOVIE_PATH})"
        )
    else:
        print("GT STL not available -> skipping movie rendering.")

    summary = {
        "asteroid_id": ASTEROID_ID,
        "target_origin": TARGET_ORIGIN,
        "target_curve_type": TARGET_CURVE_TYPE,
        "bsdf": BSDF,
        "gcn_architecture": "plain",
        "train_camera_count": len(train_indices),
        "validation_camera_count": len(validation_indices),
        "stage1_steps": STAGE1_STEPS,
        "stage1_elapsed_seconds": stage1_elapsed,
        "stage1_best_step": stage1_result.best_step,
        "stage1_best_validation_score": stage1_result.best_score,
        "stage2_steps": STAGE2_STEPS,
        "stage2_elapsed_seconds": stage2_elapsed,
        "stage2_best_step": stage2_result.best_step,
        "stage2_best_validation_score": stage2_result.best_score,
        "refined_mesh_all_cameras_mean_r": float(per_camera_df["Pearson_R"].mean()),
        "refined_mesh_all_cameras_min_r": float(per_camera_df["Pearson_R"].min()),
        "refined_mesh_all_cameras_mean_mse": float(per_camera_df["MSE"].mean()),
        "inr_only_all_cameras_mean_r": float(per_camera_inr_df["Pearson_R"].mean()),
        "inr_only_all_cameras_min_r": float(per_camera_inr_df["Pearson_R"].min()),
        "inr_only_all_cameras_mean_mse": float(per_camera_inr_df["MSE"].mean()),
        "voxel_similarity_pitch": VOXEL_SIMILARITY_PITCH if reference_mesh is not None else None,
        "voxel_similarity_surface": voxel_similarity_surface,
        "voxel_similarity_solid": voxel_similarity_solid,
        # Official challenge metric (lower=better, 0=perfect) -- see
        # hac.eval.official_voxel_dissimilarity's docstring.
        "official_iou_dissimilarity": official_iou_dissimilarity,
        "official_volfrac_dissimilarity": official_volfrac_dissimilarity,
        "best_mesh_path": str(BEST_MESH_PATH),
        "stage1_only_mesh_path": str(STAGE1_MESH_PATH),
        "gt_mesh_path": str(GT_MESH_PATH) if reference_mesh is not None else None,
        "loss_plot_path": str(LOSS_PLOT_PATH),
        "true_movie": str(TRUE_MOVIE_PATH) if reference_mesh is not None else None,
        "stage1_movie": str(STAGE1_MOVIE_PATH) if reference_mesh is not None else None,
        "recon_movie": str(RECON_MOVIE_PATH) if reference_mesh is not None else None,
        "comparison_movie": str(COMPARISON_MOVIE_PATH) if reference_mesh is not None else None,
        "output_dir": str(OUTPUT_DIR),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved summary: {SUMMARY_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
