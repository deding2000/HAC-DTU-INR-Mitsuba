"""Prior-assisted neural fitting with an externally supplied, fixed forward model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from hac.lc2mesh.metrics import compute_pearson_r
from hac.lc2mesh.model import LightcurveToMesh, aggregate_neighbors
from hac.lc2mesh.utils import chamfer_distance


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 2000
    lr_inr: float = 1e-4
    lr_graph: float = 1e-4
    hull_weight: float = 0.3
    laplacian_weight: float = 0.1
    displacement_weight: float = 0.0
    eval_every: int = 25
    scheduler_patience: int = 5
    clip_grad_norm: float = 1.0
    canonicalize: bool = True
    restore_best: bool = True
    selection: str = "validation"

    def __post_init__(self):
        if self.steps < 0 or self.eval_every < 1 or self.scheduler_patience < 0:
            raise ValueError(
                "steps/patience must be nonnegative and eval_every positive"
            )
        if min(self.lr_inr, self.lr_graph, self.clip_grad_norm) <= 0:
            raise ValueError("learning rates and clipping threshold must be positive")
        if min(self.hull_weight, self.laplacian_weight, self.displacement_weight) < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.selection not in {"validation", "training"}:
            raise ValueError("selection must be 'validation' or 'training'")


@dataclass
class StageResult:
    best_step: int = 0
    best_score: float = float("inf")
    history: dict[str, list[float]] = field(
        default_factory=lambda: {
            key: []
            for key in (
                "step",
                "loss",
                "lc_mse",
                "chamfer",
                "laplacian",
                "displacement",
                "mean_r",
                "min_r",
                "grad_norm",
                "lr",
            )
        }
    )
    evaluations: list[dict[str, float]] = field(default_factory=list)


def split_camera_indices(
    num_cameras: int,
    *,
    seed: int = 1729,
    validation_fraction: float = 0.25,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed camera-level split; do not randomly split adjacent rotation frames."""
    if num_cameras < 2 or not 0 < validation_fraction < 1:
        raise ValueError(
            "Need at least two cameras and a fraction strictly between 0 and 1"
        )
    permutation = torch.randperm(
        num_cameras, generator=torch.Generator().manual_seed(seed)
    )
    count = min(num_cameras - 1, max(1, round(num_cameras * validation_fraction)))
    return permutation[count:].sort().values.to(device), permutation[
        :count
    ].sort().values.to(device)


def displacement_laplacian(
    residual: torch.Tensor, adjacency: torch.Tensor
) -> torch.Tensor:
    """Row-normalized one-ring energy on displacements, not on absolute positions."""
    row_sum = aggregate_neighbors(torch.ones_like(residual[..., :1]), adjacency)
    neighbor_mean = aggregate_neighbors(residual, adjacency) / row_sum.clamp_min(1e-12)
    return (residual - neighbor_mean).square().mean()


def train_stage(
    net: LightcurveToMesh,
    base_vertices: torch.Tensor,
    base_normals: torch.Tensor | None,
    predict_lightcurves: Callable[[torch.Tensor], torch.Tensor],
    observed: torch.Tensor,
    hull_points: torch.Tensor | None,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    config: TrainingConfig,
    *,
    stage: int,
    progress: Callable[[dict[str, float]], None] | None = None,
) -> StageResult:
    """Fit one asteroid; no true mesh or geometry metric enters optimization.

    ``predict_lightcurves`` receives [V, 3] and returns normalized [C, P].
    Every evaluation scores the exact current weights, including step zero.
    Stage 2 regularizes total displacement from the restored coarse solution,
    so INR updates cannot evade the refinement tether. Validation is conditional
    on the supplied prior: an externally fitted prior may contain held-out data.
    """
    if stage not in {1, 2} or base_vertices.ndim != 3 or base_vertices.shape[0] != 1:
        raise ValueError("stage must be 1 or 2 and base_vertices must be [1, V, 3]")
    for indices in (train_indices, validation_indices):
        if indices.ndim != 1 or indices.numel() == 0 or indices.dtype != torch.long:
            raise ValueError(
                "camera indices must be nonempty one-dimensional long tensors"
            )
        if indices.min() < 0 or indices.max() >= observed.shape[0]:
            raise ValueError("camera index out of bounds")
    if torch.isin(train_indices, validation_indices).any():
        raise ValueError("training and validation cameras must be disjoint")
    if config.hull_weight and hull_points is None:
        raise ValueError(
            "A positive hull weight requires an explicit prior, not a silent LC-only fallback"
        )
    if stage == 2 and config.canonicalize != net.canonicalize_coarse:
        raise ValueError(
            "Stage-2 canonicalization must match the model's canonicalize_coarse setting"
        )

    def current_vertices():
        if stage == 1:
            vertices, _ = net.inr(base_vertices, base_normals)
            return (
                net.canonicalize_to_cylinder(vertices)
                if config.canonicalize
                else vertices
            )
        return net(base_vertices, base_normals)[1]

    with torch.no_grad():
        if stage == 1:
            reference = (
                net.canonicalize_to_cylinder(base_vertices)
                if config.canonicalize
                else base_vertices
            )
        else:
            reference, _ = net.inr(base_vertices, base_normals)
            reference = net.canonicalize_to_cylinder(reference)
        reference = reference.detach().clone()

    parameters = list(net.inr.parameters()) if stage == 1 else list(net.parameters())
    groups = [{"params": net.inr.parameters(), "lr": config.lr_inr}]
    if stage == 2:
        groups.append({"params": net.gcn.parameters(), "lr": config.lr_graph})
    optimizer = torch.optim.Adam(groups)
    scheduler = None
    if config.scheduler_patience:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=config.scheduler_patience,
            min_lr=[group["lr"] * 0.05 for group in optimizer.param_groups],
        )
    result = StageResult()
    best_state = None

    def objective(vertices, prediction):
        mse = (prediction[train_indices] - observed[train_indices]).square().mean()
        zero = mse.new_zeros(())
        hull = chamfer_distance(vertices, hull_points) if config.hull_weight else zero
        residual = vertices - reference
        laplacian = (
            displacement_laplacian(residual, net.gcn.adjacency)
            if config.laplacian_weight
            else zero
        )
        displacement = residual.square().mean() if config.displacement_weight else zero
        loss = (
            mse
            + config.hull_weight * hull
            + config.laplacian_weight * laplacian
            + config.displacement_weight * displacement
        )
        return loss, mse, hull, laplacian, displacement

    @torch.no_grad()
    def evaluate(step):
        nonlocal best_state
        net.eval()
        vertices = current_vertices()
        prediction = predict_lightcurves(vertices[0])
        loss, mse, *_ = objective(vertices, prediction)
        validation = (
            (prediction[validation_indices] - observed[validation_indices])
            .square()
            .mean()
        )
        score = validation if config.selection == "validation" else loss
        if not torch.isfinite(score) or not torch.isfinite(loss):
            raise FloatingPointError(
                f"Nonfinite evaluation at stage {stage}, step {step}"
            )
        row = {
            "step": step,
            "train_mse": float(mse),
            "validation_mse": float(validation),
            "score": float(score),
        }
        result.evaluations.append(row)
        if float(score) < result.best_score:
            result.best_score = float(score)
            result.best_step = step
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in net.state_dict().items()
            }
        if scheduler is not None and step:
            scheduler.step(float(score))
        if progress is not None:
            progress(row)
        net.train()

    evaluate(0)
    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        vertices = current_vertices()
        prediction = predict_lightcurves(vertices[0])
        loss, mse, hull, laplacian, displacement = objective(vertices, prediction)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at stage {stage}, step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, config.clip_grad_norm, error_if_nonfinite=True
        )
        optimizer.step()
        with torch.no_grad():
            correlations = compute_pearson_r(
                observed[train_indices], prediction[train_indices]
            )
        values = (
            step,
            loss,
            mse,
            hull,
            laplacian,
            displacement,
            correlations.mean(),
            correlations.min(),
            grad_norm,
            optimizer.param_groups[0]["lr"],
        )
        for key, value in zip(result.history, values):
            result.history[key].append(
                float(value.detach())
                if isinstance(value, torch.Tensor)
                else float(value)
            )
        if step % config.eval_every == 0 or step == config.steps:
            evaluate(step)
    if config.restore_best:
        net.load_state_dict(best_state)
    return result
