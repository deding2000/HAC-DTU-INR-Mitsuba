"""INR model definitions for lightcurve-driven mesh reconstruction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SirenLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        w0: float = 30.0,
        is_first: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.w0 = float(w0)
        self.is_first = is_first
        self.is_last = is_last
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = (
            1.0 / self.in_dim
            if self.is_first
            else math.sqrt(6.0 / self.in_dim) / self.w0
        )
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        return y if self.is_last else torch.sin(self.w0 * y)


class RadialNetwork(nn.Module):
    """Pure SIREN deformation field driven by raw 3D spatial coordinates."""

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 256,
        depth: int = 8,
        first_w0: float = 15.0,
        w0: float = 30.0,
        out_dim: int = 1,
        output_activation: str = "identity",
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        if output_activation not in {"tanh", "identity"}:
            raise ValueError("output_activation must be 'tanh' or 'identity'")
        self.output_activation = output_activation

        depth = max(3, int(depth))
        layers: list[SirenLayer] = [
            # GaussianPositionalEncoder removed. First layer takes raw coords.
            SirenLayer(in_dim, hidden_dim, w0=first_w0, is_first=True),
        ]
        for _ in range(depth - 2):
            layers.append(SirenLayer(hidden_dim, hidden_dim, w0=w0))
        layers.append(SirenLayer(hidden_dim, self.out_dim, w0=w0, is_last=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        displacement = self.model(x)
        if self.output_activation == "tanh":
            displacement = torch.tanh(displacement)
        return displacement.squeeze(-1) if self.out_dim == 1 else displacement


class INR(nn.Module):
    """Stage 1: single coarse SIREN deformation field over a star-shaped prior mesh.

    Produces the coarse INR mesh that Stage 2 (``MeshGCN`` or
    ``ResidualMeshGCN``) refines.
    Relies on explicit loss functions for regularization rather than internal
    forward-pass smoothing.

    Parameterization notes:

    - ``input_encoding='direction'``: the SIREN sees unit directions ``v/|v|``
      instead of raw ellipsoid coordinates, keeping the input domain isotropic
      (raw coordinates span (R, R, 1) per axis and bias toward boxy geometry).
    - ``deformation_mode='radial'``: multiplicative log-radial map
      ``v -> v * exp(field)``. Unbounded outward growth, radius always
      positive, mesh stays star-shaped (no fold-overs / tangential drift).
      'normal' (displace along supplied vertex normals) and 'vector'
      (free 3D displacement) reproduce the legacy behaviours.
        - ``final_init_scale`` scales the last SIREN head at initialization. A
            small value starts near the input mesh without changing the function
            class or adding parameters. There is no separate global-scale parameter.
    """

    def __init__(
        self,
        inr_hidden_dim: int = 256,
        inr_num_layers: int = 8,
        coarse_w0: float = 5.0,  # Low frequency for smooth base structure
        deformation_mode: str = "radial",
        output_activation: str = "identity",
        input_encoding: str = "direction",
        final_init_scale: float = 1.0,
    ):
        super().__init__()
        if deformation_mode not in {"radial", "normal", "vector"}:
            raise ValueError("deformation_mode must be 'radial', 'normal' or 'vector'")
        if input_encoding not in {"direction", "raw"}:
            raise ValueError("input_encoding must be 'direction' or 'raw'")
        self.deformation_mode = deformation_mode
        self.input_encoding = input_encoding
        field_dim = 3 if deformation_mode == "vector" else 1

        self.coarse_deformer = RadialNetwork(
            in_dim=3,
            hidden_dim=inr_hidden_dim,
            depth=inr_num_layers,
            first_w0=coarse_w0,
            w0=coarse_w0,
            out_dim=field_dim,
            output_activation=output_activation,
        )
        if final_init_scale != 1.0:
            with torch.no_grad():
                self.coarse_deformer.model[-1].linear.weight.mul_(final_init_scale)
                self.coarse_deformer.model[-1].linear.bias.zero_()

    def _apply_input_encoding(self, base_vertices: torch.Tensor) -> torch.Tensor:
        if self.input_encoding == "direction":
            return base_vertices / base_vertices.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        else:  # raw
            return base_vertices

    def _apply_deformation(
        self,
        base_vertices: torch.Tensor,
        field: torch.Tensor,
        vertex_normals: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.deformation_mode == "radial":
            return base_vertices * torch.exp(field).unsqueeze(-1)
        elif self.deformation_mode == "normal":
            if vertex_normals is None:
                raise ValueError("deformation_mode='normal' requires vertex_normals")
            return base_vertices + vertex_normals * field.unsqueeze(-1)
        else:  # vector
            return base_vertices + field

    def forward(
        self,
        base_vertices: torch.Tensor,
        vertex_normals: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            base_vertices: (..., N, 3) tensor of prior mesh vertices.
            vertex_normals: (..., N, 3) true vertex normals of the prior mesh.
                Required for ``deformation_mode='normal'``; ignored otherwise.
        Returns:
            vertices: Deformed vertices (..., N, 3).
            residual: The total vector displacement applied (vertices - base).
        """
        x = self._apply_input_encoding(base_vertices)
        field = self.coarse_deformer(x)
        vertices = self._apply_deformation(base_vertices, field, vertex_normals)

        residual = vertices - base_vertices

        return vertices, residual


def build_normalized_adjacency(faces: torch.Tensor, num_vertices: int) -> torch.Tensor:
    """Sparse symmetric-normalized adjacency ``A_hat = D^-1/2 (A + I) D^-1/2``.

    Nodes are mesh vertices; undirected edges come from the fixed triangle
    topology. Returns a coalesced sparse COO tensor [V, V] on ``faces.device``
    with the current default dtype.
    """
    f = faces.long()
    device = f.device
    # Both orientations of the three edges of every triangle + self-loops.
    src = torch.cat([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    dst = torch.cat([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    loops = torch.arange(num_vertices, device=device)
    idx = torch.stack([torch.cat([src, loops]), torch.cat([dst, loops])], dim=0)
    ones = torch.ones(idx.shape[1], device=device)
    adj = torch.sparse_coo_tensor(idx, ones, (num_vertices, num_vertices)).coalesce()

    # coalesce() summed duplicate edges (each edge is shared by two faces);
    # binarize back to A + I before normalizing.
    idx = adj.indices()
    vals = torch.ones_like(adj.values())
    deg = torch.zeros(num_vertices, device=device, dtype=vals.dtype)
    deg.index_add_(0, idx[0], vals)
    d_inv_sqrt = deg.clamp_min(1.0).rsqrt()
    vals = d_inv_sqrt[idx[0]] * vals * d_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, vals, (num_vertices, num_vertices)).coalesce()


class GraphConvolution(nn.Module):
    """Single GCN layer ``H' = A_hat @ H @ W + b`` (Pixel2Mesh-style).

    ``A_hat`` is the fixed sparse normalized adjacency (self-loops included),
    so the layer is permutation-consistent with the mesh topology and
    preserves the vertex count.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(int(in_dim), int(out_dim)))
        self.bias = nn.Parameter(torch.zeros(int(out_dim)))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """x: [V, C] or batched [B, V, C]; adjacency: sparse [V, V]."""
        support = x @ self.weight
        if support.dim() == 2:
            out = torch.sparse.mm(adjacency, support)
        else:
            # Sparse mm is 2D-only; every batch element shares the topology.
            out = torch.stack([torch.sparse.mm(adjacency, s) for s in support])
        return out + self.bias


class MeshGCN(nn.Module):
    """Stage 2: predicts residual vertex displacements on the fixed mesh graph.

    ``V_refined = V_inr + delta_V`` with ``delta_V = GCN(V_inr)``. Several
    graph convolutions with ReLU nonlinearities, then a linear GCN output
    layer producing 3D displacements. The output layer is initialized near
    zero (``final_init_scale``) so refinement starts at the Stage-1 mesh.
    """

    def __init__(
        self,
        faces: torch.Tensor,
        num_vertices: int,
        in_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 6,
        final_init_scale: float = 1e-3,
    ):
        super().__init__()
        num_layers = max(2, int(num_layers))
        dims = [int(in_dim)] + [int(hidden_dim)] * (num_layers - 1) + [3]
        self.layers = nn.ModuleList(
            GraphConvolution(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.activations = nn.ModuleList(
            nn.ReLU() for _ in range(num_layers - 1)
        )
        # Conservative output init: Stage 2 starts (almost) at the Stage-1 mesh.
        with torch.no_grad():
            self.layers[-1].weight.mul_(float(final_init_scale))
            self.layers[-1].bias.zero_()
        # Fixed topology -> adjacency is data, rebuilt at construction
        # (persistent=False keeps the sparse tensor out of the state_dict).
        self.register_buffer(
            "adjacency",
            build_normalized_adjacency(faces, num_vertices),
            persistent=False,
        )

    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        """vertices: (..., V, 3) -> displacement delta_V of the same shape."""
        h = vertices
        for act, layer in zip(self.activations, self.layers[:-1]):
            h = act(layer(h, self.adjacency))
        return self.layers[-1](h, self.adjacency)


def aggregate_neighbors(features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    """Apply one shared sparse graph to any batch dimensions in a single sparse mm."""
    num_vertices, channels = features.shape[-2:]
    flattened = features.reshape(-1, num_vertices, channels)
    support = flattened.permute(1, 0, 2).reshape(num_vertices, -1)
    result = torch.sparse.mm(adjacency, support)
    return result.reshape(num_vertices, -1, channels).permute(1, 0, 2).reshape_as(features)


class ResidualGraphBlock(nn.Module):
    """Pre-normalized local/neighbor update with an unfiltered identity path."""

    def __init__(self, hidden_dim: int, residual_scale: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.local = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.activation = nn.SiLU()
        self.residual_scale = residual_scale

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(features)
        update = self.local(normalized) + self.neighbor(aggregate_neighbors(normalized, adjacency))
        return features + self.residual_scale * self.activation(update)


class ResidualMeshGCN(nn.Module):
    """Compact multiscale refiner retaining coordinate and one-ring detail paths.

    LayerNorm acts on each vertex's channels, not on a batch of asteroids.
    Unlike repeated low-pass graph convolutions, the skip paths retain local
    detail. A near-zero linear head preserves the coarse mesh at initialization.
    """

    def __init__(
        self,
        faces: torch.Tensor,
        num_vertices: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        final_init_scale: float = 1e-3,
    ):
        super().__init__()
        if hidden_dim < 1 or num_layers < 1:
            raise ValueError("hidden_dim and num_layers must be positive")
        self.register_buffer("adjacency", build_normalized_adjacency(faces, num_vertices), persistent=False)
        self.input = nn.Linear(6, hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualGraphBlock(hidden_dim, num_layers ** -0.5) for _ in range(num_layers)
        )
        self.output = nn.Linear(hidden_dim, 3)
        with torch.no_grad():
            self.output.weight.mul_(final_init_scale)
            self.output.bias.zero_()

    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        detail = vertices - aggregate_neighbors(vertices, self.adjacency)
        features = self.input(torch.cat([vertices, detail], dim=-1))
        for block in self.blocks:
            features = block(features, self.adjacency)
        return self.output(features)


class LightcurveToMesh(nn.Module):
    """Full two-stage reconstructor: coarse INR + graph-CNN refinement.

    Two-stage optimization strategy:

    - Phase 1 trains ``inr`` alone (lightcurve MSE + convex-hull Chamfer) to a
      coarse, globally accurate mesh; ``gcn`` parameters stay untouched.
        - Phase 2 trains ``inr`` and ``gcn`` jointly using the refined mesh, with
            the INR in a much slower parameter group (``lr_inr_stage2 << lr_graph``).
            The external trainer controls hull and displacement regularization.

    The whole pipeline is differentiable: Stage-2 lightcurve gradients flow
    through the GCN and back into the INR.
    """

    def __init__(
        self,
        faces: torch.Tensor,
        inr_hidden_dim: int = 256,
        inr_num_layers: int = 8,
        coarse_w0: float = 5.0,
        deformation_mode: str = "radial",
        output_activation: str = "identity",
        input_encoding: str = "direction",
        gcn_hidden_dim: int = 128,
        gcn_num_layers: int = 6,
        gcn_final_init_scale: float = 1e-3,
        R: float = 1.0,
        gcn_architecture: str = "plain",
        inr_final_init_scale: float = 1.0,
        canonicalize_coarse: bool = False,
    ):
        super().__init__()
        if gcn_architecture not in {"plain", "residual"}:
            raise ValueError("gcn_architecture must be 'plain' or 'residual'")
        self.R = R
        self.canonicalize_coarse = canonicalize_coarse
        num_vertices = int(faces.max().item()) + 1
        self.inr = INR(
            inr_hidden_dim=inr_hidden_dim,
            inr_num_layers=inr_num_layers,
            coarse_w0=coarse_w0,
            deformation_mode=deformation_mode,
            output_activation=output_activation,
            input_encoding=input_encoding,
            final_init_scale=inr_final_init_scale,
        )
        refiner = MeshGCN if gcn_architecture == "plain" else ResidualMeshGCN
        self.gcn = refiner(
            faces=faces,
            num_vertices=num_vertices,
            hidden_dim=gcn_hidden_dim,
            num_layers=gcn_num_layers,
            final_init_scale=gcn_final_init_scale,
        )

    def canonicalize_to_cylinder(self, vertices: torch.Tensor) -> torch.Tensor:
        """Gauge-fix vertices [..., V, 3] to the challenge cylinder independently.

        Enforces the three exact facts from the challenge metadata:
        - containment: x^2 + y^2 <= R^2, |z| <= 1
        - touching:    min(z) = -1 and max(z) = +1 (object meets both planes)
        - tightness:   max(sqrt(x^2 + y^2)) = R (R is the minimal radius)

        Differentiable analogue of normalize_mesh_to_challenge_cylinder
        (center_xy="bounds"). Mean-normalized lightcurves cannot see uniform
        scale, so this also removes the scale gauge freedom during training.
        """
        if vertices.ndim < 2 or vertices.shape[-1] != 3:
            raise ValueError("vertices must have shape [..., V, 3]")
        z = vertices[..., 2]
        z_max = z.amax(dim=-1, keepdim=True)
        z_min = z.amin(dim=-1, keepdim=True)
        z_mid = 0.5 * (z_max + z_min)
        z_half = (0.5 * (z_max - z_min)).clamp_min(1e-9)
        xy = vertices[..., :2]
        xy_mid = 0.5 * (xy.amax(dim=-2, keepdim=True) + xy.amin(dim=-2, keepdim=True))
        xy_c = xy - xy_mid
        s_xy = self.R / xy_c.norm(dim=-1, keepdim=True).amax(dim=-2, keepdim=True).clamp_min(1e-9)
        return torch.cat([xy_c * s_xy, ((z - z_mid) / z_half).unsqueeze(-1)], dim=-1)

    def forward(
        self,
        base_vertices: torch.Tensor,
        vertex_normals: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            base_vertices: (..., V, 3) prior mesh vertices.
            vertex_normals: (..., V, 3), only used for ``deformation_mode='normal'``.
        Returns:
            vertices_inr: Stage-1 coarse mesh (..., V, 3).
            vertices_refined: Stage-2 refined mesh ``vertices_inr + delta_V``.
        """
        vertices_inr, _ = self.inr(base_vertices, vertex_normals)
        if self.canonicalize_coarse:
            vertices_inr = self.canonicalize_to_cylinder(vertices_inr)
        vertices_refined = vertices_inr + self.gcn(vertices_inr)
        vertices_refined = self.canonicalize_to_cylinder(vertices_refined)
        return vertices_inr, vertices_refined
