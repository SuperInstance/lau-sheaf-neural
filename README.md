# lau-sheaf-neural

**Sheaf-theoretic neural networks — replacing graph Laplacians with sheaf Laplacians to overcome over-squashing in GNNs.**

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

127 tests · 4,031 lines of Rust · 10 modules

---

## What This Does

Standard Graph Neural Networks (GNNs) assign a single feature vector to each node and pass messages along edges via the graph Laplacian. When the graph has low-curvature "bottleneck" edges, distant nodes cannot effectively communicate — this is **over-squashing**.

This crate solves over-squashing by replacing the graph Laplacian with the **sheaf Laplacian**. A **cellular sheaf** assigns a vector space (stalk) to each node and a linear map (restriction map) to each edge, providing richer geometry-aware information flow.

You get:
- A complete **cellular sheaf** data structure with configurable stalks and restriction maps
- **Sheaf Laplacian** (standard, normalized, connection) construction
- **Sheaf diffusion** layers for neural network message passing
- **Sheaf attention** — learn restriction maps from data (like GAT for sheaves)
- **p-Laplacian** for nonlinear sheaf diffusion (p=1 → min-cut, p=2 → linear, p>2 → contrast)
- **Multi-hop sheaf** — compose restriction maps across k-hop paths
- **Sheaf pooling** — hierarchy-aware graph coarsening
- **Over-squashing diagnosis** via sheaf curvature
- **PLATO agents** — model multi-agent communication as a sheaf neural network

---

## Key Idea

```
Standard GNN:    L_graph = D - A          (scalar edge weights)
Sheaf Neural:    L_sheaf = B^T B          (matrix restriction maps)
```

The graph Laplacian treats every edge as a scalar weight. The **sheaf Laplacian** treats every edge as a linear map between stalks. When the restriction maps are identity matrices, the sheaf Laplacian reduces to the standard graph Laplacian. When they're learned, the sheaf adapts to the geometry of the data.

**Over-squashing** occurs on edges with negative Ollivier-Ricci curvature. The sheaf can *fix* these bottlenecks by learning restriction maps that expand the effective channel capacity. This is impossible with scalar weights alone.

---

## Install

```toml
[dependencies]
lau-sheaf-neural = "0.1"
```

Requires Rust 2021 edition. Dependencies: `nalgebra` (with serde), `serde`, `rand`, `thiserror`.

---

## Quick Start

```rust
use lau_sheaf_neural::*;
use nalgebra::{DMatrix, DVector};

// 1. Create a cellular sheaf on a 4-node graph, stalk dimension 3
let mut sheaf = CellularSheaf::new_uniform(4, 3)?;

// Add edges with restriction maps
sheaf.add_edge(0, 1, DMatrix::identity(3, 3))?;
sheaf.add_edge(1, 2, DMatrix::identity(3, 3))?;
sheaf.add_edge(2, 3, DMatrix::identity(3, 3))?;

// 2. Build the sheaf Laplacian
let laplacian = SheafLaplacian::from_sheaf(&sheaf)?;

// 3. Create a diffusion layer (like a GNN layer)
let diffusion = SheafDiffusion::new(sheaf.clone(), 3, Activation::ReLU)?;

// 4. Run one forward pass
let features = DVector::from_element(sheaf.total_dim, 1.0);
let output = diffusion.forward(&features)?;
println!("Output shape: {}", output.len()); // total_dim = 4 × 3 = 12

// 5. Diagnose over-squashing
let curvature = SheafCurvature::from_sheaf(&sheaf, -0.5)?;
let bottlenecks = curvature.bottleneck_edges();
println!("Bottleneck edges: {}", bottlenecks.len());
```

---

## API Reference

### `CellularSheaf`
The core data structure. Assigns a vector space (stalk) to each node and a linear map (restriction map) to each edge.

```rust
let mut sheaf = CellularSheaf::new_uniform(num_nodes, stalk_dim)?;
sheaf.add_edge(source, target, restriction_map)?;
sheaf.validate()?;
let coboundary = sheaf.coboundary_matrix(); // For Laplacian construction
```

**Builder API** for convenient construction:
```rust
let sheaf = SheafBuilder::new(5, 4)  // 5 nodes, stalk dim 4
    .add_edge(0, 1)?
    .add_edge(1, 2)?
    .add_edge(2, 3)?
    .add_edge(3, 4)?
    .build();
```

### `SheafLaplacian`
Constructs L_Σ = B^T B from the coboundary map.

```rust
let lap = SheafLaplacian::from_sheaf(&sheaf)?;
let lap_norm = SheafLaplacian::normalized(&sheaf)?;
let eigenvalues = lap.eigenvalues();
let energy = lap.dirichlet_energy(&cochain);
```

### `ConnectionLaplacian`
For oriented sheaves where R_ji = R_ij^T. Constructs L_conn = D - A∘R.

```rust
let conn_lap = ConnectionLaplacian::from_oriented_sheaf(&sheaf)?;
let spectrum = conn_lap.eigenvalues();
```

### `SheafDiffusion`
The neural network layer. Implements `x' = σ(L_Σ x W + b)`.

```rust
let diff = SheafDiffusion::new(sheaf.clone(), out_dim, Activation::ReLU)?;
let output = diff.forward(&input)?;
let multi_layer = SheafDiffusion::new_multi_layer(sheaf.clone(), &[32, 16, 8], Activation::LeakyReLU(0.01))?;
let output = multi_layer.forward(&input)?;
```

Supported activations: `Identity`, `ReLU`, `Sigmoid`, `Tanh`, `ELU(α)`, `LeakyReLU(α)`.

### `SheafAttention`
Learns restriction maps from data, analogous to GAT.

```rust
let config = AttentionConfig::new(feature_dim, stalk_dim)
    .with_heads(4)
    .symmetric(true);
let attention = SheafAttention::new(config);
let output = attention.forward(&features, &sheaf)?;
// output.sheaf has learned restriction maps
// output.attention_weights shows attention per edge/head
```

### `SheafPLaplacian`
Nonlinear diffusion with the p-Laplacian.

```rust
let p_lap = SheafPLaplacian::new(sheaf.clone(), 1.5)?;
let delta_p = p_lap.apply(&cochain);       // Δ_p(x)
let energy = p_lap.dirichlet_energy(&cochain); // (1/p)Σ||x_v - R_vw x_w||^p
let flow = p_lap.flow(&x0, 100, 0.01);    // Gradient flow of p-Dirichlet energy
```

- **p = 2**: Standard linear sheaf Laplacian
- **p → 1**: Total variation / min-cut
- **p > 2**: Contrast enhancement, emphasizes large differences

### `MultiHopSheaf`
Composes restriction maps across k-hop paths for long-range information flow.

```rust
let multi = MultiHopSheaf::new(sheaf.clone(), 3)?; // up to 3 hops
let map_01_2hop = multi.get_map(0, 3, 2)?; // 2-hop map from node 0 to 3
let lap = multi.sheaf_laplacian()?;          // Laplacian incorporating multi-hop maps
```

### `SheafPooling`
Sheaf-aware graph coarsening for hierarchical representations.

```rust
let pooling = SheafPooling::new(MergeStrategy::Average, 0.5);
let result = pooling.pool(&sheaf, &clusters)?;
// result.coarsened_sheaf: smaller sheaf
// result.projection: down-project features
// result.lift: up-project features (pseudo-inverse)
```

Merge strategies: `DirectSum`, `MaxDim`, `Project(d)`, `Average`.

### `SheafCurvature`
Diagnoses over-squashing by computing sheaf curvature on each edge.

```rust
let curvature = SheafCurvature::from_sheaf(&sheaf, -0.5)?;
let min_kappa = curvature.min_curvature();
let bottlenecks = curvature.bottleneck_edges();
let report = curvature.summary(); // Average, min, max curvature
```

Formula: κ_Σ(i,j) = 1 - ‖R_ij + R_ji‖_F / 2. Negative curvature = bottleneck.

### `PlatoAgent` / `PlatoConfig`
Models multi-agent communication as a sheaf neural network.

```rust
let config = PlatoConfig::default();
let mut plato = PlatoSystem::new(config);
plato.add_agent("agent-0", 4, "researcher");
plato.add_agent("agent-1", 4, "coder");
plato.add_channel(0, 1, "broadcast");
let result = plato.communication_round()?;
// Agents' internal states updated via sheaf diffusion
```

---

## How It Works

The crate implements a complete sheaf neural network pipeline:

```
1. Define Graph + Sheaf       (CellularSheaf)
2. Construct Laplacian        (SheafLaplacian / ConnectionLaplacian)
3. Diagnose Over-squashing    (SheafCurvature)
4. Learn Restriction Maps     (SheafAttention)
5. Run Diffusion              (SheafDiffusion / SheafPLaplacian)
6. Extend to Multi-hop        (MultiHopSheaf)
7. Pool Hierarchically        (SheafPooling)
8. Apply to Agents            (PLATO)
```

**Step 1–2**: A cellular sheaf F on graph G assigns stalk F(v) to each node and restriction maps F_{v≺e}: F(v) → F(w) to each edge. The coboundary matrix δ₀ acts on 0-cochains (node features). The sheaf Laplacian is L_Σ = δ₀^T δ₀.

**Step 3**: Over-squashing is diagnosed via sheaf curvature κ_Σ on edges. Edges with κ < threshold are bottlenecks.

**Step 4**: Sheaf attention learns restriction maps from node features, similar to how GAT learns edge weights but with full matrix maps instead of scalars.

**Step 5**: Sheaf diffusion implements dx/dt = -σ(L_Σ x W + b), the continuous GNN on the sheaf.

**Step 6**: Multi-hop sheaf composes restriction maps across paths of length k, enabling long-range communication while respecting geometry.

**Step 7**: Sheaf pooling coarsens the graph while preserving sheaf structure — stalks are merged and restriction maps are composed.

**Step 8**: PLATO models each agent as a node with stalk = internal state, channels = restriction maps. Communication rounds are sheaf diffusion steps.

---

## The Math

### Cellular Sheaves

A **cellular sheaf** F on a graph G = (V, E) assigns:
- A vector space F(v) (the **stalk**) to each node v ∈ V
- A linear map F_{v≺e}: F(v) → F(w) (the **restriction map**) to each edge e = (v,w)

The space of **0-cochains** C⁰(G, F) = ⊕_v F(v) is the total feature space.

### Sheaf Laplacian

The **coboundary map** δ₀: C⁰ → C¹ acts on edges:
```
(δ₀ x)(e = (v,w)) = F_{w≺e}(x_w) - F_{v≺e}(x_v)
```

The **sheaf Laplacian** is:
```
L_Σ = δ₀^T δ₀
```

This is a positive semi-definite operator on C⁰. When all restriction maps are identity and all stalks have the same dimension, L_Σ reduces to the standard graph Laplacian L ⊗ I_d.

### Over-squashing and Curvature

**Over-squashing** occurs when the Jacobian ‖∂x_v^(T) / ∂x_w^(0)‖ decays exponentially with the distance between v and w. This happens on edges with negative **Ollivier-Ricci curvature**.

The **sheaf curvature** extends this notion:
```
κ_Σ(i,j) = 1 - ‖R_ij + R_ji‖_F / 2
```

When κ_Σ is negative, the edge is a bottleneck. The sheaf can *fix* it by learning restriction maps that increase the effective channel capacity.

### p-Laplacian

The **p-Laplacian** for sheaves:
```
Δ_p f(v) = Σ_{v~w} ‖f(v) - R_{vw} f(w)‖^{p-2} (f(v) - R_{vw} f(w))
```

Special cases:
- **p = 2**: Linear sheaf Laplacian (standard diffusion)
- **p → 1**: Total variation minimization (solves graph min-cut)
- **p → ∞**: Infinity Laplacian (Lipschitz extension)

The **p-Dirichlet energy** is (1/p) Σ_{v~w} ‖f(v) - R_{vw} f(w)‖^p. Gradient flow on this energy is the p-Laplacian flow.

### Connection Laplacian

For **oriented sheaves** where R_ji = R_ij^T:
```
L_conn = D_block - Σ_{i~j} (E_{ij} ⊗ R_{ij} + E_{ji} ⊗ R_{ij}^T)
```

This is the natural operator when edges carry directional information (e.g., SO(d) rotations).

---

## Module Overview

| Module | Tests | Key Types | Purpose |
|--------|-------|-----------|---------|
| `sheaf` | 18 | `CellularSheaf`, `SheafBuilder` | Core data structure |
| `laplacian` | 12 | `SheafLaplacian` | L_Σ = B^T B |
| `diffusion` | 14 | `SheafDiffusion`, `Activation` | Neural network layers |
| `curvature` | 11 | `SheafCurvature`, `EdgeCurvature` | Over-squashing diagnosis |
| `connection` | 9 | `ConnectionLaplacian` | Oriented sheaf operator |
| `attention` | 10 | `SheafAttention`, `AttentionConfig` | Learned restriction maps |
| `p_laplacian` | 13 | `SheafPLaplacian` | Nonlinear diffusion |
| `multihop` | 13 | `MultiHopSheaf`, `ComposedRestrictionMap` | Long-range communication |
| `pooling` | 13 | `SheafPooling`, `PoolingResult` | Hierarchical coarsening |
| `plato` | 14 | `PlatoAgent`, `PlatoConfig` | Agent communication |

---

## References

- **Sheaf Neural Networks**: Hansen & Gebhart, "Sheaf Neural Networks" (2020)
- **Over-squashing**: Topping et al., "Understanding over-squashing and bottlenecks on graphs via curvature" (ICLR 2022)
- **Sheaf Laplacian**: Robinson, "Sheaves are the natural data structure for heterogeneous networks" (2022)
- **p-Laplacian**: Bodnarchuk et al., "Sheaf hypergraph Laplacians" (2023)

---

## License

MIT
