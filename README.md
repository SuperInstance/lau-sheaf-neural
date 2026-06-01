# lau-sheaf-neural

**Sheaf-theoretic neural networks** — replacing graph Laplacians with sheaf Laplacians to overcome over-squashing in GNNs. Implements cellular sheaves on graphs, sheaf Laplacians, sheaf diffusion layers, curvature-based bottleneck detection, connection Laplacians, sheaf attention, p-Laplacian diffusion, multi-hop composition, sheaf pooling, and the PLATO agent communication framework.

## What This Does

Standard GNNs assign a single feature vector to each node and pass messages via the graph Laplacian. When the graph has low-curvature "bottleneck" edges, distant nodes can't effectively communicate — this is **over-squashing**.

A **cellular sheaf** on a graph assigns a vector space (stalk) to each node and a linear map (restriction map) to each edge. The resulting **sheaf Laplacian** respects the local geometry, providing richer information flow. When restriction maps are learned from data (sheaf attention), the network can adaptively route information around bottlenecks.

You get:

- **Cellular sheaves** on arbitrary graphs with heterogeneous stalk dimensions
- **Sheaf Laplacians** (normalized and unnormalized) via coboundary maps
- **Sheaf diffusion layers** with configurable activation functions
- **Connection Laplacians** for oriented sheaves
- **Sheaf attention** — learn restriction maps dynamically from node features (analogous to GAT)
- **p-Laplacian** nonlinear diffusion for contrast enhancement or total variation minimization
- **Multi-hop sheaf** — compose restriction maps across k-hop neighborhoods
- **Sheaf-aware pooling** — coarsen graphs while preserving sheaf structure
- **Curvature analysis** — diagnose over-squashing bottlenecks via sheaf curvature
- **PLATO framework** — model multi-agent communication as sheaf neural networks

## Key Idea

The sheaf Laplacian L_Σ = B^T B (where B is the coboundary map) generalizes the graph Laplacian. For the trivial sheaf (identity restriction maps), L_Σ reduces to the standard graph Laplacian. But with learned restriction maps, L_Σ can:

- Route information preferentially along semantically relevant edges
- Overcome topological bottlenecks by increasing stalk dimensions on bottleneck edges
- Detect over-squashing via sheaf curvature: κ(i,j) = 1 − ‖R_{ij} + R_{ji}‖_F / 2

## Install

```toml
[dependencies]
lau-sheaf-neural = "0.1.0"
```

Requires `nalgebra` (with `serde-serialize`), `serde`, `rand`, and `thiserror`.

## Quick Start

```rust
use lau_sheaf_neural::{CellularSheaf, SheafLaplacian, SheafDiffusion, SheafCurvature};
use nalgebra::DVector;

// Create a sheaf on a 4-node graph with stalk dimension 2
let mut sheaf = CellularSheaf::new_uniform(4, 2)?;
// Add edges with identity restriction maps
sheaf.add_edge(0, 1)?;
sheaf.add_edge(1, 2)?;
sheaf.add_edge(2, 3)?;

// Build the sheaf Laplacian
let laplacian = SheafLaplacian::from_sheaf(&sheaf)?;
println!("Laplacian {}x{}", laplacian.total_dim, laplacian.total_dim);

// Diagnose over-squashing
let curvature = SheafCurvature::from_sheaf(&sheaf, -0.5)?;
for ec in &curvature.edge_curvatures {
    if ec.is_bottleneck {
        println!("Bottleneck edge ({}, {}): curvature = {:.3}",
            ec.source, ec.target, ec.curvature);
    }
}

// Run sheaf diffusion
let mut diffusion = SheafDiffusion::new(sheaf.clone());
let x = DVector::from_vec(vec![1.0; sheaf.total_dim]);
let output = diffusion.forward(&x, 3);
```

## API Reference

### `sheaf` — Cellular Sheaf Data Structure

| Type | Description |
|------|-------------|
| `CellularSheaf` | A cellular sheaf on a graph: stalks on nodes, restriction maps on edges. |
| `SheafEdge` | An edge (source, target) with its restriction map matrix. |
| `SheafBuilder` | Builder pattern for constructing sheaves. |
| `SheafError` | Error type: NodeNotFound, EdgeNotFound, DimensionMismatch, etc. |

**`CellularSheaf` methods:**

- `new_uniform(num_nodes, stalk_dim)` — All nodes same stalk dim, no edges.
- `new_heterogeneous(stalk_dims)` — Different stalk dimensions per node.
- `add_edge(source, target)` — Add edge with identity restriction map.
- `add_edge_with_map(source, target, restriction_map)` — Custom restriction map.
- `validate()` — Check all dimensions are consistent.
- `stalk_dim(node) → usize` — Get stalk dimension at a node.
- `restriction_map(source, target) → Option<&DMatrix<f64>>`
- `coboundary_matrix() → DMatrix<f64>` — The coboundary operator B.
- `node_offset(node) → usize` — Offset into the stacked cochain vector.
- `extract_stalk(node, &cochain) → DVector<f64>` — Get node's feature slice.
- `set_stalk(node, &features, &mut cochain)` — Set node's features.
- `total_cochain(&per_node_features) → DVector<f64>` — Stack all node features.

---

### `laplacian` — Sheaf Laplacian

| Type | Description |
|------|-------------|
| `SheafLaplacian` | The sheaf Laplacian L_Σ = B^T B, a PSD matrix of size total_dim × total_dim. |

**Methods:**

- `from_sheaf(&sheaf) → Result<Self>` — Construct from a sheaf.
- `normalized(&sheaf) → Result<Self>` — D^{-1/2} L D^{-1/2} normalized form.
- `eigenvalues() → Vec<f64>` — Eigenvalues of L_Σ.
- `is_psd(tol) → bool` — Positive semi-definiteness check.
- `apply(&x) → DVector<f64>` — Multiply L_Σ x.
- `dirichlet_energy(&x) → f64` — x^T L_Σ x.

---

### `diffusion` — Sheaf Diffusion Layers

| Type | Description |
|------|-------------|
| `Activation` | Enum: Identity, ReLU, Sigmoid, Tanh, ELU(α), LeakyReLU(α). |
| `SheafDiffusionLayer` | Single layer: x' = σ(L_Σ x W + b). |
| `SheafDiffusion` | Multi-layer sheaf diffusion network. |

**`Activation` methods:**

- `apply(x) → f64`, `apply_vec(&v) → DVector<f64>`

**`SheafDiffusionLayer`:**

- `new(in_dim, out_dim, activation)` — Random initialization.
- `forward(&x, &laplacian) → DVector<f64>`

**`SheafDiffusion`:**

- `new(sheaf)` — Create with default architecture.
- `with_layers(sheaf, &[hidden_dims], activation)` — Custom architecture.
- `forward(&x, num_steps) → DVector<f64>` — Run through all layers.

---

### `curvature` — Sheaf Curvature and Over-Squashing

| Type | Description |
|------|-------------|
| `EdgeCurvature` | Curvature result for one edge: curvature value, restriction norm, bottleneck flag. |
| `SheafCurvature` | Curvature analysis for all edges. |

**`SheafCurvature`:**

- `from_sheaf(&sheaf, threshold) → Result<Self>` — Compute all edge curvatures.
- `min_curvature() → Option<f64>` — Most negative curvature.
- `bottlenecks() → Vec<&EdgeCurvature>` — Edges below threshold.
- `mean_curvature() → f64`, `num_bottlenecks() → usize`

---

### `connection` — Connection Laplacian

| Type | Description |
|------|-------------|
| `ConnectionLaplacian` | L_conn = D − A ∘ R for oriented sheaves. |

**Methods:**

- `from_oriented_sheaf(&sheaf) → Result<Self>` — Construct using R_{ji} = R_{ij}^T.
- `eigenvalues() → Vec<f64>`, `is_psd(tol) → bool`
- `apply(&x) → DVector<f64>`

---

### `attention` — Sheaf Attention (Learned Restriction Maps)

| Type | Description |
|------|-------------|
| `AttentionConfig` | Configuration: in_dim, stalk_dim, num_heads, symmetric, dropout, concat. |
| `SheafAttention` | Learn restriction maps via query-key-value attention. |
| `AttentionOutput` | Result: updated sheaf, attention weights, loss. |

**`SheafAttention`:**

- `new(config) → Self` — Random initialization.
- `forward(&sheaf, &features) → AttentionOutput` — Compute attention-weighted restriction maps.

---

### `p_laplacian` — Nonlinear Sheaf Diffusion

| Type | Description |
|------|-------------|
| `SheafPLaplacian` | The p-Laplacian: Δ_p with configurable exponent p. |

**Methods:**

- `new(sheaf, p) → Result<Self>` — p > 0. Use p=2 for linear, p→1 for total variation.
- `apply(&x) → DVector<f64>` — Apply Δ_p to a cochain.
- `dirichlet_energy(&x) → f64` — The p-Dirichlet energy.
- `gradient_descent(&x, dt, steps) → DVector<f64>` — Minimize p-Dirichlet energy.

---

### `multihop` — Multi-Hop Sheaf Composition

| Type | Description |
|------|-------------|
| `ComposedRestrictionMap` | A composed map across a k-hop path. |
| `MultiHopSheaf` | Pre-computed k-hop restriction maps. |

**`MultiHopSheaf`:**

- `new(sheaf, max_hops) → Result<Self>` — Compute all k-hop maps.
- `k_hop_laplacian(k) → Result<SheafLaplacian>` — Sheaf Laplacian using k-hop maps.
- `get_map(source, target, hops) → Option<&ComposedRestrictionMap>`
- `diffuse(&x, dt, steps) → DVector<f64>` — Multi-hop diffusion.

---

### `pooling` — Sheaf-Aware Graph Pooling

| Type | Description |
|------|-------------|
| `MergeStrategy` | Enum: DirectSum, MaxDim, Project(d), Average. |
| `PoolingResult` | Coarsened sheaf, node assignments, projection and lift matrices. |
| `SheafPooling` | Pooling module with configurable merge strategy and ratio. |

**`SheafPooling`:**

- `new(merge_strategy, pool_ratio) → Self`
- `pool(&sheaf, &clusters) → Result<PoolingResult>` — One pooling step.
- `random_clustering(num_nodes, num_clusters) → Vec<Vec<usize>>`

---

### `plato` — PLATO Agent Communication Framework

| Type | Description |
|------|-------------|
| `PlatoAgent` | An agent with id, node index, state dimension, current state, and role. |
| `CommunicationChannel` | A channel between agents with a channel type and restriction map. |
| `PlatoConfig` | Configuration: state_dim, diffusion_steps, dt, learned_maps, curvature_threshold. |
| `CommunicationRoundResult` | Updated states, bottleneck info, curvature analysis. |
| `PlatoNetwork` | The full PLATO multi-agent system. |

**`PlatoNetwork`:**

- `new(config) → Self` — Empty network.
- `add_agent(&mut self, id, role)` — Add an agent.
- `add_channel(&mut self, source, target, channel_type)` — Add communication channel.
- `communicate(&mut self) → CommunicationRoundResult` — Run one round of sheaf diffusion.
- `detect_bottlenecks(&self) → Vec<EdgeCurvature>` — Find over-squashing channels.
- `get_agent(&self, id) → Option<&PlatoAgent>`, `num_agents() → usize`

## How It Works

1. **Sheaf construction**: Each node gets a stalk (vector space of dimension d_i). Each directed edge (i→j) gets a restriction map R_{ij}: F(i) → F(j), a matrix of shape (d_j × d_i).

2. **Coboundary map**: The coboundary operator B maps 0-cochains to 1-cochains: (Bx)_{ij} = x_j − R_{ji} x_i for each edge. It's a matrix of size (total_edge_dim × total_dim).

3. **Sheaf Laplacian**: L_Σ = B^T B. This is the key operator — it's always PSD, and its kernel consists of global sections (0-cochains that are compatible across all edges).

4. **Diffusion**: The continuous flow dx/dt = −L_Σ x (discretized as x ← x − dt · L_Σ x) propagates information across the graph, respecting the sheaf geometry. With nonlinear activation, this becomes a neural network layer.

5. **Attention**: Instead of fixed restriction maps, learn them from data: R_{ij} = softmax(a^T [W x_i || W x_j]) · V, where W and V are learnable projections and a is an attention vector.

6. **Curvature**: The sheaf curvature κ(i,j) = 1 − ‖R_{ij} + R_{ji}‖_F / 2 measures how well information flows through edge (i,j). Negative curvature → bottleneck.

7. **Pooling**: Merge clusters of nodes into supernodes. The new stalk is the direct sum (or projection) of the merged stalks, and restriction maps are composed accordingly.

## The Math

### Cellular Sheaves

A **cellular sheaf** F on a graph G assigns:
- A vector space F(v) (the **stalk**) to each node v
- A linear map F_{v≺e}: F(v) → F(w) (the **restriction map**) to each edge e = (v,w)

The **0-cochain space** C⁰(G,F) = ⊕_v F(v) is the space of node features. A **global section** is a 0-cochain x where x_v = R_{vw}(x_w) for all edges — the sheaf Laplacian's kernel.

### Sheaf Laplacian

L_Σ = B^T B where B is the coboundary. This is always PSD with ker(L_Σ) = global sections. For the trivial sheaf (all maps = identity), L_Σ = L_graph ⊗ I_d.

### p-Laplacian

Δ_p x(v) = Σ_{v~w} ‖x_v − R_{vw} x_w‖^{p−2} (x_v − R_{vw} x_w). At p=2 this is linear (the sheaf Laplacian). At p=1 it solves the sheaf min-cut problem. At p>2 it enhances contrast.

### Over-Squashing

Information bottlenecks arise when graph curvature is too negative. The sheaf can "fix" bottlenecks by increasing stalk dimensions or learning restriction maps that route information around them. The Bernoulli free energy of the sheaf Laplacian provides a theoretical bound on information flow.

## License

MIT
