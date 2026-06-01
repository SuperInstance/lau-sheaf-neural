# lau-sheaf-neural

**Sheaf-theoretic neural networks — replacing graph Laplacians with sheaf Laplacians.**

A Rust library implementing sheaf neural networks (SheafNet), which overcome **over-squashing** in GNNs by replacing the graph Laplacian with the **sheaf Laplacian** — a richer operator that encodes local geometry via restriction maps between stalks.

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

---

## What This Does

Standard graph neural networks (GNNs) suffer from **over-squashing**: on graphs with bottleneck edges (low curvature), exponentially-growing neighborhoods get compressed into fixed-size vectors, preventing distant nodes from communicating effectively.

**Sheaf neural networks** solve this by enriching the graph with a **cellular sheaf**:

- Each node v gets a **stalk** F(v) — a vector space (possibly higher-dimensional than the feature vector)
- Each edge (v, w) gets a **restriction map** F_{v≺e}: F(v) → F(w) — a linear map encoding the relationship
- The **sheaf Laplacian** LΣ respects these maps, providing directed, geometry-aware message passing

This crate provides:

- **Cellular sheaves** — the core data structure
- **Sheaf Laplacian** — the operator powering sheaf diffusion
- **Sheaf diffusion** — continuous message passing layers (dx/dt = −σ(LΣ x + b))
- **Sheaf attention** — learn restriction maps from data (attention mechanism)
- **Connection Laplacian** — for oriented sheaves with unitary restriction maps
- **Sheaf curvature** — diagnose over-squashing via Ollivier-Ricci-style curvature
- **p-Laplacian** — nonlinear sheaf diffusion (p > 2 for stronger smoothing)
- **Multi-hop sheaf** — compose restriction maps across k-hop neighborhoods
- **Sheaf pooling** — hierarchy-preserving graph coarsening
- **PLATO agent communication** — multi-agent systems modeled as sheaf neural networks

---

## Key Idea

The **graph Laplacian** L = D − A treats all edges identically — information flows equally in all directions. The **sheaf Laplacian** generalizes this:

> LΣ = B · diag(|Σ_e|²) · B^T

where B is the incidence matrix and Σ_e are the restriction maps. When Σ_e = identity for all edges, LΣ reduces to L. But when Σ_e encodes meaningful relationships (rotations, projections, embeddings), LΣ provides **directed, geometry-aware** information flow.

The sheaf curvature at edge (i, j):

> κ_Σ(i, j) = 1 − ||Σ_{ji} + Σ_{ij}||_F / 2

Negative curvature edges are bottlenecks. The sheaf can **fix** them by learning restriction maps that increase curvature, improving information flow.

---

## Install

```toml
[dependencies]
lau-sheaf-neural = "0.1"
```

---

## Quick Start

```rust
use lau_sheaf_neural::{
    CellularSheaf, SheafLaplacian, SheafDiffusion, SheafCurvature,
    SheafAttention, SheafPLaplacian, SheafPooling, MultiHopSheaf,
};
use nalgebra::{DMatrix, DVector};

// --- Build a cellular sheaf ---
let mut sheaf = CellularSheaf::new_uniform(6, 2)?;  // 6 nodes, 2D stalks
sheaf.add_edge(0, 1, DMatrix::identity(2, 2))?;
sheaf.add_edge(1, 2, DMatrix::identity(2, 2))?;
sheaf.add_edge(2, 3, DMatrix::identity(2, 2))?;
sheaf.add_edge(3, 4, DMatrix::identity(2, 2))?;
sheaf.add_edge(4, 5, DMatrix::identity(2, 2))?;
sheaf.add_edge(5, 0, DMatrix::identity(2, 2))?;

// --- Diagnose over-squashing ---
let curvature = SheafCurvature::from_sheaf(&sheaf, -0.5)?;
let bottlenecks = curvature.bottleneck_edges();
let score = curvature.over_squashing_score();
println!("Over-squashing score: {:.3} (0=safe, 1=severe)", score);

// --- Sheaf Laplacian ---
let lap = SheafLaplacian::from_sheaf(&sheaf)?;
let kernel = lap.kernel();          // Global sections (H⁰)
let gap = lap.spectral_gap();       // Information flow speed
println!("Spectral gap: {:.4} (higher = faster mixing)", gap);

// --- Sheaf diffusion (message passing) ---
let diffusion = SheafDiffusion::new(&sheaf, 0.1)?;  // dt = 0.1
let x0 = DVector::from_element(sheaf.total_dim, 1.0);
let trajectory = diffusion.integrate(&x0, 100);  // 100 diffusion steps

// --- Learn restriction maps with attention ---
let attention = SheafAttention::new(&sheaf, 4)?;  // 4 attention heads
let updated = attention.forward(&x0);

// --- Sheaf pooling (coarsen graph) ---
let pooling = SheafPooling::new(MergeStrategy::MaxDim, 0.5);
let result = pooling.pool_auto(&sheaf)?;
println!("Coarsened: {} → {} nodes", sheaf.num_nodes, result.coarsened_sheaf.num_nodes);

// --- Multi-hop sheaf (long-range communication) ---
let multi = MultiHopSheaf::new(&sheaf, 3)?;  // 3-hop neighborhood
let multi_lap = multi.sheaf_laplacian()?;
```

---

## API Reference

### `CellularSheaf` — The Core Data Structure

```rust
pub struct CellularSheaf {
    pub num_nodes: usize,
    pub stalk_dims: Vec<usize>,
    pub edges: Vec<SheafEdge>,
    pub total_dim: usize,
}
```

| Method | Description |
|--------|-------------|
| `new_uniform(n, dim)` | Uniform stalk dimensions, no edges |
| `new(stalk_dims, &edges)` | Custom stalk dims, edge list |
| `with_restriction_maps(stalk_dims, edges_with_maps)` | Full control over maps |
| `add_edge(source, target, map)` | Add an edge with restriction map |
| `restriction_map(source, target) → &DMatrix` | Get map for edge |
| `restriction_map_mut(source, target) → &mut DMatrix` | Mutable access |
| `extract_stalk(node, cochain) → DVector` | Get node features from stacked vector |
| `set_stalk(node, cochain, value)` | Set node features |
| `coboundary_matrix() → DMatrix` | The δ⁰ operator |
| `neighbors(node) → Vec<usize>` | Outgoing neighbors |
| `in_neighbors(node) → Vec<usize>` | Incoming neighbors |
| `validate() → Result` | Check consistency |
| `trivial(n, dim)` | Trivial sheaf (identity maps) |
| `random(n, dim, edges, seed)` | Random restriction maps |
| `nn_sheaf(n, dim, edges, hidden)` | Neural-network-style sheaf |

### `SheafLaplacian` — The Sheaf Laplacian Operator

```rust
pub struct SheafLaplacian {
    pub matrix: DMatrix<f64>,
    pub total_dim: usize,
}
```

| Method | Description |
|--------|-------------|
| `from_sheaf(&sheaf)` | Build from cellular sheaf |
| `from_sheaf_weighted(&sheaf, weights)` | Edge-weighted variant |
| `kernel() → Vec<DVector>` | Global sections (ker LΣ) |
| `kernel_dimension() → usize` | dim H⁰ |
| `eigenvalues() → Vec<f64>` | Full spectrum |
| `spectral_gap() → f64` | Smallest non-zero eigenvalue |
| `apply(v) → DVector` | LΣ · v |
| `is_global_section(v) → bool` | v ∈ ker LΣ? |

### `SheafDiffusion` — Continuous Message Passing

```rust
pub struct SheafDiffusion { /* sheaf + Laplacian + time step */ }
```

| Method | Description |
|--------|-------------|
| `new(&sheaf, dt)` | Create with time step |
| `step(x, weight, bias)` | One diffusion step: x ← x − dt·σ(LΣx + b) |
| `integrate(x0, steps)` | Full trajectory |
| `with_nonlinearity(activation)` | ReLU, sigmoid, tanh, or custom |

### `SheafAttention` — Learn Restriction Maps

```rust
pub struct SheafAttention { /* attention heads over sheaf edges */ }
```

| Method | Description |
|--------|-------------|
| `new(&sheaf, n_heads)` | Create with attention heads |
| `forward(x) → DVector` | One forward pass (updates restriction maps + features) |
| `attention_weights() → Vec<f64>` | Current attention scores |

### `ConnectionLaplacian` — Oriented Sheaf Laplacian

```rust
pub struct ConnectionLaplacian { /* PSD matrix for oriented sheaves */ }
```

| Method | Description |
|--------|-------------|
| `from_oriented_sheaf(&sheaf)` | Build for oriented sheaf |
| `from_oriented_edges(&sheaf, &edges)` | Select subset of edges |
| `is_psd() → bool` | Positive semi-definite check |
| `connection_energy(x) → f64` | x^T L_conn x |
| `spectral_gap() → f64` | Smallest non-zero eigenvalue |
| `eigendecompose() → (evals, evecs)` | Full decomposition |
| `to_connection_matrix() → DMatrix` | The connection matrix |

### `SheafCurvature` — Over-Squashing Diagnosis

```rust
pub struct SheafCurvature {
    pub edge_curvatures: Vec<EdgeCurvature>,
    pub bottleneck_threshold: f64,
}
```

| Method | Description |
|--------|-------------|
| `from_sheaf(&sheaf, threshold)` | Compute all curvatures |
| `min_curvature() → Option<f64>` | Worst bottleneck |
| `mean_curvature() → f64` | Average curvature |
| `bottleneck_edges() → Vec<&EdgeCurvature>` | All bottleneck edges |
| `over_squashing_score() → f64` | Fraction of bottleneck edges (0–1) |
| `effective_resistance() → Vec<(usize, usize, f64)>` | Per-edge resistance |
| `suggest_dimension_increases(&sheaf)` | Stalk dimension recommendations |

### `SheafPLaplacian` — Nonlinear Sheaf Diffusion

```rust
pub struct SheafPLaplacian { /* p value + sheaf */ }
```

| Method | Description |
|--------|-------------|
| `new(&sheaf, p)` | Create with p value (p > 2 for stronger diffusion) |
| `apply(x) → DVector` | Δ_p(x) = div(|∇_Σ x|^{p−2} ∇_Σ x) |
| `p_energy(x) → f64` | The p-Dirichlet energy |
| `gradient_flow(x0, dt, steps)` | Minimize p-energy via gradient descent |

### `MultiHopSheaf` — Long-Range Communication

```rust
pub struct MultiHopSheaf { /* composed restriction maps across k hops */ }
```

| Method | Description |
|--------|-------------|
| `new(&sheaf, k)` | Build k-hop sheaf (compose maps up to k edges) |
| `sheaf_laplacian() → SheafLaplacian` | Laplacian of the multi-hop sheaf |
| `composed_map(source, target) → Option<DMatrix>` | k-hop restriction map |

### `SheafPooling` — Graph Coarsening

```rust
pub enum MergeStrategy { MaxDim, Average, Project(usize) }

pub struct SheafPooling { /* strategy + ratio */ }
```

| Method | Description |
|--------|-------------|
| `new(strategy, ratio)` | Create (ratio = target fraction of nodes) |
| `pool(&sheaf, &clusters)` | Coarsen with given clusters |
| `pool_auto(&sheaf)` | Automatic clustering |
| `hierarchical_pool(&sheaf, levels)` | Multi-level coarsening |
| `project_features(result, features)` | Features → coarsened features |
| `lift_features(result, coarse_features)` | Coarsened → original features |

### `Plato` — Multi-Agent Communication as Sheaf NN

The `plato` module models PLATO-style agent communication as a sheaf neural network, where agents are nodes, communication channels are edges with restriction maps, and the diffusion process implements message passing.

---

## How It Works

### Architecture

```
CellularSheaf (stalks + restriction maps)
  ├─ SheafLaplacian (LΣ = B diag(|Σ|²) B^T)
  │    ├─ SheafDiffusion (dx/dt = −σ(LΣ x + b))
  │    ├─ SheafPLaplacian (nonlinear p-diffusion)
  │    └─ ConnectionLaplacian (oriented sheaves)
  ├─ SheafAttention (learned restriction maps)
  ├─ SheafCurvature (κ(i,j) = 1 − ||Σ_{ij} + Σ_{ji}||_F/2)
  ├─ MultiHopSheaf (k-hop composed maps)
  └─ SheafPooling (hierarchical coarsening)
```

### Module Map

| Module | Contents |
|--------|----------|
| `sheaf` | `CellularSheaf`, `SheafEdge`, `SheafError` — core data structure |
| `laplacian` | `SheafLaplacian` — construction, spectrum, kernel |
| `diffusion` | `SheafDiffusion` — continuous message passing |
| `curvature` | `SheafCurvature` — over-squashing diagnosis |
| `connection` | `ConnectionLaplacian` — oriented sheaf operator |
| `attention` | `SheafAttention` — learned restriction maps |
| `p_laplacian` | `SheafPLaplacian` — nonlinear diffusion |
| `multihop` | `MultiHopSheaf` — long-range composed maps |
| `pooling` | `SheafPooling` — graph coarsening |
| `plato` | PLATO agent communication as sheaf neural network |

---

## The Math

### Cellular Sheaves on Graphs

A **cellular sheaf** F on a graph G assigns:
- A **stalk** F(v) ≅ R^{d_v} to each node v (a vector space)
- A **restriction map** F_{v≺e}: F(v) → F(w) to each directed edge e = (v, w)

The **0-cochain space** C⁰(G, F) = ⊕_v F(v) is the total feature space. A **cochain** x ∈ C⁰ assigns a vector x_v ∈ F(v) to each node.

### The Coboundary Map

The **coboundary** δ⁰: C⁰ → C¹ maps node features to edge features:

> (δ⁰ x)_{(i→j)} = F_{j≺e}(x_j) − F_{i≺e}(x_i)

This measures the "mismatch" between adjacent nodes' features after mapping through the restriction maps.

### The Sheaf Laplacian

> LΣ = (δ⁰)^T δ⁰

In matrix form, for each edge (i→j):

> (LΣ)_{ii} += Σ_e^T Σ_e  
> (LΣ)_{ij} += −Σ_e^T

The sheaf Laplacian is always **positive semi-definite**. Its kernel consists of **global sections** — cochains where x_j = F_{j≺e}^{-1} F_{i≺e}(x_i) for all edges (perfectly consistent across the graph).

### Over-Squashing and Curvature

**Over-squashing** occurs when the **balanced form curvature** of the graph is too negative at certain edges. For the sheaf:

> κ_Σ(i, j) = 1 − ||Σ_{ji} + Σ_{ij}||_F / 2

- κ > 0: Information flows freely (expanding neighborhood)
- κ ≈ 0: Marginal
- κ < 0: Bottleneck — information gets compressed

The sheaf can **cure** over-squashing: by learning restriction maps that make κ more positive, we increase the channel capacity of bottleneck edges.

### Sheaf Diffusion

The continuous sheaf neural network evolves features via:

> dx/dt = −σ(LΣ x + b)

where σ is a nonlinearity (ReLU, etc.) and b is a bias. This is **heat equation on the sheaf** — features diffuse along edges according to the restriction maps.

**Key property**: Unlike standard GNN diffusion, sheaf diffusion respects the local geometry. Two nodes connected by a rotation map exchange rotational information; nodes connected by projection maps exchange projected information.

### The p-Laplacian

The **p-Laplacian** generalizes diffusion with a nonlinearity:

> Δ_p(x) = div(|∇_Σ x|^{p−2} ∇_Σ x)

For p = 2, this reduces to the standard sheaf Laplacian. For p > 2, diffusion is **stronger** in high-gradient regions, providing more aggressive smoothing. The p-Dirichlet energy:

> E_p(x) = (1/p) Σ_e |(δ⁰ x)_e|^p

### Multi-Hop Sheaves

To enable **long-range communication**, we compose restriction maps across k edges:

> F_{i→j}^{(k)} = F_{j_{k-1}≺e_{k-1}} ∘ … ∘ F_{j_0≺e_0}

The k-hop sheaf has the same stalks but composed restriction maps, allowing information to flow across distant nodes in a single diffusion step.

### Connection Laplacian

For **oriented sheaves** (where restriction maps are orthogonal/unitary), the connection Laplacian is:

> L_conn = ½ (LΣ + LΣ^T)

with special structure: off-diagonal blocks are −R_{ij} (not −R_{ij}^T R_{ij}). This is the operator used in **spectral clustering on manifolds** and **synchronization problems**.

---

## Testing

```bash
cargo test
```

**127 tests** covering:

- Cellular sheaf construction (uniform, custom dims, restriction maps)
- Stalk extraction and assignment
- Edge operations (add, query, neighbors)
- Coboundary matrix construction
- Sheaf Laplacian (construction, kernel, spectrum, spectral gap)
- Sheaf diffusion (integration, nonlinearity, stability)
- Connection Laplacian (PSD verification, energy, eigendecomposition)
- Sheaf curvature (identity/zero/scaled maps, bottleneck detection)
- Over-squashing diagnosis (score, effective resistance, suggestions)
- p-Laplacian (energy, gradient flow)
- Multi-hop sheaf (map composition, k-hop Laplacian)
- Sheaf pooling (merge strategies, auto-clustering, hierarchical)
- Sheaf attention (forward pass, weight updates)
- Property-based tests (via proptest)

---

## License

MIT
