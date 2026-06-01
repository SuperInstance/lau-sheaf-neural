# lau-sheaf-neural

**Sheaf-theoretic neural networks — replacing graph Laplacians with sheaf Laplacians to overcome over-squashing in GNNs.**

A Rust library that implements cellular sheaves on graphs, the sheaf Laplacian, sheaf diffusion (message passing), curvature-based over-squashing diagnosis, connection Laplacians, learned attention-based restriction maps, nonlinear p-Laplacian diffusion, multi-hop sheaf composition, sheaf-aware graph pooling, and a PLATO multi-agent communication framework — all built on the insight that sheaves provide richer geometry than graphs alone.

[![127 tests passing](https://img.shields.io/badge/tests-127%20passing-brightgreen)]()

---

## What This Does

Standard Graph Neural Networks (GNNs) assign one feature vector per node and pass messages via the graph Laplacian. When the graph has low-curvature bottleneck edges, distant nodes can't communicate effectively — this is **over-squashing**.

A **cellular sheaf** enriches the graph: each node gets a vector space (the *stalk*), and each edge gets a linear map (the *restriction map*). The resulting **sheaf Laplacian** respects this local geometry, enabling richer information flow. This library implements the full pipeline:

```
Graph + Sheaf → Sheaf Laplacian → Sheaf Diffusion → Output
                    ↑
              Restriction maps
              (fixed or learned)
```

## Key Idea

The sheaf Laplacian **L_Σ = B^T B** generalizes the graph Laplacian by incorporating restriction maps. When restriction maps are identity, it reduces to the standard graph Laplacian. When they're learned from data (via sheaf attention), the sheaf adapts to break bottlenecks and improve information flow. Over-squashing is diagnosed via sheaf curvature: κ(i,j) = 1 − ‖R_ij + R_ji‖_F / 2. Edges with very negative curvature are bottlenecks.

## Install

```toml
[dependencies]
lau-sheaf-neural = "0.1.0"
```

Requires Rust 2021 edition. Dependencies: `nalgebra` (with serde), `serde`, `serde_json`, `rand`, `rand_distr`, `thiserror`, `approx`.

## Quick Start

```rust
use lau_sheaf_neural::{CellularSheaf, SheafLaplacian, SheafDiffusion};
use nalgebra::{DMatrix, DVector};

fn main() {
    // Create a sheaf: 4 nodes, each with 3-dimensional stalks
    let mut sheaf = CellularSheaf::new_uniform(4, 3).unwrap();
    
    // Add edges with identity restriction maps
    sheaf.add_edge(0, 1, DMatrix::identity(3, 3)).unwrap();
    sheaf.add_edge(1, 0, DMatrix::identity(3, 3)).unwrap();
    sheaf.add_edge(1, 2, DMatrix::identity(3, 3)).unwrap();
    sheaf.add_edge(2, 1, DMatrix::identity(3, 3)).unwrap();
    sheaf.add_edge(2, 3, DMatrix::identity(3, 3)).unwrap();
    sheaf.add_edge(3, 2, DMatrix::identity(3, 3)).unwrap();

    // Build the sheaf Laplacian
    let laplacian = SheafLaplacian::from_sheaf(&sheaf).unwrap();
    println!("Spectral gap: {:.4}", laplacian.spectral_gap());
    println!("Dirichlet energy of constant signal: {:.6}", 
        laplacian.dirichlet_energy(&DVector::from_element(12, 1.0)));

    // Create a sheaf diffusion network and propagate
    let diffusion = SheafDiffusion::new(sheaf.clone(), vec![8], 0.1).unwrap();
    let input = DVector::from_element(12, 1.0);
    let output = diffusion.forward(&input);
    println!("Output dimension: {}", output.len());
}
```

## API Reference

### Module: `sheaf` — Cellular Sheaf Data Structure

| Type / Method | Description |
|---|---|
| `CellularSheaf` | Core type: graph with stalks and restriction maps |
| `CellularSheaf::new_uniform(n, d)` | n nodes, each with d-dimensional stalk, no edges |
| `CellularSheaf::new(stalk_dims, edges)` | Heterogeneous stalk dimensions with edge pairs |
| `CellularSheaf::with_restriction_maps(stalk_dims, edges_with_maps)` | Full constructor with explicit maps |
| `.add_edge(source, target, map)` | Add an edge with restriction map |
| `.restriction_map(source, target) → &DMatrix` | Get the restriction map for an edge |
| `.restriction_map_mut(source, target) → &mut DMatrix` | Mutable reference to restriction map |
| `.has_edge(source, target) → bool` | Check edge existence |
| `.neighbors(node) → Vec<usize>` | Outgoing neighbors |
| `.in_neighbors(node) → Vec<usize>` | Incoming neighbors |
| `.degree(node) → usize` | Outgoing edge count |
| `.num_edges() → usize` | Total edges |
| `.stalk_dim(node) → usize` | Stalk dimension at node |
| `.node_offset(node) → usize` | Starting index in stacked vector |
| `.extract_stalk(node, cochain) → DVector` | Extract node's features from stacked vector |
| `.set_stalk(node, cochain, value)` | Set node's features in stacked vector |
| `.coboundary_matrix() → DMatrix` | Build coboundary B for Laplacian construction |
| `.adjacency() → Vec<Vec<usize>>` | Adjacency list |
| `.validate() → Result` | Check dimension consistency |
| `.rebuild_edge_index()` | Rebuild after deserialization |
| `SheafEdge` | Edge with source, target, restriction_map |
| `SheafBuilder` | Builder pattern for sheaves |
| `SheafBuilder::new(n, d).edge(s, t).build()` | Builder API |
| `SheafBuilder::edge_with_map(s, t, map)` | Builder with custom restriction map |
| `SheafError` | Error enum: NodeNotFound, EdgeNotFound, DimensionMismatch, etc. |

### Module: `laplacian` — Sheaf Laplacian

| Type / Method | Description |
|---|---|
| `SheafLaplacian` | The L_Σ = B^T B operator |
| `SheafLaplacian::from_sheaf(sheaf)` | Construct from coboundary matrix |
| `SheafLaplacian::normalized(sheaf)` | D^{-1/2} L D^{-1/2} normalized Laplacian |
| `SheafLaplacian::random_walk_normalized(sheaf)` | D^{-1} L random walk Laplacian |
| `.apply(x) → DVector` | Compute L·x |
| `.dirichlet_energy(x) → f64` | x^T L x (measures "non-harmonicity") |
| `.is_psd() → bool` | Check all eigenvalues ≥ 0 |
| `.smallest_eigenvalue() → f64` | Min eigenvalue |
| `.spectral_gap() → f64` | Second-smallest eigenvalue (governs mixing rate) |
| `.algebraic_connectivity() → f64` | Same as spectral gap |
| `.trace() → f64` | Trace of Laplacian |
| `.kernel_dimension() → usize` | Count of zero eigenvalues |
| `.matrix` | Raw DMatrix<f64> |
| `.total_dim`, `.num_nodes` | Dimensions |

### Module: `diffusion` — Sheaf Diffusion Layers

| Type / Method | Description |
|---|---|
| `enum Activation` | Identity, ReLU, Sigmoid, Tanh, ELU(α), LeakyReLU(α) |
| `Activation::apply(x) → f64` | Scalar activation |
| `Activation::apply_vec(v) → DVector` | Element-wise activation |
| `SheafDiffusionLayer` | Single layer: x' = σ((I − dt·L) x W + b) |
| `SheafDiffusionLayer::new(in_dim, out_dim, activation)` | Random init |
| `SheafDiffusionLayer::with_weights(w, b, activation)` | Specific weights |
| `.forward(x, laplacian) → DVector` | Forward pass with Laplacian diffusion |
| `.forward_linear(x) → DVector` | Forward without Laplacian |
| `.weight`, `.bias`, `.activation`, `.dt` | Fields |
| `SheafDiffusion` | Multi-layer diffusion network |
| `SheafDiffusion::new(sheaf, layer_dims, dt)` | Build multi-layer network |
| `.forward(x) → DVector` | Full forward pass through all layers |
| `.laplacian` | The underlying SheafLaplacian |
| `.layers` | Vec of SheafDiffusionLayer |
| `.sheaf` | Reference to underlying CellularSheaf |

### Module: `curvature` — Over-Squashing Diagnosis

| Type / Method | Description |
|---|---|
| `EdgeCurvature` | Per-edge: source, target, curvature, restriction_norm, is_bottleneck |
| `SheafCurvature` | Curvature analyzer for all edges |
| `SheafCurvature::from_sheaf(sheaf, threshold)` | Compute curvatures; edges below threshold are bottlenecks |
| `.min_curvature() → Option<f64>` | Most negative curvature |
| `.max_curvature() → Option<f64>` | Largest curvature |
| `.mean_curvature() → f64` | Average curvature |
| `.bottleneck_edges() → Vec<&EdgeCurvature>` | All bottleneck edges |
| `.num_bottlenecks() → usize` | Count of bottlenecks |
| `.over_squashing_score() → f64` | Fraction of edges that are bottlenecks (0–1) |

### Module: `connection` — Connection Laplacian

| Type / Method | Description |
|---|---|
| `ConnectionLaplacian` | L_conn = D − A ∘ R for oriented sheaves |
| `ConnectionLaplacian::from_oriented_sheaf(sheaf)` | Auto-uses transpose for reverse edges |
| `ConnectionLaplacian::from_oriented_edges(sheaf, edges)` | Explicit oriented edges |
| `.apply(x) → DVector` | Compute L_conn · x |
| `.dirichlet_energy(x) → f64` | x^T L x |
| `.is_psd() → bool` | Check positive semi-definiteness |
| `.matrix`, `.total_dim`, `.num_nodes` | Fields |

### Module: `attention` — Sheaf Attention (Learned Restriction Maps)

| Type / Method | Description |
|---|---|
| `AttentionConfig` | Config: in_dim, stalk_dim, num_heads, symmetric, dropout, concat |
| `AttentionConfig::new(in_dim, stalk_dim)` | Default config |
| `SheafAttention` | Multi-head attention module for learning restriction maps |
| `SheafAttention::new(config)` | Random initialization |
| `.compute(sheaf, features) → AttentionOutput` | Learn restriction maps from features |
| `.compute_symmetric(sheaf, features) → AttentionOutput` | Force R_ij = R_ji^T |
| `AttentionOutput` | Result: sheaf (with learned maps), attention_weights, loss |

### Module: `p_laplacian` — Nonlinear Sheaf Diffusion

| Type / Method | Description |
|---|---|
| `SheafPLaplacian` | The p-Laplacian operator Δ_p |
| `SheafPLaplacian::new(sheaf, p)` | Create with exponent p (p=2 is linear) |
| `.apply(x) → DVector` | Δ_p(x): weighted by ‖diff‖^{p−2} |
| `.dirichlet_energy(x) → f64` | (1/p) Σ ‖x_v − R x_w‖^p |
| `.flow(x, dt, steps) → DVector` | Run p-Laplacian diffusion for steps |
| `.sheaf`, `.p`, `.epsilon` | Fields |

### Module: `multihop` — Multi-Hop Sheaf

| Type / Method | Description |
|---|---|
| `ComposedRestrictionMap` | A k-hop composed map: source, target, hops, map, path |
| `MultiHopSheaf` | Pre-computed k-hop restriction maps |
| `MultiHopSheaf::new(sheaf, max_hops)` | Build composed maps up to max_hops |
| `.composed_map(source, target, hops) → Option<&ComposedRestrictionMap>` | Get k-hop map |
| `.all_maps(hops) → Vec<&ComposedRestrictionMap>` | All maps at given hop distance |
| `.k_hop_laplacian(k) → SheafLaplacian` | Laplacian using k-hop composed maps |
| `.sheaf`, `.max_hops` | Fields |

### Module: `pooling` — Sheaf-Aware Graph Pooling

| Type / Method | Description |
|---|---|
| `enum MergeStrategy` | DirectSum, MaxDim, Project(dim), Average |
| `SheafPooling` | Pooling module |
| `SheafPooling::new(strategy, ratio)` | Create with merge strategy and keep ratio |
| `.pool(sheaf, clusters) → PoolingResult` | Pool nodes into supernodes |
| `PoolingResult` | coarsened_sheaf, assignment, projection, lift matrices |
| `.diffpool(sheaf, features) → PoolingResult` | Differentiable-style pooling (learned assignments) |

### Module: `plato` — PLATO Agent Communication

| Type / Method | Description |
|---|---|
| `PlatoAgent` | Agent: id, node_index, state_dim, state, role |
| `CommunicationChannel` | Channel: source, target, channel_type, restriction_map |
| `PlatoConfig` | Config: state_dim, diffusion_steps, dt, learned_maps, curvature_threshold |
| `PlatoSystem` | Full multi-agent sheaf communication system |
| `PlatoSystem::new(config)` | Create empty system |
| `.register_agent(agent)` | Add an agent |
| `.add_channel(channel)` | Add communication channel |
| `.build_sheaf() → CellularSheaf` | Construct sheaf from agents + channels |
| `.communicate() → CommunicationRoundResult` | Run one round of sheaf diffusion |
| `.diagnose() → SheafCurvature` | Detect over-squashing in agent communication |
| `.get_agent_state(id) → Option<DVector>` | Read agent state |
| `.set_agent_state(id, state)` | Update agent state |
| `CommunicationRoundResult` | updated_states, energy_before/after, num_bottlenecks, over_squashing_score |

## How It Works

The library is structured in layers:

1. **`sheaf`** — The foundation. A `CellularSheaf` is a graph where each node has a vector space (stalk) and each edge has a linear map (restriction map). The cochain space C⁰(G,F) = ⊕ F(v) is the space of node features stacked into one big vector.

2. **`laplacian`** — The sheaf Laplacian L_Σ = B^T B, where B is the coboundary (gradient) matrix. For each edge (i→j), B has a block row [-R_ij | I]. This reduces to the standard graph Laplacian when restriction maps are identity. Normalized variants (symmetric, random-walk) are provided.

3. **`diffusion`** — Neural network layers that use the sheaf Laplacian for message passing: x' = σ((I − dt·L) x W + b). Multi-layer networks stack these with learnable weight matrices and activations.

4. **`curvature`** — Diagnoses over-squashing by computing sheaf curvature κ(i,j) = 1 − ‖R_ij + R_ji‖_F/2. Negative curvature edges are bottlenecks that impede information flow.

5. **`connection`** — For oriented sheaves (where the reverse edge uses the adjoint), the connection Laplacian provides a different spectral structure useful for synchronization problems.

6. **`attention`** — Learns restriction maps from data, analogous to GAT. Uses query/key/value projections to compute attention-weighted restriction maps, allowing the sheaf to adapt to the task.

7. **`p_laplacian`** — Nonlinear diffusion: the p-Laplacian weights differences by ‖diff‖^{p−2}. p=2 is linear, p→1 gives total variation (graph cut), p>2 enhances contrast.

8. **`multihop`** — Composes restriction maps across k-hop paths, allowing information to flow across longer distances while respecting sheaf geometry. Useful for overcoming over-squashing without adding edges.

9. **`pooling`** — Coarsens the sheaf by merging nodes, producing hierarchical representations. Stalks are merged via direct sum, max dimension, projection, or averaging. Restriction maps are composed accordingly.

10. **`plato`** — Models multi-agent communication as a sheaf neural network. Each agent is a node with an internal state. Communication channels are edges with restriction maps (encoding how agents translate for each other). Sheaf diffusion = consensus. Over-squashing = communication bottleneck.

## The Math

### Cellular Sheaves
A cellular sheaf F on a graph G assigns a vector space F(v) (the *stalk*) to each node v and a linear map F_{v≺e}: F(v) → F(w) (the *restriction map*) to each oriented edge e = (v,w). The cochain space C⁰(G,F) = ⊕_v F(v) is the space of assignments of vectors to stalks.

### Coboundary Map
The coboundary δ⁰: C⁰ → C¹ maps a 0-cochain f to (δf)_e = f(v) − F_{v≺e} f(w) for each edge e = (v,w). In matrix form, this is the block matrix B where each edge contributes a block row.

### Sheaf Laplacian
L_Σ = (δ⁰)* δ⁰ = B^T B is positive semi-definite. It acts on stacked cochains: (L_Σ f)(v) = Σ_{v~w} (f(v) − F_{vw} f(w)). The kernel consists of *global sections* — assignments that are compatible across all edges.

### Dirichlet Energy
E(f) = f^T L_Σ f = Σ_{edges} ‖f(v) − F_{vw} f(w)‖² measures how far f is from being a global section. Zero energy means f is a global section.

### Spectral Gap
The second-smallest eigenvalue λ₂ of L_Σ governs the mixing rate of sheaf diffusion and the Cheeger-type isoperimetric constant. Small λ₂ means slow mixing (over-squashing).

### Sheaf Curvature
κ(i,j) = 1 − ‖F_{ij} + F_{ji}‖_F / 2 generalizes Ollivier-Ricci curvature. Negative curvature indicates bottlenecks. The sheaf can "fix" these by learning restriction maps that increase curvature.

### p-Laplacian
Δ_p f(v) = Σ_{v~w} ‖f(v) − F_{vw} f(w)‖^{p−2} (f(v) − F_{vw} f(w)). For p=2 this is the linear sheaf Laplacian. For p→1 it approaches total variation (related to graph cuts). For p>2 it amplifies large differences.

### Connection Laplacian
For an oriented sheaf where F_{ji} = F_{ij}^T, L_conn = D − A ∘ R combines degree matrix with the block adjacency weighted by restriction maps. Used in synchronization and phase retrieval.

### Sheaf Attention
R_{ij} = softmax(a^T [Wq x_i ‖ Wk x_j]) · V learns restriction maps from data, analogous to multi-head attention in transformers but producing linear maps instead of scalars.

### Multi-Hop Composition
R^{(k)}_{i→j} = R_{j_k j_{k−1}} ∘ ⋯ ∘ R_{j_1 j_0} composes restriction maps along k-hop paths, extending the receptive field while respecting sheaf geometry.

## License

MIT
