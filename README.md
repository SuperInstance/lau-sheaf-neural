# lau-sheaf-neural

**Sheaf-theoretic neural networks** — replacing the graph Laplacian with the sheaf Laplacian to overcome over-squashing in graph neural networks.

## Why?

Standard Graph Neural Networks (GNNs) suffer from **over-squashing**: information bottleneck on edges with low curvature prevents distant nodes from communicating effectively. Sheaf Neural Networks solve this by assigning vector spaces to nodes and linear maps to edges, so the sheaf Laplacian respects local geometry.

## Architecture

```
Graph + Sheaf → Sheaf Laplacian → Sheaf Diffusion → Output
                    ↑
              Restriction maps
              (fixed or learned)
```

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Cellular Sheaf** | Vector space per node (stalk), linear map per edge (restriction map) |
| **Sheaf Laplacian** | L_Σ = B^T B — generalizes graph Laplacian |
| **Sheaf Diffusion** | Message passing via sheaf Laplacian flow: dx/dt = -σ(L_Σ x + b) |
| **Over-squashing** | Diagnosed via sheaf curvature; negative curvature = bottleneck |
| **Connection Laplacian** | For oriented sheaves (links to lau-connection-matrix) |
| **Sheaf Attention** | Learn restriction maps from data (GAT + sheaf structure) |
| **p-Laplacian** | Nonlinear sheaf diffusion (p=2 standard, p→1 total variation) |
| **Multi-hop Sheaf** | Compose restriction maps across k-hop neighborhoods |
| **Sheaf Pooling** | Coarsen the sheaf (not just the graph) for hierarchical classification |
| **PLATO** | Agent communication modeled as sheaf neural network |

## Quick Start

```rust
use lau_sheaf_neural::{CellularSheaf, SheafLaplacian, SheafDiffusion, Activation};

// Build a cellular sheaf on a 5-node path graph
let mut sheaf = CellularSheaf::new_uniform(5, 3).unwrap();
for i in 0..4 {
    sheaf.add_edge(i, i + 1, nalgebra::DMatrix::identity(3, 3)).unwrap();
    sheaf.add_edge(i + 1, i, nalgebra::DMatrix::identity(3, 3)).unwrap();
}

// Compute the sheaf Laplacian
let laplacian = SheafLaplacian::from_sheaf(&sheaf).unwrap();
println!("Spectral gap: {}", laplacian.spectral_gap());

// Run sheaf diffusion
let diffusion = SheafDiffusion::new(sheaf, &[16, 8], Activation::ReLU).unwrap();
let features = nalgebra::DVector::from_element(15, 1.0);
let output = diffusion.forward(&features);
```

## Modules

- `sheaf` — Cellular sheaf data structure
- `laplacian` — Sheaf Laplacian (standard, normalized, random-walk)
- `diffusion` — Sheaf diffusion layers and networks
- `curvature` — Over-squashing diagnosis via sheaf curvature
- `connection` — Connection Laplacian for oriented sheaves
- `attention` — Sheaf attention (learnable restriction maps)
- `p_laplacian` — Nonlinear p-Laplacian diffusion
- `multihop` — Multi-hop sheaf with composed restriction maps
- `pooling` — Sheaf-aware graph pooling
- `plato` — PLATO agent communication framework

## Testing

```bash
cargo test  # 127 tests
```

## License

MIT
