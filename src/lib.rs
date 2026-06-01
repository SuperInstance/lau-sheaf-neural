//! # lau-sheaf-neural
//!
//! Sheaf-theoretic neural networks: replacing the graph Laplacian with the
//! sheaf Laplacian to overcome over-squashing in GNNs.
//!
//! ## Core idea
//!
//! Standard GNNs assign a single feature vector to each node and pass messages
//! along edges via the graph Laplacian. When the graph has low curvature
//! "bottleneck" edges, distant nodes cannot effectively communicate — this is
//! **over-squashing**.
//!
//! A **cellular sheaf** on a graph assigns a vector space to each node (the
//! *stalk*) and a linear map to each edge (the *restriction map*). The resulting
//! **sheaf Laplacian** respects the local geometry of these maps, providing
//! richer information flow than the plain graph Laplacian.
//!
//! ## Architecture
//!
//! ```text
//! Graph + Sheaf → Sheaf Laplacian → Sheaf Diffusion → Output
//!                    ↑
//!              Restriction maps
//!              (fixed or learned)
//! ```
//!
//! ## Modules
//!
//! - [`sheaf`] — Cellular sheaf data structure
//! - [`laplacian`] — Sheaf Laplacian construction
//! - [`diffusion`] — Sheaf diffusion layers (message passing)
//! - [`curvature`] — Over-squashing diagnosis via sheaf curvature
//! - [`connection`] — Connection Laplacian for oriented sheaves
//! - [`attention`] — Sheaf attention: learn restriction maps from data
//! - [`p_laplacian`] — Nonlinear sheaf diffusion (p-Laplacian)
//! - [`multihop`] — Multi-hop sheaf: compose restriction maps
//! - [`pooling`] — Sheaf-aware graph pooling
//! - [`plato`] — PLATO agent communication as sheaf neural network

pub mod sheaf;
pub mod laplacian;
pub mod diffusion;
pub mod curvature;
pub mod connection;
pub mod attention;
pub mod p_laplacian;
pub mod multihop;
pub mod pooling;
pub mod plato;

pub use sheaf::CellularSheaf;
pub use laplacian::SheafLaplacian;
pub use diffusion::SheafDiffusion;
pub use curvature::SheafCurvature;
pub use connection::ConnectionLaplacian;
pub use attention::SheafAttention;
pub use p_laplacian::SheafPLaplacian;
pub use multihop::MultiHopSheaf;
pub use pooling::SheafPooling;
