//! Sheaf Laplacian construction.
//!
//! The sheaf Laplacian generalizes the graph Laplacian:
//!
//!   L_Σ = Σ_{i~j} (x_i - Σ_{ji} x_j)²
//!
//! In matrix form: L_Σ = B^T B where B is the coboundary map.
//! For the trivial sheaf (identity restriction maps), this reduces to the
//! standard graph Laplacian.

use crate::sheaf::{CellularSheaf, SheafError};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// The sheaf Laplacian operator.
///
/// L_Σ is a (total_dim × total_dim) positive semi-definite matrix that acts
/// on the stacked cochain vector. It captures the sheaf geometry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SheafLaplacian {
    /// The Laplacian matrix: L = B^T B.
    pub matrix: DMatrix<f64>,
    /// Total feature dimension.
    pub total_dim: usize,
    /// Number of nodes.
    pub num_nodes: usize,
}

impl SheafLaplacian {
    /// Construct the sheaf Laplacian from a cellular sheaf.
    ///
    /// L_Σ = B^T B where B is the coboundary map.
    pub fn from_sheaf(sheaf: &CellularSheaf) -> Result<Self, SheafError> {
        sheaf.validate()?;
        let b = sheaf.coboundary_matrix();
        let l = &b.transpose() * &b;
        Ok(SheafLaplacian {
            matrix: l,
            total_dim: sheaf.total_dim,
            num_nodes: sheaf.num_nodes,
        })
    }

    /// Construct the normalized sheaf Laplacian: D^{-1/2} L D^{-1/2}.
    ///
    /// D is the block-diagonal degree matrix (each block is degree × I_{stalk_dim}).
    pub fn normalized(sheaf: &CellularSheaf) -> Result<Self, SheafError> {
        sheaf.validate()?;
        let b = sheaf.coboundary_matrix();
        let l_raw = &b.transpose() * &b;

        // Build degree normalization: for each node, count edges touching it
        let mut degrees = vec![0usize; sheaf.num_nodes];
        for edge in &sheaf.edges {
            degrees[edge.source] += 1;
            degrees[edge.target] += 1;
        }

        // D^{-1/2} as a block diagonal
        let mut d_inv_sqrt = DMatrix::zeros(sheaf.total_dim, sheaf.total_dim);
        for i in 0..sheaf.num_nodes {
            let offset = sheaf.node_offset(i);
            let d = sheaf.stalk_dims[i];
            let val = if degrees[i] > 0 {
                1.0 / (degrees[i] as f64).sqrt()
            } else {
                0.0
            };
            for k in 0..d {
                d_inv_sqrt[(offset + k, offset + k)] = val;
            }
        }

        let l_norm = &d_inv_sqrt * &l_raw * &d_inv_sqrt;
        Ok(SheafLaplacian {
            matrix: l_norm,
            total_dim: sheaf.total_dim,
            num_nodes: sheaf.num_nodes,
        })
    }

    /// Construct the random-walk normalized Laplacian: D^{-1} L.
    pub fn random_walk_normalized(sheaf: &CellularSheaf) -> Result<Self, SheafError> {
        sheaf.validate()?;
        let b = sheaf.coboundary_matrix();
        let l_raw = &b.transpose() * &b;

        let mut degrees = vec![0usize; sheaf.num_nodes];
        for edge in &sheaf.edges {
            degrees[edge.source] += 1;
            degrees[edge.target] += 1;
        }

        let mut d_inv = DMatrix::zeros(sheaf.total_dim, sheaf.total_dim);
        for i in 0..sheaf.num_nodes {
            let offset = sheaf.node_offset(i);
            let d = sheaf.stalk_dims[i];
            let val = if degrees[i] > 0 {
                1.0 / degrees[i] as f64
            } else {
                0.0
            };
            for k in 0..d {
                d_inv[(offset + k, offset + k)] = val;
            }
        }

        let l_rw = &d_inv * &l_raw;
        Ok(SheafLaplacian {
            matrix: l_rw,
            total_dim: sheaf.total_dim,
            num_nodes: sheaf.num_nodes,
        })
    }

    /// Apply the Laplacian to a cochain vector: L·x.
    pub fn apply(&self, x: &DVector<f64>) -> DVector<f64> {
        &self.matrix * x
    }

    /// Compute the Dirichlet energy: x^T L x.
    ///
    /// This measures how "non-harmonic" the signal x is with respect to the sheaf.
    pub fn dirichlet_energy(&self, x: &DVector<f64>) -> f64 {
        let lx = self.apply(x);
        x.dot(&lx)
    }

    /// Check positive semi-definiteness: all eigenvalues ≥ 0.
    pub fn is_psd(&self) -> bool {
        let eigenvalues = self.matrix.symmetric_eigenvalues();
        eigenvalues.iter().all(|&v| v >= -1e-10)
    }

    /// Compute the smallest eigenvalue (should be ≥ 0 for PSD).
    pub fn smallest_eigenvalue(&self) -> f64 {
        let eigenvalues = self.matrix.symmetric_eigenvalues();
        eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Compute the spectral gap (second-smallest eigenvalue).
    ///
    /// The spectral gap governs the mixing rate of sheaf diffusion.
    pub fn spectral_gap(&self) -> f64 {
        let mut eigenvalues: Vec<f64> = self.matrix.symmetric_eigenvalues().iter().copied().collect();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if eigenvalues.len() < 2 {
            0.0
        } else {
            eigenvalues[1]
        }
    }

    /// Compute the algebraic connectivity (same as spectral gap).
    pub fn algebraic_connectivity(&self) -> f64 {
        self.spectral_gap()
    }

    /// Get the trace of the Laplacian (sum of diagonal).
    pub fn trace(&self) -> f64 {
        self.matrix.trace()
    }

    /// Number of zero eigenvalues (multiplicity of the kernel).
    pub fn kernel_dimension(&self) -> usize {
        let eigenvalues = self.matrix.symmetric_eigenvalues();
        eigenvalues.iter().filter(|&&v| v.abs() < 1e-10).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_line_sheaf(n: usize, d: usize) -> CellularSheaf {
        let mut sheaf = CellularSheaf::new_uniform(n, d).unwrap();
        for i in 0..n - 1 {
            sheaf
                .add_edge(i, i + 1, DMatrix::identity(d, d))
                .unwrap();
            sheaf
                .add_edge(i + 1, i, DMatrix::identity(d, d))
                .unwrap();
        }
        sheaf
    }

    #[test]
    fn test_sheaf_laplacian_basic() {
        let sheaf = make_line_sheaf(3, 2);
        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
        assert_eq!(lap.total_dim, 6);
        assert_eq!(lap.matrix.nrows(), 6);
        assert_eq!(lap.matrix.ncols(), 6);
    }

    #[test]
    fn test_psd_property() {
        let sheaf = make_line_sheaf(5, 2);
        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
        assert!(lap.is_psd());
    }

    #[test]
    fn test_dirichlet_energy_constant_signal() {
        let sheaf = make_line_sheaf(3, 2);
        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
        // Constant signal should have zero Dirichlet energy for trivial sheaf
        let x = DVector::from_element(6, 1.0);
        let energy = lap.dirichlet_energy(&x);
        assert!(energy.abs() < 1e-10, "Energy should be ~0, got {}", energy);
    }

    #[test]
    fn test_dirichlet_energy_nonconstant() {
        let sheaf = make_line_sheaf(3, 1);
        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
        let x = DVector::from_vec(vec![1.0, 0.0, -1.0]);
        let energy = lap.dirichlet_energy(&x);
        assert!(energy > 0.0);
    }

    #[test]
    fn test_spectral_gap_line_graph() {
        // Line graph on 4 nodes should have small spectral gap
        let sheaf = make_line_sheaf(4, 1);
        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
        let gap = lap.spectral_gap();
        assert!(gap > 0.0);
        assert!(gap < 2.0); // Line graph has small spectral gap
    }

    #[test]
    fn test_trivial_sheaf_equals_graph_laplacian() {
        // For trivial sheaf (identity maps) on a simple graph, verify
        // the sheaf Laplacian matches the graph Laplacian
        let mut sheaf = CellularSheaf::new_uniform(3, 1).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 0, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(2, 1, DMatrix::identity(1, 1)).unwrap();

        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
        // With bidirectional edges, the Laplacian is 2x the graph Laplacian:
        // [[2, -2, 0], [-2, 4, -2], [0, -2, 2]]
        assert_relative_eq!(lap.matrix[(0, 0)], 2.0, epsilon = 1e-10);
        assert_relative_eq!(lap.matrix[(0, 1)], -2.0, epsilon = 1e-10);
        assert_relative_eq!(lap.matrix[(1, 0)], -2.0, epsilon = 1e-10);
        assert_relative_eq!(lap.matrix[(1, 1)], 4.0, epsilon = 1e-10);
        assert_relative_eq!(lap.matrix[(1, 2)], -2.0, epsilon = 1e-10);
        assert_relative_eq!(lap.matrix[(2, 1)], -2.0, epsilon = 1e-10);
        assert_relative_eq!(lap.matrix[(2, 2)], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nontrivial_restriction_map() {
        let mut sheaf = CellularSheaf::new_uniform(2, 2).unwrap();
        let map = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        sheaf.add_edge(0, 1, map.clone()).unwrap();
        sheaf.add_edge(1, 0, map).unwrap();

        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
        // With scaling by 2, the Laplacian should have larger eigenvalues
        assert!(lap.is_psd());
        let energy_nonconstant = lap.dirichlet_energy(&DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]));
        assert!(energy_nonconstant > 0.0);
    }

    #[test]
    fn test_normalized_laplacian() {
        let mut sheaf = CellularSheaf::new_uniform(3, 1).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 0, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(2, 1, DMatrix::identity(1, 1)).unwrap();

        let lap = SheafLaplacian::normalized(&sheaf).unwrap();
        // Normalized Laplacian diagonal should be ≤ 2
        assert!(lap.matrix[(0, 0)] <= 2.0 + 1e-10);
        assert!(lap.is_psd());
    }

    #[test]
    fn test_random_walk_normalized() {
        let mut sheaf = CellularSheaf::new_uniform(3, 1).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 0, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(2, 1, DMatrix::identity(1, 1)).unwrap();

        let lap = SheafLaplacian::random_walk_normalized(&sheaf).unwrap();
        // Random walk Laplacian is not symmetric, so symmetric_eigenvalues may not be real
        // Just verify the matrix was constructed
        assert_eq!(lap.total_dim, 3);
    }

    #[test]
    fn test_kernel_dimension() {
        // Connected graph with trivial sheaf: kernel is 1-dimensional (constant signal)
        let mut sheaf = CellularSheaf::new_uniform(3, 1).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 0, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(2, 1, DMatrix::identity(1, 1)).unwrap();

        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
        // For 1D stalks on connected graph: kernel dimension = 1
        assert_eq!(lap.kernel_dimension(), 1);
    }

    #[test]
    fn test_trace() {
        let sheaf = make_line_sheaf(3, 1);
        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
        // For path graph 0-1-2 with bidirectional edges: trace = 8 (2x due to bidirectional)
        assert_relative_eq!(lap.trace(), 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_apply() {
        let sheaf = make_line_sheaf(3, 1);
        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
        let x = DVector::from_vec(vec![1.0, 0.0, -1.0]);
        let lx = lap.apply(&x);
        assert_eq!(lx.len(), 3);
    }
}
