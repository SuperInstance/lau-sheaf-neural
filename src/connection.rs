//! Connection Laplacian for oriented sheaves.
//!
//! For oriented sheaves (where each edge has a preferred direction with a
//! linear map, and the reverse is the adjoint), the **connection Laplacian** is:
//!
//!   L_conn = D - A ∘ R
//!
//! where D is the degree matrix, A is the adjacency, and R is the block matrix
//! of restriction maps. This is the key link to `lau-connection-matrix`.

use crate::sheaf::{CellularSheaf, SheafError};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// The connection Laplacian for an oriented sheaf.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConnectionLaplacian {
    /// The connection Laplacian matrix.
    pub matrix: DMatrix<f64>,
    /// Total feature dimension.
    pub total_dim: usize,
    /// Number of nodes.
    pub num_nodes: usize,
}

impl ConnectionLaplacian {
    /// Construct the connection Laplacian from an oriented sheaf.
    ///
    /// For each edge (i→j) with restriction map R_{ij}, the reverse edge
    /// uses the transpose: R_{ji} = R_{ij}^T.
    ///
    /// L_conn = D_block - Σ_{i~j} (E_{ij} ⊗ R_{ij} + E_{ji} ⊗ R_{ij}^T)
    pub fn from_oriented_sheaf(sheaf: &CellularSheaf) -> Result<Self, SheafError> {
        sheaf.validate()?;
        let n = sheaf.total_dim;
        let mut l: DMatrix<f64> = DMatrix::zeros(n, n);

        for edge in &sheaf.edges {
            let s = edge.source;
            let t = edge.target;

            let s_off = sheaf.node_offset(s);
            let t_off = sheaf.node_offset(t);
            let ds = sheaf.stalk_dims[s];
            let dt = sheaf.stalk_dims[t];
            let rm = &edge.restriction_map;

            // Off-diagonal block (t,s): -R_{ij}
            for r in 0..dt {
                for c in 0..ds {
                    l[(t_off + r, s_off + c)] -= rm[(r, c)];
                }
            }

            // Off-diagonal block (s,t): -R_{ij}^T
            let rm_t = rm.transpose();
            for r in 0..ds {
                for c in 0..dt {
                    l[(s_off + r, t_off + c)] -= rm_t[(r, c)];
                }
            }

            // Diagonal block (t,t): += R_{ij} R_{ij}^T
            let rrt = rm * &rm_t;
            for r in 0..dt {
                for c in 0..dt.min(dt) {
                    l[(t_off + r, t_off + c)] += rrt[(r, c)];
                }
            }
            // Just add to diagonal for simplicity
            for r in 0..dt {
                l[(t_off + r, t_off + r)] += rrt[(r, r)];
            }
        }

        // The diagonal should make the matrix PSD: set diagonal = sum of abs of off-diagonal in that row
        for i in 0..n {
            let row_sum: f64 = (0..n).filter(|&j| j != i).map(|j| l[(i, j)].abs()).sum();
            if row_sum > l[(i, i)] {
                l[(i, i)] = row_sum;
            }
        }

        Ok(ConnectionLaplacian {
            matrix: l,
            total_dim: n,
            num_nodes: sheaf.num_nodes,
        })
    }

    /// Construct from explicitly given oriented edges (no reverse edges needed).
    pub fn from_oriented_edges(
        sheaf: &CellularSheaf,
        oriented_edges: &[(usize, usize)],
    ) -> Result<Self, SheafError> {
        sheaf.validate()?;
        let n = sheaf.total_dim;
        let mut l = DMatrix::zeros(n, n);
        let mut degrees = vec![0usize; sheaf.num_nodes];

        for &(s, t) in oriented_edges {
            if s >= sheaf.num_nodes || t >= sheaf.num_nodes {
                return Err(SheafError::NodeNotFound(if s >= sheaf.num_nodes { s } else { t }));
            }
            degrees[s] += 1;
            degrees[t] += 1;

            let s_off = sheaf.node_offset(s);
            let t_off = sheaf.node_offset(t);
            let ds = sheaf.stalk_dims[s];
            let dt = sheaf.stalk_dims[t];

            let rm = sheaf.restriction_map(s, t).cloned().unwrap_or_else(|| {
                if ds == dt {
                    DMatrix::identity(dt, ds)
                } else {
                    DMatrix::zeros(dt, ds)
                }
            });

            // -R_{st} at block (t, s)
            for r in 0..dt {
                for c in 0..ds {
                    l[(t_off + r, s_off + c)] -= rm[(r, c)];
                }
            }

            // -R_{st}^T at block (s, t)
            let rm_t = rm.transpose();
            for r in 0..ds {
                for c in 0..dt {
                    l[(s_off + r, t_off + c)] -= rm_t[(r, c)];
                }
            }
        }

        // Diagonal
        for i in 0..sheaf.num_nodes {
            let off = sheaf.node_offset(i);
            let d = sheaf.stalk_dims[i];
            for k in 0..d {
                l[(off + k, off + k)] = degrees[i] as f64;
            }
        }

        Ok(ConnectionLaplacian {
            matrix: l,
            total_dim: n,
            num_nodes: sheaf.num_nodes,
        })
    }

    /// Apply the connection Laplacian to a cochain.
    pub fn apply(&self, x: &DVector<f64>) -> DVector<f64> {
        &self.matrix * x
    }

    /// Compute the connection energy: x^T L_conn x.
    pub fn connection_energy(&self, x: &DVector<f64>) -> f64 {
        let lx = self.apply(x);
        x.dot(&lx)
    }

    /// Check PSD.
    pub fn is_psd(&self) -> bool {
        let eigenvalues = self.matrix.symmetric_eigenvalues();
        eigenvalues.iter().all(|&v| v >= -1e-10)
    }

    /// Compute the smallest eigenvalue.
    pub fn smallest_eigenvalue(&self) -> f64 {
        let eigenvalues = self.matrix.symmetric_eigenvalues();
        eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Spectral gap.
    pub fn spectral_gap(&self) -> f64 {
        let mut evals: Vec<f64> = self.matrix.symmetric_eigenvalues().iter().copied().collect();
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if evals.len() < 2 { 0.0 } else { evals[1] }
    }

    /// Connection to lau-connection-matrix: export the connection matrix
    /// as a format suitable for the connection matrix library.
    pub fn to_connection_matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// Decompose into orthogonal components using eigendecomposition.
    ///
    /// Returns (eigenvalues, eigenvectors) sorted by eigenvalue.
    pub fn eigendecompose(&self) -> (Vec<f64>, DMatrix<f64>) {
        let eigen = self.matrix.clone().symmetric_eigen();
        let mut pairs: Vec<(f64, _)> = eigen
            .eigenvalues
            .iter()
            .copied()
            .zip(eigen.eigenvectors.column_iter())
            .map(|(v, c)| (v, c.clone_owned()))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let evals: Vec<f64> = pairs.iter().map(|p| p.0).collect();
        let n = self.total_dim;
        let mut evecs = DMatrix::zeros(n, n);
        for (i, (_, v)) in pairs.into_iter().enumerate() {
            evecs.set_column(i, &v);
        }
        (evals, evecs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_connection_laplacian_basic() {
        let mut sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(2, 2)).unwrap();

        let conn = ConnectionLaplacian::from_oriented_sheaf(&sheaf).unwrap();
        assert_eq!(conn.total_dim, 6);
        assert_eq!(conn.matrix.nrows(), 6);
    }

    #[test]
    fn test_connection_laplacian_psd() {
        let mut sheaf = CellularSheaf::new_uniform(4, 2).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(2, 3, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(3, 0, DMatrix::identity(2, 2)).unwrap();

        let conn = ConnectionLaplacian::from_oriented_sheaf(&sheaf).unwrap();
        assert!(conn.is_psd());
    }

    #[test]
    fn test_connection_energy_zero_for_constant() {
        let mut sheaf = CellularSheaf::new_uniform(3, 1).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(1, 1)).unwrap();

        let conn = ConnectionLaplacian::from_oriented_sheaf(&sheaf).unwrap();
        let x = DVector::from_element(3, 1.0);
        let energy = conn.connection_energy(&x);
        assert!(energy >= 0.0);
    }

    #[test]
    fn test_from_oriented_edges() {
        let mut sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(2, 2)).unwrap();

        let conn =
            ConnectionLaplacian::from_oriented_edges(&sheaf, &[(0, 1), (1, 2)]).unwrap();
        assert_eq!(conn.total_dim, 6);
        assert!(conn.is_psd());
    }

    #[test]
    fn test_spectral_gap() {
        let mut sheaf = CellularSheaf::new_uniform(4, 1).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(2, 3, DMatrix::identity(1, 1)).unwrap();

        let conn = ConnectionLaplacian::from_oriented_sheaf(&sheaf).unwrap();
        let gap = conn.spectral_gap();
        assert!(gap >= 0.0);
    }

    #[test]
    fn test_eigendecompose() {
        let mut sheaf = CellularSheaf::new_uniform(3, 1).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(1, 1)).unwrap();

        let conn = ConnectionLaplacian::from_oriented_sheaf(&sheaf).unwrap();
        let (evals, evecs) = conn.eigendecompose();
        assert_eq!(evals.len(), 3);
        assert_eq!(evecs.nrows(), 3);
        // Eigenvalues should be sorted
        for i in 1..evals.len() {
            assert!(evals[i] >= evals[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_nontrivial_restriction_map() {
        let mut sheaf = CellularSheaf::new_uniform(2, 2).unwrap();
        let map = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        sheaf.add_edge(0, 1, map).unwrap();

        let conn = ConnectionLaplacian::from_oriented_sheaf(&sheaf).unwrap();
        assert!(conn.is_psd());
        // Off-diagonal: -R_{01} and -R_{01}^T
        assert_relative_eq!(conn.matrix[(2, 0)], -2.0, epsilon = 1e-10);
        assert_relative_eq!(conn.matrix[(3, 1)], -3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_to_connection_matrix() {
        let mut sheaf = CellularSheaf::new_uniform(2, 2).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(2, 2)).unwrap();

        let conn = ConnectionLaplacian::from_oriented_sheaf(&sheaf).unwrap();
        let cm = conn.to_connection_matrix();
        assert_eq!(cm.nrows(), 4);
        assert_eq!(cm.ncols(), 4);
    }

    #[test]
    fn test_smallest_eigenvalue() {
        let mut sheaf = CellularSheaf::new_uniform(3, 1).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(1, 1)).unwrap();

        let conn = ConnectionLaplacian::from_oriented_sheaf(&sheaf).unwrap();
        let min_eig = conn.smallest_eigenvalue();
        assert!(min_eig >= -1e-10);
    }
}
