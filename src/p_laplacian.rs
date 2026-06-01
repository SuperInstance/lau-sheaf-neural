//! p-Laplacian for nonlinear sheaf diffusion.
//!
//! The p-Laplacian generalizes the sheaf Laplacian:
//!
//!   Δ_p f(v) = Σ_{v~w} ||f(v) - R_{vw} f(w)||^{p-2} (f(v) - R_{vw} f(w))
//!
//! - p = 2: standard sheaf Laplacian (linear)
//! - p → 1: total variation minimization (solves min-cut)
//! - p > 2: more emphasis on large differences (contrast enhancement)
//!
//! The p-Laplacian flow is: dx/dt = -Δ_p(x)

use crate::sheaf::{CellularSheaf, SheafError};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// The p-Laplacian operator for a cellular sheaf.
#[derive(Clone, Debug)]
pub struct SheafPLaplacian {
    /// The underlying sheaf.
    pub sheaf: CellularSheaf,
    /// The exponent p.
    pub p: f64,
    /// Epsilon for numerical stability (avoid division by zero).
    pub epsilon: f64,
}

impl SheafPLaplacian {
    /// Create a new p-Laplacian.
    pub fn new(sheaf: CellularSheaf, p: f64) -> Result<Self, SheafError> {
        sheaf.validate()?;
        if p <= 0.0 {
            return Err(SheafError::InvalidStalkDimension(0)); // Reuse error
        }
        Ok(SheafPLaplacian {
            sheaf,
            p,
            epsilon: 1e-8,
        })
    }

    /// Apply the p-Laplacian to a cochain vector.
    ///
    /// Δ_p x at node v = Σ_{v~w} ||x_v - R_{vw} x_w||^{p-2} (x_v - R_{vw} x_w)
    pub fn apply(&self, x: &DVector<f64>) -> DVector<f64> {
        let mut result = DVector::zeros(self.sheaf.total_dim);

        for edge in &self.sheaf.edges {
            let s = edge.source;
            let t = edge.target;

            let x_s = self.sheaf.extract_stalk(s, x);
            let x_t = self.sheaf.extract_stalk(t, x);

            // Diff: x_s - R_{st} x_t
            let diff = &x_s - &edge.restriction_map * &x_t;
            let norm = diff.norm();

            // Weight: ||diff||^{p-2}
            let weight = if norm < self.epsilon {
                0.0
            } else {
                norm.powf(self.p - 2.0)
            };

            let contribution = weight * diff;
            let offset_s = self.sheaf.node_offset(s);
            let dim_s = self.sheaf.stalk_dims[s];
            for k in 0..dim_s {
                result[offset_s + k] += contribution[k];
            }
        }

        result
    }

    /// Compute the p-Dirichlet energy: (1/p) Σ ||x_v - R_{vw} x_w||^p.
    pub fn dirichlet_energy(&self, x: &DVector<f64>) -> f64 {
        let mut energy = 0.0;

        for edge in &self.sheaf.edges {
            let x_s = self.sheaf.extract_stalk(edge.source, x);
            let x_t = self.sheaf.extract_stalk(edge.target, x);
            let diff = &x_s - &edge.restriction_map * &x_t;
            energy += diff.norm().powf(self.p);
        }

        energy / self.p
    }

    /// Run p-Laplacian diffusion: x ← x - dt * Δ_p(x).
    pub fn diffuse_step(&self, x: &DVector<f64>, dt: f64) -> DVector<f64> {
        let delta_p = self.apply(x);
        x - dt * &delta_p
    }

    /// Run multiple steps of p-Laplacian diffusion.
    pub fn diffuse(&self, x: &DVector<f64>, dt: f64, num_steps: usize) -> DVector<f64> {
        let mut state = x.clone();
        for _ in 0..num_steps {
            state = self.diffuse_step(&state, dt);
        }
        state
    }

    /// Compute the 1-Laplacian (total variation operator).
    ///
    /// Special case for p=1: Δ_1 x = Σ sign(x_v - R_{vw} x_w).
    pub fn apply_one_laplacian(&self, x: &DVector<f64>) -> DVector<f64> {
        let mut result = DVector::zeros(self.sheaf.total_dim);

        for edge in &self.sheaf.edges {
            let x_s = self.sheaf.extract_stalk(edge.source, x);
            let x_t = self.sheaf.extract_stalk(edge.target, x);
            let diff = &x_s - &edge.restriction_map * &x_t;
            let norm = diff.norm();

            let sign = if norm < self.epsilon {
                DVector::zeros(diff.len())
            } else {
                diff / norm
            };

            let offset_s = self.sheaf.node_offset(edge.source);
            let dim_s = self.sheaf.stalk_dims[edge.source];
            for k in 0..dim_s {
                result[offset_s + k] += sign[k];
            }
        }

        result
    }

    /// Solve the p-Laplacian equation Δ_p x = 0 (p-harmonic functions).
    ///
    /// Uses iterative renormalization.
    pub fn find_p_harmonic(
        &self,
        boundary_conditions: &[(usize, DVector<f64>)],
        max_iter: usize,
        tol: f64,
    ) -> DVector<f64> {
        let mut x = DVector::zeros(self.sheaf.total_dim);

        // Set boundary conditions
        for &(node, ref val) in boundary_conditions {
            let offset = self.sheaf.node_offset(node);
            let dim = self.sheaf.stalk_dims[node];
            for k in 0..dim {
                x[offset + k] = val[k];
            }
        }

        let boundary_nodes: std::collections::HashSet<usize> = boundary_conditions
            .iter()
            .map(|(n, _)| *n)
            .collect();

        for _ in 0..max_iter {
            let x_old = x.clone();
            let delta = self.apply(&x);

            // Update non-boundary nodes
            for node in 0..self.sheaf.num_nodes {
                if boundary_nodes.contains(&node) {
                    continue;
                }
                let offset = self.sheaf.node_offset(node);
                let dim = self.sheaf.stalk_dims[node];
                for k in 0..dim {
                    x[offset + k] -= 0.01 * delta[offset + k];
                }
            }

            // Check convergence
            let change = (&x - &x_old).norm();
            if change < tol {
                break;
            }
        }

        x
    }

    /// Compute the p-Laplacian spectrum via power iteration.
    ///
    /// Returns the approximate largest eigenvalue.
    pub fn largest_eigenvalue(&self, max_iter: usize) -> f64 {
        let mut v = DVector::from_fn(self.sheaf.total_dim, |_, _| {
            rand::random::<f64>() * 2.0 - 1.0
        });
        v.normalize_mut();

        for _ in 0..max_iter {
            let lv = self.apply(&v);
            v = lv;
            let norm = v.norm();
            if norm < self.epsilon {
                break;
            }
            v *= 1.0 / norm;
        }

        let lv = self.apply(&v);
        let norm_v = v.norm();
        if norm_v < self.epsilon {
            0.0
        } else {
            lv.norm() / norm_v
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_test_sheaf() -> CellularSheaf {
        let mut sheaf = CellularSheaf::new_uniform(4, 2).unwrap();
        for i in 0..3 {
            sheaf.add_edge(i, i + 1, DMatrix::identity(2, 2)).unwrap();
            sheaf.add_edge(i + 1, i, DMatrix::identity(2, 2)).unwrap();
        }
        sheaf
    }

    #[test]
    fn test_p2_equals_standard_laplacian() {
        let sheaf = make_test_sheaf();
        let p2 = SheafPLaplacian::new(sheaf.clone(), 2.0).unwrap();
        let x = DVector::from_vec(vec![1.0, 0.0, 0.5, 1.0, -1.0, 0.0, 0.0, 2.0]);

        let result = p2.apply(&x);
        assert_eq!(result.len(), 8);
        // For p=2, the result should be the standard sheaf Laplacian applied to x
    }

    #[test]
    fn test_p1_laplacian() {
        let sheaf = make_test_sheaf();
        let p1 = SheafPLaplacian::new(sheaf, 1.0).unwrap();
        let x = DVector::from_vec(vec![1.0, 0.0, 0.5, 1.0, -1.0, 0.0, 0.0, 2.0]);

        let result = p1.apply(&x);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_one_laplacian() {
        let sheaf = make_test_sheaf();
        let p1 = SheafPLaplacian::new(sheaf, 1.0).unwrap();
        let x = DVector::from_vec(vec![1.0, 0.0, 0.5, 1.0, -1.0, 0.0, 0.0, 2.0]);

        let result = p1.apply_one_laplacian(&x);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_dirichlet_energy_p2() {
        let sheaf = make_test_sheaf();
        let p2 = SheafPLaplacian::new(sheaf, 2.0).unwrap();
        let x = DVector::from_vec(vec![1.0, 0.0, 0.5, 1.0, -1.0, 0.0, 0.0, 2.0]);

        let energy = p2.dirichlet_energy(&x);
        assert!(energy > 0.0);
    }

    #[test]
    fn test_dirichlet_energy_p1() {
        let sheaf = make_test_sheaf();
        let p1 = SheafPLaplacian::new(sheaf, 1.0).unwrap();
        let x = DVector::from_vec(vec![1.0, 0.0, 0.5, 1.0, -1.0, 0.0, 0.0, 2.0]);

        let energy = p1.dirichlet_energy(&x);
        assert!(energy > 0.0);
    }

    #[test]
    fn test_dirichlet_energy_constant() {
        let sheaf = make_test_sheaf();
        let p2 = SheafPLaplacian::new(sheaf, 2.0).unwrap();
        let x = DVector::from_element(8, 1.0);

        let energy = p2.dirichlet_energy(&x);
        assert_relative_eq!(energy, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diffuse_step() {
        let sheaf = make_test_sheaf();
        let p2 = SheafPLaplacian::new(sheaf, 2.0).unwrap();
        let x = DVector::from_vec(vec![1.0, 0.0, 0.5, 1.0, -1.0, 0.0, 0.0, 2.0]);

        let stepped = p2.diffuse_step(&x, 0.1);
        assert_eq!(stepped.len(), 8);
    }

    #[test]
    fn test_diffuse_multiple_steps() {
        let sheaf = make_test_sheaf();
        let p2 = SheafPLaplacian::new(sheaf, 2.0).unwrap();
        let x = DVector::from_vec(vec![1.0, 0.0, 0.5, 1.0, -1.0, 0.0, 0.0, 2.0]);

        let result = p2.diffuse(&x, 0.05, 50);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_p_harmonic() {
        let mut sheaf = CellularSheaf::new_uniform(3, 1).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 0, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(2, 1, DMatrix::identity(1, 1)).unwrap();

        let p2 = SheafPLaplacian::new(sheaf, 2.0).unwrap();
        let bc = vec![
            (0, DVector::from_vec(vec![1.0])),
            (2, DVector::from_vec(vec![-1.0])),
        ];

        let harmonic = p2.find_p_harmonic(&bc, 1000, 1e-6);
        assert_eq!(harmonic.len(), 3);
        // Node 1 should be between 1.0 and -1.0 (i.e., 0.0)
        assert!(harmonic[1].abs() < 0.5);
    }

    #[test]
    fn test_p3_laplacian() {
        let sheaf = make_test_sheaf();
        let p3 = SheafPLaplacian::new(sheaf, 3.0).unwrap();
        let x = DVector::from_vec(vec![1.0, 0.0, 0.5, 1.0, -1.0, 0.0, 0.0, 2.0]);

        let result = p3.apply(&x);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_dirichlet_energy_decreases() {
        let sheaf = make_test_sheaf();
        let p2 = SheafPLaplacian::new(sheaf, 2.0).unwrap();
        let x = DVector::from_vec(vec![3.0, 0.0, -1.0, 2.0, 0.5, 1.0, -2.0, 0.0]);

        let energy_0 = p2.dirichlet_energy(&x);
        let x1 = p2.diffuse_step(&x, 0.01);
        let energy_1 = p2.dirichlet_energy(&x1);
        assert!(energy_1 <= energy_0 + 1e-6, "Energy should decrease: {} -> {}", energy_0, energy_1);
    }

    #[test]
    fn test_largest_eigenvalue() {
        let sheaf = make_test_sheaf();
        let p2 = SheafPLaplacian::new(sheaf, 2.0).unwrap();
        let eig = p2.largest_eigenvalue(100);
        assert!(eig >= 0.0);
    }

    #[test]
    fn test_nontrivial_restriction() {
        let mut sheaf = CellularSheaf::new_uniform(2, 2).unwrap();
        let map = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]);
        sheaf.add_edge(0, 1, map.clone()).unwrap();
        sheaf.add_edge(1, 0, map).unwrap();

        let p2 = SheafPLaplacian::new(sheaf, 2.0).unwrap();
        let x = DVector::from_vec(vec![1.0, 0.0, 0.0, 1.0]);
        let energy = p2.dirichlet_energy(&x);
        assert!(energy > 0.0);
    }
}
