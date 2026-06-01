//! Sheaf diffusion: message passing via sheaf Laplacian flow.
//!
//! Sheaf diffusion replaces the standard GNN message passing with:
//!
//!   dx/dt = -σ(L_Σ x + b)
//!
//! where L_Σ is the sheaf Laplacian and σ is a nonlinearity.
//! This is the continuous sheaf neural network (SheafNet).

use crate::laplacian::SheafLaplacian;
use crate::sheaf::{CellularSheaf, SheafError};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Activation functions for sheaf diffusion layers.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Activation {
    Identity,
    ReLU,
    Sigmoid,
    Tanh,
    ELU(f64),
    LeakyReLU(f64),
}

impl Activation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Identity => x,
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::ELU(alpha) => {
                if x >= 0.0 {
                    x
                } else {
                    alpha * (x.exp() - 1.0)
                }
            }
            Activation::LeakyReLU(alpha) => {
                if x >= 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
        }
    }

    pub fn apply_vec(&self, v: &DVector<f64>) -> DVector<f64> {
        v.map(|x| self.apply(x))
    }
}

/// A single sheaf diffusion layer.
///
/// Implements: x' = σ(L_Σ x W + b)
/// where W is a learnable weight matrix and b is bias.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SheafDiffusionLayer {
    /// Weight matrix: (total_dim × out_dim).
    pub weight: DMatrix<f64>,
    /// Bias vector: (out_dim).
    pub bias: DVector<f64>,
    /// Activation function.
    pub activation: Activation,
    /// Diffusion time step.
    pub dt: f64,
}

impl SheafDiffusionLayer {
    /// Create a new diffusion layer with random weights.
    pub fn new(in_dim: usize, out_dim: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let weight = DMatrix::from_fn(in_dim, out_dim, |_, _| {
            rng.gen::<f64>() * 0.1 - 0.05
        });
        let bias = DVector::from_fn(out_dim, |i, _| rng.gen::<f64>() * 0.01);

        SheafDiffusionLayer {
            weight,
            bias,
            activation,
            dt: 0.1,
        }
    }

    /// Create with specific weights.
    pub fn with_weights(
        weight: DMatrix<f64>,
        bias: DVector<f64>,
        activation: Activation,
    ) -> Self {
        SheafDiffusionLayer {
            weight,
            bias,
            activation,
            dt: 0.1,
        }
    }

    /// Forward pass: x' = σ((I - dt·L) x W + b)
    pub fn forward(&self, x: &DVector<f64>, laplacian: &SheafLaplacian) -> DVector<f64> {
        let n = laplacian.total_dim;
        let identity = DMatrix::identity(n, n);
        let diffused = &identity - self.dt * &laplacian.matrix;
        let propagated: DVector<f64> = &diffused * x;
        let projected = &self.weight.tr_mul(&propagated) + &self.bias;
        self.activation.apply_vec(&projected)
    }

    /// Forward pass without Laplacian (just linear transform).
    pub fn forward_linear(&self, x: &DVector<f64>) -> DVector<f64> {
        let projected = self.weight.tr_mul(x) + &self.bias;
        self.activation.apply_vec(&projected)
    }
}

/// Full sheaf diffusion network (multi-layer).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SheafDiffusion {
    /// The underlying sheaf.
    pub sheaf: CellularSheaf,
    /// The precomputed Laplacian.
    pub laplacian: SheafLaplacian,
    /// Diffusion layers.
    pub layers: Vec<SheafDiffusionLayer>,
}

impl SheafDiffusion {
    /// Create a new sheaf diffusion network.
    pub fn new(
        sheaf: CellularSheaf,
        hidden_dims: &[usize],
        activation: Activation,
    ) -> Result<Self, SheafError> {
        sheaf.validate()?;
        let laplacian = SheafLaplacian::from_sheaf(&sheaf)?;

        let mut layers = Vec::new();
        let mut in_dim = sheaf.total_dim;
        for &out_dim in hidden_dims {
            layers.push(SheafDiffusionLayer::new(in_dim, out_dim, activation.clone()));
            in_dim = out_dim;
        }

        Ok(SheafDiffusion {
            sheaf,
            laplacian,
            layers,
        })
    }

    /// Forward pass through all layers.
    pub fn forward(&self, x: &DVector<f64>) -> DVector<f64> {
        let mut h = x.clone();
        for layer in &self.layers {
            h = layer.forward(&h, &self.laplacian);
        }
        h
    }

    /// Run continuous sheaf diffusion for `num_steps` steps.
    ///
    /// Simulates: dx/dt = -L_Σ x (heat equation on the sheaf).
    pub fn diffuse(&self, x: &DVector<f64>, dt: f64, num_steps: usize) -> DVector<f64> {
        let n = self.laplacian.total_dim;
        let identity = DMatrix::identity(n, n);
        let step_matrix = &identity - dt * &self.laplacian.matrix;
        let mut state = x.clone();
        for _ in 0..num_steps {
            state = &step_matrix * &state;
        }
        state
    }

    /// Explicit Euler step of sheaf diffusion.
    pub fn euler_step(&self, x: &DVector<f64>, dt: f64) -> DVector<f64> {
        let lx = self.laplacian.apply(x);
        x - dt * lx
    }

    /// Midpoint (RK2) step for more accurate diffusion.
    pub fn midpoint_step(&self, x: &DVector<f64>, dt: f64) -> DVector<f64> {
        let k1 = self.laplacian.apply(x);
        let x_mid = x - (dt / 2.0) * &k1;
        let k2 = self.laplacian.apply(&x_mid);
        x - dt * &k2
    }

    /// Compute the diffusion matrix for `t` time: exp(-t·L).
    pub fn diffusion_matrix(&self, t: f64) -> DMatrix<f64> {
        // Approximate exp(-tL) via eigendecomposition
        let eigen = self.laplacian.matrix.clone().symmetric_eigen();
        let exp_diag = DMatrix::from_diagonal(
            &DVector::from_fn(eigen.eigenvalues.len(), |i, _| (-t * eigen.eigenvalues[i]).exp()),
        );
        &eigen.eigenvectors * exp_diag * &eigen.eigenvectors.transpose()
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_simple_diffusion() -> SheafDiffusion {
        let mut sheaf = CellularSheaf::new_uniform(4, 2).unwrap();
        for i in 0..3 {
            sheaf.add_edge(i, i + 1, DMatrix::identity(2, 2)).unwrap();
            sheaf.add_edge(i + 1, i, DMatrix::identity(2, 2)).unwrap();
        }
        SheafDiffusion::new(sheaf, &[8, 4], Activation::ReLU).unwrap()
    }

    #[test]
    fn test_activation_relu() {
        let act = Activation::ReLU;
        assert_eq!(act.apply(2.0), 2.0);
        assert_eq!(act.apply(-1.0), 0.0);
    }

    #[test]
    fn test_activation_sigmoid() {
        let act = Activation::Sigmoid;
        let val = act.apply(0.0);
        assert_relative_eq!(val, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_activation_tanh() {
        let act = Activation::Tanh;
        let val = act.apply(0.0);
        assert_relative_eq!(val, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_activation_elu() {
        let act = Activation::ELU(1.0);
        assert_eq!(act.apply(2.0), 2.0);
        assert!(act.apply(-1.0) < 0.0);
    }

    #[test]
    fn test_activation_leaky_relu() {
        let act = Activation::LeakyReLU(0.01);
        assert_eq!(act.apply(2.0), 2.0);
        assert_relative_eq!(act.apply(-2.0), -0.02, epsilon = 1e-10);
    }

    #[test]
    fn test_diffusion_layer_forward() {
        let mut sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(1, 0, DMatrix::identity(2, 2)).unwrap();
        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();

        let layer = SheafDiffusionLayer::new(6, 4, Activation::ReLU);
        let x = DVector::from_element(6, 1.0);
        let out = layer.forward(&x, &lap);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_diffusion_network_creation() {
        let net = make_simple_diffusion();
        assert_eq!(net.num_layers(), 2);
    }

    #[test]
    fn test_diffusion_forward() {
        let net = make_simple_diffusion();
        let x = DVector::from_element(8, 0.5);
        let out = net.forward(&x);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_continuous_diffusion_converges() {
        let mut sheaf = CellularSheaf::new_uniform(3, 1).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 0, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(2, 1, DMatrix::identity(1, 1)).unwrap();
        let net = SheafDiffusion::new(sheaf, &[], Activation::Identity).unwrap();

        let x = DVector::from_vec(vec![3.0, 0.0, -1.0]);
        let diffused = net.diffuse(&x, 0.1, 100);
        // Should converge towards the mean (2/3)
        let mean = diffused.mean();
        assert_relative_eq!(mean, 2.0 / 3.0, epsilon = 0.2);
    }

    #[test]
    fn test_euler_step() {
        let mut sheaf = CellularSheaf::new_uniform(2, 1).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 0, DMatrix::identity(1, 1)).unwrap();
        let net = SheafDiffusion::new(sheaf, &[], Activation::Identity).unwrap();

        let x = DVector::from_vec(vec![1.0, 0.0]);
        let stepped = net.euler_step(&x, 0.1);
        assert_eq!(stepped.len(), 2);
    }

    #[test]
    fn test_midpoint_step() {
        let mut sheaf = CellularSheaf::new_uniform(2, 1).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 0, DMatrix::identity(1, 1)).unwrap();
        let net = SheafDiffusion::new(sheaf, &[], Activation::Identity).unwrap();

        let x = DVector::from_vec(vec![1.0, 0.0]);
        let stepped = net.midpoint_step(&x, 0.1);
        assert_eq!(stepped.len(), 2);
    }

    #[test]
    fn test_diffusion_matrix() {
        let mut sheaf = CellularSheaf::new_uniform(3, 1).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 0, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(1, 1)).unwrap();
        sheaf.add_edge(2, 1, DMatrix::identity(1, 1)).unwrap();
        let net = SheafDiffusion::new(sheaf, &[], Activation::Identity).unwrap();

        let dm = net.diffusion_matrix(1.0);
        assert_eq!(dm.nrows(), 3);
        assert_eq!(dm.ncols(), 3);
        // All entries should be positive for connected graph
        for i in 0..3 {
            for j in 0..3 {
                assert!(dm[(i, j)] > 0.0);
            }
        }
    }

    #[test]
    fn test_layer_with_weights() {
        let w = DMatrix::identity(4, 4);
        let b = DVector::zeros(4);
        let layer = SheafDiffusionLayer::with_weights(w, b, Activation::Identity);

        let mut sheaf = CellularSheaf::new_uniform(2, 2).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(1, 0, DMatrix::identity(2, 2)).unwrap();
        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();

        let x = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let out = layer.forward(&x, &lap);
        // With identity weights and zero bias, output is diffused input
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_forward_linear() {
        let w = DMatrix::identity(3, 3);
        let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let layer = SheafDiffusionLayer::with_weights(w, b, Activation::Identity);
        let x = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let out = layer.forward_linear(&x);
        assert_relative_eq!(out[0], 2.0);
        assert_relative_eq!(out[1], 3.0);
        assert_relative_eq!(out[2], 4.0);
    }
}
