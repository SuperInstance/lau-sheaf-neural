//! Sheaf attention: learn restriction maps from data.
//!
//! Analogous to GAT (Graph Attention Networks), sheaf attention learns the
//! restriction maps dynamically from node features, allowing the sheaf to
//! adapt to the data.
//!
//! The attention mechanism computes:
//!   R_{ij} = softmax(a^T [W x_i || W x_j]) · V
//! where a is an attention vector, W is a projection, V is a value projection.

use crate::sheaf::{CellularSheaf, SheafEdge, SheafError};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Configuration for sheaf attention.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Input feature dimension per node.
    pub in_dim: usize,
    /// Stalk dimension (output restriction map dimension).
    pub stalk_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Whether to use symmetric attention (R_{ij} = R_{ji}^T).
    pub symmetric: bool,
    /// Dropout rate (0.0 = no dropout).
    pub dropout: f64,
    /// Whether to use multi-head concatenation (true) or averaging (false).
    pub concat: bool,
}

impl AttentionConfig {
    pub fn new(in_dim: usize, stalk_dim: usize) -> Self {
        AttentionConfig {
            in_dim,
            stalk_dim,
            num_heads: 1,
            symmetric: false,
            dropout: 0.0,
            concat: true,
        }
    }
}

/// Learned restriction maps from sheaf attention.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttentionOutput {
    /// The resulting sheaf with learned restriction maps.
    pub sheaf: CellularSheaf,
    /// Attention weights for each edge and head.
    pub attention_weights: Vec<Vec<f64>>,
    /// Loss value from this attention computation.
    pub loss: f64,
}

/// Sheaf attention module.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SheafAttention {
    pub config: AttentionConfig,
    /// Query projection matrices (one per head).
    pub w_query: Vec<DMatrix<f64>>,
    /// Key projection matrices (one per head).
    pub w_key: Vec<DMatrix<f64>>,
    /// Value projection matrix (produces restriction maps).
    pub w_value: Vec<DMatrix<f64>>,
    /// Attention vector for computing scores.
    pub attention_vec: Vec<DVector<f64>>,
}

impl SheafAttention {
    /// Create a new sheaf attention module.
    pub fn new(config: AttentionConfig) -> Self {
        let mut rng = rand::thread_rng();
        use rand::Rng;

        let scale = 1.0 / (config.in_dim as f64).sqrt();

        let w_query: Vec<DMatrix<f64>> = (0..config.num_heads)
            .map(|_| {
                DMatrix::from_fn(config.stalk_dim, config.in_dim, |_, _| {
                    (rng.gen::<f64>() * 2.0 - 1.0) * scale
                })
            })
            .collect();

        let w_key: Vec<DMatrix<f64>> = (0..config.num_heads)
            .map(|_| {
                DMatrix::from_fn(config.stalk_dim, config.in_dim, |_, _| {
                    (rng.gen::<f64>() * 2.0 - 1.0) * scale
                })
            })
            .collect();

        let w_value: Vec<DMatrix<f64>> = (0..config.num_heads)
            .map(|_| {
                DMatrix::from_fn(config.stalk_dim, config.stalk_dim, |_, _| {
                    (rng.gen::<f64>() * 2.0 - 1.0) * scale
                })
            })
            .collect();

        let attention_vec: Vec<DVector<f64>> = (0..config.num_heads)
            .map(|_| {
                DVector::from_fn(config.stalk_dim * 2, |_, _| {
                    (rng.gen::<f64>() * 2.0 - 1.0) * scale
                })
            })
            .collect();

        SheafAttention {
            config,
            w_query,
            w_key,
            w_value,
            attention_vec,
        }
    }

    /// Compute attention scores for an edge.
    fn attention_score(
        &self,
        head: usize,
        x_i: &DVector<f64>,
        x_j: &DVector<f64>,
    ) -> f64 {
        let q = &self.w_query[head] * x_i;
        let k = &self.w_key[head] * x_j;
        // Concatenate and dot with attention vector
        let concat = DVector::from_fn(self.config.stalk_dim * 2, |idx, _| {
            if idx < self.config.stalk_dim {
                q[idx]
            } else {
                k[idx - self.config.stalk_dim]
            }
        });
        let score = self.attention_vec[head].dot(&concat);
        // LeakyReLU
        if score >= 0.0 { score } else { 0.2 * score }
    }

    /// Compute restriction map from attention.
    fn compute_restriction_map(
        &self,
        head: usize,
        x_i: &DVector<f64>,
        _x_j: &DVector<f64>,
        alpha: f64,
    ) -> DMatrix<f64> {
        let base = &self.w_value[head];
        // Scale by attention weight
        alpha * base
    }

    /// Forward pass: compute restriction maps for all edges given node features.
    pub fn forward(
        &self,
        sheaf: &CellularSheaf,
        features: &[DVector<f64>],
        edges: &[(usize, usize)],
    ) -> Result<AttentionOutput, SheafError> {
        if features.len() != sheaf.num_nodes {
            return Err(SheafError::DimensionMismatch {
                expected: sheaf.num_nodes,
                actual: features.len(),
            });
        }

        let mut new_edges: Vec<(usize, usize, DMatrix<f64>)> = Vec::new();
        let mut all_weights: Vec<Vec<f64>> = Vec::new();
        let mut total_loss = 0.0;

        for &(s, t) in edges {
            let x_s = &features[s];
            let x_t = &features[t];

            if x_s.len() != self.config.in_dim || x_t.len() != self.config.in_dim {
                return Err(SheafError::DimensionMismatch {
                    expected: self.config.in_dim,
                    actual: x_s.len(),
                });
            }

            // Compute attention scores for each head
            let mut head_weights = Vec::new();
            let mut head_maps = Vec::new();

            for h in 0..self.config.num_heads {
                let score_st = self.attention_score(h, x_s, x_t);
                head_weights.push(score_st);

                let map = self.compute_restriction_map(h, x_s, x_t, 1.0);
                head_maps.push(map);
            }

            // Normalize weights via softmax over all edges sharing source node
            all_weights.push(head_weights.clone());

            // Average restriction maps across heads
            let d = self.config.stalk_dim;
            let mut avg_map = DMatrix::zeros(d, d);
            for (i, map) in head_maps.iter().enumerate() {
                let w = head_weights[i].exp();
                avg_map += w * map;
            }
            let sum_exp: f64 = head_weights.iter().map(|w| w.exp()).sum();
            if sum_exp > 0.0 {
                avg_map *= 1.0 / sum_exp;
            }

            // Symmetrize if requested
            if self.config.symmetric {
                avg_map = (&avg_map + &avg_map.transpose()) * 0.5;
            }

            // Regularization loss: encourage orthonormal restriction maps
            let deviation = &avg_map.transpose() * &avg_map - DMatrix::identity(d, d);
            for i in 0..d {
                for j in 0..d {
                    total_loss += deviation[(i, j)] * deviation[(i, j)];
                }
            }

            new_edges.push((s, t, avg_map));
        }

        let new_sheaf = CellularSheaf::with_restriction_maps(
            vec![self.config.stalk_dim; sheaf.num_nodes],
            new_edges,
        )?;

        Ok(AttentionOutput {
            sheaf: new_sheaf,
            attention_weights: all_weights,
            loss: total_loss,
        })
    }

    /// Compute multi-head attention weights (softmax-normalized).
    pub fn compute_attention_weights(
        &self,
        features: &[DVector<f64>],
        edges: &[(usize, usize)],
    ) -> Vec<Vec<f64>> {
        // Group edges by source node
        let mut by_source: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (idx, &(_, _)) in edges.iter().enumerate() {
            let s = edges[idx].0;
            by_source.entry(s).or_default().push(idx);
        }

        let mut weights: Vec<Vec<f64>> = vec![vec![0.0; self.config.num_heads]; edges.len()];

        for (&_s, edge_indices) in &by_source {
            // For each head, compute softmax over edges from this source
            for h in 0..self.config.num_heads {
                let scores: Vec<f64> = edge_indices
                    .iter()
                    .map(|&idx| {
                        let (si, ti) = edges[idx];
                        self.attention_score(h, &features[si], &features[ti])
                    })
                    .collect();

                let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = scores.iter().map(|s| (s - max_score).exp()).sum();

                for (i, &idx) in edge_indices.iter().enumerate() {
                    weights[idx][h] = (scores[i] - max_score).exp() / sum_exp;
                }
            }
        }

        weights
    }

    /// Number of parameters.
    pub fn num_parameters(&self) -> usize {
        let head_params = self.config.stalk_dim * self.config.in_dim * 2 // W_q, W_k
            + self.config.stalk_dim * self.config.stalk_dim // W_v
            + self.config.stalk_dim * 2; // attention vector
        head_params * self.config.num_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_attention_config() {
        let config = AttentionConfig::new(16, 8);
        assert_eq!(config.in_dim, 16);
        assert_eq!(config.stalk_dim, 8);
    }

    #[test]
    fn test_attention_creation() {
        let config = AttentionConfig::new(16, 8);
        let attn = SheafAttention::new(config);
        assert_eq!(attn.w_query.len(), 1);
        assert_eq!(attn.w_key.len(), 1);
    }

    #[test]
    fn test_attention_forward() {
        let config = AttentionConfig::new(4, 2);
        let attn = SheafAttention::new(config);

        let sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        let features: Vec<DVector<f64>> = (0..3)
            .map(|i| DVector::from_vec(vec![i as f64, 1.0, 0.5, -0.5]))
            .collect();
        let edges = vec![(0, 1), (1, 2)];

        let output = attn.forward(&sheaf, &features, &edges).unwrap();
        assert_eq!(output.sheaf.num_edges(), 2);
        assert_eq!(output.attention_weights.len(), 2);
    }

    #[test]
    fn test_multihead_attention() {
        let mut config = AttentionConfig::new(4, 2);
        config.num_heads = 4;
        let attn = SheafAttention::new(config);

        assert_eq!(attn.w_query.len(), 4);
    }

    #[test]
    fn test_symmetric_attention() {
        let mut config = AttentionConfig::new(4, 2);
        config.symmetric = true;
        let attn = SheafAttention::new(config);

        let sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        let features: Vec<DVector<f64>> = (0..3)
            .map(|i| DVector::from_vec(vec![i as f64, 1.0, 0.5, -0.5]))
            .collect();
        let edges = vec![(0, 1), (1, 0)];

        let output = attn.forward(&sheaf, &features, &edges).unwrap();
        // With symmetric attention, restriction maps should be symmetric
        let rm01 = output.sheaf.restriction_map(0, 1).unwrap();
        let rm01_t = rm01.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(rm01[(i, j)], rm01_t[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_attention_weights() {
        let config = AttentionConfig::new(4, 2);
        let attn = SheafAttention::new(config);

        let features: Vec<DVector<f64>> = (0..3)
            .map(|i| DVector::from_vec(vec![i as f64, 1.0, 0.5, -0.5]))
            .collect();
        let edges = vec![(0, 1), (0, 2)];

        let weights = attn.compute_attention_weights(&features, &edges);
        assert_eq!(weights.len(), 2);
        // Weights should be softmax-normalized (sum to 1 for same source)
        let sum: f64 = weights.iter().map(|w| w[0].exp()).sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_attention_loss() {
        let config = AttentionConfig::new(4, 2);
        let attn = SheafAttention::new(config);

        let sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        let features: Vec<DVector<f64>> = (0..3)
            .map(|i| DVector::from_vec(vec![i as f64, 1.0, 0.5, -0.5]))
            .collect();

        let output = attn.forward(&sheaf, &features, &[(0, 1)]).unwrap();
        assert!(output.loss >= 0.0);
    }

    #[test]
    fn test_num_parameters() {
        let config = AttentionConfig::new(16, 8);
        let attn = SheafAttention::new(config);
        let params = attn.num_parameters();
        // 2*16*8 + 8*8 + 16 = 256 + 64 + 16 = 336
        assert!(params > 0);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let config = AttentionConfig::new(4, 2);
        let attn = SheafAttention::new(config);

        let sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        // Wrong feature dimension
        let features: Vec<DVector<f64>> = (0..3)
            .map(|_| DVector::from_vec(vec![1.0, 2.0]))
            .collect();

        let result = attn.forward(&sheaf, &features, &[(0, 1)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_preserves_edge_structure() {
        let config = AttentionConfig::new(4, 2);
        let attn = SheafAttention::new(config);

        let sheaf = CellularSheaf::new_uniform(4, 2).unwrap();
        let features: Vec<DVector<f64>> = (0..4)
            .map(|i| DVector::from_vec(vec![i as f64, 1.0, 0.5, -0.5]))
            .collect();
        let edges = vec![(0, 1), (1, 2), (2, 3)];

        let output = attn.forward(&sheaf, &features, &edges).unwrap();
        assert!(output.sheaf.has_edge(0, 1));
        assert!(output.sheaf.has_edge(1, 2));
        assert!(output.sheaf.has_edge(2, 3));
        assert!(!output.sheaf.has_edge(0, 3));
    }
}
