//! Sheaf curvature and over-squashing diagnosis.
//!
//! Over-squashing occurs when the graph curvature is too negative on certain
//! edges, preventing effective information flow. The **sheaf curvature** extends
//! the notion of Ollivier-Ricci curvature to sheaves:
//!
//!   κ_Σ(i,j) = 1 - ||Σ_{ji} + Σ_{ij}||_F / 2
//!
//! When restriction maps are identity, this reduces to graph curvature.
//! Negative curvature edges are bottlenecks; the sheaf can "fix" them by
//! learning appropriate restriction maps.

use crate::sheaf::{CellularSheaf, SheafError};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Curvature analysis result for a single edge.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeCurvature {
    pub source: usize,
    pub target: usize,
    /// Sheaf curvature (Ollivier-Ricci style).
    pub curvature: f64,
    /// Frobenius norm of the restriction map.
    pub restriction_norm: f64,
    /// Whether this edge is an over-squashing bottleneck.
    pub is_bottleneck: bool,
}

/// Sheaf curvature analyzer.
#[derive(Clone, Debug)]
pub struct SheafCurvature {
    /// Curvature of each edge.
    pub edge_curvatures: Vec<EdgeCurvature>,
    /// Bottleneck threshold (edges below this are problematic).
    pub bottleneck_threshold: f64,
}

impl SheafCurvature {
    /// Compute sheaf curvatures for all edges.
    pub fn from_sheaf(sheaf: &CellularSheaf, threshold: f64) -> Result<Self, SheafError> {
        sheaf.validate()?;
        let mut curvatures = Vec::new();

        for edge in &sheaf.edges {
            let s = edge.source;
            let t = edge.target;

            let r_st = &edge.restriction_map;
            let r_ts = sheaf
                .restriction_map(t, s)
                .cloned()
                .unwrap_or_else(|| DMatrix::zeros(sheaf.stalk_dims[s], sheaf.stalk_dims[t]));

            // Sheaf curvature: κ(i,j) = 1 - ||R_{ij} + R_{ji}||_F / 2
            let sum = r_st + &r_ts;
            let frob = frobenius_norm(&sum);
            let kappa = 1.0 - frob / 2.0;

            curvatures.push(EdgeCurvature {
                source: s,
                target: t,
                curvature: kappa,
                restriction_norm: frobenius_norm(r_st),
                is_bottleneck: kappa < threshold,
            });
        }

        Ok(SheafCurvature {
            edge_curvatures: curvatures,
            bottleneck_threshold: threshold,
        })
    }

    /// Get the most negative curvature (worst bottleneck).
    pub fn min_curvature(&self) -> Option<f64> {
        self.edge_curvatures
            .iter()
            .map(|e| e.curvature)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Get the maximum curvature.
    pub fn max_curvature(&self) -> Option<f64> {
        self.edge_curvatures
            .iter()
            .map(|e| e.curvature)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Get the average curvature.
    pub fn mean_curvature(&self) -> f64 {
        if self.edge_curvatures.is_empty() {
            return 0.0;
        }
        self.edge_curvatures.iter().map(|e| e.curvature).sum::<f64>()
            / self.edge_curvatures.len() as f64
    }

    /// Get all bottleneck edges (curvature below threshold).
    pub fn bottleneck_edges(&self) -> Vec<&EdgeCurvature> {
        self.edge_curvatures
            .iter()
            .filter(|e| e.is_bottleneck)
            .collect()
    }

    /// Count of bottleneck edges.
    pub fn num_bottlenecks(&self) -> usize {
        self.edge_curvatures.iter().filter(|e| e.is_bottleneck).count()
    }

    /// Diagnose over-squashing: compute the "over-squashing score" for the graph.
    ///
    /// Score = (number of bottleneck edges) / (total edges).
    /// 0 = no over-squashing, 1 = all edges are bottlenecks.
    pub fn over_squashing_score(&self) -> f64 {
        if self.edge_curvatures.is_empty() {
            return 0.0;
        }
        self.num_bottlenecks() as f64 / self.edge_curvatures.len() as f64
    }

    /// Compute the effective resistance for each edge.
    ///
    /// Effective resistance = 1 / (1 + κ(i,j)) for positive curvature.
    pub fn effective_resistance(&self) -> Vec<(usize, usize, f64)> {
        self.edge_curvatures
            .iter()
            .map(|e| {
                let r = if e.curvature > -1.0 {
                    1.0 / (1.0 + e.curvature)
                } else {
                    f64::INFINITY
                };
                (e.source, e.target, r)
            })
            .collect()
    }

    /// Suggest stalk dimension increases for bottleneck edges.
    ///
    /// The idea: increase stalk dimensions at bottleneck nodes to allow
    /// more information to flow through.
    pub fn suggest_dimension_increases(&self, sheaf: &CellularSheaf) -> Vec<(usize, usize)> {
        let mut suggestions = Vec::new();
        for edge in &self.edge_curvatures {
            if edge.is_bottleneck {
                let current_dim = sheaf.stalk_dim(edge.source);
                // Suggest doubling the dimension
                suggestions.push((edge.source, current_dim * 2));
            }
        }
        suggestions
    }

    /// Get curvature for a specific edge.
    pub fn curvature_for_edge(&self, source: usize, target: usize) -> Option<f64> {
        self.edge_curvatures
            .iter()
            .find(|e| e.source == source && e.target == target)
            .map(|e| e.curvature)
    }
}

/// Compute the Frobenius norm of a matrix.
fn frobenius_norm(m: &DMatrix<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            sum += m[(i, j)] * m[(i, j)];
        }
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_bipartite_sheaf() -> CellularSheaf {
        // A bottleneck graph: two clusters connected by a single edge
        let mut sheaf = CellularSheaf::new_uniform(6, 2).unwrap();
        // Cluster 1
        sheaf.add_edge(0, 1, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(1, 0, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(2, 1, DMatrix::identity(2, 2)).unwrap();
        // Cluster 2
        sheaf.add_edge(3, 4, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(4, 3, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(4, 5, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(5, 4, DMatrix::identity(2, 2)).unwrap();
        // Bottleneck
        sheaf.add_edge(2, 3, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(3, 2, DMatrix::identity(2, 2)).unwrap();
        sheaf
    }

    #[test]
    fn test_curvature_identity_restriction() {
        let mut sheaf = CellularSheaf::new_uniform(2, 2).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(1, 0, DMatrix::identity(2, 2)).unwrap();

        let curv = SheafCurvature::from_sheaf(&sheaf, -0.5).unwrap();
        // Identity restriction: ||I + I||_F = 2*sqrt(2), curvature = 1 - sqrt(2) ≈ -0.414
        let kappa = curv.curvature_for_edge(0, 1).unwrap();
        assert_relative_eq!(kappa, 1.0 - 2_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_curvature_zero_restriction() {
        let mut sheaf = CellularSheaf::new_uniform(2, 2).unwrap();
        sheaf
            .add_edge(0, 1, DMatrix::zeros(2, 2))
            .unwrap();
        sheaf
            .add_edge(1, 0, DMatrix::zeros(2, 2))
            .unwrap();

        let curv = SheafCurvature::from_sheaf(&sheaf, -0.5).unwrap();
        let kappa = curv.curvature_for_edge(0, 1).unwrap();
        // Zero maps: curvature = 1 - 0 = 1
        assert_relative_eq!(kappa, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bottleneck_detection() {
        let sheaf = make_bipartite_sheaf();
        let curv = SheafCurvature::from_sheaf(&sheaf, -0.5).unwrap();
        // All edges have the same curvature with identity maps, so none should
        // be bottlenecks unless threshold is set appropriately
        assert!(curv.edge_curvatures.len() > 0);
    }

    #[test]
    fn test_min_max_curvature() {
        let sheaf = make_bipartite_sheaf();
        let curv = SheafCurvature::from_sheaf(&sheaf, -1.0).unwrap();
        let min = curv.min_curvature().unwrap();
        let max = curv.max_curvature().unwrap();
        assert!(min <= max);
    }

    #[test]
    fn test_mean_curvature() {
        let sheaf = make_bipartite_sheaf();
        let curv = SheafCurvature::from_sheaf(&sheaf, -1.0).unwrap();
        let mean = curv.mean_curvature();
        assert!(mean.is_finite());
    }

    #[test]
    fn test_over_squashing_score() {
        let sheaf = make_bipartite_sheaf();
        let curv = SheafCurvature::from_sheaf(&sheaf, -1.0).unwrap();
        let score = curv.over_squashing_score();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_effective_resistance() {
        let sheaf = make_bipartite_sheaf();
        let curv = SheafCurvature::from_sheaf(&sheaf, -1.0).unwrap();
        let resistances = curv.effective_resistance();
        assert_eq!(resistances.len(), sheaf.num_edges());
        for (_, _, r) in &resistances {
            assert!(r.is_finite() && *r > 0.0);
        }
    }

    #[test]
    fn test_suggest_dimension_increases() {
        let sheaf = make_bipartite_sheaf();
        let curv = SheafCurvature::from_sheaf(&sheaf, 0.0).unwrap();
        let suggestions = curv.suggest_dimension_increases(&sheaf);
        // With threshold 0.0 and identity maps, edges with curvature < 0 are bottlenecks
        // κ = 1 - √2 ≈ -0.414 < 0, so all edges are bottlenecks
        assert!(suggestions.len() > 0);
    }

    #[test]
    fn test_empty_sheaf_curvature() {
        let sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        let curv = SheafCurvature::from_sheaf(&sheaf, -1.0).unwrap();
        assert_eq!(curv.edge_curvatures.len(), 0);
        assert_eq!(curv.over_squashing_score(), 0.0);
    }

    #[test]
    fn test_curvature_scaled_restriction() {
        // Scaling restriction maps should change curvature
        let mut sheaf1 = CellularSheaf::new_uniform(2, 2).unwrap();
        sheaf1.add_edge(0, 1, DMatrix::identity(2, 2)).unwrap();
        sheaf1.add_edge(1, 0, DMatrix::identity(2, 2)).unwrap();

        let mut sheaf2 = CellularSheaf::new_uniform(2, 2).unwrap();
        let scaled = DMatrix::from_diagonal(&nalgebra::DVector::from_element(2, 0.5));
        sheaf2.add_edge(0, 1, scaled.clone()).unwrap();
        sheaf2.add_edge(1, 0, scaled).unwrap();

        let curv1 = SheafCurvature::from_sheaf(&sheaf1, -1.0).unwrap();
        let curv2 = SheafCurvature::from_sheaf(&sheaf2, -1.0).unwrap();
        let k1 = curv1.curvature_for_edge(0, 1).unwrap();
        let k2 = curv2.curvature_for_edge(0, 1).unwrap();
        // With scaled maps, ||0.5I + 0.5I||_F = ||I||_F = √2, so curvature is same
        // Actually: ||I + I||_F = 2√2, ||0.5I + 0.5I||_F = √2
        assert!(k1 < k2); // Scaled maps → smaller sum → higher curvature
    }

    #[test]
    fn test_num_bottlenecks() {
        let sheaf = make_bipartite_sheaf();
        let curv = SheafCurvature::from_sheaf(&sheaf, 0.0).unwrap();
        // With threshold 0: κ = 1 - √2 ≈ -0.414 < 0 for all edges
        assert_eq!(curv.num_bottlenecks(), sheaf.num_edges());
    }
}
