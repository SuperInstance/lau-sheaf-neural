//! Sheaf-aware graph pooling.
//!
//! Standard graph pooling (earsening) only coarsens the graph structure.
//! Sheaf pooling coarsens the sheaf: it merges stalks and composes restriction
//! maps, producing a hierarchical representation.
//!
//! The key insight: when merging nodes i and j into a supernode,
//! the new stalk is F(i) ⊕ F(j) (or a subspace thereof), and the
//! restriction maps are composed accordingly.

use crate::laplacian::SheafLaplacian;
use crate::sheaf::{CellularSheaf, SheafEdge, SheafError};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Strategy for merging stalks when pooling nodes.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum MergeStrategy {
    /// Direct sum: new stalk dim = dim_i + dim_j.
    DirectSum,
    /// Take the maximum dimension (pad smaller with zeros).
    MaxDim,
    /// Project onto a common subspace of given dimension.
    Project(usize),
    /// Average the stalks (only valid for same-dimension stalks).
    Average,
}

/// Result of a single pooling step.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolingResult {
    /// The coarsened sheaf.
    pub coarsened_sheaf: CellularSheaf,
    /// Assignment of original nodes to supernodes.
    pub assignment: Vec<usize>,
    /// Number of supernodes.
    pub num_supernodes: usize,
    /// Projection matrix: (new_total_dim × old_total_dim).
    pub projection: DMatrix<f64>,
    /// Lift matrix: (old_total_dim × new_total_dim), pseudo-inverse of projection.
    pub lift: DMatrix<f64>,
}

/// Sheaf pooling module.
#[derive(Clone, Debug)]
pub struct SheafPooling {
    /// Merge strategy.
    pub merge_strategy: MergeStrategy,
    /// Ratio of nodes to keep (0.0 to 1.0).
    pub pool_ratio: f64,
}

impl SheafPooling {
    /// Create a new sheaf pooling module.
    pub fn new(merge_strategy: MergeStrategy, pool_ratio: f64) -> Self {
        SheafPooling {
            merge_strategy,
            pool_ratio: pool_ratio.clamp(0.1, 1.0),
        }
    }

    /// Perform one step of sheaf pooling.
    ///
    /// Groups nodes into clusters and merges their stalks.
    pub fn pool(&self, sheaf: &CellularSheaf, clusters: &[Vec<usize>]) -> Result<PoolingResult, SheafError> {
        sheaf.validate()?;

        let num_supernodes = clusters.len();
        if num_supernodes == 0 {
            return Err(SheafError::EmptySheaf);
        }

        // Compute new stalk dimensions based on merge strategy
        let mut new_stalk_dims = Vec::new();
        for cluster in clusters {
            let dim = match self.merge_strategy {
                MergeStrategy::DirectSum => cluster.iter().map(|&n| sheaf.stalk_dims[n]).sum(),
                MergeStrategy::MaxDim => cluster.iter().map(|&n| sheaf.stalk_dims[n]).max().unwrap_or(1),
                MergeStrategy::Project(d) => d.min(cluster.iter().map(|&n| sheaf.stalk_dims[n]).sum()),
                MergeStrategy::Average => cluster.iter().map(|&n| sheaf.stalk_dims[n]).max().unwrap_or(1),
            };
            new_stalk_dims.push(dim);
        }

        let new_total_dim: usize = new_stalk_dims.iter().sum();

        // Build assignment mapping
        let mut assignment = vec![0usize; sheaf.num_nodes];
        for (super_idx, cluster) in clusters.iter().enumerate() {
            for &node in cluster {
                assignment[node] = super_idx;
            }
        }

        // Build projection matrix
        let mut projection = DMatrix::zeros(new_total_dim, sheaf.total_dim);

        for (super_idx, cluster) in clusters.iter().enumerate() {
            let new_offset: usize = new_stalk_dims[..super_idx].iter().sum();
            let new_dim = new_stalk_dims[super_idx];

            match self.merge_strategy {
                MergeStrategy::DirectSum => {
                    let mut col_offset = 0;
                    for &node in cluster {
                        let old_offset = sheaf.node_offset(node);
                        let old_dim = sheaf.stalk_dims[node];
                        for k in 0..old_dim.min(new_dim) {
                            if col_offset + k < new_dim {
                                projection[(new_offset + col_offset + k, old_offset + k)] = 1.0;
                            }
                        }
                        col_offset += old_dim;
                    }
                }
                MergeStrategy::MaxDim | MergeStrategy::Average => {
                    let dim = new_dim;
                    let n = cluster.len() as f64;
                    for &node in cluster {
                        let old_offset = sheaf.node_offset(node);
                        for k in 0..dim.min(sheaf.stalk_dims[node]) {
                            let weight = match self.merge_strategy {
                                MergeStrategy::Average => 1.0 / n,
                                _ => 1.0,
                            };
                            projection[(new_offset + k, old_offset + k)] = weight;
                        }
                    }
                }
                MergeStrategy::Project(target_dim) => {
                    let total_cluster_dim: usize = cluster.iter().map(|&n| sheaf.stalk_dims[n]).sum();
                    for k in 0..target_dim.min(total_cluster_dim).min(new_dim) {
                        projection[(new_offset + k, k)] = 1.0;
                    }
                }
            }
        }

        // Build lift matrix (pseudo-inverse: transpose of projection for simplicity)
        let lift = projection.transpose();

        // Build edges for the coarsened sheaf
        let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
        for edge in &sheaf.edges {
            let new_s = assignment[edge.source];
            let new_t = assignment[edge.target];
            if new_s != new_t {
                edge_set.insert((new_s, new_t));
            }
        }

        let mut new_edges: Vec<(usize, usize, DMatrix<f64>)> = Vec::new();
        for &(s, t) in &edge_set {
            let sd = new_stalk_dims[s];
            let td = new_stalk_dims[t];
            // Use identity for new restriction maps (could be more sophisticated)
            let map = if sd == td {
                DMatrix::identity(td, sd)
            } else {
                DMatrix::zeros(td, sd)
            };
            new_edges.push((s, t, map));
        }

        let coarsened_sheaf = CellularSheaf::with_restriction_maps(new_stalk_dims, new_edges)?;

        Ok(PoolingResult {
            coarsened_sheaf,
            assignment,
            num_supernodes,
            projection,
            lift,
        })
    }

    /// Auto-cluster nodes based on graph structure (greedy pooling).
    pub fn auto_cluster(&self, sheaf: &CellularSheaf) -> Vec<Vec<usize>> {
        let target_clusters = ((sheaf.num_nodes as f64) * self.pool_ratio).ceil() as usize;
        let target_clusters = target_clusters.max(1).min(sheaf.num_nodes);

        // Simple greedy clustering: pair up adjacent nodes
        let mut used: HashSet<usize> = HashSet::new();
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        for edge in &sheaf.edges {
            if used.contains(&edge.source) || used.contains(&edge.target) {
                continue;
            }
            clusters.push(vec![edge.source, edge.target]);
            used.insert(edge.source);
            used.insert(edge.target);
        }

        // Add remaining unclustered nodes
        for i in 0..sheaf.num_nodes {
            if !used.contains(&i) {
                clusters.push(vec![i]);
            }
        }

        // Trim to target if needed
        while clusters.len() > target_clusters && clusters.len() > 1 {
            if let Some(last) = clusters.pop() {
                if let Some(cluster) = clusters.last_mut() {
                    cluster.extend(last);
                }
            }
        }

        clusters
    }

    /// Full pooling pipeline: auto-cluster and pool.
    pub fn pool_auto(&self, sheaf: &CellularSheaf) -> Result<PoolingResult, SheafError> {
        let clusters = self.auto_cluster(sheaf);
        self.pool(sheaf, &clusters)
    }

    /// Project features from original sheaf to coarsened sheaf.
    pub fn project_features(pool_result: &PoolingResult, features: &DVector<f64>) -> DVector<f64> {
        &pool_result.projection * features
    }

    /// Lift features from coarsened sheaf back to original sheaf.
    pub fn lift_features(pool_result: &PoolingResult, features: &DVector<f64>) -> DVector<f64> {
        &pool_result.lift * features
    }

    /// Multi-level hierarchical pooling.
    pub fn hierarchical_pool(
        &self,
        sheaf: &CellularSheaf,
        levels: usize,
    ) -> Result<Vec<PoolingResult>, SheafError> {
        let mut results = Vec::new();
        let mut current_sheaf = sheaf.clone();

        for _ in 0..levels {
            let result = self.pool_auto(&current_sheaf)?;
            if result.coarsened_sheaf.num_nodes <= 1 {
                break;
            }
            current_sheaf = result.coarsened_sheaf.clone();
            results.push(result);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_grid_sheaf() -> CellularSheaf {
        // 2x2 grid
        let mut sheaf = CellularSheaf::new_uniform(4, 2).unwrap();
        let edges = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1), (2, 3), (3, 2)];
        for (s, t) in edges {
            sheaf.add_edge(s, t, DMatrix::identity(2, 2)).unwrap();
        }
        sheaf
    }

    #[test]
    fn test_pooling_basic() {
        let sheaf = make_grid_sheaf();
        let pooling = SheafPooling::new(MergeStrategy::MaxDim, 0.5);
        let clusters = vec![vec![0, 1], vec![2, 3]];
        let result = pooling.pool(&sheaf, &clusters).unwrap();

        assert_eq!(result.num_supernodes, 2);
        assert_eq!(result.assignment[0], 0);
        assert_eq!(result.assignment[1], 0);
        assert_eq!(result.assignment[2], 1);
        assert_eq!(result.assignment[3], 1);
    }

    #[test]
    fn test_pooling_coarsened_sheaf() {
        let sheaf = make_grid_sheaf();
        let pooling = SheafPooling::new(MergeStrategy::MaxDim, 0.5);
        let clusters = vec![vec![0, 1], vec![2, 3]];
        let result = pooling.pool(&sheaf, &clusters).unwrap();

        assert_eq!(result.coarsened_sheaf.num_nodes, 2);
        // Should have edges between supernodes
        assert!(result.coarsened_sheaf.num_edges() > 0);
    }

    #[test]
    fn test_direct_sum_merge() {
        let sheaf = make_grid_sheaf();
        let pooling = SheafPooling::new(MergeStrategy::DirectSum, 0.5);
        let clusters = vec![vec![0, 1], vec![2, 3]];
        let result = pooling.pool(&sheaf, &clusters).unwrap();

        // Direct sum: each supernode has stalk dim 2+2=4
        assert_eq!(result.coarsened_sheaf.total_dim, 8);
    }

    #[test]
    fn test_max_dim_merge() {
        let sheaf = make_grid_sheaf();
        let pooling = SheafPooling::new(MergeStrategy::MaxDim, 0.5);
        let clusters = vec![vec![0, 1], vec![2, 3]];
        let result = pooling.pool(&sheaf, &clusters).unwrap();

        // Max dim: each supernode has stalk dim 2
        assert_eq!(result.coarsened_sheaf.total_dim, 4);
    }

    #[test]
    fn test_average_merge() {
        let sheaf = make_grid_sheaf();
        let pooling = SheafPooling::new(MergeStrategy::Average, 0.5);
        let clusters = vec![vec![0, 1], vec![2, 3]];
        let result = pooling.pool(&sheaf, &clusters).unwrap();

        assert_eq!(result.coarsened_sheaf.num_nodes, 2);
    }

    #[test]
    fn test_project_merge() {
        let sheaf = make_grid_sheaf();
        let pooling = SheafPooling::new(MergeStrategy::Project(3), 0.5);
        let clusters = vec![vec![0, 1], vec![2, 3]];
        let result = pooling.pool(&sheaf, &clusters).unwrap();
        assert_eq!(result.coarsened_sheaf.num_nodes, 2);
    }

    #[test]
    fn test_project_features() {
        let sheaf = make_grid_sheaf();
        let pooling = SheafPooling::new(MergeStrategy::MaxDim, 0.5);
        let clusters = vec![vec![0, 1], vec![2, 3]];
        let result = pooling.pool(&sheaf, &clusters).unwrap();

        let features = DVector::from_element(8, 1.0);
        let projected = SheafPooling::project_features(&result, &features);
        assert_eq!(projected.len(), result.coarsened_sheaf.total_dim);
    }

    #[test]
    fn test_lift_features() {
        let sheaf = make_grid_sheaf();
        let pooling = SheafPooling::new(MergeStrategy::MaxDim, 0.5);
        let clusters = vec![vec![0, 1], vec![2, 3]];
        let result = pooling.pool(&sheaf, &clusters).unwrap();

        let coarse_features = DVector::from_element(4, 1.0);
        let lifted = SheafPooling::lift_features(&result, &coarse_features);
        assert_eq!(lifted.len(), 8);
    }

    #[test]
    fn test_auto_cluster() {
        let sheaf = make_grid_sheaf();
        let pooling = SheafPooling::new(MergeStrategy::MaxDim, 0.5);
        let clusters = pooling.auto_cluster(&sheaf);
        assert!(clusters.len() <= 4);
        assert!(clusters.len() >= 1);
        // All nodes should be assigned
        let all_nodes: HashSet<usize> = clusters.iter().flatten().copied().collect();
        assert_eq!(all_nodes.len(), 4);
    }

    #[test]
    fn test_pool_auto() {
        let sheaf = make_grid_sheaf();
        let pooling = SheafPooling::new(MergeStrategy::MaxDim, 0.5);
        let result = pooling.pool_auto(&sheaf).unwrap();
        assert!(result.coarsened_sheaf.num_nodes < sheaf.num_nodes);
    }

    #[test]
    fn test_hierarchical_pool() {
        // Larger graph for hierarchical pooling
        let mut sheaf = CellularSheaf::new_uniform(8, 2).unwrap();
        for i in 0..7 {
            sheaf.add_edge(i, i + 1, DMatrix::identity(2, 2)).unwrap();
            sheaf.add_edge(i + 1, i, DMatrix::identity(2, 2)).unwrap();
        }

        let pooling = SheafPooling::new(MergeStrategy::MaxDim, 0.5);
        let results = pooling.hierarchical_pool(&sheaf, 3).unwrap();
        assert!(results.len() >= 1);
        // Each level should have fewer nodes
        for i in 1..results.len() {
            assert!(results[i].coarsened_sheaf.num_nodes < results[i - 1].coarsened_sheaf.num_nodes);
        }
    }

    #[test]
    fn test_single_node_clusters() {
        let sheaf = make_grid_sheaf();
        let pooling = SheafPooling::new(MergeStrategy::MaxDim, 1.0);
        let clusters = vec![vec![0], vec![1], vec![2], vec![3]];
        let result = pooling.pool(&sheaf, &clusters).unwrap();
        assert_eq!(result.coarsened_sheaf.num_nodes, 4);
    }

    #[test]
    fn test_empty_clusters_error() {
        let sheaf = make_grid_sheaf();
        let pooling = SheafPooling::new(MergeStrategy::MaxDim, 0.5);
        let result = pooling.pool(&sheaf, &[]);
        assert!(result.is_err());
    }
}
