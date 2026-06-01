//! Multi-hop sheaf: compose restriction maps across k-hop neighborhoods.
//!
//! Instead of only passing messages along direct edges, multi-hop sheaf
//! diffusion composes restriction maps across paths of length k:
//!
//!   R_{i→j}^{(k)} = R_{j_k j_{k-1}} ∘ ... ∘ R_{j_1 j_0}
//!
//! where (j_0=i, j_1, ..., j_k=j) is a path of length k.
//! This allows information to flow across longer distances while
//! respecting the sheaf geometry.

use crate::laplacian::SheafLaplacian;
use crate::sheaf::{CellularSheaf, SheafError};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A multi-hop composed restriction map.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComposedRestrictionMap {
    /// Source node.
    pub source: usize,
    /// Target node.
    pub target: usize,
    /// Number of hops.
    pub hops: usize,
    /// The composed restriction map.
    pub map: DMatrix<f64>,
    /// The path through which this map was composed.
    pub path: Vec<usize>,
}

/// Multi-hop sheaf: pre-computed k-hop restriction maps.
#[derive(Clone, Debug)]
pub struct MultiHopSheaf {
    /// The base sheaf.
    pub sheaf: CellularSheaf,
    /// Maximum hop distance.
    pub max_hops: usize,
    /// Composed restriction maps, indexed by (source, target, hops).
    composed_maps: HashMap<(usize, usize, usize), ComposedRestrictionMap>,
}

impl MultiHopSheaf {
    /// Build multi-hop restriction maps up to `max_hops` hops.
    pub fn new(sheaf: CellularSheaf, max_hops: usize) -> Result<Self, SheafError> {
        sheaf.validate()?;
        let mut composed: HashMap<(usize, usize, usize), ComposedRestrictionMap> = HashMap::new();

        // 1-hop: just the original restriction maps
        for edge in &sheaf.edges {
            composed.insert(
                (edge.source, edge.target, 1),
                ComposedRestrictionMap {
                    source: edge.source,
                    target: edge.target,
                    hops: 1,
                    map: edge.restriction_map.clone(),
                    path: vec![edge.source, edge.target],
                },
            );
        }

        // Build k-hop maps iteratively
        for k in 2..=max_hops {
            // For each (k-1)-hop map (s, m), extend by 1-hop maps from m
            let prev_maps: Vec<ComposedRestrictionMap> = composed
                .values()
                .filter(|m| m.hops == k - 1)
                .cloned()
                .collect();

            for prev in prev_maps {
                let mid = prev.target;
                for edge in &sheaf.edges {
                    if edge.source != mid {
                        continue;
                    }
                    let target = edge.target;
                    // Compose: R_{s→target} = R_{mid→target} ∘ R_{s→mid}
                    let composed_map = &edge.restriction_map * &prev.map;

                    let mut new_path = prev.path.clone();
                    new_path.push(target);

                    composed.insert(
                        (prev.source, target, k),
                        ComposedRestrictionMap {
                            source: prev.source,
                            target,
                            hops: k,
                            map: composed_map,
                            path: new_path,
                        },
                    );
                }
            }
        }

        Ok(MultiHopSheaf {
            sheaf,
            max_hops,
            composed_maps: composed,
        })
    }

    /// Get the k-hop restriction map from source to target.
    pub fn restriction_map(&self, source: usize, target: usize, k: usize) -> Option<&DMatrix<f64>> {
        self.composed_maps
            .get(&(source, target, k))
            .map(|m| &m.map)
    }

    /// Get all k-hop restriction maps from a source node.
    pub fn k_hop_maps_from(&self, source: usize, k: usize) -> Vec<&ComposedRestrictionMap> {
        self.composed_maps
            .values()
            .filter(|m| m.source == source && m.hops == k)
            .collect()
    }

    /// Get all restriction maps to a target node.
    pub fn maps_to(&self, target: usize) -> Vec<&ComposedRestrictionMap> {
        self.composed_maps
            .values()
            .filter(|m| m.target == target)
            .collect()
    }

    /// Build the k-hop sheaf Laplacian.
    ///
    /// Uses composed restriction maps for k-hop neighborhoods.
    pub fn k_hop_laplacian(&self, k: usize) -> Option<SheafLaplacian> {
        // Create a new sheaf with k-hop composed maps as edges
        let mut edges: Vec<(usize, usize, DMatrix<f64>)> = Vec::new();

        for m in self.composed_maps.values().filter(|m| m.hops == k) {
            edges.push((m.source, m.target, m.map.clone()));
        }

        if edges.is_empty() {
            return None;
        }

        let new_sheaf = CellularSheaf::with_restriction_maps(
            self.sheaf.stalk_dims.clone(),
            edges,
        )
        .ok()?;

        SheafLaplacian::from_sheaf(&new_sheaf).ok()
    }

    /// Multi-hop message passing: aggregate information from all hops up to max_hops.
    pub fn multi_hop_message_passing(
        &self,
        features: &DVector<f64>,
        weights: &[f64],
    ) -> DVector<f64> {
        let mut result = DVector::zeros(self.sheaf.total_dim);

        for k in 1..=self.max_hops {
            let w = weights.get(k - 1).copied().unwrap_or(1.0 / k as f64);

            for m in self.composed_maps.values().filter(|m| m.hops == k) {
                let x_source = self.sheaf.extract_stalk(m.source, features);
                let msg = &m.map * &x_source;
                let offset = self.sheaf.node_offset(m.target);
                let dim = self.sheaf.stalk_dims[m.target];

                for i in 0..dim {
                    if i < msg.len() {
                        result[offset + i] += w * msg[i];
                    }
                }
            }
        }

        result
    }

    /// Number of composed maps.
    pub fn num_composed_maps(&self) -> usize {
        self.composed_maps.len()
    }

    /// Count maps at a specific hop level.
    pub fn num_maps_at_hop(&self, k: usize) -> usize {
        self.composed_maps.values().filter(|m| m.hops == k).count()
    }

    /// Get the path for a specific composed map.
    pub fn path(&self, source: usize, target: usize, k: usize) -> Option<&Vec<usize>> {
        self.composed_maps
            .get(&(source, target, k))
            .map(|m| &m.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_chain_sheaf() -> CellularSheaf {
        let mut sheaf = CellularSheaf::new_uniform(5, 2).unwrap();
        for i in 0..4 {
            sheaf.add_edge(i, i + 1, DMatrix::identity(2, 2)).unwrap();
            sheaf.add_edge(i + 1, i, DMatrix::identity(2, 2)).unwrap();
        }
        sheaf
    }

    #[test]
    fn test_1hop_maps() {
        let sheaf = make_chain_sheaf();
        let mh = MultiHopSheaf::new(sheaf, 2).unwrap();
        // 1-hop maps should match original edges
        assert_eq!(mh.num_maps_at_hop(1), 8); // 4 edges × 2 directions
    }

    #[test]
    fn test_2hop_maps() {
        let sheaf = make_chain_sheaf();
        let mh = MultiHopSheaf::new(sheaf, 2).unwrap();
        // 2-hop: node 0→2, 1→3, 2→4, 4→2, 3→1, 2→0
        assert!(mh.num_maps_at_hop(2) > 0);
    }

    #[test]
    fn test_3hop_maps() {
        let sheaf = make_chain_sheaf();
        let mh = MultiHopSheaf::new(sheaf, 3).unwrap();
        assert!(mh.num_maps_at_hop(3) > 0);
    }

    #[test]
    fn test_2hop_identity_composition() {
        // For identity restriction maps, 2-hop composition should give identity
        let sheaf = make_chain_sheaf();
        let mh = MultiHopSheaf::new(sheaf, 2).unwrap();
        let map_02 = mh.restriction_map(0, 2, 2).unwrap();
        assert_relative_eq!(map_02[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(map_02[(1, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scaled_2hop() {
        let mut sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        let scale = DMatrix::from_diagonal(&DVector::from_element(2, 0.5));
        sheaf.add_edge(0, 1, scale.clone()).unwrap();
        sheaf.add_edge(1, 2, scale).unwrap();

        let mh = MultiHopSheaf::new(sheaf, 2).unwrap();
        let map_02 = mh.restriction_map(0, 2, 2).unwrap();
        // 0.5 * 0.5 = 0.25
        assert_relative_eq!(map_02[(0, 0)], 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_k_hop_laplacian() {
        let sheaf = make_chain_sheaf();
        let mh = MultiHopSheaf::new(sheaf, 2).unwrap();
        let lap = mh.k_hop_laplacian(1);
        assert!(lap.is_some());
        let lap = lap.unwrap();
        assert!(lap.is_psd());
    }

    #[test]
    fn test_multi_hop_message_passing() {
        let sheaf = make_chain_sheaf();
        let mh = MultiHopSheaf::new(sheaf, 2).unwrap();
        let features = DVector::from_element(10, 1.0);
        let weights = vec![1.0, 0.5];
        let result = mh.multi_hop_message_passing(&features, &weights);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_path_tracking() {
        let sheaf = make_chain_sheaf();
        let mh = MultiHopSheaf::new(sheaf, 3).unwrap();
        let path = mh.path(0, 3, 3);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path, &vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_k_hop_maps_from() {
        let sheaf = make_chain_sheaf();
        let mh = MultiHopSheaf::new(sheaf, 2).unwrap();
        let from_0 = mh.k_hop_maps_from(0, 1);
        assert_eq!(from_0.len(), 1); // Only edge 0→1
    }

    #[test]
    fn test_maps_to() {
        let sheaf = make_chain_sheaf();
        let mh = MultiHopSheaf::new(sheaf, 2).unwrap();
        let to_2 = mh.maps_to(2);
        assert!(to_2.len() > 0);
    }

    #[test]
    fn test_num_composed_maps() {
        let sheaf = make_chain_sheaf();
        let mh = MultiHopSheaf::new(sheaf, 2).unwrap();
        let total = mh.num_composed_maps();
        assert!(total > 8); // More than just the original edges
    }

    #[test]
    fn test_empty_k_hop_laplacian() {
        let sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        let mh = MultiHopSheaf::new(sheaf, 2).unwrap();
        let lap = mh.k_hop_laplacian(1);
        assert!(lap.is_none()); // No edges
    }

    #[test]
    fn test_cycle_graph_multihop() {
        let mut sheaf = CellularSheaf::new_uniform(4, 2).unwrap();
        for i in 0..4 {
            sheaf.add_edge(i, (i + 1) % 4, DMatrix::identity(2, 2)).unwrap();
        }
        let mh = MultiHopSheaf::new(sheaf, 2).unwrap();
        // In a cycle, 2-hop from node 0 should reach node 2
        let map_02 = mh.restriction_map(0, 2, 2);
        assert!(map_02.is_some());
    }
}
