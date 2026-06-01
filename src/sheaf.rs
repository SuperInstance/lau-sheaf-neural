//! Cellular sheaf on a graph.
//!
//! A cellular sheaf assigns:
//! - A vector space (stalk) F(v) to each node v
//! - A linear map (restriction map) F_{v≺e} : F(v) → F(w) to each edge e = (v,w)
//!
//! The cochain complex C⁰(G, F) = ⊕_v F(v) is the space of node features.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors for sheaf operations.
#[derive(Error, Debug)]
pub enum SheafError {
    #[error("node {0} not found in sheaf")]
    NodeNotFound(usize),
    #[error("edge ({0}, {1}) not found in sheaf")]
    EdgeNotFound(usize, usize),
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("restriction map dimension error: source stalk dim {src_dim}, map rows {rows}")]
    RestrictionMapDimension { src_dim: usize, rows: usize },
    #[error("empty sheaf")]
    EmptySheaf,
    #[error("invalid stalk dimension: {0}")]
    InvalidStalkDimension(usize),
}

/// An edge in the sheaf graph, stored as (source, target) with a restriction map.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SheafEdge {
    /// Source node index.
    pub source: usize,
    /// Target node index.
    pub target: usize,
    /// Restriction map from source stalk to target stalk: F(source) → F(target).
    /// Shape: (target_dim × source_dim).
    pub restriction_map: DMatrix<f64>,
}

/// A cellular sheaf on a graph.
///
/// Type parameters:
/// - Each node `i` has a stalk of dimension `stalk_dims[i]`
/// - Each edge `(i,j)` has a restriction map `F(i) → F(j)`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CellularSheaf {
    /// Number of nodes.
    pub num_nodes: usize,
    /// Stalk dimension for each node.
    pub stalk_dims: Vec<usize>,
    /// Edges with their restriction maps.
    pub edges: Vec<SheafEdge>,
    /// Total feature dimension: sum of all stalk dims.
    pub total_dim: usize,
    /// Mapping from (source, target) to edge index.
    #[serde(skip)]
    pub edge_index: HashMap<(usize, usize), usize>,
}

impl CellularSheaf {
    /// Rebuild the edge index from edges (call after deserialization).
    pub fn rebuild_edge_index(&mut self) {
        self.edge_index.clear();
        for (idx, edge) in self.edges.iter().enumerate() {
            self.edge_index.insert((edge.source, edge.target), idx);
        }
    }

    /// Create a new cellular sheaf with uniform stalk dimension.
    ///
    /// All restriction maps are initialized to the identity (trivial sheaf).
    pub fn new_uniform(num_nodes: usize, stalk_dim: usize) -> Result<Self, SheafError> {
        if num_nodes == 0 {
            return Err(SheafError::EmptySheaf);
        }
        if stalk_dim == 0 {
            return Err(SheafError::InvalidStalkDimension(0));
        }
        let stalk_dims = vec![stalk_dim; num_nodes];
        Self::new(stalk_dims, &[])
    }

    /// Create a sheaf with given stalk dimensions and edges.
    ///
    /// Edges are specified as (source, target) pairs; restriction maps default to identity
    /// (only valid when source and target have the same stalk dimension).
    pub fn new(
        stalk_dims: Vec<usize>,
        edges: &[(usize, usize)],
    ) -> Result<Self, SheafError> {
        let num_nodes = stalk_dims.len();
        if num_nodes == 0 {
            return Err(SheafError::EmptySheaf);
        }
        for &d in &stalk_dims {
            if d == 0 {
                return Err(SheafError::InvalidStalkDimension(0));
            }
        }

        let total_dim: usize = stalk_dims.iter().sum();

        let mut sheaf_edges = Vec::new();
        let mut edge_index = HashMap::new();

        for &(s, t) in edges {
            if s >= num_nodes {
                return Err(SheafError::NodeNotFound(s));
            }
            if t >= num_nodes {
                return Err(SheafError::NodeNotFound(t));
            }
            let sd = stalk_dims[s];
            let td = stalk_dims[t];
            // For uniform stalk dims, use identity; otherwise use zero map
            let restriction_map = if sd == td {
                DMatrix::identity(sd, td)
            } else {
                DMatrix::zeros(td, sd)
            };
            let idx = sheaf_edges.len();
            edge_index.insert((s, t), idx);
            sheaf_edges.push(SheafEdge {
                source: s,
                target: t,
                restriction_map,
            });
        }

        Ok(CellularSheaf {
            num_nodes,
            stalk_dims,
            edges: sheaf_edges,
            total_dim,
            edge_index,
        })
    }

    /// Create a sheaf with given stalk dimensions, edges, and explicit restriction maps.
    pub fn with_restriction_maps(
        stalk_dims: Vec<usize>,
        edges: Vec<(usize, usize, DMatrix<f64>)>,
    ) -> Result<Self, SheafError> {
        let num_nodes = stalk_dims.len();
        if num_nodes == 0 {
            return Err(SheafError::EmptySheaf);
        }
        let total_dim: usize = stalk_dims.iter().sum();

        let mut sheaf_edges = Vec::new();
        let mut edge_index = HashMap::new();

        for (s, t, map) in edges {
            if s >= num_nodes {
                return Err(SheafError::NodeNotFound(s));
            }
            if t >= num_nodes {
                return Err(SheafError::NodeNotFound(t));
            }
            let expected_rows = stalk_dims[t];
            let expected_cols = stalk_dims[s];
            if map.nrows() != expected_rows || map.ncols() != expected_cols {
                return Err(SheafError::RestrictionMapDimension {
                    src_dim: expected_cols,
                    rows: map.nrows(),
                });
            }
            let idx = sheaf_edges.len();
            edge_index.insert((s, t), idx);
            sheaf_edges.push(SheafEdge {
                source: s,
                target: t,
                restriction_map: map,
            });
        }

        Ok(CellularSheaf {
            num_nodes,
            stalk_dims,
            edges: sheaf_edges,
            total_dim,
            edge_index,
        })
    }

    /// Get the stalk dimension at node `i`.
    pub fn stalk_dim(&self, node: usize) -> usize {
        self.stalk_dims[node]
    }

    /// Get the starting index of node `i` in the stacked feature vector.
    pub fn node_offset(&self, node: usize) -> usize {
        self.stalk_dims[..node].iter().sum()
    }

    /// Add an edge with a restriction map.
    pub fn add_edge(
        &mut self,
        source: usize,
        target: usize,
        restriction_map: DMatrix<f64>,
    ) -> Result<(), SheafError> {
        if source >= self.num_nodes {
            return Err(SheafError::NodeNotFound(source));
        }
        if target >= self.num_nodes {
            return Err(SheafError::NodeNotFound(target));
        }
        let idx = self.edges.len();
        self.edge_index.insert((source, target), idx);
        self.edges.push(SheafEdge {
            source,
            target,
            restriction_map,
        });
        Ok(())
    }

    /// Get the restriction map for edge (source, target).
    pub fn restriction_map(&self, source: usize, target: usize) -> Option<&DMatrix<f64>> {
        self.edge_index
            .get(&(source, target))
            .map(|&idx| &self.edges[idx].restriction_map)
    }

    /// Get a mutable reference to the restriction map for edge (source, target).
    pub fn restriction_map_mut(
        &mut self,
        source: usize,
        target: usize,
    ) -> Option<&mut DMatrix<f64>> {
        self.edge_index
            .get(&(source, target))
            .map(|&idx| &mut self.edges[idx].restriction_map)
    }

    /// Check if edge (source, target) exists.
    pub fn has_edge(&self, source: usize, target: usize) -> bool {
        self.edge_index.contains_key(&(source, target))
    }

    /// Get all neighbors of node `i` (nodes that `i` has edges to).
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|e| e.source == node)
            .map(|e| e.target)
            .collect()
    }

    /// Get all incoming neighbors of node `i`.
    pub fn in_neighbors(&self, node: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|e| e.target == node)
            .map(|e| e.source)
            .collect()
    }

    /// Get degree of node (number of outgoing edges).
    pub fn degree(&self, node: usize) -> usize {
        self.edges.iter().filter(|e| e.source == node).count()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Extract the stalk vector for node `i` from a stacked cochain vector.
    pub fn extract_stalk(&self, node: usize, cochain: &DVector<f64>) -> DVector<f64> {
        let offset = self.node_offset(node);
        let dim = self.stalk_dims[node];
        cochain.rows(offset, dim).into()
    }

    /// Set the stalk vector for node `i` in a stacked cochain vector.
    pub fn set_stalk(
        &self,
        node: usize,
        cochain: &mut DVector<f64>,
        value: &DVector<f64>,
    ) -> Result<(), SheafError> {
        let offset = self.node_offset(node);
        let dim = self.stalk_dims[node];
        if value.len() != dim {
            return Err(SheafError::DimensionMismatch {
                expected: dim,
                actual: value.len(),
            });
        }
        cochain.rows_mut(offset, dim).copy_from(value);
        Ok(())
    }

    /// Construct the coboundary map (gradient) δ⁰ as a sparse-like dense matrix.
    ///
    /// For each edge (i→j), the coboundary maps the stalk at i to the stalk at j
    /// via the restriction map. The full coboundary δ: C⁰ → C¹ has dimensions
    /// (num_edges × stalk_dims[target]) rows and total_dim columns.
    ///
    /// Actually we build the transpose for the Laplacian construction.
    /// Returns the matrix B such that L_Σ = B^T B is the sheaf Laplacian.
    pub fn coboundary_matrix(&self) -> DMatrix<f64> {
        // Each edge (i,j) contributes a block row:
        // row block e = (i,j): [ 0 ... -R_{ij} ... 0 ... I_{dj} ... 0 ]
        // where R_{ij} is at column offset of node i, I is at offset of node j
        let num_edge_components: usize = self
            .edges
            .iter()
            .map(|e| self.stalk_dims[e.target])
            .sum();

        let mut b = DMatrix::zeros(num_edge_components, self.total_dim);

        let mut row_offset = 0;
        for edge in &self.edges {
            let dt = self.stalk_dims[edge.target];
            let ds = self.stalk_dims[edge.source];
            let s_off = self.node_offset(edge.source);
            let t_off = self.node_offset(edge.target);

            // -R_{ij} block
            for r in 0..dt {
                for c in 0..ds {
                    b[(row_offset + r, s_off + c)] -= edge.restriction_map[(r, c)];
                }
            }
            // I block at target
            for r in 0..dt {
                b[(row_offset + r, t_off + r)] += 1.0;
            }
            row_offset += dt;
        }

        b
    }

    /// Build the adjacency list representation.
    pub fn adjacency(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); self.num_nodes];
        for edge in &self.edges {
            adj[edge.source].push(edge.target);
        }
        adj
    }

    /// Validate the sheaf: check all dimensions are consistent.
    pub fn validate(&self) -> Result<(), SheafError> {
        for edge in &self.edges {
            let expected_rows = self.stalk_dims[edge.target];
            let expected_cols = self.stalk_dims[edge.source];
            if edge.restriction_map.nrows() != expected_rows
                || edge.restriction_map.ncols() != expected_cols
            {
                return Err(SheafError::RestrictionMapDimension {
                    src_dim: expected_cols,
                    rows: edge.restriction_map.nrows(),
                });
            }
        }
        Ok(())
    }
}

/// Builder for constructing cellular sheaves incrementally.
pub struct SheafBuilder {
    stalk_dims: Vec<usize>,
    edges: Vec<(usize, usize, DMatrix<f64>)>,
}

impl SheafBuilder {
    /// Start building a sheaf with `num_nodes` nodes, all with dimension `stalk_dim`.
    pub fn new(num_nodes: usize, stalk_dim: usize) -> Self {
        SheafBuilder {
            stalk_dims: vec![stalk_dim; num_nodes],
            edges: Vec::new(),
        }
    }

    /// Set the stalk dimension for a specific node.
    pub fn stalk_dim(mut self, node: usize, dim: usize) -> Self {
        self.stalk_dims[node] = dim;
        self
    }

    /// Add an edge with an identity restriction map.
    pub fn edge(mut self, source: usize, target: usize) -> Self {
        let sd = self.stalk_dims[source];
        let td = self.stalk_dims[target];
        let map = if sd == td {
            DMatrix::identity(td, sd)
        } else {
            DMatrix::zeros(td, sd)
        };
        self.edges.push((source, target, map));
        self
    }

    /// Add an edge with a custom restriction map.
    pub fn edge_with_map(
        mut self,
        source: usize,
        target: usize,
        map: DMatrix<f64>,
    ) -> Self {
        self.edges.push((source, target, map));
        self
    }

    /// Build the cellular sheaf.
    pub fn build(self) -> Result<CellularSheaf, SheafError> {
        CellularSheaf::with_restriction_maps(self.stalk_dims, self.edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_uniform_sheaf_creation() {
        let sheaf = CellularSheaf::new_uniform(5, 3).unwrap();
        assert_eq!(sheaf.num_nodes, 5);
        assert_eq!(sheaf.total_dim, 15);
        assert_eq!(sheaf.stalk_dims, vec![3; 5]);
    }

    #[test]
    fn test_empty_sheaf_error() {
        let result = CellularSheaf::new_uniform(0, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_stalk_dim_error() {
        let result = CellularSheaf::new_uniform(3, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_edges() {
        let mut sheaf = CellularSheaf::new_uniform(4, 2).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(2, 3, DMatrix::identity(2, 2)).unwrap();
        assert_eq!(sheaf.num_edges(), 3);
        assert!(sheaf.has_edge(0, 1));
        assert!(!sheaf.has_edge(3, 0));
    }

    #[test]
    fn test_restriction_map() {
        let mut sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        let custom_map = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        sheaf.add_edge(0, 1, custom_map.clone()).unwrap();
        let rm = sheaf.restriction_map(0, 1).unwrap();
        assert_relative_eq!(rm[(0, 0)], 1.0);
        assert_relative_eq!(rm[(0, 1)], 2.0);
        assert_relative_eq!(rm[(1, 0)], 3.0);
        assert_relative_eq!(rm[(1, 1)], 4.0);
    }

    #[test]
    fn test_neighbors() {
        let mut sheaf = CellularSheaf::new_uniform(4, 2).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(0, 2, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(1, 3, DMatrix::identity(2, 2)).unwrap();
        let neighbors = sheaf.neighbors(0);
        assert_eq!(neighbors, vec![1, 2]);
    }

    #[test]
    fn test_degree() {
        let mut sheaf = CellularSheaf::new_uniform(4, 2).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(0, 2, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(1, 3, DMatrix::identity(2, 2)).unwrap();
        assert_eq!(sheaf.degree(0), 2);
        assert_eq!(sheaf.degree(1), 1);
        assert_eq!(sheaf.degree(2), 0);
    }

    #[test]
    fn test_extract_set_stalk() {
        let mut sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        let mut cochain = DVector::zeros(6);
        let val = DVector::from_vec(vec![3.0, 4.0]);
        sheaf.set_stalk(1, &mut cochain, &val).unwrap();
        let extracted = sheaf.extract_stalk(1, &cochain);
        assert_relative_eq!(extracted[(0, 0)], 3.0);
        assert_relative_eq!(extracted[(1, 0)], 4.0);
    }

    #[test]
    fn test_node_offset() {
        let sheaf = CellularSheaf::new_uniform(4, 3).unwrap();
        assert_eq!(sheaf.node_offset(0), 0);
        assert_eq!(sheaf.node_offset(1), 3);
        assert_eq!(sheaf.node_offset(2), 6);
        assert_eq!(sheaf.node_offset(3), 9);
    }

    #[test]
    fn test_heterogeneous_stalk_dims() {
        let dims = vec![2, 3, 4];
        let mut sheaf = CellularSheaf::new(dims, &[(0, 1), (1, 2)]).unwrap();
        assert_eq!(sheaf.total_dim, 9);
        assert_eq!(sheaf.stalk_dim(0), 2);
        assert_eq!(sheaf.stalk_dim(1), 3);
        assert_eq!(sheaf.stalk_dim(2), 4);
    }

    #[test]
    fn test_validate() {
        let mut sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(2, 2)).unwrap();
        assert!(sheaf.validate().is_ok());
    }

    #[test]
    fn test_builder_pattern() {
        let sheaf = SheafBuilder::new(4, 3)
            .edge(0, 1)
            .edge(1, 2)
            .edge(2, 3)
            .edge(3, 0)
            .build()
            .unwrap();
        assert_eq!(sheaf.num_nodes, 4);
        assert_eq!(sheaf.num_edges(), 4);
        assert_eq!(sheaf.total_dim, 12);
    }

    #[test]
    fn test_builder_custom_map() {
        let map = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]);
        let sheaf = SheafBuilder::new(3, 2)
            .edge_with_map(0, 1, map.clone())
            .build()
            .unwrap();
        let rm = sheaf.restriction_map(0, 1).unwrap();
        assert_relative_eq!(rm[(0, 0)], 0.5);
    }

    #[test]
    fn test_invalid_node_error() {
        let mut sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        let result = sheaf.add_edge(0, 5, DMatrix::identity(2, 2));
        assert!(result.is_err());
    }

    #[test]
    fn test_coboundary_matrix_trivial() {
        // 2 nodes, identity restriction map
        let mut sheaf = CellularSheaf::new_uniform(2, 2).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(2, 2)).unwrap();
        let b = sheaf.coboundary_matrix();
        // Edge (0,1): [-I | I]
        // Row 0: [-1, 0, 1, 0]
        // Row 1: [0, -1, 0, 1]
        assert_eq!(b.nrows(), 2);
        assert_eq!(b.ncols(), 4);
        assert_relative_eq!(b[(0, 0)], -1.0);
        assert_relative_eq!(b[(0, 2)], 1.0);
        assert_relative_eq!(b[(1, 1)], -1.0);
        assert_relative_eq!(b[(1, 3)], 1.0);
    }

    #[test]
    fn test_in_neighbors() {
        let mut sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        sheaf.add_edge(0, 2, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(1, 2, DMatrix::identity(2, 2)).unwrap();
        let inn = sheaf.in_neighbors(2);
        assert_eq!(inn, vec![0, 1]);
    }

    #[test]
    fn test_serialization() {
        let mut sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(2, 2)).unwrap();
        let json = serde_json::to_string(&sheaf).unwrap();
        let mut deserialized: CellularSheaf = serde_json::from_str(&json).unwrap();
        deserialized.rebuild_edge_index();
        assert_eq!(deserialized.num_nodes, 3);
        assert_eq!(deserialized.num_edges(), 1);
        assert!(deserialized.has_edge(0, 1));
    }

    #[test]
    fn test_adjacency() {
        let mut sheaf = CellularSheaf::new_uniform(4, 2).unwrap();
        sheaf.add_edge(0, 1, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(0, 2, DMatrix::identity(2, 2)).unwrap();
        sheaf.add_edge(2, 3, DMatrix::identity(2, 2)).unwrap();
        let adj = sheaf.adjacency();
        assert_eq!(adj[0], vec![1, 2]);
        assert_eq!(adj[1], Vec::<usize>::new());
        assert_eq!(adj[2], vec![3]);
    }
}
