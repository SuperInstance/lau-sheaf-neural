//! PLATO: Agent communication modeled as sheaf neural network.
//!
//! In the PLATO framework, each agent has a local state space (stalk) and
//! communication channels (edges with restriction maps). The sheaf structure
//! captures how agents translate their internal representations for others.
//!
//! Key ideas:
//! - Each agent is a node with a stalk (its internal state space)
//! - Communication between agents is via restriction maps (encoding/decoding)
//! - Sheaf diffusion = consensus/belief propagation
//! - Over-squashing = communication bottleneck between agents
//! - Sheaf attention = agents learn how to communicate

use crate::attention::{AttentionConfig, SheafAttention};
use crate::curvature::SheafCurvature;
use crate::diffusion::{Activation, SheafDiffusion};
use crate::laplacian::SheafLaplacian;
use crate::sheaf::{CellularSheaf, SheafBuilder, SheafError};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// An agent in the PLATO system.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlatoAgent {
    /// Unique agent identifier.
    pub id: String,
    /// Node index in the sheaf.
    pub node_index: usize,
    /// Internal state dimension.
    pub state_dim: usize,
    /// Current internal state.
    pub state: DVector<f64>,
    /// Agent role/capability description.
    pub role: String,
}

/// Communication channel between agents.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommunicationChannel {
    /// Source agent node index.
    pub source: usize,
    /// Target agent node index.
    pub target: usize,
    /// Channel type (e.g., "broadcast", "request", "response").
    pub channel_type: String,
    /// Custom restriction map for this channel.
    pub restriction_map: DMatrix<f64>,
}

/// PLATO configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlatoConfig {
    /// State dimension for all agents.
    pub state_dim: usize,
    /// Number of diffusion steps per round.
    pub diffusion_steps: usize,
    /// Diffusion time step.
    pub dt: f64,
    /// Whether to use learned restriction maps.
    pub learned_maps: bool,
    /// Over-squashing detection threshold.
    pub curvature_threshold: f64,
}

impl Default for PlatoConfig {
    fn default() -> Self {
        PlatoConfig {
            state_dim: 8,
            diffusion_steps: 5,
            dt: 0.1,
            learned_maps: false,
            curvature_threshold: -0.5,
        }
    }
}

/// Result of a PLATO communication round.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommunicationRoundResult {
    /// Updated agent states after diffusion.
    pub updated_states: Vec<DVector<f64>>,
    /// Dirichlet energy before communication.
    pub energy_before: f64,
    /// Dirichlet energy after communication.
    pub energy_after: f64,
    /// Number of bottleneck channels detected.
    pub num_bottlenecks: usize,
    /// Over-squashing score.
    pub over_squashing_score: f64,
}

/// The PLATO agent communication system.
#[derive(Clone, Debug)]
pub struct PlatoSystem {
    /// Configuration.
    pub config: PlatoConfig,
    /// Registered agents.
    pub agents: Vec<PlatoAgent>,
    /// Communication channels.
    pub channels: Vec<CommunicationChannel>,
    /// The underlying sheaf (rebuilt when agents/channels change).
    pub sheaf: Option<CellularSheaf>,
    /// Attention module for learned maps.
    pub attention: Option<SheafAttention>,
}

impl PlatoSystem {
    /// Create a new PLATO system.
    pub fn new(config: PlatoConfig) -> Self {
        let attention = if config.learned_maps {
            let attn_config = AttentionConfig::new(config.state_dim, config.state_dim);
            Some(SheafAttention::new(attn_config))
        } else {
            None
        };

        PlatoSystem {
            config,
            agents: Vec::new(),
            channels: Vec::new(),
            sheaf: None,
            attention,
        }
    }

    /// Register a new agent.
    pub fn register_agent(&mut self, id: &str, role: &str, initial_state: DVector<f64>) -> usize {
        let node_index = self.agents.len();
        self.agents.push(PlatoAgent {
            id: id.to_string(),
            node_index,
            state_dim: initial_state.len(),
            state: initial_state,
            role: role.to_string(),
        });
        node_index
    }

    /// Add a communication channel between agents.
    pub fn add_channel(
        &mut self,
        source: usize,
        target: usize,
        channel_type: &str,
        restriction_map: Option<DMatrix<f64>>,
    ) -> Result<(), SheafError> {
        if source >= self.agents.len() {
            return Err(SheafError::NodeNotFound(source));
        }
        if target >= self.agents.len() {
            return Err(SheafError::NodeNotFound(target));
        }

        let dim = self.config.state_dim;
        let map = restriction_map.unwrap_or_else(|| DMatrix::identity(dim, dim));

        self.channels.push(CommunicationChannel {
            source,
            target,
            channel_type: channel_type.to_string(),
            restriction_map: map,
        });

        Ok(())
    }

    /// Build the sheaf from current agents and channels.
    pub fn build_sheaf(&mut self) -> Result<(), SheafError> {
        let dim = self.config.state_dim;
        let mut builder = SheafBuilder::new(self.agents.len(), dim);

        for channel in &self.channels {
            builder = builder.edge_with_map(
                channel.source,
                channel.target,
                channel.restriction_map.clone(),
            );
        }

        self.sheaf = Some(builder.build()?);
        Ok(())
    }

    /// Run one round of communication (sheaf diffusion).
    pub fn communicate(&mut self) -> Result<CommunicationRoundResult, SheafError> {
        if self.sheaf.is_none() {
            self.build_sheaf()?;
        }

        let sheaf = self.sheaf.as_ref().unwrap();
        let diffusion = SheafDiffusion::new(sheaf.clone(), &[], Activation::Identity)?;

        // Build stacked state vector
        let dim = self.config.state_dim;
        let n = self.agents.len();
        let mut state = DVector::zeros(n * dim);
        for agent in &self.agents {
            let offset = agent.node_index * dim;
            for k in 0..agent.state.len().min(dim) {
                state[offset + k] = agent.state[k];
            }
        }

        // Compute energy before
        let laplacian = SheafLaplacian::from_sheaf(sheaf)?;
        let energy_before = laplacian.dirichlet_energy(&state);

        // Run diffusion
        let diffused = diffusion.diffuse(&state, self.config.dt, self.config.diffusion_steps);

        // Compute energy after
        let energy_after = laplacian.dirichlet_energy(&diffused);

        // Diagnose over-squashing
        let curvature = SheafCurvature::from_sheaf(sheaf, self.config.curvature_threshold)?;
        let num_bottlenecks = curvature.num_bottlenecks();
        let over_squashing_score = curvature.over_squashing_score();

        // Update agent states
        let mut updated_states = Vec::new();
        for agent in &mut self.agents {
            let offset = agent.node_index * dim;
            let mut new_state = DVector::zeros(dim);
            for k in 0..dim {
                new_state[k] = diffused[offset + k];
            }
            agent.state = new_state.clone();
            updated_states.push(new_state);
        }

        Ok(CommunicationRoundResult {
            updated_states,
            energy_before,
            energy_after,
            num_bottlenecks,
            over_squashing_score,
        })
    }

    /// Get the current state of an agent.
    pub fn agent_state(&self, id: &str) -> Option<&DVector<f64>> {
        self.agents.iter().find(|a| a.id == id).map(|a| &a.state)
    }

    /// Diagnose communication bottlenecks.
    pub fn diagnose(&self) -> Result<Option<Vec<(String, String, f64)>>, SheafError> {
        if let Some(ref sheaf) = self.sheaf {
            let curvature = SheafCurvature::from_sheaf(sheaf, self.config.curvature_threshold)?;
            let bottlenecks = curvature.bottleneck_edges();

            let result: Vec<(String, String, f64)> = bottlenecks
                .iter()
                .map(|e| {
                    let src = self.agents.get(e.source).map(|a| a.id.clone()).unwrap_or_default();
                    let tgt = self.agents.get(e.target).map(|a| a.id.clone()).unwrap_or_default();
                    (src, tgt, e.curvature)
                })
                .collect();

            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Inject a message from an external source into an agent.
    pub fn inject_message(&mut self, agent_id: &str, message: &DVector<f64>) -> Result<(), SheafError> {
        if let Some(agent) = self.agents.iter_mut().find(|a| a.id == agent_id) {
            if message.len() != agent.state.len() {
                return Err(SheafError::DimensionMismatch {
                    expected: agent.state.len(),
                    actual: message.len(),
                });
            }
            agent.state = message.clone();
            Ok(())
        } else {
            Err(SheafError::NodeNotFound(0))
        }
    }

    /// Number of registered agents.
    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }

    /// Number of communication channels.
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Compute consensus (average state across agents).
    pub fn consensus_state(&self) -> Option<DVector<f64>> {
        if self.agents.is_empty() {
            return None;
        }
        let dim = self.config.state_dim;
        let mut avg = DVector::zeros(dim);
        for agent in &self.agents {
            avg += &agent.state;
        }
        avg *= 1.0 / self.agents.len() as f64;
        Some(avg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_simple_plato() -> PlatoSystem {
        let config = PlatoConfig {
            state_dim: 4,
            diffusion_steps: 3,
            dt: 0.1,
            ..Default::default()
        };
        let mut system = PlatoSystem::new(config);

        system.register_agent("alice", "coordinator", DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0]));
        system.register_agent("bob", "worker", DVector::from_vec(vec![0.0, 1.0, 0.0, 0.0]));
        system.register_agent("carol", "analyst", DVector::from_vec(vec![0.0, 0.0, 1.0, 0.0]));

        system.add_channel(0, 1, "broadcast", None).unwrap();
        system.add_channel(1, 0, "response", None).unwrap();
        system.add_channel(1, 2, "broadcast", None).unwrap();
        system.add_channel(2, 0, "report", None).unwrap();

        system
    }

    #[test]
    fn test_plato_creation() {
        let system = make_simple_plato();
        assert_eq!(system.num_agents(), 3);
        assert_eq!(system.num_channels(), 4);
    }

    #[test]
    fn test_build_sheaf() {
        let mut system = make_simple_plato();
        system.build_sheaf().unwrap();
        assert!(system.sheaf.is_some());
        let sheaf = system.sheaf.unwrap();
        assert_eq!(sheaf.num_nodes, 3);
        assert_eq!(sheaf.num_edges(), 4);
    }

    #[test]
    fn test_communication_round() {
        let mut system = make_simple_plato();
        let result = system.communicate().unwrap();

        assert_eq!(result.updated_states.len(), 3);
        assert!(result.energy_after <= result.energy_before + 1e-6);
        assert!(result.over_squashing_score >= 0.0);
        assert!(result.over_squashing_score <= 1.0);
    }

    #[test]
    fn test_agent_state_access() {
        let system = make_simple_plato();
        let alice_state = system.agent_state("alice").unwrap();
        assert_eq!(alice_state.len(), 4);
        assert_relative_eq!(alice_state[0], 1.0);
    }

    #[test]
    fn test_nonexistent_agent() {
        let system = make_simple_plato();
        assert!(system.agent_state("dave").is_none());
    }

    #[test]
    fn test_diagnose() {
        let mut system = make_simple_plato();
        system.build_sheaf().unwrap();
        let diagnosis = system.diagnose().unwrap();
        assert!(diagnosis.is_some());
    }

    #[test]
    fn test_inject_message() {
        let mut system = make_simple_plato();
        let msg = DVector::from_vec(vec![5.0, 5.0, 5.0, 5.0]);
        system.inject_message("alice", &msg).unwrap();
        let state = system.agent_state("alice").unwrap();
        assert_relative_eq!(state[0], 5.0);
    }

    #[test]
    fn test_inject_wrong_dim() {
        let mut system = make_simple_plato();
        let msg = DVector::from_vec(vec![1.0, 2.0]);
        let result = system.inject_message("alice", &msg);
        assert!(result.is_err());
    }

    #[test]
    fn test_consensus() {
        let mut system = make_simple_plato();
        system.communicate().unwrap();
        let consensus = system.consensus_state().unwrap();
        assert_eq!(consensus.len(), 4);
    }

    #[test]
    fn test_consensus_empty() {
        let config = PlatoConfig::default();
        let system = PlatoSystem::new(config);
        assert!(system.consensus_state().is_none());
    }

    #[test]
    fn test_multiple_communication_rounds() {
        let mut system = make_simple_plato();
        for _ in 0..5 {
            let result = system.communicate().unwrap();
            assert_eq!(result.updated_states.len(), 3);
        }
    }

    #[test]
    fn test_custom_restriction_map() {
        let config = PlatoConfig {
            state_dim: 2,
            ..Default::default()
        };
        let mut system = PlatoSystem::new(config);
        system.register_agent("a", "test", DVector::from_vec(vec![1.0, 0.0]));
        system.register_agent("b", "test", DVector::from_vec(vec![0.0, 1.0]));

        let rotation = DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]); // 90° rotation
        system.add_channel(0, 1, "rotate", Some(rotation)).unwrap();

        let result = system.communicate().unwrap();
        assert_eq!(result.updated_states.len(), 2);
    }

    #[test]
    fn test_channel_type() {
        let mut system = make_simple_plato();
        assert_eq!(system.channels[0].channel_type, "broadcast");
        assert_eq!(system.channels[1].channel_type, "response");
    }

    #[test]
    fn test_over_squashing_detection() {
        // Create a bottleneck topology
        let config = PlatoConfig {
            state_dim: 2,
            curvature_threshold: 0.0, // Detect all negative curvature
            ..Default::default()
        };
        let mut system = PlatoSystem::new(config);

        // Two clusters with single bottleneck
        system.register_agent("a1", "cluster1", DVector::from_vec(vec![1.0, 0.0]));
        system.register_agent("a2", "cluster1", DVector::from_vec(vec![1.0, 0.0]));
        system.register_agent("b1", "cluster2", DVector::from_vec(vec![0.0, 1.0]));
        system.register_agent("b2", "cluster2", DVector::from_vec(vec![0.0, 1.0]));

        // Dense intra-cluster
        system.add_channel(0, 1, "intra", None).unwrap();
        system.add_channel(1, 0, "intra", None).unwrap();
        system.add_channel(2, 3, "intra", None).unwrap();
        system.add_channel(3, 2, "intra", None).unwrap();
        // Single bottleneck
        system.add_channel(1, 2, "bottleneck", None).unwrap();
        system.add_channel(2, 1, "bottleneck", None).unwrap();

        let result = system.communicate().unwrap();
        assert!(result.over_squashing_score > 0.0);
    }
}
