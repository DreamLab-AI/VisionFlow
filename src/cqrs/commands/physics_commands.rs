// src/cqrs/commands/physics_commands.rs
//! GPU Physics Commands
//!
//! Write operations for GPU physics adapter.

use crate::cqrs::types::{Command, Result};
use crate::models::graph::GraphData;
use crate::ports::gpu_physics_adapter::PhysicsParameters;
use std::sync::Arc;

///
#[derive(Debug, Clone)]
pub struct InitializePhysicsCommand {
    pub graph: Arc<GraphData>,
    pub params: PhysicsParameters,
}

impl Command for InitializePhysicsCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "InitializePhysics"
    }

    fn validate(&self) -> Result<()> {
        if self.params.time_step <= 0.0 {
            return Err(anyhow::anyhow!("Time step must be positive"));
        }
        if self.params.damping < 0.0 || self.params.damping > 1.0 {
            return Err(anyhow::anyhow!("Damping must be between 0 and 1"));
        }
        if self.params.max_velocity <= 0.0 {
            return Err(anyhow::anyhow!("Max velocity must be positive"));
        }
        if self.params.convergence_threshold <= 0.0 {
            return Err(anyhow::anyhow!("Convergence threshold must be positive"));
        }
        Ok(())
    }
}

///
#[derive(Debug, Clone)]
pub struct UpdatePhysicsParametersCommand {
    pub params: PhysicsParameters,
}

impl Command for UpdatePhysicsParametersCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "UpdatePhysicsParameters"
    }

    fn validate(&self) -> Result<()> {
        if self.params.time_step <= 0.0 {
            return Err(anyhow::anyhow!("Time step must be positive"));
        }
        if self.params.damping < 0.0 || self.params.damping > 1.0 {
            return Err(anyhow::anyhow!("Damping must be between 0 and 1"));
        }
        Ok(())
    }
}

///
#[derive(Debug, Clone)]
pub struct UpdateGraphDataCommand {
    pub graph: Arc<GraphData>,
}

impl Command for UpdateGraphDataCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "UpdateGraphData"
    }
}

///
#[derive(Debug, Clone)]
pub struct ApplyExternalForcesCommand {
    pub forces: Vec<(u32, f32, f32, f32)>, 
}

impl Command for ApplyExternalForcesCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "ApplyExternalForces"
    }

    fn validate(&self) -> Result<()> {
        if self.forces.is_empty() {
            return Err(anyhow::anyhow!("Must provide at least one force"));
        }
        for (node_id, fx, fy, fz) in &self.forces {
            if fx.is_nan() || fy.is_nan() || fz.is_nan() {
                return Err(anyhow::anyhow!(
                    "Force components cannot be NaN for node {}",
                    node_id
                ));
            }
            if fx.is_infinite() || fy.is_infinite() || fz.is_infinite() {
                return Err(anyhow::anyhow!(
                    "Force components cannot be infinite for node {}",
                    node_id
                ));
            }
        }
        Ok(())
    }
}

///
#[derive(Debug, Clone)]
pub struct PinNodesCommand {
    pub nodes: Vec<(u32, f32, f32, f32)>, 
}

impl Command for PinNodesCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "PinNodes"
    }

    fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(anyhow::anyhow!("Must provide at least one node to pin"));
        }
        for (node_id, x, y, z) in &self.nodes {
            if x.is_nan() || y.is_nan() || z.is_nan() {
                return Err(anyhow::anyhow!(
                    "Position coordinates cannot be NaN for node {}",
                    node_id
                ));
            }
            if x.is_infinite() || y.is_infinite() || z.is_infinite() {
                return Err(anyhow::anyhow!(
                    "Position coordinates cannot be infinite for node {}",
                    node_id
                ));
            }
        }
        Ok(())
    }
}

///
#[derive(Debug, Clone)]
pub struct UnpinNodesCommand {
    pub node_ids: Vec<u32>,
}

impl Command for UnpinNodesCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "UnpinNodes"
    }

    fn validate(&self) -> Result<()> {
        if self.node_ids.is_empty() {
            return Err(anyhow::anyhow!("Must provide at least one node ID"));
        }
        Ok(())
    }
}

///
#[derive(Debug, Clone)]
pub struct ResetPhysicsCommand;

impl Command for ResetPhysicsCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "ResetPhysics"
    }
}

///
#[derive(Debug, Clone)]
pub struct CleanupPhysicsCommand;

impl Command for CleanupPhysicsCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "CleanupPhysics"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_physics_validation() {
        let graph = Arc::new(GraphData::default());
        let params = PhysicsParameters::default();
        let cmd = InitializePhysicsCommand { graph, params };
        assert!(cmd.validate().is_ok());

        let graph = Arc::new(GraphData::default());
        let mut params = PhysicsParameters::default();
        params.time_step = -1.0;
        let cmd = InitializePhysicsCommand { graph, params };
        assert!(cmd.validate().is_err());
    }

    #[test]
    fn test_apply_forces_validation() {
        let cmd = ApplyExternalForcesCommand {
            forces: vec![(1, 1.0, 2.0, 3.0)],
        };
        assert!(cmd.validate().is_ok());

        let cmd = ApplyExternalForcesCommand {
            forces: vec![(1, f32::NAN, 2.0, 3.0)],
        };
        assert!(cmd.validate().is_err());

        let cmd = ApplyExternalForcesCommand { forces: vec![] };
        assert!(cmd.validate().is_err());
    }

    #[test]
    fn test_pin_nodes_validation() {
        let cmd = PinNodesCommand {
            nodes: vec![(1, 0.0, 0.0, 0.0)],
        };
        assert!(cmd.validate().is_ok());

        let cmd = PinNodesCommand {
            nodes: vec![(1, f32::INFINITY, 0.0, 0.0)],
        };
        assert!(cmd.validate().is_err());
    }
}
