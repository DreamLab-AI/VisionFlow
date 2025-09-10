//! Constraint Actor - Handles constraint management and updates

use actix::prelude::*;
use log::{debug, error, info, warn};

use crate::actors::messages::*;
use crate::models::constraints::{Constraint, ConstraintSet, ConstraintData};
use super::shared::{SharedGPUContext, GPUState};

/// Constraint Actor - handles constraint management and GPU uploads
pub struct ConstraintActor {
    /// Shared GPU state
    gpu_state: GPUState,
    
    /// Shared GPU context reference
    shared_context: Option<SharedGPUContext>,
    
    /// Current constraint set
    constraints: Vec<Constraint>,
}

impl ConstraintActor {
    pub fn new() -> Self {
        Self {
            gpu_state: GPUState::default(),
            shared_context: None,
            constraints: Vec::new(),
        }
    }
    
    /// Update constraints and upload to GPU
    fn update_constraints(&mut self, new_constraints: Vec<Constraint>) -> Result<(), String> {
        info!("ConstraintActor: Updating constraints - {} current, {} new", 
              self.constraints.len(), new_constraints.len());
        
        // Update local constraint storage
        self.constraints = new_constraints;
        
        // Upload constraints to GPU if available
        if self.shared_context.is_some() {
            self.upload_constraints_to_gpu()?;
        } else {
            info!("ConstraintActor: GPU not initialized, constraints stored locally");
        }
        
        info!("ConstraintActor: Constraint update completed - {} total constraints", 
              self.constraints.len());
        Ok(())
    }
    
    /// Upload constraints to GPU
    fn upload_constraints_to_gpu(&self) -> Result<(), String> {
        info!("ConstraintActor: Uploading {} constraints to GPU", self.constraints.len());
        
        let mut unified_compute = match &self.shared_context {
            Some(ctx) => {
                ctx.unified_compute.lock()
                    .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?
            },
            None => {
                return Err("GPU context not initialized".to_string());
            }
        };
        
        // Convert constraints to GPU format
        let constraint_data = self.convert_constraints_to_gpu_format()?;
        
        if constraint_data.is_empty() {
            info!("ConstraintActor: No constraints to upload, clearing GPU constraints");
            unified_compute.clear_constraints()
                .map_err(|e| format!("Failed to clear GPU constraints: {}", e))?;
        } else {
            // Upload constraint data to GPU
            unified_compute.upload_constraints(&constraint_data)
                .map_err(|e| format!("Failed to upload constraints to GPU: {}", e))?;
            
            info!("ConstraintActor: Successfully uploaded {} constraint entries to GPU", 
                  constraint_data.len());
        }
        
        Ok(())
    }
    
    /// Convert constraints to GPU-compatible format
    fn convert_constraints_to_gpu_format(&self) -> Result<Vec<ConstraintData>, String> {
        let mut constraint_data = Vec::new();
        
        for (_constraint_idx, _constraint) in self.constraints.iter().enumerate() {
            // TODO: Fix constraint enum definitions - all constraint processing disabled for now
            /*
            match constraint {
                Constraint::Distance { node1_id, node2_id, target_distance, strength } => {
                    // Find node indices from IDs
                    let node1_idx = self.gpu_state.node_indices.get(node1_id)
                        .copied()
                        .ok_or_else(|| format!("Node {} not found in node indices", node1_id))? as u32;
                    
                    let node2_idx = self.gpu_state.node_indices.get(node2_id)
                        .copied()
                        .ok_or_else(|| format!("Node {} not found in node indices", node2_id))? as u32;
                    
                    constraint_data.push(ConstraintData {
                        constraint_type: 0, // Distance constraint type
                        node1_idx,
                        node2_idx,
                        target_value: *target_distance,
                        strength: *strength,
                        active: true,
                    });
                },
                
                Constraint::Angle { node1_id, node2_id, node3_id, target_angle, strength } => {
                    // Find node indices from IDs
                    let node1_idx = self.gpu_state.node_indices.get(node1_id)
                        .copied()
                        .ok_or_else(|| format!("Node {} not found in node indices", node1_id))? as u32;
                    
                    let node2_idx = self.gpu_state.node_indices.get(node2_id)
                        .copied()
                        .ok_or_else(|| format!("Node {} not found in node indices", node2_id))? as u32;
                    
                    let node3_idx = self.gpu_state.node_indices.get(node3_id)
                        .copied()
                        .ok_or_else(|| format!("Node {} not found in node indices", node3_id))? as u32;
                    
                    constraint_data.push(ConstraintData {
                        constraint_type: 1, // Angle constraint type
                        node1_idx,
                        node2_idx, // Center node for angle
                        target_value: target_angle.to_radians(), // Convert to radians for GPU
                        strength: *strength,
                        active: true,
                    });
                    
                    // For angle constraints, we also need the third node information
                    // This could be encoded as additional constraint data or handled differently
                    // For now, we'll create a second entry for the third node
                    constraint_data.push(ConstraintData {
                        constraint_type: 1, // Angle constraint type continuation
                        node1_idx: node3_idx,
                        node2_idx: constraint_idx as u32, // Link back to main constraint
                        target_value: 0.0, // Unused for continuation entry
                        strength: 0.0, // Unused for continuation entry
                        active: true,
                    });
                },
                
                Constraint::Position { node_id, target_position, strength } => {
                    // Find node index from ID
                    let node_idx = self.gpu_state.node_indices.get(node_id)
                        .copied()
                        .ok_or_else(|| format!("Node {} not found in node indices", node_id))? as u32;
                    
                    // Create three constraint entries for x, y, z components
                    constraint_data.push(ConstraintData {
                        constraint_type: 2, // Position X constraint type
                        node1_idx: node_idx,
                        node2_idx: 0, // Unused for position constraints
                        target_value: target_position.x,
                        strength: *strength,
                        active: true,
                    });
                    
                    constraint_data.push(ConstraintData {
                        constraint_type: 3, // Position Y constraint type
                        node1_idx: node_idx,
                        node2_idx: 0, // Unused for position constraints
                        target_value: target_position.y,
                        strength: *strength,
                        active: true,
                    });
                    
                    constraint_data.push(ConstraintData {
                        constraint_type: 4, // Position Z constraint type
                        node1_idx: node_idx,
                        node2_idx: 0, // Unused for position constraints
                        target_value: target_position.z,
                        strength: *strength,
                        active: true,
                    });
                },
                
                Constraint::Cluster { node_ids, center_position, strength } => {
                    // For cluster constraints, create a center attraction for each node
                    for node_id in node_ids {
                        let node_idx = self.gpu_state.node_indices.get(node_id)
                            .copied()
                            .ok_or_else(|| format!("Node {} not found in node indices", node_id))? as u32;
                        
                        // Create three constraint entries for x, y, z components of cluster center attraction
                        constraint_data.push(ConstraintData {
                            constraint_type: 5, // Cluster X attraction type
                            node1_idx: node_idx,
                            node2_idx: 0, // Unused for cluster constraints
                            target_value: center_position.x,
                            strength: *strength,
                            active: true,
                        });
                        
                        constraint_data.push(ConstraintData {
                            constraint_type: 6, // Cluster Y attraction type
                            node1_idx: node_idx,
                            node2_idx: 0, // Unused for cluster constraints
                            target_value: center_position.y,
                            strength: *strength,
                            active: true,
                        });
                        
                        constraint_data.push(ConstraintData {
                            constraint_type: 7, // Cluster Z attraction type
                            node1_idx: node_idx,
                            node2_idx: 0, // Unused for cluster constraints
                            target_value: center_position.z,
                            strength: *strength,
                            active: true,
                        });
                    }
                },
            }
            */
        }
        
        info!("ConstraintActor: Converted {} constraints to {} GPU constraint entries", 
              self.constraints.len(), constraint_data.len());
        
        Ok(constraint_data)
    }
    
    /// Get current constraint set
    fn get_current_constraints(&self) -> ConstraintSet {
        ConstraintSet {
            constraints: self.constraints.clone(),
            groups: std::collections::HashMap::new(), // Add missing groups field
        }
    }
    
    /// Clear all constraints
    fn clear_constraints(&mut self) -> Result<(), String> {
        info!("ConstraintActor: Clearing all constraints");
        
        self.constraints.clear();
        
        // Clear constraints on GPU if available
        if let Some(ctx) = &self.shared_context {
            let mut unified_compute = ctx.unified_compute.lock()
                .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?;
            
            unified_compute.clear_constraints()
                .map_err(|e| format!("Failed to clear GPU constraints: {}", e))?;
            
            info!("ConstraintActor: GPU constraints cleared");
        }
        
        info!("ConstraintActor: All constraints cleared successfully");
        Ok(())
    }
    
    /// Get constraint statistics
    fn get_constraint_statistics(&self) -> ConstraintStatistics {
        let stats = ConstraintStatistics {
            total_constraints: self.constraints.len(),
            distance_constraints: 0,
            angle_constraints: 0,
            position_constraints: 0,
            cluster_constraints: 0,
            active_constraints: self.constraints.len(), // All are active by default
        };
        
        // TODO: Fix constraint enum definitions
        /*
        for constraint in &self.constraints {
            match constraint {
                Constraint::Distance { .. } => stats.distance_constraints += 1,
                Constraint::Angle { .. } => stats.angle_constraints += 1,
                Constraint::Position { .. } => stats.position_constraints += 1,
                Constraint::Cluster { node_ids, .. } => {
                    stats.cluster_constraints += 1;
                    // Each cluster constraint affects multiple nodes
                    stats.total_constraints += node_ids.len().saturating_sub(1);
                },
            }
        }
        */
        
        stats
    }
}

impl Actor for ConstraintActor {
    type Context = Context<Self>;
    
    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("Constraint Actor started");
    }
    
    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("Constraint Actor stopped");
    }
}

// === Message Handlers ===

impl Handler<UpdateConstraints> for ConstraintActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: UpdateConstraints, _ctx: &mut Self::Context) -> Self::Result {
        info!("ConstraintActor: UpdateConstraints received");
        // Convert Value to Vec<Constraint> - placeholder implementation
        let constraints: Vec<Constraint> = Vec::new(); // TODO: Parse from msg.constraint_data
        self.update_constraints(constraints)
    }
}

impl Handler<GetConstraints> for ConstraintActor {
    type Result = Result<ConstraintSet, String>;
    
    fn handle(&mut self, _msg: GetConstraints, _ctx: &mut Self::Context) -> Self::Result {
        debug!("ConstraintActor: GetConstraints request");
        Ok(self.get_current_constraints())
    }
}

impl Handler<UploadConstraintsToGPU> for ConstraintActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: UploadConstraintsToGPU, _ctx: &mut Self::Context) -> Self::Result {
        info!("ConstraintActor: UploadConstraintsToGPU received - {} constraint entries", 
              msg.constraint_data.len());
        
        let mut unified_compute = match &self.shared_context {
            Some(ctx) => {
                ctx.unified_compute.lock()
                    .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?
            },
            None => {
                return Err("GPU context not initialized".to_string());
            }
        };
        
        // Upload the provided constraint data directly
        unified_compute.upload_constraints(&msg.constraint_data)
            .map_err(|e| format!("Failed to upload constraints to GPU: {}", e))?;
        
        info!("ConstraintActor: Successfully uploaded {} constraint entries to GPU", 
              msg.constraint_data.len());
        Ok(())
    }
}

// Custom message handlers for constraint management
impl Handler<ClearConstraints> for ConstraintActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, _msg: ClearConstraints, _ctx: &mut Self::Context) -> Self::Result {
        self.clear_constraints()
    }
}

impl Handler<GetConstraintStatistics> for ConstraintActor {
    type Result = Result<ConstraintStatistics, String>;
    
    fn handle(&mut self, _msg: GetConstraintStatistics, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.get_constraint_statistics())
    }
}

// Custom messages for constraint management
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ClearConstraints;

#[derive(Message)]
#[rtype(result = "Result<ConstraintStatistics, String>")]
pub struct GetConstraintStatistics;

// Constraint statistics structure
#[derive(Debug, Clone)]
pub struct ConstraintStatistics {
    pub total_constraints: usize,
    pub distance_constraints: usize,
    pub angle_constraints: usize,
    pub position_constraints: usize,
    pub cluster_constraints: usize,
    pub active_constraints: usize,
}