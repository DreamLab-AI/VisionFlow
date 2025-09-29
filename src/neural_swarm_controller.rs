//! Neural-enhanced swarm controller with codex-syntaptic integration
//! Provides mesh networks, swarm intelligence, and cognitive coordination

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use anyhow::{Result, Context};
use tracing::{info, warn, error, debug};
use nalgebra::{Vector3, Matrix3};

use crate::neural_actor_system::{NeuralActor, CognitivePattern, NeuralActorSystem};
use crate::neural_memory::{NeuralMemory, MemoryType, ExperienceData};
use crate::neural_consensus::{NeuralConsensus, ConsensusResult};
use crate::gpu::compute_service::ComputeService;

/// Neural swarm topology types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmTopology {
    Mesh {
        connectivity: f32,
        redundancy: u32,
    },
    Hierarchical {
        levels: u32,
        branching_factor: u32,
    },
    Ring {
        bidirectional: bool,
        cluster_size: u32,
    },
    Star {
        hub_capacity: u32,
        failover_hubs: Vec<Uuid>,
    },
    Adaptive {
        base_topology: Box<SwarmTopology>,
        adaptation_rate: f32,
        performance_threshold: f32,
    },
}

/// Neural swarm agent with cognitive capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralSwarmAgent {
    pub id: Uuid,
    pub role: AgentRole,
    pub cognitive_pattern: CognitivePattern,
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub connections: HashSet<Uuid>,
    pub neural_state: NeuralState,
    pub last_activity: DateTime<Utc>,
    pub performance_metrics: PerformanceMetrics,
    pub capabilities: Vec<String>,
    pub workload: f32,
    pub trust_score: f32,
}

/// Agent roles in the neural swarm
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AgentRole {
    Coordinator,
    Researcher,
    Coder,
    Analyzer,
    Optimizer,
    Validator,
    Monitor,
    Specialist(String),
}

/// Neural state of an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralState {
    pub activation_level: f32,
    pub cognitive_load: f32,
    pub learning_rate: f32,
    pub attention_weights: HashMap<String, f32>,
    pub memory_utilization: f32,
    pub neural_connections: u32,
    pub synaptic_strength: f32,
}

/// Performance metrics for neural agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub task_completion_rate: f32,
    pub response_time: f32,
    pub accuracy_score: f32,
    pub collaboration_score: f32,
    pub innovation_index: f32,
    pub energy_efficiency: f32,
    pub adaptation_speed: f32,
}

/// Swarm intelligence patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmPattern {
    Flocking {
        separation_weight: f32,
        alignment_weight: f32,
        cohesion_weight: f32,
    },
    Foraging {
        exploration_bias: f32,
        exploitation_bias: f32,
        pheromone_decay: f32,
    },
    Clustering {
        cluster_radius: f32,
        min_cluster_size: u32,
        merge_threshold: f32,
    },
    Emergent {
        emergence_threshold: f32,
        pattern_stability: f32,
        collective_memory: bool,
    },
}

/// Neural swarm task with cognitive requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralSwarmTask {
    pub id: Uuid,
    pub description: String,
    pub cognitive_requirements: Vec<CognitivePattern>,
    pub priority: TaskPriority,
    pub complexity: f32,
    pub estimated_duration: chrono::Duration,
    pub required_agents: u32,
    pub dependencies: Vec<Uuid>,
    pub neural_constraints: NeuralConstraints,
    pub collaboration_type: CollaborationType,
}

/// Task priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

/// Neural constraints for tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConstraints {
    pub min_activation_level: f32,
    pub max_cognitive_load: f32,
    pub required_trust_score: f32,
    pub neural_synchronization: bool,
    pub collective_intelligence: bool,
}

/// Collaboration types for neural tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollaborationType {
    Independent,
    Sequential,
    Parallel,
    Hierarchical,
    Mesh,
    Swarm,
}

/// Neural swarm controller configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralSwarmConfig {
    pub max_agents: u32,
    pub topology: SwarmTopology,
    pub swarm_pattern: SwarmPattern,
    pub cognitive_diversity: f32,
    pub neural_plasticity: f32,
    pub learning_rate: f32,
    pub adaptation_threshold: f32,
    pub consensus_threshold: f32,
    pub memory_retention: f32,
    pub gpu_acceleration: bool,
}

/// Main neural swarm controller
#[derive(Debug)]
pub struct NeuralSwarmController {
    pub id: Uuid,
    pub config: NeuralSwarmConfig,
    pub agents: Arc<RwLock<HashMap<Uuid, NeuralSwarmAgent>>>,
    pub tasks: Arc<RwLock<HashMap<Uuid, NeuralSwarmTask>>>,
    pub neural_memory: Arc<NeuralMemory>,
    pub neural_consensus: Arc<NeuralConsensus>,
    pub actor_system: Arc<NeuralActorSystem>,
    pub compute_service: Option<Arc<ComputeService>>,
    pub swarm_metrics: Arc<RwLock<SwarmMetrics>>,
    pub topology_matrix: Arc<RwLock<Matrix3<f32>>>,
    pub active_patterns: Arc<RwLock<HashMap<String, SwarmPattern>>>,
}

/// Swarm-level metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmMetrics {
    pub collective_intelligence: f32,
    pub swarm_coherence: f32,
    pub task_throughput: f32,
    pub adaptation_rate: f32,
    pub energy_efficiency: f32,
    pub fault_tolerance: f32,
    pub learning_velocity: f32,
    pub emergence_index: f32,
}

impl Default for NeuralSwarmConfig {
    fn default() -> Self {
        Self {
            max_agents: 100,
            topology: SwarmTopology::Mesh {
                connectivity: 0.7,
                redundancy: 3,
            },
            swarm_pattern: SwarmPattern::Emergent {
                emergence_threshold: 0.8,
                pattern_stability: 0.9,
                collective_memory: true,
            },
            cognitive_diversity: 0.8,
            neural_plasticity: 0.7,
            learning_rate: 0.01,
            adaptation_threshold: 0.75,
            consensus_threshold: 0.85,
            memory_retention: 0.9,
            gpu_acceleration: true,
        }
    }
}

impl NeuralSwarmController {
    /// Create a new neural swarm controller
    pub async fn new(
        config: NeuralSwarmConfig,
        compute_service: Option<Arc<ComputeService>>,
    ) -> Result<Self> {
        let id = Uuid::new_v4();
        let neural_memory = Arc::new(NeuralMemory::new().await?);
        let neural_consensus = Arc::new(NeuralConsensus::new().await?);
        let actor_system = Arc::new(NeuralActorSystem::new().await?);
        
        info!("Initializing neural swarm controller with ID: {}", id);
        
        Ok(Self {
            id,
            config,
            agents: Arc::new(RwLock::new(HashMap::new())),
            tasks: Arc::new(RwLock::new(HashMap::new())),
            neural_memory,
            neural_consensus,
            actor_system,
            compute_service,
            swarm_metrics: Arc::new(RwLock::new(SwarmMetrics::default())),
            topology_matrix: Arc::new(RwLock::new(Matrix3::identity())),
            active_patterns: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Add a neural agent to the swarm
    pub async fn add_agent(
        &self,
        role: AgentRole,
        cognitive_pattern: CognitivePattern,
        capabilities: Vec<String>,
    ) -> Result<Uuid> {
        let agent_id = Uuid::new_v4();
        let agent = NeuralSwarmAgent {
            id: agent_id,
            role,
            cognitive_pattern,
            position: Vector3::new(
                fastrand::f32() * 100.0,
                fastrand::f32() * 100.0,
                fastrand::f32() * 100.0,
            ),
            velocity: Vector3::zeros(),
            connections: HashSet::new(),
            neural_state: NeuralState {
                activation_level: 0.5,
                cognitive_load: 0.0,
                learning_rate: self.config.learning_rate,
                attention_weights: HashMap::new(),
                memory_utilization: 0.0,
                neural_connections: 0,
                synaptic_strength: 0.5,
            },
            last_activity: Utc::now(),
            performance_metrics: PerformanceMetrics::default(),
            capabilities,
            workload: 0.0,
            trust_score: 0.5,
        };

        // Register with actor system
        self.actor_system.add_neural_actor(
            agent_id,
            cognitive_pattern.clone(),
            agent.capabilities.clone(),
        ).await?;

        // Store in memory
        self.neural_memory.store_experience(
            MemoryType::Agent,
            agent_id.to_string(),
            ExperienceData::AgentCreation {
                agent_id,
                role: agent.role.clone(),
                cognitive_pattern: cognitive_pattern.clone(),
                timestamp: Utc::now(),
            },
        ).await?;

        // Add to swarm
        let mut agents = self.agents.write().await;
        agents.insert(agent_id, agent);
        
        // Update topology
        self.update_topology().await?;
        
        info!("Added neural agent {} with role {:?}", agent_id, role);
        Ok(agent_id)
    }

    /// Remove an agent from the swarm
    pub async fn remove_agent(&self, agent_id: Uuid) -> Result<()> {
        let mut agents = self.agents.write().await;
        if let Some(agent) = agents.remove(&agent_id) {
            // Remove from actor system
            self.actor_system.remove_neural_actor(agent_id).await?;
            
            // Update connections
            for other_agent in agents.values_mut() {
                other_agent.connections.remove(&agent_id);
            }
            
            // Store removal in memory
            self.neural_memory.store_experience(
                MemoryType::Agent,
                agent_id.to_string(),
                ExperienceData::AgentRemoval {
                    agent_id,
                    role: agent.role,
                    timestamp: Utc::now(),
                },
            ).await?;
            
            // Update topology
            self.update_topology().await?;
            
            info!("Removed neural agent {}", agent_id);
        }
        Ok(())
    }

    /// Submit a task to the neural swarm
    pub async fn submit_task(
        &self,
        description: String,
        cognitive_requirements: Vec<CognitivePattern>,
        priority: TaskPriority,
        complexity: f32,
    ) -> Result<Uuid> {
        let task_id = Uuid::new_v4();
        let task = NeuralSwarmTask {
            id: task_id,
            description: description.clone(),
            cognitive_requirements,
            priority,
            complexity,
            estimated_duration: chrono::Duration::minutes((complexity * 60.0) as i64),
            required_agents: ((complexity * 5.0) as u32).max(1),
            dependencies: Vec::new(),
            neural_constraints: NeuralConstraints {
                min_activation_level: 0.6,
                max_cognitive_load: 0.8,
                required_trust_score: 0.7,
                neural_synchronization: complexity > 0.7,
                collective_intelligence: complexity > 0.8,
            },
            collaboration_type: if complexity > 0.8 {
                CollaborationType::Swarm
            } else if complexity > 0.5 {
                CollaborationType::Mesh
            } else {
                CollaborationType::Parallel
            },
        };

        // Store task
        let mut tasks = self.tasks.write().await;
        tasks.insert(task_id, task.clone());
        drop(tasks);

        // Store in neural memory
        self.neural_memory.store_experience(
            MemoryType::Task,
            task_id.to_string(),
            ExperienceData::TaskSubmission {
                task_id,
                description,
                priority,
                complexity,
                timestamp: Utc::now(),
            },
        ).await?;

        // Assign agents using neural selection
        self.assign_agents_to_task(task_id).await?;
        
        info!("Submitted neural swarm task {} with complexity {}", task_id, complexity);
        Ok(task_id)
    }

    /// Assign agents to a task using neural selection
    async fn assign_agents_to_task(&self, task_id: Uuid) -> Result<()> {
        let tasks = self.tasks.read().await;
        let task = tasks.get(&task_id)
            .context("Task not found")?;
        
        let agents = self.agents.read().await;
        let mut suitable_agents = Vec::new();
        
        // Find agents matching cognitive requirements
        for agent in agents.values() {
            if self.is_agent_suitable(agent, task) {
                suitable_agents.push(agent.id);
            }
        }
        
        // Sort by fitness score
        suitable_agents.sort_by(|a, b| {
            let score_a = self.calculate_fitness_score(&agents[a], task);
            let score_b = self.calculate_fitness_score(&agents[b], task);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Select top agents
        let selected_agents = suitable_agents
            .into_iter()
            .take(task.required_agents as usize)
            .collect::<Vec<_>>();
        
        // Assign task to selected agents
        for agent_id in selected_agents {
            self.actor_system.assign_task(agent_id, task_id, task.clone()).await?;
        }
        
        debug!("Assigned {} agents to task {}", task.required_agents, task_id);
        Ok(())
    }

    /// Check if an agent is suitable for a task
    fn is_agent_suitable(&self, agent: &NeuralSwarmAgent, task: &NeuralSwarmTask) -> bool {
        // Check cognitive pattern compatibility
        let cognitive_match = task.cognitive_requirements.contains(&agent.cognitive_pattern);
        
        // Check neural constraints
        let neural_constraints_met = 
            agent.neural_state.activation_level >= task.neural_constraints.min_activation_level &&
            agent.neural_state.cognitive_load <= task.neural_constraints.max_cognitive_load &&
            agent.trust_score >= task.neural_constraints.required_trust_score;
        
        // Check workload
        let workload_ok = agent.workload < 0.8;
        
        cognitive_match && neural_constraints_met && workload_ok
    }

    /// Calculate fitness score for agent-task pairing
    fn calculate_fitness_score(&self, agent: &NeuralSwarmAgent, task: &NeuralSwarmTask) -> f32 {
        let mut score = 0.0;
        
        // Cognitive compatibility
        if task.cognitive_requirements.contains(&agent.cognitive_pattern) {
            score += 0.3;
        }
        
        // Performance metrics
        score += agent.performance_metrics.task_completion_rate * 0.2;
        score += agent.performance_metrics.accuracy_score * 0.2;
        score += (1.0 - agent.performance_metrics.response_time.min(1.0)) * 0.1;
        
        // Neural state
        score += agent.neural_state.activation_level * 0.1;
        score += (1.0 - agent.neural_state.cognitive_load) * 0.1;
        
        // Trust and collaboration
        score += agent.trust_score * 0.1;
        
        // Workload balance
        score += (1.0 - agent.workload) * 0.1;
        
        score
    }

    /// Update swarm topology based on current state
    async fn update_topology(&self) -> Result<()> {
        let agents = self.agents.read().await;
        let agent_count = agents.len();
        
        if agent_count < 2 {
            return Ok(());
        }
        
        match &self.config.topology {
            SwarmTopology::Mesh { connectivity, .. } => {
                self.update_mesh_topology(*connectivity, &agents).await?
            },
            SwarmTopology::Hierarchical { levels, branching_factor } => {
                self.update_hierarchical_topology(*levels, *branching_factor, &agents).await?
            },
            SwarmTopology::Ring { bidirectional, cluster_size } => {
                self.update_ring_topology(*bidirectional, *cluster_size, &agents).await?
            },
            SwarmTopology::Star { hub_capacity, failover_hubs } => {
                self.update_star_topology(*hub_capacity, failover_hubs, &agents).await?
            },
            SwarmTopology::Adaptive { base_topology, adaptation_rate, .. } => {
                self.update_adaptive_topology(base_topology, *adaptation_rate, &agents).await?
            },
        }
        
        debug!("Updated swarm topology for {} agents", agent_count);
        Ok(())
    }

    /// Update mesh topology
    async fn update_mesh_topology(
        &self,
        connectivity: f32,
        agents: &HashMap<Uuid, NeuralSwarmAgent>,
    ) -> Result<()> {
        let agent_ids: Vec<Uuid> = agents.keys().cloned().collect();
        let target_connections = (agent_ids.len() as f32 * connectivity) as usize;
        
        for agent_id in &agent_ids {
            let mut connections = HashSet::new();
            
            // Add random connections
            while connections.len() < target_connections {
                let target = agent_ids[fastrand::usize(..agent_ids.len())];
                if target != *agent_id {
                    connections.insert(target);
                }
            }
            
            // Update agent connections through actor system
            self.actor_system.update_connections(*agent_id, connections).await?;
        }
        
        Ok(())
    }

    /// Update hierarchical topology
    async fn update_hierarchical_topology(
        &self,
        levels: u32,
        branching_factor: u32,
        agents: &HashMap<Uuid, NeuralSwarmAgent>,
    ) -> Result<()> {
        let agent_ids: Vec<Uuid> = agents.keys().cloned().collect();
        let mut level_assignments = HashMap::new();
        
        // Assign agents to levels
        for (i, agent_id) in agent_ids.iter().enumerate() {
            let level = i % levels as usize;
            level_assignments.entry(level).or_insert_with(Vec::new).push(*agent_id);
        }
        
        // Create hierarchical connections
        for level in 0..levels as usize {
            if let Some(current_level_agents) = level_assignments.get(&level) {
                for agent_id in current_level_agents {
                    let mut connections = HashSet::new();
                    
                    // Connect to parent level
                    if level > 0 {
                        if let Some(parent_level_agents) = level_assignments.get(&(level - 1)) {
                            let parent_index = fastrand::usize(..parent_level_agents.len());
                            connections.insert(parent_level_agents[parent_index]);
                        }
                    }
                    
                    // Connect to child level
                    if level < (levels as usize - 1) {
                        if let Some(child_level_agents) = level_assignments.get(&(level + 1)) {
                            let children_count = (branching_factor as usize).min(child_level_agents.len());
                            for i in 0..children_count {
                                connections.insert(child_level_agents[i]);
                            }
                        }
                    }
                    
                    self.actor_system.update_connections(*agent_id, connections).await?;
                }
            }
        }
        
        Ok(())
    }

    /// Update ring topology
    async fn update_ring_topology(
        &self,
        bidirectional: bool,
        cluster_size: u32,
        agents: &HashMap<Uuid, NeuralSwarmAgent>,
    ) -> Result<()> {
        let agent_ids: Vec<Uuid> = agents.keys().cloned().collect();
        
        for (i, agent_id) in agent_ids.iter().enumerate() {
            let mut connections = HashSet::new();
            
            // Connect to next agent in ring
            let next_index = (i + 1) % agent_ids.len();
            connections.insert(agent_ids[next_index]);
            
            // Connect to previous agent if bidirectional
            if bidirectional {
                let prev_index = if i == 0 { agent_ids.len() - 1 } else { i - 1 };
                connections.insert(agent_ids[prev_index]);
            }
            
            // Add cluster connections
            for offset in 1..=cluster_size as usize {
                if offset < agent_ids.len() / 2 {
                    let cluster_index = (i + offset) % agent_ids.len();
                    connections.insert(agent_ids[cluster_index]);
                    
                    if bidirectional {
                        let cluster_prev_index = if i >= offset {
                            i - offset
                        } else {
                            agent_ids.len() - (offset - i)
                        };
                        connections.insert(agent_ids[cluster_prev_index]);
                    }
                }
            }
            
            self.actor_system.update_connections(*agent_id, connections).await?;
        }
        
        Ok(())
    }

    /// Update star topology
    async fn update_star_topology(
        &self,
        hub_capacity: u32,
        failover_hubs: &[Uuid],
        agents: &HashMap<Uuid, NeuralSwarmAgent>,
    ) -> Result<()> {
        let agent_ids: Vec<Uuid> = agents.keys().cloned().collect();
        
        // Select hub agents
        let mut hubs = Vec::new();
        
        // Add specified failover hubs if they exist
        for hub_id in failover_hubs {
            if agents.contains_key(hub_id) {
                hubs.push(*hub_id);
            }
        }
        
        // Add additional hubs if needed
        while hubs.len() < (agent_ids.len() / hub_capacity as usize).max(1) {
            let hub_candidate = agent_ids[fastrand::usize(..agent_ids.len())];
            if !hubs.contains(&hub_candidate) {
                hubs.push(hub_candidate);
            }
        }
        
        // Connect spoke agents to hubs
        for agent_id in &agent_ids {
            if !hubs.contains(agent_id) {
                let hub_index = fastrand::usize(..hubs.len());
                let hub_id = hubs[hub_index];
                
                let connections = std::iter::once(hub_id).collect();
                self.actor_system.update_connections(*agent_id, connections).await?;
            }
        }
        
        // Connect hubs to all spoke agents
        for hub_id in &hubs {
            let connections = agent_ids.iter()
                .filter(|id| !hubs.contains(id))
                .cloned()
                .collect();
            self.actor_system.update_connections(*hub_id, connections).await?;
        }
        
        Ok(())
    }

    /// Update adaptive topology
    async fn update_adaptive_topology(
        &self,
        base_topology: &SwarmTopology,
        adaptation_rate: f32,
        agents: &HashMap<Uuid, NeuralSwarmAgent>,
    ) -> Result<()> {
        // First apply base topology
        match base_topology.as_ref() {
            SwarmTopology::Mesh { connectivity, .. } => {
                self.update_mesh_topology(*connectivity, agents).await?
            },
            SwarmTopology::Hierarchical { levels, branching_factor } => {
                self.update_hierarchical_topology(*levels, *branching_factor, agents).await?
            },
            SwarmTopology::Ring { bidirectional, cluster_size } => {
                self.update_ring_topology(*bidirectional, *cluster_size, agents).await?
            },
            SwarmTopology::Star { hub_capacity, failover_hubs } => {
                self.update_star_topology(*hub_capacity, failover_hubs, agents).await?
            },
            _ => {}, // Avoid infinite recursion
        }
        
        // Apply adaptive modifications based on performance
        let metrics = self.swarm_metrics.read().await;
        if metrics.collective_intelligence < self.config.adaptation_threshold {
            // Increase connectivity for better collaboration
            self.adapt_connectivity(adaptation_rate, agents).await?;
        }
        
        Ok(())
    }

    /// Adapt connectivity based on performance
    async fn adapt_connectivity(
        &self,
        adaptation_rate: f32,
        agents: &HashMap<Uuid, NeuralSwarmAgent>,
    ) -> Result<()> {
        let agent_ids: Vec<Uuid> = agents.keys().cloned().collect();
        
        for agent_id in &agent_ids {
            if let Some(agent) = agents.get(agent_id) {
                // Add connections to high-performing agents
                let mut new_connections = agent.connections.clone();
                
                for other_agent in agents.values() {
                    if other_agent.id != *agent_id && 
                       !new_connections.contains(&other_agent.id) &&
                       other_agent.performance_metrics.task_completion_rate > 0.8 {
                        
                        if fastrand::f32() < adaptation_rate {
                            new_connections.insert(other_agent.id);
                        }
                    }
                }
                
                self.actor_system.update_connections(*agent_id, new_connections).await?;
            }
        }
        
        Ok(())
    }

    /// Update swarm metrics
    pub async fn update_metrics(&self) -> Result<()> {
        let agents = self.agents.read().await;
        let tasks = self.tasks.read().await;
        
        let mut metrics = SwarmMetrics::default();
        
        if !agents.is_empty() {
            // Calculate collective intelligence
            let total_intelligence: f32 = agents.values()
                .map(|a| a.performance_metrics.accuracy_score * a.neural_state.activation_level)
                .sum();
            metrics.collective_intelligence = total_intelligence / agents.len() as f32;
            
            // Calculate swarm coherence
            let connection_density = agents.values()
                .map(|a| a.connections.len() as f32)
                .sum::<f32>() / (agents.len() as f32 * (agents.len() - 1) as f32);
            metrics.swarm_coherence = connection_density;
            
            // Calculate energy efficiency
            let total_efficiency: f32 = agents.values()
                .map(|a| a.performance_metrics.energy_efficiency)
                .sum();
            metrics.energy_efficiency = total_efficiency / agents.len() as f32;
            
            // Calculate adaptation rate
            let total_adaptation: f32 = agents.values()
                .map(|a| a.performance_metrics.adaptation_speed)
                .sum();
            metrics.adaptation_rate = total_adaptation / agents.len() as f32;
        }
        
        // Calculate task throughput
        let active_tasks = tasks.len() as f32;
        metrics.task_throughput = active_tasks;
        
        // Update stored metrics
        let mut stored_metrics = self.swarm_metrics.write().await;
        *stored_metrics = metrics;
        
        debug!("Updated swarm metrics: collective_intelligence={:.3}, coherence={:.3}", 
               stored_metrics.collective_intelligence, stored_metrics.swarm_coherence);
        
        Ok(())
    }

    /// Get current swarm status
    pub async fn get_status(&self) -> Result<SwarmStatus> {
        let agents = self.agents.read().await;
        let tasks = self.tasks.read().await;
        let metrics = self.swarm_metrics.read().await;
        
        Ok(SwarmStatus {
            swarm_id: self.id,
            agent_count: agents.len() as u32,
            active_tasks: tasks.len() as u32,
            topology: self.config.topology.clone(),
            metrics: metrics.clone(),
            uptime: Utc::now().timestamp(),
        })
    }

    /// Initiate neural consensus for critical decisions
    pub async fn initiate_consensus(
        &self,
        proposal: String,
        participating_agents: Vec<Uuid>,
    ) -> Result<ConsensusResult> {
        self.neural_consensus.initiate_consensus(
            proposal,
            participating_agents,
            self.config.consensus_threshold,
        ).await
    }

    /// Execute swarm intelligence pattern
    pub async fn execute_swarm_pattern(&self, pattern: SwarmPattern) -> Result<()> {
        let pattern_id = Uuid::new_v4().to_string();
        
        // Store pattern in active patterns
        let mut patterns = self.active_patterns.write().await;
        patterns.insert(pattern_id.clone(), pattern.clone());
        drop(patterns);
        
        match pattern {
            SwarmPattern::Flocking { separation_weight, alignment_weight, cohesion_weight } => {
                self.execute_flocking_pattern(separation_weight, alignment_weight, cohesion_weight).await?
            },
            SwarmPattern::Foraging { exploration_bias, exploitation_bias, pheromone_decay } => {
                self.execute_foraging_pattern(exploration_bias, exploitation_bias, pheromone_decay).await?
            },
            SwarmPattern::Clustering { cluster_radius, min_cluster_size, merge_threshold } => {
                self.execute_clustering_pattern(cluster_radius, min_cluster_size, merge_threshold).await?
            },
            SwarmPattern::Emergent { emergence_threshold, pattern_stability, collective_memory } => {
                self.execute_emergent_pattern(emergence_threshold, pattern_stability, collective_memory).await?
            },
        }
        
        info!("Executed swarm pattern: {}", pattern_id);
        Ok(())
    }

    /// Execute flocking pattern
    async fn execute_flocking_pattern(
        &self,
        separation_weight: f32,
        alignment_weight: f32,
        cohesion_weight: f32,
    ) -> Result<()> {
        let mut agents = self.agents.write().await;
        let agent_positions: HashMap<Uuid, Vector3<f32>> = agents.iter()
            .map(|(id, agent)| (*id, agent.position))
            .collect();
        
        for agent in agents.values_mut() {
            let mut separation = Vector3::zeros();
            let mut alignment = Vector3::zeros();
            let mut cohesion = Vector3::zeros();
            let mut neighbor_count = 0;
            
            // Calculate forces from connected agents
            for &neighbor_id in &agent.connections {
                if let Some(&neighbor_pos) = agent_positions.get(&neighbor_id) {
                    let distance = (neighbor_pos - agent.position).magnitude();
                    
                    if distance > 0.0 && distance < 50.0 { // Within influence radius
                        // Separation: steer away from nearby agents
                        if distance < 10.0 {
                            separation += (agent.position - neighbor_pos).normalize() / distance;
                        }
                        
                        // Alignment: match velocity with neighbors
                        if let Some(neighbor_agent) = agents.get(&neighbor_id) {
                            alignment += neighbor_agent.velocity;
                        }
                        
                        // Cohesion: steer toward average position
                        cohesion += neighbor_pos;
                        neighbor_count += 1;
                    }
                }
            }
            
            if neighbor_count > 0 {
                alignment /= neighbor_count as f32;
                cohesion = (cohesion / neighbor_count as f32 - agent.position).normalize();
                
                // Apply flocking forces
                let force = separation * separation_weight + 
                           alignment * alignment_weight + 
                           cohesion * cohesion_weight;
                
                agent.velocity = (agent.velocity + force * 0.1).normalize() * 5.0;
                agent.position += agent.velocity * 0.1;
            }
        }
        
        Ok(())
    }

    /// Execute foraging pattern
    async fn execute_foraging_pattern(
        &self,
        exploration_bias: f32,
        exploitation_bias: f32,
        pheromone_decay: f32,
    ) -> Result<()> {
        // Implementation of foraging behavior
        // Agents explore for resources (tasks) and exploit known good areas
        let agents = self.agents.read().await;
        
        for agent in agents.values() {
            // Determine exploration vs exploitation
            if fastrand::f32() < exploration_bias {
                // Exploration: move to unexplored areas
                self.actor_system.explore_new_area(agent.id).await?;
            } else {
                // Exploitation: focus on productive areas
                self.actor_system.exploit_known_area(agent.id, exploitation_bias).await?;
            }
        }
        
        // Apply pheromone decay
        self.neural_memory.decay_pheromone_trails(pheromone_decay).await?;
        
        Ok(())
    }

    /// Execute clustering pattern
    async fn execute_clustering_pattern(
        &self,
        cluster_radius: f32,
        min_cluster_size: u32,
        merge_threshold: f32,
    ) -> Result<()> {
        let agents = self.agents.read().await;
        let mut clusters = Vec::new();
        let mut assigned_agents = HashSet::new();
        
        // Form clusters based on position and cognitive similarity
        for agent in agents.values() {
            if assigned_agents.contains(&agent.id) {
                continue;
            }
            
            let mut cluster = vec![agent.id];
            assigned_agents.insert(agent.id);
            
            // Find nearby agents with similar cognitive patterns
            for other_agent in agents.values() {
                if assigned_agents.contains(&other_agent.id) {
                    continue;
                }
                
                let distance = (other_agent.position - agent.position).magnitude();
                let cognitive_similarity = self.calculate_cognitive_similarity(&agent.cognitive_pattern, &other_agent.cognitive_pattern);
                
                if distance <= cluster_radius && cognitive_similarity > merge_threshold {
                    cluster.push(other_agent.id);
                    assigned_agents.insert(other_agent.id);
                }
            }
            
            if cluster.len() >= min_cluster_size as usize {
                clusters.push(cluster);
            }
        }
        
        // Update cluster information in actor system
        for (cluster_id, cluster) in clusters.iter().enumerate() {
            self.actor_system.form_cluster(cluster_id, cluster.clone()).await?;
        }
        
        debug!("Formed {} clusters with minimum size {}", clusters.len(), min_cluster_size);
        Ok(())
    }

    /// Execute emergent pattern
    async fn execute_emergent_pattern(
        &self,
        emergence_threshold: f32,
        pattern_stability: f32,
        collective_memory: bool,
    ) -> Result<()> {
        let metrics = self.swarm_metrics.read().await;
        
        // Check if emergence conditions are met
        if metrics.collective_intelligence > emergence_threshold {
            // Enable emergent behaviors
            self.actor_system.enable_emergent_behaviors(pattern_stability).await?;
            
            if collective_memory {
                // Activate collective memory system
                self.neural_memory.activate_collective_memory().await?;
            }
            
            info!("Emergent pattern activated with stability {}", pattern_stability);
        }
        
        Ok(())
    }

    /// Calculate cognitive similarity between two patterns
    fn calculate_cognitive_similarity(&self, pattern1: &CognitivePattern, pattern2: &CognitivePattern) -> f32 {
        // Simplified similarity calculation
        if pattern1 == pattern2 {
            1.0
        } else {
            // More sophisticated similarity calculation could be implemented
            0.5
        }
    }

    /// Shutdown the neural swarm controller
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down neural swarm controller {}", self.id);
        
        // Stop all agents
        let agents = self.agents.read().await;
        for agent_id in agents.keys() {
            self.actor_system.stop_neural_actor(*agent_id).await?;
        }
        
        // Shutdown actor system
        self.actor_system.shutdown().await?;
        
        // Store final metrics in memory
        let metrics = self.swarm_metrics.read().await;
        self.neural_memory.store_experience(
            MemoryType::System,
            "shutdown".to_string(),
            ExperienceData::SystemShutdown {
                swarm_id: self.id,
                final_metrics: metrics.clone(),
                timestamp: Utc::now(),
            },
        ).await?;
        
        Ok(())
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            task_completion_rate: 0.0,
            response_time: 1.0,
            accuracy_score: 0.5,
            collaboration_score: 0.5,
            innovation_index: 0.5,
            energy_efficiency: 0.5,
            adaptation_speed: 0.5,
        }
    }
}

impl Default for SwarmMetrics {
    fn default() -> Self {
        Self {
            collective_intelligence: 0.0,
            swarm_coherence: 0.0,
            task_throughput: 0.0,
            adaptation_rate: 0.0,
            energy_efficiency: 0.0,
            fault_tolerance: 0.0,
            learning_velocity: 0.0,
            emergence_index: 0.0,
        }
    }
}

/// Swarm status for external monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmStatus {
    pub swarm_id: Uuid,
    pub agent_count: u32,
    pub active_tasks: u32,
    pub topology: SwarmTopology,
    pub metrics: SwarmMetrics,
    pub uptime: i64,
}
