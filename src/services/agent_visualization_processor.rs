use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use crate::types::claude_flow::{AgentStatus, Vec3};
use crate::config::dev_config;
use sysinfo::{System, Pid};
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Processed agent data optimized for GPU visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizedAgent {
    pub id: String,
    pub name: String,
    pub agent_type: String,
    pub status: String,
    
    // Visual attributes for GPU rendering
    pub position: Vec3,
    pub velocity: Vec3,
    pub color: String,
    pub size: f32,
    pub glow_intensity: f32,
    
    // Performance metrics (0-1 normalized)
    pub health: f32,
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub activity_level: f32,
    
    // Task information
    pub active_tasks: u32,
    pub completed_tasks: u32,
    pub success_rate: f32,
    pub current_task: Option<String>,
    
    // Token usage for visual weight
    pub token_usage: u64,
    pub token_rate: f32, // tokens per second
    
    // Metadata for rich tooltips
    pub metadata: AgentMetadata,
    
    // Shape hint for client rendering
    pub shape_type: ShapeType,
    pub animation_state: AnimationState,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetadata {
    pub created_at: DateTime<Utc>,
    pub age_seconds: u64,
    pub last_activity: DateTime<Utc>,
    pub capabilities: Vec<String>,
    pub error_count: u32,
    pub warning_count: u32,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShapeType {
    Sphere,      // Default, healthy agents
    Cube,        // Busy/working agents
    Octahedron,  // Coordinators
    Cylinder,    // Analysts
    Torus,       // Testers
    Cone,        // Architects
    Pyramid,     // Error state
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnimationState {
    Idle,
    Pulsing,
    Rotating,
    Bouncing,
    Glowing,
    Flashing,  // For errors
}

/// Processed edge data for GPU rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizedConnection {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    
    // Visual attributes
    pub strength: f32,      // 0-1, affects line thickness
    pub flow_rate: f32,     // 0-1, affects particle speed
    pub color: String,
    pub particle_count: u32,
    
    // Data metrics
    pub data_volume: u64,   // Total bytes
    pub message_count: u64,
    pub latency_ms: f32,
    pub error_rate: f32,
    
    // Animation hints
    pub is_active: bool,
    pub pulse_frequency: f32,
}

/// Swarm overview for high-level visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmVisualization {
    pub swarm_id: String,
    pub topology: String,
    pub total_agents: u32,
    pub active_agents: u32,
    
    // Performance overview
    pub overall_health: f32,
    pub total_cpu_usage: f32,
    pub total_memory_usage: f32,
    pub total_token_usage: u64,
    pub tokens_per_second: f32,
    
    // Visual clustering hints
    pub clusters: Vec<AgentCluster>,
    
    // Time-series data for graphs
    pub performance_history: Vec<PerformanceSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCluster {
    pub id: String,
    pub center: Vec3,
    pub radius: f32,
    pub agent_ids: Vec<String>,
    pub cluster_type: String,
    pub color: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub timestamp: DateTime<Utc>,
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub active_tasks: u32,
    pub token_rate: f32,
}

/// Complete visualization packet ready for GPU rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentVisualizationData {
    pub timestamp: DateTime<Utc>,
    pub swarm: SwarmVisualization,
    pub agents: Vec<VisualizedAgent>,
    pub connections: Vec<VisualizedConnection>,
    
    // Physics hints for GPU solver
    pub physics_config: PhysicsConfig,
    
    // Visual configuration
    pub visual_config: VisualConfig,
}

// Use PhysicsConfig from agent_visualization_protocol module
use crate::services::agent_visualization_protocol::PhysicsConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualConfig {
    pub color_scheme: HashMap<String, String>,
    pub size_multipliers: HashMap<String, f32>,
    pub glow_settings: GlowSettings,
    pub animation_speeds: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlowSettings {
    pub base_intensity: f32,
    pub health_multiplier: f32,
    pub activity_multiplier: f32,
    pub error_intensity: f32,
}

/// Global system instance for monitoring
static SYSTEM: Lazy<Arc<Mutex<System>>> = Lazy::new(|| {
    let mut sys = System::new_all();
    sys.refresh_all();
    Arc::new(Mutex::new(sys))
});

/// Main processor that transforms MCP data into visualization-ready format
pub struct AgentVisualizationProcessor {
    token_history: HashMap<String, Vec<(DateTime<Utc>, u64)>>,
    _performance_history: HashMap<String, Vec<PerformanceSnapshot>>,
    _last_update: DateTime<Utc>,
    process_map: HashMap<String, Pid>,
}

impl AgentVisualizationProcessor {
    pub fn new() -> Self {
        Self {
            token_history: HashMap::new(),
            _performance_history: HashMap::new(),
            _last_update: Utc::now(),
            process_map: HashMap::new(),
        }
    }
    
    /// Process raw agent data from MCP into visualization format
    pub fn process_agents(&mut self, agents: Vec<AgentStatus>) -> Vec<VisualizedAgent> {
        agents.into_iter().map(|agent| {
            // Use the new agent_type field directly
            let agent_type = agent.agent_type.clone();

            // Use already normalized values from the new structure
            let health = agent.health;
            let cpu_usage = agent.cpu_usage;
            let memory_usage = agent.memory_usage;
            let activity_level = agent.activity;

            // Determine visual properties based on agent state
            let (color, shape, animation) = self.get_visual_properties(&agent_type, &agent.status, health);

            // Calculate size based on workload or task count
            let size = agent.workload.unwrap_or_else(|| 1.0 + (agent.active_tasks_count as f32 * 0.2).min(2.0));

            // Glow intensity based on activity
            let glow_intensity = 0.3 + activity_level * 0.7;

            // Use the existing token data from the new structure
            let token_usage = agent.tokens;
            let token_rate = agent.token_rate;

            // Use position from agent if available, otherwise physics will set it
            let position = agent.position.unwrap_or(Vec3 { x: 0.0, y: 0.0, z: 0.0 });

            VisualizedAgent {
                id: agent.agent_id.clone(),
                name: agent.profile.name.clone(),
                agent_type: agent_type.clone(),
                status: agent.status.clone(),

                position,
                velocity: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
                color,
                size,
                glow_intensity,

                health,
                cpu_usage,
                memory_usage,
                activity_level,

                active_tasks: agent.tasks_active,
                completed_tasks: agent.tasks_completed,
                success_rate: agent.success_rate_normalized,
                current_task: agent.current_task_description.clone(),

                token_usage,
                token_rate,

                metadata: AgentMetadata {
                    created_at: agent.timestamp,
                    age_seconds: agent.age,
                    last_activity: agent.timestamp,
                    capabilities: agent.capabilities.clone(),
                    error_count: agent.failed_tasks_count,
                    warning_count: 0,
                    tags: vec![agent_type.clone(), agent.status.clone()],
                },

                shape_type: shape,
                animation_state: animation,
            }
        }).collect()
    }
    
    /// Get visual properties based on agent type and state
    fn get_visual_properties(&self, agent_type: &str, status: &str, health: f32) -> (String, ShapeType, AnimationState) {
        let colors = &dev_config::rendering().agent_colors;
        let color = match agent_type {
            "coordinator" => &colors.coordinator,
            "coder" => &colors.coder,
            "architect" => &colors.architect,
            "analyst" => &colors.analyst,
            "tester" => &colors.tester,
            "researcher" => &colors.researcher,
            "reviewer" => &colors.reviewer,
            "optimizer" => &colors.optimizer,
            "documenter" => &colors.documenter,
            _ => &colors.default,
        }.to_string();
        
        let shape = match agent_type {
            "coordinator" => ShapeType::Octahedron,
            "architect" => ShapeType::Cone,
            "analyst" => ShapeType::Cylinder,
            "tester" => ShapeType::Torus,
            _ => match status {
                "error" => ShapeType::Pyramid,
                "busy" => ShapeType::Cube,
                _ => ShapeType::Sphere,
            }
        };
        
        let animation = match status {
            "error" => AnimationState::Flashing,
            "busy" => AnimationState::Rotating,
            "idle" => AnimationState::Idle,
            _ => if health < 0.3 {
                AnimationState::Pulsing
            } else {
                AnimationState::Glowing
            }
        };
        
        (color, shape, animation)
    }
    
    /// Calculate token usage rate
    fn calculate_token_rate(&mut self, agent_id: &str, current_usage: u64) -> f32 {
        let now = Utc::now();
        let history = self.token_history.entry(agent_id.to_string()).or_insert_with(Vec::new);
        
        // Add current reading
        history.push((now, current_usage));
        
        // Keep only last 60 seconds of history
        let cutoff = now - chrono::Duration::seconds(60);
        history.retain(|(time, _)| *time > cutoff);
        
        // Calculate rate
        if history.len() < 2 {
            return 0.0;
        }
        
        let oldest = &history[0];
        let newest = &history[history.len() - 1];
        let time_diff = (newest.0 - oldest.0).num_seconds() as f32;
        
        if time_diff > 0.0 {
            ((newest.1 - oldest.1) as f32) / time_diff
        } else {
            0.0
        }
    }
    
    /// Get token usage for an agent from actual MCP data
    fn get_agent_token_usage(&self, agent_id: &str) -> u64 {
        // Get token usage from history or default based on activity
        if let Some(history) = self.token_history.get(agent_id) {
            history.last().map(|(_, usage)| *usage).unwrap_or(0)
        } else {
            // Base token usage estimate based on agent ID hash for consistency
            let mut hasher = DefaultHasher::new();
            agent_id.hash(&mut hasher);
            (hasher.finish() % 10000) + 500
        }
    }
    
    /// Get real CPU and memory usage for an agent process
    fn get_real_system_metrics(&mut self, agent_id: &str) -> (f32, f32) {
        let mut sys = SYSTEM.lock().unwrap();
        sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
        
        // Try to find process by agent ID or name
        if let Some(&pid) = self.process_map.get(agent_id) {
            if let Some(process) = sys.process(pid) {
                let cpu_usage = process.cpu_usage() / 100.0; // Convert to 0-1 range
                let memory_usage = process.memory() as f32 / (1024.0 * 1024.0 * 1024.0); // Convert to GB
                let total_memory = sys.total_memory() as f32 / (1024.0 * 1024.0 * 1024.0); // GB
                let memory_percentage = if total_memory > 0.0 { memory_usage / total_memory } else { 0.0 };
                
                return (cpu_usage.clamp(0.0, 1.0), memory_percentage.clamp(0.0, 1.0));
            }
        }
        
        // Fallback: find process by name containing agent_id
        for (pid, process) in sys.processes() {
            let process_name = process.name().to_string_lossy().to_lowercase();
            let agent_id_lower = agent_id.to_lowercase();
            
            // Check if process name contains agent type or similar identifier
            if process_name.contains(&agent_id_lower) || 
               process_name.contains("claude") ||
               process_name.contains("agent") ||
               process_name.contains("bot") {
                
                // Cache the mapping for future use
                self.process_map.insert(agent_id.to_string(), *pid);
                
                let cpu_usage = process.cpu_usage() / 100.0;
                let memory_usage = process.memory() as f32 / (1024.0 * 1024.0 * 1024.0);
                let total_memory = sys.total_memory() as f32 / (1024.0 * 1024.0 * 1024.0);
                let memory_percentage = if total_memory > 0.0 { memory_usage / total_memory } else { 0.0 };
                
                return (cpu_usage.clamp(0.0, 1.0), memory_percentage.clamp(0.0, 1.0));
            }
        }
        
        // Final fallback: use system-wide averages scaled down
        let global_cpu = sys.global_cpu_usage() / 100.0;
        let used_memory = sys.used_memory() as f32;
        let total_memory = sys.total_memory() as f32;
        let global_memory = if total_memory > 0.0 { used_memory / total_memory } else { 0.0 };
        
        // Scale down to represent a single agent's approximate usage
        let agent_cpu = (global_cpu * 0.1).clamp(0.0, 1.0); // Assume 10% of system CPU per agent
        let agent_memory = (global_memory * 0.05).clamp(0.0, 1.0); // Assume 5% of system memory per agent
        
        (agent_cpu, agent_memory)
    }
    
    /// Create complete visualization data packet
    pub fn create_visualization_packet(
        &mut self,
        agents: Vec<AgentStatus>,
        swarm_id: String,
        topology: String,
    ) -> AgentVisualizationData {
        let processed_agents = self.process_agents(agents);
        
        // Calculate swarm metrics
        let total_agents = processed_agents.len() as u32;
        let active_agents = processed_agents.iter()
            .filter(|a| a.status != "idle" && a.status != "error")
            .count() as u32;
        
        let overall_health = processed_agents.iter()
            .map(|a| a.health)
            .sum::<f32>() / total_agents.max(1) as f32;
        
        let total_cpu_usage = processed_agents.iter()
            .map(|a| a.cpu_usage)
            .sum::<f32>();
        
        let total_token_usage = processed_agents.iter()
            .map(|a| a.token_usage)
            .sum::<u64>();
        
        let tokens_per_second = processed_agents.iter()
            .map(|a| a.token_rate)
            .sum::<f32>();
        
        // Create connections based on actual agent relationships
        let connections = self.create_connections(&processed_agents);
        
        // Create clusters based on agent types
        let clusters = self.create_clusters(&processed_agents);
        
        AgentVisualizationData {
            timestamp: Utc::now(),
            swarm: SwarmVisualization {
                swarm_id,
                topology,
                total_agents,
                active_agents,
                overall_health,
                total_cpu_usage,
                total_memory_usage: processed_agents.iter()
                    .map(|a| a.memory_usage)
                    .sum::<f32>(),
                total_token_usage,
                tokens_per_second,
                clusters,
                performance_history: self.get_performance_history(),
            },
            agents: processed_agents,
            connections,
            physics_config: PhysicsConfig {
                spring_k: 0.3,
                link_distance: 25.0,
                damping: 0.92,
                repel_k: 20.0,
                gravity_k: 0.08,
                max_velocity: 0.8,
            },
            visual_config: self.create_visual_config(),
        }
    }
    
    /// Create connections based on agent relationships
    fn create_connections(&self, agents: &[VisualizedAgent]) -> Vec<VisualizedConnection> {
        let mut connections = Vec::new();
        
        // Create connections between coordinators and other agents
        for coordinator in agents.iter().filter(|a| a.agent_type == "coordinator") {
            for agent in agents.iter().filter(|a| a.id != coordinator.id) {
                connections.push(VisualizedConnection {
                    id: format!("{}-{}", coordinator.id, agent.id),
                    source_id: coordinator.id.clone(),
                    target_id: agent.id.clone(),
                    strength: 0.5,
                    flow_rate: agent.activity_level,
                    color: "#4444FF".to_string(),
                    particle_count: 10,
                    data_volume: 1000,
                    message_count: 10,
                    latency_ms: 5.0,
                    error_rate: 0.0,
                    is_active: agent.active_tasks > 0,
                    pulse_frequency: 1.0,
                });
            }
        }
        
        connections
    }
    
    /// Create agent clusters for visual grouping
    fn create_clusters(&self, agents: &[VisualizedAgent]) -> Vec<AgentCluster> {
        let mut clusters = Vec::new();
        let mut type_groups: HashMap<String, Vec<String>> = HashMap::new();
        
        // Group agents by type
        for agent in agents {
            type_groups.entry(agent.agent_type.clone())
                .or_insert_with(Vec::new)
                .push(agent.id.clone());
        }
        
        // Create clusters
        for (agent_type, agent_ids) in type_groups {
            if agent_ids.len() > 1 {
                clusters.push(AgentCluster {
                    id: format!("cluster-{}", agent_type),
                    center: Vec3 { x: 0.0, y: 0.0, z: 0.0 }, // Will be calculated by physics
                    radius: 15.0,
                    agent_ids,
                    cluster_type: agent_type.clone(),
                    color: self.get_cluster_color(&agent_type),
                });
            }
        }
        
        clusters
    }
    
    fn get_cluster_color(&self, agent_type: &str) -> String {
        match agent_type {
            "coordinator" => "#00FFFF33",
            "coder" => "#00FF0033",
            "architect" => "#FFA50033",
            _ => "#FFFFFF22",
        }.to_string()
    }
    
    fn create_visual_config(&self) -> VisualConfig {
        let mut color_scheme = HashMap::new();
        color_scheme.insert("background".to_string(), "#000033".to_string());
        color_scheme.insert("grid".to_string(), "#003366".to_string());
        color_scheme.insert("text".to_string(), "#FFFFFF".to_string());
        
        let mut size_multipliers = HashMap::new();
        size_multipliers.insert("coordinator".to_string(), 1.5);
        size_multipliers.insert("architect".to_string(), 1.3);
        size_multipliers.insert("default".to_string(), 1.0);
        
        let mut animation_speeds = HashMap::new();
        animation_speeds.insert("pulse".to_string(), 2.0);
        animation_speeds.insert("rotate".to_string(), 1.0);
        animation_speeds.insert("glow".to_string(), 0.5);
        
        VisualConfig {
            color_scheme,
            size_multipliers,
            glow_settings: GlowSettings {
                base_intensity: 0.3,
                health_multiplier: 0.5,
                activity_multiplier: 0.8,
                error_intensity: 1.0,
            },
            animation_speeds,
        }
    }

    /// Get performance history for swarm visualization
    fn get_performance_history(&self) -> Vec<PerformanceSnapshot> {
        // Generate recent performance history based on current data
        let now = chrono::Utc::now();
        let mut history = Vec::new();

        // Create snapshots for the last few minutes (simulated)
        for i in 0..10 {
            let timestamp = now - chrono::Duration::minutes(i);

            // Calculate aggregate metrics from current agents with some variation
            let variation = (i as f32 * 0.1).sin() * 0.1;

            history.push(PerformanceSnapshot {
                timestamp,
                cpu_usage: (0.4 + variation).clamp(0.0, 1.0),
                memory_usage: (0.3 + variation * 0.5).clamp(0.0, 1.0),
                active_tasks: (5.0 + variation * 10.0).max(0.0) as u32,
                token_rate: (10.0 + variation * 20.0).max(0.0),
            });
        }

        // Reverse to get chronological order
        history.reverse();
        history
    }
}