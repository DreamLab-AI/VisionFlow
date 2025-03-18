use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::RwLock;
use std::collections::{HashMap, HashSet};
use actix_web::web;
use rand::distributions::{Alphanumeric, DistString};
use rand::Rng;
use rand::seq::SliceRandom;
use std::io::{Error, ErrorKind};
use serde_json;
use std::pin::Pin;
use std::time::{Duration, Instant};
use futures::Future;
use log::{info, warn, error, debug};
use scopeguard;

use tokio::fs::File as TokioFile;
use crate::models::graph::GraphData;
use crate::utils::socket_flow_messages::Node;
use crate::models::edge::Edge;
use crate::models::metadata::MetadataStore;
use crate::app_state::AppState;
use crate::config::Settings;
use crate::utils::gpu_compute::GPUCompute;
use crate::models::simulation_params::{SimulationParams, SimulationPhase, SimulationMode};
use crate::handlers::socket_flow_handler::ClientManager;
use crate::models::pagination::PaginatedGraphData;
use crate::handlers::socket_flow_handler::ClientManager;
use tokio::sync::Mutex;
use once_cell::sync::Lazy;

// Static flag to prevent multiple simultaneous graph rebuilds
static GRAPH_REBUILD_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

// Static flag to track if a simulation loop is already running and current simulation ID
static SIMULATION_LOOP_RUNNING: AtomicBool = AtomicBool::new(false);

// A mutex to synchronize simulation loop creation and shutdown
// This is necessary to avoid race conditions when a new GraphService is created
// while an old one is being shut down
static SIMULATION_MUTEX: Lazy<Mutex<String>> = Lazy::new(|| Mutex::new(String::new()));


// Physics simulation constants for independent processing
const CONTINUOUS_SIMULATION_INTERVAL_MS: u64 = 16; // ~60fps
const STABLE_THRESHOLD_ITERATIONS: usize = 100; // Number of iterations with minimal movement to consider graph stable

// Cache configuration
const NODE_POSITION_CACHE_TTL_MS: u64 = 50; // 50ms cache time
const METADATA_FILE_WAIT_TIMEOUT_MS: u64 = 5000; // 5 second wait timeout
const SHUTDOWN_TIMEOUT_MS: u64 = 5000; // 5 second shutdown timeout

// Physics stabilization constants
const STABLE_THRESHOLD_ITERATIONS: usize = 100; // Number of iterations with minimal movement
const POSITION_STABILITY_THRESHOLD: f32 = 0.001; // 1mm threshold for stability

// Rate limiting and conflict resolution constants
const UPDATE_RATE_LIMIT_MS: u64 = 16; // ~60fps max update rate
const POSITION_CONFLICT_THRESHOLD: f32 = 0.001; // 1mm threshold for position conflicts
const MAX_CONCURRENT_UPDATES: usize = 100; // Maximum number of node updates per batch
const METADATA_FILE_CHECK_INTERVAL_MS: u64 = 100; // Check every 100ms
// Constants for GPU retry mechanism
const MAX_GPU_CALCULATION_RETRIES: u32 = 3;
const GPU_RETRY_DELAY_MS: u64 = 500; // 500ms delay between retries

#[derive(Clone)]
pub struct GraphService {
    graph_data: Arc<RwLock<GraphData>>,
    shutdown_complete: Arc<AtomicBool>,
    node_map: Arc<RwLock<HashMap<String, Node>>>,
    gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
    node_positions_cache: Arc<RwLock<Option<(Vec<Node>, Instant)>>>,
    last_update: Arc<RwLock<Instant>>,
    pending_updates: Arc<RwLock<HashMap<String, (Node, Instant)>>>,
    cache_enabled: bool,
    simulation_id: String,
    client_manager: Option<Arc<ClientManager>>,
    is_initialized: Arc<AtomicBool>,
    shutdown_requested: Arc<AtomicBool>,
}

impl GraphService {
    pub async fn new(
        settings: Arc<RwLock<Settings>>, 
        gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
        client_manager: Option<Arc<ClientManager>>) -> Self {
        // Get physics settings
        let physics_settings = settings.read().await.visualization.physics.clone();

        // Generate a unique ID for this GraphService instance
        let simulation_id = Alphanumeric.sample_string(&mut rand::thread_rng(), 8);
        info!("[GraphService::new] Creating independent GraphService instance with ID: {}", simulation_id);
        
        // Acquire the mutex to ensure exclusive access during initialization
        let mut guard = SIMULATION_MUTEX.lock().await;
        
        // Check if there's already an instance running
        let is_running = SIMULATION_LOOP_RUNNING.load(Ordering::SeqCst);
        if is_running {
            error!("[GraphService::new] 🚨 CRITICAL: A simulation loop is already running with ID: {}! Creating a new GraphService without shutting down the previous one may cause dual simulation loops.", *guard);
            warn!("[GraphService::new] Current simulation ID: {} will replace previous ID: {}", simulation_id, *guard);
        }
        
        // Create the shared node map
        let node_map = Arc::new(RwLock::new(HashMap::new()));

        if gpu_compute.is_some() {
            info!("[GraphService] GPU compute is enabled - continuous physics simulation will run");
            info!("[GraphService] Testing GPU compute functionality at startup");
            tokio::spawn(Self::test_gpu_at_startup(gpu_compute.clone()));
        } else {
            error!("[GraphService] GPU compute is NOT enabled - physics simulation will use CPU fallback");
        }

        // Create shutdown signal
        let shutdown_requested = Arc::new(AtomicBool::new(false));
        // Create the GraphService with caching enabled 
        let _cache = Arc::new(RwLock::new(Option::<(Vec<Node>, Instant)>::None));
        
        // Create atomic flags for graph status tracking
        let is_initialized = Arc::new(AtomicBool::new(false));
        let is_physics_stable = Arc::new(AtomicBool::new(false));
        
        let graph_service = Self {
            graph_data: Arc::new(RwLock::new(GraphData::default())),
            shutdown_complete: Arc::new(AtomicBool::new(false)),
            node_map: node_map.clone(),
            gpu_compute,
            last_update: Arc::new(RwLock::new(Instant::now())),
            pending_updates: Arc::new(RwLock::new(HashMap::new())),
            node_positions_cache: Arc::new(RwLock::new(None)),
            cache_enabled: true,
            client_manager,
            is_initialized: Arc::new(AtomicBool::new(false)),
            simulation_id: simulation_id.clone(),
            shutdown_requested: shutdown_requested.clone(),
        };

        // Initialize the graph data from metadata if available
        // This runs immediately on startup, before any clients connect
        let graph_service_clone = graph_service.clone();
        tokio::spawn(async move {
            // Allow a small delay for other systems to initialize
            tokio::time::sleep(Duration::from_millis(500)).await;
            info!("[GraphService] Starting initial graph initialization");
            
            if let Err(e) = graph_service_clone.initialize_graph().await {
                error!("[GraphService] Failed to initialize graph data: {}", e);
            }
        });
        
        // Prepare for simulation loop
        let graph_data = Arc::clone(&graph_service.graph_data);
        let node_positions_cache = Arc::clone(&graph_service.node_positions_cache);
        let gpu_compute = graph_service.gpu_compute.clone();
        let loop_simulation_id = simulation_id.clone();
        
        // Log more detailed information about the GPU compute status
        if gpu_compute.is_some() {
            info!("[GraphService] 🔹 GPU compute is enabled and will be used for physics simulation (ID: {})", simulation_id);
            // Try to gather device information
            if let Some(gpu) = &gpu_compute {
                if let Ok(gpu_lock) = gpu.try_read() {
                    info!("[GraphService] GPU device information: iterations={} (ID: {})", gpu_lock.iteration_count, simulation_id);
                }
            }
        } else {
            warn!("[GraphService] 🔸 GPU compute is NOT available - will use CPU fallback for physics (ID: {})", simulation_id);
        }
        
        // Update the current simulation ID in the shared mutex
        *guard = simulation_id.clone();
        
        // Check if a simulation loop is already running and attempt to replace it
        if SIMULATION_LOOP_RUNNING.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
            warn!("[GraphService] Simulation loop already running, attempting to replace it (new ID: {})", simulation_id);
            // We're replacing an existing simulation, wait for the flag to be reset
            // by forcing a reset ourselves since we have the mutex
            SIMULATION_LOOP_RUNNING.store(false, Ordering::SeqCst);
            // Then set it again for our new loop
            SIMULATION_LOOP_RUNNING.store(true, Ordering::SeqCst);
        }
        
        // Release the mutex before spawning the task
        drop(guard);
        
        info!("[GraphService] Starting physics simulation loop (ID: {})", loop_simulation_id);
        
        // Clone graph_service twice - one for the async block and one for return
        let graph_service_clone = graph_service.clone();
        let return_service = graph_service.clone();
        tokio::spawn(async move {
            // Configure simulation parameters
            let params = SimulationParams {
                iterations: physics_settings.iterations, 
                spring_strength: physics_settings.spring_strength,
                repulsion: physics_settings.repulsion_strength,
                damping: physics_settings.damping,
                max_repulsion_distance: physics_settings.repulsion_distance,
                viewport_bounds: physics_settings.bounds_size,
                mass_scale: physics_settings.mass_scale,
                boundary_damping: physics_settings.boundary_damping,
                enable_bounds: physics_settings.enable_bounds,
                time_step: 0.016,  // ~60fps
                phase: SimulationPhase::Dynamic,
                mode: SimulationMode::Remote,
            };
            
            // Create a guard to reset the flag when the task exits
            let loop_guard = scopeguard::guard((), |_| { 
                info!("[Graph] Physics simulation loop exiting, resetting SIMULATION_LOOP_RUNNING flag (ID: {})", loop_simulation_id);
                // Use compare_exchange to safely reset the flag
                if SIMULATION_LOOP_RUNNING.compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    graph_service.shutdown_complete.store(true, Ordering::SeqCst);
                } else {
                    error!("[Graph] Failed to reset SIMULATION_LOOP_RUNNING flag - was already false (ID: {})", 
                           loop_simulation_id);
                }
            });
            
            loop {
                // Check if shutdown was requested
                if shutdown_requested.load(Ordering::SeqCst) {
                    info!("[Graph] Shutdown requested for simulation loop (ID: {})", loop_simulation_id);
                    break;
                }
                
                // Continuously update positions - using loop ID in logs to track which loop is running
                debug!("[Graph:{}] Starting physics calculation iteration", loop_simulation_id);
                let mut graph = graph_data.write().await;
                let mut node_map = node_map.write().await;

                let gpu_status = if gpu_compute.is_some() { "available" } else { "NOT available" };
                debug!("[Graph:{}] GPU compute status: {}, physics enabled: {}", 
                       loop_simulation_id, gpu_status, physics_settings.enabled);
                       
                if physics_settings.enabled {
                    let mut positions_stable = true;
                    
                    if let Some(gpu) = &gpu_compute {
                        if let Err(e) = Self::calculate_layout_with_retry(gpu, &mut graph, &mut node_map, &params).await {
                            error!("[Graph:{}] Error updating positions: {}", loop_simulation_id, e);
                            positions_stable = false;
                        } else {
                            debug!("[Graph:{}] GPU calculation completed successfully", loop_simulation_id);
                            debug!("[Graph:{}] Successfully calculated layout for {} nodes", loop_simulation_id, graph.nodes.len());
                            
                            // Broadcast position updates to all clients
                            Self::broadcast_positions(&graph_service_clone, &graph.nodes).await;
                        }
                    } else {
                        // Use CPU fallback when GPU is not available
                        debug!("[Graph:{}] GPU compute not available - using CPU fallback for physics calculation", loop_simulation_id);
                        if let Err(e) = Self::calculate_layout_cpu(&mut graph, &mut node_map, &params) {
                            error!("[Graph:{}] Error updating positions with CPU fallback: {}", loop_simulation_id, e);
                            positions_stable = false;
                        } else {
                            debug!("[Graph:{}] CPU calculation completed successfully", loop_simulation_id);
                            debug!("[Graph:{}] Successfully calculated layout with CPU fallback for {} nodes", loop_simulation_id, graph.nodes.len());
                            
                            // Broadcast position updates to all clients
                            Self::broadcast_positions(&graph_service_clone, &graph.nodes).await;
                        }
                    }
                } else {
                    debug!("[Graph:{}] Physics disabled in settings - skipping physics calculation", loop_simulation_id);
                }
                
                // Mark graph as initialized after first layout calculation
                is_initialized.store(true, Ordering::SeqCst);
                
                drop(graph); // Release locks before sleep
                drop(node_map);
                tokio::time::sleep(tokio::time::Duration::from_millis(16)).await;
                let mut cache = node_positions_cache.write().await;
                *cache = None;
            }
            drop(loop_guard); // Explicitly drop the guard to trigger the cleanup
        }); 

        return_service
    }
    
    // Helper method to check for update rate limiting
    async fn should_rate_limit(&self) -> bool {
        let now = Instant::now();
        let last = *self.last_update.read().await;
        if now.duration_since(last).as_millis() < UPDATE_RATE_LIMIT_MS as u128 {
            return true;
        }
        *self.last_update.write().await = now;
        false
    }

    // Helper method to resolve position conflicts
    fn resolve_position_conflict(current: &Node, update: &Node) -> Node {
        let mut resolved = current.clone();
        
        // Calculate position differences
        let dx = update.data.position.x - current.data.position.x;
        let dy = update.data.position.y - current.data.position.y;
        let dz = update.data.position.z - current.data.position.z;
        
        // If difference is significant, use update position
        if dx*dx + dy*dy + dz*dz > POSITION_CONFLICT_THRESHOLD*POSITION_CONFLICT_THRESHOLD {
            resolved.data.position = update.data.position.clone();
            
            // Average the velocities to smooth transitions
            resolved.data.velocity.x = (current.data.velocity.x + update.data.velocity.x) * 0.5;
            resolved.data.velocity.y = (current.data.velocity.y + update.data.velocity.y) * 0.5;
            resolved.data.velocity.z = (current.data.velocity.z + update.data.velocity.z) * 0.5;
        }
        
        // Preserve mass and flags from current node
        resolved.data.mass = current.data.mass;
        resolved.data.flags = current.data.flags;
        
        resolved
    }

    // Helper method to clean up old pending updates
    async fn cleanup_pending_updates(&self) {
        let mut pending = self.pending_updates.write().await;
        let now = Instant::now();
        pending.retain(|_, (_, timestamp)| {
            now.duration_since(*timestamp).as_millis() < UPDATE_RATE_LIMIT_MS as u128
        });
    }

    // Helper method to broadcast position updates to all clients
    async fn broadcast_positions(service: &GraphService, nodes: &[Node]) {
        if let Some(client_manager) = &service.client_manager {
            // Clone nodes for broadcasting
            let nodes_to_broadcast = nodes.to_vec();
            
            // Broadcast to all clients through the client manager
            client_manager.broadcast_node_positions(nodes_to_broadcast).await;
        } else {
            debug!("No client manager available for broadcasting positions");
        }
    }

    /// Shutdown the simulation loop to allow creating a new instance
    pub async fn shutdown(&self) {
        info!("[GraphService] Shutting down simulation loop (ID: {})", self.simulation_id);
        
        // Acquire the mutex to ensure we don't have race conditions during shutdown
        let guard = SIMULATION_MUTEX.lock().await;
        
        // Check if this is the currently running simulation
        if *guard != self.simulation_id {
            warn!("[GraphService] Cannot shutdown simulation - current running loop has different ID: {} (this instance ID: {})", 
                  *guard, self.simulation_id);
            return;
        }
        
        // Signal the loop to stop by setting the shutdown flag
        self.shutdown_requested.store(true, Ordering::SeqCst);
        info!("[GraphService] Set shutdown flag for simulation loop (ID: {})", self.simulation_id);
        
        // Reset shutdown complete flag before waiting
        self.shutdown_complete.store(false, Ordering::SeqCst);
        
        // Wait for the loop to fully exit with a 5 second timeout
        let max_attempts = SHUTDOWN_TIMEOUT_MS / 50; // 5 seconds total at 50ms intervals
        for attempt in 0..max_attempts {
            if !SIMULATION_LOOP_RUNNING.load(Ordering::SeqCst) {
                // Double check that shutdown is complete
                if self.shutdown_complete.load(Ordering::SeqCst) {
                    info!("[GraphService] Simulation loop successfully stopped (ID: {})", self.simulation_id);
                    return;
                }
            }
            
            // Log progress every second
            if attempt % 20 == 0 {
                info!("[GraphService] Waiting for simulation loop to stop (attempt {}/{})", attempt, max_attempts);
            }
            
            tokio::time::sleep(Duration::from_millis(50)).await;
            if attempt == max_attempts - 1 {
                error!("[GraphService] Shutdown timeout after {}ms for simulation (ID: {})", 
                    SHUTDOWN_TIMEOUT_MS, self.simulation_id);
            }
        }
    }
    
    /// Get diagnostic information about the simulation status
    pub async fn get_simulation_diagnostics(&self) -> String {
        // Get the current simulation ID from the mutex
        let current_id = match SIMULATION_MUTEX.try_lock() {
            Ok(guard) => {
                let id = guard.clone();
                // Drop the guard immediately to avoid holding it
                drop(guard);
                id
            },
            Err(_) => "Unable to acquire mutex".to_string(),
        };
        
        // Check if this is the active simulation
        let is_active = current_id == self.simulation_id;
        
        // Check the global running flag
        let is_running = SIMULATION_LOOP_RUNNING.load(Ordering::SeqCst);
        
        // Check if shutdown has been requested for this instance
        let shutdown_requested = self.shutdown_requested.load(Ordering::SeqCst);
        
        format!(
            "Simulation Diagnostics:\n- This instance ID: {}\n- Current active ID: {}\n- Is this instance active: {}\n- Global running flag: {}\n- Shutdown requested: {}\n- Has GPU compute: {}",
            self.simulation_id,
            current_id,
            is_active,
            is_running,
            shutdown_requested,
            self.gpu_compute.is_some()
        )
    }
    
    /// Test GPU compute at startup to verify it's working
    async fn test_gpu_at_startup(gpu_compute: Option<Arc<RwLock<GPUCompute>>>) {
        // Add a small delay to let other initialization complete
        tokio::time::sleep(Duration::from_millis(1000)).await;
        
        info!("[GraphService] Running GPU startup test");
        
        if let Some(gpu) = &gpu_compute {
            match gpu.read().await.test_compute() {
                Ok(_) => {
                    info!("[GraphService] ✅ GPU test computation succeeded - GPU physics is working");
                },
                Err(e) => {
                    error!("[GraphService] ❌ GPU test computation failed: {}", e);
                    error!("[GraphService] The system will fall back to CPU physics which may be slower");
                    
                    // Try initializing a new GPU instance
                    info!("[GraphService] Attempting to reinitialize GPU...");
                    let _new_gpu = GPUCompute::new(&GraphData::default()).await; // Using _ to avoid unused warning
                }
            }
        } else {
            error!("[GraphService] ❌ No GPU compute instance available for testing");
        }
    }
    
    /// Wait for metadata file to be available (mounted by Docker)
    pub async fn wait_for_metadata_file() -> bool {
        info!("Checking for metadata file from Docker volume mount...");
        
        // Path to metadata file
        let metadata_path = std::path::Path::new("/app/data/metadata/metadata.json");
        
        // Start timer
        let start_time = Instant::now();
        let timeout = Duration::from_millis(METADATA_FILE_WAIT_TIMEOUT_MS);
        
        // Loop until timeout
        while start_time.elapsed() < timeout {
            // Check if file exists and is not empty
            match tokio::fs::metadata(&metadata_path).await {
                Ok(metadata) => {
                    if metadata.is_file() && metadata.len() > 0 {
                        // Try to open and validate the file
                        match TokioFile::open(&metadata_path).await {
                            Ok(_) => {
                                let elapsed = start_time.elapsed();
                                info!("Metadata file found and accessible after {:?}", elapsed);
                                return true;
                            }
                            Err(e) => {
                                debug!("Metadata file exists but couldn't be opened: {}", e);
                                // Continue waiting - might still be being written to
                            }
                        }
                    } else {
                        debug!("Metadata file exists but is empty or not a regular file");
                    }
                }
                Err(e) => {
                    debug!("Waiting for metadata file to be mounted: {}", e);
                }
            }
            
            // Sleep before checking again
            tokio::time::sleep(Duration::from_millis(METADATA_FILE_CHECK_INTERVAL_MS)).await;
        }
        
        // Timeout reached
        warn!("Timed out waiting for metadata file after {:?}", timeout);
        
        // Timeout reached, file not found or accessible
        false
    }

    pub async fn build_graph_from_metadata(metadata: &MetadataStore) -> Result<GraphData, Box<dyn std::error::Error + Send + Sync>> {
        // Check if a rebuild is already in progress
        info!("Building graph from {} metadata entries", metadata.len());
        debug!("Building graph from {} metadata entries", metadata.len());
        
        if GRAPH_REBUILD_IN_PROGRESS.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
            warn!("Graph rebuild already in progress, skipping duplicate rebuild");
            return Err("Graph rebuild already in progress".into());
        }      
        
        // Create a guard struct to ensure the flag is reset when this function returns
        struct RebuildGuard;
        impl Drop for RebuildGuard {
            fn drop(&mut self) {
                GRAPH_REBUILD_IN_PROGRESS.store(false, Ordering::SeqCst);
            }
        }
        // This guard will reset the flag when it goes out of scope
        debug!("Starting graph rebuild from metadata");
        let _guard = RebuildGuard;
        
        let mut graph = GraphData::new();
        let mut edge_map = HashMap::new();
        let mut node_map = HashMap::new();

        // Track used numeric IDs to prevent duplicates
        let mut used_numeric_ids = HashSet::new();
        let mut next_id_counter = 1; // Start at 1 to avoid node 0 issues
        // First pass: Create nodes from files in metadata
        let mut valid_nodes = HashSet::new();
        debug!("Creating nodes from {} metadata entries", metadata.len());
        for file_name in metadata.keys() {
            let node_id = file_name.trim_end_matches(".md").to_string();
            valid_nodes.insert(node_id);
        }
        debug!("Created valid_nodes set with {} nodes", valid_nodes.len());

        // Create nodes for all valid node IDs
        for node_id in &valid_nodes {
            // Skip if this node ID already has a numeric ID to prevent duplicates
            if graph.id_to_metadata.values().any(|id| id == node_id) {
                debug!("Skipping duplicate node for metadata ID: {}", node_id); 
                continue;
            }
            // Get metadata for this node, including the node_id if available
            let metadata_entry = graph.metadata.get(&format!("{}.md", node_id));
            let stored_node_id = metadata_entry.map(|m| m.node_id.clone());

            // Validate the stored node ID or generate a new one to guarantee uniqueness
            let final_node_id = if let Some(id_str) = stored_node_id {
                if let Ok(id_num) = id_str.parse::<u16>() {
                    if id_num == 0 || used_numeric_ids.contains(&id_str) {
                        // Don't use 0 or already used IDs
                        let new_id = next_id_counter.to_string();
                        next_id_counter += 1;
                        new_id
                    } else {
                        id_str
                    }
                } else {
                    let new_id = next_id_counter.to_string();
                    next_id_counter += 1;
                    new_id
                }
            } else {
                // No stored ID, generate a new unique one
                let new_id = next_id_counter.to_string();
                next_id_counter += 1;
                new_id
            };
            used_numeric_ids.insert(final_node_id.clone());
            
            // Create node with guaranteed unique ID
            let mut node = Node::new_with_id(node_id.clone(), Some(final_node_id));
            debug!("Created node {} with numeric ID {}", node_id, node.id);
            
            // CRITICAL: Double-check to ensure no node 0
            if node.id == "0" {
                debug!("Found node with ID 0 after creation, assigning new ID");
                // Generate a new unique ID
                let new_id = next_id_counter.to_string();
                next_id_counter += 1;
                
                // Create a new node with the same metadata but different ID
                let mut replacement_node = node.clone();
                replacement_node.id = new_id;
                node = replacement_node;
            }
            graph.id_to_metadata.insert(node.id.clone(), node_id.clone());

            // Get metadata for this node
            if let Some(metadata) = metadata.get(&format!("{}.md", node_id)) {
                // Set file size which also calculates mass
                node.set_file_size(metadata.file_size as u64);  // This will update both file_size and mass
                
                // Set the node label to the file name without extension
                // This will be used as the display name for the node
                node.label = metadata.file_name.trim_end_matches(".md").to_string();
                
                // Set visual properties from metadata
                node.size = Some(metadata.node_size as f32);
                
                // Add metadata fields to node's metadata map
                // Add all relevant metadata fields to ensure consistency
                node.metadata.insert("fileName".to_string(), metadata.file_name.clone());
                
                // Add name field (without .md extension) for client-side metadata ID mapping
                if metadata.file_name.ends_with(".md") {
                    let name = metadata.file_name[..metadata.file_name.len() - 3].to_string();
                    node.metadata.insert("name".to_string(), name.clone());
                    node.metadata.insert("metadataId".to_string(), name);
                } else {
                    node.metadata.insert("name".to_string(), metadata.file_name.clone());
                    node.metadata.insert("metadataId".to_string(), metadata.file_name.clone());
                }
                
                node.metadata.insert("fileSize".to_string(), metadata.file_size.to_string());
                node.metadata.insert("nodeSize".to_string(), metadata.node_size.to_string());
                node.metadata.insert("hyperlinkCount".to_string(), metadata.hyperlink_count.to_string());
                node.metadata.insert("sha1".to_string(), metadata.sha1.clone());
                node.metadata.insert("lastModified".to_string(), metadata.last_modified.to_string());
                
                if !metadata.perplexity_link.is_empty() {
                    node.metadata.insert("perplexityLink".to_string(), metadata.perplexity_link.clone());
                }
                
                if let Some(last_process) = metadata.last_perplexity_process {
                    node.metadata.insert("lastPerplexityProcess".to_string(), last_process.to_string());
                }
                
                // We don't add topic_counts to metadata as it would create circular references
                // and is already used to create edges
                
                // Ensure flags is set to 1 (default active state)
                node.data.flags = 1;
            }

            let node_clone = node.clone();
            graph.nodes.push(node_clone);
            // Store nodes in map by numeric ID for efficient lookups
            debug!("Added node with numeric ID: {}, metadata ID: {}", node.id, node.metadata_id);
            node_map.insert(node.id.clone(), node);
        }

        // Store metadata in graph
        debug!("Storing {} metadata entries in graph", metadata.len());
        graph.metadata = metadata.clone();
        debug!("Created {} nodes in graph", graph.nodes.len());
        
        // Final check for any duplicate node IDs that might have slipped through
        let mut unique_ids = HashSet::new();
        let original_count = graph.nodes.len();
        graph.nodes.retain(|node| unique_ids.insert(node.id.clone()));
        if original_count != graph.nodes.len() {
            debug!("Removed {} duplicate nodes in final validation", original_count - graph.nodes.len());
        }
        
        // Second pass: Create edges from topic counts
        for (source_file, metadata) in metadata.iter() {
            let source_id = source_file.trim_end_matches(".md").to_string();
            // Find the node with this metadata_id to get its numeric ID
            let source_node = graph.nodes.iter().find(|n| n.metadata_id == source_id);
            if source_node.is_none() {
                continue; // Skip if node not found
            }
            let source_numeric_id = source_node.unwrap().id.clone();
            
            debug!("Processing edges for source: {} (ID: {})", source_id, source_numeric_id);
            for (target_file, count) in &metadata.topic_counts {
                let target_id = target_file.trim_end_matches(".md").to_string();
                // Find the node with this metadata_id to get its numeric ID
                let target_node = graph.nodes.iter().find(|n| n.metadata_id == target_id);
                if target_node.is_none() {
                    continue; // Skip if node not found
                }
                let target_numeric_id = target_node.unwrap().id.clone();

                debug!("  Edge: {} -> {} (weight: {})", source_numeric_id, target_numeric_id, count);

                // Only create edge if both nodes exist and they're different
                if source_numeric_id != target_numeric_id {
                    let edge_key = if source_numeric_id < target_numeric_id {
                        (source_numeric_id.clone(), target_numeric_id.clone())
                    } else {
                        (target_numeric_id.clone(), source_numeric_id.clone())
                    };

                    edge_map.entry(edge_key)
                        .and_modify(|weight| *weight += *count as f32)
                        .or_insert(*count as f32);
                }
            }
        }

        // Convert edge map to edges
        debug!("Edge map contains {} unique connections", edge_map.len());
        for ((source, target), weight) in &edge_map {
            debug!("Edge map entry: {} -- {} (weight: {})", source, target, weight);
        }

        debug!("Converting edge map to {} edges", edge_map.len());
        graph.edges = edge_map.into_iter()
            .map(|((source, target), weight)| {
                Edge::new(source, target, weight)
            })
            .collect();

        // Initialize random positions
        Self::initialize_random_positions(&mut graph);

        info!("Built graph with {} nodes and {} edges", graph.nodes.len(), graph.edges.len());
        debug!("Completed graph build: {} nodes, {} edges", graph.nodes.len(), graph.edges.len());
        Ok(graph)
    }

    pub async fn build_graph(state: &web::Data<AppState>) -> Result<GraphData, Box<dyn std::error::Error + Send + Sync>> {
        info!("Building graph from app state");
        // Check if a rebuild is already in progress
        if GRAPH_REBUILD_IN_PROGRESS.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
            warn!("Graph rebuild already in progress, skipping duplicate rebuild");
            return Err("Graph rebuild already in progress".into());
        }
        
        // Create a guard struct to ensure the flag is reset when this function returns
        struct RebuildGuard;
        impl Drop for RebuildGuard {
            fn drop(&mut self) {
                GRAPH_REBUILD_IN_PROGRESS.store(false, Ordering::SeqCst);
            }
        }
        // This guard will reset the flag when it goes out of scope
        let _guard = RebuildGuard;
        
        let current_graph = state.graph_service.get_graph_data_mut().await;
        let mut graph = GraphData::new();
        let mut node_map = HashMap::new();
        debug!("Starting graph build process");
        
        // Track used numeric IDs to prevent duplicates
        let mut used_numeric_ids: HashSet<String> = HashSet::new();
        let mut next_id_counter = 1; // Start at 1 to avoid node 0 issues
        
        // Track all existing node IDs to prevent duplicates
        let mut existing_numeric_ids = HashSet::new();

        // Copy metadata from current graph
        graph.metadata = current_graph.metadata.clone();
        debug!("Copied {} metadata entries from current graph", graph.metadata.len());
        
        let mut edge_map = HashMap::new();
        
        // Copy id_to_metadata mapping to prevent duplicate node creation
        graph.id_to_metadata = current_graph.id_to_metadata.clone();

        // Create nodes from metadata entries - first build a set of valid nodes
        // to avoid creating duplicate nodes or nodes not tied to metadata
        debug!("Building node set from metadata entries");
        let mut valid_nodes = HashSet::new();
        for file_name in graph.metadata.keys() {
            let node_id = file_name.trim_end_matches(".md").to_string();
            valid_nodes.insert(node_id);
        }
        debug!("Created valid_nodes set with {} nodes", valid_nodes.len());

        // Create nodes for all valid node IDs
        for node_id in &valid_nodes {
            // Skip if this node ID already has a numeric ID to prevent duplicates
            if graph.id_to_metadata.values().any(|id| id == node_id) {
                debug!("Skipping duplicate node for metadata ID: {}", node_id);
                continue;
            }
            
            // Get metadata for this node, including the node_id if available
            let metadata_entry = graph.metadata.get(&format!("{}.md", node_id));
            let stored_node_id = metadata_entry.map(|m| m.node_id.clone());
            
            // Create node with stored ID or generate a new one if not available
            let mut node = Node::new_with_id(node_id.clone(), stored_node_id);
            graph.id_to_metadata.insert(node.id.clone(), node_id.clone());

            // CRITICAL: Check for node 0 and assign a new ID if found
            if node.id == "0" {
                debug!("Found node with ID 0, assigning new ID");
                // Generate a new unique ID
                let new_id = next_id_counter.to_string();
                next_id_counter += 1;
                
                // Create a new node with the same metadata but different ID
                let mut replacement_node = node.clone();
                replacement_node.id = new_id;
                node = replacement_node;
            }

            // Track numeric ID to avoid duplicates
            if !existing_numeric_ids.insert(node.id.clone()) {
                debug!("Detected duplicate numeric ID: {} for node: {}", node.id, node_id);
                continue;  // Skip this node as it has a duplicate ID
            }

            // Get metadata for this node
            if let Some(metadata) = graph.metadata.get(&format!("{}.md", node_id)) {
                // Set file size which also calculates mass
                node.set_file_size(metadata.file_size as u64);  // This will update both file_size and mass
                
                // Set the node label to the file name without extension
                // This will be used as the display name for the node
                node.label = metadata.file_name.trim_end_matches(".md").to_string();
                
                // Set visual properties from metadata
                node.size = Some(metadata.node_size as f32);
                
                // Add metadata fields to node's metadata map
                // Add all relevant metadata fields to ensure consistency
                node.metadata.insert("fileName".to_string(), metadata.file_name.clone());
                
                // Add name field (without .md extension) for client-side metadata ID mapping
                if metadata.file_name.ends_with(".md") {
                    let name = metadata.file_name[..metadata.file_name.len() - 3].to_string();
                    node.metadata.insert("name".to_string(), name.clone());
                    node.metadata.insert("metadataId".to_string(), name);
                } else {
                    node.metadata.insert("name".to_string(), metadata.file_name.clone());
                    node.metadata.insert("metadataId".to_string(), metadata.file_name.clone());
                }
                
                node.metadata.insert("fileSize".to_string(), metadata.file_size.to_string());
                node.metadata.insert("nodeSize".to_string(), metadata.node_size.to_string());
                node.metadata.insert("hyperlinkCount".to_string(), metadata.hyperlink_count.to_string());
                node.metadata.insert("sha1".to_string(), metadata.sha1.clone());
                node.metadata.insert("lastModified".to_string(), metadata.last_modified.to_string());
                
                if !metadata.perplexity_link.is_empty() {
                    node.metadata.insert("perplexityLink".to_string(), metadata.perplexity_link.clone());
                }
                
                if let Some(last_process) = metadata.last_perplexity_process {
                    node.metadata.insert("lastPerplexityProcess".to_string(), last_process.to_string());
                }
                
                // We don't add topic_counts to metadata as it would create circular references
                // and is already used to create edges
                
                // Ensure flags is set to 1 (default active state)
                node.data.flags = 1;
            }
            
            let node_clone = node.clone();
            graph.nodes.push(node_clone);
            // Store nodes in map by numeric ID for efficient lookups
            node_map.insert(node.id.clone(), node);
        }

        // Create edges from metadata topic counts
        for (source_file, metadata) in graph.metadata.iter() {
            let source_id = source_file.trim_end_matches(".md").to_string();
            debug!("Processing edges for source file: {}", source_file);
            // Find the node with this metadata_id to get its numeric ID
            let source_node = graph.nodes.iter().find(|n| n.metadata_id == source_id);
            if source_node.is_none() {
                continue; // Skip if node not found
            }
            let source_numeric_id = source_node.unwrap().id.clone();
            
            // Process outbound links from this file to other topics
            for (target_file, count) in &metadata.topic_counts {
                let target_id = target_file.trim_end_matches(".md").to_string();
                // Find the node with this metadata_id to get its numeric ID
                let target_node = graph.nodes.iter().find(|n| n.metadata_id == target_id);
                debug!("  Processing potential edge: {} -> {} (count: {})", source_id, target_id, count);
                if target_node.is_none() {
                    continue; // Skip if node not found
                }
                let target_numeric_id = target_node.unwrap().id.clone();
                debug!("  Found target node: {} (ID: {})", target_id, target_numeric_id);

                // Only create edge if both nodes exist and they're different
                if source_numeric_id != target_numeric_id {
                    let edge_key = if source_numeric_id < target_numeric_id {
                        (source_numeric_id.clone(), target_numeric_id.clone())
                    } else {
                        (target_numeric_id.clone(), source_numeric_id.clone())
                    };

                    debug!("  Creating/updating edge: {:?} with weight {}", edge_key, count);
                    // Sum the weights for bi-directional references
                    edge_map.entry(edge_key)
                        .and_modify(|w| *w += *count as f32)
                        .or_insert(*count as f32);
                }
            }
        }

        // Log edge_map contents before transformation
        debug!("Edge map contains {} unique connections", edge_map.len());
        for ((source, target), weight) in &edge_map {
            debug!("Edge map entry: {} -- {} (weight: {})", source, target, weight);
        }

        // Convert edge map to edges
        debug!("Converting edge map to {} edges", edge_map.len());
        graph.edges = edge_map.into_iter()
            .map(|((source, target), weight)| {
                Edge::new(source, target, weight)
            })
            .collect(); 
        
        // Final check for any duplicate node IDs that might have slipped through
        let mut unique_ids = HashSet::new();
        let original_count = graph.nodes.len();
        graph.nodes.retain(|node| unique_ids.insert(node.id.clone()));
        if original_count != graph.nodes.len() {
            debug!("Removed {} duplicate nodes in final validation", original_count - graph.nodes.len());
        }

        // Initialize random positions for all nodes
        Self::initialize_random_positions(&mut graph);

        info!("Built graph with {} nodes and {} edges", graph.nodes.len(), graph.edges.len());
        debug!("Completed graph build: {} nodes, {} edges", graph.nodes.len(), graph.edges.len());
        Ok(graph)
    }

    fn initialize_random_positions(graph: &mut GraphData) {
        let mut rng = rand::thread_rng();
        let node_count = graph.nodes.len() as f32;
        let initial_radius = 3.0; // Increasing radius for better visibility
        // Slightly vary the golden ratio to avoid perfect symmetry
        let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0 + rng.gen_range(-0.01..0.01);
        
        // Log the initialization process
        info!("Initializing random positions for {} nodes with radius {}", 
             node_count, initial_radius);
        info!("First 5 node numeric IDs: {}", graph.nodes.iter().take(5).map(|n| n.id.clone()).collect::<Vec<_>>().join(", "));
        info!("First 5 node metadata IDs: {}", graph.nodes.iter().take(5).map(|n| n.metadata_id.clone()).collect::<Vec<_>>().join(", "));
        
        // Use Fibonacci sphere distribution for more uniform initial positions
        for (i, node) in graph.nodes.iter_mut().enumerate() {
            let i_float: f32 = i as f32;
            
            // Calculate Fibonacci sphere coordinates
            // Add randomness to prevent perfect alignment
            let _i_perturbed = i_float + rng.gen_range(-0.2..0.2);
            let _golden_angle = 2.0 * std::f32::consts::PI / golden_ratio + rng.gen_range(-0.01..0.01);
            let theta = 2.0 * std::f32::consts::PI * i_float / golden_ratio;
            let phi = (1.0 - 2.0 * (i_float + 0.5) / node_count).acos();

            // Add slight randomness to prevent exact overlaps
            // CRITICAL FIX: Use different random values for each coordinate component
            // This prevents nodes from aligning along axis lines which was causing
            // node proliferation issues. By using independent randomization factors
            // for each coordinate, we ensure better 3D distribution and prevent
            // nodes from clustering along specific axes.
            let r_x = initial_radius * (0.85 + rng.gen_range(0.0..0.3));
            let r_y = initial_radius * (0.85 + rng.gen_range(0.0..0.3));
            let r_z = initial_radius * (0.85 + rng.gen_range(0.0..0.3));
            
            node.set_x(r_x * phi.sin() * theta.cos());
            node.set_y(r_y * phi.sin() * theta.sin());
            node.set_z(r_z * phi.cos());
            
            // Initialize with zero velocity
            node.set_vx(0.0);
            node.set_vy(0.0);
            node.set_vz(0.0);

            // Log first 5 nodes for debugging
            if i < 5 {
                info!("Initialized node {}: id={}, pos=[{:.3},{:.3},{:.3}]", 
                     i,
                     node.id,
                     node.data.position.x, 
                     node.data.position.y, 
                     node.data.position.z);
            }
        }
    }

    /// Helper function to retry GPU layout calculation with exponential backoff
    pub async fn calculate_layout_with_retry(
        gpu_compute: &Arc<RwLock<GPUCompute>>,
        graph: &mut GraphData,
        node_map: &mut HashMap<String, Node>, 
        params: &SimulationParams,
    ) -> std::io::Result<()> {
        debug!("[calculate_layout_with_retry] Starting GPU calculation with retry mechanism");
        let mut last_error: Option<Error> = None;
        
        for attempt in 0..MAX_GPU_CALCULATION_RETRIES {
            match Self::calculate_layout(gpu_compute, graph, node_map, params).await {
                Ok(()) => {
                    if attempt > 0 {
                        info!("[calculate_layout] Succeeded after {} retries", attempt);
                        debug!("[calculate_layout_with_retry] GPU calculation succeeded after retries");
                    }
                    return Ok(());
                }
                Err(e) => {
                    let delay = GPU_RETRY_DELAY_MS * (1 << attempt); // Exponential backoff
                    warn!("[calculate_layout] Failed (attempt {}/{}): {}. Retrying in {}ms...", 
                          attempt + 1, MAX_GPU_CALCULATION_RETRIES, e, delay);
                    last_error = Some(e);
                    
                    if attempt + 1 < MAX_GPU_CALCULATION_RETRIES {
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                    }
                }
            }
        }
        
        // If we get here, all attempts failed
        debug!("[calculate_layout_with_retry] All GPU attempts failed, falling back to CPU");
        error!("[calculate_layout] Failed after {} attempts, falling back to CPU", MAX_GPU_CALCULATION_RETRIES);
        
        // As a fallback, try CPU calculation when GPU fails repeatedly
        match Self::calculate_layout_cpu(graph, node_map, params) {
            Ok(()) => {
                info!("[calculate_layout] Successfully fell back to CPU calculation");
                Ok(())
            }
            Err(cpu_err) => {
                error!("[calculate_layout] CPU fallback also failed: {}", cpu_err);
                // Return the last GPU error as it's likely more relevant
                Err(last_error.unwrap_or_else(|| Error::new(ErrorKind::Other, 
                    format!("All {} GPU retry attempts failed and CPU fallback failed", MAX_GPU_CALCULATION_RETRIES))))
            }
        }
    }

    pub async fn calculate_layout(
        gpu_compute: &Arc<RwLock<GPUCompute>>,
        graph: &mut GraphData,
        node_map: &mut HashMap<String, Node>, 
        params: &SimulationParams,
    ) -> std::io::Result<()> {
        {
            info!("[calculate_layout] Starting GPU physics calculation for {} nodes, {} edges with mode {:?}", 
                  graph.nodes.len(), graph.edges.len(), params.mode);
            
            // Get current timestamp for performance tracking
            let start_time = std::time::Instant::now();

            let mut gpu_compute = gpu_compute.write().await;

            info!("[calculate_layout] params: iterations={}, spring_strength={:.3}, repulsion={:.3}, damping={:.3}",
                 params.iterations, params.spring_strength, params.repulsion, params.damping);
            
            // Update data and parameters
            if let Err(e) = gpu_compute.update_graph_data(graph) {
                error!("[calculate_layout] Failed to update graph data in GPU: {}, node count: {}", 
                      e, graph.nodes.len());
                // Log more details about the graph for debugging
                if !graph.nodes.is_empty() {
                    debug!("First node: id={}, position=[{:.3},{:.3},{:.3}]", graph.nodes[0].id, graph.nodes[0].data.position.x, graph.nodes[0].data.position.y, graph.nodes[0].data.position.z);
                }
                return Err(e);
            }
            
            if let Err(e) = gpu_compute.update_simulation_params(params) {
                error!("[calculate_layout] Failed to update simulation parameters in GPU: {}", e);
                return Err(e);
            }
            
            // Perform computation step
            if let Err(e) = gpu_compute.step() {
                error!("[calculate_layout] Failed to execute physics step: {}, graph has {} nodes and {} edges", 
                       e, graph.nodes.len(), graph.edges.len());
                return Err(e);
            }
            
            // Get updated positions
            let updated_nodes = match gpu_compute.get_node_data() {
                Ok(nodes) => {
                    info!("[calculate_layout] Successfully retrieved {} nodes from GPU", nodes.len());
                    nodes
                },
                Err(e) => {
                    error!("[calculate_layout] Failed to get node data from GPU: {}", e);
                    return Err(e);
                }
            };
            
            // Update graph with new positions
            let mut nodes_updated = 0;
            for (i, node) in graph.nodes.iter_mut().enumerate() {
                if i >= updated_nodes.len() {
                    error!("[calculate_layout] Node index out of range: {} >= {}", i, updated_nodes.len());
                    continue;
                }
                
                // Update position and velocity from GPU data
                node.data = updated_nodes[i];
                nodes_updated += 1;
                
                // Update node_map as well
                if let Some(map_node) = node_map.get_mut(&node.id) {
                    map_node.data = updated_nodes[i];
                } else {
                    warn!("[calculate_layout] Node {} not found in node_map", node.id);
                }
            }
            
            // Log performance info
            let elapsed = start_time.elapsed();
            
                // Log sample positions for debugging (first 2 nodes)
                let sample_positions = if graph.nodes.len() >= 2 {
                    format!("[{:.2},{:.2},{:.2}], [{:.2},{:.2},{:.2}]", 
                        graph.nodes[0].data.position.x, graph.nodes[0].data.position.y, graph.nodes[0].data.position.z,
                        graph.nodes[1].data.position.x, graph.nodes[1].data.position.y, graph.nodes[1].data.position.z)
                } else if graph.nodes.len() == 1 {
                    format!("[{:.2},{:.2},{:.2}]", graph.nodes[0].data.position.x, graph.nodes[0].data.position.y, graph.nodes[0].data.position.z)
                } else { "no nodes".to_string() };
            
                info!("[calculate_layout] Updated positions for {}/{} nodes in {:?}. Sample positions: {}", nodes_updated, graph.nodes.len(), elapsed, sample_positions);
            
            Ok(())
        }
    }

    /// CPU fallback implementation of force-directed graph layout
    pub fn calculate_layout_cpu(
        graph: &mut GraphData,
        node_map: &mut HashMap<String, Node>,
        params: &SimulationParams,
    ) -> std::io::Result<()> {
        let nodes_len = graph.nodes.len();
        debug!("[calculate_layout_cpu] Starting CPU calculation with {} nodes", nodes_len);
        
        // Early return if there are no nodes to process
        if nodes_len == 0 {
            return Ok(());
        }
        
        // Initialize force accumulators for each node
        let mut forces = vec![(0.0, 0.0, 0.0); nodes_len];
        
        // Calculate repulsive forces between all pairs of nodes
        for i in 0..nodes_len {
            for j in (i+1)..nodes_len {
                let node_i = &graph.nodes[i];
                let node_j = &graph.nodes[j];
                
                // Calculate distance between nodes
                let dx = node_j.data.position.x - node_i.data.position.x;
                let dy = node_j.data.position.y - node_i.data.position.y;
                let dz = node_j.data.position.z - node_i.data.position.z;
                let distance_squared = dx * dx + dy * dy + dz * dz;
                
                // Avoid division by zero and limit maximum repulsion distance
                if distance_squared < 0.0001 { continue; }
                let distance = distance_squared.sqrt();
                if distance > params.max_repulsion_distance { continue; }
                
                // Calculate repulsion strength based on node masses (stored in data.mass) and distance
                let mass_i = (node_i.data.mass as f32 / 255.0) * 10.0 * params.mass_scale;
                let mass_j = (node_j.data.mass as f32 / 255.0) * 10.0 * params.mass_scale;
                let repulsion_factor = params.repulsion * mass_i * mass_j / distance_squared;
                
                // Normalize direction
                let nx = dx / distance;
                let ny = dy / distance;
                let nz = dz / distance;
                
                // Calculate forces (nodes push each other away)
                let fx = nx * repulsion_factor;
                let fy = ny * repulsion_factor;
                let fz = nz * repulsion_factor;
                
                // Apply forces to both nodes (equal and opposite)
                forces[i].0 -= fx;
                forces[i].1 -= fy;
                forces[i].2 -= fz;
                forces[j].0 += fx;
                forces[j].1 += fy;
                forces[j].2 += fz;
            }
        }
        
        // Calculate attractive forces for edges (spring forces)
        for edge in &graph.edges {
            let source_idx = graph.nodes.iter().position(|n| n.id == edge.source);
            let target_idx = graph.nodes.iter().position(|n| n.id == edge.target);
            
            if let (Some(i), Some(j)) = (source_idx, target_idx) {
                let node_i = &graph.nodes[i];
                let node_j = &graph.nodes[j];
                
                // Calculate distance between nodes
                let dx = node_j.data.position.x - node_i.data.position.x;
                let dy = node_j.data.position.y - node_i.data.position.y;
                let dz = node_j.data.position.z - node_i.data.position.z;
                let distance_squared = dx * dx + dy * dy + dz * dz;
                if distance_squared < 0.0001 { continue; }
                let distance = distance_squared.sqrt();
                
                // Spring force increases with distance and edge weight
                let spring_factor = params.spring_strength * edge.weight * distance;
                
                // Normalize direction
                let nx = dx / distance;
                let ny = dy / distance;
                let nz = dz / distance;
                
                // Calculate spring forces (edges pull nodes together)
                let fx = nx * spring_factor;
                let fy = ny * spring_factor;
                let fz = nz * spring_factor;
                
                // Apply spring forces 
                forces[i].0 += fx;
                forces[i].1 += fy;
                forces[i].2 += fz;
                forces[j].0 -= fx;
                forces[j].1 -= fy;
                forces[j].2 -= fz;
            }
        }
        
        // Update velocities and positions for all nodes
        for (i, node) in graph.nodes.iter_mut().enumerate() {            
            // Apply force to velocity with damping
            node.set_vx(node.data.velocity.x * params.damping + forces[i].0 * params.time_step);
            node.set_vy(node.data.velocity.y * params.damping + forces[i].1 * params.time_step);
            node.set_vz(node.data.velocity.z * params.damping + forces[i].2 * params.time_step);
            
            // Update position based on velocity
            node.set_x(node.data.position.x + node.data.velocity.x * params.time_step);
            node.set_y(node.data.position.y + node.data.velocity.y * params.time_step);
            node.set_z(node.data.position.z + node.data.velocity.z * params.time_step);
            
            // Update node_map as well
            if let Some(map_node) = node_map.get_mut(&node.id) {
                map_node.data = node.data.clone();
            }
        }
        
        Ok(())
    }

    pub async fn get_paginated_graph_data(
        &self,
        page: u32,
        page_size: u32,
    ) -> Result<PaginatedGraphData, Box<dyn std::error::Error + Send + Sync>> {
        let graph = self.graph_data.read().await;
        
        // Convert page and page_size to usize for vector operations
        let page = page as usize;
        let page_size = page_size as usize;
        let total_nodes = graph.nodes.len();
        
        let start = page * page_size;
        let end = std::cmp::min((page + 1) * page_size, total_nodes);

        let page_nodes: Vec<Node> = graph.nodes
            .iter()
            .skip(start)
            .take(end - start)
            .cloned() 
            .collect();

        // Get edges that connect to these nodes
        let node_ids: HashSet<String> = page_nodes.iter()
            .map(|n| n.id.clone())
            .collect();

        let edges: Vec<Edge> = graph.edges
            .iter()
            .filter(|e| node_ids.contains(&e.source) || node_ids.contains(&e.target))
            .cloned()
            .collect();

        Ok(PaginatedGraphData {
            nodes: page_nodes,
            edges,
            metadata: serde_json::to_value(graph.metadata.clone()).unwrap_or_default(),
            total_nodes,
            total_edges: graph.edges.len(),
            total_pages: ((total_nodes as f32 / page_size as f32).ceil()) as u32,
            current_page: page as u32,
        })
    }
    
    // Clear position cache to force a refresh on next request
    pub async fn clear_position_cache(&self) {
        let mut cache = self.node_positions_cache.write().await;
        *cache = None;
    }

    pub async fn get_node_positions(&self) -> Vec<Node> {
        let start_time = Instant::now();

        // First check if we have a valid cached result
        if self.cache_enabled {
            let cache = self.node_positions_cache.read().await;
            if let Some((cached_nodes, timestamp)) = &*cache {
                let age = start_time.duration_since(*timestamp);
                
                // If cache is still fresh, use it
                if age < Duration::from_millis(NODE_POSITION_CACHE_TTL_MS) {
                    debug!("Using cached node positions ({} nodes, age: {:?})", 
                           cached_nodes.len(), age);
                    return cached_nodes.clone();
                }
            }
        }

        // No valid cache, fetch from graph data
        let nodes = {
            let graph = self.graph_data.read().await;
            
            // Only log node position data in debug level
            debug!("get_node_positions: reading {} nodes from graph (cache miss)", graph.nodes.len());
            
            // Clone the nodes vector 
            graph.nodes.clone()
        };

        // Update cache with new result
        if self.cache_enabled {
            let mut cache = self.node_positions_cache.write().await;
            *cache = Some((nodes.clone(), start_time));
        }

        let elapsed = start_time.elapsed();
        debug!("Node position fetch completed in {:?} for {} nodes", elapsed, nodes.len());
        
        // Log first 5 nodes only when debug is enabled
        let sample_size = std::cmp::min(5, nodes.len());
        if sample_size > 0 && log::log_enabled!(log::Level::Debug) {
            debug!("Node position sample: {} samples of {} nodes", sample_size, nodes.len());
        }
        nodes
    }

    pub async fn get_graph_data_mut(&self) -> tokio::sync::RwLockWriteGuard<'_, GraphData> {
        self.graph_data.write().await
    }

    pub async fn get_node_map_mut(&self) -> tokio::sync::RwLockWriteGuard<'_, HashMap<String, Node>> {
        self.node_map.write().await
    }

    /// Fix any instances of node ID 0 throughout the graph
    pub async fn fix_node_zero_issue(&self) -> Result<usize, Error> {
        debug!("Checking for problematic node 0 instances");
        let mut graph = self.graph_data.write().await;
        let mut node_map = self.node_map.write().await;
        
        let mut counter = 0;
        let mut next_id = 100; // Start at higher number to avoid conflicts
        
        // Fix any node with ID 0
        for node in &mut graph.nodes {
            if node.id == "0" {
                debug!("Found node with ID 0, replacing with ID {}", next_id);
                let old_id = node.id.clone();
                node.id = next_id.to_string();
                next_id += 1;
                counter += 1;
                node_map.remove(&old_id);
                node_map.insert(node.id.clone(), node.clone());
            }
        }
        
        Ok(counter)
    }
    
    // Add a method to check for duplicate nodes and clean them
    pub async fn clean_duplicate_nodes(&self) -> Result<usize, Error> {
        debug!("Checking for duplicate nodes in the graph");
        let mut graph = self.graph_data.write().await;
        let mut node_map = self.node_map.write().await;
        
        // Create a set to track unique node IDs
        let mut unique_ids = HashSet::new();
        let original_count = graph.nodes.len();
        
        // Remove duplicate nodes (keeping the first instance)
        graph.nodes.retain(|node| unique_ids.insert(node.id.clone()));
        
        // Update node map to match
        *node_map = graph.nodes.iter().map(|n| (n.id.clone(), n.clone())).collect();
        let nodes_removed = original_count - graph.nodes.len();
        debug!("Removed {} duplicate nodes from graph", nodes_removed);
        Ok(nodes_removed)
    }
    
    // Add method to get GPU compute instance
    pub async fn get_gpu_compute(&self) -> Option<Arc<RwLock<GPUCompute>>> {
        self.gpu_compute.clone()
    }

    pub async fn update_node_positions(&self, updates: Vec<(u16, Node)>) -> Result<(), Error> {
        let mut graph = self.graph_data.write().await;
        let mut node_map = self.node_map.write().await;
        
        // Process node updates efficiently
        let mut updated_count = 0;
        let mut skipped_count = 0;
        
        // Process updates in batches
        for (node_id_u16, update_node) in updates {
            let node_id = node_id_u16.to_string(); 

            // Skip if this is a redundant update based on rate limiting
            if self.should_rate_limit().await {
                skipped_count += 1;
                continue;
            }
            
            // Apply update with conflict resolution if node exists
            if let Some(existing_node) = node_map.get_mut(&node_id) {
                // Create a new node with updated position/velocity but preserving other data
                let mut resolved_node = update_node.clone();
                
                // Preserve important attributes from existing node
                resolved_node.data.mass = existing_node.data.mass;
                resolved_node.data.flags = existing_node.data.flags;
                resolved_node.metadata = existing_node.metadata.clone();
                
                // Update the node in the map
                *existing_node = resolved_node;
                updated_count += 1;
            }
        }
        
        // Sync graph nodes with node_map
        graph.nodes.iter_mut().for_each(|node| {
            if let Some(map_node) = node_map.get(&node.id) {
                node.data = map_node.data.clone();
            }
        });
        
        // Broadcast all positions 
        Self::broadcast_positions(self, &graph.nodes).await;
        
        Ok(())
    }

    pub fn update_positions(&mut self) -> Pin<Box<dyn Future<Output = Result<(), Error>> + '_>> {
        Box::pin(async move {
            if let Some(gpu) = &self.gpu_compute {
                let mut gpu = gpu.write().await;
                gpu.compute_forces()?;
                Ok(())
            } else {
                // Initialize GPU if not already done
                if self.gpu_compute.is_none() {
                    let settings = Arc::new(RwLock::new(Settings::default()));
                    let graph_data = GraphData::default(); // Or get your actual graph data
                    self.initialize_gpu(settings, &graph_data).await?;
                    return self.update_positions().await;
                }
                Err(Error::new(ErrorKind::Other, "GPU compute not initialized"))
            }
        })
    }

    pub async fn initialize_gpu(&mut self, _settings: Arc<RwLock<Settings>>, graph_data: &GraphData) -> Result<(), Error> {
        info!("Initializing GPU compute system...");

        // If GPU is already initialized, don't reinitialize
        if self.gpu_compute.is_some() {
            info!("GPU compute is already initialized, skipping initialization");
            return Ok(());
        }

        match GPUCompute::new(graph_data).await {
            Ok(gpu_instance) => {
                // Try a test computation before accepting the GPU
                {
                    let mut gpu = gpu_instance.write().await;
                    if let Err(e) = gpu.compute_forces() {
                        error!("GPU test computation failed: {}", e);
                        return Err(Error::new(ErrorKind::Other, format!("GPU test computation failed: {}", e)));
                    }
                    info!("GPU test computation succeeded");
                }

                self.gpu_compute = Some(gpu_instance);
                info!("GPU compute system successfully initialized");
                Ok(())
            }
            Err(e) => {
                error!("Failed to initialize GPU compute: {}. Physics simulation will not work.", e);
                Err(Error::new(ErrorKind::Other, format!("GPU initialization failed: {}", e)))
            }
        }
    }

    /// Helper method to check GPU availability and print detailed diagnostics
    pub fn diagnose_gpu_status(gpu_compute: Option<Arc<RwLock<GPUCompute>>>) -> Pin<Box<dyn Future<Output = bool> + Send>> {
        Box::pin(async move {
            info!("[GraphService] Diagnosing GPU status...");
            
            match gpu_compute {
                Some(gpu) => {
                    info!("[GraphService] GPU compute is available in service");
                    // Try a test computation 
                    if let Ok(gpu_lock) = gpu.try_read() {
                        match gpu_lock.test_compute() {
                            Ok(_) => {
                                info!("[GraphService] GPU test computation succeeded");
                                true
                            },
                            Err(e) => {
                                error!("[GraphService] GPU test computation failed: {}", e);
                                false
                            }
                        }
                    } else {
                        info!("[GraphService] Could not acquire GPU lock for diagnostics");
                        false
                    }
                },
                None => {
                    error!("[GraphService] GPU compute is NOT available in service");
                    
                    // Try to initialize it
                    info!("[GraphService] Attempting to initialize GPU on demand...");
                    false
                }
            }
        })
    }

    // Development test function to verify metadata transfer
    #[cfg(test)]
    pub async fn test_metadata_transfer() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use chrono::Utc;
        use std::collections::HashMap;
        use crate::models::metadata::Metadata;

        // Create test metadata
        let mut metadata = crate::models::metadata::MetadataStore::new();
        let file_name = "test.md";
        
        // Create a test metadata entry
        let meta = Metadata {
            file_name: file_name.to_string(),
            file_size: 1000,
            node_size: 1.5,
            hyperlink_count: 5,
            sha1: "abc123".to_string(),
            node_id: "1".to_string(),
            last_modified: Utc::now(),
            perplexity_link: "https://example.com".to_string(),
            last_perplexity_process: Some(Utc::now()),
            topic_counts: HashMap::new(),
        };
        
        metadata.insert(file_name.to_string(), meta.clone());
        
        // Build graph from metadata
        let graph = Self::build_graph_from_metadata(&metadata).await?;
        
        // Check that the graph has one node with the correct metadata
        assert_eq!(graph.nodes.len(), 1);
        
        // Verify metadata_id
        let node = &graph.nodes[0];
        assert_eq!(node.metadata_id, "test");
        
        // Verify metadata fields
        assert!(node.metadata.contains_key("fileName"));
        assert_eq!(node.metadata.get("fileName").unwrap(), "test.md");
        
        assert!(node.metadata.contains_key("fileSize"));
        assert_eq!(node.metadata.get("fileSize").unwrap(), "1000");
        
        assert!(node.metadata.contains_key("nodeSize"));
        assert_eq!(node.metadata.get("nodeSize").unwrap(), "1.5");
        
        assert!(node.metadata.contains_key("hyperlinkCount"));
        assert_eq!(node.metadata.get("hyperlinkCount").unwrap(), "5");
        
        assert!(node.metadata.contains_key("sha1"));
        assert!(node.metadata.contains_key("lastModified"));
        
        // Check flags
        assert_eq!(node.data.flags, 1);

        println!("All metadata tests passed!");
        Ok(())
    }
}
