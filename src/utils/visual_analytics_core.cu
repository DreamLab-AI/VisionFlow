// Visual Analytics Core - Research-Grade GPU Implementation
// Temporal-Spatial Force Webs with Real-Time Element Isolation
// Designed for A6000 (48GB VRAM, 10752 cores, 768 GB/s bandwidth)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cub/cub.cuh>
#include <cmath>

namespace cg = cooperative_groups;

extern "C" {

// ============================================================================
// CORE DATA STRUCTURES - Designed for GPU-First Architecture
// ============================================================================

// Constants for A6000 optimization
#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 1024
#define SHARED_MEM_SIZE 49152  // 48KB shared memory per SM
#define MAX_LAYERS 16          // Visual isolation layers
#define TEMPORAL_WINDOW 128    // Frames of history
#define MAX_COMMUNITIES 256    
#define TOPOLOGY_FEATURES 32   // Topological descriptors per node

// Advanced 4D vector (3D + time)
struct Vec4 {
    float x, y, z, t;
    
    __device__ inline float spatial_length() const {
        return sqrtf(x*x + y*y + z*z);
    }
    
    __device__ inline float temporal_distance(const Vec4& other) const {
        return fabsf(t - other.t);
    }
};

// Temporal-Spatial Node - Core visualization element
struct TSNode {
    // Current state
    Vec4 position;           // 3D position + time
    Vec4 velocity;          
    Vec4 acceleration;
    
    // Temporal dynamics
    Vec4 trajectory[8];      // Recent trajectory (compressed)
    float temporal_coherence;
    float motion_saliency;
    
    // Multi-resolution hierarchy
    int hierarchy_level;     // 0=leaf, higher=abstract
    int parent_idx;
    int children[4];         // Quad-tree style
    float lod_importance;    // Level-of-detail importance
    
    // Visual isolation layers
    float layer_membership[MAX_LAYERS];  // Soft assignment to layers
    int primary_layer;
    float isolation_strength;
    
    // Topological features
    float topology[TOPOLOGY_FEATURES];
    float betweenness_centrality;
    float clustering_coefficient;
    float pagerank;
    int community_id;
    
    // Semantic embeddings (compressed)
    float semantic_vector[16];  // PCA-reduced embedding
    float semantic_drift;       // Change over time
    
    // Visual importance
    float visual_saliency;
    float information_content;
    float attention_weight;
    
    // Force modulation
    float force_scale;
    float damping_local;
    int constraint_mask;
};

// Enhanced edge with temporal dynamics
struct TSEdge {
    int source, target;
    
    // Multi-dimensional weights
    float structural_weight;
    float semantic_weight;
    float temporal_weight;
    float causal_weight;      // Directional causality
    
    // Temporal dynamics
    float weight_history[8];   // Recent weight changes
    float formation_time;      // When edge was created
    float stability;           // How stable over time
    
    // Visual properties
    float bundling_strength;   // For edge bundling
    Vec4 control_points[2];    // Bezier control points
    int layer_mask;            // Which layers to show in
    
    // Information flow
    float information_flow;    // Directed information transfer
    float latency;            // Communication delay
};

// GPU-optimized isolation layer
struct IsolationLayer {
    // Layer properties
    int layer_id;
    float opacity;
    float z_offset;           // Depth separation
    
    // Focus definition
    Vec4 focus_center;        // Center of attention
    float focus_radius;       // Region of interest
    float context_falloff;    // How quickly context fades
    
    // Filter criteria
    float importance_threshold;
    int community_filter;     // -1 for all
    int topology_filter_mask;
    float temporal_range[2];  // Time window
    
    // Visual style
    float force_modulation;
    float edge_opacity;
    int color_scheme;
};

// Unified parameters for visual analytics
struct VisualAnalyticsParams {
    // GPU optimization
    int total_nodes;
    int total_edges;
    int active_layers;
    int hierarchy_depth;
    
    // Temporal dynamics
    int current_frame;
    float time_step;
    float temporal_decay;
    float history_weight;
    
    // Force parameters (multi-resolution)
    float force_scale[4];     // Per hierarchy level
    float damping[4];
    float temperature[4];
    
    // Isolation and focus
    float isolation_strength;
    float focus_gamma;        // Focus enhancement
    int primary_focus_node;
    float context_alpha;
    
    // Visual comprehension
    float complexity_threshold;
    float saliency_boost;
    float information_bandwidth;
    
    // Topology analysis
    int community_algorithm;   // 0=Louvain, 1=InfoMap, 2=Spectral
    float modularity_resolution;
    int topology_update_interval;
    
    // Semantic analysis  
    float semantic_influence;
    float drift_threshold;
    int embedding_dims;
    
    // Viewport and interaction
    Vec4 camera_position;
    Vec4 viewport_bounds;
    float zoom_level;
    float time_window;
};

// ============================================================================
// SHARED MEMORY STRUCTURES - Optimized for Coalesced Access
// ============================================================================

__shared__ float shared_positions[MAX_BLOCK_SIZE * 4];  // x,y,z,t
__shared__ float shared_importance[MAX_BLOCK_SIZE];
__shared__ int shared_communities[MAX_BLOCK_SIZE];
__shared__ float shared_topology[WARP_SIZE * TOPOLOGY_FEATURES];

// ============================================================================
// ADVANCED FORCE CALCULATIONS - Research-Based Algorithms
// ============================================================================

// Multi-resolution force calculation with temporal coherence
__device__ Vec4 calculate_hierarchical_forces(
    int node_idx,
    const TSNode* nodes,
    const TSEdge* edges,
    int num_edges,
    const VisualAnalyticsParams& params,
    cg::thread_block block
) {
    const TSNode& node = nodes[node_idx];
    Vec4 force = {0, 0, 0, 0};
    
    // Adaptive neighborhood based on hierarchy level
    float influence_radius = 1000.0f * powf(2.0f, node.hierarchy_level);
    int samples = max(32, 256 >> node.hierarchy_level);  // Fewer samples for abstract nodes
    
    // Barnes-Hut style optimization - process by spatial tiles
    for (int tile = 0; tile < params.total_nodes; tile += MAX_BLOCK_SIZE) {
        // Cooperative loading into shared memory
        int tid = threadIdx.x;
        int other_idx = tile + tid;
        
        if (other_idx < params.total_nodes) {
            shared_positions[tid * 4] = nodes[other_idx].position.x;
            shared_positions[tid * 4 + 1] = nodes[other_idx].position.y;
            shared_positions[tid * 4 + 2] = nodes[other_idx].position.z;
            shared_positions[tid * 4 + 3] = nodes[other_idx].position.t;
            shared_importance[tid] = nodes[other_idx].visual_saliency;
            shared_communities[tid] = nodes[other_idx].community_id;
        }
        
        block.sync();
        
        // Process this tile
        int limit = min(MAX_BLOCK_SIZE, params.total_nodes - tile);
        for (int j = 0; j < limit; j++) {
            if (tile + j == node_idx) continue;
            
            Vec4 other_pos = {
                shared_positions[j * 4],
                shared_positions[j * 4 + 1],
                shared_positions[j * 4 + 2],
                shared_positions[j * 4 + 3]
            };
            
            // Spatial distance
            float dx = node.position.x - other_pos.x;
            float dy = node.position.y - other_pos.y;
            float dz = node.position.z - other_pos.z;
            float dist_sq = dx*dx + dy*dy + dz*dz;
            
            if (dist_sq > influence_radius * influence_radius) continue;
            
            float dist = sqrtf(dist_sq + 0.001f);
            
            // Temporal coherence factor
            float temporal_dist = fabsf(node.position.t - other_pos.t);
            float temporal_factor = expf(-temporal_dist * params.temporal_decay);
            
            // Community-based modulation
            float community_factor = 1.0f;
            if (node.community_id == shared_communities[j] && node.community_id >= 0) {
                community_factor = 2.0f;  // Stronger forces within communities
            }
            
            // Importance-weighted repulsion
            float importance = (node.visual_saliency + shared_importance[j]) * 0.5f;
            float repulsion = params.force_scale[node.hierarchy_level] * 
                            importance * community_factor * temporal_factor / dist_sq;
            
            // Apply force
            force.x += (dx / dist) * repulsion;
            force.y += (dy / dist) * repulsion;
            force.z += (dz / dist) * repulsion;
            
            // Temporal force (nodes should maintain temporal coherence)
            force.t += (node.position.t - other_pos.t) * temporal_factor * 0.01f;
        }
        
        block.sync();
    }
    
    return force;
}

// GPU-based topological feature extraction
__device__ void extract_topological_features(
    int node_idx,
    const TSNode* nodes,
    const TSEdge* edges,
    int num_edges,
    TSNode* output_node
) {
    // Calculate degree centrality
    int degree = 0;
    float weighted_degree = 0.0f;
    
    for (int e = 0; e < num_edges; e++) {
        if (edges[e].source == node_idx || edges[e].target == node_idx) {
            degree++;
            weighted_degree += edges[e].structural_weight;
        }
    }
    
    output_node->topology[0] = float(degree);
    output_node->topology[1] = weighted_degree;
    
    // Calculate local clustering coefficient
    // (Simplified for GPU - full implementation would use adjacency matrix)
    float clustering = 0.0f;
    if (degree > 1) {
        int triangles = 0;
        int possible_triangles = degree * (degree - 1) / 2;
        
        // Count triangles (simplified)
        for (int e1 = 0; e1 < num_edges; e1++) {
            if (edges[e1].source != node_idx && edges[e1].target != node_idx) continue;
            int neighbor1 = (edges[e1].source == node_idx) ? edges[e1].target : edges[e1].source;
            
            for (int e2 = e1 + 1; e2 < num_edges; e2++) {
                if (edges[e2].source != node_idx && edges[e2].target != node_idx) continue;
                int neighbor2 = (edges[e2].source == node_idx) ? edges[e2].target : edges[e2].source;
                
                // Check if neighbors are connected
                for (int e3 = 0; e3 < num_edges; e3++) {
                    if ((edges[e3].source == neighbor1 && edges[e3].target == neighbor2) ||
                        (edges[e3].source == neighbor2 && edges[e3].target == neighbor1)) {
                        triangles++;
                        break;
                    }
                }
            }
        }
        
        clustering = float(triangles) / float(possible_triangles);
    }
    
    output_node->topology[2] = clustering;
    output_node->clustering_coefficient = clustering;
}

// Focus + Context isolation with smooth transitions
__device__ float calculate_isolation_weight(
    const TSNode& node,
    const IsolationLayer& layer,
    const VisualAnalyticsParams& params
) {
    // Distance from focus center
    float dx = node.position.x - layer.focus_center.x;
    float dy = node.position.y - layer.focus_center.y;
    float dz = node.position.z - layer.focus_center.z;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    
    // Smooth falloff function (research: DOI-based fisheye)
    float focus_weight = 1.0f;
    if (dist > layer.focus_radius) {
        float falloff_dist = dist - layer.focus_radius;
        focus_weight = expf(-falloff_dist * layer.context_falloff);
    }
    
    // Importance filtering
    float importance_weight = (node.visual_saliency >= layer.importance_threshold) ? 1.0f : 0.2f;
    
    // Community filtering
    float community_weight = 1.0f;
    if (layer.community_filter >= 0 && node.community_id != layer.community_filter) {
        community_weight = 0.1f;
    }
    
    // Temporal filtering
    float temporal_weight = 1.0f;
    if (node.position.t < layer.temporal_range[0] || node.position.t > layer.temporal_range[1]) {
        temporal_weight = 0.05f;
    }
    
    // Combined isolation weight
    return focus_weight * importance_weight * community_weight * temporal_weight * layer.opacity;
}

// ============================================================================
// MAIN VISUAL ANALYTICS KERNEL - Orchestrates All Components
// ============================================================================

__global__ void visual_analytics_kernel(
    TSNode* nodes,
    const TSEdge* edges,
    const IsolationLayer* layers,
    const VisualAnalyticsParams params,
    float* output_positions,    // For rendering
    float* output_colors,        // For rendering
    float* output_importance     // For rendering
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.total_nodes) return;
    
    auto block = cg::this_thread_block();
    TSNode& node = nodes[idx];
    
    // 1. Calculate multi-resolution forces
    Vec4 force = calculate_hierarchical_forces(idx, nodes, edges, 
                                              params.total_edges, params, block);
    
    // 2. Apply edge-based forces
    for (int e = 0; e < params.total_edges; e++) {
        const TSEdge& edge = edges[e];
        int other = -1;
        
        if (edge.source == idx) other = edge.target;
        else if (edge.target == idx) other = edge.source;
        else continue;
        
        Vec4 other_pos = nodes[other].position;
        float dx = other_pos.x - node.position.x;
        float dy = other_pos.y - node.position.y;
        float dz = other_pos.z - node.position.z;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz + 0.001f);
        
        // Multi-dimensional spring force
        float ideal_dist = 100.0f * (2.0f - edge.semantic_weight);
        float spring = (dist - ideal_dist) * edge.structural_weight * 0.1f;
        
        force.x += (dx / dist) * spring;
        force.y += (dy / dist) * spring;
        force.z += (dz / dist) * spring;
        
        // Information flow creates directional bias
        if (edge.source == idx) {
            force.x += edge.information_flow * dx * 0.01f;
            force.y += edge.information_flow * dy * 0.01f;
        }
    }
    
    // 3. Update topological features periodically
    if (params.current_frame % params.topology_update_interval == 0) {
        extract_topological_features(idx, nodes, edges, params.total_edges, &node);
    }
    
    // 4. Calculate visual saliency
    node.visual_saliency = node.betweenness_centrality * 0.3f +
                          node.information_content * 0.3f +
                          node.motion_saliency * 0.2f +
                          node.semantic_drift * 0.2f;
    
    // 5. Apply isolation layers
    float total_isolation = 0.0f;
    Vec4 isolated_force = {0, 0, 0, 0};
    
    for (int l = 0; l < params.active_layers; l++) {
        float weight = calculate_isolation_weight(node, layers[l], params);
        total_isolation += weight;
        
        // Layer-specific force modulation
        isolated_force.x += force.x * weight * layers[l].force_modulation;
        isolated_force.y += force.y * weight * layers[l].force_modulation;
        isolated_force.z += force.z * weight * (layers[l].force_modulation + layers[l].z_offset);
    }
    
    if (total_isolation > 0.001f) {
        force = isolated_force;
        force.x /= total_isolation;
        force.y /= total_isolation;
        force.z /= total_isolation;
    }
    
    // 6. Temporal coherence maintenance
    node.temporal_coherence = 0.0f;
    for (int i = 0; i < 7; i++) {
        float dx = node.trajectory[i].x - node.trajectory[i+1].x;
        float dy = node.trajectory[i].y - node.trajectory[i+1].y;
        float dz = node.trajectory[i].z - node.trajectory[i+1].z;
        node.temporal_coherence += sqrtf(dx*dx + dy*dy + dz*dz);
    }
    node.temporal_coherence = 1.0f / (1.0f + node.temporal_coherence);
    
    // 7. Update dynamics with temporal smoothing
    float damping = params.damping[node.hierarchy_level];
    node.acceleration = force;
    node.velocity.x = node.velocity.x * (1.0f - damping) + node.acceleration.x * params.time_step;
    node.velocity.y = node.velocity.y * (1.0f - damping) + node.acceleration.y * params.time_step;
    node.velocity.z = node.velocity.z * (1.0f - damping) + node.acceleration.z * params.time_step;
    
    // 8. Update position with velocity clamping
    float vel_mag = node.velocity.spatial_length();
    if (vel_mag > 100.0f) {
        float scale = 100.0f / vel_mag;
        node.velocity.x *= scale;
        node.velocity.y *= scale;
        node.velocity.z *= scale;
    }
    
    node.position.x += node.velocity.x * params.time_step;
    node.position.y += node.velocity.y * params.time_step;
    node.position.z += node.velocity.z * params.time_step;
    node.position.t = float(params.current_frame);
    
    // 9. Update trajectory history
    for (int i = 7; i > 0; i--) {
        node.trajectory[i] = node.trajectory[i-1];
    }
    node.trajectory[0] = node.position;
    
    // 10. Calculate motion saliency
    float motion = node.velocity.spatial_length();
    node.motion_saliency = motion / (motion + 10.0f);  // Normalize
    
    // 11. Prepare output for rendering
    int out_idx = idx * 4;
    output_positions[out_idx] = node.position.x;
    output_positions[out_idx + 1] = node.position.y;
    output_positions[out_idx + 2] = node.position.z;
    output_positions[out_idx + 3] = node.position.t;
    
    // Color based on community and importance
    int color_idx = idx * 4;
    float hue = float(node.community_id % 12) / 12.0f;
    output_colors[color_idx] = hue;  // Hue
    output_colors[color_idx + 1] = 0.7f + 0.3f * node.visual_saliency;  // Saturation
    output_colors[color_idx + 2] = 0.5f + 0.5f * node.visual_saliency;  // Value
    output_colors[color_idx + 3] = total_isolation;  // Alpha
    
    output_importance[idx] = node.visual_saliency * total_isolation;
}

// ============================================================================
// COMMUNITY DETECTION - GPU Louvain Algorithm
// ============================================================================

__global__ void louvain_community_detection(
    TSNode* nodes,
    const TSEdge* edges,
    int num_edges,
    float* modularity,
    int iteration
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= gridDim.x * blockDim.x) return;
    
    // Simplified Louvain - full implementation would be more complex
    // This demonstrates the structure for GPU-based community detection
    
    TSNode& node = nodes[idx];
    
    // Calculate modularity gain for each neighbor community
    float best_gain = 0.0f;
    int best_community = node.community_id;
    
    for (int e = 0; e < num_edges; e++) {
        if (edges[e].source != idx && edges[e].target != idx) continue;
        
        int neighbor = (edges[e].source == idx) ? edges[e].target : edges[e].source;
        int neighbor_community = nodes[neighbor].community_id;
        
        if (neighbor_community != node.community_id) {
            // Calculate modularity gain (simplified)
            float gain = edges[e].structural_weight * edges[e].semantic_weight;
            
            if (gain > best_gain) {
                best_gain = gain;
                best_community = neighbor_community;
            }
        }
    }
    
    // Update community assignment
    if (best_gain > 0.01f) {
        node.community_id = best_community;
    }
}

// ============================================================================
// ENTRY POINTS - Interface with Rust
// ============================================================================

struct VisualAnalyticsContext {
    TSNode* d_nodes;
    TSEdge* d_edges;
    IsolationLayer* d_layers;
    float* d_output_positions;
    float* d_output_colors;
    float* d_output_importance;
    float* d_modularity;
    
    int max_nodes;
    int max_edges;
    int max_layers;
};

extern "C" {
    // Initialize visual analytics context
    VisualAnalyticsContext* init_visual_analytics(
        int max_nodes,
        int max_edges,
        int max_layers
    ) {
        VisualAnalyticsContext* ctx = new VisualAnalyticsContext;
        
        cudaMalloc(&ctx->d_nodes, max_nodes * sizeof(TSNode));
        cudaMalloc(&ctx->d_edges, max_edges * sizeof(TSEdge));
        cudaMalloc(&ctx->d_layers, max_layers * sizeof(IsolationLayer));
        cudaMalloc(&ctx->d_output_positions, max_nodes * 4 * sizeof(float));
        cudaMalloc(&ctx->d_output_colors, max_nodes * 4 * sizeof(float));
        cudaMalloc(&ctx->d_output_importance, max_nodes * sizeof(float));
        cudaMalloc(&ctx->d_modularity, sizeof(float));
        
        ctx->max_nodes = max_nodes;
        ctx->max_edges = max_edges;
        ctx->max_layers = max_layers;
        
        return ctx;
    }
    
    // Execute visual analytics pipeline
    void execute_visual_analytics(
        VisualAnalyticsContext* ctx,
        const VisualAnalyticsParams* params,
        int num_nodes,
        int num_edges,
        int num_layers
    ) {
        const int block_size = 256;
        const int grid_size = (num_nodes + block_size - 1) / block_size;
        
        // Run community detection periodically
        if (params->current_frame % 30 == 0) {
            louvain_community_detection<<<grid_size, block_size>>>(
                ctx->d_nodes, ctx->d_edges, num_edges,
                ctx->d_modularity, params->current_frame
            );
        }
        
        // Main visual analytics kernel
        visual_analytics_kernel<<<grid_size, block_size>>>(
            ctx->d_nodes, ctx->d_edges, ctx->d_layers,
            *params,
            ctx->d_output_positions,
            ctx->d_output_colors,
            ctx->d_output_importance
        );
        
        cudaDeviceSynchronize();
    }
    
    // Cleanup
    void destroy_visual_analytics(VisualAnalyticsContext* ctx) {
        cudaFree(ctx->d_nodes);
        cudaFree(ctx->d_edges);
        cudaFree(ctx->d_layers);
        cudaFree(ctx->d_output_positions);
        cudaFree(ctx->d_output_colors);
        cudaFree(ctx->d_output_importance);
        cudaFree(ctx->d_modularity);
        delete ctx;
    }
}

} // extern "C"