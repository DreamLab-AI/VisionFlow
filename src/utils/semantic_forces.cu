// Semantic Forces GPU Kernel - Type-aware physics for knowledge graphs
// Implements DAG layout, type clustering, collision detection, and attribute-weighted springs.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <cmath>

extern "C" {

// =============================================================================
// Configuration Structures
// =============================================================================

// DAG layout configuration
struct DAGConfig {
    float vertical_spacing;      // Vertical separation between hierarchy levels
    float horizontal_spacing;    // Minimum horizontal separation within a level
    float level_attraction;      // Strength of attraction to target level
    float sibling_repulsion;     // Repulsion between nodes at same level
    bool enabled;
};

// Type clustering configuration
struct TypeClusterConfig {
    float cluster_attraction;    // Attraction between nodes of same type
    float cluster_radius;        // Target radius for type clusters
    float inter_cluster_repulsion; // Repulsion between different type clusters
    bool enabled;
};

// Collision detection configuration
struct CollisionConfig {
    float min_distance;          // Minimum allowed distance between nodes
    float collision_strength;    // Force strength when colliding
    float node_radius;           // Default node radius
    bool enabled;
};

// Attribute-weighted spring configuration
struct AttributeSpringConfig {
    float base_spring_k;         // Base spring constant
    float weight_multiplier;     // Multiplier for edge weight influence
    float rest_length_min;       // Minimum rest length
    float rest_length_max;       // Maximum rest length
    bool enabled;
};

// Unified semantic configuration
struct SemanticConfig {
    DAGConfig dag;
    TypeClusterConfig type_cluster;
    CollisionConfig collision;
    AttributeSpringConfig attribute_spring;
};

// Global constant memory for semantic configuration
__constant__ SemanticConfig c_semantic_config;

// =============================================================================
// Helper Functions
// =============================================================================

__device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ inline float3 normalize(const float3& v) {
    float len = length(v);
    if (len > 1e-6f) {
        return make_float3(v.x / len, v.y / len, v.z / len);
    }
    return make_float3(0.0f, 0.0f, 0.0f);
}

// =============================================================================
// DAG Layout Kernel
// =============================================================================

// Apply hierarchical layout forces based on node hierarchy levels
__global__ void apply_dag_force(
    const int* node_hierarchy_levels,  // Hierarchy level for each node
    const int* node_types,             // Node type for each node
    float3* positions,                 // Current positions
    float3* forces,                    // Force accumulator
    const int num_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    if (!c_semantic_config.dag.enabled) return;

    int level = node_hierarchy_levels[idx];
    if (level < 0) return; // Skip nodes not in DAG

    // Calculate target Y position based on hierarchy level
    float target_y = level * c_semantic_config.dag.vertical_spacing;
    float current_y = positions[idx].y;
    float dy = target_y - current_y;

    // Apply vertical attraction to target level
    float3 level_force = make_float3(
        0.0f,
        dy * c_semantic_config.dag.level_attraction,
        0.0f
    );

    // Apply horizontal repulsion from siblings at same level
    float3 sibling_repulsion = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < num_nodes; i++) {
        if (i == idx) continue;
        if (node_hierarchy_levels[i] != level) continue;

        float3 delta = positions[idx] - positions[i];
        delta.y = 0.0f; // Only horizontal repulsion
        float dist = length(delta);

        if (dist < c_semantic_config.dag.horizontal_spacing && dist > 1e-6f) {
            float force_mag = c_semantic_config.dag.sibling_repulsion *
                            (c_semantic_config.dag.horizontal_spacing - dist) / dist;
            sibling_repulsion = sibling_repulsion + (delta * force_mag);
        }
    }

    // Accumulate forces
    atomicAdd(&forces[idx].x, level_force.x + sibling_repulsion.x);
    atomicAdd(&forces[idx].y, level_force.y + sibling_repulsion.y);
    atomicAdd(&forces[idx].z, level_force.z + sibling_repulsion.z);
}

// =============================================================================
// Type Clustering Kernel
// =============================================================================

// Apply type-based clustering forces
__global__ void apply_type_cluster_force(
    const int* node_types,             // Node type for each node
    const float3* type_centroids,      // Centroid position for each type
    float3* positions,                 // Current positions
    float3* forces,                    // Force accumulator
    const int num_nodes,
    const int num_types
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    if (!c_semantic_config.type_cluster.enabled) return;

    int node_type = node_types[idx];
    if (node_type < 0 || node_type >= num_types) return;

    // Attraction to type centroid
    float3 to_centroid = type_centroids[node_type] - positions[idx];
    float dist_to_centroid = length(to_centroid);

    float3 cluster_force = make_float3(0.0f, 0.0f, 0.0f);
    if (dist_to_centroid > c_semantic_config.type_cluster.cluster_radius) {
        // Outside cluster radius - attract inward
        float force_mag = c_semantic_config.type_cluster.cluster_attraction *
                        (dist_to_centroid - c_semantic_config.type_cluster.cluster_radius);
        cluster_force = normalize(to_centroid) * force_mag;
    }

    // Repulsion from nodes of different types
    float3 inter_cluster_repulsion = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < num_nodes; i++) {
        if (i == idx) continue;
        if (node_types[i] == node_type) continue; // Same type

        float3 delta = positions[idx] - positions[i];
        float dist = length(delta);

        if (dist < c_semantic_config.type_cluster.cluster_radius * 2.0f && dist > 1e-6f) {
            float force_mag = c_semantic_config.type_cluster.inter_cluster_repulsion / (dist * dist);
            inter_cluster_repulsion = inter_cluster_repulsion + (normalize(delta) * force_mag);
        }
    }

    // Accumulate forces
    atomicAdd(&forces[idx].x, cluster_force.x + inter_cluster_repulsion.x);
    atomicAdd(&forces[idx].y, cluster_force.y + inter_cluster_repulsion.y);
    atomicAdd(&forces[idx].z, cluster_force.z + inter_cluster_repulsion.z);
}

// =============================================================================
// Collision Detection Kernel
// =============================================================================

// Apply collision detection and response forces
__global__ void apply_collision_force(
    const float* node_radii,           // Radius for each node (can be NULL for default)
    float3* positions,                 // Current positions
    float3* forces,                    // Force accumulator
    const int num_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    if (!c_semantic_config.collision.enabled) return;

    float radius_a = node_radii ? node_radii[idx] : c_semantic_config.collision.node_radius;

    float3 collision_force = make_float3(0.0f, 0.0f, 0.0f);

    // Check collision with all other nodes
    for (int i = 0; i < num_nodes; i++) {
        if (i == idx) continue;

        float radius_b = node_radii ? node_radii[i] : c_semantic_config.collision.node_radius;
        float min_dist = radius_a + radius_b + c_semantic_config.collision.min_distance;

        float3 delta = positions[idx] - positions[i];
        float dist = length(delta);

        if (dist < min_dist && dist > 1e-6f) {
            // Collision detected - apply repulsion
            float overlap = min_dist - dist;
            float force_mag = c_semantic_config.collision.collision_strength * overlap;
            collision_force = collision_force + (normalize(delta) * force_mag);
        }
    }

    // Accumulate forces
    atomicAdd(&forces[idx].x, collision_force.x);
    atomicAdd(&forces[idx].y, collision_force.y);
    atomicAdd(&forces[idx].z, collision_force.z);
}

// =============================================================================
// Attribute-Weighted Spring Kernel
// =============================================================================

// Apply spring forces weighted by edge attributes
__global__ void apply_attribute_spring_force(
    const int* edge_sources,           // Source node index for each edge
    const int* edge_targets,           // Target node index for each edge
    const float* edge_weights,         // Weight/strength for each edge
    const int* edge_types,             // Type for each edge
    float3* positions,                 // Current positions
    float3* forces,                    // Force accumulator
    const int num_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;

    if (!c_semantic_config.attribute_spring.enabled) return;

    int src = edge_sources[idx];
    int tgt = edge_targets[idx];
    float weight = edge_weights[idx];

    // Calculate spring force
    float3 delta = positions[tgt] - positions[src];
    float dist = length(delta);

    if (dist < 1e-6f) return;

    // Weight influences spring constant and rest length
    float spring_k = c_semantic_config.attribute_spring.base_spring_k *
                    (1.0f + weight * c_semantic_config.attribute_spring.weight_multiplier);

    // Rest length inversely proportional to weight (stronger edges = shorter rest length)
    float rest_length = c_semantic_config.attribute_spring.rest_length_max -
                       (weight * (c_semantic_config.attribute_spring.rest_length_max -
                                c_semantic_config.attribute_spring.rest_length_min));

    // Hooke's law: F = -k * (x - x0)
    float displacement = dist - rest_length;
    float force_mag = spring_k * displacement;

    float3 spring_force = normalize(delta) * force_mag;

    // Apply equal and opposite forces
    atomicAdd(&forces[src].x, spring_force.x);
    atomicAdd(&forces[src].y, spring_force.y);
    atomicAdd(&forces[src].z, spring_force.z);

    atomicAdd(&forces[tgt].x, -spring_force.x);
    atomicAdd(&forces[tgt].y, -spring_force.y);
    atomicAdd(&forces[tgt].z, -spring_force.z);
}

// =============================================================================
// Hierarchy Level Calculation (Utility)
// =============================================================================

// Calculate hierarchy levels for DAG layout using BFS-style parallel approach
__global__ void calculate_hierarchy_levels(
    const int* edge_sources,           // Source node index for each edge
    const int* edge_targets,           // Target node index for each edge
    const int* edge_types,             // Edge type (use hierarchy type = 2)
    int* node_levels,                  // Output: hierarchy level for each node
    bool* changed,                     // Flag indicating if any level changed
    const int num_edges,
    const int num_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;

    int edge_type = edge_types[idx];

    // Only process hierarchy edges (type 2 = Hierarchy)
    if (edge_type != 2) return;

    int parent = edge_sources[idx];
    int child = edge_targets[idx];

    int parent_level = node_levels[parent];
    if (parent_level >= 0) {
        int new_child_level = parent_level + 1;
        int old_child_level = atomicMax(&node_levels[child], new_child_level);

        if (old_child_level < new_child_level) {
            *changed = true;
        }
    }
}

// =============================================================================
// Type Centroid Calculation (Utility)
// =============================================================================

// Calculate centroid positions for each node type
__global__ void calculate_type_centroids(
    const int* node_types,             // Node type for each node
    const float3* positions,           // Current positions
    float3* type_centroids,            // Output: centroid for each type
    int* type_counts,                  // Output: count for each type
    const int num_nodes,
    const int num_types
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    int node_type = node_types[idx];
    if (node_type < 0 || node_type >= num_types) return;

    // Atomic add to accumulate positions
    atomicAdd(&type_centroids[node_type].x, positions[idx].x);
    atomicAdd(&type_centroids[node_type].y, positions[idx].y);
    atomicAdd(&type_centroids[node_type].z, positions[idx].z);
    atomicAdd(&type_counts[node_type], 1);
}

// Finalize centroids by dividing by count
__global__ void finalize_type_centroids(
    float3* type_centroids,            // Centroids to finalize
    const int* type_counts,            // Count for each type
    const int num_types
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_types) return;

    int count = type_counts[idx];
    if (count > 0) {
        type_centroids[idx].x /= count;
        type_centroids[idx].y /= count;
        type_centroids[idx].z /= count;
    }
}

// =============================================================================
// Configuration Setup
// =============================================================================

// Upload semantic configuration to constant memory
void set_semantic_config(const SemanticConfig* config) {
    cudaMemcpyToSymbol(c_semantic_config, config, sizeof(SemanticConfig));
}

} // extern "C"
