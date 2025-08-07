#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// CUDA math functions
__device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ inline float3 operator/(const float3& a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__device__ inline float3& operator*=(float3& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

__device__ inline float length(const float3& v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

extern "C" {
    // Vec3Data struct definition to match Rust's Vec3Data
    struct Vec3Data {
        float x;    // 4 bytes
        float y;    // 4 bytes
        float z;    // 4 bytes
    };

    // BinaryNodeData struct to match Rust's memory layout
    struct BinaryNodeData {
        Vec3Data position;    // 12 bytes
        Vec3Data velocity;    // 12 bytes
        unsigned char mass;   // 1 byte
        unsigned char flags;  // 1 byte
        unsigned char padding[2]; // 2 bytes
    };

    // EdgeData struct for communication intensity
    struct EdgeData {
        int source_idx;    // 4 bytes - source node index
        int target_idx;    // 4 bytes - target node index
        float weight;      // 4 bytes - communication intensity (0.0 to 1.0)
    };

    // Physics parameters for each graph type
    struct GraphPhysicsParams {
        float spring_k;
        float damping;
        float repel_k;
        float max_velocity;
        float cluster_strength;  // For knowledge graph topic clustering
        float communication_factor; // For agent graph dynamic connections
    };

    __global__ void compute_dual_graph_forces(
        BinaryNodeData* knowledge_nodes,
        BinaryNodeData* agent_nodes,
        EdgeData* knowledge_edges,
        EdgeData* agent_edges,
        int num_knowledge_nodes,
        int num_agent_nodes,
        int num_knowledge_edges,
        int num_agent_edges,
        GraphPhysicsParams knowledge_params,
        GraphPhysicsParams agent_params,
        float dt,
        float max_repulsion_dist,
        float viewport_bounds,
        int iteration_count,
        bool process_knowledge,  // Flag to enable/disable knowledge graph processing
        bool process_agents      // Flag to enable/disable agent graph processing
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Determine which graph this thread processes
        bool is_knowledge_node = idx < num_knowledge_nodes && process_knowledge;
        bool is_agent_node = idx >= num_knowledge_nodes && 
                            idx < (num_knowledge_nodes + num_agent_nodes) && 
                            process_agents;
        
        if (!is_knowledge_node && !is_agent_node) return;
        
        // Select appropriate node array and parameters
        BinaryNodeData* nodes;
        EdgeData* edges;
        int num_nodes;
        int num_edges;
        GraphPhysicsParams params;
        int local_idx;
        
        if (is_knowledge_node) {
            nodes = knowledge_nodes;
            edges = knowledge_edges;
            num_nodes = num_knowledge_nodes;
            num_edges = num_knowledge_edges;
            params = knowledge_params;
            local_idx = idx;
        } else {
            nodes = agent_nodes;
            edges = agent_edges;
            num_nodes = num_agent_nodes;
            num_edges = num_agent_edges;
            params = agent_params;
            local_idx = idx - num_knowledge_nodes;
        }
        
        if (local_idx >= num_nodes) return;
        
        const float MIN_DISTANCE = 0.15f;
        const float MAX_FORCE = is_knowledge_node ? 2.0f : 3.0f; // Gentler forces for knowledge graph
        
        // Progressive force application for stability
        const int WARMUP_ITERATIONS = 100;
        float ramp_up_factor = 1.0f;
        
        if (iteration_count < WARMUP_ITERATIONS) {
            ramp_up_factor = 0.01f + (iteration_count / (float)WARMUP_ITERATIONS) * 0.99f;
            // Higher damping during warmup
            params.damping = fmaxf(params.damping, 0.9f - 0.4f * (iteration_count / (float)WARMUP_ITERATIONS));
        }
        
        float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
        float3 pos = make_float3(
            nodes[local_idx].position.x, 
            nodes[local_idx].position.y, 
            nodes[local_idx].position.z
        );
        float3 vel = make_float3(
            nodes[local_idx].velocity.x, 
            nodes[local_idx].velocity.y, 
            nodes[local_idx].velocity.z
        );
        
        // Zero velocity in first iterations to prevent explosion
        if (iteration_count < 5) {
            vel = make_float3(0.0f, 0.0f, 0.0f);
        }
        
        // Convert mass from u8 to float
        float mass = nodes[local_idx].mass == 0 ? 0.5f : (nodes[local_idx].mass + 1.0f) / 256.0f;
        
        // Process edge-based spring forces
        for (int edge_idx = 0; edge_idx < num_edges; edge_idx++) {
            EdgeData edge = edges[edge_idx];
            
            int other_node_idx = -1;
            if (edge.source_idx == local_idx) {
                other_node_idx = edge.target_idx;
            } else if (edge.target_idx == local_idx) {
                other_node_idx = edge.source_idx;
            }
            
            if (other_node_idx >= 0 && other_node_idx < num_nodes) {
                float3 other_pos = make_float3(
                    nodes[other_node_idx].position.x,
                    nodes[other_node_idx].position.y,
                    nodes[other_node_idx].position.z
                );
                
                float3 diff = other_pos - pos;
                float dist = length(diff);
                
                if (dist > MIN_DISTANCE) {
                    float3 direction = diff / dist;
                    
                    // Spring force with weight-based strength
                    float spring_force = params.spring_k * edge.weight * (dist - 1.0f);
                    
                    // Knowledge graph: Add clustering force for related topics
                    if (is_knowledge_node) {
                        spring_force *= (1.0f + params.cluster_strength * edge.weight);
                    }
                    // Agent graph: Dynamic force based on communication intensity
                    else {
                        spring_force *= (1.0f + params.communication_factor * edge.weight);
                    }
                    
                    total_force += direction * spring_force;
                }
            }
        }
        
        // Process repulsion forces (all-to-all within same graph)
        for (int other_idx = 0; other_idx < num_nodes; other_idx++) {
            if (other_idx != local_idx) {
                float3 other_pos = make_float3(
                    nodes[other_idx].position.x,
                    nodes[other_idx].position.y,
                    nodes[other_idx].position.z
                );
                
                float3 diff = pos - other_pos;
                float dist = length(diff);
                
                if (dist < max_repulsion_dist && dist > MIN_DISTANCE) {
                    float3 direction = diff / dist;
                    float repulsion = params.repel_k / (dist * dist);
                    
                    // Knowledge graph: Stronger repulsion for unrelated nodes
                    if (is_knowledge_node) {
                        // Check if nodes are connected
                        bool connected = false;
                        for (int e = 0; e < num_edges; e++) {
                            EdgeData edge = edges[e];
                            if ((edge.source_idx == local_idx && edge.target_idx == other_idx) ||
                                (edge.target_idx == local_idx && edge.source_idx == other_idx)) {
                                connected = true;
                                break;
                            }
                        }
                        if (!connected) {
                            repulsion *= 1.5f; // Stronger repulsion for unconnected nodes
                        }
                    }
                    
                    total_force += direction * repulsion;
                }
            }
        }
        
        // Apply ramp-up factor during warmup
        total_force *= ramp_up_factor;
        
        // Clamp total force magnitude
        float force_magnitude = length(total_force);
        if (force_magnitude > MAX_FORCE) {
            total_force = (total_force / force_magnitude) * MAX_FORCE;
        }
        
        // Update velocity with damping
        vel += total_force * dt / mass;
        vel *= params.damping;
        
        // Clamp velocity
        float vel_magnitude = length(vel);
        if (vel_magnitude > params.max_velocity) {
            vel = (vel / vel_magnitude) * params.max_velocity;
        }
        
        // Update position
        pos += vel * dt;
        
        // Apply viewport bounds (keep nodes visible)
        pos.x = fmaxf(-viewport_bounds, fminf(viewport_bounds, pos.x));
        pos.y = fmaxf(-viewport_bounds, fminf(viewport_bounds, pos.y));
        pos.z = fmaxf(-viewport_bounds * 0.5f, fminf(viewport_bounds * 0.5f, pos.z));
        
        // Write back results
        nodes[local_idx].position.x = pos.x;
        nodes[local_idx].position.y = pos.y;
        nodes[local_idx].position.z = pos.z;
        nodes[local_idx].velocity.x = vel.x;
        nodes[local_idx].velocity.y = vel.y;
        nodes[local_idx].velocity.z = vel.z;
    }
    
    // Helper kernel to mark node types based on metadata
    __global__ void mark_node_types(
        BinaryNodeData* nodes,
        int num_nodes,
        unsigned char knowledge_flag,
        unsigned char agent_flag,
        bool* is_agent_array  // Array indicating if each node is an agent
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_nodes) return;
        
        // Set flags based on node type
        if (is_agent_array[idx]) {
            nodes[idx].flags |= agent_flag;
        } else {
            nodes[idx].flags |= knowledge_flag;
        }
    }
}