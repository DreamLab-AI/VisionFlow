#include <cuda_runtime.h>

extern "C" {
    // Vec3Data struct definition to match Rust's Vec3Data
    struct Vec3Data {
        float x;    // 4 bytes
        float y;    // 4 bytes
        float z;    // 4 bytes
    };

    // Updated BinaryNodeData struct to match Rust's memory layout
    // Previous version used arrays which caused memory layout mismatches
    struct BinaryNodeData {
        // Now using Vec3Data structs instead of arrays to match Rust memory layout
        Vec3Data position;    // 12 bytes - matches Rust Vec3Data struct
        Vec3Data velocity;    // 12 bytes - matches Rust Vec3Data struct

        // These fields remain unchanged and are still
        // used internally but not transmitted over the wire
        // The binary_protocol.rs still sets default values when decoding

        unsigned char mass;   // 1 byte  - matches Rust u8
        unsigned char flags;  // 1 byte  - matches Rust u8
        unsigned char padding[2]; // 2 bytes - matches Rust padding
    };

    // EdgeData struct to match Rust's EdgeData for communication intensity
    struct EdgeData {
        int source_idx;    // 4 bytes - source node index
        int target_idx;    // 4 bytes - target node index
        float weight;      // 4 bytes - communication intensity (0.0 to 1.0)
    };

    __global__ void compute_forces_kernel(
        BinaryNodeData* nodes,
        EdgeData* edges,
        int num_nodes,
        int num_edges,
        float spring_k,
        float damping,
        float repel_k,
        float dt,
        float max_repulsion_dist,
        float viewport_bounds,
        int iteration_count
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_nodes) return;

        const float MAX_FORCE = 3.0f; // Reduced maximum force magnitude
        const float MAX_VELOCITY = 0.02f; // Stricter velocity cap to prevent momentum buildup
        const float MIN_DISTANCE = 0.15f; // Slightly increased minimum distance

        // Progressive force application parameters
        // First 100 iterations use a ramp-up factor
        const int WARMUP_ITERATIONS = 100;
        float ramp_up_factor = 1.0f;

        if (iteration_count < WARMUP_ITERATIONS) {
            // Gradually increase from 0.01 to 1.0 over WARMUP_ITERATIONS
            ramp_up_factor = 0.01f + (iteration_count / (float)WARMUP_ITERATIONS) * 0.99f;

            // Also use higher damping in initial iterations to stabilize the system
            damping = fmaxf(damping, 0.9f - 0.4f * (iteration_count / (float)WARMUP_ITERATIONS));
        }

        float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
        float3 pos = make_float3(nodes[idx].position.x, nodes[idx].position.y, nodes[idx].position.z);
        float3 vel = make_float3(nodes[idx].velocity.x, nodes[idx].velocity.y, nodes[idx].velocity.z);

        // Zero out velocity in the very first iterations to prevent explosion
        if (iteration_count < 5) {
            vel = make_float3(0.0f, 0.0f, 0.0f);
        }

        // Convert mass from u8 to float (approximately 0-1 range)
        float mass;
        if (nodes[idx].mass == 0) {
            mass = 0.5f; // Default mid-range mass value
        } else {
            mass = (nodes[idx].mass + 1.0f) / 256.0f; // Add 1 to avoid zero mass
        }

        bool is_active = true; // All nodes are active by default

        if (!is_active) return; // Skip inactive nodes

        // Process edge-based interactions using communication intensity
        for (int edge_idx = 0; edge_idx < num_edges; edge_idx++) {
            EdgeData edge = edges[edge_idx];
            
            // Check if this edge involves the current node
            int other_node_idx = -1;
            
            if (edge.source_idx == idx) {
                other_node_idx = edge.target_idx;
            } else if (edge.target_idx == idx) {
                other_node_idx = edge.source_idx;
            }
            
            // Skip if edge doesn't involve current node
            if (other_node_idx == -1 || other_node_idx >= num_nodes) continue;

            // Get other node data
            float other_mass = (nodes[other_node_idx].mass == 0) ? 0.5f : (nodes[other_node_idx].mass + 1.0f) / 256.0f;

            float3 other_pos = make_float3(
                nodes[other_node_idx].position.x,
                nodes[other_node_idx].position.y,
                nodes[other_node_idx].position.z
            );

            float3 diff = make_float3(
                other_pos.x - pos.x,
                other_pos.y - pos.y,
                other_pos.z - pos.z
            );

            float dist = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
            
            // Only process if nodes are at a meaningful distance apart
            if (dist > MIN_DISTANCE) {
                float3 dir = make_float3(
                    diff.x / dist,
                    diff.y / dist,
                    diff.z / dist
                );

                // Use communication intensity (edge weight) to modulate spring force
                float communication_intensity = edge.weight; // 0.0 to 1.0 range
                
                // Natural length scales with communication intensity
                // Higher communication = shorter desired distance (stronger attraction)
                float base_natural_length = 1.5f;
                float natural_length = base_natural_length * (1.0f - communication_intensity * 0.5f);

                // Spring force proportional to communication intensity
                float spring_force = -spring_k * ramp_up_factor * communication_intensity * (dist - natural_length);

                // Apply progressively stronger springs for very distant highly-communicating nodes
                if (dist > natural_length * 2.0f && communication_intensity > 0.5f) {
                    spring_force *= (1.0f + (dist - natural_length * 2.0f) * communication_intensity);
                }

                float spring_scale = mass * other_mass;
                float force_magnitude = spring_force * spring_scale;

                // Repulsion forces - weaker for high-communication pairs
                if (dist < max_repulsion_dist) {
                    float repel_scale = repel_k * mass * other_mass * (1.0f - communication_intensity * 0.7f);
                    float dist_sq = fmaxf(dist * dist, MIN_DISTANCE);
                    float repel_force = fminf(repel_scale / dist_sq, repel_scale * 2.0f);
                    
                    total_force.x -= dir.x * repel_force;
                    total_force.y -= dir.y * repel_force;
                    total_force.z -= dir.z * repel_force;
                } else {
                    // Apply communication-weighted spring forces
                    total_force.x -= dir.x * force_magnitude;
                    total_force.y -= dir.y * force_magnitude;
                    total_force.z -= dir.z * force_magnitude;
                }
            }
        }
        
        // Add weak repulsion between non-connected nodes to prevent clustering
        for (int j = 0; j < num_nodes; j++) {
            if (j == idx) continue;
            
            // Check if there's an edge between these nodes
            bool has_edge = false;
            for (int edge_idx = 0; edge_idx < num_edges; edge_idx++) {
                EdgeData edge = edges[edge_idx];
                if ((edge.source_idx == idx && edge.target_idx == j) || 
                    (edge.source_idx == j && edge.target_idx == idx)) {
                    has_edge = true;
                    break;
                }
            }
            
            // Apply weak repulsion only to non-connected nodes
            if (!has_edge) {
                float3 other_pos = make_float3(
                    nodes[j].position.x,
                    nodes[j].position.y,
                    nodes[j].position.z
                );
                
                float3 diff = make_float3(
                    other_pos.x - pos.x,
                    other_pos.y - pos.y,
                    other_pos.z - pos.z
                );
                
                float dist = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
                
                if (dist > 0.0f && dist < max_repulsion_dist * 2.0f) {
                    float3 dir = make_float3(diff.x / dist, diff.y / dist, diff.z / dist);
                    float weak_repel = repel_k * 0.1f * mass / (dist * dist + 0.1f);
                    
                    total_force.x -= dir.x * weak_repel;
                    total_force.y -= dir.y * weak_repel;
                    total_force.z -= dir.z * weak_repel;
                }
            }
        }

        // Stronger center gravity to prevent nodes from drifting too far
        float center_strength = 0.015f * mass * ramp_up_factor; // Apply ramp_up to center gravity too
        float center_dist = sqrtf(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);
        if (center_dist > 3.0f) { // Apply at shorter distances
            float center_factor = center_strength * (center_dist - 3.0f) / center_dist;
            total_force.x -= pos.x * center_factor;
            total_force.y -= pos.y * center_factor;
            total_force.z -= pos.z * center_factor;
        }

        // Calculate total force magnitude
        float force_magnitude = sqrtf(
            total_force.x*total_force.x +
            total_force.y*total_force.y +
            total_force.z*total_force.z);

        // Scale down excessive forces to prevent explosion
        if (force_magnitude > MAX_FORCE) {
            float scale_factor = MAX_FORCE / force_magnitude;
            total_force.x *= scale_factor;
            total_force.y *= scale_factor;
            total_force.z *= scale_factor;

            // Additional logging to help debug extreme forces after randomization
            // if (idx == 0 && iteration_count < 5)
            //     printf("Force clamped from %f to %f (iteration %d)\n", force_magnitude, MAX_FORCE, iteration_count);
        }

        // Apply damping and bounded forces to velocity
        vel.x = vel.x * (1.0f - damping) + fminf(MAX_FORCE, fmaxf(-MAX_FORCE, total_force.x)) * dt;
        vel.y = vel.y * (1.0f - damping) + fminf(MAX_FORCE, fmaxf(-MAX_FORCE, total_force.y)) * dt;
        vel.z = vel.z * (1.0f - damping) + fminf(MAX_FORCE, fmaxf(-MAX_FORCE, total_force.z)) * dt;

        // Apply STRICT velocity cap to prevent runaway momentum
        float vel_magnitude = sqrtf(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
        if (vel_magnitude > MAX_VELOCITY) {
            float scale_factor = MAX_VELOCITY / vel_magnitude;
            vel.x *= scale_factor;
            vel.y *= scale_factor;
            vel.z *= scale_factor;
        }

        // Update position
        pos.x += vel.x * dt;
        pos.y += vel.y * dt;
        pos.z += vel.z * dt;

        // Progressive boundary approach - stronger the further you go
        if (viewport_bounds > 0.0f && iteration_count > 10) { // Only apply boundary after initial stabilization
            float soft_margin = 0.3f * viewport_bounds; // 30% soft boundary
            float bound_with_margin = viewport_bounds - soft_margin;

            // Apply progressively stronger boundary forces
            if (fabsf(pos.x) > bound_with_margin) {
                pos.x *= 0.92f; // Pull back by 8%
                // Also add dampening to velocity in this direction
                vel.x *= 0.85f;
            }
            if (fabsf(pos.y) > bound_with_margin) {
                pos.y *= 0.92f; // Pull back by 8%
                vel.y *= 0.85f;
            }
            if (fabsf(pos.z) > bound_with_margin) {
                pos.z *= 0.92f; // Pull back by 8%
                vel.z *= 0.85f;
            }
        }

        // Store results back
        nodes[idx].position.x = pos.x;
        nodes[idx].position.y = pos.y;
        nodes[idx].position.z = pos.z;
        nodes[idx].velocity.x = vel.x;
        nodes[idx].velocity.y = vel.y;
        nodes[idx].velocity.z = vel.z;

        // Debug output for first node
        // if (idx == 0 && (iteration_count < 5 || iteration_count % 20 == 0)) {
        //     float force_mag = sqrtf(
        //         total_force.x * total_force.x +
        //         total_force.y * total_force.y +
        //         total_force.z * total_force.z
        //     );
        //     printf("Node %d: force_mag=%f, pos=(%f,%f,%f), vel=(%f,%f,%f)\n",
        //         idx, force_mag,
        //         pos.x, pos.y, pos.z,
        //         vel.x, vel.y, vel.z);

        //     // More detailed logging during initialization
        //     if (iteration_count < WARMUP_ITERATIONS)
        //         printf("Node %d: iteration=%d, ramp_up=%f, damping=%f\n", idx, iteration_count, ramp_up_factor, damping);
        // }
    }
}
