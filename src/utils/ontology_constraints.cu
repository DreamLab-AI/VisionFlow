// CUDA Kernels for Ontology Constraint Physics
// GPU-accelerated constraint enforcement for multi-graph ontology simulations
// Target: ~2ms per frame for 10K nodes

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cstdint>

// 64-byte aligned data structures for optimal GPU memory access
struct OntologyNode {
    uint32_t graph_id;
    uint32_t node_id;
    uint32_t ontology_type;      // bits: class/individual/property
    uint32_t constraint_flags;
    float3 position;
    float3 velocity;
    float mass;
    float radius;
    uint32_t parent_class;
    uint32_t property_count;
    uint32_t padding[6];         // Align to 64 bytes
};

struct OntologyConstraint {
    uint32_t type;               // DisjointClasses=1, SubClassOf=2, etc
    uint32_t source_id;
    uint32_t target_id;
    uint32_t graph_id;
    float strength;
    float distance;
    // PERF: Pre-computed indices eliminate O(N) lookup per constraint
    // These are populated on host before kernel launch, converting O(N²) to O(N)
    int32_t source_idx;          // Pre-computed index into nodes array (-1 if not found)
    int32_t target_idx;          // Pre-computed index into nodes array (-1 if not found)
    float padding[8];            // Align to 64 bytes
};

// Constraint type constants
#define CONSTRAINT_DISJOINT_CLASSES 1
#define CONSTRAINT_SUBCLASS_OF 2
#define CONSTRAINT_SAMEAS 3
#define CONSTRAINT_INVERSE_OF 4
#define CONSTRAINT_FUNCTIONAL 5

// Ontology type flags
#define ONTOLOGY_CLASS 0x01
#define ONTOLOGY_INDIVIDUAL 0x02
#define ONTOLOGY_PROPERTY 0x04

// Performance constants
#define BLOCK_SIZE 256
#define EPSILON 1e-6f
#define MAX_FORCE 1000.0f

// Device helper functions
__device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline float dot(const float3& a, const float3& b) {
    // Use FMA for better performance and accuracy
    return fmaf(a.x, b.x, fmaf(a.y, b.y, a.z * b.z));
}

__device__ inline float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__device__ inline float3 normalize(const float3& v) {
    float len = length(v);
    if (len < EPSILON) return make_float3(0.0f, 0.0f, 0.0f);
    return v * (1.0f / len);
}

__device__ inline float3 clamp_force(const float3& force) {
    float mag = length(force);
    if (mag > MAX_FORCE) {
        return force * (MAX_FORCE / mag);
    }
    return force;
}

// Atomic add for float3 (requires atomicAdd for float)
__device__ inline void atomic_add_float3(float3* addr, const float3& val) {
    atomicAdd(&(addr->x), val.x);
    atomicAdd(&(addr->y), val.y);
    atomicAdd(&(addr->z), val.z);
}

// Kernel 1: DisjointClasses - Apply separation forces between disjoint class instances
// PERF: Uses pre-computed indices (O(1) lookup vs O(N) linear search)
__global__ void apply_disjoint_classes_kernel(
    OntologyNode* nodes,
    int num_nodes,
    OntologyConstraint* constraints,
    int num_constraints,
    float delta_time,
    float separation_strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_constraints) return;

    OntologyConstraint constraint = constraints[idx];

    if (constraint.type != CONSTRAINT_DISJOINT_CLASSES) return;

    // PERF: Use pre-computed indices instead of O(N) linear search
    // Indices are computed on host before kernel launch
    int source_idx = constraint.source_idx;
    int target_idx = constraint.target_idx;

    // Validate pre-computed indices
    if (source_idx < 0 || source_idx >= num_nodes ||
        target_idx < 0 || target_idx >= num_nodes) return;

    OntologyNode source = nodes[source_idx];
    OntologyNode target = nodes[target_idx];

    // Calculate repulsion force
    float3 delta = target.position - source.position;
    float dist = length(delta);
    float min_distance = source.radius + target.radius + constraint.distance;

    if (dist < min_distance && dist > EPSILON) {
        float3 direction = normalize(delta);
        float penetration = min_distance - dist;

        // Repulsion force: stronger when closer
        float force_magnitude = separation_strength * constraint.strength * penetration;
        float3 force = direction * (-force_magnitude);
        force = clamp_force(force);

        // Apply forces with mass consideration
        float3 source_accel = force * (1.0f / fmaxf(source.mass, EPSILON));
        float3 target_accel = force * (-1.0f / fmaxf(target.mass, EPSILON));

        // Update velocities
        atomic_add_float3(&nodes[source_idx].velocity, source_accel * delta_time);
        atomic_add_float3(&nodes[target_idx].velocity, target_accel * delta_time);
    }
}

// Kernel 2: SubClassOf - Apply hierarchical alignment forces
// PERF: Uses pre-computed indices (O(1) lookup vs O(N) linear search)
__global__ void apply_subclass_hierarchy_kernel(
    OntologyNode* nodes,
    int num_nodes,
    OntologyConstraint* constraints,
    int num_constraints,
    float delta_time,
    float alignment_strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_constraints) return;

    OntologyConstraint constraint = constraints[idx];

    if (constraint.type != CONSTRAINT_SUBCLASS_OF) return;

    // PERF: Use pre-computed indices instead of O(N) linear search
    int source_idx = constraint.source_idx;
    int target_idx = constraint.target_idx;

    // Validate pre-computed indices
    if (source_idx < 0 || source_idx >= num_nodes ||
        target_idx < 0 || target_idx >= num_nodes) return;

    OntologyNode source = nodes[source_idx];
    OntologyNode target = nodes[target_idx];

    // Calculate spring force towards ideal distance
    float3 delta = target.position - source.position;
    float dist = length(delta);
    float ideal_distance = constraint.distance;

    if (dist > EPSILON) {
        float3 direction = normalize(delta);
        float displacement = dist - ideal_distance;

        // Spring force: F = k * x
        float force_magnitude = alignment_strength * constraint.strength * displacement;
        float3 force = direction * force_magnitude;
        force = clamp_force(force);

        // Apply forces with mass consideration
        float3 source_accel = force * (1.0f / fmaxf(source.mass, EPSILON));
        float3 target_accel = force * (-1.0f / fmaxf(target.mass, EPSILON));

        // Update velocities
        atomic_add_float3(&nodes[source_idx].velocity, source_accel * delta_time);
        atomic_add_float3(&nodes[target_idx].velocity, target_accel * delta_time);
    }
}

// Kernel 3: SameAs - Apply co-location forces
// PERF: Uses pre-computed indices (O(1) lookup vs O(N) linear search)
__global__ void apply_sameas_colocate_kernel(
    OntologyNode* nodes,
    int num_nodes,
    OntologyConstraint* constraints,
    int num_constraints,
    float delta_time,
    float colocate_strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_constraints) return;

    OntologyConstraint constraint = constraints[idx];

    if (constraint.type != CONSTRAINT_SAMEAS) return;

    // PERF: Use pre-computed indices instead of O(N) linear search
    int source_idx = constraint.source_idx;
    int target_idx = constraint.target_idx;

    // Validate pre-computed indices
    if (source_idx < 0 || source_idx >= num_nodes ||
        target_idx < 0 || target_idx >= num_nodes) return;

    OntologyNode source = nodes[source_idx];
    OntologyNode target = nodes[target_idx];

    // Calculate strong attraction towards same position
    float3 delta = target.position - source.position;
    float dist = length(delta);

    if (dist > EPSILON) {
        float3 direction = normalize(delta);

        // Strong spring force to minimize distance
        float force_magnitude = colocate_strength * constraint.strength * dist;
        float3 force = direction * force_magnitude;
        force = clamp_force(force);

        // Apply forces with mass consideration
        float3 source_accel = force * (1.0f / fmaxf(source.mass, EPSILON));
        float3 target_accel = force * (-1.0f / fmaxf(target.mass, EPSILON));

        // Update velocities
        atomic_add_float3(&nodes[source_idx].velocity, source_accel * delta_time);
        atomic_add_float3(&nodes[target_idx].velocity, target_accel * delta_time);

        // Additional velocity damping to converge faster
        float damping = 0.95f;
        nodes[source_idx].velocity = nodes[source_idx].velocity * damping;
        nodes[target_idx].velocity = nodes[target_idx].velocity * damping;
    }
}

// Kernel 4: InverseOf - Apply symmetry enforcement
// PERF: Uses pre-computed indices (O(1) lookup vs O(N) linear search)
__global__ void apply_inverse_symmetry_kernel(
    OntologyNode* nodes,
    int num_nodes,
    OntologyConstraint* constraints,
    int num_constraints,
    float delta_time,
    float symmetry_strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_constraints) return;

    OntologyConstraint constraint = constraints[idx];

    if (constraint.type != CONSTRAINT_INVERSE_OF) return;

    // PERF: Use pre-computed indices instead of O(N) linear search
    int source_idx = constraint.source_idx;
    int target_idx = constraint.target_idx;

    // Validate pre-computed indices and property type
    if (source_idx < 0 || source_idx >= num_nodes ||
        target_idx < 0 || target_idx >= num_nodes) return;

    // Verify they are property nodes
    if (!(nodes[source_idx].ontology_type & ONTOLOGY_PROPERTY) ||
        !(nodes[target_idx].ontology_type & ONTOLOGY_PROPERTY)) return;

    OntologyNode source = nodes[source_idx];
    OntologyNode target = nodes[target_idx];

    // Calculate symmetry constraint
    float3 delta = target.position - source.position;
    float dist = length(delta);

    // For inverse properties, enforce symmetric positioning
    // Calculate midpoint and push nodes to be equidistant
    float3 midpoint = (source.position + target.position) * 0.5f;

    float3 source_to_mid = midpoint - source.position;
    float3 target_to_mid = midpoint - target.position;

    // Apply corrective forces
    float force_magnitude = symmetry_strength * constraint.strength;

    float3 source_force = source_to_mid * force_magnitude;
    float3 target_force = target_to_mid * force_magnitude;

    source_force = clamp_force(source_force);
    target_force = clamp_force(target_force);

    // Apply forces with mass consideration
    float3 source_accel = source_force * (1.0f / fmaxf(source.mass, EPSILON));
    float3 target_accel = target_force * (1.0f / fmaxf(target.mass, EPSILON));

    // Update velocities
    atomic_add_float3(&nodes[source_idx].velocity, source_accel * delta_time);
    atomic_add_float3(&nodes[target_idx].velocity, target_accel * delta_time);
}

// Kernel 5: FunctionalProperty - Apply cardinality constraints
__global__ void apply_functional_cardinality_kernel(
    OntologyNode* nodes,
    int num_nodes,
    OntologyConstraint* constraints,
    int num_constraints,
    float delta_time,
    float cardinality_penalty
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_nodes) return;

    OntologyNode node = nodes[idx];

    // Only apply to properties
    if (!(node.ontology_type & ONTOLOGY_PROPERTY)) return;

    // Count constraints involving this property
    int constraint_count = 0;
    float3 centroid = make_float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < num_constraints; i++) {
        OntologyConstraint constraint = constraints[i];

        if (constraint.type == CONSTRAINT_FUNCTIONAL &&
            constraint.graph_id == node.graph_id &&
            (constraint.source_id == node.node_id || constraint.target_id == node.node_id)) {

            constraint_count++;

            // Find the other node in the constraint
            uint32_t other_id = (constraint.source_id == node.node_id) ?
                                constraint.target_id : constraint.source_id;

            for (int j = 0; j < num_nodes; j++) {
                if (nodes[j].node_id == other_id &&
                    nodes[j].graph_id == node.graph_id) {
                    centroid = centroid + nodes[j].position;
                    break;
                }
            }
        }
    }

    // Functional property: at most one value
    // If property_count > 1, apply penalty force
    if (node.property_count > 1 && constraint_count > 0) {
        centroid = centroid * (1.0f / (float)constraint_count);

        float3 delta = centroid - node.position;
        float dist = length(delta);

        if (dist > EPSILON) {
            // Penalty force increases with cardinality violation
            float violation = (float)(node.property_count - 1);
            float force_magnitude = cardinality_penalty * violation;

            float3 direction = normalize(delta);
            float3 force = direction * force_magnitude;
            force = clamp_force(force);

            // Apply force
            float3 accel = force * (1.0f / fmaxf(node.mass, EPSILON));
            atomic_add_float3(&nodes[idx].velocity, accel * delta_time);

            // Additional damping to stabilize
            nodes[idx].velocity = nodes[idx].velocity * 0.9f;
        }
    }
}

// Host functions for kernel launch
extern "C" {

// PERF: Pre-compute constraint indices on host (O(N+M) instead of O(N*M) on GPU)
// This is called once before kernel launches and dramatically improves performance
// by converting O(N²) GPU lookups to O(1) indexed access
void precompute_constraint_indices(
    OntologyNode* h_nodes, int num_nodes,
    OntologyConstraint* h_constraints, int num_constraints
) {
    // Build node_id -> index lookup table (O(N))
    // Using simple array - could use hash map for very large node counts
    // Assumes node_ids are reasonably dense (< 10x num_nodes)

    // Find max node_id to size lookup table
    uint32_t max_node_id = 0;
    uint32_t max_graph_id = 0;
    for (int i = 0; i < num_nodes; i++) {
        if (h_nodes[i].node_id > max_node_id) max_node_id = h_nodes[i].node_id;
        if (h_nodes[i].graph_id > max_graph_id) max_graph_id = h_nodes[i].graph_id;
    }

    // Create lookup table: index = graph_id * (max_node_id+1) + node_id
    // This handles multi-graph scenarios where same node_id exists in different graphs
    size_t table_size = (size_t)(max_graph_id + 1) * (max_node_id + 1);

    // Limit table size to prevent excessive memory usage
    // For very sparse or large ID spaces, fall back to linear search
    if (table_size > 10000000) {
        // Fallback: O(N) per constraint on host (still better than O(N) per constraint on GPU)
        for (int c = 0; c < num_constraints; c++) {
            h_constraints[c].source_idx = -1;
            h_constraints[c].target_idx = -1;

            for (int n = 0; n < num_nodes; n++) {
                if (h_nodes[n].node_id == h_constraints[c].source_id &&
                    h_nodes[n].graph_id == h_constraints[c].graph_id) {
                    h_constraints[c].source_idx = n;
                }
                if (h_nodes[n].node_id == h_constraints[c].target_id &&
                    h_nodes[n].graph_id == h_constraints[c].graph_id) {
                    h_constraints[c].target_idx = n;
                }
                if (h_constraints[c].source_idx >= 0 && h_constraints[c].target_idx >= 0) break;
            }
        }
        return;
    }

    // Allocate and initialize lookup table
    int* lookup = (int*)malloc(table_size * sizeof(int));
    for (size_t i = 0; i < table_size; i++) {
        lookup[i] = -1;
    }

    // Populate lookup table (O(N))
    for (int i = 0; i < num_nodes; i++) {
        size_t key = (size_t)h_nodes[i].graph_id * (max_node_id + 1) + h_nodes[i].node_id;
        lookup[key] = i;
    }

    // Resolve all constraint indices (O(M))
    for (int c = 0; c < num_constraints; c++) {
        size_t source_key = (size_t)h_constraints[c].graph_id * (max_node_id + 1) + h_constraints[c].source_id;
        size_t target_key = (size_t)h_constraints[c].graph_id * (max_node_id + 1) + h_constraints[c].target_id;

        h_constraints[c].source_idx = lookup[source_key];
        h_constraints[c].target_idx = lookup[target_key];
    }

    free(lookup);
}

void launch_disjoint_classes_kernel(
    OntologyNode* d_nodes, int num_nodes,
    OntologyConstraint* d_constraints, int num_constraints,
    float delta_time, float separation_strength
) {
    int grid_size = (num_constraints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_disjoint_classes_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_nodes, num_nodes, d_constraints, num_constraints,
        delta_time, separation_strength
    );
}

void launch_subclass_hierarchy_kernel(
    OntologyNode* d_nodes, int num_nodes,
    OntologyConstraint* d_constraints, int num_constraints,
    float delta_time, float alignment_strength
) {
    int grid_size = (num_constraints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_subclass_hierarchy_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_nodes, num_nodes, d_constraints, num_constraints,
        delta_time, alignment_strength
    );
}

void launch_sameas_colocate_kernel(
    OntologyNode* d_nodes, int num_nodes,
    OntologyConstraint* d_constraints, int num_constraints,
    float delta_time, float colocate_strength
) {
    int grid_size = (num_constraints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_sameas_colocate_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_nodes, num_nodes, d_constraints, num_constraints,
        delta_time, colocate_strength
    );
}

void launch_inverse_symmetry_kernel(
    OntologyNode* d_nodes, int num_nodes,
    OntologyConstraint* d_constraints, int num_constraints,
    float delta_time, float symmetry_strength
) {
    int grid_size = (num_constraints + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_inverse_symmetry_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_nodes, num_nodes, d_constraints, num_constraints,
        delta_time, symmetry_strength
    );
}

void launch_functional_cardinality_kernel(
    OntologyNode* d_nodes, int num_nodes,
    OntologyConstraint* d_constraints, int num_constraints,
    float delta_time, float cardinality_penalty
) {
    int grid_size = (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_functional_cardinality_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_nodes, num_nodes, d_constraints, num_constraints,
        delta_time, cardinality_penalty
    );
}

} // extern "C"
