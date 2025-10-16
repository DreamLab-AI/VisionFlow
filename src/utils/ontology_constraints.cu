// CUDA Kernels for Ontology Constraint Physics
// GPU-accelerated constraint enforcement for multi-graph ontology simulations
// Target: ~2ms per frame for 10K nodes

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

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
    float padding[10];           // Align to 64 bytes
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
    return a.x * b.x + a.y * b.y + a.z * b.z;
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

    // Find source and target nodes
    int source_idx = -1;
    int target_idx = -1;

    for (int i = 0; i < num_nodes; i++) {
        if (nodes[i].node_id == constraint.source_id &&
            nodes[i].graph_id == constraint.graph_id) {
            source_idx = i;
        }
        if (nodes[i].node_id == constraint.target_id &&
            nodes[i].graph_id == constraint.graph_id) {
            target_idx = i;
        }
        if (source_idx >= 0 && target_idx >= 0) break;
    }

    if (source_idx < 0 || target_idx < 0) return;

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

    // Find source (subclass) and target (superclass) nodes
    int source_idx = -1;
    int target_idx = -1;

    for (int i = 0; i < num_nodes; i++) {
        if (nodes[i].node_id == constraint.source_id &&
            nodes[i].graph_id == constraint.graph_id) {
            source_idx = i;
        }
        if (nodes[i].node_id == constraint.target_id &&
            nodes[i].graph_id == constraint.graph_id) {
            target_idx = i;
        }
        if (source_idx >= 0 && target_idx >= 0) break;
    }

    if (source_idx < 0 || target_idx < 0) return;

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

    // Find source and target nodes
    int source_idx = -1;
    int target_idx = -1;

    for (int i = 0; i < num_nodes; i++) {
        if (nodes[i].node_id == constraint.source_id &&
            nodes[i].graph_id == constraint.graph_id) {
            source_idx = i;
        }
        if (nodes[i].node_id == constraint.target_id &&
            nodes[i].graph_id == constraint.graph_id) {
            target_idx = i;
        }
        if (source_idx >= 0 && target_idx >= 0) break;
    }

    if (source_idx < 0 || target_idx < 0) return;

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

    // Find source and target property nodes
    int source_idx = -1;
    int target_idx = -1;

    for (int i = 0; i < num_nodes; i++) {
        if (nodes[i].node_id == constraint.source_id &&
            nodes[i].graph_id == constraint.graph_id &&
            (nodes[i].ontology_type & ONTOLOGY_PROPERTY)) {
            source_idx = i;
        }
        if (nodes[i].node_id == constraint.target_id &&
            nodes[i].graph_id == constraint.graph_id &&
            (nodes[i].ontology_type & ONTOLOGY_PROPERTY)) {
            target_idx = i;
        }
        if (source_idx >= 0 && target_idx >= 0) break;
    }

    if (source_idx < 0 || target_idx < 0) return;

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
