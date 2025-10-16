# Ontology Integration Protocol Design
**Version:** 2.0.0
**Date:** 2025-10-16
**Author:** Protocol Design Agent

## Executive Summary

This document defines backward-compatible client-server protocol updates and configuration schemas for ontology integration into the Hive Mind system. All changes maintain v1.x client compatibility through protocol version negotiation.

---

## 1. GraphType Enum Update

**Location:** `/home/devuser/workspace/project/src/models/graph_types.rs`

### Updated Enum
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GraphType {
    /// Standard node-edge graph
    Standard,
    /// Multi-agent system graph
    MultiAgent,
    /// Force-directed layout graph
    ForceDirected,
    /// Hierarchical graph structure
    Hierarchical,
    /// Network topology graph
    Network,
    /// OWL/RDF ontology graph with semantic constraints
    Ontology,  // NEW
}
```

### String Parsing Support
```rust
impl std::str::FromStr for GraphType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "standard" => Ok(GraphType::Standard),
            "multi-agent" | "multiagent" => Ok(GraphType::MultiAgent),
            "force-directed" | "forcedirected" => Ok(GraphType::ForceDirected),
            "hierarchical" => Ok(GraphType::Hierarchical),
            "network" => Ok(GraphType::Network),
            "ontology" => Ok(GraphType::Ontology),  // NEW
            _ => Err(format!("Unknown graph type: {}", s)),
        }
    }
}
```

**Serialization Format:**
- JSON: `"ontology"`
- Display: `"ontology"`
- Case-insensitive parsing

---

## 2. Protocol Message Format for Ontology Operations

### 2.1 WebSocket Message Structure

**Base Message Format** (extends `RealtimeWebSocketMessage`):
```json
{
  "type": "ontology_operation",
  "protocol_version": "2.0.0",
  "data": { ... },
  "timestamp": 1729123456789,
  "client_id": "uuid",
  "session_id": "uuid"
}
```

### 2.2 Ontology-Specific Message Types

#### Load Ontology Request
```json
{
  "type": "ontology_load",
  "protocol_version": "2.0.0",
  "data": {
    "graph_id": "ontology-graph-uuid",
    "graph_type": "ontology",
    "source": {
      "format": "turtle|rdf-xml|json-ld",
      "uri": "http://example.org/ontology.ttl",
      "content": "optional base64-encoded content"
    },
    "mapping_config": "optional path to mapping.toml",
    "physics_config": {
      "enable_constraints": true,
      "constraint_groups": ["disjoint", "hierarchy", "cardinality"]
    }
  }
}
```

#### Ontology Validation Status
```json
{
  "type": "ontology_validation",
  "protocol_version": "2.0.0",
  "data": {
    "graph_id": "ontology-graph-uuid",
    "status": "valid|invalid|processing",
    "consistency": true,
    "violations": [
      {
        "type": "disjoint_violation",
        "severity": "error|warning",
        "nodes": [123, 456],
        "message": "Class A and B are disjoint but share instances"
      }
    ],
    "metrics": {
      "class_count": 150,
      "property_count": 75,
      "axiom_count": 320,
      "reasoning_time_ms": 450
    }
  }
}
```

#### Constraint Update (Ontology-Specific)
```json
{
  "type": "ontology_constraint_update",
  "protocol_version": "2.0.0",
  "data": {
    "graph_id": "ontology-graph-uuid",
    "constraints": [
      {
        "type": "disjoint_separation",
        "class_uris": ["Class_A", "Class_B"],
        "min_distance": 50.0,
        "strength": 0.8
      },
      {
        "type": "hierarchy_alignment",
        "parent_uri": "SuperClass",
        "child_uri": "SubClass",
        "vertical_spacing": 30.0,
        "strength": 0.6
      }
    ],
    "enable_gpu": true,
    "convergence_threshold": 0.001
  }
}
```

#### Reasoning Request
```json
{
  "type": "ontology_reasoning",
  "protocol_version": "2.0.0",
  "data": {
    "graph_id": "ontology-graph-uuid",
    "reasoner": "hermit|elk|pellet",
    "inference_level": "rdfs|owl-dl|owl-full",
    "materialize_inferences": false,
    "timeout_ms": 30000
  }
}
```

#### Class/Property Query
```json
{
  "type": "ontology_query",
  "protocol_version": "2.0.0",
  "data": {
    "graph_id": "ontology-graph-uuid",
    "query_type": "class_hierarchy|property_domain|instances",
    "subject_uri": "http://example.org/Class_A",
    "include_inferred": true
  }
}
```

### 2.3 Ontology Node Type Flags

**Binary Protocol Extension** (compatible with existing flag system):

```rust
// Node type flags (30-bit ID space, 2 flag bits)
const ONTOLOGY_CLASS_FLAG: u32    = 0x10000000;  // Bit 28: OWL Class
const ONTOLOGY_PROPERTY_FLAG: u32 = 0x08000000;  // Bit 27: OWL Property
const ONTOLOGY_INDIVIDUAL_FLAG: u32 = 0x04000000; // Bit 26: OWL Individual

// Combined with existing flags (bits 30-31):
// Bit 31: Agent node
// Bit 30: Knowledge graph node
// Bit 29: Reserved
// Bit 28: Ontology Class
// Bit 27: Ontology Property
// Bit 26: Ontology Individual
// Bits 0-25: Node ID (67M node capacity)
```

**Wire Format Addition:**
```rust
pub struct OntologyNodeMetadata {
    pub node_id: u32,
    pub uri: String,
    pub label: String,
    pub node_type: OntologyNodeType,
    pub inferred: bool,
}

pub enum OntologyNodeType {
    Class,
    ObjectProperty,
    DataProperty,
    Individual,
    AnnotationProperty,
}
```

---

## 3. settings.yaml Schema Additions

**Location:** `/home/devuser/workspace/project/data/settings.yaml`

### New Top-Level Section: `ontology`

```yaml
# Ontology visualization and reasoning configuration
ontology:
  # Enable/disable ontology features
  enabled: true

  # GPU acceleration for constraint solving
  enable_gpu: true
  gpu_device_id: 0

  # Reasoner configuration
  reasoner:
    default_engine: "elk"  # elk|hermit|pellet
    timeout_ms: 30000
    inference_level: "owl-dl"  # rdfs|owl-dl|owl-full
    cache_inferences: true

  # Constraint group settings
  constraint_groups:
    disjoint:
      enabled: true
      separation_strength: 0.8
      max_separation_distance: 50.0

    hierarchy:
      enabled: true
      alignment_strength: 0.6
      vertical_spacing: 30.0
      horizontal_clustering: true

    same_as:
      enabled: true
      colocation_strength: 0.9
      min_colocation_distance: 2.0

    cardinality:
      enabled: true
      boundary_strength: 0.7
      visual_indicators: true

  # Validation settings
  validation:
    auto_validate_on_load: true
    check_consistency: true
    report_warnings: true
    max_violation_display: 50

  # Namespace management
  namespaces:
    auto_discover: true
    display_prefixes: true
    default_namespaces:
      owl: "http://www.w3.org/2002/07/owl#"
      rdf: "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
      rdfs: "http://www.w3.org/2000/01/rdf-schema#"
      xsd: "http://www.w3.org/2001/XMLSchema#"

# Graph-specific ontology visualization settings
visualisation:
  graphs:
    ontology:
      nodes:
        baseColor: '#4A90E2'
        metalness: 0.2
        opacity: 0.9
        roughness: 0.4
        nodeSize: 2.0
        quality: high
        enableInstancing: true
        enableHologram: false
        enableMetadataShape: true
        enableMetadataVisualisation: true

        # Node type-specific colors
        classColor: '#4A90E2'       # Blue for OWL Classes
        propertyColor: '#E8B339'    # Gold for Properties
        individualColor: '#7ED321'  # Green for Individuals
        inferredOpacity: 0.6        # Dimmer for inferred nodes

      edges:
        arrowSize: 0.03
        baseWidth: 0.5
        enableArrows: true
        opacity: 0.6
        widthRange: [0.2, 2.0]
        quality: high

        # Edge type-specific colors
        subClassOfColor: '#4A90E2'     # Hierarchy edges
        propertyEdgeColor: '#E8B339'   # Property connections
        instanceOfColor: '#7ED321'     # Instance relationships
        inferredEdgeStyle: 'dashed'

      labels:
        desktopFontSize: 1.2
        enableLabels: true
        textColor: '#FFFFFF'
        textOutlineColor: '#000000'
        textOutlineWidth: 0.01
        textResolution: 64
        textPadding: 0.4
        billboardMode: camera
        showMetadata: true
        maxLabelWidth: 8.0

        # URI display options
        showFullURI: false
        showPrefix: true
        showLocalName: true

      physics:
        enabled: true
        iterations: 100
        dt: 0.016
        damping: 0.7

        # Ontology-specific force parameters
        springK: 8.0                    # Stronger springs for hierarchy
        repelK: 20.0                    # Stronger repulsion for separation
        massScale: 1.2                  # Heavier nodes for stability

        # Constraint forces
        constraintForceWeight: 1.5      # High priority for ontology constraints
        hierarchyAlignmentForce: 0.6
        disjointSeparationForce: 0.8
        sameAsColocationForce: 0.9

        # GPU kernel parameters
        computeMode: 1                  # GPU-accelerated
        warmupIterations: 150
        coolingRate: 0.001

        autoBalance: true
        autoBalanceIntervalMs: 500

        autoPause:
          enabled: true
          equilibriumVelocityThreshold: 0.05
          equilibriumCheckFrames: 60
          equilibriumEnergyThreshold: 0.005
```

---

## 4. ontology_physics.toml Configuration File

**Location:** `/home/devuser/workspace/project/data/ontology_physics.toml`

```toml
# Ontology-Specific Physics Configuration
# Server-side parameters for ontology constraint solving

[metadata]
version = "1.0.0"
description = "Physics parameters for OWL ontology visualization"

[kernel]
# GPU kernel parameters
enable_gpu = true
gpu_device_id = 0
kernel_time_limit_ms = 5000
max_nodes = 1000000
max_edges = 10000000

# Simulation parameters
dt = 0.016
iterations_per_frame = 100
warmup_iterations = 150
cooling_rate = 0.001
convergence_threshold = 0.001

[forces]
# Base force strengths
spring_k = 8.0
repel_k = 20.0
center_gravity_k = 0.001
damping = 0.7

# Force cutoffs and limits
rest_length = 80.0
repulsion_cutoff = 120.0
repulsion_softening_epsilon = 0.0001
max_force = 100.0
max_velocity = 200.0

# Spatial grid for optimization
grid_cell_size = 60.0

[constraints.disjoint_separation]
enabled = true
base_strength = 0.8
min_separation_distance = 50.0
max_separation_distance = 200.0
force_falloff = "inverse_square"  # linear|inverse_square
enable_caching = true

[constraints.hierarchy_alignment]
enabled = true
base_strength = 0.6
vertical_spacing = 30.0
horizontal_spacing = 25.0
alignment_tolerance = 5.0
cluster_subclasses = true
enable_layering = true

[constraints.same_as_colocation]
enabled = true
base_strength = 0.9
min_colocation_distance = 2.0
max_colocation_distance = 10.0
force_type = "spring"  # spring|magnetic

[constraints.cardinality_boundary]
enabled = true
base_strength = 0.7
boundary_type = "soft"  # soft|hard
violation_penalty_multiplier = 2.0
visual_feedback = true

[constraints.property_domain_range]
enabled = true
base_strength = 0.5
domain_proximity_bonus = 1.2
range_proximity_bonus = 1.2
enable_path_constraints = true

[reasoning]
# Reasoning engine integration
auto_materialize_inferences = false
update_on_constraint_violation = true
incremental_reasoning = true
reasoning_interval_ms = 5000

[optimization]
# Convergence detection
stability_check_interval = 30
stability_variance_threshold = 0.01
min_stable_frames = 60

# Adaptive parameter tuning
enable_adaptive_forces = true
force_adjustment_rate = 0.05
min_force_multiplier = 0.5
max_force_multiplier = 2.0

[visualization]
# Constraint visualization settings
show_constraint_forces = false
show_violation_indicators = true
highlight_inferred_nodes = true
inferred_node_opacity = 0.6

[performance]
# Performance tuning
enable_constraint_caching = true
cache_invalidation_strategy = "on_topology_change"
max_cache_size_mb = 256
batch_constraint_updates = true
batch_size = 100

[debug]
# Debug logging
log_constraint_violations = true
log_force_magnitudes = false
log_convergence_metrics = true
export_constraint_graph = false
```

---

## 5. Migration Guide for Breaking Changes

### 5.1 Client Compatibility Matrix

| Client Version | Server Version | Compatibility | Notes |
|----------------|----------------|---------------|-------|
| v1.0-v1.9     | v2.0+         | ✅ Full       | Auto-negotiation to v1 protocol |
| v2.0+         | v2.0+         | ✅ Full       | Full ontology support |
| v2.0+         | v1.9          | ⚠️ Degraded   | Ontology features disabled |

### 5.2 Protocol Version Negotiation

**Client Connection Handshake:**
```json
{
  "type": "client_hello",
  "protocol_version": "2.0.0",
  "supported_features": [
    "binary_protocol_v2",
    "ontology_graphs",
    "reasoning_queries",
    "constraint_updates"
  ],
  "fallback_version": "1.9.0"
}
```

**Server Response:**
```json
{
  "type": "server_hello",
  "protocol_version": "2.0.0",
  "negotiated_features": [
    "binary_protocol_v2",
    "ontology_graphs"
  ],
  "available_graph_types": [
    "standard",
    "multi-agent",
    "force-directed",
    "hierarchical",
    "network",
    "ontology"
  ]
}
```

### 5.3 Breaking Changes Summary

**None** - All changes are additive and backward-compatible:

1. **GraphType Enum Addition**: New variant `Ontology` added without changing existing variants
2. **Protocol Messages**: New message types don't conflict with existing types
3. **Binary Protocol**: Node flags use previously unused bits (26-28)
4. **Configuration**: New sections added to existing YAML without modifying current settings

### 5.4 Migration Steps for Existing Deployments

#### Step 1: Update Server Code
```bash
# Pull latest code with GraphType update
git pull origin main

# Update Cargo dependencies if needed
cargo update

# Run tests to verify compatibility
cargo test --test ontology_integration_tests
```

#### Step 2: Add Configuration Files
```bash
# Copy default ontology_physics.toml
cp data/ontology_physics.toml.example data/ontology_physics.toml

# Update settings.yaml with ontology section
# (Merge with existing settings, don't replace)
```

#### Step 3: Restart Server
```bash
# Server will automatically detect new config sections
# v1.x clients will continue working without changes
cargo run --release
```

#### Step 4: Update Clients (Optional)
- v1.x clients: No action needed, will continue working
- v2.x clients: Update to support ontology graph type and new message types

### 5.5 Rollback Procedure

If issues occur:

1. **Revert GraphType changes:**
   ```bash
   git revert <commit-hash>
   ```

2. **Remove ontology section from settings.yaml:**
   ```bash
   # Comment out or delete the 'ontology:' section
   ```

3. **Restart server:**
   ```bash
   cargo run --release
   ```

All v1.x functionality will be fully restored.

---

## 6. Version Negotiation Strategy

### 6.1 Protocol Version Format

**Semantic Versioning:** `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes requiring client updates
- **MINOR**: New features, backward-compatible
- **PATCH**: Bug fixes, fully compatible

**Current Version:** `2.0.0`
- MAJOR=2: Binary protocol v2 (32-bit node IDs)
- MINOR=0: Ontology graph type added
- PATCH=0: Initial release

### 6.2 Feature Detection

Clients query server capabilities at connection:

```rust
pub struct ServerCapabilities {
    pub protocol_version: String,
    pub binary_protocol_version: u8,
    pub supported_graph_types: Vec<GraphType>,
    pub features: HashSet<String>,
}

impl ServerCapabilities {
    pub fn new() -> Self {
        Self {
            protocol_version: "2.0.0".to_string(),
            binary_protocol_version: 2,
            supported_graph_types: vec![
                GraphType::Standard,
                GraphType::MultiAgent,
                GraphType::ForceDirected,
                GraphType::Hierarchical,
                GraphType::Network,
                GraphType::Ontology,  // NEW
            ],
            features: HashSet::from([
                "websocket_realtime".to_string(),
                "binary_positions".to_string(),
                "constraint_updates".to_string(),
                "ontology_reasoning".to_string(),     // NEW
                "ontology_validation".to_string(),    // NEW
                "semantic_constraints".to_string(),   // NEW
            ]),
        }
    }
}
```

### 6.3 Graceful Degradation

When older clients connect:

```rust
pub fn negotiate_protocol(client_version: &str, server_version: &str) -> NegotiatedProtocol {
    let client_major = parse_major_version(client_version);
    let server_major = parse_major_version(server_version);

    if client_major < 2 {
        // v1.x client - disable ontology features
        NegotiatedProtocol {
            version: "1.9.0".to_string(),
            binary_protocol: 2,  // Still use v2 binary for efficiency
            enabled_features: vec![
                "standard_graphs",
                "multi_agent_graphs",
                "websocket_realtime",
            ],
            disabled_features: vec![
                "ontology_graphs",
                "ontology_reasoning",
                "semantic_constraints",
            ],
        }
    } else {
        // v2.x client - full feature set
        NegotiatedProtocol {
            version: server_version.to_string(),
            binary_protocol: 2,
            enabled_features: vec![
                "all",
            ],
            disabled_features: vec![],
        }
    }
}
```

### 6.4 Error Handling for Unsupported Features

When v1.x client requests ontology features:

```json
{
  "type": "error",
  "code": "FEATURE_NOT_SUPPORTED",
  "message": "Ontology graphs require protocol version 2.0.0 or higher",
  "current_protocol": "1.9.0",
  "required_protocol": "2.0.0",
  "upgrade_url": "https://docs.visionflow.info/upgrade"
}
```

---

## 7. Implementation Checklist

### Server-Side
- [x] Update GraphType enum with Ontology variant
- [x] Add FromStr implementation for GraphType parsing
- [ ] Implement ontology WebSocket message handlers
- [ ] Add binary protocol node flags for ontology types
- [ ] Create OntologyNodeMetadata struct
- [ ] Implement protocol version negotiation
- [ ] Add ServerCapabilities query endpoint
- [ ] Parse ontology section from settings.yaml
- [ ] Load ontology_physics.toml configuration
- [ ] Integrate with OntologyConstraintConfig

### Client-Side (Future Work)
- [ ] Add ontology graph type to UI selectors
- [ ] Implement ontology message handlers
- [ ] Add ontology-specific rendering pipeline
- [ ] Create constraint violation visualizations
- [ ] Add reasoning query interface
- [ ] Implement namespace prefix display

### Testing
- [ ] Unit tests for GraphType parsing
- [ ] Integration tests for protocol negotiation
- [ ] Compatibility tests with v1.x clients
- [ ] Load tests with large ontologies (>10k nodes)
- [ ] Validation tests for constraint solving

### Documentation
- [x] Protocol design document (this file)
- [ ] API documentation for new message types
- [ ] Client integration guide
- [ ] Configuration reference
- [ ] Performance tuning guide

---

## 8. References

### Existing System Components
- **GraphType Enum:** `/home/devuser/workspace/project/src/models/graph_types.rs`
- **Binary Protocol:** `/home/devuser/workspace/project/src/utils/binary_protocol.rs`
- **WebSocket Handler:** `/home/devuser/workspace/project/src/handlers/realtime_websocket_handler.rs`
- **Settings Config:** `/home/devuser/workspace/project/src/config/mod.rs`
- **Dev Config:** `/home/devuser/workspace/project/src/config/dev_config.rs`

### Related Documentation
- **Ontology Constraints:** `/home/devuser/workspace/project/src/physics/ontology_constraints.rs`
- **OWL Validator:** `/home/devuser/workspace/project/src/ontology/services/owl_validator.rs`
- **Test Mapping:** `/home/devuser/workspace/project/tests/fixtures/ontology/test_mapping.toml`

### External Standards
- OWL 2 Web Ontology Language: https://www.w3.org/TR/owl2-overview/
- RDF 1.1 Concepts: https://www.w3.org/TR/rdf11-concepts/
- SPARQL 1.1 Query Language: https://www.w3.org/TR/sparql11-query/

---

## Version History

| Version | Date       | Changes |
|---------|------------|---------|
| 1.0.0   | 2025-10-16 | Initial protocol design for ontology integration |

---

**Status:** ✅ Ready for Implementation
**Next Steps:** Begin server-side message handler implementation
