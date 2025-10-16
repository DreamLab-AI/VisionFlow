use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::types::vec3::Vec3Data;
use crate::models::constraints::{Constraint, AdvancedParams};
use log::{trace, debug};
use serde::{Serialize, Deserialize};
use serde_json;

// Protocol versions for wire format
const PROTOCOL_V1: u8 = 1;  // Legacy 16-bit node IDs (34 bytes per node)
const PROTOCOL_V2: u8 = 2;  // Full 32-bit node IDs (38 bytes per node)

// Node type flag constants for u32 (server-side)
const AGENT_NODE_FLAG: u32 = 0x80000000;     // Bit 31 indicates agent node
const KNOWLEDGE_NODE_FLAG: u32 = 0x40000000; // Bit 30 indicates knowledge graph node

// Ontology node type flags (bits 26-28, only valid when GraphType::Ontology)
const ONTOLOGY_TYPE_MASK: u32 = 0x1C000000;      // Bits 26-28: Ontology node types
const ONTOLOGY_CLASS_FLAG: u32 = 0x04000000;     // Bit 26: OWL Class
const ONTOLOGY_INDIVIDUAL_FLAG: u32 = 0x08000000; // Bit 27: OWL Individual
const ONTOLOGY_PROPERTY_FLAG: u32 = 0x10000000;   // Bit 28: OWL Property

const NODE_ID_MASK: u32 = 0x3FFFFFFF;        // Mask to extract actual node ID (bits 0-29)

// Node type flag constants for u16 (wire format v1 - DEPRECATED)
// BUG: These constants truncate node IDs > 16383, causing collisions
// FIXED: Use PROTOCOL_V2 with full u32 IDs for node_id > 16383
const WIRE_V1_AGENT_FLAG: u16 = 0x8000;         // Bit 15 indicates agent node
const WIRE_V1_KNOWLEDGE_FLAG: u16 = 0x4000;     // Bit 14 indicates knowledge graph node
const WIRE_V1_NODE_ID_MASK: u16 = 0x3FFF;       // Mask to extract actual node ID (bits 0-13)

// Node type flag constants for u32 (wire format v2)
const WIRE_V2_AGENT_FLAG: u32 = 0x80000000;     // Bit 31 indicates agent node
const WIRE_V2_KNOWLEDGE_FLAG: u32 = 0x40000000; // Bit 30 indicates knowledge graph node
const WIRE_V2_NODE_ID_MASK: u32 = 0x3FFFFFFF;   // Mask to extract actual node ID (bits 0-29)

/// Wire format v1 struct (LEGACY - 34 bytes)
/// BUG: Truncates node IDs to 14 bits (max 16383), causing collisions
/// DEPRECATED: Use WireNodeDataItemV2 for new implementations
pub struct WireNodeDataItemV1 {
    pub id: u16,                // 2 bytes - TRUNCATED to 14 bits + 2 flag bits
    pub position: Vec3Data,     // 12 bytes
    pub velocity: Vec3Data,     // 12 bytes
    pub sssp_distance: f32,     // 4 bytes - SSSP distance from source
    pub sssp_parent: i32,       // 4 bytes - Parent node for path reconstruction
    // Total: 34 bytes
}

/// Wire format v2 struct (FIXED - 38 bytes)
/// FIXES: Uses full 32-bit node IDs (30 bits + 2 flag bits)
/// Supports node IDs up to 1,073,741,823 (2^30 - 1)
pub struct WireNodeDataItemV2 {
    pub id: u32,                // 4 bytes - Full 32-bit with 30 bits for ID + 2 flag bits
    pub position: Vec3Data,     // 12 bytes
    pub velocity: Vec3Data,     // 12 bytes
    pub sssp_distance: f32,     // 4 bytes - SSSP distance from source
    pub sssp_parent: i32,       // 4 bytes - Parent node for path reconstruction
    // Total: 38 bytes
}

// Backwards compatibility alias - DEPRECATED
pub type WireNodeDataItem = WireNodeDataItemV2;

// Constants for wire format sizes
const WIRE_V1_ID_SIZE: usize = 2;  // u16 (LEGACY)
const WIRE_V2_ID_SIZE: usize = 4;  // u32 (FIXED)
const WIRE_VEC3_SIZE: usize = 12;  // 3 * f32
const WIRE_F32_SIZE: usize = 4;    // f32
const WIRE_I32_SIZE: usize = 4;    // i32
const WIRE_V1_ITEM_SIZE: usize = WIRE_V1_ID_SIZE + WIRE_VEC3_SIZE + WIRE_VEC3_SIZE + WIRE_F32_SIZE + WIRE_I32_SIZE; // 34 bytes (2+12+12+4+4)
const WIRE_V2_ITEM_SIZE: usize = WIRE_V2_ID_SIZE + WIRE_VEC3_SIZE + WIRE_VEC3_SIZE + WIRE_F32_SIZE + WIRE_I32_SIZE; // 36 bytes (4+12+12+4+4) NOT 38!

// Backwards compatibility alias - DEPRECATED
const WIRE_ID_SIZE: usize = WIRE_V2_ID_SIZE;
const WIRE_ITEM_SIZE: usize = WIRE_V2_ITEM_SIZE;

// Binary format (explicit):
//
// PROTOCOL V2 (CURRENT - FIXES node ID truncation bug):
// - Wire format sent to client (36 bytes total):
//   - Node Index: 4 bytes (u32) - Bits 30-31 for flags, bits 0-29 for ID
//   - Position: 3 × 4 bytes = 12 bytes
//   - Velocity: 3 × 4 bytes = 12 bytes
//   - SSSP Distance: 4 bytes (f32)
//   - SSSP Parent: 4 bytes (i32)
// Total: 36 bytes per node (NOT 38 - that was a documentation error!)
// Supports node IDs: 0 to 1,073,741,823 (2^30 - 1)
//
// PROTOCOL V1 (LEGACY - HAS BUG):
// - Wire format sent to client (34 bytes total):
//   - Node Index: 2 bytes (u16) - High bit (0x8000) indicates agent node
//   - Position: 3 × 4 bytes = 12 bytes
//   - Velocity: 3 × 4 bytes = 12 bytes
//   - SSSP Distance: 4 bytes (f32)
//   - SSSP Parent: 4 bytes (i32)
// Total: 34 bytes per node
// BUG: Only supports node IDs 0-16383 (14 bits). IDs > 16383 get truncated!
//
// - Server format (BinaryNodeData - 28 bytes total):
//   - Node ID: 4 bytes (u32)
//   - Position: 3 × 4 bytes = 12 bytes
//   - Velocity: 3 × 4 bytes = 12 bytes
// Total: 28 bytes per node
//
// Node Type Flags:
// - V2: Bits 30-31 of u32 ID (Bit 31 = Agent, Bit 30 = Knowledge)
// - V2: Bits 26-28 of u32 ID for Ontology types (Bit 26 = Class, Bit 27 = Individual, Bit 28 = Property)
// - V1: Bits 14-15 of u16 ID (Bit 15 = Agent, Bit 14 = Knowledge) [BUGGY]
// This allows the client to distinguish between different node types for visualization.

/// Utility functions for node type flag manipulation
pub fn set_agent_flag(node_id: u32) -> u32 {
    (node_id & NODE_ID_MASK) | AGENT_NODE_FLAG
}

pub fn set_knowledge_flag(node_id: u32) -> u32 {
    (node_id & NODE_ID_MASK) | KNOWLEDGE_NODE_FLAG
}

pub fn clear_agent_flag(node_id: u32) -> u32 {
    node_id & !AGENT_NODE_FLAG
}

pub fn clear_all_flags(node_id: u32) -> u32 {
    node_id & NODE_ID_MASK
}

pub fn is_agent_node(node_id: u32) -> bool {
    (node_id & AGENT_NODE_FLAG) != 0
}

pub fn is_knowledge_node(node_id: u32) -> bool {
    (node_id & KNOWLEDGE_NODE_FLAG) != 0
}

pub fn get_actual_node_id(node_id: u32) -> u32 {
    node_id & NODE_ID_MASK
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeType {
    Knowledge,
    Agent,
    OntologyClass,
    OntologyIndividual,
    OntologyProperty,
    Unknown,
}

pub fn get_node_type(node_id: u32) -> NodeType {
    if is_agent_node(node_id) {
        NodeType::Agent
    } else if is_knowledge_node(node_id) {
        NodeType::Knowledge
    } else if is_ontology_class(node_id) {
        NodeType::OntologyClass
    } else if is_ontology_individual(node_id) {
        NodeType::OntologyIndividual
    } else if is_ontology_property(node_id) {
        NodeType::OntologyProperty
    } else {
        NodeType::Unknown
    }
}

/// Ontology node type utility functions
pub fn set_ontology_class_flag(node_id: u32) -> u32 {
    (node_id & NODE_ID_MASK) | ONTOLOGY_CLASS_FLAG
}

pub fn set_ontology_individual_flag(node_id: u32) -> u32 {
    (node_id & NODE_ID_MASK) | ONTOLOGY_INDIVIDUAL_FLAG
}

pub fn set_ontology_property_flag(node_id: u32) -> u32 {
    (node_id & NODE_ID_MASK) | ONTOLOGY_PROPERTY_FLAG
}

pub fn is_ontology_class(node_id: u32) -> bool {
    (node_id & ONTOLOGY_TYPE_MASK) == ONTOLOGY_CLASS_FLAG
}

pub fn is_ontology_individual(node_id: u32) -> bool {
    (node_id & ONTOLOGY_TYPE_MASK) == ONTOLOGY_INDIVIDUAL_FLAG
}

pub fn is_ontology_property(node_id: u32) -> bool {
    (node_id & ONTOLOGY_TYPE_MASK) == ONTOLOGY_PROPERTY_FLAG
}

pub fn is_ontology_node(node_id: u32) -> bool {
    (node_id & ONTOLOGY_TYPE_MASK) != 0
}

/// Convert u32 node ID with flags to u16 wire format (V1 - LEGACY)
/// BUG: Truncates node IDs to 14 bits! Use to_wire_id_v2 instead.
/// DEPRECATED: Only kept for backwards compatibility with old clients
#[deprecated(note = "Use to_wire_id_v2 for full 32-bit node ID support")]
pub fn to_wire_id_v1(node_id: u32) -> u16 {
    let actual_id = get_actual_node_id(node_id);
    let wire_id = (actual_id & 0x3FFF) as u16; // BUG: Truncates to 14 bits!

    // Preserve node type flags
    if is_agent_node(node_id) {
        wire_id | WIRE_V1_AGENT_FLAG
    } else if is_knowledge_node(node_id) {
        wire_id | WIRE_V1_KNOWLEDGE_FLAG
    } else {
        wire_id
    }
}

/// Convert u16 wire ID back to u32 preserving flags (V1 - LEGACY)
/// DEPRECATED: Only kept for backwards compatibility with old clients
#[deprecated(note = "Use from_wire_id_v2 for full 32-bit node ID support")]
pub fn from_wire_id_v1(wire_id: u16) -> u32 {
    let actual_id = (wire_id & WIRE_V1_NODE_ID_MASK) as u32;

    // Restore node type flags
    if (wire_id & WIRE_V1_AGENT_FLAG) != 0 {
        actual_id | AGENT_NODE_FLAG
    } else if (wire_id & WIRE_V1_KNOWLEDGE_FLAG) != 0 {
        actual_id | KNOWLEDGE_NODE_FLAG
    } else {
        actual_id
    }
}

/// Convert u32 node ID with flags to u32 wire format (V2 - FIXED)
/// FIXED: Preserves full 32-bit node ID without truncation
pub fn to_wire_id_v2(node_id: u32) -> u32 {
    // No truncation needed - wire format uses full u32
    // Flags are already in the correct bit positions (30-31)
    node_id
}

/// Convert u32 wire ID back to u32 preserving flags (V2 - FIXED)
/// FIXED: No data loss, full 32-bit support
pub fn from_wire_id_v2(wire_id: u32) -> u32 {
    // No conversion needed - direct passthrough
    wire_id
}

// Backwards compatibility aliases - use V2 by default
pub fn to_wire_id(node_id: u32) -> u32 {
    to_wire_id_v2(node_id)
}

pub fn from_wire_id(wire_id: u32) -> u32 {
    from_wire_id_v2(wire_id)
}

/// Convert BinaryNodeData to wire format
impl BinaryNodeData {
    pub fn to_wire_format(&self, node_id: u32) -> WireNodeDataItem {
        WireNodeDataItem {
            id: to_wire_id(node_id),
            position: self.position(),
            velocity: self.velocity(),
            sssp_distance: f32::INFINITY,  // Default for client data
            sssp_parent: -1,               // Default for client data
        }
    }
}

/// Determine if we need V2 protocol (any node_id > 16383)
/// FIXED: Automatically detects when V2 is needed to prevent truncation
pub fn needs_v2_protocol(nodes: &[(u32, BinaryNodeData)]) -> bool {
    nodes.iter().any(|(node_id, _)| {
        let actual_id = get_actual_node_id(*node_id);
        actual_id > 0x3FFF // > 16383
    })
}

/// Enhanced encoding function that accepts metadata about node types including ontology nodes
/// FIXED: Always uses V2 protocol by default (full 32-bit node ID support)
/// Only falls back to V1 for backwards compatibility when all IDs are small
pub fn encode_node_data_with_types(
    nodes: &[(u32, BinaryNodeData)],
    agent_node_ids: &[u32],
    knowledge_node_ids: &[u32]
) -> Vec<u8> {
    encode_node_data_extended(nodes, agent_node_ids, knowledge_node_ids, &[], &[], &[])
}

/// Extended encoding function with ontology node type support
pub fn encode_node_data_extended(
    nodes: &[(u32, BinaryNodeData)],
    agent_node_ids: &[u32],
    knowledge_node_ids: &[u32],
    ontology_class_ids: &[u32],
    ontology_individual_ids: &[u32],
    ontology_property_ids: &[u32]
) -> Vec<u8> {
    // Always use V2 protocol to prevent truncation bugs (V1 is only for backwards compat)
    let use_v2 = true; // Force V2 for all new encoding
    let item_size = WIRE_V2_ITEM_SIZE;
    let protocol_version = PROTOCOL_V2;

    // Only log non-empty node transmissions to reduce spam
    if nodes.len() > 0 {
        trace!("Encoding {} nodes with agent flags using protocol v{} (item_size={})",
               nodes.len(), protocol_version, item_size);
    }

    // Reserve space for version byte + node data
    let mut buffer = Vec::with_capacity(1 + nodes.len() * item_size);

    // Write protocol version as first byte
    buffer.push(protocol_version);

    // Log some samples of the encoded data
    let sample_size = std::cmp::min(3, nodes.len());
    if sample_size > 0 {
        trace!("Sample of nodes being encoded with agent flags (protocol v{}):", protocol_version);
    }

    for (node_id, node) in nodes {
        // Check node type and set the appropriate flag
        // Priority: Agent > Knowledge > Ontology types > None
        let flagged_id = if agent_node_ids.contains(node_id) {
            set_agent_flag(*node_id)
        } else if knowledge_node_ids.contains(node_id) {
            set_knowledge_flag(*node_id)
        } else if ontology_class_ids.contains(node_id) {
            set_ontology_class_flag(*node_id)
        } else if ontology_individual_ids.contains(node_id) {
            set_ontology_individual_flag(*node_id)
        } else if ontology_property_ids.contains(node_id) {
            set_ontology_property_flag(*node_id)
        } else {
            *node_id  // No flags for unknown nodes
        };

        // Log the first few nodes for debugging
        if sample_size > 0 && *node_id < sample_size as u32 {
            trace!("Encoding node {}: pos=[{:.3},{:.3},{:.3}], vel=[{:.3},{:.3},{:.3}], is_agent={}",
                node_id,
                node.x, node.y, node.z,
                node.vx, node.vy, node.vz,
                agent_node_ids.contains(node_id));
        }

        if use_v2 {
            // V2: Write u32 ID (4 bytes) - NO TRUNCATION
            let wire_id = to_wire_id_v2(flagged_id);
            buffer.extend_from_slice(&wire_id.to_le_bytes());
        } else {
            // V1: Write u16 ID (2 bytes) - TRUNCATES (legacy support only)
            #[allow(deprecated)]
            let wire_id = to_wire_id_v1(flagged_id);
            buffer.extend_from_slice(&wire_id.to_le_bytes());
        }

        // Write position (12 bytes = 3 * f32)
        buffer.extend_from_slice(&node.x.to_le_bytes());
        buffer.extend_from_slice(&node.y.to_le_bytes());
        buffer.extend_from_slice(&node.z.to_le_bytes());

        // Write velocity (12 bytes = 3 * f32)
        buffer.extend_from_slice(&node.vx.to_le_bytes());
        buffer.extend_from_slice(&node.vy.to_le_bytes());
        buffer.extend_from_slice(&node.vz.to_le_bytes());

        // SSSP fields (8 bytes)
        buffer.extend_from_slice(&f32::INFINITY.to_le_bytes());
        buffer.extend_from_slice(&(-1i32).to_le_bytes());
    }

    // Only log non-empty node transmissions to reduce spam
    if nodes.len() > 0 {
        trace!("Encoded binary data with agent flags (v{}): {} bytes for {} nodes",
               protocol_version, buffer.len(), nodes.len());
    }
    buffer
}

/// Backwards-compatible encoding function for agent nodes only
pub fn encode_node_data_with_flags(nodes: &[(u32, BinaryNodeData)], agent_node_ids: &[u32]) -> Vec<u8> {
    encode_node_data_with_types(nodes, agent_node_ids, &[])
}

/// Basic encoding function without node type metadata
/// FIXED: Automatically uses V2 protocol when node IDs > 16383
pub fn encode_node_data(nodes: &[(u32, BinaryNodeData)]) -> Vec<u8> {
    encode_node_data_with_types(nodes, &[], &[])
}

pub fn decode_node_data(data: &[u8]) -> Result<Vec<(u32, BinaryNodeData)>, String> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    // First byte is protocol version
    if data.len() < 1 {
        return Err("Data too small for protocol version".to_string());
    }

    let protocol_version = data[0];
    let payload = &data[1..];

    match protocol_version {
        PROTOCOL_V1 => decode_node_data_v1(payload),
        PROTOCOL_V2 => decode_node_data_v2(payload),
        v => Err(format!("Unknown protocol version: {}", v)),
    }
}

/// Decode V1 protocol (LEGACY - 34 bytes per node, u16 IDs)
fn decode_node_data_v1(data: &[u8]) -> Result<Vec<(u32, BinaryNodeData)>, String> {
    // Check if data is properly sized
    if data.len() % WIRE_V1_ITEM_SIZE != 0 {
        return Err(format!(
            "Data size {} is not a multiple of V1 wire item size {}",
            data.len(),
            WIRE_V1_ITEM_SIZE
        ));
    }

    let expected_nodes = data.len() / WIRE_V1_ITEM_SIZE;
    debug!(
        "Decoding V1 binary data: size={} bytes, expected nodes={}",
        data.len(),
        expected_nodes
    );

    let mut updates = Vec::with_capacity(expected_nodes);
    let max_samples = 3;
    let mut samples_logged = 0;

    // Process data in chunks of WIRE_V1_ITEM_SIZE bytes
    for chunk in data.chunks_exact(WIRE_V1_ITEM_SIZE) {
        let mut cursor = 0;

        // Read u16 ID (2 bytes)
        let wire_id = u16::from_le_bytes([chunk[cursor], chunk[cursor + 1]]);
        cursor += 2;

        // Read position (12 bytes = 3 * f32)
        let pos_x = f32::from_le_bytes([chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]]);
        cursor += 4;
        let pos_y = f32::from_le_bytes([chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]]);
        cursor += 4;
        let pos_z = f32::from_le_bytes([chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]]);
        cursor += 4;

        // Read velocity (12 bytes = 3 * f32)
        let vel_x = f32::from_le_bytes([chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]]);
        cursor += 4;
        let vel_y = f32::from_le_bytes([chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]]);
        cursor += 4;
        let vel_z = f32::from_le_bytes([chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]]);
        cursor += 4;

        // Read SSSP data (8 bytes)
        let _sssp_distance = f32::from_le_bytes([chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]]);
        cursor += 4;
        let _sssp_parent = i32::from_le_bytes([chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]]);

        // Convert V1 wire ID back to u32 with flags
        #[allow(deprecated)]
        let full_node_id = from_wire_id_v1(wire_id);

        if samples_logged < max_samples {
            let is_agent = is_agent_node(full_node_id);
            let actual_id = get_actual_node_id(full_node_id);
            debug!(
                "Decoded V1 node wire_id={} -> full_id={} (actual_id={}, is_agent={}): pos=[{:.3},{:.3},{:.3}], vel=[{:.3},{:.3},{:.3}]",
                wire_id, full_node_id, actual_id, is_agent,
                pos_x, pos_y, pos_z,
                vel_x, vel_y, vel_z
            );
            samples_logged += 1;
        }

        let actual_id = get_actual_node_id(full_node_id);
        let server_node_data = BinaryNodeData {
            node_id: actual_id,
            x: pos_x,
            y: pos_y,
            z: pos_z,
            vx: vel_x,
            vy: vel_y,
            vz: vel_z,
        };

        updates.push((actual_id, server_node_data));
    }

    debug!("Successfully decoded {} V1 nodes from binary data", updates.len());
    Ok(updates)
}

/// Decode V2 protocol (FIXED - 38 bytes per node, u32 IDs)
fn decode_node_data_v2(data: &[u8]) -> Result<Vec<(u32, BinaryNodeData)>, String> {
    // Check if data is properly sized
    if data.len() % WIRE_V2_ITEM_SIZE != 0 {
        return Err(format!(
            "Data size {} is not a multiple of V2 wire item size {}",
            data.len(),
            WIRE_V2_ITEM_SIZE
        ));
    }

    let expected_nodes = data.len() / WIRE_V2_ITEM_SIZE;
    debug!(
        "Decoding V2 binary data: size={} bytes, expected nodes={}",
        data.len(),
        expected_nodes
    );

    let mut updates = Vec::with_capacity(expected_nodes);
    let max_samples = 3;
    let mut samples_logged = 0;

    // Process data in chunks of WIRE_V2_ITEM_SIZE bytes
    for chunk in data.chunks_exact(WIRE_V2_ITEM_SIZE) {
        let mut cursor = 0;

        // Read u32 ID (4 bytes) - FIXED: Full 32-bit support
        let wire_id = u32::from_le_bytes([
            chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]
        ]);
        cursor += 4;

        // Read position (12 bytes = 3 * f32)
        let pos_x = f32::from_le_bytes([chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]]);
        cursor += 4;
        let pos_y = f32::from_le_bytes([chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]]);
        cursor += 4;
        let pos_z = f32::from_le_bytes([chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]]);
        cursor += 4;

        // Read velocity (12 bytes = 3 * f32)
        let vel_x = f32::from_le_bytes([chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]]);
        cursor += 4;
        let vel_y = f32::from_le_bytes([chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]]);
        cursor += 4;
        let vel_z = f32::from_le_bytes([chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]]);
        cursor += 4;

        // Read SSSP data (8 bytes)
        let _sssp_distance = f32::from_le_bytes([chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]]);
        cursor += 4;
        let _sssp_parent = i32::from_le_bytes([chunk[cursor], chunk[cursor + 1], chunk[cursor + 2], chunk[cursor + 3]]);

        // Convert V2 wire ID back to u32 with flags (no truncation)
        let full_node_id = from_wire_id_v2(wire_id);

        if samples_logged < max_samples {
            let is_agent = is_agent_node(full_node_id);
            let actual_id = get_actual_node_id(full_node_id);
            debug!(
                "Decoded V2 node wire_id={} -> full_id={} (actual_id={}, is_agent={}): pos=[{:.3},{:.3},{:.3}], vel=[{:.3},{:.3},{:.3}]",
                wire_id, full_node_id, actual_id, is_agent,
                pos_x, pos_y, pos_z,
                vel_x, vel_y, vel_z
            );
            samples_logged += 1;
        }

        let actual_id = get_actual_node_id(full_node_id);
        let server_node_data = BinaryNodeData {
            node_id: actual_id,
            x: pos_x,
            y: pos_y,
            z: pos_z,
            vx: vel_x,
            vy: vel_y,
            vz: vel_z,
        };

        updates.push((actual_id, server_node_data));
    }

    debug!("Successfully decoded {} V2 nodes from binary data", updates.len());
    Ok(updates)
}

pub fn calculate_message_size(updates: &[(u32, BinaryNodeData)]) -> usize {
    // Determine protocol version based on node IDs
    let use_v2 = needs_v2_protocol(updates);
    let item_size = if use_v2 { WIRE_V2_ITEM_SIZE } else { WIRE_V1_ITEM_SIZE };
    // 1 byte for protocol version + items
    1 + updates.len() * item_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wire_format_size() {
        // Verify V1 format is 34 bytes (legacy)
        assert_eq!(WIRE_V1_ITEM_SIZE, 34);
        // Verify V2 format is 38 bytes (current)
        assert_eq!(WIRE_V2_ITEM_SIZE, 38);
        assert_eq!(WIRE_ITEM_SIZE, WIRE_V2_ITEM_SIZE); // Default is V2
        assert_eq!(WIRE_ID_SIZE + WIRE_VEC3_SIZE + WIRE_VEC3_SIZE + WIRE_F32_SIZE + WIRE_I32_SIZE, 38);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let nodes = vec![
            (1u32, BinaryNodeData {
                node_id: 1,
                x: 1.0, y: 2.0, z: 3.0,
                vx: 0.1, vy: 0.2, vz: 0.3,
            }),
            (2u32, BinaryNodeData {
                node_id: 2,
                x: 4.0, y: 5.0, z: 6.0,
                vx: 0.4, vy: 0.5, vz: 0.6,
            }),
        ];

        let encoded = encode_node_data(&nodes);

        // Verify encoded size: 1 byte version + nodes * V2 item size (small IDs use V2 by default now)
        assert_eq!(encoded.len(), 1 + nodes.len() * WIRE_V2_ITEM_SIZE);

        let decoded = decode_node_data(&encoded).unwrap();
        assert_eq!(nodes.len(), decoded.len());

        for ((orig_id, orig_data), (dec_id, dec_data)) in nodes.iter().zip(decoded.iter()) {
            assert_eq!(orig_id, dec_id);
            assert_eq!(orig_data.position(), dec_data.position());
            assert_eq!(orig_data.velocity(), dec_data.velocity());
        }
    }

    #[test]
    fn test_decode_invalid_data() {
        // Test with V2 protocol marker but invalid data size (not multiple of 38 bytes)
        let mut data = vec![PROTOCOL_V2]; // Version byte
        data.extend_from_slice(&[0u8; 37]); // 37 bytes (not multiple of 38)
        let result = decode_node_data(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not a multiple of"));

        // Test with just version byte (valid empty message)
        let result = decode_node_data(&[PROTOCOL_V2]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_message_size_calculation() {
        let nodes = vec![
            (1u32, BinaryNodeData {
                node_id: 1,
                x: 1.0, y: 2.0, z: 3.0,
                vx: 0.1, vy: 0.2, vz: 0.3,
            }),
        ];

        let size = calculate_message_size(&nodes);
        // V2 format: 1 byte version + 38 bytes per node
        assert_eq!(size, 1 + 38);

        let encoded = encode_node_data(&nodes);
        assert_eq!(encoded.len(), size);
    }

    #[test]
    fn test_agent_flag_functions() {
        let node_id = 42u32;
        
        // Test setting agent flag
        let flagged_id = set_agent_flag(node_id);
        assert_eq!(flagged_id, node_id | AGENT_NODE_FLAG);
        assert!(is_agent_node(flagged_id));
        
        // Test getting actual ID
        let actual_id = get_actual_node_id(flagged_id);
        assert_eq!(actual_id, node_id);
        
        // Test clearing agent flag
        let cleared_id = clear_agent_flag(flagged_id);
        assert_eq!(cleared_id, node_id);
        assert!(!is_agent_node(cleared_id));
        
        // Test non-agent node
        assert!(!is_agent_node(node_id));
    }

    #[test]
    fn test_wire_id_conversion() {
        // Test basic conversion (V2 uses u32, no truncation)
        let node_id = 42u32;
        let wire_id = to_wire_id(node_id);
        assert_eq!(wire_id, 42u32); // V2 returns u32
        assert_eq!(from_wire_id(wire_id), node_id);

        // Test agent flag preservation
        let agent_id = set_agent_flag(node_id);
        let agent_wire_id = to_wire_id(agent_id);
        assert_eq!(agent_wire_id & WIRE_V2_NODE_ID_MASK, 42u32);
        assert!((agent_wire_id & WIRE_V2_AGENT_FLAG) != 0);
        assert_eq!(from_wire_id(agent_wire_id), agent_id);

        // Test knowledge flag preservation
        let knowledge_id = set_knowledge_flag(node_id);
        let knowledge_wire_id = to_wire_id(knowledge_id);
        assert_eq!(knowledge_wire_id & WIRE_V2_NODE_ID_MASK, 42u32);
        assert!((knowledge_wire_id & WIRE_V2_KNOWLEDGE_FLAG) != 0);
        assert_eq!(from_wire_id(knowledge_wire_id), knowledge_id);

        // Test large node ID (V2 no longer truncates)
        let large_id = 0x5432u32;
        let wire_id = to_wire_id(large_id);
        assert_eq!(wire_id, 0x5432u32); // V2: No truncation!
        assert_eq!(from_wire_id(wire_id), large_id);
    }

    #[test]
    fn test_encode_with_agent_flags() {
        let nodes = vec![
            (1u32, BinaryNodeData {
                node_id: 1,
                x: 1.0, y: 2.0, z: 3.0,
                vx: 0.1, vy: 0.2, vz: 0.3,
            }),
            (2u32, BinaryNodeData {
                node_id: 2,
                x: 4.0, y: 5.0, z: 6.0,
                vx: 0.4, vy: 0.5, vz: 0.6,
            }),
        ];

        // Mark node 2 as an agent
        let agent_ids = vec![2u32];
        let encoded = encode_node_data_with_flags(&nodes, &agent_ids);

        // Verify encoded size: 1 byte version + nodes * V2 item size
        assert_eq!(encoded.len(), 1 + nodes.len() * WIRE_V2_ITEM_SIZE);

        let decoded = decode_node_data(&encoded).unwrap();
        assert_eq!(nodes.len(), decoded.len());

        // Verify that positions and velocities are preserved
        for ((orig_id, orig_data), (dec_id, dec_data)) in nodes.iter().zip(decoded.iter()) {
            assert_eq!(orig_id, dec_id); // IDs should match after flag stripping
            assert_eq!(orig_data.position(), dec_data.position());
            assert_eq!(orig_data.velocity(), dec_data.velocity());
        }
    }

    #[test]
    fn test_large_node_id_no_truncation() {
        // Test that V2 protocol handles large node IDs correctly
        let large_nodes = vec![
            (20000u32, BinaryNodeData {
                node_id: 20000,
                x: 1.0, y: 2.0, z: 3.0,
                vx: 0.1, vy: 0.2, vz: 0.3,
            }),
            (100000u32, BinaryNodeData {
                node_id: 100000,
                x: 4.0, y: 5.0, z: 6.0,
                vx: 0.4, vy: 0.5, vz: 0.6,
            }),
        ];

        // Verify V2 is automatically selected
        assert!(needs_v2_protocol(&large_nodes));

        let encoded = encode_node_data(&large_nodes);

        // First byte should be PROTOCOL_V2
        assert_eq!(encoded[0], PROTOCOL_V2);

        let decoded = decode_node_data(&encoded).unwrap();
        assert_eq!(large_nodes.len(), decoded.len());

        // Verify node IDs are preserved without truncation
        assert_eq!(decoded[0].0, 20000u32);
        assert_eq!(decoded[1].0, 100000u32);
    }

    #[test]
    fn test_ontology_node_flags() {
        let node_id = 123u32;

        // Test ontology class flag
        let class_id = set_ontology_class_flag(node_id);
        assert!(is_ontology_class(class_id));
        assert!(is_ontology_node(class_id));
        assert!(!is_ontology_individual(class_id));
        assert!(!is_ontology_property(class_id));
        assert_eq!(get_actual_node_id(class_id), node_id);
        assert_eq!(get_node_type(class_id), NodeType::OntologyClass);

        // Test ontology individual flag
        let individual_id = set_ontology_individual_flag(node_id);
        assert!(is_ontology_individual(individual_id));
        assert!(is_ontology_node(individual_id));
        assert!(!is_ontology_class(individual_id));
        assert!(!is_ontology_property(individual_id));
        assert_eq!(get_actual_node_id(individual_id), node_id);
        assert_eq!(get_node_type(individual_id), NodeType::OntologyIndividual);

        // Test ontology property flag
        let property_id = set_ontology_property_flag(node_id);
        assert!(is_ontology_property(property_id));
        assert!(is_ontology_node(property_id));
        assert!(!is_ontology_class(property_id));
        assert!(!is_ontology_individual(property_id));
        assert_eq!(get_actual_node_id(property_id), node_id);
        assert_eq!(get_node_type(property_id), NodeType::OntologyProperty);

        // Test non-ontology node
        assert!(!is_ontology_node(node_id));
        assert!(!is_ontology_class(node_id));
        assert!(!is_ontology_individual(node_id));
        assert!(!is_ontology_property(node_id));
    }

    #[test]
    fn test_encode_with_ontology_types() {
        let nodes = vec![
            (1u32, BinaryNodeData {
                node_id: 1,
                x: 1.0, y: 2.0, z: 3.0,
                vx: 0.1, vy: 0.2, vz: 0.3,
            }),
            (2u32, BinaryNodeData {
                node_id: 2,
                x: 4.0, y: 5.0, z: 6.0,
                vx: 0.4, vy: 0.5, vz: 0.6,
            }),
            (3u32, BinaryNodeData {
                node_id: 3,
                x: 7.0, y: 8.0, z: 9.0,
                vx: 0.7, vy: 0.8, vz: 0.9,
            }),
        ];

        // Mark nodes with different ontology types
        let class_ids = vec![1u32];
        let individual_ids = vec![2u32];
        let property_ids = vec![3u32];

        let encoded = encode_node_data_extended(&nodes, &[], &[], &class_ids, &individual_ids, &property_ids);

        // Verify encoded size: 1 byte version + nodes * V2 item size
        assert_eq!(encoded.len(), 1 + nodes.len() * WIRE_V2_ITEM_SIZE);

        let decoded = decode_node_data(&encoded).unwrap();
        assert_eq!(nodes.len(), decoded.len());

        // Verify positions and velocities are preserved
        for ((orig_id, orig_data), (dec_id, dec_data)) in nodes.iter().zip(decoded.iter()) {
            assert_eq!(orig_id, dec_id);
            assert_eq!(orig_data.position(), dec_data.position());
            assert_eq!(orig_data.velocity(), dec_data.velocity());
        }
    }

    #[test]
    fn test_ontology_flags_preserved_in_wire_format() {
        let nodes = vec![
            (100u32, BinaryNodeData {
                node_id: 100,
                x: 1.0, y: 2.0, z: 3.0,
                vx: 0.1, vy: 0.2, vz: 0.3,
            }),
        ];

        let class_ids = vec![100u32];
        let encoded = encode_node_data_extended(&nodes, &[], &[], &class_ids, &[], &[]);

        // First byte is protocol version
        assert_eq!(encoded[0], PROTOCOL_V2);

        // Next 4 bytes are the node ID with flags
        let wire_id = u32::from_le_bytes([
            encoded[1], encoded[2], encoded[3], encoded[4]
        ]);

        // Verify the class flag is set
        assert_eq!(wire_id & ONTOLOGY_TYPE_MASK, ONTOLOGY_CLASS_FLAG);
        assert_eq!(wire_id & NODE_ID_MASK, 100u32);
    }

    #[test]
    fn test_v1_backwards_compatibility() {
        // Test that small node IDs can use V1 if needed
        let small_nodes = vec![
            (100u32, BinaryNodeData {
                node_id: 100,
                x: 1.0, y: 2.0, z: 3.0,
                vx: 0.1, vy: 0.2, vz: 0.3,
            }),
        ];

        // V1 not needed, but should decode correctly if received
        let mut v1_encoded = vec![PROTOCOL_V1];
        #[allow(deprecated)]
        {
            let wire_id = to_wire_id_v1(100);
            v1_encoded.extend_from_slice(&wire_id.to_le_bytes());
        }
        // Add position
        v1_encoded.extend_from_slice(&1.0f32.to_le_bytes());
        v1_encoded.extend_from_slice(&2.0f32.to_le_bytes());
        v1_encoded.extend_from_slice(&3.0f32.to_le_bytes());
        // Add velocity
        v1_encoded.extend_from_slice(&0.1f32.to_le_bytes());
        v1_encoded.extend_from_slice(&0.2f32.to_le_bytes());
        v1_encoded.extend_from_slice(&0.3f32.to_le_bytes());
        // Add SSSP fields
        v1_encoded.extend_from_slice(&f32::INFINITY.to_le_bytes());
        v1_encoded.extend_from_slice(&(-1i32).to_le_bytes());

        let decoded = decode_node_data(&v1_encoded).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].0, 100u32);
    }
}

// Control frame structures for constraint and parameter updates

/// Control frame types for WebSocket communication
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ControlFrame {
    /// Update constraints on the server
    #[serde(rename = "constraints_update")]
    ConstraintsUpdate {
        version: u32,
        constraints: Vec<Constraint>,
        #[serde(skip_serializing_if = "Option::is_none")]
        advanced_params: Option<AdvancedParams>,
    },
    
    /// Request specific view lens configuration
    #[serde(rename = "lens_request")]
    LensRequest {
        lens_type: String,
        parameters: serde_json::Value,
    },
    
    /// Server acknowledgment of control frame
    #[serde(rename = "control_ack")]
    ControlAck {
        frame_type: String,
        success: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
    
    /// Update advanced physics parameters
    #[serde(rename = "physics_params")]
    PhysicsParams {
        advanced_params: AdvancedParams,
    },
    
    /// Request constraint preset
    #[serde(rename = "preset_request")]
    PresetRequest {
        preset_name: String,
    },
}

impl ControlFrame {
    /// Serialize control frame to JSON bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }
    
    /// Deserialize control frame from JSON bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
    
    /// Create a constraints update frame
    pub fn constraints_update(constraints: Vec<Constraint>, params: Option<AdvancedParams>) -> Self {
        ControlFrame::ConstraintsUpdate {
            version: 1,
            constraints,
            advanced_params: params,
        }
    }
    
    /// Create an acknowledgment frame
    pub fn ack(frame_type: &str, success: bool, message: Option<String>) -> Self {
        ControlFrame::ControlAck {
            frame_type: frame_type.to_string(),
            success,
            message,
        }
    }
}

/// Message type indicator for multiplexed WebSocket communication
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MessageType {
    /// Binary position data (28 bytes per node)
    BinaryPositions = 0,
    /// JSON control frame
    ControlFrame = 1,
}

/// Multiplexed message wrapper for WebSocket
pub struct MultiplexedMessage {
    pub msg_type: MessageType,
    pub data: Vec<u8>,
}

impl MultiplexedMessage {
    /// Create a binary positions message
    pub fn positions(node_data: &[(u32, BinaryNodeData)]) -> Self {
        Self {
            msg_type: MessageType::BinaryPositions,
            data: encode_node_data(node_data),
        }
    }
    
    /// Create a control frame message
    pub fn control(frame: &ControlFrame) -> Result<Self, serde_json::Error> {
        Ok(Self {
            msg_type: MessageType::ControlFrame,
            data: frame.to_bytes()?,
        })
    }
    
    /// Encode message with type indicator prefix
    pub fn encode(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(1 + self.data.len());
        result.push(self.msg_type as u8);
        result.extend_from_slice(&self.data);
        result
    }
    
    /// Decode message from bytes
    pub fn decode(data: &[u8]) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Empty message".to_string());
        }
        
        let msg_type = match data[0] {
            0 => MessageType::BinaryPositions,
            1 => MessageType::ControlFrame,
            t => return Err(format!("Unknown message type: {}", t)),
        };
        
        Ok(Self {
            msg_type,
            data: data[1..].to_vec(),
        })
    }
}

#[cfg(test)]
mod control_frame_tests {
    use super::*;
    use crate::models::constraints::ConstraintKind;
    
    #[test]
    fn test_control_frame_serialization() {
        let constraint = Constraint {
            kind: ConstraintKind::Separation,
            node_indices: vec![1, 2],
            params: vec![100.0],
            weight: 0.8,
            active: true,
        };
        
        let frame = ControlFrame::constraints_update(vec![constraint], None);
        let bytes = frame.to_bytes().unwrap();
        let decoded = ControlFrame::from_bytes(&bytes).unwrap();
        
        match decoded {
            ControlFrame::ConstraintsUpdate { version, constraints, .. } => {
                assert_eq!(version, 1);
                assert_eq!(constraints.len(), 1);
                assert_eq!(constraints[0].kind, ConstraintKind::Separation);
            },
            _ => panic!("Wrong frame type"),
        }
    }
    
    #[test]
    fn test_multiplexed_message() {
        let nodes = vec![
            (1u32, BinaryNodeData {
                position: Vec3Data::new(1.0, 2.0, 3.0),
                velocity: Vec3Data::new(0.1, 0.2, 0.3),
                mass: 100,
                flags: 1,
                padding: [0, 0],
            }),
        ];
        
        let msg = MultiplexedMessage::positions(&nodes);
        let encoded = msg.encode();
        
        assert_eq!(encoded[0], 0); // Binary positions type
        assert_eq!(encoded.len(), 1 + 28); // Type byte + one node
        
        let decoded = MultiplexedMessage::decode(&encoded).unwrap();
        assert_eq!(decoded.msg_type, MessageType::BinaryPositions);
        assert_eq!(decoded.data.len(), 28);
    }
}
