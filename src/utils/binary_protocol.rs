#![allow(dead_code)]
use crate::models::constraints::{AdvancedParams, Constraint};
use crate::types::vec3::Vec3Data;
use crate::utils::socket_flow_messages::BinaryNodeData;
use log::{debug, trace};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;

// Protocol versions for wire format (V1 REMOVED - no backward compatibility)
const PROTOCOL_V2: u8 = 2;
const PROTOCOL_V3: u8 = 3; // Analytics extension protocol (P0-4) - CURRENT
const PROTOCOL_V4: u8 = 4; // Delta encoding protocol

// Node type flag constants for u32 (server-side)
const AGENT_NODE_FLAG: u32 = 0x80000000; 
const KNOWLEDGE_NODE_FLAG: u32 = 0x40000000; 

// Ontology node type flags (bits 26-28, only valid when GraphType::Ontology)
const ONTOLOGY_TYPE_MASK: u32 = 0x1C000000;
const ONTOLOGY_CLASS_FLAG: u32 = 0x04000000;
const ONTOLOGY_INDIVIDUAL_FLAG: u32 = 0x08000000;
const ONTOLOGY_PROPERTY_FLAG: u32 = 0x10000000;

// Node ID mask: bits 0-25 only (excludes bits 26-31 for all flags)
// Supports node IDs: 0 to 67,108,863 (2^26 - 1)
const NODE_ID_MASK: u32 = 0x03FFFFFF; 

// V1 wire format constants REMOVED - caused node ID truncation bugs
// V2+ uses full u32 IDs with no truncation

// Node type flag constants for u32 (wire format v2)
const WIRE_V2_AGENT_FLAG: u32 = 0x80000000;
const WIRE_V2_KNOWLEDGE_FLAG: u32 = 0x40000000;
// Wire V2 node ID mask: bits 0-25 (excludes flag bits 26-31)
const WIRE_V2_NODE_ID_MASK: u32 = 0x03FFFFFF; 

// WireNodeDataItemV1 REMOVED - V1 protocol no longer supported

/// Wire format V2 - 36 bytes per node
/// Basic pathfinding + node type flags
pub struct WireNodeDataItemV2 {
    pub id: u32,
    pub position: Vec3Data,
    pub velocity: Vec3Data,
    pub sssp_distance: f32,
    pub sssp_parent: i32,

}

/// Wire format V3 - 48 bytes per node (P0-4 Analytics Extension)
/// Adds clustering, anomaly detection, and community detection
pub struct WireNodeDataItemV3 {
    pub id: u32,
    pub position: Vec3Data,
    pub velocity: Vec3Data,
    pub sssp_distance: f32,
    pub sssp_parent: i32,
    pub cluster_id: u32,
    pub anomaly_score: f32,
    pub community_id: u32,
}

// Backwards compatibility alias - now defaults to V3
pub type WireNodeDataItem = WireNodeDataItemV3;

// ============================================================================
// DELTA ENCODING (Protocol V4) - P1-3 Feature
// ============================================================================

/// Delta-encoded position update (16 bytes per changed node)
/// Used in frames 1-59 to send only changes from previous frame
/// Achieves 60-80% bandwidth reduction compared to full state updates
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeltaNodeData {
    pub id: u32,            // 4 bytes - node ID with flags
    pub change_flags: u8,   // 1 byte - bits indicate which fields changed
    pub _padding: [u8; 3],  // 3 bytes - alignment padding
    pub dx: i16,            // 2 bytes - delta position x (scaled)
    pub dy: i16,            // 2 bytes - delta position y (scaled)
    pub dz: i16,            // 2 bytes - delta position z (scaled)
    pub dvx: i16,           // 2 bytes - delta velocity x (scaled)
    pub dvy: i16,           // 2 bytes - delta velocity y (scaled)
    pub dvz: i16,           // 2 bytes - delta velocity z (scaled)
}

// Change flags for delta encoding
const DELTA_POSITION_CHANGED: u8 = 0x01;
const DELTA_VELOCITY_CHANGED: u8 = 0x02;
const DELTA_ALL_CHANGED: u8 = DELTA_POSITION_CHANGED | DELTA_VELOCITY_CHANGED;

// Delta encoding constants
const DELTA_SCALE_FACTOR: f32 = 100.0; // Scale factor for i16 precision
const DELTA_ITEM_SIZE: usize = 16;     // Size of DeltaNodeData in bytes
const DELTA_RESYNC_INTERVAL: u64 = 60; // Full state every 60 frames

// Constants for wire format sizes (V1 removed)
const WIRE_V2_ID_SIZE: usize = 4;
const WIRE_VEC3_SIZE: usize = 12;
const WIRE_F32_SIZE: usize = 4;
const WIRE_I32_SIZE: usize = 4;
const WIRE_U32_SIZE: usize = 4;
const WIRE_V2_ITEM_SIZE: usize =
    WIRE_V2_ID_SIZE + WIRE_VEC3_SIZE + WIRE_VEC3_SIZE + WIRE_F32_SIZE + WIRE_I32_SIZE;
const WIRE_V3_ITEM_SIZE: usize =
    WIRE_V2_ID_SIZE + WIRE_VEC3_SIZE + WIRE_VEC3_SIZE + WIRE_F32_SIZE + WIRE_I32_SIZE +
    WIRE_U32_SIZE + WIRE_F32_SIZE + WIRE_U32_SIZE; // V2 + cluster_id + anomaly_score + community_id 

// Backwards compatibility alias - now defaults to V3
const WIRE_ID_SIZE: usize = WIRE_V2_ID_SIZE;
const WIRE_ITEM_SIZE: usize = WIRE_V3_ITEM_SIZE;

// Binary format (explicit):
//
// PROTOCOL V3 (CURRENT - P0-4 Analytics Extension):
// - Wire format sent to client (48 bytes total):
//   - Node Index: 4 bytes (u32) - Bits 30-31 for agent/knowledge, bits 26-28 for ontology, bits 0-25 for ID
//   - Position: 3 × 4 bytes = 12 bytes
//   - Velocity: 3 × 4 bytes = 12 bytes
//   - SSSP Distance: 4 bytes (f32)
//   - SSSP Parent: 4 bytes (i32)
//   - Cluster ID: 4 bytes (u32) - K-means cluster assignment
//   - Anomaly Score: 4 bytes (f32) - LOF anomaly score (0.0-1.0)
//   - Community ID: 4 bytes (u32) - Louvain community assignment
// Total: 48 bytes per node
// Supports node IDs: 0 to 67,108,863 (2^26 - 1)
//
// PROTOCOL V2 (STABLE - FIXES node ID truncation bug):
// - Wire format sent to client (36 bytes total):
//   - Node Index: 4 bytes (u32) - Bits 30-31 for agent/knowledge, bits 26-28 for ontology, bits 0-25 for ID
//   - Position: 3 × 4 bytes = 12 bytes
//   - Velocity: 3 × 4 bytes = 12 bytes
//   - SSSP Distance: 4 bytes (f32)
//   - SSSP Parent: 4 bytes (i32)
// Total: 36 bytes per node (NOT 38 - that was a documentation error!)
// Supports node IDs: 0 to 67,108,863 (2^26 - 1)
//
// PROTOCOL V1 REMOVED - Had node ID truncation bug (IDs > 16383 were corrupted)
//
// - Server format (BinaryNodeData - 28 bytes total):
//   - Node ID: 4 bytes (u32)
//   - Position: 3 × 4 bytes = 12 bytes
//   - Velocity: 3 × 4 bytes = 12 bytes
// Total: 28 bytes per node
//
// Node Type Flags:
// - V2/V3: Bits 30-31 of u32 ID (Bit 31 = Agent, Bit 30 = Knowledge)
// - V2/V3: Bits 26-28 of u32 ID for Ontology types (Bit 26 = Class, Bit 27 = Individual, Bit 28 = Property)
// This allows the client to distinguish between different node types for visualization.

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

// to_wire_id_v1 and from_wire_id_v1 REMOVED - V1 protocol no longer supported
// Use to_wire_id_v2/from_wire_id_v2 for full 32-bit node ID support

pub fn to_wire_id_v2(node_id: u32) -> u32 {
    
    
    node_id
}

pub fn from_wire_id_v2(wire_id: u32) -> u32 {
    
    wire_id
}

// Backwards compatibility aliases - use V2 by default
pub fn to_wire_id(node_id: u32) -> u32 {
    to_wire_id_v2(node_id)
}

pub fn from_wire_id(wire_id: u32) -> u32 {
    from_wire_id_v2(wire_id)
}

/// Convert BinaryNodeData to wire format V3
impl BinaryNodeData {
    pub fn to_wire_format(&self, node_id: u32) -> WireNodeDataItem {
        self.to_wire_format_with_data(node_id, None, None)
    }

    /// Convert to wire format V3 with optional SSSP and analytics data.
    /// `sssp`: (distance, parent_id). Defaults to (INFINITY, -1).
    /// `analytics`: (cluster_id, anomaly_score, community_id). Defaults to (0, 0.0, 0).
    pub fn to_wire_format_with_data(
        &self,
        node_id: u32,
        sssp: Option<(f32, i32)>,
        analytics: Option<(u32, f32, u32)>,
    ) -> WireNodeDataItem {
        let (sssp_distance, sssp_parent) = sssp.unwrap_or((f32::INFINITY, -1));
        let (cluster_id, anomaly_score, community_id) = analytics.unwrap_or((0, 0.0, 0));
        WireNodeDataItem {
            id: to_wire_id(node_id),
            position: self.position(),
            velocity: self.velocity(),
            sssp_distance,
            sssp_parent,
            cluster_id,
            anomaly_score,
            community_id,
        }
    }
}

pub fn needs_v2_protocol(nodes: &[(u32, BinaryNodeData)]) -> bool {
    nodes.iter().any(|(node_id, _)| {
        let actual_id = get_actual_node_id(*node_id);
        actual_id > 0x3FFF 
    })
}

pub fn encode_node_data_with_types(
    nodes: &[(u32, BinaryNodeData)],
    agent_node_ids: &[u32],
    knowledge_node_ids: &[u32],
) -> Vec<u8> {
    encode_node_data_extended(nodes, agent_node_ids, knowledge_node_ids, &[], &[], &[])
}

pub fn encode_node_data_extended(
    nodes: &[(u32, BinaryNodeData)],
    agent_node_ids: &[u32],
    knowledge_node_ids: &[u32],
    ontology_class_ids: &[u32],
    ontology_individual_ids: &[u32],
    ontology_property_ids: &[u32],
) -> Vec<u8> {
    encode_node_data_extended_with_sssp(
        nodes,
        agent_node_ids,
        knowledge_node_ids,
        ontology_class_ids,
        ontology_individual_ids,
        ontology_property_ids,
        None,
    )
}

/// Encode node data with optional per-node SSSP distances and analytics.
/// `sssp_data` maps node_id -> (distance, parent_id).
/// When absent for a node, defaults to (INFINITY, -1).
pub fn encode_node_data_extended_with_sssp(
    nodes: &[(u32, BinaryNodeData)],
    agent_node_ids: &[u32],
    knowledge_node_ids: &[u32],
    ontology_class_ids: &[u32],
    ontology_individual_ids: &[u32],
    ontology_property_ids: &[u32],
    sssp_data: Option<&HashMap<u32, (f32, i32)>>,
) -> Vec<u8> {
    // Always use V3 as the default protocol (P0-4 Analytics Extension)
    let protocol_version = PROTOCOL_V3;
    let item_size = WIRE_V3_ITEM_SIZE;

    if !nodes.is_empty() {
        trace!(
            "Encoding {} nodes with agent flags using protocol v{} (item_size={})",
            nodes.len(),
            protocol_version,
            item_size
        );
    }

    let mut buffer = Vec::with_capacity(1 + nodes.len() * item_size);

    buffer.push(protocol_version);

    let sample_size = std::cmp::min(3, nodes.len());
    if sample_size > 0 {
        trace!(
            "Sample of nodes being encoded with agent flags (protocol v{}):",
            protocol_version
        );
    }

    for (node_id, node) in nodes {

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
            *node_id
        };

        if sample_size > 0 && *node_id < sample_size as u32 {
            trace!(
                "Encoding node {}: pos=[{:.3},{:.3},{:.3}], vel=[{:.3},{:.3},{:.3}], is_agent={}",
                node_id,
                node.x,
                node.y,
                node.z,
                node.vx,
                node.vy,
                node.vz,
                agent_node_ids.contains(node_id)
            );
        }

        // V3 always uses u32 IDs
        let wire_id = to_wire_id_v2(flagged_id);
        buffer.extend_from_slice(&wire_id.to_le_bytes());

        // Position (12 bytes)
        buffer.extend_from_slice(&node.x.to_le_bytes());
        buffer.extend_from_slice(&node.y.to_le_bytes());
        buffer.extend_from_slice(&node.z.to_le_bytes());

        // Velocity (12 bytes)
        buffer.extend_from_slice(&node.vx.to_le_bytes());
        buffer.extend_from_slice(&node.vy.to_le_bytes());
        buffer.extend_from_slice(&node.vz.to_le_bytes());

        // SSSP data (8 bytes) - read from sssp_data if available
        let (sssp_distance, sssp_parent) = sssp_data
            .and_then(|m| m.get(node_id))
            .copied()
            .unwrap_or((f32::INFINITY, -1));
        buffer.extend_from_slice(&sssp_distance.to_le_bytes());
        buffer.extend_from_slice(&sssp_parent.to_le_bytes());

        // Analytics data (12 bytes) - V3 extension with default values
        // For actual analytics data, use encode_node_data_with_analytics()
        buffer.extend_from_slice(&0u32.to_le_bytes());   // cluster_id
        buffer.extend_from_slice(&0.0f32.to_le_bytes()); // anomaly_score
        buffer.extend_from_slice(&0u32.to_le_bytes());   // community_id
    }

    
    if nodes.len() > 0 {
        trace!(
            "Encoded binary data with agent flags (v{}): {} bytes for {} nodes",
            protocol_version,
            buffer.len(),
            nodes.len()
        );
    }
    buffer
}

pub fn encode_node_data_with_flags(
    nodes: &[(u32, BinaryNodeData)],
    agent_node_ids: &[u32],
) -> Vec<u8> {
    encode_node_data_with_types(nodes, agent_node_ids, &[])
}

/// Encode node data with analytics (Protocol V3 - P0-4)
/// Extends V2 with cluster_id, anomaly_score, and community_id
pub fn encode_node_data_with_analytics(
    nodes: &[(u32, BinaryNodeData)],
    agent_node_ids: &[u32],
    knowledge_node_ids: &[u32],
    ontology_class_ids: &[u32],
    ontology_individual_ids: &[u32],
    ontology_property_ids: &[u32],
    analytics: &HashMap<u32, (u32, f32, u32)>, // (cluster_id, anomaly_score, community_id)
) -> Vec<u8> {
    encode_node_data_with_all(
        nodes,
        agent_node_ids,
        knowledge_node_ids,
        ontology_class_ids,
        ontology_individual_ids,
        ontology_property_ids,
        None,
        analytics,
    )
}

/// Encode node data with both SSSP and analytics data (Protocol V3).
/// `sssp_data` maps node_id -> (distance, parent_id).
/// `analytics` maps node_id -> (cluster_id, anomaly_score, community_id).
pub fn encode_node_data_with_all(
    nodes: &[(u32, BinaryNodeData)],
    agent_node_ids: &[u32],
    knowledge_node_ids: &[u32],
    ontology_class_ids: &[u32],
    ontology_individual_ids: &[u32],
    ontology_property_ids: &[u32],
    sssp_data: Option<&HashMap<u32, (f32, i32)>>,
    analytics: &HashMap<u32, (u32, f32, u32)>,
) -> Vec<u8> {
    let protocol_version = PROTOCOL_V3;
    let item_size = WIRE_V3_ITEM_SIZE;

    if !nodes.is_empty() {
        trace!(
            "Encoding {} nodes with analytics using protocol v{} (item_size={})",
            nodes.len(),
            protocol_version,
            item_size
        );
    }

    let mut buffer = Vec::with_capacity(1 + nodes.len() * item_size);
    buffer.push(protocol_version);

    for (node_id, node) in nodes {
        // Apply node type flags
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
            *node_id
        };

        let wire_id = to_wire_id_v2(flagged_id);
        buffer.extend_from_slice(&wire_id.to_le_bytes());

        // Position (12 bytes)
        buffer.extend_from_slice(&node.x.to_le_bytes());
        buffer.extend_from_slice(&node.y.to_le_bytes());
        buffer.extend_from_slice(&node.z.to_le_bytes());

        // Velocity (12 bytes)
        buffer.extend_from_slice(&node.vx.to_le_bytes());
        buffer.extend_from_slice(&node.vy.to_le_bytes());
        buffer.extend_from_slice(&node.vz.to_le_bytes());

        // SSSP data (8 bytes) - read from sssp_data if available
        let (sssp_distance, sssp_parent) = sssp_data
            .and_then(|m| m.get(node_id))
            .copied()
            .unwrap_or((f32::INFINITY, -1));
        buffer.extend_from_slice(&sssp_distance.to_le_bytes());
        buffer.extend_from_slice(&sssp_parent.to_le_bytes());

        // Analytics data (12 bytes) - V3 extension
        let (cluster_id, anomaly_score, community_id) = analytics
            .get(node_id)
            .copied()
            .unwrap_or((0, 0.0, 0)); // Default values if no analytics

        buffer.extend_from_slice(&cluster_id.to_le_bytes());
        buffer.extend_from_slice(&anomaly_score.to_le_bytes());
        buffer.extend_from_slice(&community_id.to_le_bytes());
    }

    if !nodes.is_empty() {
        trace!(
            "Encoded binary data with analytics (v{}): {} bytes for {} nodes",
            protocol_version,
            buffer.len(),
            nodes.len()
        );
    }

    buffer
}

pub fn encode_node_data(nodes: &[(u32, BinaryNodeData)]) -> Vec<u8> {
    encode_node_data_with_types(nodes, &[], &[])
}

pub fn decode_node_data(data: &[u8]) -> Result<Vec<(u32, BinaryNodeData)>, String> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    
    if data.len() < 1 {
        return Err("Data too small for protocol version".to_string());
    }

    let protocol_version = data[0];
    let payload = &data[1..];

    match protocol_version {
        1 => Err("Protocol V1 is no longer supported. Please upgrade client.".to_string()),
        PROTOCOL_V2 => decode_node_data_v2(payload),
        PROTOCOL_V3 => decode_node_data_v3(payload),
        v => Err(format!("Unknown protocol version: {}", v)),
    }
}

// decode_node_data_v1 REMOVED - V1 protocol no longer supported

fn decode_node_data_v2(data: &[u8]) -> Result<Vec<(u32, BinaryNodeData)>, String> {
    
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

    
    for chunk in data.chunks_exact(WIRE_V2_ITEM_SIZE) {
        let mut cursor = 0;

        
        let wire_id = u32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;

        
        let pos_x = f32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;
        let pos_y = f32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;
        let pos_z = f32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;

        
        let vel_x = f32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;
        let vel_y = f32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;
        let vel_z = f32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;

        
        let _sssp_distance = f32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;
        let _sssp_parent = i32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);

        
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

    debug!(
        "Successfully decoded {} V2 nodes from binary data",
        updates.len()
    );
    Ok(updates)
}

/// Decode Protocol V3 with analytics data (P0-4)
/// Returns standard BinaryNodeData (analytics data is discarded in basic decode)
fn decode_node_data_v3(data: &[u8]) -> Result<Vec<(u32, BinaryNodeData)>, String> {
    if data.len() % WIRE_V3_ITEM_SIZE != 0 {
        return Err(format!(
            "Data size {} is not a multiple of V3 wire item size {}",
            data.len(),
            WIRE_V3_ITEM_SIZE
        ));
    }

    let expected_nodes = data.len() / WIRE_V3_ITEM_SIZE;
    debug!(
        "Decoding V3 binary data with analytics: size={} bytes, expected nodes={}",
        data.len(),
        expected_nodes
    );

    let mut updates = Vec::with_capacity(expected_nodes);
    let max_samples = 3;
    let mut samples_logged = 0;

    for chunk in data.chunks_exact(WIRE_V3_ITEM_SIZE) {
        let mut cursor = 0;

        // Node ID (4 bytes)
        let wire_id = u32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;

        // Position (12 bytes)
        let pos_x = f32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;
        let pos_y = f32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;
        let pos_z = f32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;

        // Velocity (12 bytes)
        let vel_x = f32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;
        let vel_y = f32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;
        let vel_z = f32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;

        // SSSP data (8 bytes) - read but not used
        let _sssp_distance = f32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;
        let _sssp_parent = i32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;

        // Analytics data (12 bytes) - NEW in V3
        let _cluster_id = u32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;
        let _anomaly_score = f32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);
        cursor += 4;
        let _community_id = u32::from_le_bytes([
            chunk[cursor],
            chunk[cursor + 1],
            chunk[cursor + 2],
            chunk[cursor + 3],
        ]);

        let full_node_id = from_wire_id_v2(wire_id);

        if samples_logged < max_samples {
            let is_agent = is_agent_node(full_node_id);
            let actual_id = get_actual_node_id(full_node_id);
            debug!(
                "Decoded V3 node wire_id={} -> full_id={} (actual_id={}, is_agent={}): pos=[{:.3},{:.3},{:.3}], vel=[{:.3},{:.3},{:.3}], cluster={}, anomaly={:.3}, community={}",
                wire_id, full_node_id, actual_id, is_agent,
                pos_x, pos_y, pos_z,
                vel_x, vel_y, vel_z,
                _cluster_id, _anomaly_score, _community_id
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

    debug!(
        "Successfully decoded {} V3 nodes with analytics from binary data",
        updates.len()
    );
    Ok(updates)
}

pub fn calculate_message_size(updates: &[(u32, BinaryNodeData)]) -> usize {
    // V3 is now the default protocol (48 bytes per node)
    1 + updates.len() * WIRE_V3_ITEM_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wire_format_size() {
        // V1 REMOVED - was 34 bytes, caused node ID truncation
        // V2: 4 + 12 + 12 + 4 + 4 = 36 bytes
        assert_eq!(WIRE_V2_ITEM_SIZE, 36);
        // V3: 4 + 12 + 12 + 4 + 4 + 4 + 4 + 4 = 48 bytes (CURRENT)
        assert_eq!(WIRE_V3_ITEM_SIZE, 48);
        assert_eq!(WIRE_ITEM_SIZE, WIRE_V3_ITEM_SIZE); // Default is now V3
        assert_eq!(
            WIRE_ID_SIZE + WIRE_VEC3_SIZE + WIRE_VEC3_SIZE + WIRE_F32_SIZE + WIRE_I32_SIZE +
            WIRE_U32_SIZE + WIRE_F32_SIZE + WIRE_U32_SIZE,
            48
        );
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let nodes = vec![
            (
                1u32,
                BinaryNodeData {
                    node_id: 1,
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                    vx: 0.1,
                    vy: 0.2,
                    vz: 0.3,
                },
            ),
            (
                2u32,
                BinaryNodeData {
                    node_id: 2,
                    x: 4.0,
                    y: 5.0,
                    z: 6.0,
                    vx: 0.4,
                    vy: 0.5,
                    vz: 0.6,
                },
            ),
        ];

        let encoded = encode_node_data(&nodes);

        // V3 is now the default: 1 header byte + nodes * 48 bytes
        assert_eq!(encoded.len(), 1 + nodes.len() * WIRE_V3_ITEM_SIZE);

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
        
        let mut data = vec![PROTOCOL_V2]; 
        data.extend_from_slice(&[0u8; 37]); 
        let result = decode_node_data(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not a multiple of"));

        
        let result = decode_node_data(&[PROTOCOL_V2]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_message_size_calculation() {
        let nodes = vec![(
            1u32,
            BinaryNodeData {
                node_id: 1,
                x: 1.0,
                y: 2.0,
                z: 3.0,
                vx: 0.1,
                vy: 0.2,
                vz: 0.3,
            },
        )];

        let size = calculate_message_size(&nodes);
        // V3: 1 header + 48 bytes per node
        assert_eq!(size, 1 + 48);

        let encoded = encode_node_data(&nodes);
        assert_eq!(encoded.len(), size);
    }

    #[test]
    fn test_agent_flag_functions() {
        let node_id = 42u32;

        
        let flagged_id = set_agent_flag(node_id);
        assert_eq!(flagged_id, node_id | AGENT_NODE_FLAG);
        assert!(is_agent_node(flagged_id));

        
        let actual_id = get_actual_node_id(flagged_id);
        assert_eq!(actual_id, node_id);

        
        let cleared_id = clear_agent_flag(flagged_id);
        assert_eq!(cleared_id, node_id);
        assert!(!is_agent_node(cleared_id));

        
        assert!(!is_agent_node(node_id));
    }

    #[test]
    fn test_wire_id_conversion() {
        
        let node_id = 42u32;
        let wire_id = to_wire_id(node_id);
        assert_eq!(wire_id, 42u32); 
        assert_eq!(from_wire_id(wire_id), node_id);

        
        let agent_id = set_agent_flag(node_id);
        let agent_wire_id = to_wire_id(agent_id);
        assert_eq!(agent_wire_id & WIRE_V2_NODE_ID_MASK, 42u32);
        assert!((agent_wire_id & WIRE_V2_AGENT_FLAG) != 0);
        assert_eq!(from_wire_id(agent_wire_id), agent_id);

        
        let knowledge_id = set_knowledge_flag(node_id);
        let knowledge_wire_id = to_wire_id(knowledge_id);
        assert_eq!(knowledge_wire_id & WIRE_V2_NODE_ID_MASK, 42u32);
        assert!((knowledge_wire_id & WIRE_V2_KNOWLEDGE_FLAG) != 0);
        assert_eq!(from_wire_id(knowledge_wire_id), knowledge_id);

        
        let large_id = 0x5432u32;
        let wire_id = to_wire_id(large_id);
        assert_eq!(wire_id, 0x5432u32); 
        assert_eq!(from_wire_id(wire_id), large_id);
    }

    #[test]
    fn test_encode_with_agent_flags() {
        let nodes = vec![
            (
                1u32,
                BinaryNodeData {
                    node_id: 1,
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                    vx: 0.1,
                    vy: 0.2,
                    vz: 0.3,
                },
            ),
            (
                2u32,
                BinaryNodeData {
                    node_id: 2,
                    x: 4.0,
                    y: 5.0,
                    z: 6.0,
                    vx: 0.4,
                    vy: 0.5,
                    vz: 0.6,
                },
            ),
        ];

        // Mark node 2 as agent
        let agent_ids = vec![2u32];
        let encoded = encode_node_data_with_flags(&nodes, &agent_ids);

        // V3 format: 1 header + nodes * 48 bytes
        assert_eq!(encoded.len(), 1 + nodes.len() * WIRE_V3_ITEM_SIZE);

        let decoded = decode_node_data(&encoded).unwrap();
        assert_eq!(nodes.len(), decoded.len());

        
        for ((orig_id, orig_data), (dec_id, dec_data)) in nodes.iter().zip(decoded.iter()) {
            assert_eq!(orig_id, dec_id); 
            assert_eq!(orig_data.position(), dec_data.position());
            assert_eq!(orig_data.velocity(), dec_data.velocity());
        }
    }

    #[test]
    fn test_large_node_id_no_truncation() {
        
        let large_nodes = vec![
            (
                20000u32,
                BinaryNodeData {
                    node_id: 20000,
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                    vx: 0.1,
                    vy: 0.2,
                    vz: 0.3,
                },
            ),
            (
                100000u32,
                BinaryNodeData {
                    node_id: 100000,
                    x: 4.0,
                    y: 5.0,
                    z: 6.0,
                    vx: 0.4,
                    vy: 0.5,
                    vz: 0.6,
                },
            ),
        ];

        
        assert!(needs_v2_protocol(&large_nodes));

        let encoded = encode_node_data(&large_nodes);

        // V3 is now the default protocol
        assert_eq!(encoded[0], PROTOCOL_V3);

        let decoded = decode_node_data(&encoded).unwrap();
        assert_eq!(large_nodes.len(), decoded.len());

        
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
        // get_actual_node_id masks out all flags including ontology flags
        // The flagged ID includes the ontology bits, but actual ID strips them
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

        // Test that unflagged node is not an ontology node
        assert!(!is_ontology_node(node_id));
        assert!(!is_ontology_class(node_id));
        assert!(!is_ontology_individual(node_id));
        assert!(!is_ontology_property(node_id));
    }

    #[test]
    fn test_encode_with_ontology_types() {
        let nodes = vec![
            (
                1u32,
                BinaryNodeData {
                    node_id: 1,
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                    vx: 0.1,
                    vy: 0.2,
                    vz: 0.3,
                },
            ),
            (
                2u32,
                BinaryNodeData {
                    node_id: 2,
                    x: 4.0,
                    y: 5.0,
                    z: 6.0,
                    vx: 0.4,
                    vy: 0.5,
                    vz: 0.6,
                },
            ),
            (
                3u32,
                BinaryNodeData {
                    node_id: 3,
                    x: 7.0,
                    y: 8.0,
                    z: 9.0,
                    vx: 0.7,
                    vy: 0.8,
                    vz: 0.9,
                },
            ),
        ];

        // Mark nodes with ontology types
        let class_ids = vec![1u32];
        let individual_ids = vec![2u32];
        let property_ids = vec![3u32];

        let encoded =
            encode_node_data_extended(&nodes, &[], &[], &class_ids, &individual_ids, &property_ids);

        // V3 format: 1 header + nodes * 48 bytes
        assert_eq!(encoded.len(), 1 + nodes.len() * WIRE_V3_ITEM_SIZE);

        let decoded = decode_node_data(&encoded).unwrap();
        assert_eq!(nodes.len(), decoded.len());

        // After decoding, the actual node IDs should match (flags are stripped)
        // decode_node_data strips flags via get_actual_node_id
        for ((orig_id, orig_data), (dec_id, dec_data)) in nodes.iter().zip(decoded.iter()) {
            assert_eq!(*orig_id, *dec_id);
            assert_eq!(orig_data.position(), dec_data.position());
            assert_eq!(orig_data.velocity(), dec_data.velocity());
        }
    }

    #[test]
    fn test_ontology_flags_preserved_in_wire_format() {
        let nodes = vec![(
            100u32,
            BinaryNodeData {
                node_id: 100,
                x: 1.0,
                y: 2.0,
                z: 3.0,
                vx: 0.1,
                vy: 0.2,
                vz: 0.3,
            },
        )];

        let class_ids = vec![100u32];
        let encoded = encode_node_data_extended(&nodes, &[], &[], &class_ids, &[], &[]);

        // V3 is now the default protocol
        assert_eq!(encoded[0], PROTOCOL_V3);

        // Wire ID is at offset 1
        let wire_id = u32::from_le_bytes([encoded[1], encoded[2], encoded[3], encoded[4]]);

        // Verify ontology flag is set in the wire format
        assert_eq!(wire_id & ONTOLOGY_TYPE_MASK, ONTOLOGY_CLASS_FLAG);
        // Verify the actual node ID is preserved (using NODE_ID_MASK to extract it)
        assert_eq!(wire_id & NODE_ID_MASK, 100u32);
    }

    #[test]
    fn test_v1_protocol_rejected() {
        // V1 protocol should be rejected with clear error message
        let v1_encoded = vec![1u8]; // Protocol version 1
        let result = decode_node_data(&v1_encoded);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no longer supported"));
    }
}

// ============================================================================
// AGENT ACTION EVENTS (Protocol 0x23) - Ephemeral Connection Visualization
// ============================================================================

/// Action types for agent-to-data interactions
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AgentActionType {
    Query = 0,      // Agent querying data node (blue)
    Update = 1,     // Agent updating data node (yellow)
    Create = 2,     // Agent creating data node (green)
    Delete = 3,     // Agent deleting data node (red)
    Link = 4,       // Agent linking nodes (purple)
    Transform = 5,  // Agent transforming data (cyan)
}

impl From<u8> for AgentActionType {
    fn from(value: u8) -> Self {
        match value {
            0 => AgentActionType::Query,
            1 => AgentActionType::Update,
            2 => AgentActionType::Create,
            3 => AgentActionType::Delete,
            4 => AgentActionType::Link,
            5 => AgentActionType::Transform,
            _ => AgentActionType::Query, // Default
        }
    }
}

/// Agent action event for visualization (15 bytes header + variable payload)
/// Used to render ephemeral connections between agent nodes and data nodes
#[repr(C)]
#[derive(Debug, Clone)]
pub struct AgentActionEvent {
    pub source_agent_id: u32,   // 4 bytes - ID of the acting agent
    pub target_node_id: u32,    // 4 bytes - ID of the target data node
    pub action_type: u8,        // 1 byte - AgentActionType
    pub timestamp: u32,         // 4 bytes - Event timestamp (ms)
    pub duration_ms: u16,       // 2 bytes - Animation duration hint
    pub payload: Vec<u8>,       // Variable - Optional metadata
}

// Wire format size (fixed header only, payload is variable)
const AGENT_ACTION_HEADER_SIZE: usize = 15;

impl AgentActionEvent {
    /// Create a new agent action event
    pub fn new(
        source_agent_id: u32,
        target_node_id: u32,
        action_type: AgentActionType,
        duration_ms: u16,
    ) -> Self {
        Self {
            source_agent_id,
            target_node_id,
            action_type: action_type as u8,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| (d.as_millis() % u32::MAX as u128) as u32)
                .unwrap_or(0),
            duration_ms,
            payload: Vec::new(),
        }
    }

    /// Encode to wire format
    pub fn encode(&self) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(1 + AGENT_ACTION_HEADER_SIZE + self.payload.len());

        // Message type header
        buffer.push(MessageType::AgentAction as u8);

        // Fixed header (15 bytes)
        buffer.extend_from_slice(&self.source_agent_id.to_le_bytes());
        buffer.extend_from_slice(&self.target_node_id.to_le_bytes());
        buffer.push(self.action_type);
        buffer.extend_from_slice(&self.timestamp.to_le_bytes());
        buffer.extend_from_slice(&self.duration_ms.to_le_bytes());

        // Variable payload
        buffer.extend_from_slice(&self.payload);

        buffer
    }

    /// Decode from wire format (excludes message type byte)
    pub fn decode(data: &[u8]) -> Result<Self, String> {
        if data.len() < AGENT_ACTION_HEADER_SIZE {
            return Err(format!(
                "AgentActionEvent data too small: {} < {}",
                data.len(),
                AGENT_ACTION_HEADER_SIZE
            ));
        }

        let source_agent_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let target_node_id = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let action_type = data[8];
        let timestamp = u32::from_le_bytes([data[9], data[10], data[11], data[12]]);
        let duration_ms = u16::from_le_bytes([data[13], data[14]]);
        let payload = if data.len() > AGENT_ACTION_HEADER_SIZE {
            data[AGENT_ACTION_HEADER_SIZE..].to_vec()
        } else {
            Vec::new()
        };

        Ok(Self {
            source_agent_id,
            target_node_id,
            action_type,
            timestamp,
            duration_ms,
            payload,
        })
    }

    /// Get the action type as enum
    pub fn get_action_type(&self) -> AgentActionType {
        AgentActionType::from(self.action_type)
    }
}

/// Batch encode multiple agent action events
pub fn encode_agent_actions(events: &[AgentActionEvent]) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(
        1 + events.len() * (AGENT_ACTION_HEADER_SIZE + 16) // Estimate with avg payload
    );

    // Message type
    buffer.push(MessageType::AgentAction as u8);

    // Event count (u16)
    buffer.extend_from_slice(&(events.len() as u16).to_le_bytes());

    // Each event (length-prefixed)
    for event in events {
        let event_data = event.encode();
        let event_len = (event_data.len() - 1) as u16; // Exclude msg type byte
        buffer.extend_from_slice(&event_len.to_le_bytes());
        buffer.extend_from_slice(&event_data[1..]); // Skip msg type byte
    }

    buffer
}

/// Decode batch of agent action events
pub fn decode_agent_actions(data: &[u8]) -> Result<Vec<AgentActionEvent>, String> {
    if data.len() < 2 {
        return Err("AgentAction batch data too small".to_string());
    }

    let event_count = u16::from_le_bytes([data[0], data[1]]) as usize;
    let mut events = Vec::with_capacity(event_count);
    let mut offset = 2;

    for _ in 0..event_count {
        if offset + 2 > data.len() {
            return Err("Truncated event length".to_string());
        }

        let event_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        if offset + event_len > data.len() {
            return Err("Truncated event data".to_string());
        }

        let event = AgentActionEvent::decode(&data[offset..offset + event_len])?;
        events.push(event);
        offset += event_len;
    }

    Ok(events)
}

// Control frame structures for constraint and parameter updates

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ControlFrame {
    
    #[serde(rename = "constraints_update")]
    ConstraintsUpdate {
        version: u32,
        constraints: Vec<Constraint>,
        #[serde(skip_serializing_if = "Option::is_none")]
        advanced_params: Option<AdvancedParams>,
    },

    
    #[serde(rename = "lens_request")]
    LensRequest {
        lens_type: String,
        parameters: serde_json::Value,
    },

    
    #[serde(rename = "control_ack")]
    ControlAck {
        frame_type: String,
        success: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },

    
    #[serde(rename = "physics_params")]
    PhysicsParams { advanced_params: AdvancedParams },

    
    #[serde(rename = "preset_request")]
    PresetRequest { preset_name: String },
}

impl ControlFrame {
    
    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    
    pub fn constraints_update(
        constraints: Vec<Constraint>,
        params: Option<AdvancedParams>,
    ) -> Self {
        ControlFrame::ConstraintsUpdate {
            version: 1,
            constraints,
            advanced_params: params,
        }
    }

    
    pub fn ack(frame_type: &str, success: bool, message: Option<String>) -> Self {
        ControlFrame::ControlAck {
            frame_type: frame_type.to_string(),
            success,
            message,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MessageType {

    /// Binary position updates using Protocol V3 (48 bytes/node)
    BinaryPositions = 0,

    VoiceData = 0x02,

    ControlFrame = 0x03,

    /// Delta-encoded position updates (Protocol V4)
    /// Frame 0: FULL state, Frames 1-59: DELTA, Frame 60: FULL resync
    PositionDelta = 0x04,

    /// Client acknowledgement of position broadcast (Protocol V3 backpressure)
    /// Enables true end-to-end flow control vs queue-only confirmation
    BroadcastAck = 0x34,

    /// Agent action event for visualization of agent-to-data interactions
    /// Used for ephemeral connection visualization in 3D space
    AgentAction = 0x23,
}

/// WebSocket message types for voice and acknowledgements
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    VoiceData { audio: Vec<u8> },

    /// Client acknowledgement of position broadcast for backpressure flow control
    BroadcastAck {
        sequence_id: u64,    // Correlates with server broadcast sequence
        nodes_received: u32, // Number of nodes client processed
        timestamp: u64,      // Client receive timestamp (ms since epoch)
    },
}

#[derive(Debug)]
pub enum ProtocolError {
    InvalidMessageType(u8),
    InvalidPayloadSize(String),
    EncodingError(String),
    DecodingError(String),
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtocolError::InvalidMessageType(t) => write!(f, "Invalid message type: {}", t),
            ProtocolError::InvalidPayloadSize(s) => write!(f, "Invalid payload size: {}", s),
            ProtocolError::EncodingError(s) => write!(f, "Encoding error: {}", s),
            ProtocolError::DecodingError(s) => write!(f, "Decoding error: {}", s),
        }
    }
}

impl std::error::Error for ProtocolError {}

/// Binary protocol utilities for voice and acknowledgement messages
pub struct BinaryProtocol;

impl BinaryProtocol {
    /// Decode incoming WebSocket messages (voice data and acknowledgements)
    pub fn decode_message(data: &[u8]) -> Result<Message, ProtocolError> {
        if data.is_empty() {
            return Err(ProtocolError::DecodingError("Empty message".to_string()));
        }

        let message_type = data[0];

        match message_type {
            0x02 => Self::decode_voice_data(&data[1..]),
            0x34 => Self::decode_broadcast_ack(&data[1..]),
            _ => Err(ProtocolError::InvalidMessageType(message_type)),
        }
    }

    fn decode_voice_data(data: &[u8]) -> Result<Message, ProtocolError> {
        Ok(Message::VoiceData {
            audio: data.to_vec(),
        })
    }

    /// Decode client broadcast acknowledgement for backpressure flow control
    /// Payload: 8 bytes sequence_id + 4 bytes nodes_received + 8 bytes timestamp = 20 bytes
    fn decode_broadcast_ack(data: &[u8]) -> Result<Message, ProtocolError> {
        if data.len() < 20 {
            return Err(ProtocolError::InvalidPayloadSize(format!(
                "BroadcastAck payload size {} is less than required 20 bytes",
                data.len()
            )));
        }

        // Decode sequence_id (u64, little-endian)
        let sequence_id = u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]);

        // Decode nodes_received (u32, little-endian)
        let nodes_received = u32::from_le_bytes([
            data[8], data[9], data[10], data[11],
        ]);

        // Decode timestamp (u64, little-endian)
        let timestamp = u64::from_le_bytes([
            data[12], data[13], data[14], data[15],
            data[16], data[17], data[18], data[19],
        ]);

        Ok(Message::BroadcastAck {
            sequence_id,
            nodes_received,
            timestamp,
        })
    }

    
    
    pub fn encode_voice_data(audio: &[u8]) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(1 + audio.len());
        buffer.push(MessageType::VoiceData as u8);
        buffer.extend_from_slice(audio);
        buffer
    }
}

pub struct MultiplexedMessage {
    pub msg_type: MessageType,
    pub data: Vec<u8>,
}

impl MultiplexedMessage {
    
    pub fn positions(node_data: &[(u32, BinaryNodeData)]) -> Self {
        Self {
            msg_type: MessageType::BinaryPositions,
            data: encode_node_data(node_data),
        }
    }

    
    pub fn control(frame: &ControlFrame) -> Result<Self, serde_json::Error> {
        Ok(Self {
            msg_type: MessageType::ControlFrame,
            data: frame.to_bytes()?,
        })
    }

    
    pub fn encode(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(1 + self.data.len());
        result.push(self.msg_type as u8);
        result.extend_from_slice(&self.data);
        result
    }

    /// Decode multiplexed message from wire format
    pub fn decode(data: &[u8]) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Empty message".to_string());
        }

        let msg_type = match data[0] {
            0 => MessageType::BinaryPositions,
            0x02 => MessageType::VoiceData,
            0x03 => MessageType::ControlFrame,
            0x04 => MessageType::PositionDelta,
            0x23 => MessageType::AgentAction,
            0x34 => MessageType::BroadcastAck,
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
        let bytes = frame.to_bytes().expect("Serialization failed");
        let decoded = ControlFrame::from_bytes(&bytes).expect("Deserialization failed");

        match decoded {
            ControlFrame::ConstraintsUpdate {
                version,
                constraints,
                ..
            } => {
                assert_eq!(version, 1);
                assert_eq!(constraints.len(), 1);
                assert_eq!(constraints[0].kind, ConstraintKind::Separation);
            }
            _ => panic!("Wrong frame type"),
        }
    }

    #[test]
    fn test_multiplexed_message() {
        let nodes = vec![(
            1u32,
            BinaryNodeData {
                node_id: 1,
                x: 1.0,
                y: 2.0,
                z: 3.0,
                vx: 0.1,
                vy: 0.2,
                vz: 0.3,
            },
        )];

        let msg = MultiplexedMessage::positions(&nodes);
        let encoded = msg.encode();

        assert_eq!(encoded[0], 0); 

        let decoded = MultiplexedMessage::decode(&encoded).expect("Decode failed");
        assert_eq!(decoded.msg_type, MessageType::BinaryPositions);
    }

    #[test]
    fn test_simplified_protocol_voice_data() {
        let audio = vec![0x12, 0x34, 0x56, 0x78];

        let encoded = BinaryProtocol::encode_voice_data(&audio);
        assert_eq!(encoded[0], 0x02); 
        assert_eq!(encoded.len(), 1 + audio.len());

        let decoded = BinaryProtocol::decode_message(&encoded).expect("Message decode failed");
        match decoded {
            Message::VoiceData {
                audio: decoded_audio,
            } => {
                assert_eq!(decoded_audio, audio);
            }
            _ => panic!("Expected VoiceData message"),
        }
    }

    #[test]
    fn test_protocol_error_handling() {
        // Empty message
        let result = BinaryProtocol::decode_message(&[]);
        assert!(matches!(result, Err(ProtocolError::DecodingError(_))));

        // Invalid message type
        let result = BinaryProtocol::decode_message(&[0xFF]);
        assert!(matches!(
            result,
            Err(ProtocolError::InvalidMessageType(0xFF))
        ));
    }

    #[test]
    fn test_agent_action_type_conversion() {
        assert_eq!(AgentActionType::from(0), AgentActionType::Query);
        assert_eq!(AgentActionType::from(1), AgentActionType::Update);
        assert_eq!(AgentActionType::from(2), AgentActionType::Create);
        assert_eq!(AgentActionType::from(3), AgentActionType::Delete);
        assert_eq!(AgentActionType::from(4), AgentActionType::Link);
        assert_eq!(AgentActionType::from(5), AgentActionType::Transform);
        assert_eq!(AgentActionType::from(255), AgentActionType::Query); // Default
    }

    #[test]
    fn test_agent_action_event_encode_decode() {
        let event = AgentActionEvent::new(
            42,   // source_agent_id
            100,  // target_node_id
            AgentActionType::Update,
            500,  // duration_ms
        );

        let encoded = event.encode();

        // Verify message type header
        assert_eq!(encoded[0], MessageType::AgentAction as u8);
        assert_eq!(encoded[0], 0x23);

        // Verify header size: 1 (msg type) + 15 (header) = 16 bytes minimum
        assert!(encoded.len() >= 16);

        // Decode (skip msg type byte)
        let decoded = AgentActionEvent::decode(&encoded[1..]).expect("Decode failed");

        assert_eq!(decoded.source_agent_id, 42);
        assert_eq!(decoded.target_node_id, 100);
        assert_eq!(decoded.get_action_type(), AgentActionType::Update);
        assert_eq!(decoded.duration_ms, 500);
        assert!(decoded.payload.is_empty());
    }

    #[test]
    fn test_agent_action_event_with_payload() {
        let mut event = AgentActionEvent::new(
            1,
            2,
            AgentActionType::Create,
            1000,
        );
        event.payload = vec![0xDE, 0xAD, 0xBE, 0xEF];

        let encoded = event.encode();

        // 1 (msg type) + 15 (header) + 4 (payload) = 20 bytes
        assert_eq!(encoded.len(), 20);

        let decoded = AgentActionEvent::decode(&encoded[1..]).expect("Decode failed");
        assert_eq!(decoded.payload, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_agent_action_batch_encode_decode() {
        let events = vec![
            AgentActionEvent::new(1, 10, AgentActionType::Query, 100),
            AgentActionEvent::new(2, 20, AgentActionType::Update, 200),
            AgentActionEvent::new(3, 30, AgentActionType::Delete, 300),
        ];

        let encoded = encode_agent_actions(&events);

        // First byte is message type
        assert_eq!(encoded[0], MessageType::AgentAction as u8);

        // Decode batch (skip msg type byte)
        let decoded = decode_agent_actions(&encoded[1..]).expect("Batch decode failed");

        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].source_agent_id, 1);
        assert_eq!(decoded[0].target_node_id, 10);
        assert_eq!(decoded[0].get_action_type(), AgentActionType::Query);

        assert_eq!(decoded[1].source_agent_id, 2);
        assert_eq!(decoded[1].get_action_type(), AgentActionType::Update);

        assert_eq!(decoded[2].source_agent_id, 3);
        assert_eq!(decoded[2].get_action_type(), AgentActionType::Delete);
    }

    #[test]
    fn test_agent_action_decode_error() {
        // Data too small
        let result = AgentActionEvent::decode(&[0; 10]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too small"));
    }

    #[test]
    fn test_multiplexed_agent_action() {
        let event = AgentActionEvent::new(5, 50, AgentActionType::Link, 750);
        let encoded = event.encode();

        let msg = MultiplexedMessage::decode(&encoded).expect("Decode failed");
        assert_eq!(msg.msg_type, MessageType::AgentAction);
    }

    #[test]
    fn test_message_type_values() {
        // Verify message type constants match spec
        assert_eq!(MessageType::BinaryPositions as u8, 0x00);
        assert_eq!(MessageType::VoiceData as u8, 0x02);
        assert_eq!(MessageType::ControlFrame as u8, 0x03);
        assert_eq!(MessageType::PositionDelta as u8, 0x04);
        assert_eq!(MessageType::AgentAction as u8, 0x23);
        assert_eq!(MessageType::BroadcastAck as u8, 0x34);
    }
}
