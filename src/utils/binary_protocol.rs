use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::types::vec3::Vec3Data;
use crate::models::constraints::{Constraint, AdvancedParams};
use bytemuck::{Pod, Zeroable};
use log::{trace, debug};
use serde::{Serialize, Deserialize};
use serde_json;

// Node type flag constants
const AGENT_NODE_FLAG: u32 = 0x80000000;     // Bit 31 indicates agent node
const KNOWLEDGE_NODE_FLAG: u32 = 0x40000000; // Bit 30 indicates knowledge graph node
const NODE_ID_MASK: u32 = 0x3FFFFFFF;        // Mask to extract actual node ID (bits 0-29)

/// Explicit wire format struct for WebSocket binary protocol
/// This struct represents exactly what is sent over the wire
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WireNodeDataItem {
    pub id: u32,           // 4 bytes (changed from u16)
    pub position: Vec3Data, // 12 bytes
    pub velocity: Vec3Data, // 12 bytes
    // Total: 28 bytes
}

// Compile-time assertion to ensure wire format is exactly 28 bytes
static_assertions::const_assert_eq!(std::mem::size_of::<WireNodeDataItem>(), 28);

// Binary format (explicit):
// - For each node (28 bytes total):
//   - Node Index: 4 bytes (u32) - High bit indicates agent node (0x80000000)
//   - Position: 3 × 4 bytes = 12 bytes
//   - Velocity: 3 × 4 bytes = 12 bytes
// Total: 28 bytes per node
//
// Agent Node Flag: The high bit (0x80000000) of the node ID is used to indicate
// that this node represents an agent rather than a document/metadata node.
// This allows the client to distinguish between different node types for visualization.

/// Utility functions for node type flag manipulation
pub fn set_agent_flag(node_id: u32) -> u32 {
    (node_id & NODE_ID_MASK) | AGENT_NODE_FLAG
}

pub fn set_knowledge_flag(node_id: u32) -> u32 {
    (node_id & NODE_ID_MASK) | KNOWLEDGE_NODE_FLAG
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
    Unknown,
}

pub fn get_node_type(node_id: u32) -> NodeType {
    if is_agent_node(node_id) {
        NodeType::Agent
    } else if is_knowledge_node(node_id) {
        NodeType::Knowledge
    } else {
        NodeType::Unknown
    }
}

/// Enhanced encoding function that accepts metadata about node types
pub fn encode_node_data_with_types(
    nodes: &[(u32, BinaryNodeData)], 
    agent_node_ids: &[u32],
    knowledge_node_ids: &[u32]
) -> Vec<u8> {
    // Only log non-empty node transmissions to reduce spam
    if nodes.len() > 0 {
        trace!("Encoding {} nodes with agent flags for binary transmission", nodes.len());
    }
    
    let mut buffer = Vec::with_capacity(nodes.len() * std::mem::size_of::<WireNodeDataItem>());
    
    // Log some samples of the encoded data
    let sample_size = std::cmp::min(3, nodes.len());
    if sample_size > 0 {
        trace!("Sample of nodes being encoded with agent flags:");
    }
    
    for (node_id, node) in nodes {
        // Check node type and set the appropriate flag
        let flagged_id = if agent_node_ids.contains(node_id) {
            set_agent_flag(*node_id)
        } else if knowledge_node_ids.contains(node_id) {
            set_knowledge_flag(*node_id)
        } else {
            *node_id  // No flags for unknown nodes
        };
        
        // Log the first few nodes for debugging
        if sample_size > 0 && *node_id < sample_size as u32 {
            trace!("Encoding node {}: pos=[{:.3},{:.3},{:.3}], vel=[{:.3},{:.3},{:.3}], is_agent={}",
                node_id,
                node.position.x, node.position.y, node.position.z,
                node.velocity.x, node.velocity.y, node.velocity.z,
                agent_node_ids.contains(node_id));
        }
        
        // Create explicit wire format item with potentially flagged ID
        let wire_item = WireNodeDataItem {
            id: flagged_id,
            position: node.position,
            velocity: node.velocity,
        };
        
        // Use bytemuck for safe, direct memory layout conversion
        let item_bytes = bytemuck::bytes_of(&wire_item);
        buffer.extend_from_slice(item_bytes);
    }

    // Only log non-empty node transmissions to reduce spam
    if nodes.len() > 0 {
        trace!("Encoded binary data with agent flags: {} bytes for {} nodes", buffer.len(), nodes.len());
    }
    buffer
}

/// Backwards-compatible encoding function for agent nodes only
pub fn encode_node_data_with_flags(nodes: &[(u32, BinaryNodeData)], agent_node_ids: &[u32]) -> Vec<u8> {
    encode_node_data_with_types(nodes, agent_node_ids, &[])
}

pub fn encode_node_data(nodes: &[(u32, BinaryNodeData)]) -> Vec<u8> {
    // Only log non-empty node transmissions to reduce spam
    if nodes.len() > 0 {
        trace!("Encoding {} nodes for binary transmission", nodes.len());
    }
    
    let mut buffer = Vec::with_capacity(nodes.len() * std::mem::size_of::<WireNodeDataItem>());
    
    // Log some samples of the encoded data
    let sample_size = std::cmp::min(3, nodes.len());
    if sample_size > 0 {
        trace!("Sample of nodes being encoded:");
    }
    
    for (node_id, node) in nodes {
        // Log the first few nodes for debugging
        if sample_size > 0 && *node_id < sample_size as u32 {
            trace!("Encoding node {}: pos=[{:.3},{:.3},{:.3}], vel=[{:.3},{:.3},{:.3}]",
                node_id,
                node.position.x, node.position.y, node.position.z,
                node.velocity.x, node.velocity.y, node.velocity.z);
        }
        
        // Create explicit wire format item
        let wire_item = WireNodeDataItem {
            id: *node_id,
            position: node.position,
            velocity: node.velocity,
        };
        
        // Use bytemuck for safe, direct memory layout conversion
        let item_bytes = bytemuck::bytes_of(&wire_item);
        buffer.extend_from_slice(item_bytes);
        
        // Mass, flags, and padding are server-side only and not transmitted over wire
    }

    // Only log non-empty node transmissions to reduce spam
    if nodes.len() > 0 {
        trace!("Encoded binary data: {} bytes for {} nodes", buffer.len(), nodes.len());
    }
    buffer
}

pub fn decode_node_data(data: &[u8]) -> Result<Vec<(u32, BinaryNodeData)>, String> {
    const WIRE_ITEM_SIZE: usize = std::mem::size_of::<WireNodeDataItem>();
    
    // Check if data is properly sized
    if data.len() % WIRE_ITEM_SIZE != 0 {
        return Err(format!(
            "Data size {} is not a multiple of wire item size {}",
            data.len(),
            WIRE_ITEM_SIZE
        ));
    }
    
    if data.is_empty() {
        return Ok(Vec::new());
    }
    
    let expected_nodes = data.len() / WIRE_ITEM_SIZE;
    debug!(
        "Decoding binary data: size={} bytes, expected nodes={}",
        data.len(),
        expected_nodes
    );
    
    let mut updates = Vec::with_capacity(expected_nodes);
    
    // Set up sample logging
    let max_samples = 3;
    let mut samples_logged = 0;
    
    debug!("Starting binary data decode, expecting nodes with position and velocity data");
    
    // Process data in chunks of WIRE_ITEM_SIZE bytes
    for chunk in data.chunks_exact(WIRE_ITEM_SIZE) {
        // Use bytemuck for safe deserialization from bytes
        let wire_item: WireNodeDataItem = *bytemuck::from_bytes(chunk);
        
        // Log the first few decoded items as samples
        if samples_logged < max_samples {
            let is_agent = is_agent_node(wire_item.id);
            let actual_id = get_actual_node_id(wire_item.id);
            debug!(
                "Decoded node {} (actual_id={}, is_agent={}): pos=[{:.3},{:.3},{:.3}], vel=[{:.3},{:.3},{:.3}]",
                wire_item.id, actual_id, is_agent,
                wire_item.position.x, wire_item.position.y, wire_item.position.z,
                wire_item.velocity.x, wire_item.velocity.y, wire_item.velocity.z
            );
            samples_logged += 1;
        }
        
        // Convert wire format to server format (BinaryNodeData)
        // Server-side fields (mass, flags, padding) get default values
        let server_node_data = BinaryNodeData {
            position: wire_item.position,
            velocity: wire_item.velocity,
            mass: 100u8,     // Default mass - will be replaced with actual value from node_map
            flags: 0u8,      // Default flags - will be replaced with actual value from node_map
            padding: [0u8, 0u8], // Default padding
        };
        
        // Use the actual node ID (with agent flag stripped) for server-side processing
        let actual_id = get_actual_node_id(wire_item.id);
        updates.push((actual_id, server_node_data));
    }
    
    debug!("Successfully decoded {} nodes from binary data", updates.len());
    Ok(updates)
}

pub fn calculate_message_size(updates: &[(u32, BinaryNodeData)]) -> usize {
    // Each update uses WireNodeDataItem size
    updates.len() * std::mem::size_of::<WireNodeDataItem>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wire_format_size() {
        // Verify that WireNodeDataItem is exactly 28 bytes
        assert_eq!(std::mem::size_of::<WireNodeDataItem>(), 28);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let nodes = vec![
            (1u32, BinaryNodeData {
                position: crate::types::vec3::Vec3Data::new(1.0, 2.0, 3.0),
                velocity: crate::types::vec3::Vec3Data::new(0.1, 0.2, 0.3),
                mass: 100,
                flags: 1,
                padding: [0, 0],
            }),
            (2u32, BinaryNodeData {
                position: crate::types::vec3::Vec3Data::new(4.0, 5.0, 6.0),
                velocity: crate::types::vec3::Vec3Data::new(0.4, 0.5, 0.6),
                mass: 200,
                flags: 1,
                padding: [0, 0],
            }),
        ];

        let encoded = encode_node_data(&nodes);
        
        // Verify encoded size matches expected wire format
        assert_eq!(encoded.len(), nodes.len() * std::mem::size_of::<WireNodeDataItem>());
        assert_eq!(encoded.len(), nodes.len() * 28);
        
        let decoded = decode_node_data(&encoded).unwrap();
        assert_eq!(nodes.len(), decoded.len());

        for ((orig_id, orig_data), (dec_id, dec_data)) in nodes.iter().zip(decoded.iter()) {
            assert_eq!(orig_id, dec_id);
            assert_eq!(orig_data.position, dec_data.position);
            assert_eq!(orig_data.velocity, dec_data.velocity);
            // Note: mass, flags, and padding are not transmitted over the wire,
            // so decoded values will have defaults (100, 0, [0,0])
            assert_eq!(dec_data.mass, 100u8);
            assert_eq!(dec_data.flags, 0u8);
            assert_eq!(dec_data.padding, [0u8, 0u8]);
        }
    }

    #[test]
    fn test_decode_invalid_data() {
        // Test with data that's not a multiple of wire item size (28 bytes)
        let result = decode_node_data(&[0u8; 27]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not a multiple of wire item size"));
        
        // Test with data that's too short but multiple of wire size
        let result = decode_node_data(&[0u8; 0]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_message_size_calculation() {
        let nodes = vec![
            (1u32, BinaryNodeData {
                position: crate::types::vec3::Vec3Data::new(1.0, 2.0, 3.0),
                velocity: crate::types::vec3::Vec3Data::new(0.1, 0.2, 0.3),
                mass: 100,
                flags: 1,
                padding: [0, 0],
            }),
        ];

        let size = calculate_message_size(&nodes);
        assert_eq!(size, 28);
        
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
    fn test_encode_with_agent_flags() {
        let nodes = vec![
            (1u32, BinaryNodeData {
                position: crate::types::vec3::Vec3Data::new(1.0, 2.0, 3.0),
                velocity: crate::types::vec3::Vec3Data::new(0.1, 0.2, 0.3),
                mass: 100,
                flags: 1,
                padding: [0, 0],
            }),
            (2u32, BinaryNodeData {
                position: crate::types::vec3::Vec3Data::new(4.0, 5.0, 6.0),
                velocity: crate::types::vec3::Vec3Data::new(0.4, 0.5, 0.6),
                mass: 200,
                flags: 1,
                padding: [0, 0],
            }),
        ];

        // Mark node 2 as an agent
        let agent_ids = vec![2u32];
        let encoded = encode_node_data_with_flags(&nodes, &agent_ids);
        
        // Verify encoded size
        assert_eq!(encoded.len(), nodes.len() * 28);
        
        let decoded = decode_node_data(&encoded).unwrap();
        assert_eq!(nodes.len(), decoded.len());

        // Verify that positions and velocities are preserved
        for ((orig_id, orig_data), (dec_id, dec_data)) in nodes.iter().zip(decoded.iter()) {
            assert_eq!(orig_id, dec_id); // IDs should match after flag stripping
            assert_eq!(orig_data.position, dec_data.position);
            assert_eq!(orig_data.velocity, dec_data.velocity);
        }
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
