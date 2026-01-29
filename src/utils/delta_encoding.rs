// src/utils/delta_encoding.rs
//! Delta Encoding (Protocol V4) - P1-3 Feature
//!
//! Implements 60-80% bandwidth reduction for WebSocket position updates
//! by sending only changed node data in frames 1-59, with full state
//! resync every 60 frames.

use crate::utils::binary_protocol::*;
use crate::utils::socket_flow_messages::BinaryNodeData;
use log::{debug, trace};
use std::collections::HashMap;

// Delta encoding constants
const DELTA_POSITION_CHANGED: u8 = 0x01;
const DELTA_VELOCITY_CHANGED: u8 = 0x02;
const DELTA_ALL_CHANGED: u8 = DELTA_POSITION_CHANGED | DELTA_VELOCITY_CHANGED;

const DELTA_SCALE_FACTOR: f32 = 100.0; // Scale factor for i16 precision
const DELTA_ITEM_SIZE: usize = 20;     // Size of DeltaNodeData in bytes
const DELTA_RESYNC_INTERVAL: u64 = 60; // Full state every 60 frames
const PROTOCOL_V4: u8 = 4;

/// Maximum frames to keep in history (2 seconds at 60fps)
/// Prevents unbounded memory growth in delta encoding history
pub const MAX_HISTORY_FRAMES: usize = 120;

// Wire format sizes from binary_protocol
const WIRE_V2_ITEM_SIZE: usize = 36;

/// Delta-encoded position update (20 bytes per changed node)
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

/// Encodes nodes as delta updates (Protocol V4)
/// Frame 0 & 60: Full state (36 bytes per node)
/// Frames 1-59: Delta encoding (20 bytes per changed node only)
/// # Arguments
/// * `nodes` - Current node positions/velocities
/// * `previous_nodes` - Previous frame's node data (for delta calculation)
/// * `frame_number` - Current frame number (0-59, resets at 60)
/// * `agent_node_ids` - IDs of agent nodes (for flag setting)
/// * `knowledge_node_ids` - IDs of knowledge nodes (for flag setting)
/// # Returns
/// Binary-encoded delta frame with protocol version header
pub fn encode_node_data_delta(
    nodes: &[(u32, BinaryNodeData)],
    previous_nodes: &HashMap<u32, BinaryNodeData>,
    frame_number: u64,
    agent_node_ids: &[u32],
    knowledge_node_ids: &[u32],
) -> Vec<u8> {
    // Frame 0 or every 60th frame: send full state for resync
    if frame_number % DELTA_RESYNC_INTERVAL == 0 {
        trace!(
            "Delta encoding: Frame {} - FULL state resync ({} nodes)",
            frame_number,
            nodes.len()
        );
        return encode_node_data_extended(nodes, agent_node_ids, knowledge_node_ids, &[], &[], &[]);
    }

    // Frames 1-59: send only changes
    let mut changed_nodes = Vec::new();

    for (node_id, node) in nodes {
        if let Some(prev_node) = previous_nodes.get(node_id) {
            // Calculate deltas
            let dx = node.x - prev_node.x;
            let dy = node.y - prev_node.y;
            let dz = node.z - prev_node.z;
            let dvx = node.vx - prev_node.vx;
            let dvy = node.vy - prev_node.vy;
            let dvz = node.vz - prev_node.vz;

            // Determine what changed (with threshold to avoid noise)
            let position_changed = dx.abs() > 0.001 || dy.abs() > 0.001 || dz.abs() > 0.001;
            let velocity_changed = dvx.abs() > 0.0001 || dvy.abs() > 0.0001 || dvz.abs() > 0.0001;

            if position_changed || velocity_changed {
                let mut change_flags = 0u8;
                if position_changed {
                    change_flags |= DELTA_POSITION_CHANGED;
                }
                if velocity_changed {
                    change_flags |= DELTA_VELOCITY_CHANGED;
                }

                // Apply node type flags
                let flagged_id = if agent_node_ids.contains(node_id) {
                    set_agent_flag(*node_id)
                } else if knowledge_node_ids.contains(node_id) {
                    set_knowledge_flag(*node_id)
                } else {
                    *node_id
                };

                changed_nodes.push((*node_id, change_flags, dx, dy, dz, dvx, dvy, dvz, flagged_id));
            }
        } else {
            // New node not in previous frame - treat as full change
            let flagged_id = if agent_node_ids.contains(node_id) {
                set_agent_flag(*node_id)
            } else if knowledge_node_ids.contains(node_id) {
                set_knowledge_flag(*node_id)
            } else {
                *node_id
            };

            changed_nodes.push((
                *node_id,
                DELTA_ALL_CHANGED,
                node.x,
                node.y,
                node.z,
                node.vx,
                node.vy,
                node.vz,
                flagged_id,
            ));
        }
    }

    trace!(
        "Delta encoding: Frame {} - DELTA ({} changed out of {} total nodes)",
        frame_number,
        changed_nodes.len(),
        nodes.len()
    );

    // Encode delta frame
    let mut buffer = Vec::with_capacity(4 + changed_nodes.len() * DELTA_ITEM_SIZE);

    // Protocol version
    buffer.push(PROTOCOL_V4);

    // Frame number (1 byte, modulo 256)
    buffer.push((frame_number % 256) as u8);

    // Number of changed nodes (2 bytes)
    let num_changed = changed_nodes.len() as u16;
    buffer.extend_from_slice(&num_changed.to_le_bytes());

    // Encode each changed node
    for (_node_id, change_flags, dx, dy, dz, dvx, dvy, dvz, flagged_id) in changed_nodes {
        // Node ID with flags
        buffer.extend_from_slice(&flagged_id.to_le_bytes());

        // Change flags
        buffer.push(change_flags);

        // Padding
        buffer.extend_from_slice(&[0u8; 3]);

        // Delta values (scaled to i16)
        let dx_scaled = (dx * DELTA_SCALE_FACTOR).clamp(-32768.0, 32767.0) as i16;
        let dy_scaled = (dy * DELTA_SCALE_FACTOR).clamp(-32768.0, 32767.0) as i16;
        let dz_scaled = (dz * DELTA_SCALE_FACTOR).clamp(-32768.0, 32767.0) as i16;
        let dvx_scaled = (dvx * DELTA_SCALE_FACTOR).clamp(-32768.0, 32767.0) as i16;
        let dvy_scaled = (dvy * DELTA_SCALE_FACTOR).clamp(-32768.0, 32767.0) as i16;
        let dvz_scaled = (dvz * DELTA_SCALE_FACTOR).clamp(-32768.0, 32767.0) as i16;

        buffer.extend_from_slice(&dx_scaled.to_le_bytes());
        buffer.extend_from_slice(&dy_scaled.to_le_bytes());
        buffer.extend_from_slice(&dz_scaled.to_le_bytes());
        buffer.extend_from_slice(&dvx_scaled.to_le_bytes());
        buffer.extend_from_slice(&dvy_scaled.to_le_bytes());
        buffer.extend_from_slice(&dvz_scaled.to_le_bytes());
    }

    debug!(
        "Encoded delta frame {}: {} bytes for {} changed nodes (vs {} bytes for full state)",
        frame_number,
        buffer.len(),
        num_changed,
        1 + nodes.len() * WIRE_V2_ITEM_SIZE
    );

    buffer
}

/// Decodes delta-encoded node data (Protocol V4)
/// Reconstructs full node state by applying deltas to previous state
/// # Arguments
/// * `data` - Encoded delta frame data (without protocol version byte)
/// * `previous_state` - Previous frame's complete node state
/// # Returns
/// Updated node state after applying deltas
pub fn decode_node_data_delta(
    data: &[u8],
    previous_state: &HashMap<u32, BinaryNodeData>,
) -> Result<HashMap<u32, BinaryNodeData>, String> {
    if data.len() < 3 {
        return Err("Delta data too small".to_string());
    }

    // Parse header
    let frame_number = data[0];
    let num_changed = u16::from_le_bytes([data[1], data[2]]) as usize;

    let expected_size = 3 + num_changed * DELTA_ITEM_SIZE;
    if data.len() != expected_size {
        return Err(format!(
            "Invalid delta data size: expected {}, got {}",
            expected_size,
            data.len()
        ));
    }

    // Start with previous state
    let mut new_state = previous_state.clone();

    // Apply deltas
    let mut cursor = 3;
    for _ in 0..num_changed {
        if cursor + DELTA_ITEM_SIZE > data.len() {
            return Err("Unexpected end of delta data".to_string());
        }

        let chunk = &data[cursor..cursor + DELTA_ITEM_SIZE];

        // Parse delta item
        let wire_id = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let change_flags = chunk[4];
        // Skip padding bytes 5-7
        let dx_scaled = i16::from_le_bytes([chunk[8], chunk[9]]);
        let dy_scaled = i16::from_le_bytes([chunk[10], chunk[11]]);
        let dz_scaled = i16::from_le_bytes([chunk[12], chunk[13]]);
        let dvx_scaled = i16::from_le_bytes([chunk[14], chunk[15]]);
        let dvy_scaled = i16::from_le_bytes([chunk[16], chunk[17]]);
        let dvz_scaled = i16::from_le_bytes([chunk[18], chunk[19]]);

        // Convert deltas back to f32
        let dx = dx_scaled as f32 / DELTA_SCALE_FACTOR;
        let dy = dy_scaled as f32 / DELTA_SCALE_FACTOR;
        let dz = dz_scaled as f32 / DELTA_SCALE_FACTOR;
        let dvx = dvx_scaled as f32 / DELTA_SCALE_FACTOR;
        let dvy = dvy_scaled as f32 / DELTA_SCALE_FACTOR;
        let dvz = dvz_scaled as f32 / DELTA_SCALE_FACTOR;

        let actual_id = get_actual_node_id(wire_id);

        // Apply delta to previous state
        if let Some(prev_node) = new_state.get(&actual_id) {
            let mut updated_node = *prev_node;

            if (change_flags & DELTA_POSITION_CHANGED) != 0 {
                updated_node.x += dx;
                updated_node.y += dy;
                updated_node.z += dz;
            }

            if (change_flags & DELTA_VELOCITY_CHANGED) != 0 {
                updated_node.vx += dvx;
                updated_node.vy += dvy;
                updated_node.vz += dvz;
            }

            new_state.insert(actual_id, updated_node);
        } else {
            // New node not in previous state - use absolute values
            new_state.insert(
                actual_id,
                BinaryNodeData {
                    node_id: actual_id,
                    x: dx,
                    y: dy,
                    z: dz,
                    vx: dvx,
                    vy: dvy,
                    vz: dvz,
                },
            );
        }

        cursor += DELTA_ITEM_SIZE;
    }

    debug!(
        "Decoded delta frame {}: {} nodes updated",
        frame_number,
        num_changed
    );

    Ok(new_state)
}

/// Enforces maximum history size to prevent unbounded memory growth.
/// Call this after adding new frames to the history.
/// # Arguments
/// * `history` - Mutable reference to the history VecDeque
pub fn enforce_history_limit<T>(history: &mut std::collections::VecDeque<T>) {
    while history.len() > MAX_HISTORY_FRAMES {
        history.pop_front();
    }
}

/// Enforces maximum history size for Vec-based history storage.
/// Call this after adding new frames to the history.
/// # Arguments
/// * `history` - Mutable reference to the history Vec
pub fn enforce_history_limit_vec<T>(history: &mut Vec<T>) {
    if history.len() > MAX_HISTORY_FRAMES {
        let excess = history.len() - MAX_HISTORY_FRAMES;
        history.drain(0..excess);
    }
}

/// Calculate bandwidth savings from delta encoding
pub fn calculate_delta_savings(
    total_nodes: usize,
    changed_nodes: usize,
    frame_number: u64,
) -> (usize, usize, f32) {
    let full_size = 1 + total_nodes * WIRE_V2_ITEM_SIZE;

    let delta_size = if frame_number % DELTA_RESYNC_INTERVAL == 0 {
        full_size // Resync frame
    } else {
        4 + changed_nodes * DELTA_ITEM_SIZE // Delta frame
    };

    let savings = if full_size > 0 {
        ((full_size - delta_size) as f32 / full_size as f32) * 100.0
    } else {
        0.0
    };

    (full_size, delta_size, savings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_encoding_full_frame() {
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

        let previous = HashMap::new();

        // Frame 0 should be full state
        let encoded = encode_node_data_delta(&nodes, &previous, 0, &[], &[]);

        // Should use Protocol V3 for full state (encode_node_data_extended now uses V3)
        assert_eq!(encoded[0], 3); // PROTOCOL_V3
    }

    #[test]
    fn test_delta_encoding_changes_only() {
        let nodes = vec![
            (
                1u32,
                BinaryNodeData {
                    node_id: 1,
                    x: 1.1, // Changed
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

        let mut previous = HashMap::new();
        previous.insert(
            1u32,
            BinaryNodeData {
                node_id: 1,
                x: 1.0, // Previous value
                y: 2.0,
                z: 3.0,
                vx: 0.1,
                vy: 0.2,
                vz: 0.3,
            },
        );
        previous.insert(
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
        );

        // Frame 1 should be delta
        let encoded = encode_node_data_delta(&nodes, &previous, 1, &[], &[]);

        assert_eq!(encoded[0], PROTOCOL_V4);
        assert_eq!(encoded[1], 1); // Frame number

        // Only node 1 changed
        let num_changed = u16::from_le_bytes([encoded[2], encoded[3]]);
        assert_eq!(num_changed, 1);
    }

    #[test]
    fn test_delta_savings_calculation() {
        // Test with 100K nodes, 10% changing
        let (full_size, delta_size, savings) = calculate_delta_savings(100000, 10000, 1);

        // Full state: 1 + 100000 * 36 = 3,600,001 bytes
        assert_eq!(full_size, 3_600_001);

        // Delta frame: 4 + 10000 * 20 = 200,004 bytes
        assert_eq!(delta_size, 200_004);

        // Savings: ~94.4%
        assert!(savings > 94.0 && savings < 95.0);
    }

    #[test]
    fn test_resync_interval() {
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

        let previous = HashMap::new();

        // Frame 60 should be full resync
        let encoded = encode_node_data_delta(&nodes, &previous, 60, &[], &[]);
        assert_eq!(encoded[0], 3); // PROTOCOL_V3 for full state (encode_node_data_extended uses V3)

        // Frame 120 should also be full resync
        let encoded = encode_node_data_delta(&nodes, &previous, 120, &[], &[]);
        assert_eq!(encoded[0], 3); // PROTOCOL_V3 for full state
    }

    #[test]
    fn test_delta_decode_roundtrip() {
        let mut nodes = vec![
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

        let mut previous: HashMap<u32, BinaryNodeData> = nodes.iter().cloned().collect();

        // Make a change
        nodes[0].1.x = 1.5;

        // Encode delta
        let encoded = encode_node_data_delta(&nodes, &previous, 1, &[], &[]);

        assert_eq!(encoded[0], PROTOCOL_V4);

        // Decode delta
        let payload = &encoded[1..];
        let decoded = decode_node_data_delta(payload, &previous).unwrap();

        // Verify the change was applied
        assert!((decoded.get(&1).unwrap().x - 1.5).abs() < 0.01);
        assert_eq!(decoded.get(&2).unwrap().x, 4.0); // Unchanged node
    }
}
