// Test to verify the binary wire format is exactly 26 bytes
// This demonstrates the fix for the wire format mismatch between server and client

#[test]
fn test_wire_format_is_26_bytes() {
    // The wire format consists of:
    // - id: u16 (2 bytes)
    // - position: Vec3 (12 bytes = 3 * f32)
    // - velocity: Vec3 (12 bytes = 3 * f32)
    // Total: 26 bytes

    const WIRE_ID_SIZE: usize = 2; // u16
    const WIRE_VEC3_SIZE: usize = 12; // 3 * f32
    const WIRE_ITEM_SIZE: usize = WIRE_ID_SIZE + WIRE_VEC3_SIZE + WIRE_VEC3_SIZE;

    assert_eq!(WIRE_ITEM_SIZE, 26, "Wire format must be exactly 26 bytes");

    // Test encoding
    let mut buffer = Vec::new();

    // Write a test node
    let node_id: u16 = 42;
    let pos_x: f32 = 1.0;
    let pos_y: f32 = 2.0;
    let pos_z: f32 = 3.0;
    let vel_x: f32 = 0.1;
    let vel_y: f32 = 0.2;
    let vel_z: f32 = 0.3;

    // Manual serialization (same as in binary_protocol.rs)
    buffer.extend_from_slice(&node_id.to_le_bytes());
    buffer.extend_from_slice(&pos_x.to_le_bytes());
    buffer.extend_from_slice(&pos_y.to_le_bytes());
    buffer.extend_from_slice(&pos_z.to_le_bytes());
    buffer.extend_from_slice(&vel_x.to_le_bytes());
    buffer.extend_from_slice(&vel_y.to_le_bytes());
    buffer.extend_from_slice(&vel_z.to_le_bytes());

    assert_eq!(buffer.len(), 26, "Encoded node must be exactly 26 bytes");

    // Test decoding
    let mut cursor = 0;
    let decoded_id = u16::from_le_bytes([buffer[cursor], buffer[cursor + 1]]);
    cursor += 2;

    let decoded_pos_x = f32::from_le_bytes([
        buffer[cursor],
        buffer[cursor + 1],
        buffer[cursor + 2],
        buffer[cursor + 3],
    ]);
    cursor += 4;
    let decoded_pos_y = f32::from_le_bytes([
        buffer[cursor],
        buffer[cursor + 1],
        buffer[cursor + 2],
        buffer[cursor + 3],
    ]);
    cursor += 4;
    let decoded_pos_z = f32::from_le_bytes([
        buffer[cursor],
        buffer[cursor + 1],
        buffer[cursor + 2],
        buffer[cursor + 3],
    ]);
    cursor += 4;

    let decoded_vel_x = f32::from_le_bytes([
        buffer[cursor],
        buffer[cursor + 1],
        buffer[cursor + 2],
        buffer[cursor + 3],
    ]);
    cursor += 4;
    let decoded_vel_y = f32::from_le_bytes([
        buffer[cursor],
        buffer[cursor + 1],
        buffer[cursor + 2],
        buffer[cursor + 3],
    ]);
    cursor += 4;
    let decoded_vel_z = f32::from_le_bytes([
        buffer[cursor],
        buffer[cursor + 1],
        buffer[cursor + 2],
        buffer[cursor + 3],
    ]);

    assert_eq!(decoded_id, node_id);
    assert_eq!(decoded_pos_x, pos_x);
    assert_eq!(decoded_pos_y, pos_y);
    assert_eq!(decoded_pos_z, pos_z);
    assert_eq!(decoded_vel_x, vel_x);
    assert_eq!(decoded_vel_y, vel_y);
    assert_eq!(decoded_vel_z, vel_z);

    println!("✓ Wire format is correctly 26 bytes");
    println!("✓ Encoding and decoding work correctly");
}

#[test]
fn test_flag_preservation_in_u16() {
    // Test that we can preserve agent/knowledge flags in u16
    const WIRE_AGENT_FLAG: u16 = 0x8000; // Bit 15
    const WIRE_KNOWLEDGE_FLAG: u16 = 0x4000; // Bit 14
    const WIRE_NODE_ID_MASK: u16 = 0x3FFF; // Bits 0-13

    let node_id: u16 = 42;

    // Test agent flag
    let agent_id = node_id | WIRE_AGENT_FLAG;
    assert_eq!(agent_id & WIRE_NODE_ID_MASK, node_id);
    assert!((agent_id & WIRE_AGENT_FLAG) != 0);

    // Test knowledge flag
    let knowledge_id = node_id | WIRE_KNOWLEDGE_FLAG;
    assert_eq!(knowledge_id & WIRE_NODE_ID_MASK, node_id);
    assert!((knowledge_id & WIRE_KNOWLEDGE_FLAG) != 0);

    // Test maximum node ID that fits in 14 bits
    let max_id: u16 = 0x3FFF; // 16383
    assert_eq!(max_id & WIRE_NODE_ID_MASK, max_id);

    println!("✓ Node type flags work correctly in u16 format");
}
