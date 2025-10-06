// Test suite for node ID truncation bug fix
// Verifies that V2 protocol correctly handles node IDs > 16383

use crate::utils::binary_protocol::*;
use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::types::vec3::Vec3Data;

#[test]
fn test_v1_truncation_bug() {
    // Verify that V1 protocol DOES truncate (this is the bug we're fixing)
    let large_node_id = 20000u32; // Larger than 14-bit max (16383)

    #[allow(deprecated)]
    let wire_id_v1 = to_wire_id_v1(large_node_id);

    // V1 truncates to 14 bits
    assert_eq!(wire_id_v1 & 0x3FFF, 3616u16); // 20000 & 0x3FFF = 3616

    #[allow(deprecated)]
    let recovered_id = from_wire_id_v1(wire_id_v1);

    // Data loss: 20000 became 3616
    assert_eq!(recovered_id, 3616u32);
    assert_ne!(recovered_id, large_node_id);

    println!("✓ V1 protocol exhibits truncation bug as expected");
}

#[test]
fn test_v2_no_truncation() {
    // Verify that V2 protocol does NOT truncate
    let large_node_id = 20000u32;

    let wire_id_v2 = to_wire_id_v2(large_node_id);
    let recovered_id = from_wire_id_v2(wire_id_v2);

    // No data loss: 20000 stays 20000
    assert_eq!(recovered_id, large_node_id);

    println!("✓ V2 protocol preserves large node IDs");
}

#[test]
fn test_collision_scenarios() {
    // Test that different node IDs don't collide in V2
    let node_ids = vec![
        100u32,
        16383u32,  // Maximum V1 safe ID
        16384u32,  // First ID that would be truncated in V1
        20000u32,
        50000u32,
        100000u32,
        1000000u32,
    ];

    let mut wire_ids = Vec::new();

    for node_id in &node_ids {
        let wire_id = to_wire_id_v2(*node_id);
        wire_ids.push(wire_id);

        let recovered = from_wire_id_v2(wire_id);
        assert_eq!(recovered, *node_id, "Node ID {} was corrupted", node_id);
    }

    // Verify no collisions
    for i in 0..wire_ids.len() {
        for j in (i+1)..wire_ids.len() {
            assert_ne!(
                wire_ids[i] & 0x3FFFFFFF,
                wire_ids[j] & 0x3FFFFFFF,
                "Collision between node IDs {} and {}",
                node_ids[i],
                node_ids[j]
            );
        }
    }

    println!("✓ No collisions detected in V2 protocol");
}

#[test]
fn test_v1_collision_demonstration() {
    // Demonstrate that V1 would have collisions
    let id1 = 100u32;
    let id2 = 16384u32 + 100u32; // 16484

    #[allow(deprecated)]
    let wire1 = to_wire_id_v1(id1);
    #[allow(deprecated)]
    let wire2 = to_wire_id_v1(id2);

    // These should collide in V1 (both truncate to 100)
    assert_eq!(wire1 & 0x3FFF, wire2 & 0x3FFF);

    println!("✓ V1 collision demonstrated: {} and {} both truncate to {}", id1, id2, wire1 & 0x3FFF);
}

#[test]
fn test_v2_encode_decode_large_ids() {
    // Test full encode/decode with large node IDs
    let nodes = vec![
        (100u32, create_test_node(1.0, 2.0, 3.0)),
        (16383u32, create_test_node(4.0, 5.0, 6.0)),  // V1 max
        (16384u32, create_test_node(7.0, 8.0, 9.0)),  // V1 would fail
        (50000u32, create_test_node(10.0, 11.0, 12.0)),
        (100000u32, create_test_node(13.0, 14.0, 15.0)),
    ];

    // Encode with V2 protocol
    let encoded = encode_node_data(&nodes);

    // First byte should be protocol version 2
    assert_eq!(encoded[0], 2u8, "Protocol version should be V2");

    // Decode
    let decoded = decode_node_data(&encoded).expect("Decode should succeed");

    // Verify all node IDs are preserved
    assert_eq!(decoded.len(), nodes.len());

    for ((orig_id, orig_data), (dec_id, dec_data)) in nodes.iter().zip(decoded.iter()) {
        assert_eq!(orig_id, dec_id, "Node ID mismatch");
        assert_eq!(orig_data.x, dec_data.x, "X position mismatch");
        assert_eq!(orig_data.y, dec_data.y, "Y position mismatch");
        assert_eq!(orig_data.z, dec_data.z, "Z position mismatch");
    }

    println!("✓ V2 encode/decode preserves all large node IDs");
}

#[test]
fn test_automatic_v2_selection() {
    // Verify that V2 is automatically selected when needed
    let small_nodes = vec![
        (100u32, create_test_node(1.0, 2.0, 3.0)),
        (200u32, create_test_node(4.0, 5.0, 6.0)),
    ];

    let large_nodes = vec![
        (100u32, create_test_node(1.0, 2.0, 3.0)),
        (20000u32, create_test_node(4.0, 5.0, 6.0)),
    ];

    // Small IDs could use V1
    assert!(!needs_v2_protocol(&small_nodes), "Small IDs don't need V2");

    // Large IDs require V2
    assert!(needs_v2_protocol(&large_nodes), "Large IDs require V2");

    println!("✓ Automatic V2 selection works correctly");
}

#[test]
fn test_v2_with_flags() {
    // Test that agent/knowledge flags work with large node IDs
    let node_id = 50000u32;

    // Set agent flag
    let agent_id = set_agent_flag(node_id);
    assert!(is_agent_node(agent_id));
    assert_eq!(get_actual_node_id(agent_id), node_id);

    // Set knowledge flag
    let knowledge_id = set_knowledge_flag(node_id);
    assert!(is_knowledge_node(knowledge_id));
    assert_eq!(get_actual_node_id(knowledge_id), node_id);

    // Test wire format preservation
    let wire_agent = to_wire_id_v2(agent_id);
    let recovered_agent = from_wire_id_v2(wire_agent);
    assert_eq!(recovered_agent, agent_id);
    assert!(is_agent_node(recovered_agent));
    assert_eq!(get_actual_node_id(recovered_agent), node_id);

    println!("✓ V2 protocol preserves flags with large node IDs");
}

#[test]
fn test_v2_wire_format_size() {
    // Verify V2 wire format is exactly 38 bytes per node
    let nodes = vec![
        (50000u32, create_test_node(1.0, 2.0, 3.0)),
    ];

    let encoded = encode_node_data(&nodes);

    // 1 byte version + 38 bytes per node
    assert_eq!(encoded.len(), 1 + 38, "V2 wire format should be 39 bytes total (1 version + 38 data)");

    println!("✓ V2 wire format size is correct");
}

#[test]
fn test_v1_compatibility() {
    // Test that V1 format can still be decoded (for legacy support)
    let nodes = vec![
        (100u32, create_test_node(1.0, 2.0, 3.0)),
        (200u32, create_test_node(4.0, 5.0, 6.0)),
    ];

    // Manually encode as V1
    let mut buffer = Vec::new();
    buffer.push(1u8); // V1 protocol version

    for (node_id, node) in &nodes {
        #[allow(deprecated)]
        let wire_id = to_wire_id_v1(*node_id);
        buffer.extend_from_slice(&wire_id.to_le_bytes());
        buffer.extend_from_slice(&node.x.to_le_bytes());
        buffer.extend_from_slice(&node.y.to_le_bytes());
        buffer.extend_from_slice(&node.z.to_le_bytes());
        buffer.extend_from_slice(&node.vx.to_le_bytes());
        buffer.extend_from_slice(&node.vy.to_le_bytes());
        buffer.extend_from_slice(&node.vz.to_le_bytes());
        buffer.extend_from_slice(&f32::INFINITY.to_le_bytes());
        buffer.extend_from_slice(&(-1i32).to_le_bytes());
    }

    // Should decode successfully
    let decoded = decode_node_data(&buffer).expect("V1 decode should succeed");
    assert_eq!(decoded.len(), nodes.len());

    println!("✓ V1 format can still be decoded for legacy support");
}

#[test]
fn test_maximum_node_id() {
    // Test maximum supported node ID (30 bits = 1,073,741,823)
    let max_id = 0x3FFFFFFFu32; // 30 bits all set

    let wire_id = to_wire_id_v2(max_id);
    let recovered = from_wire_id_v2(wire_id);

    assert_eq!(get_actual_node_id(recovered), max_id);

    println!("✓ Maximum 30-bit node ID (1,073,741,823) supported");
}

#[test]
fn test_encode_with_types_v2() {
    // Test encoding with agent and knowledge node types using V2
    let nodes = vec![
        (100u32, create_test_node(1.0, 2.0, 3.0)),
        (20000u32, create_test_node(4.0, 5.0, 6.0)),
        (50000u32, create_test_node(7.0, 8.0, 9.0)),
    ];

    let agent_ids = vec![20000u32];
    let knowledge_ids = vec![50000u32];

    let encoded = encode_node_data_with_types(&nodes, &agent_ids, &knowledge_ids);

    // Should use V2 due to large IDs
    assert_eq!(encoded[0], 2u8);

    let decoded = decode_node_data(&encoded).expect("Decode should succeed");

    // Verify node IDs are preserved (flags are stripped in decode)
    assert_eq!(decoded.len(), 3);
    assert_eq!(decoded[0].0, 100u32);
    assert_eq!(decoded[1].0, 20000u32);
    assert_eq!(decoded[2].0, 50000u32);

    println!("✓ V2 encoding with node types works correctly");
}

// Helper function to create test node data
fn create_test_node(x: f32, y: f32, z: f32) -> BinaryNodeData {
    BinaryNodeData {
        node_id: 0,
        x,
        y,
        z,
        vx: 0.1,
        vy: 0.2,
        vz: 0.3,
    }
}

#[test]
fn test_stress_large_node_count() {
    // Stress test with many large node IDs
    let mut nodes = Vec::new();
    for i in 0..1000 {
        let node_id = 20000u32 + i;
        nodes.push((node_id, create_test_node(i as f32, i as f32, i as f32)));
    }

    let encoded = encode_node_data(&nodes);
    let decoded = decode_node_data(&encoded).expect("Decode should succeed");

    assert_eq!(decoded.len(), 1000);

    for (i, (node_id, _)) in decoded.iter().enumerate() {
        assert_eq!(*node_id, 20000u32 + i as u32, "Node ID {} mismatch", i);
    }

    println!("✓ Stress test with 1000 large node IDs passed");
}

#[test]
fn test_performance_comparison() {
    // Compare V1 vs V2 encoding size
    let nodes = vec![
        (100u32, create_test_node(1.0, 2.0, 3.0)),
    ];

    // V2 encoding
    let v2_encoded = encode_node_data(&nodes);
    let v2_size = v2_encoded.len();

    // V1 would be: 1 byte version + 34 bytes per node = 35 bytes
    // V2 is: 1 byte version + 38 bytes per node = 39 bytes

    assert_eq!(v2_size, 39);

    // Extra 4 bytes per node for u32 IDs
    let overhead_per_node = 4;
    let overhead_percentage = (overhead_per_node as f32 / 34.0) * 100.0;

    println!("✓ V2 overhead: {} bytes per node ({:.1}% increase)", overhead_per_node, overhead_percentage);
    println!("  Trade-off: +4 bytes/node for 65536x more unique IDs");
}
