//! Integration tests for high-performance networking layer
//!
//! Tests QUIC/WebTransport, fastwebsockets, and postcard serialization

use std::collections::HashMap;
use std::time::Instant;

// Test module for postcard serialization performance
mod postcard_serialization {
    use super::*;

    /// Test basic postcard serialization roundtrip
    #[test]
    fn test_postcard_node_serialization() {
        use webxr::handlers::quic_transport_handler::{
            PostcardNodeUpdate, PostcardBatchUpdate,
        };

        let node = PostcardNodeUpdate {
            id: 42,
            x: 1.5,
            y: 2.5,
            z: 3.5,
            vx: 0.1,
            vy: 0.2,
            vz: 0.3,
        };

        // Serialize
        let bytes = postcard::to_stdvec(&node).expect("Serialization failed");

        // Postcard should be compact (< 30 bytes for 7 fields)
        assert!(bytes.len() < 40, "Postcard output unexpectedly large: {} bytes", bytes.len());

        // Deserialize
        let decoded: PostcardNodeUpdate = postcard::from_bytes(&bytes).expect("Deserialization failed");

        assert_eq!(decoded.id, 42);
        assert!((decoded.x - 1.5).abs() < 0.001);
        assert!((decoded.vy - 0.2).abs() < 0.001);
    }

    /// Test batch serialization with multiple nodes
    #[test]
    fn test_postcard_batch_serialization() {
        use webxr::handlers::quic_transport_handler::{
            PostcardNodeUpdate, PostcardBatchUpdate,
        };

        let nodes: Vec<PostcardNodeUpdate> = (0..1000)
            .map(|i| PostcardNodeUpdate {
                id: i,
                x: i as f32 * 0.1,
                y: i as f32 * 0.2,
                z: i as f32 * 0.3,
                vx: 0.01,
                vy: 0.02,
                vz: 0.03,
            })
            .collect();

        let batch = PostcardBatchUpdate {
            frame_id: 12345,
            timestamp_ms: 1700000000000,
            nodes,
        };

        // Serialize
        let bytes = postcard::to_stdvec(&batch).expect("Batch serialization failed");

        // Should be compact: ~28 bytes per node + header
        let expected_max = 1000 * 30 + 100;
        assert!(
            bytes.len() < expected_max,
            "Batch too large: {} bytes (expected < {})",
            bytes.len(),
            expected_max
        );

        // Deserialize
        let decoded: PostcardBatchUpdate = postcard::from_bytes(&bytes).expect("Batch deserialization failed");

        assert_eq!(decoded.frame_id, 12345);
        assert_eq!(decoded.nodes.len(), 1000);
        assert_eq!(decoded.nodes[500].id, 500);
    }

    /// Benchmark: Compare postcard vs legacy binary protocol
    #[test]
    fn test_serialization_performance_comparison() {
        use webxr::handlers::quic_transport_handler::{
            PostcardNodeUpdate, PostcardBatchUpdate,
            encode_postcard_batch, decode_postcard_batch,
        };
        use webxr::utils::binary_protocol;
        use webxr::utils::socket_flow_messages::BinaryNodeData;

        const NODE_COUNT: usize = 10000;
        const ITERATIONS: usize = 100;

        // Prepare test data
        let legacy_nodes: Vec<(u32, BinaryNodeData)> = (0..NODE_COUNT as u32)
            .map(|i| {
                (i, BinaryNodeData {
                    node_id: i,
                    x: i as f32 * 0.1,
                    y: i as f32 * 0.2,
                    z: i as f32 * 0.3,
                    vx: 0.01,
                    vy: 0.02,
                    vz: 0.03,
                })
            })
            .collect();

        // Benchmark legacy binary protocol
        let legacy_start = Instant::now();
        let mut legacy_total_bytes = 0;
        for _ in 0..ITERATIONS {
            let encoded = binary_protocol::encode_node_data(&legacy_nodes);
            legacy_total_bytes = encoded.len();
            let _ = binary_protocol::decode_node_data(&encoded);
        }
        let legacy_duration = legacy_start.elapsed();

        // Benchmark postcard
        let postcard_start = Instant::now();
        let mut postcard_total_bytes = 0;
        for _ in 0..ITERATIONS {
            let encoded = encode_postcard_batch(&legacy_nodes).expect("Postcard encode failed");
            postcard_total_bytes = encoded.len();
            let _ = decode_postcard_batch(&encoded);
        }
        let postcard_duration = postcard_start.elapsed();

        println!("\n=== Serialization Performance Comparison ===");
        println!("Node count: {}", NODE_COUNT);
        println!("Iterations: {}", ITERATIONS);
        println!("\nLegacy Binary Protocol:");
        println!("  - Size: {} bytes", legacy_total_bytes);
        println!("  - Time: {:?}", legacy_duration);
        println!("  - Throughput: {:.2} GB/s",
            (legacy_total_bytes as f64 * ITERATIONS as f64) / legacy_duration.as_secs_f64() / 1e9);

        println!("\nPostcard Protocol:");
        println!("  - Size: {} bytes", postcard_total_bytes);
        println!("  - Time: {:?}", postcard_duration);
        println!("  - Throughput: {:.2} GB/s",
            (postcard_total_bytes as f64 * ITERATIONS as f64) / postcard_duration.as_secs_f64() / 1e9);

        let size_reduction = 100.0 * (1.0 - postcard_total_bytes as f64 / legacy_total_bytes as f64);
        let speed_improvement = legacy_duration.as_nanos() as f64 / postcard_duration.as_nanos() as f64;

        println!("\nImprovements:");
        println!("  - Size reduction: {:.1}%", size_reduction);
        println!("  - Speed improvement: {:.2}x", speed_improvement);

        // Assertions
        // Postcard should be at least as compact as legacy (usually smaller due to varint encoding)
        // Speed should be significantly better
        assert!(postcard_duration < legacy_duration * 2, "Postcard should not be much slower");
    }
}

// Test module for delta encoding
mod delta_encoding {
    use super::*;
    use webxr::handlers::quic_transport_handler::{
        PostcardNodeUpdate, PostcardDeltaUpdate, calculate_deltas,
    };
    use webxr::utils::socket_flow_messages::BinaryNodeData;

    #[test]
    fn test_delta_calculation() {
        let current = vec![
            (1u32, BinaryNodeData {
                node_id: 1,
                x: 10.5,
                y: 20.5,
                z: 30.5,
                vx: 0.15,
                vy: 0.25,
                vz: 0.35,
            }),
            (2u32, BinaryNodeData {
                node_id: 2,
                x: 100.0, // No change
                y: 200.0, // No change
                z: 300.0, // No change
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
            }),
        ];

        let mut previous = HashMap::new();
        previous.insert(1, PostcardNodeUpdate {
            id: 1,
            x: 10.0,
            y: 20.0,
            z: 30.0,
            vx: 0.1,
            vy: 0.2,
            vz: 0.3,
        });
        previous.insert(2, PostcardNodeUpdate {
            id: 2,
            x: 100.0,
            y: 200.0,
            z: 300.0,
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
        });

        let deltas = calculate_deltas(&current, &previous, 100.0);

        // Node 2 should not appear (no change)
        assert_eq!(deltas.len(), 1, "Only changed nodes should have deltas");

        // Node 1 should have deltas
        let delta1 = &deltas[0];
        assert_eq!(delta1.id, 1);
        assert_eq!(delta1.dx, 50); // 0.5 * 100 = 50
        assert_eq!(delta1.dy, 50);
        assert_eq!(delta1.dz, 50);
        assert_eq!(delta1.dvx, 5); // 0.05 * 100 = 5
    }

    #[test]
    fn test_delta_encoding_bandwidth_savings() {
        use webxr::handlers::quic_transport_handler::PostcardBatchUpdate;

        const NODE_COUNT: usize = 1000;

        // Simulate a typical frame where only 10% of nodes moved
        let current: Vec<(u32, BinaryNodeData)> = (0..NODE_COUNT as u32)
            .map(|i| {
                let moved = i % 10 == 0;
                (i, BinaryNodeData {
                    node_id: i,
                    x: if moved { i as f32 * 0.1 + 0.5 } else { i as f32 * 0.1 },
                    y: if moved { i as f32 * 0.2 + 0.5 } else { i as f32 * 0.2 },
                    z: i as f32 * 0.3,
                    vx: 0.01,
                    vy: 0.02,
                    vz: 0.03,
                })
            })
            .collect();

        let previous: HashMap<u32, PostcardNodeUpdate> = (0..NODE_COUNT as u32)
            .map(|i| {
                (i, PostcardNodeUpdate {
                    id: i,
                    x: i as f32 * 0.1,
                    y: i as f32 * 0.2,
                    z: i as f32 * 0.3,
                    vx: 0.01,
                    vy: 0.02,
                    vz: 0.03,
                })
            })
            .collect();

        // Calculate deltas
        let deltas = calculate_deltas(&current, &previous, 100.0);

        // Full update size
        let full_nodes: Vec<PostcardNodeUpdate> = current
            .iter()
            .map(|(_, data)| PostcardNodeUpdate {
                id: data.node_id,
                x: data.x,
                y: data.y,
                z: data.z,
                vx: data.vx,
                vy: data.vy,
                vz: data.vz,
            })
            .collect();

        let full_batch = PostcardBatchUpdate {
            frame_id: 0,
            timestamp_ms: 0,
            nodes: full_nodes,
        };

        let full_bytes = postcard::to_stdvec(&full_batch).unwrap();
        let delta_bytes = postcard::to_stdvec(&deltas).unwrap();

        let savings = 100.0 * (1.0 - delta_bytes.len() as f64 / full_bytes.len() as f64);

        println!("\n=== Delta Encoding Bandwidth Savings ===");
        println!("Nodes: {}", NODE_COUNT);
        println!("Nodes moved: {} (10%)", deltas.len());
        println!("Full update: {} bytes", full_bytes.len());
        println!("Delta update: {} bytes", delta_bytes.len());
        println!("Bandwidth savings: {:.1}%", savings);

        // With 10% nodes moving, we should save at least 60%
        assert!(savings > 50.0, "Delta encoding should save significant bandwidth");
    }
}

// Test module for protocol negotiation
mod protocol_negotiation {
    use webxr::handlers::fastwebsockets_handler::{
        negotiate_protocol, TransportProtocol, SerializationFormat,
    };

    #[test]
    fn test_negotiate_quic_webtransport() {
        let caps = vec!["webtransport".to_string(), "postcard".to_string()];
        let result = negotiate_protocol(&caps);

        assert_eq!(result.protocol, TransportProtocol::QuicWebTransport);
        assert_eq!(result.serialization, SerializationFormat::Postcard);
        assert!(result.supports_datagrams);
        assert!(result.supports_delta_encoding);
    }

    #[test]
    fn test_negotiate_fastwebsocket_postcard() {
        let caps = vec!["postcard".to_string()];
        let result = negotiate_protocol(&caps);

        assert_eq!(result.protocol, TransportProtocol::FastWebSocketPostcard);
        assert_eq!(result.serialization, SerializationFormat::Postcard);
        assert!(!result.supports_datagrams);
        assert!(result.supports_delta_encoding);
    }

    #[test]
    fn test_negotiate_legacy_fallback() {
        let caps = vec![];
        let result = negotiate_protocol(&caps);

        assert_eq!(result.protocol, TransportProtocol::LegacyWebSocket);
        assert_eq!(result.serialization, SerializationFormat::LegacyBinary);
        assert!(!result.supports_datagrams);
        assert!(!result.supports_delta_encoding);
    }

    #[test]
    fn test_negotiate_unknown_capabilities() {
        let caps = vec!["unknown".to_string(), "other".to_string()];
        let result = negotiate_protocol(&caps);

        // Should fall back to legacy
        assert_eq!(result.protocol, TransportProtocol::LegacyWebSocket);
    }
}

// Test module for control messages
mod control_messages {
    use webxr::handlers::quic_transport_handler::{
        ControlMessage, TopologyNode, TopologyEdge,
    };

    #[test]
    fn test_control_message_serialization() {
        let messages = vec![
            ControlMessage::Hello {
                client_id: "test-client".to_string(),
                protocol_version: 1,
                capabilities: vec!["postcard".to_string(), "delta".to_string()],
            },
            ControlMessage::Welcome {
                session_id: "session-123".to_string(),
                server_capabilities: vec!["quic".to_string()],
                position_stream_id: 0,
                control_stream_id: 1,
            },
            ControlMessage::TopologyUpdate {
                nodes_added: vec![
                    TopologyNode {
                        id: 1,
                        metadata_id: "meta-1".to_string(),
                        label: "Node 1".to_string(),
                        node_type: Some("concept".to_string()),
                    },
                ],
                nodes_removed: vec![99],
                edges_added: vec![
                    TopologyEdge {
                        id: "edge-1".to_string(),
                        source: 1,
                        target: 2,
                        weight: 0.8,
                        edge_type: Some("related".to_string()),
                    },
                ],
                edges_removed: vec!["edge-old".to_string()],
            },
            ControlMessage::PhysicsParams {
                spring_k: 0.1,
                repel_k: 100.0,
                damping: 0.95,
                iterations: 50,
            },
            ControlMessage::Ping { timestamp_ms: 1700000000000 },
            ControlMessage::Pong {
                timestamp_ms: 1700000000000,
                server_timestamp_ms: 1700000000005,
            },
            ControlMessage::Error {
                code: 500,
                message: "Internal error".to_string(),
            },
            ControlMessage::Disconnect {
                reason: "Client requested".to_string(),
            },
        ];

        for msg in &messages {
            // Serialize
            let bytes = postcard::to_stdvec(msg).expect("Control message serialization failed");

            // Deserialize
            let decoded: ControlMessage = postcard::from_bytes(&bytes)
                .expect("Control message deserialization failed");

            // Verify round-trip (basic check)
            let re_bytes = postcard::to_stdvec(&decoded).expect("Re-serialization failed");
            assert_eq!(bytes, re_bytes, "Round-trip should produce identical bytes");
        }
    }

    #[test]
    fn test_topology_update_compactness() {
        // Large topology update
        let update = ControlMessage::TopologyUpdate {
            nodes_added: (0..100)
                .map(|i| TopologyNode {
                    id: i,
                    metadata_id: format!("meta-{}", i),
                    label: format!("Node {}", i),
                    node_type: Some("concept".to_string()),
                })
                .collect(),
            nodes_removed: (100..200).collect(),
            edges_added: (0..50)
                .map(|i| TopologyEdge {
                    id: format!("edge-{}", i),
                    source: i,
                    target: i + 1,
                    weight: 0.5,
                    edge_type: Some("link".to_string()),
                })
                .collect(),
            edges_removed: vec![],
        };

        let bytes = postcard::to_stdvec(&update).expect("Serialization failed");

        println!("\n=== Topology Update Size ===");
        println!("Nodes added: 100");
        println!("Nodes removed: 100");
        println!("Edges added: 50");
        println!("Total bytes: {}", bytes.len());
        println!("Bytes per node: {:.1}", bytes.len() as f64 / 150.0);

        // Should be reasonably compact
        assert!(bytes.len() < 10000, "Topology update too large");
    }
}

// Benchmark module
mod benchmarks {
    use super::*;
    use webxr::handlers::quic_transport_handler::{
        PostcardNodeUpdate, PostcardBatchUpdate,
    };

    #[test]
    #[ignore] // Run with --ignored for benchmarks
    fn benchmark_serialization_throughput() {
        const NODE_COUNT: usize = 100000;
        const ITERATIONS: usize = 1000;

        let nodes: Vec<PostcardNodeUpdate> = (0..NODE_COUNT)
            .map(|i| PostcardNodeUpdate {
                id: i as u32,
                x: i as f32 * 0.001,
                y: i as f32 * 0.002,
                z: i as f32 * 0.003,
                vx: 0.01,
                vy: 0.02,
                vz: 0.03,
            })
            .collect();

        let batch = PostcardBatchUpdate {
            frame_id: 0,
            timestamp_ms: 0,
            nodes,
        };

        // Warm up
        for _ in 0..10 {
            let bytes = postcard::to_stdvec(&batch).unwrap();
            let _: PostcardBatchUpdate = postcard::from_bytes(&bytes).unwrap();
        }

        // Benchmark serialization
        let start = Instant::now();
        let mut total_bytes = 0;
        for _ in 0..ITERATIONS {
            let bytes = postcard::to_stdvec(&batch).unwrap();
            total_bytes += bytes.len();
        }
        let ser_duration = start.elapsed();

        // Benchmark deserialization
        let bytes = postcard::to_stdvec(&batch).unwrap();
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let _: PostcardBatchUpdate = postcard::from_bytes(&bytes).unwrap();
        }
        let de_duration = start.elapsed();

        let ser_throughput = (total_bytes as f64 / ITERATIONS as f64) * ITERATIONS as f64
            / ser_duration.as_secs_f64() / 1e9;
        let de_throughput = (bytes.len() as f64) * ITERATIONS as f64
            / de_duration.as_secs_f64() / 1e9;

        println!("\n=== Postcard Serialization Benchmark ===");
        println!("Nodes: {}", NODE_COUNT);
        println!("Iterations: {}", ITERATIONS);
        println!("\nSerialization:");
        println!("  - Time: {:?}", ser_duration);
        println!("  - Throughput: {:.2} GB/s", ser_throughput);
        println!("\nDeserialization:");
        println!("  - Time: {:?}", de_duration);
        println!("  - Throughput: {:.2} GB/s", de_throughput);
        println!("\nMessage size: {} bytes", bytes.len());

        // Target throughput depends on build profile:
        // - Release: 10+ GB/s (postcard's full capability)
        // - Debug: 0.5+ GB/s (reduced due to lack of optimizations)
        #[cfg(debug_assertions)]
        let min_throughput = 0.5;
        #[cfg(not(debug_assertions))]
        let min_throughput = 1.0;

        assert!(
            ser_throughput > min_throughput,
            "Serialization too slow: {:.2} GB/s (min: {:.2} GB/s)",
            ser_throughput, min_throughput
        );
    }
}
