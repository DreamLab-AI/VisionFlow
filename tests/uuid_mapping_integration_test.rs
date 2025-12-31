//! Integration tests for UUID mapping functionality in MCP Session Bridge
//!
//! Tests cover:
//! - Bidirectional UUID <-> Swarm ID mapping
//! - Session discovery and linking
//! - Concurrent mapping operations
//! - Cache invalidation and refresh
//! - Telemetry query routing via UUID
//!
//! NOTE: These tests are disabled because the mcp_session_bridge module
//! has been removed or relocated. Re-enable when the module is available.

// Module webxr::services::mcp_session_bridge does not exist
// Commenting out all tests until the module is restored or relocated

/*
use std::sync::Arc;
use tokio::time::Duration;
use webxr::services::mcp_session_bridge::{McpSessionBridge, MonitoredSessionMetadata};

/// Test basic UUID to Swarm ID mapping
#[tokio::test]
async fn test_uuid_to_swarm_mapping() {
    let bridge = McpSessionBridge::new("test-container".to_string());

    let uuid = "session-uuid-001";
    let swarm_id = "swarm-abc123";

    // Link session to swarm
    bridge.link_session_to_swarm(uuid, swarm_id).await;

    // Verify forward mapping
    assert_eq!(
        bridge.get_swarm_id_for_session(uuid).await,
        Some(swarm_id.to_string()),
        "UUID should map to swarm ID"
    );

    // Verify reverse mapping
    assert_eq!(
        bridge.get_session_for_swarm(swarm_id).await,
        Some(uuid.to_string()),
        "Swarm ID should map back to UUID"
    );
}

/// Test bidirectional mapping consistency
#[tokio::test]
async fn test_bidirectional_mapping_consistency() {
    let bridge = McpSessionBridge::new("test-container".to_string());

    let test_pairs = vec![
        ("uuid-1", "swarm-1"),
        ("uuid-2", "swarm-2"),
        ("uuid-3", "swarm-3"),
    ];

    // Create all mappings
    for (uuid, swarm_id) in &test_pairs {
        bridge.link_session_to_swarm(uuid, swarm_id).await;
    }

    // Verify all forward mappings
    for (uuid, expected_swarm) in &test_pairs {
        let swarm_id = bridge.get_swarm_id_for_session(uuid).await;
        assert_eq!(
            swarm_id.as_deref(),
            Some(*expected_swarm),
            "Forward mapping failed for UUID: {}",
            uuid
        );
    }

    // Verify all reverse mappings
    for (expected_uuid, swarm_id) in &test_pairs {
        let uuid = bridge.get_session_for_swarm(swarm_id).await;
        assert_eq!(
            uuid.as_deref(),
            Some(*expected_uuid),
            "Reverse mapping failed for swarm: {}",
            swarm_id
        );
    }
}

/// Test mapping with non-existent UUIDs
#[tokio::test]
async fn test_mapping_with_nonexistent_uuid() {
    let bridge = McpSessionBridge::new("test-container".to_string());

    // Query non-existent UUID
    let result = bridge.get_swarm_id_for_session("non-existent-uuid").await;
    assert_eq!(result, None, "Non-existent UUID should return None");

    // Query non-existent swarm ID
    let result = bridge.get_session_for_swarm("non-existent-swarm").await;
    assert_eq!(result, None, "Non-existent swarm ID should return None");
}

/// Test concurrent mapping operations
#[tokio::test]
async fn test_concurrent_mapping_operations() {
    let bridge = Arc::new(McpSessionBridge::new("test-container".to_string()));
    let mut handles = Vec::new();

    // Spawn concurrent mapping tasks
    for i in 0..10 {
        let bridge_clone = Arc::clone(&bridge);
        let uuid = format!("concurrent-uuid-{}", i);
        let swarm_id = format!("concurrent-swarm-{}", i);

        let handle = tokio::spawn(async move {
            bridge_clone.link_session_to_swarm(&uuid, &swarm_id).await;

            // Verify mapping immediately
            let retrieved_swarm = bridge_clone.get_swarm_id_for_session(&uuid).await;
            assert_eq!(retrieved_swarm.as_deref(), Some(swarm_id.as_str()));
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle
            .await
            .expect("Concurrent task should complete successfully");
    }

    // Verify all mappings persisted
    for i in 0..10 {
        let uuid = format!("concurrent-uuid-{}", i);
        let expected_swarm = format!("concurrent-swarm-{}", i);

        let swarm_id = bridge.get_swarm_id_for_session(&uuid).await;
        assert_eq!(
            swarm_id.as_deref(),
            Some(expected_swarm.as_str()),
            "Concurrent mapping should persist for UUID: {}",
            uuid
        );
    }
}

/// Test mapping overwrite behavior
#[tokio::test]
async fn test_mapping_overwrite() {
    let bridge = McpSessionBridge::new("test-container".to_string());

    let uuid = "reusable-uuid";
    let swarm_id_1 = "swarm-first";
    let swarm_id_2 = "swarm-second";

    // Create initial mapping
    bridge.link_session_to_swarm(uuid, swarm_id_1).await;
    assert_eq!(
        bridge.get_swarm_id_for_session(uuid).await.as_deref(),
        Some(swarm_id_1)
    );

    // Overwrite mapping
    bridge.link_session_to_swarm(uuid, swarm_id_2).await;

    // Verify new mapping
    assert_eq!(
        bridge.get_swarm_id_for_session(uuid).await.as_deref(),
        Some(swarm_id_2),
        "Mapping should be overwritten"
    );

    // Old swarm ID should not reverse-map to UUID anymore
    assert_eq!(
        bridge.get_session_for_swarm(swarm_id_1).await,
        None,
        "Old swarm ID should not map to UUID after overwrite"
    );

    // New swarm ID should reverse-map correctly
    assert_eq!(
        bridge.get_session_for_swarm(swarm_id_2).await.as_deref(),
        Some(uuid),
        "New swarm ID should map to UUID"
    );
}

/// Test metadata cache initialization
#[tokio::test]
async fn test_metadata_cache_with_mapping() {
    let bridge = McpSessionBridge::new("test-container".to_string());

    let uuid = "metadata-test-uuid";
    let swarm_id = "metadata-test-swarm";

    // Create metadata by spawning (simulated via link)
    bridge.link_session_to_swarm(uuid, swarm_id).await;

    // List monitored sessions
    let sessions = bridge.list_monitored_sessions().await;

    // Should have at least one session in cache
    let session_exists = sessions.iter().any(|s| s.uuid == uuid);
    assert!(
        session_exists,
        "Session should be in metadata cache after linking"
    );

    // Verify swarm_id is set in metadata
    let session_metadata = sessions.iter().find(|s| s.uuid == uuid);
    if let Some(metadata) = session_metadata {
        assert_eq!(
            metadata.swarm_id.as_deref(),
            Some(swarm_id),
            "Metadata should contain swarm ID"
        );
    }
}

/// Test UUID format validation patterns
#[tokio::test]
async fn test_uuid_format_patterns() {
    let bridge = McpSessionBridge::new("test-container".to_string());

    let test_cases = vec![
        ("valid-uuid-v4-format", "swarm-1"),
        ("550e8400-e29b-41d4-a716-446655440000", "swarm-2"), // UUID v4
        ("session-2024-10-06", "swarm-3"),
        ("hyphenated-session-id", "swarm-4"),
        ("UPPER-CASE-UUID", "swarm-5"),
    ];

    for (uuid, swarm_id) in test_cases {
        bridge.link_session_to_swarm(uuid, swarm_id).await;

        let retrieved = bridge.get_swarm_id_for_session(uuid).await;
        assert_eq!(
            retrieved.as_deref(),
            Some(swarm_id),
            "UUID format '{}' should be handled correctly",
            uuid
        );
    }
}

/// Test cleanup of completed sessions
#[tokio::test]
async fn test_cleanup_completed_sessions() {
    let bridge = McpSessionBridge::new("test-container".to_string());

    // Create several mappings
    bridge
        .link_session_to_swarm("active-session", "swarm-active")
        .await;
    bridge
        .link_session_to_swarm("completed-session", "swarm-completed")
        .await;

    // Simulate cleanup (note: actual cleanup requires status = "Completed")
    // This test verifies the cleanup method can be called without errors
    let cleaned = bridge.cleanup_completed_sessions().await;

    // Since we haven't marked any as completed in this test, should be 0
    assert_eq!(
        cleaned, 0,
        "No sessions should be cleaned if none are completed"
    );
}

/// Test swarm ID retrieval with special characters
#[tokio::test]
async fn test_special_characters_in_identifiers() {
    let bridge = McpSessionBridge::new("test-container".to_string());

    let special_cases = vec![
        ("uuid-with-numbers-123", "swarm-456-abc"),
        ("uuid_with_underscores", "swarm_with_underscores"),
        ("uuid.with.dots", "swarm.with.dots"),
    ];

    for (uuid, swarm_id) in special_cases {
        bridge.link_session_to_swarm(uuid, swarm_id).await;

        assert_eq!(
            bridge.get_swarm_id_for_session(uuid).await.as_deref(),
            Some(swarm_id),
            "Special character handling failed for: {}",
            uuid
        );
    }
}

/// Test mapping persistence across multiple queries
#[tokio::test]
async fn test_mapping_persistence() {
    let bridge = McpSessionBridge::new("test-container".to_string());

    let uuid = "persistent-uuid";
    let swarm_id = "persistent-swarm";

    bridge.link_session_to_swarm(uuid, swarm_id).await;

    // Query multiple times
    for i in 0..5 {
        let result = bridge.get_swarm_id_for_session(uuid).await;
        assert_eq!(
            result.as_deref(),
            Some(swarm_id),
            "Mapping should persist across query #{}",
            i + 1
        );
    }
}

/// Integration test: Full lifecycle
#[tokio::test]
async fn test_full_session_lifecycle() {
    let bridge = Arc::new(McpSessionBridge::new("test-container".to_string()));

    let uuid = "lifecycle-test-uuid";
    let swarm_id = "lifecycle-test-swarm";

    // 1. Create mapping
    bridge.link_session_to_swarm(uuid, swarm_id).await;

    // 2. Verify bidirectional mapping
    assert_eq!(
        bridge.get_swarm_id_for_session(uuid).await.as_deref(),
        Some(swarm_id)
    );
    assert_eq!(
        bridge.get_session_for_swarm(swarm_id).await.as_deref(),
        Some(uuid)
    );

    // 3. List sessions
    let sessions = bridge.list_monitored_sessions().await;
    assert!(sessions.iter().any(|s| s.uuid == uuid));

    // 4. Simulate refresh (should not create duplicates)
    let discovered = bridge.refresh_mappings().await.unwrap_or(0);

    // 5. Verify mapping still intact
    assert_eq!(
        bridge.get_swarm_id_for_session(uuid).await.as_deref(),
        Some(swarm_id),
        "Mapping should survive refresh"
    );
}
*/
