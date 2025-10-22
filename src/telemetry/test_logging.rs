//! Test module for verifying telemetry logging functionality
//!
//! This module contains tests and examples for the telemetry system.

use super::agent_telemetry::{
    AgentTelemetryLogger, CorrelationId, LogLevel, Position3D, TelemetryEvent,
};
use serde_json;
use std::collections::HashMap;

/// Test telemetry logging functionality
#[cfg(test)]
pub fn test_telemetry_logging() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger for testing
    let logger = AgentTelemetryLogger::new("/tmp/test_logs", 10)?;

    // Test 1: Agent spawn with origin position bug detection
    let agent_id = "test_agent_001";
    let origin_position = Position3D::origin();
    let normal_position = Position3D::new(100.0, 50.0, 25.0);

    let mut metadata = HashMap::new();
    metadata.insert("test_mode".to_string(), serde_json::json!(true));

    // Log origin position (should trigger bug detection)
    logger.log_agent_spawn(agent_id, origin_position, metadata.clone());

    // Log normal position
    logger.log_agent_spawn("test_agent_002", normal_position, metadata);

    // Test 2: Position updates
    let old_pos = Position3D::new(0.0, 0.0, 0.0);
    let new_pos = Position3D::new(10.0, 20.0, 30.0);
    logger.log_position_update(agent_id, old_pos, new_pos, "test_gpu_kernel");

    // Test 3: GPU execution
    logger.log_gpu_execution("test_force_kernel", 100, 15.5, 2.3);

    // Test 4: MCP message flow
    logger.log_mcp_message("test_message", "inbound", 1024, "success");

    // Test 5: Graph state change
    let mut graph_details = HashMap::new();
    graph_details.insert(
        "layout_algorithm".to_string(),
        serde_json::json!("force_directed"),
    );
    graph_details.insert("convergence_threshold".to_string(), serde_json::json!(0.01));
    logger.log_graph_state_change("node_added", 50, 75, graph_details);

    // Force flush all events
    logger.flush();

    println!("✅ Telemetry logging test completed successfully");
    println!("📁 Test logs written to: /tmp/test_logs/");

    Ok(())
}

/// Example usage demonstrating the telemetry system
pub fn example_usage() {
    use crate::telemetry::agent_telemetry::get_telemetry_logger;

    // Initialize global logger (this would be done in main.rs)
    if crate::telemetry::agent_telemetry::init_telemetry_logger("/app/logs", 100).is_ok() {
        println!("✅ Global telemetry logger initialized");
    }

    // Usage in application code
    if let Some(logger) = get_telemetry_logger() {
        // Example: Log agent spawn
        let correlation_id = CorrelationId::from_agent_id("example_agent");
        let position = Position3D::new(150.0, 200.0, 100.0);
        let mut metadata = HashMap::new();
        metadata.insert(
            "spawn_method".to_string(),
            serde_json::json!("websocket_connection"),
        );

        logger.log_agent_spawn("example_agent", position, metadata);

        // Example: Manual event creation with custom data
        let event = TelemetryEvent::new(
            correlation_id,
            LogLevel::DEBUG,
            "custom_category",
            "custom_event",
            "This is a custom telemetry event",
            "example_component",
        )
        .with_metadata("custom_field", serde_json::json!("custom_value"))
        .with_duration(42.5);

        logger.log_event(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_origin_detection() {
        let origin = Position3D::origin();
        assert!(origin.is_origin());
        assert_eq!(origin.magnitude, 0.0);

        let non_origin = Position3D::new(1.0, 0.0, 0.0);
        assert!(!non_origin.is_origin());
        assert_eq!(non_origin.magnitude, 1.0);
    }

    #[test]
    fn test_correlation_id_generation() {
        let id1 = CorrelationId::new();
        let id2 = CorrelationId::new();
        assert_ne!(id1.as_str(), id2.as_str());

        let agent_id = CorrelationId::from_agent_id("test");
        assert!(agent_id.as_str().contains("agent-test"));
    }

    #[test]
    fn test_telemetry_event_builder() {
        let correlation_id = CorrelationId::new();
        let event = TelemetryEvent::new(
            correlation_id,
            LogLevel::INFO,
            "test_category",
            "test_event",
            "Test message",
            "test_component",
        )
        .with_agent_id("test_agent")
        .with_position(Position3D::new(1.0, 2.0, 3.0))
        .with_duration(100.5)
        .with_metadata("key", serde_json::json!("value"));

        assert_eq!(event.category, "test_category");
        assert_eq!(event.agent_id, Some("test_agent".to_string()));
        assert!(event.position.is_some());
        assert_eq!(event.duration_ms, Some(100.5));
        assert!(event.metadata.contains_key("key"));
    }
}
