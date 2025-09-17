//! Telemetry module for comprehensive system monitoring and logging
//!
//! Provides structured logging, metrics collection, and observability
//! for the WebXR graph visualization system.

pub mod agent_telemetry;

#[cfg(test)]
pub mod test_logging;

pub use agent_telemetry::{
    AgentTelemetryLogger, CorrelationId, Position3D, TelemetryEvent, LogLevel,
    init_telemetry_logger, get_telemetry_logger
};