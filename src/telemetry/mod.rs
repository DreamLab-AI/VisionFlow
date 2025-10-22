//! Telemetry module for comprehensive system monitoring and logging
//!
//! Provides structured logging, metrics collection, and observability
//! for the WebXR graph visualization system.

pub mod agent_telemetry;

#[cfg(test)]
pub mod test_logging;

pub use agent_telemetry::{
    get_telemetry_logger, init_telemetry_logger, AgentTelemetryLogger, CorrelationId, LogLevel,
    Position3D, TelemetryEvent,
};
