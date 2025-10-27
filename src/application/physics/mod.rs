// src/application/physics/mod.rs
//! Physics Domain - CQRS Application Layer
//!
//! Handles physics simulation control and status queries.

pub mod directives;
pub mod queries;

pub use directives::{
    ApplyConstraints, ApplyConstraintsHandler, StartSimulation, StartSimulationHandler,
    StopSimulation, StopSimulationHandler, UpdatePhysicsParams, UpdatePhysicsParamsHandler,
};
pub use queries::{
    GetPhysicsStatus, GetPhysicsStatusHandler, GetSimulationParams, GetSimulationParamsHandler,
};
