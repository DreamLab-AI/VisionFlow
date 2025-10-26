pub mod queries;

// Re-export query structs and handlers
pub use queries::{
    ComputeShortestPaths,
    ComputeShortestPathsHandler,
    GetAutoBalanceNotifications,
    GetAutoBalanceNotificationsHandler,
    GetBotsGraphData,
    GetBotsGraphDataHandler,
    GetConstraints,
    GetConstraintsHandler,
    GetEquilibriumStatus,
    GetEquilibriumStatusHandler,
    GetGraphData,
    GetGraphDataHandler,
    GetNodeMap,
    GetNodeMapHandler,
    GetPhysicsState,
    GetPhysicsStateHandler,
};
