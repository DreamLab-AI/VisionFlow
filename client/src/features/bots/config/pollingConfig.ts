import type { PollingConfig } from '../services/AgentPollingService';

/**
 * Default polling configurations for different scenarios
 */
export const POLLING_PRESETS = {
  // Real-time mode for active development
  realtime: {
    activePollingInterval: 500,   // 0.5s for very responsive updates
    idlePollingInterval: 2000,    // 2s when idle
    enableSmartPolling: true,
    maxRetries: 5,
    retryDelay: 1000
  } as PollingConfig,

  // Standard mode for normal operation
  standard: {
    activePollingInterval: 1000,  // 1s for active tasks
    idlePollingInterval: 5000,    // 5s when idle
    enableSmartPolling: true,
    maxRetries: 3,
    retryDelay: 2000
  } as PollingConfig,

  // Performance mode for reduced load
  performance: {
    activePollingInterval: 2000,  // 2s for active tasks
    idlePollingInterval: 10000,   // 10s when idle
    enableSmartPolling: true,
    maxRetries: 3,
    retryDelay: 3000
  } as PollingConfig,

  // Debug mode with very frequent updates
  debug: {
    activePollingInterval: 250,   // 0.25s for debugging
    idlePollingInterval: 1000,    // 1s when idle
    enableSmartPolling: false,    // Always use active interval
    maxRetries: 10,
    retryDelay: 500
  } as PollingConfig
};

/**
 * Activity thresholds for smart polling
 */
export const ACTIVITY_THRESHOLDS = {
  // Percentage of active agents to trigger active polling
  activeAgentThreshold: 0.2,      // 20% of agents active
  
  // Number of pending tasks to trigger active polling
  pendingTaskThreshold: 1,        // At least 1 pending task
  
  // Time since last data change to switch to idle (ms)
  idleTimeThreshold: 30000,       // 30 seconds of no changes
  
  // Minimum time between activity level changes (ms)
  activityDebounceTime: 5000      // 5 seconds
};