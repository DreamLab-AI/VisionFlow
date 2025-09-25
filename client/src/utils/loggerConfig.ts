/**
 * Dynamic logger configuration that integrates with Control Center settings
 * Supports both environment variables and runtime configuration changes
 */

import { createLogger as createBaseLogger, createErrorMetadata, createDataMetadata, LoggerOptions, DEFAULT_LOG_LEVEL } from './baseLogger';
import { clientDebugState } from './clientDebugState';

// Re-export utility functions for backward compatibility
export { createErrorMetadata, createDataMetadata };

// Agent-specific telemetry types
export interface AgentTelemetryData {
  agentId: string;
  agentType: string;
  action: string;
  timestamp: Date;
  metadata?: Record<string, any>;
  position?: { x: number; y: number; z: number };
  performance?: {
    renderTime?: number;
    frameRate?: number;
    meshCount?: number;
    triangleCount?: number;
  };
}

export interface WebSocketTelemetryData {
  messageType: string;
  direction: 'incoming' | 'outgoing';
  timestamp: Date;
  size?: number;
  metadata?: Record<string, any>;
}

export interface ThreeJSTelemetryData {
  action: 'position_update' | 'mesh_create' | 'animation_frame' | 'force_applied';
  objectId: string;
  position?: { x: number; y: number; z: number };
  rotation?: { x: number; y: number; z: number };
  timestamp: Date;
  metadata?: Record<string, any>;
}

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const MAX_TELEMETRY_ENTRIES = 1000;

// Enhanced agent logger with telemetry capabilities
function createBaseAgentLogger(namespace: string, options: LoggerOptions = {}) {
  const baseLogger = createBaseLogger(namespace, options);

  const agentTelemetry: AgentTelemetryData[] = [];
  const webSocketTelemetry: WebSocketTelemetryData[] = [];
  const threeJSTelemetry: ThreeJSTelemetryData[] = [];

  function trimTelemetryArray<T>(array: T[]) {
    if (array.length > MAX_TELEMETRY_ENTRIES) {
      array.splice(0, array.length - MAX_TELEMETRY_ENTRIES);
    }
  }

  return {
    ...baseLogger,

    // Agent-specific telemetry
    logAgentAction(agentId: string, agentType: string, action: string, metadata?: Record<string, any>, position?: { x: number; y: number; z: number }) {
      const telemetryData: AgentTelemetryData = {
        agentId,
        agentType,
        action,
        timestamp: new Date(),
        metadata,
        position
      };

      agentTelemetry.push(telemetryData);
      trimTelemetryArray(agentTelemetry);

      baseLogger.debug(`[AGENT:${agentType}:${agentId}] ${action}`, { metadata, position });

      try {
        const storedTelemetry = JSON.parse(localStorage.getItem('agent-telemetry') || '[]');
        storedTelemetry.push(telemetryData);
        if (storedTelemetry.length > MAX_TELEMETRY_ENTRIES) {
          storedTelemetry.splice(0, storedTelemetry.length - MAX_TELEMETRY_ENTRIES);
        }
        localStorage.setItem('agent-telemetry', JSON.stringify(storedTelemetry));
      } catch (e) {
        baseLogger.warn('Failed to store agent telemetry in localStorage:', e);
      }
    },

    // WebSocket message telemetry
    logWebSocketMessage(messageType: string, direction: 'incoming' | 'outgoing', metadata?: Record<string, any>, size?: number) {
      const telemetryData: WebSocketTelemetryData = {
        messageType,
        direction,
        timestamp: new Date(),
        size,
        metadata
      };

      webSocketTelemetry.push(telemetryData);
      trimTelemetryArray(webSocketTelemetry);

      baseLogger.debug(`[WS:${direction.toUpperCase()}:${messageType}] ${size ? `${size} bytes` : 'no size'}`, metadata);
    },

    // Three.js telemetry
    logThreeJSAction(action: ThreeJSTelemetryData['action'], objectId: string, position?: { x: number; y: number; z: number }, rotation?: { x: number; y: number; z: number }, metadata?: Record<string, any>) {
      const telemetryData: ThreeJSTelemetryData = {
        action,
        objectId,
        position,
        rotation,
        timestamp: new Date(),
        metadata
      };

      threeJSTelemetry.push(telemetryData);
      trimTelemetryArray(threeJSTelemetry);

      baseLogger.debug(`[THREE.JS:${action.toUpperCase()}:${objectId}]`, { position, rotation, metadata });
    },

    // Performance telemetry
    logPerformance(operation: string, duration: number, metadata?: Record<string, any>) {
      baseLogger.info(`[PERF:${operation}] ${duration.toFixed(2)}ms`, metadata);
    },

    // Get telemetry data
    getAgentTelemetry: () => [...agentTelemetry],
    getWebSocketTelemetry: () => [...webSocketTelemetry],
    getThreeJSTelemetry: () => [...threeJSTelemetry],

    // Clear telemetry
    clearTelemetry() {
      agentTelemetry.length = 0;
      webSocketTelemetry.length = 0;
      threeJSTelemetry.length = 0;
      localStorage.removeItem('agent-telemetry');
      baseLogger.info('Telemetry data cleared');
    }
  };
}

// Track created loggers for dynamic updates
const createdLoggers = new Set<{
  logger: any;
  namespace: string;
  options: LoggerOptions;
  updateLevel: (level: LogLevel, enabled: boolean) => void;
}>();

/**
 * Validate if a string is a valid log level
 */
function isValidLogLevel(level: any): level is LogLevel {
  const validLevels: LogLevel[] = ['debug', 'info', 'warn', 'error'];
  return validLevels.includes(level);
}

/**
 * Get the effective log level considering both environment variables and Control Center settings
 */
function getEffectiveLogLevel(options: LoggerOptions = {}): LogLevel {
  // If runtime settings should be ignored, use environment variables only
  if (options.respectRuntimeSettings === false) {
    return DEFAULT_LOG_LEVEL;
  }

  // Check if console logging is enabled in Control Center
  const consoleLoggingEnabled = clientDebugState.get('consoleLogging');
  if (!consoleLoggingEnabled) {
    // If console logging is disabled, still return a level but the logger will be disabled
    return options.level || DEFAULT_LOG_LEVEL;
  }

  // Get log level from Control Center settings
  const runtimeLogLevel = clientDebugState.get('logLevel') as LogLevel;
  if (runtimeLogLevel && isValidLogLevel(runtimeLogLevel)) {
    return runtimeLogLevel;
  }

  // Fallback to provided level or environment variables
  return options.level || DEFAULT_LOG_LEVEL;
}

/**
 * Check if logging should be enabled based on Control Center settings
 */
function shouldEnableLogging(options: LoggerOptions): boolean {
  if (options.respectRuntimeSettings === false) {
    return !options.disabled;
  }

  const consoleLoggingEnabled = clientDebugState.get('consoleLogging');
  return !options.disabled && consoleLoggingEnabled;
}

/**
 * Create a dynamic logger that responds to Control Center settings changes
 * @deprecated Use createDynamicLogger instead for new code
 */
export function createLogger(namespace: string, options: LoggerOptions = {}) {
  return createDynamicLogger(namespace, options);
}

/**
 * Create a dynamic logger that responds to Control Center settings changes
 */
export function createDynamicLogger(namespace: string, options: LoggerOptions = {}) {
  const effectiveLevel = getEffectiveLogLevel(options);
  const enabled = shouldEnableLogging(options);

  // Create the base logger (now supports dynamic updates)
  const logger = createBaseLogger(namespace, {
    ...options,
    level: effectiveLevel,
    disabled: !enabled
  });

  const updateLevel = (newLevel: LogLevel, newEnabled: boolean) => {
    // Use the base logger's dynamic configuration methods
    logger.updateLevel(newLevel);
    logger.setEnabled(newEnabled);
  };

  // Track this logger for updates
  const loggerInfo = {
    logger,
    namespace,
    options,
    updateLevel
  };
  createdLoggers.add(loggerInfo);

  // Return the logger with additional dynamic methods
  const dynamicLogger = {
    ...logger, // Include all base logger methods

    // Add method to manually update the logger
    updateSettings: () => {
      const newLevel = getEffectiveLogLevel(options);
      const newEnabled = shouldEnableLogging(options);
      updateLevel(newLevel, newEnabled);
    },

    // Add method to remove from tracking when logger is no longer needed
    destroy: () => {
      createdLoggers.delete(loggerInfo);
    }
  };

  return dynamicLogger;
}

/**
 * Create an agent logger with dynamic configuration
 * @deprecated Use createDynamicAgentLogger instead for new code
 */
export function createAgentLogger(namespace: string, options: LoggerOptions = {}) {
  return createDynamicAgentLogger(namespace, options);
}

/**
 * Create a dynamic agent logger that responds to Control Center settings changes
 */
export function createDynamicAgentLogger(namespace: string, options: LoggerOptions = {}) {
  const effectiveLevel = getEffectiveLogLevel(options);
  const enabled = shouldEnableLogging(options);

  // Create the base agent logger (now supports dynamic updates)
  const agentLogger = createBaseAgentLogger(namespace, {
    ...options,
    level: effectiveLevel,
    disabled: !enabled
  });

  const updateLevel = (newLevel: LogLevel, newEnabled: boolean) => {
    // Use the base logger's dynamic configuration methods
    agentLogger.updateLevel(newLevel);
    agentLogger.setEnabled(newEnabled);
  };

  // Track this logger for updates
  const loggerInfo = {
    logger: agentLogger,
    namespace,
    options,
    updateLevel
  };
  createdLoggers.add(loggerInfo);

  // Return the agent logger with additional dynamic methods
  const dynamicAgentLogger = {
    ...agentLogger, // Include all base agent logger methods

    // Add method to manually update the logger
    updateSettings: () => {
      const newLevel = getEffectiveLogLevel(options);
      const newEnabled = shouldEnableLogging(options);
      updateLevel(newLevel, newEnabled);
    },

    // Add method to remove from tracking when logger is no longer needed
    destroy: () => {
      createdLoggers.delete(loggerInfo);
    }
  };

  return dynamicAgentLogger;
}

/**
 * Update all tracked loggers when settings change
 * This should be called when Control Center settings are updated
 */
export function updateAllLoggers() {
  createdLoggers.forEach(({ updateLevel, options }) => {
    const newLevel = getEffectiveLogLevel(options);
    const newEnabled = shouldEnableLogging(options);
    updateLevel(newLevel, newEnabled);
  });
}

/**
 * Subscribe to Control Center settings changes and update loggers automatically
 */
function setupAutoUpdate() {
  // Subscribe to console logging toggle
  clientDebugState.subscribe('consoleLogging', () => {
    updateAllLoggers();
  });

  // Subscribe to log level changes
  clientDebugState.subscribe('logLevel', () => {
    updateAllLoggers();
  });
}

// Initialize auto-update when this module is loaded
if (typeof window !== 'undefined') {
  setupAutoUpdate();
}

// Export the current effective log level for reference
export { DEFAULT_LOG_LEVEL };

// Backward compatibility exports (deprecated)
export { createLogger as createLegacyLogger, createAgentLogger as createLegacyAgentLogger };

// Dynamic telemetry logger for system-wide telemetry that respects Control Center settings
export const dynamicAgentTelemetryLogger = createDynamicAgentLogger('agent-telemetry');

// Default logger instance for backward compatibility
export const logger = createDynamicLogger('app');