/**
 * Dynamic logger configuration that integrates with Control Center settings
 * Supports both environment variables and runtime configuration changes
 */

import { createLogger as baseCreateLogger, createAgentLogger as baseCreateAgentLogger } from './logger';
import { clientDebugState } from './clientDebugState';

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LoggerOptions {
  disabled?: boolean;
  level?: LogLevel;
  maxLogEntries?: number;
  respectRuntimeSettings?: boolean;
}

// Track created loggers for dynamic updates
const createdLoggers = new Set<{
  logger: any;
  namespace: string;
  options: LoggerOptions;
  updateLevel: (level: LogLevel, enabled: boolean) => void;
}>();

/**
 * Get the effective log level considering both environment variables and Control Center settings
 */
function getEffectiveLogLevel(options: LoggerOptions = {}): LogLevel {
  // If runtime settings should be ignored, use environment variables only
  if (options.respectRuntimeSettings === false) {
    return getEnvironmentLogLevel();
  }

  // Check if console logging is enabled in Control Center
  const consoleLoggingEnabled = clientDebugState.get('consoleLogging');
  if (!consoleLoggingEnabled) {
    // If console logging is disabled, still return a level but the logger will be disabled
    return options.level || getEnvironmentLogLevel();
  }

  // Get log level from Control Center settings
  const runtimeLogLevel = clientDebugState.get('logLevel') as LogLevel;
  if (runtimeLogLevel && isValidLogLevel(runtimeLogLevel)) {
    return runtimeLogLevel;
  }

  // Fallback to provided level or environment variables
  return options.level || getEnvironmentLogLevel();
}

/**
 * Get the default log level from environment variables only
 */
function getEnvironmentLogLevel(): LogLevel {
  // Read from environment variable (Vite uses VITE_ prefix)
  const envLogLevel = import.meta.env?.VITE_LOG_LEVEL || import.meta.env?.LOG_LEVEL;

  // Validate and return the log level
  const level = envLogLevel?.toLowerCase();

  if (isValidLogLevel(level)) {
    return level as LogLevel;
  }

  // Default to 'debug' in development, 'info' in production
  const isDev = import.meta.env?.DEV || import.meta.env?.MODE === 'development';
  return isDev ? 'debug' : 'info';
}

/**
 * Validate if a string is a valid log level
 */
function isValidLogLevel(level: any): level is LogLevel {
  const validLevels: LogLevel[] = ['debug', 'info', 'warn', 'error'];
  return validLevels.includes(level);
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
  const logger = baseCreateLogger(namespace, {
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

// Helper function removed - now handled by base logger

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
  const agentLogger = baseCreateAgentLogger(namespace, {
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
export const getDefaultLogLevel = getEnvironmentLogLevel;
export const DEFAULT_LOG_LEVEL = getEnvironmentLogLevel();

// Backward compatibility exports (deprecated)
export { createLogger as createLegacyLogger, createAgentLogger as createLegacyAgentLogger };

// Dynamic telemetry logger for system-wide telemetry that respects Control Center settings
export const dynamicAgentTelemetryLogger = createDynamicAgentLogger('agent-telemetry');