/**
 * Dynamic logger configuration that integrates with Control Center settings
 * Supports both environment variables and runtime configuration changes
 */

// Moved from logger.ts - utility functions for error handling
export function createErrorMetadata(error: unknown): Record<string, any> {
  if (error instanceof Error) {
    return {
      message: error.message,
      name: error.name,
      stack: error.stack,
    };
  }
  if (typeof error === 'object' && error !== null) {
    try {
      const errorKeys = Object.getOwnPropertyNames(error);
      const serializableError = errorKeys.reduce((acc, key) => {
        acc[key] = (error as any)[key];
        return acc;
      }, {} as Record<string, any>);

      const serializedErrorString = JSON.stringify(serializableError, null, 2);
      return {
        message: `Non-Error object encountered. Details: ${serializedErrorString.substring(0, 500)}${serializedErrorString.length > 500 ? '...' : ''}`,
        originalErrorType: 'Object',
      };
    } catch (e) {
      return {
        message: `Non-Error object (serialization failed): ${String(error)}`,
        originalErrorType: typeof error,
      };
    }
  }
  return {
    message: `Unknown error type: ${String(error)}`,
    originalErrorType: typeof error,
  };
}

export function createDataMetadata(data: Record<string, any>): Record<string, any> {
  return {
    ...data,
    timestamp: new Date().toISOString(),
  };
}

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
import { clientDebugState } from './clientDebugState';

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LOG_LEVEL_PRIORITY: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

const LOG_COLORS = {
  debug: '#8c8c8c', // gray
  info: '#4c9aff',  // blue
  warn: '#ffab00',  // orange
  error: '#ff5630', // red
};

// Central log storage
const logStorage: LogEntry[] = [];
const DEFAULT_MAX_LOG_ENTRIES = 1000;

interface LogEntry {
  timestamp: string;
  level: LogLevel;
  namespace: string;
  message: string;
  args: any[];
}

// Base logger implementation
function createBaseLogger(namespace: string, options: LoggerOptions = {}) {
  const { disabled = false, level = 'info', maxLogEntries = DEFAULT_MAX_LOG_ENTRIES } = options;
  let currentDisabled = disabled;
  let currentLevel = level;
  let levelPriority = LOG_LEVEL_PRIORITY[level];

  function shouldLog(msgLevel: LogLevel): boolean {
    if (currentDisabled) return false;
    return LOG_LEVEL_PRIORITY[msgLevel] >= levelPriority;
  }

  function updateLevel(newLevel: LogLevel): void {
    currentLevel = newLevel;
    levelPriority = LOG_LEVEL_PRIORITY[newLevel];
  }

  function setEnabled(enabled: boolean): void {
    currentDisabled = !enabled;
  }

  function getCurrentConfig(): { level: LogLevel; enabled: boolean } {
    return {
      level: currentLevel,
      enabled: !currentDisabled
    };
  }

  function formatMessage(message: any): string {
    if (typeof message === 'string') return message;
    if (message instanceof Error) {
      return message.stack ? message.stack : message.message;
    }
    try {
      return JSON.stringify(message, (key, value) => {
        if (typeof value === 'object' && value !== null) {
          if (value === message && key !== '') return '[Circular Reference]';
        }
        return value;
      }, 2);
    } catch (e) {
      return String(message);
    }
  }

  function formatArgs(args: any[]): any[] {
    return args.map(arg => {
      if (arg instanceof Error) {
        return { message: arg.message, name: arg.name, stack: arg.stack };
      }
      return arg;
    });
  }

  function createLogMethod(logLevel: LogLevel) {
    return function(message: any, ...args: any[]) {
      if (!shouldLog(logLevel)) return;

      const color = LOG_COLORS[logLevel];
      const now = new Date();
      const timestamp = now.toISOString();
      const consoleTimestamp = now.toISOString().split('T')[1].slice(0, -1);
      const prefix = `%c[${consoleTimestamp}] [${namespace}]`;

      const formattedArgs = formatArgs(args);
      const formattedMessage = formatMessage(message);

      logStorage.push({
        timestamp,
        level: logLevel,
        namespace,
        message: formattedMessage,
        args: formattedArgs,
      });

      if (logStorage.length > maxLogEntries) {
        logStorage.shift();
      }

      console[logLevel === 'debug' ? 'log' : logLevel](
        `${prefix} ${formattedMessage}`,
        `color: ${color}; font-weight: bold;`,
        ...args
      );
    };
  }

  function getLogs(): LogEntry[] {
    return [...logStorage];
  }

  return {
    debug: createLogMethod('debug'),
    info: createLogMethod('info'),
    warn: createLogMethod('warn'),
    error: createLogMethod('error'),
    getLogs,
    updateLevel,
    setEnabled,
    getCurrentConfig,
    isEnabled: () => !currentDisabled,
    namespace,
  };
}

// Enhanced agent logger with telemetry capabilities
function createBaseAgentLogger(namespace: string, options: LoggerOptions = {}) {
  const baseLogger = createBaseLogger(namespace, options);

  const agentTelemetry: AgentTelemetryData[] = [];
  const webSocketTelemetry: WebSocketTelemetryData[] = [];
  const threeJSTelemetry: ThreeJSTelemetryData[] = [];
  const MAX_TELEMETRY_ENTRIES = 1000;

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
export const getDefaultLogLevel = getEnvironmentLogLevel;
export const DEFAULT_LOG_LEVEL = getEnvironmentLogLevel();

// Backward compatibility exports (deprecated)
export { createLogger as createLegacyLogger, createAgentLogger as createLegacyAgentLogger };

// Dynamic telemetry logger for system-wide telemetry that respects Control Center settings
export const dynamicAgentTelemetryLogger = createDynamicAgentLogger('agent-telemetry');

// Default logger instance for backward compatibility
export const logger = createDynamicLogger('app');