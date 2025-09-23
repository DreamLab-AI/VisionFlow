/**
 * Enhanced dynamic logger system with registry management
 * Complements the existing loggerConfig.ts by providing centralized registry and monitoring
 * This provides a higher-level interface for managing logger instances across the application
 */

import {
  createDynamicLogger,
  createDynamicAgentLogger,
  updateAllLoggers as updateAllConfigLoggers,
  type LoggerOptions as ConfigLoggerOptions
} from './loggerConfig';
import { clientDebugState, type DebugKey } from './clientDebugState';

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LoggerOptions extends ConfigLoggerOptions {
  // Enhanced options for the dynamic logger registry
  category?: string; // Optional category for grouping loggers
  autoCleanup?: boolean; // Automatically remove logger when component unmounts
}

interface LoggerMetrics {
  namespace: string;
  category?: string;
  createdAt: number;
  lastUsed: number;
  messageCount: number;
  errorCount: number;
  isActive: boolean;
}

interface RegisteredLogger {
  namespace: string;
  category?: string;
  options: LoggerOptions;
  logger: ReturnType<typeof createDynamicLogger> | ReturnType<typeof createDynamicAgentLogger>;
  isAgent: boolean;
  createdAt: number;
  metrics: LoggerMetrics;
}

class DynamicLoggerManager {
  private loggerRegistry = new Map<string, RegisteredLogger>();
  private subscriptions: (() => void)[] = [];
  private initialized = false;
  private cleanupInterval: number | null = null;

  constructor() {
    this.initializeSubscriptions();
    this.startCleanupInterval();
  }

  private initializeSubscriptions(): void {
    if (this.initialized) return;

    // Subscribe to debug state changes that affect logging
    const debugKeys: DebugKey[] = ['enabled', 'consoleLogging', 'logLevel'];

    debugKeys.forEach(key => {
      const unsubscribe = clientDebugState.subscribe(key, () => {
        this.updateAllLoggers();
      });
      this.subscriptions.push(unsubscribe);
    });

    this.initialized = true;
  }

  private startCleanupInterval(): void {
    // Run cleanup every 5 minutes to remove inactive loggers
    this.cleanupInterval = window.setInterval(() => {
      this.cleanupInactiveLoggers();
    }, 5 * 60 * 1000);
  }

  private cleanupInactiveLoggers(): void {
    const now = Date.now();
    const inactiveThreshold = 10 * 60 * 1000; // 10 minutes

    const toRemove: string[] = [];
    this.loggerRegistry.forEach((registered, namespace) => {
      if (
        registered.options.autoCleanup &&
        now - registered.metrics.lastUsed > inactiveThreshold
      ) {
        toRemove.push(namespace);
      }
    });

    toRemove.forEach(namespace => {
      const registered = this.loggerRegistry.get(namespace);
      if (registered?.logger.destroy) {
        registered.logger.destroy();
      }
      this.loggerRegistry.delete(namespace);
    });

    if (toRemove.length > 0) {
      console.debug(`DynamicLoggerManager: Cleaned up ${toRemove.length} inactive loggers`);
    }
  }

  private updateAllLoggers(): void {
    // Use the existing loggerConfig system to update all loggers
    updateAllConfigLoggers();

    // Update our registry metrics
    this.loggerRegistry.forEach((registered, namespace) => {
      registered.metrics.lastUsed = Date.now();
    });
  }

  private wrapLoggerWithMetrics(
    logger: ReturnType<typeof createDynamicLogger> | ReturnType<typeof createDynamicAgentLogger>,
    namespace: string
  ): typeof logger {
    const registered = this.loggerRegistry.get(namespace);
    if (!registered) return logger;

    // Create a wrapper that tracks metrics
    const wrappedLogger = {
      ...logger,
      debug: (message: any, ...args: any[]) => {
        registered.metrics.lastUsed = Date.now();
        registered.metrics.messageCount++;
        return logger.debug(message, ...args);
      },
      info: (message: any, ...args: any[]) => {
        registered.metrics.lastUsed = Date.now();
        registered.metrics.messageCount++;
        return logger.info(message, ...args);
      },
      warn: (message: any, ...args: any[]) => {
        registered.metrics.lastUsed = Date.now();
        registered.metrics.messageCount++;
        return logger.warn(message, ...args);
      },
      error: (message: any, ...args: any[]) => {
        registered.metrics.lastUsed = Date.now();
        registered.metrics.messageCount++;
        registered.metrics.errorCount++;
        return logger.error(message, ...args);
      },
    };

    return wrappedLogger;
  }

  public createLogger(namespace: string, options: LoggerOptions = {}): ReturnType<typeof createDynamicLogger> {
    // Check if logger already exists
    const existing = this.loggerRegistry.get(namespace);
    if (existing && !existing.isAgent) {
      return existing.logger as ReturnType<typeof createDynamicLogger>;
    }

    // Create the logger using the existing dynamic system
    const logger = createDynamicLogger(namespace, options);

    // Create metrics
    const now = Date.now();
    const metrics: LoggerMetrics = {
      namespace,
      category: options.category,
      createdAt: now,
      lastUsed: now,
      messageCount: 0,
      errorCount: 0,
      isActive: true
    };

    // Register the logger for future updates and monitoring
    const registeredLogger: RegisteredLogger = {
      namespace,
      category: options.category,
      options,
      logger,
      isAgent: false,
      createdAt: now,
      metrics
    };

    this.loggerRegistry.set(namespace, registeredLogger);

    // Return wrapped logger with metrics tracking
    return this.wrapLoggerWithMetrics(logger, namespace) as ReturnType<typeof createDynamicLogger>;
  }

  public createAgentLogger(namespace: string, options: LoggerOptions = {}): ReturnType<typeof createDynamicAgentLogger> {
    // Check if logger already exists
    const existing = this.loggerRegistry.get(namespace);
    if (existing && existing.isAgent) {
      return existing.logger as ReturnType<typeof createDynamicAgentLogger>;
    }

    // Create the agent logger using the existing dynamic system
    const logger = createDynamicAgentLogger(namespace, options);

    // Create metrics
    const now = Date.now();
    const metrics: LoggerMetrics = {
      namespace,
      category: options.category,
      createdAt: now,
      lastUsed: now,
      messageCount: 0,
      errorCount: 0,
      isActive: true
    };

    // Register the logger for future updates and monitoring
    const registeredLogger: RegisteredLogger = {
      namespace,
      category: options.category,
      options,
      logger,
      isAgent: true,
      createdAt: now,
      metrics
    };

    this.loggerRegistry.set(namespace, registeredLogger);

    // Return wrapped logger with metrics tracking
    return this.wrapLoggerWithMetrics(logger, namespace) as ReturnType<typeof createDynamicAgentLogger>;
  }

  public getLogger(namespace: string): ReturnType<typeof createDynamicLogger> | ReturnType<typeof createDynamicAgentLogger> | undefined {
    const registered = this.loggerRegistry.get(namespace);
    return registered?.logger;
  }

  public unregisterLogger(namespace: string): boolean {
    const registered = this.loggerRegistry.get(namespace);
    if (registered?.logger.destroy) {
      registered.logger.destroy();
    }
    return this.loggerRegistry.delete(namespace);
  }

  public getRegisteredLoggers(): string[] {
    return Array.from(this.loggerRegistry.keys());
  }

  public getLoggersByCategory(category: string): RegisteredLogger[] {
    return Array.from(this.loggerRegistry.values()).filter(
      registered => registered.category === category
    );
  }

  public getLoggerMetrics(namespace?: string): LoggerMetrics[] {
    if (namespace) {
      const registered = this.loggerRegistry.get(namespace);
      return registered ? [registered.metrics] : [];
    }
    return Array.from(this.loggerRegistry.values()).map(registered => registered.metrics);
  }

  public getCurrentSettings(): {
    enabled: boolean;
    consoleLogging: boolean;
    logLevel: LogLevel;
    registeredCount: number;
    categoryCounts: Record<string, number>;
  } {
    const categoryCounts: Record<string, number> = {};
    this.loggerRegistry.forEach(registered => {
      const category = registered.category || 'uncategorized';
      categoryCounts[category] = (categoryCounts[category] || 0) + 1;
    });

    return {
      enabled: clientDebugState.isEnabled(),
      consoleLogging: clientDebugState.get('consoleLogging'),
      logLevel: clientDebugState.get('logLevel') || 'info',
      registeredCount: this.loggerRegistry.size,
      categoryCounts
    };
  }

  public cleanup(): void {
    // Clear cleanup interval
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }

    // Unsubscribe from all debug state changes
    this.subscriptions.forEach(unsubscribe => unsubscribe());
    this.subscriptions = [];

    // Cleanup all loggers
    this.loggerRegistry.forEach(registered => {
      if (registered.logger.destroy) {
        registered.logger.destroy();
      }
    });

    // Clear logger registry
    this.loggerRegistry.clear();

    this.initialized = false;
  }
}

// Create singleton instance
const dynamicLoggerManager = new DynamicLoggerManager();

// Enhanced logger creation functions that use the dynamic system
export function createLogger(namespace: string, options: LoggerOptions = {}): ReturnType<typeof createDynamicLogger> {
  return dynamicLoggerManager.createLogger(namespace, options);
}

export function createAgentLogger(namespace: string, options: LoggerOptions = {}): ReturnType<typeof createDynamicAgentLogger> {
  return dynamicLoggerManager.createAgentLogger(namespace, options);
}

// Utility functions for managing the logger system
export const loggerManager = {
  getLogger: (namespace: string) => dynamicLoggerManager.getLogger(namespace),
  unregisterLogger: (namespace: string) => dynamicLoggerManager.unregisterLogger(namespace),
  getRegisteredLoggers: () => dynamicLoggerManager.getRegisteredLoggers(),
  getLoggersByCategory: (category: string) => dynamicLoggerManager.getLoggersByCategory(category),
  getLoggerMetrics: (namespace?: string) => dynamicLoggerManager.getLoggerMetrics(namespace),
  getCurrentSettings: () => dynamicLoggerManager.getCurrentSettings(),
  updateAllLoggers: () => dynamicLoggerManager.updateAllLoggers(),
  cleanup: () => dynamicLoggerManager.cleanup(),
};

// Logger categories for organization
export const LoggerCategories = {
  CORE: 'core',
  UI: 'ui',
  API: 'api',
  WEBSOCKET: 'websocket',
  PHYSICS: 'physics',
  RENDERING: 'rendering',
  PERFORMANCE: 'performance',
  DEBUG: 'debug',
  TELEMETRY: 'telemetry',
} as const;

// Convenience functions for common logger types
export const loggerFactory = {
  createCoreLogger: (namespace: string, options?: LoggerOptions) =>
    createLogger(namespace, { ...options, category: LoggerCategories.CORE }),

  createUILogger: (namespace: string, options?: LoggerOptions) =>
    createLogger(namespace, { ...options, category: LoggerCategories.UI }),

  createAPILogger: (namespace: string, options?: LoggerOptions) =>
    createLogger(namespace, { ...options, category: LoggerCategories.API }),

  createWebSocketLogger: (namespace: string, options?: LoggerOptions) =>
    createAgentLogger(namespace, { ...options, category: LoggerCategories.WEBSOCKET }),

  createPhysicsLogger: (namespace: string, options?: LoggerOptions) =>
    createAgentLogger(namespace, { ...options, category: LoggerCategories.PHYSICS }),

  createRenderingLogger: (namespace: string, options?: LoggerOptions) =>
    createAgentLogger(namespace, { ...options, category: LoggerCategories.RENDERING }),

  createPerformanceLogger: (namespace: string, options?: LoggerOptions) =>
    createAgentLogger(namespace, { ...options, category: LoggerCategories.PERFORMANCE }),

  createDebugLogger: (namespace: string, options?: LoggerOptions) =>
    createLogger(namespace, { ...options, category: LoggerCategories.DEBUG, autoCleanup: true }),

  createTelemetryLogger: (namespace: string, options?: LoggerOptions) =>
    createAgentLogger(namespace, { ...options, category: LoggerCategories.TELEMETRY }),
};

// Export types for external use
export type { LoggerOptions, LogLevel, LoggerMetrics, RegisteredLogger };
export type DynamicLogger = ReturnType<typeof createLogger>;
export type DynamicAgentLogger = ReturnType<typeof createAgentLogger>;

// Hook for cleanup on application shutdown
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    dynamicLoggerManager.cleanup();
  });
}