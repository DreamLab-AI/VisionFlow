/**
 * Logger Provider - Centralized initialization and management of the logging system
 * This provides the connection between UI settings and logger instances
 */

import { clientDebugState } from './clientDebugState';
import { loggerManager, createLogger } from './dynamicLogger';
import { updateAllLoggers } from './loggerConfig';

interface LoggerProviderConfig {
  enableAutoCleanup?: boolean;
  metricsReportingInterval?: number;
  defaultLogLevel?: 'debug' | 'info' | 'warn' | 'error';
  enableMetrics?: boolean;
}

class LoggerProvider {
  private initialized = false;
  private config: LoggerProviderConfig;
  private metricsInterval: number | null = null;
  private subscriptions: (() => void)[] = [];
  private logger: ReturnType<typeof createLogger>;

  constructor(config: LoggerProviderConfig = {}) {
    this.config = {
      enableAutoCleanup: true,
      metricsReportingInterval: 60000, // 1 minute
      defaultLogLevel: 'info',
      enableMetrics: true,
      ...config
    };

    // Create logger for the provider itself
    this.logger = createLogger('LoggerProvider', {
      category: 'core',
      level: this.config.defaultLogLevel
    });
  }

  /**
   * Initialize the logger provider and set up subscriptions
   */
  public initialize(): void {
    if (this.initialized) {
      this.logger.warn('Logger provider already initialized');
      return;
    }

    this.logger.info('Initializing logger provider', this.config);

    // Set up subscriptions to debug state changes
    this.setupDebugStateSubscriptions();

    // Set up metrics reporting if enabled
    if (this.config.enableMetrics && this.config.metricsReportingInterval) {
      this.setupMetricsReporting();
    }

    // Set up environment-specific configurations
    this.setupEnvironmentConfig();

    this.initialized = true;
    this.logger.info('Logger provider initialized successfully');
  }

  /**
   * Subscribe to client debug state changes and propagate to loggers
   */
  private setupDebugStateSubscriptions(): void {
    // Subscribe to master debug toggle
    const enabledUnsubscribe = clientDebugState.subscribe('enabled', (enabled) => {
      this.logger.info(`Debug enabled changed: ${enabled}`);
      this.updateAllLoggers();
    });

    // Subscribe to console logging toggle
    const consoleUnsubscribe = clientDebugState.subscribe('consoleLogging', (consoleLogging) => {
      this.logger.info(`Console logging changed: ${consoleLogging}`);
      this.updateAllLoggers();
    });

    // Subscribe to log level changes
    const logLevelUnsubscribe = clientDebugState.subscribe('logLevel', (logLevel) => {
      this.logger.info(`Log level changed: ${logLevel}`);
      this.updateAllLoggers();
    });

    // Store unsubscribe functions for cleanup
    this.subscriptions.push(enabledUnsubscribe, consoleUnsubscribe, logLevelUnsubscribe);
  }

  /**
   * Set up periodic metrics reporting
   */
  private setupMetricsReporting(): void {
    this.metricsInterval = window.setInterval(() => {
      this.reportMetrics();
    }, this.config.metricsReportingInterval!);
  }

  /**
   * Configure based on current environment
   */
  private setupEnvironmentConfig(): void {
    const isDev = import.meta.env?.DEV || import.meta.env?.MODE === 'development';
    const isTest = import.meta.env?.MODE === 'test';

    if (isDev) {
      this.logger.info('Development environment detected - enabling verbose logging');
      // In development, ensure debug logging is available by default
      if (!clientDebugState.get('logLevel')) {
        clientDebugState.set('logLevel', 'debug');
      }
    } else if (isTest) {
      this.logger.info('Test environment detected - configuring for testing');
      // In test environment, reduce noise
      clientDebugState.set('logLevel', 'warn');
    } else {
      this.logger.info('Production environment detected - configuring for production');
      // In production, default to info level
      if (!clientDebugState.get('logLevel')) {
        clientDebugState.set('logLevel', 'info');
      }
    }
  }

  /**
   * Update all registered loggers with current settings
   */
  private updateAllLoggers(): void {
    try {
      // Use the existing loggerConfig update mechanism
      updateAllLoggers();

      // Also trigger our registry update
      loggerManager.updateAllLoggers();

      this.logger.debug('All loggers updated with new settings');
    } catch (error) {
      this.logger.error('Failed to update loggers:', error);
    }
  }

  /**
   * Report current logger metrics
   */
  private reportMetrics(): void {
    if (!this.config.enableMetrics) return;

    try {
      const settings = loggerManager.getCurrentSettings();
      const metrics = loggerManager.getLoggerMetrics();

      const report = {
        timestamp: new Date().toISOString(),
        settings,
        totalLoggers: metrics.length,
        activeLoggers: metrics.filter(m => m.isActive).length,
        totalMessages: metrics.reduce((sum, m) => sum + m.messageCount, 0),
        totalErrors: metrics.reduce((sum, m) => sum + m.errorCount, 0),
        categoryCounts: settings.categoryCounts
      };

      this.logger.debug('Logger metrics report:', report);

      // Emit custom event for any listeners
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('logger-metrics', { detail: report }));
      }
    } catch (error) {
      this.logger.error('Failed to generate metrics report:', error);
    }
  }

  /**
   * Get current logger provider status
   */
  public getStatus(): {
    initialized: boolean;
    settings: ReturnType<typeof loggerManager.getCurrentSettings>;
    registeredLoggers: string[];
    metricsEnabled: boolean;
  } {
    return {
      initialized: this.initialized,
      settings: loggerManager.getCurrentSettings(),
      registeredLoggers: loggerManager.getRegisteredLoggers(),
      metricsEnabled: this.config.enableMetrics || false
    };
  }

  /**
   * Force update all loggers (useful for testing or manual refresh)
   */
  public forceUpdate(): void {
    this.updateAllLoggers();
    this.logger.info('Forced update of all loggers');
  }

  /**
   * Reset all debug settings to defaults
   */
  public resetToDefaults(): void {
    this.logger.info('Resetting debug settings to defaults');
    clientDebugState.reset();
    this.updateAllLoggers();
  }

  /**
   * Clean up the logger provider
   */
  public cleanup(): void {
    if (!this.initialized) return;

    this.logger.info('Cleaning up logger provider');

    // Clear metrics interval
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
      this.metricsInterval = null;
    }

    // Unsubscribe from debug state changes
    this.subscriptions.forEach(unsubscribe => unsubscribe());
    this.subscriptions = [];

    // Cleanup logger manager
    loggerManager.cleanup();

    this.initialized = false;
    this.logger.info('Logger provider cleanup complete');
  }

  /**
   * Enable or disable debug mode programmatically
   */
  public setDebugMode(enabled: boolean): void {
    this.logger.info(`Setting debug mode: ${enabled}`);
    clientDebugState.setEnabled(enabled);
  }

  /**
   * Set log level programmatically
   */
  public setLogLevel(level: 'debug' | 'info' | 'warn' | 'error'): void {
    this.logger.info(`Setting log level: ${level}`);
    clientDebugState.set('logLevel', level);
  }

  /**
   * Enable or disable console logging
   */
  public setConsoleLogging(enabled: boolean): void {
    this.logger.info(`Setting console logging: ${enabled}`);
    clientDebugState.set('consoleLogging', enabled);
  }
}

// Create singleton instance
let loggerProviderInstance: LoggerProvider | null = null;

/**
 * Get the singleton logger provider instance
 */
export function getLoggerProvider(config?: LoggerProviderConfig): LoggerProvider {
  if (!loggerProviderInstance) {
    loggerProviderInstance = new LoggerProvider(config);
  }
  return loggerProviderInstance;
}

/**
 * Initialize the logger provider system
 * Should be called early in the application lifecycle
 */
export function initializeLoggerProvider(config?: LoggerProviderConfig): void {
  const provider = getLoggerProvider(config);
  provider.initialize();
}

/**
 * React hook for using logger provider in components
 */
export function useLoggerProvider() {
  const provider = getLoggerProvider();

  return {
    status: provider.getStatus(),
    forceUpdate: () => provider.forceUpdate(),
    resetToDefaults: () => provider.resetToDefaults(),
    setDebugMode: (enabled: boolean) => provider.setDebugMode(enabled),
    setLogLevel: (level: 'debug' | 'info' | 'warn' | 'error') => provider.setLogLevel(level),
    setConsoleLogging: (enabled: boolean) => provider.setConsoleLogging(enabled),
  };
}

// Auto-cleanup on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    if (loggerProviderInstance) {
      loggerProviderInstance.cleanup();
    }
  });
}

// Export types
export type { LoggerProviderConfig };
export { LoggerProvider };