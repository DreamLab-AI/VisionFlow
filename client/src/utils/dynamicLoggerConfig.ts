/**
 * Dynamic Logger Configuration Manager
 * Manages logger configuration from multiple sources with real-time updates
 */

import { clientDebugState } from './clientDebugState';
import { loggerRegistry, LoggerConfig } from './loggerRegistry';

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

class DynamicLoggerConfig {
  private currentConfig: LoggerConfig;
  private listeners: Set<(config: LoggerConfig) => void> = new Set();
  private debugStateUnsubscribe?: () => void;
  private consoleLoggingUnsubscribe?: () => void;

  constructor() {
    this.currentConfig = this.computeCurrentConfig();
    this.initializeSubscriptions();
  }

  /**
   * Get the current effective configuration
   */
  getCurrentConfig(): LoggerConfig {
    return { ...this.currentConfig };
  }

  /**
   * Subscribe to configuration changes
   */
  subscribe(callback: (config: LoggerConfig) => void): () => void {
    this.listeners.add(callback);

    // Return unsubscribe function
    return () => {
      this.listeners.delete(callback);
    };
  }

  /**
   * Compute current configuration based on priority hierarchy
   */
  private computeCurrentConfig(): LoggerConfig {
    // Priority: Runtime Settings > Environment Variables > Defaults

    const runtimeEnabled = clientDebugState.get('consoleLogging');
    const runtimeLogLevel = clientDebugState.get('logLevel') as LogLevel;

    // Check if runtime settings are explicitly set
    if (typeof runtimeEnabled === 'boolean' && runtimeLogLevel) {
      return {
        enabled: runtimeEnabled,
        level: runtimeLogLevel,
        source: 'runtime'
      };
    }

    // Fall back to environment variables
    const envLogLevel = this.getEnvironmentLogLevel();
    if (envLogLevel) {
      return {
        enabled: true, // If env var is set, assume enabled
        level: envLogLevel,
        source: 'environment'
      };
    }

    // System defaults
    return {
      enabled: this.getDefaultEnabled(),
      level: this.getDefaultLogLevel(),
      source: 'default'
    };
  }

  /**
   * Get log level from environment variables
   */
  private getEnvironmentLogLevel(): LogLevel | null {
    const envLogLevel = import.meta.env?.VITE_LOG_LEVEL || import.meta.env?.LOG_LEVEL;
    const validLevels: LogLevel[] = ['debug', 'info', 'warn', 'error'];
    const level = envLogLevel?.toLowerCase();

    if (validLevels.includes(level as LogLevel)) {
      return level as LogLevel;
    }

    return null;
  }

  /**
   * Get default enabled state
   */
  private getDefaultEnabled(): boolean {
    // Default to enabled in development, disabled in production for console logging
    return import.meta.env?.DEV || import.meta.env?.MODE === 'development';
  }

  /**
   * Get default log level
   */
  private getDefaultLogLevel(): LogLevel {
    // Default to 'debug' in development, 'info' in production
    const isDev = import.meta.env?.DEV || import.meta.env?.MODE === 'development';
    return isDev ? 'debug' : 'info';
  }

  /**
   * Initialize subscriptions to debug state changes
   */
  private initializeSubscriptions(): void {
    // Subscribe to console logging toggle changes
    this.consoleLoggingUnsubscribe = clientDebugState.subscribe('consoleLogging', () => {
      this.handleConfigChange();
    });

    // Subscribe to log level changes
    this.debugStateUnsubscribe = clientDebugState.subscribe('logLevel', () => {
      this.handleConfigChange();
    });
  }

  /**
   * Handle configuration changes
   */
  private handleConfigChange(): void {
    const newConfig = this.computeCurrentConfig();

    // Check if configuration actually changed
    const configChanged =
      this.currentConfig.enabled !== newConfig.enabled ||
      this.currentConfig.level !== newConfig.level ||
      this.currentConfig.source !== newConfig.source;

    if (configChanged) {
      this.currentConfig = newConfig;

      // Update all registered loggers
      loggerRegistry.updateAllLoggers(newConfig);

      // Notify subscribers
      this.notifyListeners(newConfig);
    }
  }

  /**
   * Notify all listeners of configuration change
   */
  private notifyListeners(config: LoggerConfig): void {
    this.listeners.forEach(listener => {
      try {
        listener(config);
      } catch (error) {
        console.warn('Error in logger config listener:', error);
      }
    });
  }

  /**
   * Force a configuration update (useful for initialization)
   */
  forceUpdate(): void {
    this.handleConfigChange();
  }

  /**
   * Clean up subscriptions
   */
  cleanup(): void {
    this.debugStateUnsubscribe?.();
    this.consoleLoggingUnsubscribe?.();
    this.listeners.clear();
  }

  /**
   * Get configuration source information
   */
  getConfigInfo(): {
    current: LoggerConfig;
    sources: {
      runtime: { enabled: boolean | null; level: LogLevel | null };
      environment: { level: LogLevel | null };
      defaults: { enabled: boolean; level: LogLevel };
    };
  } {
    return {
      current: this.getCurrentConfig(),
      sources: {
        runtime: {
          enabled: clientDebugState.get('consoleLogging'),
          level: clientDebugState.get('logLevel') as LogLevel
        },
        environment: {
          level: this.getEnvironmentLogLevel()
        },
        defaults: {
          enabled: this.getDefaultEnabled(),
          level: this.getDefaultLogLevel()
        }
      }
    };
  }
}

// Export singleton instance
export const dynamicLoggerConfig = new DynamicLoggerConfig();