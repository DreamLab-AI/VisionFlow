/**
 * Logger Integration Initialization
 * Provides centralized initialization for the dynamic logger system
 */

import { loggerDebugBridge, initializeLoggerIntegration, getLoggerIntegrationStatus } from './loggerDebugBridge';
import { dynamicLoggerConfig } from './dynamicLoggerConfig';
import { loggerRegistry } from './loggerRegistry';

interface LoggerIntegrationOptions {
  enableBridgeDebug?: boolean;
  autoStart?: boolean;
  logInitialization?: boolean;
}

class LoggerIntegrationInitializer {
  private isInitialized = false;
  private options: Required<LoggerIntegrationOptions> = {
    enableBridgeDebug: false,
    autoStart: true,
    logInitialization: true
  };

  /**
   * Initialize the complete dynamic logger integration system
   */
  initialize(options: LoggerIntegrationOptions = {}): void {
    if (this.isInitialized) {
      if (this.options.logInitialization) {
        console.warn('Logger integration system is already initialized');
      }
      return;
    }

    // Merge options with defaults
    this.options = { ...this.options, ...options };

    if (this.options.logInitialization) {
      console.info('Initializing dynamic logger integration system...');
    }

    try {
      // 1. Initialize the debug bridge
      initializeLoggerIntegration();

      // 2. Enable bridge debugging if requested
      if (this.options.enableBridgeDebug) {
        loggerDebugBridge.enableBridgeDebug();
      }

      // 3. Force initial synchronization
      this.performInitialSynchronization();

      this.isInitialized = true;

      if (this.options.logInitialization) {
        const status = getLoggerIntegrationStatus();
        console.info('Dynamic logger integration initialized successfully', status);
      }

    } catch (error) {
      console.error('Failed to initialize logger integration system:', error);
      throw error;
    }
  }

  /**
   * Perform initial synchronization of all settings
   */
  private performInitialSynchronization(): void {
    // Force configuration update to ensure everything is in sync
    dynamicLoggerConfig.forceUpdate();

    // Log initial state if debugging is enabled
    if (this.options.logInitialization) {
      const configInfo = dynamicLoggerConfig.getConfigInfo();
      const registryStats = loggerRegistry.getStats();

      console.info('Initial logger synchronization complete', {
        configuration: configInfo,
        registry: registryStats,
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Check if the integration system is initialized
   */
  isSystemInitialized(): boolean {
    return this.isInitialized;
  }

  /**
   * Get comprehensive status of the integration system
   */
  getSystemStatus() {
    return {
      initialized: this.isInitialized,
      options: this.options,
      bridge: getLoggerIntegrationStatus(),
      configuration: dynamicLoggerConfig.getConfigInfo(),
      registry: loggerRegistry.getStats(),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Refresh the entire integration system
   */
  refresh(): void {
    if (!this.isInitialized) {
      throw new Error('Logger integration system not initialized. Call initialize() first.');
    }

    loggerDebugBridge.refresh();
    this.performInitialSynchronization();

    if (this.options.logInitialization) {
      console.info('Logger integration system refreshed');
    }
  }

  /**
   * Clean up the integration system
   */
  cleanup(): void {
    if (!this.isInitialized) {
      return;
    }

    try {
      loggerDebugBridge.cleanup();
      loggerRegistry.clear();
      this.isInitialized = false;

      if (this.options.logInitialization) {
        console.info('Logger integration system cleaned up');
      }
    } catch (error) {
      console.error('Error during logger integration cleanup:', error);
    }
  }

  /**
   * Enable or disable bridge debugging at runtime
   */
  setBridgeDebug(enabled: boolean): void {
    this.options.enableBridgeDebug = enabled;

    if (enabled && this.isInitialized) {
      loggerDebugBridge.enableBridgeDebug();
    }
  }
}

// Export singleton instance
const loggerIntegrationInitializer = new LoggerIntegrationInitializer();

/**
 * Initialize the dynamic logger integration system
 * Call this once during application startup
 */
export function initializeDynamicLoggers(options?: LoggerIntegrationOptions): void {
  loggerIntegrationInitializer.initialize(options);
}

/**
 * Get the status of the logger integration system
 */
export function getLoggerSystemStatus() {
  return loggerIntegrationInitializer.getSystemStatus();
}

/**
 * Refresh the logger integration system
 */
export function refreshLoggerSystem(): void {
  loggerIntegrationInitializer.refresh();
}

/**
 * Check if the logger integration system is initialized
 */
export function isLoggerSystemInitialized(): boolean {
  return loggerIntegrationInitializer.isSystemInitialized();
}

/**
 * Clean up the logger integration system
 */
export function cleanupLoggerSystem(): void {
  loggerIntegrationInitializer.cleanup();
}

/**
 * Auto-initialize if running in browser and not explicitly disabled
 */
if (typeof window !== 'undefined') {
  // Auto-initialize on next tick to allow other modules to load first
  setTimeout(() => {
    if (!loggerIntegrationInitializer.isSystemInitialized()) {
      initializeDynamicLoggers({
        enableBridgeDebug: import.meta.env?.DEV,
        logInitialization: import.meta.env?.DEV
      });
    }
  }, 0);
}

// Export the initializer instance for advanced usage
export { loggerIntegrationInitializer };