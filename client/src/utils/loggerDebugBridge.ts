/**
 * Logger Debug Bridge - Connects clientDebugState to the logger system
 * Initializes and manages the integration between Control Center settings and loggers
 */

import { clientDebugState } from './clientDebugState';
import { dynamicLoggerConfig } from './dynamicLoggerConfig';
import { loggerRegistry } from './loggerRegistry';

class LoggerDebugBridge {
  private isInitialized = false;
  private debugStateSubscriptions: (() => void)[] = [];
  private configSubscription?: () => void;

  /**
   * Initialize the bridge between debug settings and logger system
   */
  initialize(): void {
    if (this.isInitialized) {
      console.warn('Logger Debug Bridge is already initialized');
      return;
    }

    console.info('Initializing Logger Debug Bridge...');

    // Subscribe to configuration changes from dynamic config
    this.configSubscription = dynamicLoggerConfig.subscribe((config) => {
      this.handleConfigurationChange(config);
    });

    // Force initial synchronization
    this.synchronizeSettings();

    this.isInitialized = true;
    console.info('Logger Debug Bridge initialized successfully');
  }

  /**
   * Synchronize current debug settings with logger system
   */
  synchronizeSettings(): void {
    // Force update of configuration
    dynamicLoggerConfig.forceUpdate();

    // Log current state for debugging
    const configInfo = dynamicLoggerConfig.getConfigInfo();
    const registryStats = loggerRegistry.getStats();

    console.info('Logger settings synchronized', {
      config: configInfo,
      activeLoggers: registryStats
    });
  }

  /**
   * Handle configuration changes from the dynamic config manager
   */
  private handleConfigurationChange(config: any): void {
    const registryStats = loggerRegistry.getStats();

    console.info('Logger configuration changed', {
      newConfig: config,
      affectedLoggers: registryStats.total
    });
  }

  /**
   * Get current integration status
   */
  getStatus(): {
    initialized: boolean;
    configSource: string;
    activeLoggers: number;
    enabledLoggers: number;
    currentLevel: string;
    debugSettings: {
      consoleLogging: any;
      logLevel: any;
      debugEnabled: any;
    };
  } {
    const config = dynamicLoggerConfig.getCurrentConfig();
    const stats = loggerRegistry.getStats();

    return {
      initialized: this.isInitialized,
      configSource: config.source,
      activeLoggers: stats.total,
      enabledLoggers: stats.enabled,
      currentLevel: config.level,
      debugSettings: {
        consoleLogging: clientDebugState.get('consoleLogging'),
        logLevel: clientDebugState.get('logLevel'),
        debugEnabled: clientDebugState.get('enabled')
      }
    };
  }

  /**
   * Manually trigger a settings update (useful for testing)
   */
  refresh(): void {
    this.synchronizeSettings();
  }

  /**
   * Clean up all subscriptions and reset state
   */
  cleanup(): void {
    // Clean up dynamic config subscription
    this.configSubscription?.();

    // Clean up debug state subscriptions
    this.debugStateSubscriptions.forEach(unsubscribe => unsubscribe());
    this.debugStateSubscriptions = [];

    // Clean up dynamic config
    dynamicLoggerConfig.cleanup();

    this.isInitialized = false;
    console.info('Logger Debug Bridge cleaned up');
  }

  /**
   * Enable debug mode for the bridge itself
   */
  enableBridgeDebug(): void {
    // Add detailed logging for bridge operations
    const originalConfigHandler = this.handleConfigurationChange.bind(this);
    this.handleConfigurationChange = (config: any) => {
      console.debug('[LoggerDebugBridge] Configuration changing:', {
        old: dynamicLoggerConfig.getCurrentConfig(),
        new: config,
        timestamp: new Date().toISOString()
      });
      originalConfigHandler(config);
    };
  }
}

// Export singleton instance
export const loggerDebugBridge = new LoggerDebugBridge();

/**
 * Initialize the logger debug bridge (should be called once during app startup)
 */
export function initializeLoggerIntegration(): void {
  loggerDebugBridge.initialize();
}

/**
 * Get the current status of logger integration
 */
export function getLoggerIntegrationStatus() {
  return loggerDebugBridge.getStatus();
}