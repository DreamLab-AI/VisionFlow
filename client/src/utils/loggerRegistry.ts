/**
 * Logger Registry - Tracks and manages all active logger instances
 * Enables bulk updates when configuration changes
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LoggerInstance {
  namespace: string;
  logger: any;
  updateLevel: (level: LogLevel) => void;
  setEnabled: (enabled: boolean) => void;
  isEnabled: () => boolean;
  getCurrentConfig: () => { level: LogLevel; enabled: boolean };
}

export interface LoggerConfig {
  enabled: boolean;
  level: LogLevel;
  source: 'runtime' | 'environment' | 'default';
}

class LoggerRegistry {
  private loggers: Map<string, LoggerInstance> = new Map();
  private configSubscription?: () => void;

  /**
   * Register a logger instance for dynamic configuration
   */
  register(instance: LoggerInstance): void {
    this.loggers.set(instance.namespace, instance);
  }

  /**
   * Unregister a logger instance
   */
  unregister(namespace: string): void {
    this.loggers.delete(namespace);
  }

  /**
   * Update all registered loggers with new configuration
   */
  updateAllLoggers(config: LoggerConfig): void {
    this.loggers.forEach((instance) => {
      try {
        // Update logging level
        instance.updateLevel(config.level);

        // Update enabled state
        instance.setEnabled(config.enabled);
      } catch (error) {
        console.warn(`Failed to update logger ${instance.namespace}:`, error);
      }
    });
  }

  /**
   * Get all active logger instances
   */
  getActiveLoggers(): LoggerInstance[] {
    return Array.from(this.loggers.values());
  }

  /**
   * Get logger by namespace
   */
  getLogger(namespace: string): LoggerInstance | undefined {
    return this.loggers.get(namespace);
  }

  /**
   * Get current registry statistics
   */
  getStats(): {
    total: number;
    enabled: number;
    byLevel: Record<LogLevel, number>
  } {
    const stats = {
      total: this.loggers.size,
      enabled: 0,
      byLevel: { debug: 0, info: 0, warn: 0, error: 0 } as Record<LogLevel, number>
    };

    this.loggers.forEach((instance) => {
      if (instance.isEnabled()) {
        stats.enabled++;
        const config = instance.getCurrentConfig();
        stats.byLevel[config.level]++;
      }
    });

    return stats;
  }

  /**
   * Clear all registered loggers
   */
  clear(): void {
    this.loggers.clear();
  }
}

// Export singleton instance
export const loggerRegistry = new LoggerRegistry();