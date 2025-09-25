import { createLogger } from './loggerConfig';

// Debug categories enum
export enum DebugCategory {
  GENERAL = 'general',
  VOICE = 'voice',
  WEBSOCKET = 'websocket',
  PERFORMANCE = 'performance',
  DATA = 'data',
  RENDERING = 'rendering',
  AUTH = 'auth',
  ERROR = 'error'
}

// Logger instance for debug control
const logger = createLogger('DebugControl');

class DebugControl {
  private enabled: boolean = false;
  private enabledCategories: Set<DebugCategory> = new Set();
  private dataDebugEnabled: boolean = false;
  private performanceDebugEnabled: boolean = false;

  // Enable main debug mode
  enable() {
    this.enabled = true;
    logger.info('Debug mode enabled');
  }

  // Disable main debug mode
  disable() {
    this.enabled = false;
    this.enabledCategories.clear();
    this.dataDebugEnabled = false;
    this.performanceDebugEnabled = false;
    logger.info('Debug mode disabled');
  }

  // Check if debug is enabled
  isEnabled(): boolean {
    return this.enabled;
  }

  // Enable a specific category
  enableCategory(category: DebugCategory) {
    this.enabledCategories.add(category);
    logger.debug(`Enabled debug category: ${category}`);
  }

  // Disable a specific category
  disableCategory(category: DebugCategory) {
    this.enabledCategories.delete(category);
    logger.debug(`Disabled debug category: ${category}`);
  }

  // Check if a category is enabled
  isCategoryEnabled(category: DebugCategory): boolean {
    return this.enabledCategories.has(category);
  }

  // Get all enabled categories
  getEnabledCategories(): DebugCategory[] {
    return Array.from(this.enabledCategories);
  }

  // Enable data debugging
  enableData() {
    this.dataDebugEnabled = true;
    logger.debug('Data debugging enabled');
  }

  // Disable data debugging
  disableData() {
    this.dataDebugEnabled = false;
    logger.debug('Data debugging disabled');
  }

  // Check if data debugging is enabled
  isDataEnabled(): boolean {
    return this.dataDebugEnabled;
  }

  // Enable performance debugging
  enablePerformance() {
    this.performanceDebugEnabled = true;
    logger.debug('Performance debugging enabled');
  }

  // Disable performance debugging
  disablePerformance() {
    this.performanceDebugEnabled = false;
    logger.debug('Performance debugging disabled');
  }

  // Check if performance debugging is enabled
  isPerformanceEnabled(): boolean {
    return this.performanceDebugEnabled;
  }

  // Debug presets for quick configuration
  presets = {
    // Enable all debugging
    all: () => {
      this.enable();
      Object.values(DebugCategory).forEach(cat => this.enableCategory(cat));
      this.enableData();
      this.enablePerformance();
      logger.info('All debug features enabled');
    },

    // Enable only essential debugging
    minimal: () => {
      this.enable();
      this.enableCategory(DebugCategory.GENERAL);
      this.enableCategory(DebugCategory.ERROR);
      this.disableData();
      this.disablePerformance();
      logger.info('Minimal debug features enabled');
    },

    // Enable performance analysis
    performance: () => {
      this.enable();
      this.enableCategory(DebugCategory.PERFORMANCE);
      this.enableCategory(DebugCategory.RENDERING);
      this.enablePerformance();
      logger.info('Performance debug features enabled');
    },

    // Enable network debugging
    network: () => {
      this.enable();
      this.enableCategory(DebugCategory.WEBSOCKET);
      this.enableCategory(DebugCategory.DATA);
      this.enableData();
      logger.info('Network debug features enabled');
    },

    // Disable all debugging
    none: () => {
      this.disable();
      logger.info('All debug features disabled');
    }
  };

  // Log a debug message for a specific category
  log(category: DebugCategory, message: string, data?: any) {
    if (!this.enabled || !this.isCategoryEnabled(category)) {
      return;
    }

    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      category,
      message,
      data
    };

    // Use the logger to output based on category
    switch (category) {
      case DebugCategory.ERROR:
        logger.error(message, data);
        break;
      case DebugCategory.PERFORMANCE:
        if (this.performanceDebugEnabled) {
          logger.info(`[PERF] ${message}`, data);
        }
        break;
      case DebugCategory.DATA:
        if (this.dataDebugEnabled) {
          logger.debug(`[DATA] ${message}`, data);
        }
        break;
      default:
        logger.debug(`[${category.toUpperCase()}] ${message}`, data);
    }
  }
}

// Create singleton instance
export const debugControl = new DebugControl();

// Convenience logging functions
export const debugLog = (category: DebugCategory, message: string, data?: any) => {
  debugControl.log(category, message, data);
};

export const debugError = (message: string, error?: any) => {
  debugControl.log(DebugCategory.ERROR, message, error);
};

export const debugPerformance = (message: string, metrics?: any) => {
  debugControl.log(DebugCategory.PERFORMANCE, message, metrics);
};

export const debugData = (message: string, data?: any) => {
  debugControl.log(DebugCategory.DATA, message, data);
};

// Create gatedConsole for category-based logging
export const gatedConsole = {
  voice: {
    log: (...args: any[]) => {
      if (debugControl.isCategoryEnabled(DebugCategory.VOICE)) {
        console.log('[VOICE]', ...args);
      }
    },
    error: (...args: any[]) => {
      if (debugControl.isCategoryEnabled(DebugCategory.VOICE) || debugControl.isCategoryEnabled(DebugCategory.ERROR)) {
        console.error('[VOICE ERROR]', ...args);
      }
    },
    warn: (...args: any[]) => {
      if (debugControl.isCategoryEnabled(DebugCategory.VOICE)) {
        console.warn('[VOICE]', ...args);
      }
    },
    debug: (...args: any[]) => {
      if (debugControl.isCategoryEnabled(DebugCategory.VOICE)) {
        console.debug('[VOICE]', ...args);
      }
    }
  },
  websocket: {
    log: (...args: any[]) => {
      if (debugControl.isCategoryEnabled(DebugCategory.WEBSOCKET)) {
        console.log('[WEBSOCKET]', ...args);
      }
    },
    error: (...args: any[]) => {
      if (debugControl.isCategoryEnabled(DebugCategory.WEBSOCKET) || debugControl.isCategoryEnabled(DebugCategory.ERROR)) {
        console.error('[WEBSOCKET ERROR]', ...args);
      }
    }
  },
  performance: {
    log: (...args: any[]) => {
      if (debugControl.isCategoryEnabled(DebugCategory.PERFORMANCE)) {
        console.log('[PERFORMANCE]', ...args);
      }
    }
  },
  data: {
    log: (...args: any[]) => {
      if (debugControl.isCategoryEnabled(DebugCategory.DATA)) {
        console.log('[DATA]', ...args);
      }
    }
  },
  error: (...args: any[]) => {
    if (debugControl.isCategoryEnabled(DebugCategory.ERROR)) {
      console.error('[ERROR]', ...args);
    }
  },
  log: (...args: any[]) => {
    if (debugControl.isCategoryEnabled(DebugCategory.GENERAL)) {
      console.log('[DEBUG]', ...args);
    }
  }
};

// Export for compatibility with existing code
export default debugControl;