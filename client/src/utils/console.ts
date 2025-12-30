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

  
  enable() {
    this.enabled = true;
    logger.info('Debug mode enabled');
  }

  
  disable() {
    this.enabled = false;
    this.enabledCategories.clear();
    this.dataDebugEnabled = false;
    this.performanceDebugEnabled = false;
    logger.info('Debug mode disabled');
  }

  
  isEnabled(): boolean {
    return this.enabled;
  }

  
  enableCategory(category: DebugCategory) {
    this.enabledCategories.add(category);
    logger.debug(`Enabled debug category: ${category}`);
  }

  
  disableCategory(category: DebugCategory) {
    this.enabledCategories.delete(category);
    logger.debug(`Disabled debug category: ${category}`);
  }

  
  isCategoryEnabled(category: DebugCategory): boolean {
    return this.enabledCategories.has(category);
  }

  
  getEnabledCategories(): DebugCategory[] {
    return Array.from(this.enabledCategories);
  }

  
  enableData() {
    this.dataDebugEnabled = true;
    logger.debug('Data debugging enabled');
  }

  
  disableData() {
    this.dataDebugEnabled = false;
    logger.debug('Data debugging disabled');
  }

  
  isDataEnabled(): boolean {
    return this.dataDebugEnabled;
  }

  
  enablePerformance() {
    this.performanceDebugEnabled = true;
    logger.debug('Performance debugging enabled');
  }

  
  disablePerformance() {
    this.performanceDebugEnabled = false;
    logger.debug('Performance debugging disabled');
  }

  
  isPerformanceEnabled(): boolean {
    return this.performanceDebugEnabled;
  }

  
  presets = {
    
    all: () => {
      this.enable();
      Object.values(DebugCategory).forEach(cat => this.enableCategory(cat));
      this.enableData();
      this.enablePerformance();
      logger.info('All debug features enabled');
    },

    
    minimal: () => {
      this.enable();
      this.enableCategory(DebugCategory.GENERAL);
      this.enableCategory(DebugCategory.ERROR);
      this.disableData();
      this.disablePerformance();
      logger.info('Minimal debug features enabled');
    },

    
    performance: () => {
      this.enable();
      this.enableCategory(DebugCategory.PERFORMANCE);
      this.enableCategory(DebugCategory.RENDERING);
      this.enablePerformance();
      logger.info('Performance debug features enabled');
    },

    
    network: () => {
      this.enable();
      this.enableCategory(DebugCategory.WEBSOCKET);
      this.enableCategory(DebugCategory.DATA);
      this.enableData();
      logger.info('Network debug features enabled');
    },

    
    none: () => {
      this.disable();
      logger.info('All debug features disabled');
    },

    // Alias presets for UI consistency
    off: () => {
      this.disable();
      logger.info('All debug features disabled');
    },

    standard: () => {
      this.enable();
      this.enableCategory(DebugCategory.GENERAL);
      this.enableCategory(DebugCategory.ERROR);
      this.enableCategory(DebugCategory.DATA);
      this.disableData();
      this.disablePerformance();
      logger.info('Standard debug features enabled');
    },

    verbose: () => {
      this.enable();
      Object.values(DebugCategory).forEach(cat => this.enableCategory(cat));
      this.enableData();
      this.enablePerformance();
      logger.info('Verbose debug features enabled');
    }
  };

  
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

// Function to replace global console with gated console
export const replaceGlobalConsole = () => {
  
  const originalConsole = {
    log: console.log,
    error: console.error,
    warn: console.warn,
    debug: console.debug,
    info: console.info
  };

  
  console.log = (...args: any[]) => {
    if (debugControl.isEnabled()) {
      originalConsole.log(...args);
    }
  };

  console.error = (...args: any[]) => {
    
    originalConsole.error(...args);
  };

  console.warn = (...args: any[]) => {
    if (debugControl.isEnabled()) {
      originalConsole.warn(...args);
    }
  };

  console.debug = (...args: any[]) => {
    if (debugControl.isEnabled()) {
      originalConsole.debug(...args);
    }
  };

  console.info = (...args: any[]) => {
    if (debugControl.isEnabled()) {
      originalConsole.info(...args);
    }
  };

  logger.info('Global console replaced with gated console');
};

// Export for compatibility with existing code
export default debugControl;