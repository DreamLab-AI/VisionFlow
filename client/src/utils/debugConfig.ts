

import { debugControl, replaceGlobalConsole, DebugCategory } from './console';
import { createLogger } from './loggerConfig';

const logger = createLogger('DebugConfig');


export interface DebugConfig {
  enabled: boolean;
  categories: DebugCategory[];
  replaceConsole: boolean;
  preset?: 'minimal' | 'standard' | 'verbose' | 'off';
}


export function parseDebugConfig(): DebugConfig {
  
  const env = import.meta.env || {};
  const isDev = env.DEV || env.MODE === 'development';
  
  const config: DebugConfig = {
    enabled: env.VITE_DEBUG === 'true' || isDev,
    categories: [],
    replaceConsole: env.VITE_DEBUG_REPLACE_CONSOLE === 'true',
    preset: env.VITE_DEBUG_PRESET as any,
  };

  
  if (env.VITE_DEBUG_CATEGORIES) {
    const cats = env.VITE_DEBUG_CATEGORIES.split(',').map((c: string) => c.trim());
    config.categories = cats.filter(c => Object.values(DebugCategory).includes(c as any)) as DebugCategory[];
  }

  return config;
}


export function initializeDebugSystem(): void {
  const config = parseDebugConfig();
  
  logger.info('Initializing debug system', config);

  
  if (config.preset && debugControl.presets[config.preset]) {
    debugControl.presets[config.preset]();
    logger.info(`Applied debug preset: ${config.preset}`);
  } else if (config.enabled) {
    
    debugControl.enable();
    
    
    if (config.categories.length > 0) {
      debugControl.disableAllCategories();
      config.categories.forEach(cat => {
        debugControl.enableCategory(cat);
      });
      logger.info('Enabled debug categories:', config.categories);
    }
  }

  
  if (config.replaceConsole && process.env.NODE_ENV === 'development') {
    replaceGlobalConsole(true);
  }

  
  logger.info('Debug system initialized', {
    enabled: debugControl.isEnabled(),
    categories: debugControl.getEnabledCategories(),
  });
}

