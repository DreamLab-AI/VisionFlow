

import { createLogger } from './baseLogger';

const logger = createLogger('ClientDebugState');

// localStorage keys for debug settings
const DEBUG_KEYS = {
  
  enabled: 'debug.enabled',
  dataDebug: 'debug.data',
  performanceDebug: 'debug.performance',
  
  
  consoleLogging: 'debug.consoleLogging',
  logLevel: 'debug.logLevel',
  showNodeIds: 'debug.showNodeIds',
  showEdgeWeights: 'debug.showEdgeWeights',
  enableProfiler: 'debug.enableProfiler',
  apiDebugMode: 'debug.apiDebugMode',
  
  
  enableWebsocketDebug: 'debug.enableWebsocketDebug',
  logBinaryHeaders: 'debug.logBinaryHeaders',
  logFullJson: 'debug.logFullJson',
  enablePhysicsDebug: 'debug.enablePhysicsDebug',
  enableNodeDebug: 'debug.enableNodeDebug',
  enableShaderDebug: 'debug.enableShaderDebug',
  enableMatrixDebug: 'debug.enableMatrixDebug',
  enablePerformanceDebug: 'debug.enablePerformanceDebug',
} as const;

export type DebugKey = keyof typeof DEBUG_KEYS;

class ClientDebugState {
  private listeners: Map<string, Set<(value: string | boolean) => void>> = new Map();

  constructor() {
    
    if (typeof window !== 'undefined') {
      window.addEventListener('storage', this.handleStorageChange.bind(this));
    }
  }

  private handleStorageChange(e: StorageEvent): void {
    if (e.key && (Object.values(DEBUG_KEYS) as readonly string[]).includes(e.key)) {
      const newValue: string | boolean = e.newValue === 'true' ? true :
                       e.newValue === 'false' ? false :
                       e.newValue ?? '';
      this.notifyListeners(e.key, newValue);
    }
  }

  private notifyListeners(key: string, value: string | boolean): void {
    const listeners = this.listeners.get(key);
    if (listeners) {
      listeners.forEach(listener => listener(value));
    }
  }

  
  public get(key: DebugKey): string | boolean {
    const storageKey = DEBUG_KEYS[key];
    if (typeof window === 'undefined') return false;
    
    try {
      const value = localStorage.getItem(storageKey);
      if (value === null) return this.getDefault(key);
      
      
      if (value === 'true') return true;
      if (value === 'false') return false;
      
      
      return value;
    } catch (e) {
      logger.warn(`Failed to read ${storageKey} from localStorage`);
      return this.getDefault(key);
    }
  }

  
  public set(key: DebugKey, value: string | boolean): void {
    const storageKey = DEBUG_KEYS[key];
    if (typeof window === 'undefined') return;
    
    try {
      const stringValue = String(value);
      localStorage.setItem(storageKey, stringValue);
      this.notifyListeners(storageKey, value);
      logger.info(`Debug setting ${key} set to ${stringValue}`);
    } catch (e) {
      logger.warn(`Failed to save ${storageKey} to localStorage`);
    }
  }

  
  public subscribe(key: DebugKey, listener: (value: string | boolean) => void): () => void {
    const storageKey = DEBUG_KEYS[key];
    if (!this.listeners.has(storageKey)) {
      this.listeners.set(storageKey, new Set());
    }
    this.listeners.get(storageKey)!.add(listener);
    
    
    return () => {
      const listeners = this.listeners.get(storageKey);
      if (listeners) {
        listeners.delete(listener);
      }
    };
  }

  
  private getDefault(key: DebugKey): string | boolean {
    switch (key) {
      case 'logLevel':
        return 'info';
      default:
        return false;
    }
  }

  
  public isEnabled(): boolean {
    return this.get('enabled') === true;
  }

  public setEnabled(value: boolean): void {
    this.set('enabled', value);
  }

  public isDataDebugEnabled(): boolean {
    return this.isEnabled() && this.get('dataDebug') === true;
  }

  public isPerformanceDebugEnabled(): boolean {
    return this.isEnabled() && this.get('performanceDebug') === true;
  }

  
  public getAll(): Record<DebugKey, string | boolean> {
    const result: Partial<Record<DebugKey, string | boolean>> = {};
    for (const key of Object.keys(DEBUG_KEYS) as DebugKey[]) {
      result[key] = this.get(key);
    }
    return result as Record<DebugKey, string | boolean>;
  }

  public reset(): void {
    if (typeof window === 'undefined') return;
    
    try {
      for (const storageKey of Object.values(DEBUG_KEYS)) {
        localStorage.removeItem(storageKey);
        this.notifyListeners(storageKey, this.getDefault(Object.entries(DEBUG_KEYS).find(([, v]) => v === storageKey)?.[0] as DebugKey || 'enabled'));
      }
      logger.info('All debug settings reset to defaults');
    } catch (e) {
      logger.warn('Failed to reset debug settings');
    }
  }
}

// Create singleton instance
export const clientDebugState = new ClientDebugState();

// For backward compatibility with existing code
export const debugState = {
  isEnabled: () => clientDebugState.isEnabled(),
  enableDebug: (value: boolean) => clientDebugState.setEnabled(value),
  isDataDebugEnabled: () => clientDebugState.isDataDebugEnabled(),
  enableDataDebug: (value: boolean) => clientDebugState.set('dataDebug', value),
  isPerformanceDebugEnabled: () => clientDebugState.isPerformanceDebugEnabled(),
  enablePerformanceDebug: (value: boolean) => clientDebugState.set('performanceDebug', value),
};