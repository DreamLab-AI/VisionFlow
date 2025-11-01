

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
  private listeners: Map<string, Set<(value: any) => void>> = new Map();

  constructor() {
    
    if (typeof window !== 'undefined') {
      window.addEventListener('storage', this.handleStorageChange.bind(this));
    }
  }

  private handleStorageChange(e: StorageEvent): void {
    if (e.key && Object.values(DEBUG_KEYS).includes(e.key as any)) {
      const newValue = e.newValue === 'true' ? true : 
                       e.newValue === 'false' ? false : 
                       e.newValue;
      this.notifyListeners(e.key, newValue);
    }
  }

  private notifyListeners(key: string, value: any): void {
    const listeners = this.listeners.get(key);
    if (listeners) {
      listeners.forEach(listener => listener(value));
    }
  }

  
  public get(key: DebugKey): any {
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

  
  public set(key: DebugKey, value: any): void {
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

  
  public subscribe(key: DebugKey, listener: (value: any) => void): () => void {
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

  
  private getDefault(key: DebugKey): any {
    switch (key) {
      case 'logLevel':
        return 'info';
      default:
        return false;
    }
  }

  
  public isEnabled(): boolean {
    return this.get('enabled');
  }

  public setEnabled(value: boolean): void {
    this.set('enabled', value);
  }

  public isDataDebugEnabled(): boolean {
    return this.isEnabled() && this.get('dataDebug');
  }

  public isPerformanceDebugEnabled(): boolean {
    return this.isEnabled() && this.get('performanceDebug');
  }

  
  public getAll(): Record<DebugKey, any> {
    const result: Partial<Record<DebugKey, any>> = {};
    for (const key of Object.keys(DEBUG_KEYS) as DebugKey[]) {
      result[key] = this.get(key);
    }
    return result as Record<DebugKey, any>;
  }

  public reset(): void {
    if (typeof window === 'undefined') return;
    
    try {
      for (const storageKey of Object.values(DEBUG_KEYS)) {
        localStorage.removeItem(storageKey);
        this.notifyListeners(storageKey, this.getDefault(storageKey as any));
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