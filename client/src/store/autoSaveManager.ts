import { createLogger } from '../utils/loggerConfig';
import { toast } from '../features/design-system/components/Toast';
import { settingsApi } from '../api/settingsApi';

interface BatchOperation {
  path: string;
  value: any;
}

const logger = createLogger('AutoSaveManager');


export class AutoSaveManager {
  private pendingChanges: Map<string, any> = new Map();
  private saveDebounceTimer: NodeJS.Timeout | null = null;
  private isInitialized: boolean = false;
  private retryCount: Map<string, number> = new Map();
  private readonly MAX_RETRIES = 3;
  private readonly DEBOUNCE_DELAY = 500; 
  private readonly RETRY_DELAY = 1000; 

  
  private readonly CLIENT_ONLY_PATHS = [
    'auth.nostr.connected',
    'auth.nostr.publicKey',
  ];

  
  private isClientOnlyPath(path: string): boolean {
    return this.CLIENT_ONLY_PATHS.some(clientPath =>
      path === clientPath || path.startsWith(clientPath + '.')
    );
  }

  setInitialized(initialized: boolean) {
    this.isInitialized = initialized;
  }

  
  queueChange(path: string, value: any) {
    if (!this.isInitialized) return;

    
    if (this.isClientOnlyPath(path)) {
      logger.debug(`Skipping client-only path: ${path}`);
      return;
    }

    this.pendingChanges.set(path, value);
    this.resetRetryCount(path);
    this.scheduleFlush();
  }

  
  queueChanges(changes: Map<string, any>) {
    if (!this.isInitialized) return;

    changes.forEach((value, path) => {
      
      if (this.isClientOnlyPath(path)) {
        logger.debug(`Skipping client-only path: ${path}`);
        return;
      }

      this.pendingChanges.set(path, value);
      this.resetRetryCount(path);
    });
    this.scheduleFlush();
  }

  
  private scheduleFlush() {
    if (this.saveDebounceTimer) {
      clearTimeout(this.saveDebounceTimer);
    }
    
    this.saveDebounceTimer = setTimeout(() => {
      this.flushPendingChanges();
    }, this.DEBOUNCE_DELAY);
  }

  
  async forceFlush(): Promise<void> {
    if (this.saveDebounceTimer) {
      clearTimeout(this.saveDebounceTimer);
      this.saveDebounceTimer = null;
    }
    await this.flushPendingChanges();
  }

  
  private async flushPendingChanges(): Promise<void> {
    if (this.pendingChanges.size === 0) return;
    
    const updates: BatchOperation[] = Array.from(this.pendingChanges.entries())
      .map(([path, value]) => ({ path, value }));
    
    logger.debug('Auto-save: Attempting to flush changes', { count: updates.length, paths: updates.map(u => u.path) });
    
    try {
      await settingsApi.updateSettingsByPaths(updates);
      
      
      updates.forEach(({ path }) => {
        this.pendingChanges.delete(path);
        this.resetRetryCount(path);
      });
      
      logger.info('Auto-save: Successfully flushed pending changes', { count: updates.length });
    } catch (error) {
      logger.error('Auto-save: Failed to flush changes', { error, updatesCount: updates.length });
      
      
      await this.retryFailedChanges(updates, error);
    }
  }

  
  private async retryFailedChanges(failedUpdates: BatchOperation[], error: any): Promise<void> {
    let hasRetriableChanges = false;
    let hasMaxedOutChanges = false;
    
    for (const { path } of failedUpdates) {
      const currentRetries = this.retryCount.get(path) || 0;
      
      if (currentRetries < this.MAX_RETRIES) {
        this.retryCount.set(path, currentRetries + 1);
        hasRetriableChanges = true;
        
        
        const retryDelay = this.RETRY_DELAY * Math.pow(2, currentRetries);
        
        setTimeout(() => {
          if (this.pendingChanges.has(path)) {
            logger.info(`Auto-save: Retrying save for path ${path} (attempt ${currentRetries + 1}/${this.MAX_RETRIES})`);
            this.scheduleFlush();
          } else {
            logger.debug(`Auto-save: Path ${path} no longer pending, skipping retry`);
          }
        }, retryDelay);
      } else {
        
        hasMaxedOutChanges = true;
        logger.error(`Auto-save: Max retries exceeded for path ${path}`, { error, maxRetries: this.MAX_RETRIES });
        
        
        try {
          if (typeof toast === 'function') {
            (toast as any).error?.(`Failed to save setting: ${path.split('.').pop()}. Changes are queued for retry.`);
          } else if (toast && typeof (toast as any).error === 'function') {
            (toast as any).error(`Failed to save setting: ${path.split('.').pop()}. Changes are queued for retry.`);
          }
        } catch {
          // Silently ignore toast errors
        }
      }
    }
    
    
    if (hasRetriableChanges && hasMaxedOutChanges) {
      logger.warn(`Auto-save: Some changes will be retried, others have exceeded max retries`);
    } else if (hasRetriableChanges) {
      logger.info(`Auto-save: All failed changes scheduled for retry`);
    } else if (hasMaxedOutChanges) {
      logger.error(`Auto-save: All failed changes have exceeded max retries`);
    }
  }

  private resetRetryCount(path: string) {
    this.retryCount.delete(path);
  }

  
  hasPendingChanges(): boolean {
    return this.pendingChanges.size > 0;
  }

  
  getPendingCount(): number {
    return this.pendingChanges.size;
  }
}

// Export singleton instance
export const autoSaveManager = new AutoSaveManager();