import { createLogger } from '../utils/logger';
import { toast } from '../features/design-system/components/Toast';
import { settingsApi, BatchOperation } from '../api/settingsApi';

const logger = createLogger('AutoSaveManager');

/**
 * AutoSaveManager - Provides debounced batch saving with retry logic
 * 
 * Key features:
 * - Debounced batching (500ms delay) to reduce server load
 * - Retry logic with exponential backoff (max 3 retries)
 * - Error recovery and user notifications
 * - Pending changes tracking for UI feedback
 * - Force flush capability for manual saves
 */
export class AutoSaveManager {
  private pendingChanges: Map<string, any> = new Map();
  private saveDebounceTimer: NodeJS.Timeout | null = null;
  private isInitialized: boolean = false;
  private retryCount: Map<string, number> = new Map();
  private readonly MAX_RETRIES = 3;
  private readonly DEBOUNCE_DELAY = 500; // 500ms debounce
  private readonly RETRY_DELAY = 1000; // 1s retry delay

  setInitialized(initialized: boolean) {
    this.isInitialized = initialized;
  }

  // Queue a change for auto-save
  queueChange(path: string, value: any) {
    if (!this.isInitialized) return;
    
    this.pendingChanges.set(path, value);
    this.resetRetryCount(path);
    this.scheduleFlush();
  }

  // Queue multiple changes
  queueChanges(changes: Map<string, any>) {
    if (!this.isInitialized) return;
    
    changes.forEach((value, path) => {
      this.pendingChanges.set(path, value);
      this.resetRetryCount(path);
    });
    this.scheduleFlush();
  }

  // Schedule a debounced flush
  private scheduleFlush() {
    if (this.saveDebounceTimer) {
      clearTimeout(this.saveDebounceTimer);
    }
    
    this.saveDebounceTimer = setTimeout(() => {
      this.flushPendingChanges();
    }, this.DEBOUNCE_DELAY);
  }

  // Force immediate flush (for manual save)
  async forceFlush(): Promise<void> {
    if (this.saveDebounceTimer) {
      clearTimeout(this.saveDebounceTimer);
      this.saveDebounceTimer = null;
    }
    await this.flushPendingChanges();
  }

  // Flush all pending changes to server
  private async flushPendingChanges(): Promise<void> {
    if (this.pendingChanges.size === 0) return;
    
    const updates: BatchOperation[] = Array.from(this.pendingChanges.entries())
      .map(([path, value]) => ({ path, value }));
    
    try {
      await settingsApi.updateSettingsByPaths(updates);
      
      // Clear successfully saved changes
      updates.forEach(({ path }) => {
        this.pendingChanges.delete(path);
        this.resetRetryCount(path);
      });
      
      logger.debug('Auto-save: Flushed pending changes', { count: updates.length });
    } catch (error) {
      logger.error('Auto-save: Failed to flush changes', { error, updatesCount: updates.length });
      
      // Implement retry logic for failed changes
      await this.retryFailedChanges(updates, error);
    }
  }

  // Retry failed changes with exponential backoff
  private async retryFailedChanges(failedUpdates: BatchOperation[], error: any): Promise<void> {
    for (const { path } of failedUpdates) {
      const currentRetries = this.retryCount.get(path) || 0;
      
      if (currentRetries < this.MAX_RETRIES) {
        this.retryCount.set(path, currentRetries + 1);
        
        // Schedule retry with exponential backoff
        const retryDelay = this.RETRY_DELAY * Math.pow(2, currentRetries);
        
        setTimeout(() => {
          if (this.pendingChanges.has(path)) {
            logger.info(`Auto-save: Retrying save for path ${path} (attempt ${currentRetries + 1})`);
            this.scheduleFlush();
          }
        }, retryDelay);
      } else {
        // Max retries exceeded, log error but keep change in pending
        logger.error(`Auto-save: Max retries exceeded for path ${path}`, { error });
        toast?.error?.(`Failed to save setting: ${path}`);
      }
    }
  }

  private resetRetryCount(path: string) {
    this.retryCount.delete(path);
  }

  // Check if there are pending changes
  hasPendingChanges(): boolean {
    return this.pendingChanges.size > 0;
  }

  // Get pending changes count (for UI feedback)
  getPendingCount(): number {
    return this.pendingChanges.size;
  }
}