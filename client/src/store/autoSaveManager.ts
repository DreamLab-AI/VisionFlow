import { createLogger } from '../utils/loggerConfig';
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

  // Client-only paths that should never sync to server
  private readonly CLIENT_ONLY_PATHS = [
    'auth.nostr.connected',
    'auth.nostr.publicKey',
  ];

  // Check if a path is client-only (shouldn't sync to server)
  private isClientOnlyPath(path: string): boolean {
    return this.CLIENT_ONLY_PATHS.some(clientPath =>
      path === clientPath || path.startsWith(clientPath + '.')
    );
  }

  setInitialized(initialized: boolean) {
    this.isInitialized = initialized;
  }

  // Queue a change for auto-save
  queueChange(path: string, value: any) {
    if (!this.isInitialized) return;

    // Skip client-only paths that shouldn't sync to server
    if (this.isClientOnlyPath(path)) {
      logger.debug(`Skipping client-only path: ${path}`);
      return;
    }

    this.pendingChanges.set(path, value);
    this.resetRetryCount(path);
    this.scheduleFlush();
  }

  // Queue multiple changes
  queueChanges(changes: Map<string, any>) {
    if (!this.isInitialized) return;

    changes.forEach((value, path) => {
      // Skip client-only paths that shouldn't sync to server
      if (this.isClientOnlyPath(path)) {
        logger.debug(`Skipping client-only path: ${path}`);
        return;
      }

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
    
    logger.debug('Auto-save: Attempting to flush changes', { count: updates.length, paths: updates.map(u => u.path) });
    
    try {
      await settingsApi.updateSettingsByPaths(updates);
      
      // Clear successfully saved changes
      updates.forEach(({ path }) => {
        this.pendingChanges.delete(path);
        this.resetRetryCount(path);
      });
      
      logger.info('Auto-save: Successfully flushed pending changes', { count: updates.length });
    } catch (error) {
      logger.error('Auto-save: Failed to flush changes', { error, updatesCount: updates.length });
      
      // Implement retry logic for failed changes
      await this.retryFailedChanges(updates, error);
    }
  }

  // Retry failed changes with exponential backoff
  private async retryFailedChanges(failedUpdates: BatchOperation[], error: any): Promise<void> {
    let hasRetriableChanges = false;
    let hasMaxedOutChanges = false;
    
    for (const { path } of failedUpdates) {
      const currentRetries = this.retryCount.get(path) || 0;
      
      if (currentRetries < this.MAX_RETRIES) {
        this.retryCount.set(path, currentRetries + 1);
        hasRetriableChanges = true;
        
        // Schedule retry with exponential backoff
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
        // Max retries exceeded, log error but keep change in pending for manual flush
        hasMaxedOutChanges = true;
        logger.error(`Auto-save: Max retries exceeded for path ${path}`, { error, maxRetries: this.MAX_RETRIES });
        
        // Show user-friendly notification for important failures
        if (toast?.error) {
          toast.error(`Failed to save setting: ${path.split('.').pop()}. Changes are queued for retry.`);
        }
      }
    }
    
    // Log summary of retry status
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

  // Check if there are pending changes
  hasPendingChanges(): boolean {
    return this.pendingChanges.size > 0;
  }

  // Get pending changes count (for UI feedback)
  getPendingCount(): number {
    return this.pendingChanges.size;
  }
}