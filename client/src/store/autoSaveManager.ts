import { createLogger } from '../utils/loggerConfig';
import { settingsApi } from '../api/settingsApi';
import { settingsRetryManager } from './settingsRetryManager';

interface BatchOperation {
  path: string;
  value: any;
}

const logger = createLogger('AutoSaveManager');


export class AutoSaveManager {
  private pendingChanges: Map<string, any> = new Map();
  private saveDebounceTimer: NodeJS.Timeout | null = null;
  private isInitialized: boolean = false;
  private readonly DEBOUNCE_DELAY = 500;


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
    console.warn(`[SETTINGS-DIAG] autoSaveManager.setInitialized(${initialized})`);
    this.isInitialized = initialized;
  }


  queueChange(path: string, value: any) {
    if (!this.isInitialized) {
      console.warn(`[SETTINGS-DIAG] autoSaveManager.queueChange DROPPED (not initialized): ${path}`, value);
      return;
    }
    console.warn(`[SETTINGS-DIAG] autoSaveManager.queueChange: ${path} =`, value);

    if (this.isClientOnlyPath(path)) {
      logger.debug(`Skipping client-only path: ${path}`);
      return;
    }

    this.pendingChanges.set(path, value);
    this.scheduleFlush();
  }


  queueChanges(changes: Map<string, any>) {
    if (!this.isInitialized) {
      console.warn(`[SETTINGS-DIAG] autoSaveManager.queueChanges DROPPED (not initialized): ${changes.size} changes`, [...changes.keys()]);
      return;
    }
    console.warn(`[SETTINGS-DIAG] autoSaveManager.queueChanges: ${changes.size} changes`, [...changes.keys()]);

    changes.forEach((value, path) => {
      if (this.isClientOnlyPath(path)) {
        logger.debug(`Skipping client-only path: ${path}`);
        return;
      }

      this.pendingChanges.set(path, value);
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

    // Clear pending immediately to avoid re-sending on next flush
    this.pendingChanges.clear();

    console.warn(`[SETTINGS-DIAG] autoSaveManager.flush: ${updates.length} updates`, updates.map(u => `${u.path}=${JSON.stringify(u.value)}`));

    try {
      await settingsApi.updateSettingsByPaths(updates);
      console.warn(`[SETTINGS-DIAG] autoSaveManager.flush SUCCESS: ${updates.length} updates sent to server`);
    } catch (error) {
      console.warn(`[SETTINGS-DIAG] autoSaveManager.flush FAILED:`, error);

      // Delegate all failed updates to the centralized retry manager
      for (const { path, value } of updates) {
        settingsRetryManager.addFailedUpdate(
          path,
          value,
          error instanceof Error ? error.message : 'Auto-save flush failed'
        );
      }
    }
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
