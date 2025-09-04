import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { SettingsPath, DeepPartial, Settings } from '../types/generated/settings'
import { createLogger, createErrorMetadata } from '../utils/logger'
import { debugState } from '../utils/clientDebugState'
import { produce } from 'immer';
import { toast } from '../utils/toast';
import { isViewportSetting } from '../features/settings/config/viewportSettings';
import { settingsApi, PathValuePair } from '../api/settingsApi';

const logger = createLogger('SettingsStore')

// Auto-save system with debouncing and error recovery
class AutoSaveManager {
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
    
    const updates = Array.from(this.pendingChanges.entries())
      .map(([path, value]) => ({ path, value }));
    
    try {
      await settingsApi.updateSettingsByPath(updates);
      
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
  private async retryFailedChanges(failedUpdates: PathValuePair[], error: any): Promise<void> {
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

// Global auto-save manager instance
const autoSaveManager = new AutoSaveManager();

// Helper function to find changed paths between two objects
function findChangedPaths(oldObj: any, newObj: any, path: string = ''): string[] {
  const changedPaths: string[] = [];
  
  // Handle null/undefined cases
  if (oldObj === newObj) return changedPaths;
  if (oldObj == null || newObj == null) {
    if (path) changedPaths.push(path);
    return changedPaths;
  }
  
  // Handle primitive values
  if (typeof oldObj !== 'object' || typeof newObj !== 'object') {
    if (oldObj !== newObj && path) {
      changedPaths.push(path);
    }
    return changedPaths;
  }
  
  // Handle objects and arrays
  const allKeys = new Set([...Object.keys(oldObj), ...Object.keys(newObj)]);
  
  for (const key of allKeys) {
    const currentPath = path ? `${path}.${key}` : key;
    const oldValue = oldObj[key];
    const newValue = newObj[key];
    
    if (typeof oldValue === 'object' && typeof newValue === 'object' && oldValue !== null && newValue !== null) {
      // Recursively check nested objects
      changedPaths.push(...findChangedPaths(oldValue, newValue, currentPath));
    } else if (oldValue !== newValue) {
      // Value changed
      changedPaths.push(currentPath);
    }
  }
  
  return changedPaths;
}

/**
 * Optimized Zustand Settings Store
 * 
 * OPTIMIZATION NOTES:
 * - Removed redundant custom subscription system (subscribers Map, subscribe/unsubscribe methods)
 * - Zustand provides native subscriptions that are more efficient:
 *   * Automatic re-render optimization with selector functions
 *   * Built-in shallow comparison and change detection
 *   * No manual Map management or callback tracking needed
 * - Viewport updates now use native Zustand subscription + custom events
 * - Components use useSelectiveSetting hooks that leverage native Zustand selectors
 * 
 * USAGE:
 * - Components: const value = useSelectiveSetting('path.to.setting');
 * - Viewport updates: Listen to 'viewport-update' custom event
 * - Manual subscriptions: useSettingsStore.subscribe(selector, callback, { equalityFn: shallow })
 */
interface SettingsState {
  // Partial state management - only holds what's been loaded
  partialSettings: DeepPartial<Settings>
  loadedPaths: Set<string> // Track which paths have been loaded
  loadingSections: Set<string> // Track sections currently being loaded
  
  initialized: boolean
  authenticated: boolean
  user: { isPowerUser: boolean; pubkey: string } | null
  isPowerUser: boolean
  
  // Native Zustand subscriptions handle selective re-renders automatically

  // Save state tracking (hasUnsavedChanges removed - auto-save handles this)
  lastSavedState: DeepPartial<Settings>
  saving: boolean

  // Core actions - path-based only
  initialize: () => Promise<void>
  setAuthenticated: (authenticated: boolean) => void
  setUser: (user: { isPowerUser: boolean; pubkey: string } | null) => void
  
  // Path-based getters/setters
  get: <T>(path: SettingsPath) => T | undefined
  set: <T>(path: SettingsPath, value: T) => Promise<void>
  
  // Batch operations
  batchSet: (updates: PathValuePair[]) => Promise<void>
  
  // Lazy loading - load settings on demand
  ensureLoaded: (paths: string[]) => Promise<void>
  loadSection: (section: string) => Promise<void>
  isLoaded: (path: SettingsPath) => boolean
  
  // Viewport updates for real-time rendering (native Zustand subscription handles this)
  // No need for custom subscription methods - Zustand handles it natively
  
  // UI control functions
  checkUnsavedChanges: () => boolean
  updateSettings: (updater: (draft: DeepPartial<Settings>) => void) => Promise<void>
}

// Essential paths loaded at startup for fast initialization
const ESSENTIAL_PATHS = [
  'system.debug.enabled',
  'system.websocket.updateRate',
  'system.websocket.reconnectAttempts', 
  'auth.enabled',
  'auth.required',
  'visualisation.rendering.context',
  'xr.enabled',
  'xr.mode'
];
export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      partialSettings: {},
      loadedPaths: new Set(),
      loadingSections: new Set(),
      initialized: false,
      authenticated: false,
      user: null,
      isPowerUser: false,
      // Native Zustand subscriptions - no custom Map needed
      lastSavedState: {},
      saving: false,

      initialize: async () => {
        try {
          if (debugState.isEnabled()) {
            logger.info('Initializing settings store with essential paths only')
          }

          // Load only essential settings for fast startup
          const essentialSettings = await settingsApi.getSettingsByPaths(ESSENTIAL_PATHS);

          if (debugState.isEnabled()) {
            // logger.info('Essential settings loaded:', { essentialSettings })
          }

          set(state => ({
            partialSettings: essentialSettings as DeepPartial<Settings>,
            loadedPaths: new Set(ESSENTIAL_PATHS),
            initialized: true,
            lastSavedState: JSON.parse(JSON.stringify(essentialSettings))
          }));

          // Initialize auto-save manager
          autoSaveManager.setInitialized(true);

          if (debugState.isEnabled()) {
            // logger.info('Settings store initialized with essential paths and auto-save enabled')
          }

        } catch (error) {
          logger.error('Failed to initialize settings store:', createErrorMetadata(error))
          set({ initialized: false })
          autoSaveManager.setInitialized(false);
          throw error
        }
      },

      setAuthenticated: (authenticated: boolean) => set({ authenticated }),

      setUser: (user: { isPowerUser: boolean; pubkey: string } | null) => set({
        user,
        isPowerUser: user?.isPowerUser || false
      }),

      // Path-based getter - handles unloaded state properly
      get: <T>(path: SettingsPath): T | undefined => {
        const { partialSettings, loadedPaths } = get();

        if (!path?.trim()) {
          return partialSettings as unknown as T;
        }

        // Check if this path (or a parent path) has been loaded
        const isPathLoaded = loadedPaths.has(path) || 
          [...loadedPaths].some(loadedPath => 
            path.startsWith(loadedPath + '.') || loadedPath.startsWith(path + '.')
          );

        if (!isPathLoaded) {
          if (debugState.isEnabled()) {
            logger.warn(`Accessing unloaded path: ${path} - path should be loaded before access`);
          }
          // Don't trigger loading here - it causes infinite loops
          // The calling component should use ensureLoaded in useEffect
          return undefined as unknown as T;
        }

        // Navigate the partial settings using the path
        const pathParts = path.split('.');
        let current: any = partialSettings;

        for (const part of pathParts) {
          if (current?.[part] === undefined) {
            return undefined;
          }
          current = current[part];
        }

        return current as T;
      },

      // Path-based setter - updates server immediately
      set: async <T>(path: SettingsPath, value: T): Promise<void> => {
        if (!path?.trim()) {
          throw new Error('Path cannot be empty');
        }

        try {
          await settingsApi.setSetting(path, value);
          
          // Update local state
          set(state => {
            const newPartialSettings = { ...state.partialSettings };
            setNestedValue(newPartialSettings, path, value);
            const newLoadedPaths = new Set(state.loadedPaths);
            newLoadedPaths.add(path);

            return {
              partialSettings: newPartialSettings,
              loadedPaths: newLoadedPaths
            };
          });

          // Native Zustand subscriptions handle notifications automatically

          // Viewport updates handled by native Zustand subscriptions

          if (debugState.isEnabled()) {
            // logger.info('Setting updated:', { path, value });
          }
        } catch (error) {
          logger.error('Failed to update setting:', createErrorMetadata(error));
          throw error;
        }
      },

      // Batch set multiple paths at once
      batchSet: async (updates: PathValuePair[]): Promise<void> => {
        if (!updates.length) return;

        try {
          await settingsApi.updateSettingsByPath(updates);
          
          // Update local state
          set(state => {
            const newPartialSettings = { ...state.partialSettings };
            const newLoadedPaths = new Set(state.loadedPaths);
            
            updates.forEach(({ path, value }) => {
              setNestedValue(newPartialSettings, path, value);
              newLoadedPaths.add(path);
            });

            return {
              partialSettings: newPartialSettings,
              loadedPaths: newLoadedPaths
            };
          });

          // Native Zustand subscriptions handle all notifications automatically

          if (debugState.isEnabled()) {
            // logger.info('Batch settings updated:', { updates });
          }
        } catch (error) {
          logger.error('Failed to batch update settings:', createErrorMetadata(error));
          throw error;
        }
      },

      // Ensure specific paths are loaded
      ensureLoaded: async (paths: string[]): Promise<void> => {
        const { loadedPaths } = get();
        const unloadedPaths = paths.filter(path => !loadedPaths.has(path));
        
        if (unloadedPaths.length === 0) {
          return; // All paths already loaded
        }

        try {
          const pathSettings = await settingsApi.getSettingsByPaths(unloadedPaths);
          
          set(state => {
            const newPartialSettings = { ...state.partialSettings };
            const newLoadedPaths = new Set(state.loadedPaths);
            
            Object.entries(pathSettings).forEach(([path, value]) => {
              setNestedValue(newPartialSettings, path, value);
              newLoadedPaths.add(path);
            });

            return {
              partialSettings: newPartialSettings,
              loadedPaths: newLoadedPaths
            };
          });

          // Commented out to reduce noise - only log in debug mode if really needed
          // if (debugState.isEnabled()) {
          //   logger.info('Paths loaded on demand:', { paths: unloadedPaths });
          // }
        } catch (error) {
          logger.error('Failed to load paths:', createErrorMetadata(error));
          throw error;
        }
      },

      // Load entire section (collection of related paths)  
      loadSection: async (section: string): Promise<void> => {
        const { loadingSections } = get();
        if (loadingSections.has(section)) {
          return; // Already loading
        }

        const sectionPaths = getSectionPaths(section);
        if (sectionPaths.length === 0) {
          logger.warn(`Unknown section: ${section}`);
          return;
        }

        // Mark as loading
        set(state => ({
          loadingSections: new Set(state.loadingSections).add(section)
        }));

        try {
          await get().ensureLoaded(sectionPaths);

          if (debugState.isEnabled()) {
            // logger.info(`Section loaded: ${section}`, { paths: sectionPaths });
          }
        } finally {
          // Mark as no longer loading
          set(state => {
            const newLoadingSections = new Set(state.loadingSections);
            newLoadingSections.delete(section);
            return { loadingSections: newLoadingSections };
          });
        }
      },

      // Check if a path has been loaded
      isLoaded: (path: SettingsPath): boolean => {
        const { loadedPaths } = get();
        return loadedPaths.has(path);
      },

      // Viewport updates handled by native Zustand subscription in application startup

      // Native Zustand subscriptions replace custom subscription system

      // Check if there are unsaved changes
      checkUnsavedChanges: (): boolean => {
        return autoSaveManager.hasPendingChanges();
      },

      // Update settings using immer-style updater
      updateSettings: async (updater: (draft: DeepPartial<Settings>) => void): Promise<void> => {
        const { partialSettings } = get();
        
        // Use produce to create immutable update
        const newSettings = produce(partialSettings, updater);
        
        // Check what paths changed
        const changedPaths = findChangedPaths(partialSettings, newSettings);
        
        if (changedPaths.length === 0) {
          return; // No changes
        }

        // Update state and queue auto-save
        set(state => {
          return {
            partialSettings: newSettings,
            // Add changed paths to loaded paths
            loadedPaths: new Set([...state.loadedPaths, ...changedPaths])
          };
        });

        // Queue changes for auto-save
        const changesMap = new Map<string, any>();
        changedPaths.forEach(path => {
          const pathParts = path.split('.');
          let current: any = newSettings;
          for (const part of pathParts) {
            current = current[part];
          }
          changesMap.set(path, current);
        });
        autoSaveManager.queueChanges(changesMap);

        if (debugState.isEnabled()) {
          logger.info('Settings updated via updateSettings:', { changedPaths });
        }
      },
    }),
    {
      name: 'graph-viz-settings-v2',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        // Only persist auth state and essential settings, not full partial settings
        authenticated: state.authenticated,
        user: state.user,
        isPowerUser: state.isPowerUser,
        // Only persist essential paths to avoid staleness
        essentialPaths: ESSENTIAL_PATHS.reduce((acc, path) => {
          const value = state.partialSettings[path];
          if (value !== undefined) {
            acc[path] = value;
          }
          return acc;
        }, {} as Record<string, any>)
      }),
      merge: (persistedState: any, currentState: SettingsState): SettingsState => {
        if (!persistedState) return currentState;
        
        return {
          ...currentState,
          authenticated: persistedState.authenticated || false,
          user: persistedState.user || null,
          isPowerUser: persistedState.isPowerUser || false,
          // Don't merge partial settings - let them load fresh from server
        };
      },
      onRehydrateStorage: () => (state) => {
        if (state) {
          if (debugState.isEnabled()) {
            logger.info('Settings store rehydrated from storage');
          }
        }
      }
    }
  )
)

// Helper function to get paths for a specific section
function getSectionPaths(section: string): string[] {
  const sectionPathMap: Record<string, string[]> = {
    'physics': [
      'visualisation.graphs.logseq.physics',
      'visualisation.graphs.visionflow.physics'
    ],
    'rendering': [
      'visualisation.rendering.ambientLightIntensity',
      'visualisation.rendering.backgroundColor',
      'visualisation.rendering.directionalLightIntensity',
      'visualisation.rendering.enableAmbientOcclusion',
      'visualisation.rendering.enableAntialiasing',
      'visualisation.rendering.enableShadows',
      'visualisation.rendering.environmentIntensity',
      'visualisation.rendering.shadowMapSize',
      'visualisation.rendering.shadowBias',
      'visualisation.rendering.context'
    ],
    'xr': [
      'xr.enabled',
      'xr.mode',
      'xr.enableHandTracking',
      'xr.enableHaptics',
      'xr.quality'
    ],
    'glow': [
      'visualisation.glow.enabled',
      'visualisation.glow.intensity',
      'visualisation.glow.radius',
      'visualisation.glow.threshold'
    ],
    'hologram': [
      'visualisation.hologram.ringCount',
      'visualisation.hologram.ringColor',
      'visualisation.hologram.globalRotationSpeed'
    ],
    'nodes': [
      'visualisation.graphs.logseq.nodes',
      'visualisation.graphs.visionflow.nodes'
    ],
    'edges': [
      'visualisation.graphs.logseq.edges',
      'visualisation.graphs.visionflow.edges'
    ],
    'labels': [
      'visualisation.graphs.logseq.labels',
      'visualisation.graphs.visionflow.labels'
    ]
  };

  return sectionPathMap[section] || [];
}

// Helper function to set nested value by dot notation path
function setNestedValue(obj: any, path: string, value: any): void {
  const keys = path.split('.');
  let current = obj;
  
  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];
    if (!(key in current) || typeof current[key] !== 'object' || current[key] === null) {
      current[key] = {};
    }
    current = current[key];
  }
  
  current[keys[keys.length - 1]] = value;
}

// Helper function to extract all paths from a settings object
function getAllSettingsPaths(obj: any, prefix: string = ''): string[] {
  const paths: string[] = [];
  
  if (obj && typeof obj === 'object') {
    for (const [key, value] of Object.entries(obj)) {
      const currentPath = prefix ? `${prefix}.${key}` : key;
      
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        // Recursively get paths from nested objects
        paths.push(...getAllSettingsPaths(value, currentPath));
      } else {
        // This is a leaf value
        paths.push(currentPath);
      }
    }
  }
  
  return paths;
}

// Export for testing and direct access
export const settingsStoreUtils = {
  autoSaveManager,
  getSectionPaths,
  setNestedValue,
  getAllSettingsPaths
};
