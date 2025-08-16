import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { defaultSettings } from '../features/settings/config/defaultSettings'
import { Settings, SettingsPath, GraphSettings } from '../features/settings/config/settings'
import { createLogger, createErrorMetadata } from '../utils/logger'
import { debugState } from '../utils/clientDebugState'
import { deepMerge } from '../utils/deepMerge';
import { apiService } from '../services/apiService';
import { produce } from 'immer';
import { toast } from '../features/design-system/components/Toast';
import { isViewportSetting } from '../features/settings/config/viewportSettings';



const logger = createLogger('SettingsStore')

// Debounce utility
let saveTimeoutId: ReturnType<typeof setTimeout> | null = null;

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
 * Normalize settings to server schema to avoid validation errors.
 * Maps old parameter names to new GPU-aligned names and ensures proper type conversions.
 * - system.debug.logLevel: coerce 'debug'|'info'|'warn'|'error' -> 0..3 and clamp to [0,3]
 * - physics.iterations: ensure it's an integer (JavaScript sends floats as 100.0)
 * - GPU parameters: map old names to new GPU-aligned names
 */
function normalizeSettingsForServer(settings: Settings) {
  // JSON clone to avoid mutating store state
  const normalized: any = JSON.parse(JSON.stringify(settings));
  
  // Fix system.debug.logLevel
  const sys = normalized?.system;
  const dbg = sys?.debug;
  if (dbg) {
    const map: Record<string, number> = { debug: 0, info: 1, warn: 2, error: 3 };
    const lvl = dbg.logLevel;
    if (typeof lvl === 'string') {
      if (map[lvl] !== undefined) {
        dbg.logLevel = map[lvl];
      }
    } else if (typeof lvl === 'number') {
      if (Number.isFinite(lvl)) {
        if (lvl < 0) dbg.logLevel = 0;
        if (lvl > 3) dbg.logLevel = 3;
      } else {
        dbg.logLevel = 1; // default to 'info'
      }
    }
  }
  
  // Fix physics parameters for all graphs and map old names to new GPU-aligned names
  const graphs = normalized?.visualisation?.graphs;
  if (graphs) {
    for (const graphName of Object.keys(graphs)) {
      const physics = graphs[graphName]?.physics;
      if (physics) {
        // Ensure iterations is an integer
        if (physics.iterations !== undefined) {
          physics.iterations = Math.round(physics.iterations);
        }
        
        // Map old parameter names to new GPU-aligned names
        if (physics.springStrength !== undefined && physics.springK === undefined) {
          physics.springK = physics.springStrength;
          delete physics.springStrength;
        }
        if (physics.repulsionStrength !== undefined && physics.repelK === undefined) {
          physics.repelK = physics.repulsionStrength;
          delete physics.repulsionStrength;
        }
        if (physics.attractionStrength !== undefined && physics.attractionK === undefined) {
          physics.attractionK = physics.attractionStrength;
          delete physics.attractionStrength;
        }
        if (physics.timeStep !== undefined && physics.dt === undefined) {
          physics.dt = physics.timeStep;
          delete physics.timeStep;
        }
        if (physics.repulsionDistance !== undefined && physics.maxRepulsionDist === undefined) {
          physics.maxRepulsionDist = physics.repulsionDistance;
          delete physics.repulsionDistance;
        }
        
        // Ensure GPU parameters are properly typed
        ['springK', 'repelK', 'attractionK', 'dt', 'maxVelocity', 'damping', 
         'temperature', 'maxRepulsionDist', 'warmupIterations', 'coolingRate'].forEach(param => {
          if (physics[param] !== undefined) {
            physics[param] = Number(physics[param]);
          }
        });
      }
    }
  }
  
  // Fix hologram.ringCount to ensure it's an integer
  const hologram = normalized?.visualisation?.hologram;
  if (hologram && hologram.ringCount !== undefined) {
    // Ensure ringCount is an integer
    hologram.ringCount = Math.round(hologram.ringCount);
  }
  
  // Ensure dashboard GPU status fields are properly typed
  const dashboard = normalized?.dashboard;
  if (dashboard) {
    if (dashboard.iterationCount !== undefined) {
      dashboard.iterationCount = Math.round(Number(dashboard.iterationCount));
    }
    if (dashboard.activeConstraints !== undefined) {
      dashboard.activeConstraints = Math.round(Number(dashboard.activeConstraints));
    }
  }
  
  // Ensure analytics clustering parameters are properly typed
  const analytics = normalized?.analytics;
  if (analytics?.clustering) {
    const clustering = analytics.clustering;
    if (clustering.clusterCount !== undefined) {
      clustering.clusterCount = Math.round(Number(clustering.clusterCount));
    }
    if (clustering.iterations !== undefined) {
      clustering.iterations = Math.round(Number(clustering.iterations));
    }
    if (clustering.resolution !== undefined) {
      clustering.resolution = Number(clustering.resolution);
    }
  }
  
  // Ensure performance warmup parameters are properly typed
  const performance = normalized?.performance;
  if (performance) {
    if (performance.warmupDuration !== undefined) {
      performance.warmupDuration = Number(performance.warmupDuration);
    }
    if (performance.convergenceThreshold !== undefined) {
      performance.convergenceThreshold = Number(performance.convergenceThreshold);
    }
  }
  
  return normalized;
}
// Shared debounced save function
const debouncedSaveToServer = async (settings: Settings, initialized: boolean) => {
  if (!initialized || settings.system?.persistSettings === false) {
    return;
  }

  try {
    const headers: Record<string, string> = {};

    // Add authentication headers if available
    try {
      const { nostrAuth } = await import('../services/nostrAuthService');
      if (nostrAuth.isAuthenticated()) {
        const user = nostrAuth.getCurrentUser();
        const token = nostrAuth.getSessionToken();
        if (user && token) {
          headers['X-Nostr-Pubkey'] = user.pubkey;
          headers['Authorization'] = `Bearer ${token}`;
          if (debugState.isEnabled()) {
            logger.info('Using Nostr authentication for settings sync');
          }
        }
      }
    } catch (error) {
      logger.warn('Error getting Nostr authentication:', createErrorMetadata(error));
    }

    // Log the exact payload being sent for debugging
    logger.info('[SETTINGS DEBUG] Sending settings payload to server:', {
      endpoint: '/api/settings',
      payloadKeys: Object.keys(settings),
      sampleFields: {
        'xr.enabled': settings.xr?.enabled,
        'xr.enableXrMode': (settings.xr as any)?.enableXrMode,
        'system.debug.enabled': settings.system?.debug?.enabled,
        'system.debug.enableClientDebugMode': (settings.system?.debug as any)?.enableClientDebugMode
      }
    });

    const payload = normalizeSettingsForServer(settings);
    const updatedSettings = await apiService.post('/settings', payload, headers);
    if (updatedSettings) {
      if (debugState.isEnabled()) {
        logger.info('Settings saved to server successfully');
      }
      toast({ title: "Settings Saved", description: "Your settings have been synced with the server." });
    }
  } catch (error) {
    const errorMeta = createErrorMetadata(error);
    logger.error('Failed to save settings to server:', errorMeta);
    toast({
      variant: "destructive",
      title: "Save Failed",
      description: `Could not save settings to server. ${errorMeta.message || 'Check console.'}`
    });
  }
};

// Schedule debounced save
const scheduleSave = (settings: Settings, initialized: boolean) => {
  if (saveTimeoutId) {
    clearTimeout(saveTimeoutId);
  }
  saveTimeoutId = setTimeout(() => debouncedSaveToServer(settings, initialized), 500);
};

interface SettingsState {
  settings: Settings
  initialized: boolean
  authenticated: boolean
  user: { isPowerUser: boolean; pubkey: string } | null
  isPowerUser: boolean // Direct access to power user state
  subscribers: Map<string, Set<() => void>>

  // Actions
  initialize: () => Promise<Settings>
  setAuthenticated: (authenticated: boolean) => void
  setUser: (user: { isPowerUser: boolean; pubkey: string } | null) => void
  get: <T>(path: SettingsPath) => T
  set: <T>(path: SettingsPath, value: T) => void
  subscribe: (path: SettingsPath, callback: () => void, immediate?: boolean) => () => void;
  unsubscribe: (path: SettingsPath, callback: () => void) => void;
  updateSettings: (updater: (draft: Settings) => void) => void;
  notifyViewportUpdate: (path: SettingsPath) => void; // For real-time viewport updates
  
  // GPU-specific methods
  updateComputeMode: (mode: string) => void;
  updateClustering: (config: ClusteringConfig) => void;
  updateConstraints: (constraints: ConstraintConfig[]) => void;
  updateGPUPhysics: (graphName: string, params: Partial<GPUPhysicsParams>) => void;
  updateWarmupSettings: (settings: WarmupSettings) => void;
}

// GPU-specific interfaces for type safety
interface GPUPhysicsParams {
  springK: number;
  repelK: number;
  attractionK: number;
  dt: number;
  maxVelocity: number;
  damping: number;
  temperature: number;
  maxRepulsionDist: number;
  warmupIterations: number;
  coolingRate: number;
}

interface ClusteringConfig {
  algorithm: 'none' | 'kmeans' | 'spectral' | 'louvain';
  clusterCount: number;
  resolution: number;
  iterations: number;
  exportEnabled: boolean;
  importEnabled: boolean;
}

interface ConstraintConfig {
  id: string;
  name: string;
  enabled: boolean;
  description?: string;
  icon?: string;
}

interface WarmupSettings {
  warmupDuration: number;
  convergenceThreshold: number;
  enableAdaptiveCooling: boolean;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      settings: defaultSettings,
      initialized: false,
      authenticated: false,
      user: null,
      isPowerUser: false,
      subscribers: new Map(),

      initialize: async () => {
        try {
          if (debugState.isEnabled()) {
            logger.info('Initializing settings')
          }

          // Load settings from localStorage via zustand persist
          const currentSettings = get().settings

          // Fetch settings from server if available
          try {
            // Use the settings service to fetch settings
            const serverSettings = await apiService.get('/settings')

            if (serverSettings) {
              if (debugState.isEnabled()) {
                logger.info('Fetched settings from server:', { serverSettings })
              }

              // Merge server settings with defaults and current settings using deep merge
              // This ensures all nested objects are properly merged
              const mergedSettings = deepMerge(defaultSettings, currentSettings, serverSettings)

              const migratedSettings = mergedSettings

              if (debugState.isEnabled()) {
                logger.info('Deep merged settings:', { migratedSettings })
              }

              set({
                settings: migratedSettings,
                initialized: true
              })

              if (debugState.isEnabled()) {
                logger.info('Settings loaded from server and merged')
              }

              return mergedSettings
            }
          } catch (error) {
            logger.warn('Failed to fetch settings from server:', createErrorMetadata(error))
            // Continue with local settings if server fetch fails
          }

          // Use current settings as is
          const migratedSettings = currentSettings

          // Mark as initialized
          set({
            settings: migratedSettings,
            initialized: true
          })

          if (debugState.isEnabled()) {
            logger.info('Settings initialized from local storage')
          }

          return migratedSettings
        } catch (error) {
          logger.error('Failed to initialize settings:', createErrorMetadata(error))

          // Fall back to default settings
          const migratedDefaults = defaultSettings
          set({
            settings: migratedDefaults,
            initialized: true
          })

          return migratedDefaults
        }
      },

      setAuthenticated: (authenticated: boolean) => set({ authenticated }),

      setUser: (user: { isPowerUser: boolean; pubkey: string } | null) => set({
        user,
        isPowerUser: user?.isPowerUser || false
      }),

      notifyViewportUpdate: (path: SettingsPath) => {
        // This method will be called for settings that need immediate viewport updates
        const callbacks = get().subscribers.get('viewport.update')
        if (callbacks) {
          Array.from(callbacks).forEach(callback => {
            try {
              callback()
            } catch (error) {
              logger.error(`Error in viewport update subscriber:`, createErrorMetadata(error))
            }
          })
        }
      },

      get: <T>(path: SettingsPath): T => {
        const settings = get().settings

        if (!path || path === '') {
          return settings as unknown as T
        }

        // Navigate the settings object using the path
        let current: any = settings
        const pathParts = path.split('.')

        for (const part of pathParts) {
          if (current === undefined || current === null) {
            return undefined as unknown as T
          }
          current = current[part]
        }

        return current as T
      },

      set: <T>(path: SettingsPath, value: T) => {
        const state = get();

        // Use updateSettings internally which will handle viewport updates
        state.updateSettings((draft) => {
          // If setting the entire object
          if (!path || path === '') {
            Object.assign(draft, value);
            return;
          }

          // Navigate to the correct location and update
          const pathParts = path.split('.');
          let current: any = draft;

          // Navigate to the parent of the setting we want to update
          for (let i = 0; i < pathParts.length - 1; i++) {
            const part = pathParts[i];
            if (current[part] === undefined || current[part] === null) {
              current[part] = {};
            }
            current = current[part];
          }

          // Update the value
          const finalPart = pathParts[pathParts.length - 1];
          current[finalPart] = value;
        });
      },

      subscribe: (path: SettingsPath, callback: () => void, immediate: boolean = true) => {
        set(state => {
          const subscribers = new Map(state.subscribers)

          if (!subscribers.has(path)) {
            subscribers.set(path, new Set())
          }

          subscribers.get(path)!.add(callback)

          return { subscribers }
        })

        // Call callback immediately if requested and initialized
        if (immediate && get().initialized) {
          callback()
        }

        // Return unsubscribe function
        return () => get().unsubscribe(path, callback)
      },

      unsubscribe: (path: SettingsPath, callback: () => void) => {
        set(state => {
          const subscribers = new Map(state.subscribers)

          if (subscribers.has(path)) {
            const callbacks = subscribers.get(path)!
            callbacks.delete(callback)

            if (callbacks.size === 0) {
              subscribers.delete(path)
            }
          }

          return { subscribers }
        })
      },

      // Immer-based updateSettings - the preferred method for updating settings
      updateSettings: (updater) => {
        // Get the old settings for comparison
        const oldSettings = get().settings;
        
        // Apply the update
        set((state) => produce(state, (draft) => {
          updater(draft.settings);
        }));

        // After state update, handle notifications and saving
        const state = get();
        
        // Find which paths changed
        const changedPaths = findChangedPaths(oldSettings, state.settings);
        
        if (debugState.isEnabled() && changedPaths.length > 0) {
          logger.info('Settings updated', { changedPaths });
        }

        // Check if any viewport settings were updated
        const viewportUpdated = changedPaths.some(path => isViewportSetting(path));
        
        if (viewportUpdated) {
          // Trigger immediate viewport update
          state.notifyViewportUpdate('viewport.update');
          
          if (debugState.isEnabled()) {
            logger.info('Viewport settings updated, triggering immediate update', { 
              viewportPaths: changedPaths.filter(path => isViewportSetting(path))
            });
          }
        }

        // Notify all subscribers
        const allCallbacks = new Set<() => void>();
        state.subscribers.forEach(callbacks => {
          callbacks.forEach(cb => allCallbacks.add(cb));
        });

        Array.from(allCallbacks).forEach(callback => {
          try {
            callback();
          } catch (error) {
            logger.error('Error in settings subscriber during updateSettings:', createErrorMetadata(error));
          }
        });

        // Schedule save to server
        scheduleSave(state.settings, state.initialized);
      },

      // GPU-specific methods
      updateComputeMode: (mode: string) => {
        const state = get();
        state.updateSettings((draft) => {
          if (!draft.dashboard) {
            draft.dashboard = {};
          }
          (draft.dashboard as any).computeMode = mode;
        });
      },

      updateClustering: (config: ClusteringConfig) => {
        const state = get();
        state.updateSettings((draft) => {
          if (!draft.analytics) {
            (draft as any).analytics = {};
          }
          if (!(draft as any).analytics.clustering) {
            (draft as any).analytics.clustering = {};
          }
          Object.assign((draft as any).analytics.clustering, config);
        });
      },

      updateConstraints: (constraints: ConstraintConfig[]) => {
        const state = get();
        state.updateSettings((draft) => {
          if (!draft.developer) {
            (draft as any).developer = {};
          }
          if (!(draft as any).developer.constraints) {
            (draft as any).developer.constraints = {};
          }
          (draft as any).developer.constraints.active = constraints;
        });
      },

      updateGPUPhysics: (graphName: string, params: Partial<GPUPhysicsParams>) => {
        const state = get();
        state.updateSettings((draft) => {
          const graphSettings = draft.visualisation.graphs[graphName as keyof typeof draft.visualisation.graphs];
          if (graphSettings && graphSettings.physics) {
            Object.assign(graphSettings.physics, params);
          }
        });
      },

      updateWarmupSettings: (settings: WarmupSettings) => {
        const state = get();
        state.updateSettings((draft) => {
          if (!draft.performance) {
            (draft as any).performance = {};
          }
          Object.assign((draft as any).performance, settings);
        });
      },

      // The subscribe and unsubscribe functions below were duplicated and are removed by this change.
    }),
    {
      name: 'graph-viz-settings',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        settings: state.settings,
        authenticated: state.authenticated,
        user: state.user,
        isPowerUser: state.isPowerUser
      }),
      onRehydrateStorage: () => (state) => {
        if (state && state.settings) {
          if (debugState.isEnabled()) {
            logger.info('Settings rehydrated from storage');
          }
        }
      }
    }
  )
)

// Export for testing and direct access
export const settingsStoreUtils = {
  debouncedSaveToServer,
  scheduleSave
};
