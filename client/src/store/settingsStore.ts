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
import { normalizeBloomGlowSettings, transformBloomToGlow } from '../utils/caseConversion';



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

// Shared debounced save function with bloom/glow field transformation
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

    // Transform bloom fields to glow fields for server compatibility
    const serverSettings = transformBloomToGlow(settings);

    // Log the exact payload being sent for debugging
    logger.info('[SETTINGS DEBUG] Sending settings payload to server:', {
      endpoint: '/api/settings',
      payloadKeys: Object.keys(serverSettings),
      sampleFields: {
        'xr.enabled': serverSettings.xr?.enabled,
        'xr.enableXrMode': (serverSettings.xr as any)?.enableXrMode,
        'system.debug.enabled': serverSettings.system?.debug?.enabled,
        'system.debug.enableClientDebugMode': (serverSettings.system?.debug as any)?.enableClientDebugMode,
        'visualisation.glow.enabled': serverSettings.visualisation?.glow?.enabled,
        'visualisation.bloom.enabled': serverSettings.visualisation?.bloom?.enabled
      }
    });

    // Server handles camelCase to snake_case conversion automatically
    // Send transformed settings (bloom -> glow) to server
    const updatedSettings = await apiService.post('/settings', serverSettings, headers);
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
  updatePhysics: (graphName: string, params: Partial<GPUPhysicsParams>) => void; // Alias for updateGPUPhysics
  updateWarmupSettings: (settings: WarmupSettings) => void;
  
  // WebSocket integration for real-time physics updates
  notifyPhysicsUpdate: (graphName: string, params: Partial<GPUPhysicsParams>) => void;
}

// GPU-specific interfaces for type safety
interface GPUPhysicsParams {
  springK: number;
  repelK: number;
  attractionK: number;
  gravity: number;
  dt: number;
  maxVelocity: number;
  damping: number;
  temperature: number;
  maxRepulsionDist: number;
  
  // New CUDA kernel parameters
  restLength: number;
  repulsionCutoff: number;
  repulsionSofteningEpsilon: number;
  centerGravityK: number;
  gridCellSize: number;
  featureFlags: number;
  
  // Warmup parameters
  warmupIterations: number;
  coolingRate: number;
  
  // Additional boundary and collision parameters
  enableBounds?: boolean;
  boundsSize?: number;
  boundaryDamping?: number;
  collisionRadius?: number;
  
  // Advanced parameters
  iterations?: number;
  massScale?: number;
  updateThreshold?: number;
  
  // Missing CUDA parameters
  /** Extreme boundary force multiplier (1.0-5.0) */
  boundaryExtremeMultiplier?: number;
  /** Extreme boundary force strength multiplier (1.0-20.0) */
  boundaryExtremeForceMultiplier?: number;
  /** Boundary velocity damping factor (0.0-1.0) */
  boundaryVelocityDamping?: number;
  /** Maximum force magnitude (1-1000) */
  maxForce?: number;
  /** Random seed for initialization */
  seed?: number;
  /** Current iteration count */
  iteration?: number;
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
  warmupIterations?: number; // Optional for direct warmup iteration control
  coolingRate?: number; // Optional for direct cooling rate control
}

// Clear bad physics from localStorage on startup
if (typeof window !== 'undefined' && window.localStorage) {
  try {
    const stored = localStorage.getItem('settings-storage');
    if (stored) {
      const parsed = JSON.parse(stored);
      // Clear physics to force reload from server
      if (parsed?.state?.settings?.visualisation?.graphs) {
        delete parsed.state.settings.visualisation.graphs.logseq?.physics;
        delete parsed.state.settings.visualisation.graphs.visionflow?.physics;
        localStorage.setItem('settings-storage', JSON.stringify(parsed));
      }
    }
  } catch (e) {
    // Could not clear cached physics settings
  }
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

          // ALWAYS fetch settings from server and use them as source of truth
          try {
            // Use the settings service to fetch settings
            const rawServerSettings = await apiService.get('/settings')

            if (rawServerSettings) {
              if (debugState.isEnabled()) {
                logger.info('Fetched settings from server:', { rawServerSettings })
              }

              // Transform server glow settings to client bloom settings for compatibility
              const clientCompatibleSettings = normalizeBloomGlowSettings(rawServerSettings, 'toClient');

              // Server settings OVERRIDE everything - server is source of truth
              // Only use defaults for missing fields, ignore localStorage for physics
              const mergedSettings = deepMerge(defaultSettings, clientCompatibleSettings)

              if (debugState.isEnabled()) {
                logger.info('Using server settings as source of truth with bloom/glow normalization:', { 
                  mergedSettings,
                  hasBloom: !!mergedSettings.visualisation?.bloom,
                  hasGlow: !!mergedSettings.visualisation?.glow
                })
              }

              set({
                settings: mergedSettings,
                initialized: true
              })

              if (debugState.isEnabled()) {
                logger.info('Settings loaded from server with bloom/glow field normalization')
              }

              return mergedSettings
            }
          } catch (error) {
            logger.warn('Failed to fetch settings from server:', createErrorMetadata(error))
            // Continue with safe defaults if server fetch fails
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
        
        // Validate parameter ranges for new CUDA parameters
        const validatedParams = { ...params };
        
        // Validate new CUDA parameters
        if (validatedParams.restLength !== undefined) {
          validatedParams.restLength = Math.max(0.1, Math.min(10.0, validatedParams.restLength));
        }
        if (validatedParams.repulsionCutoff !== undefined) {
          validatedParams.repulsionCutoff = Math.max(1.0, Math.min(1000.0, validatedParams.repulsionCutoff));
        }
        if (validatedParams.repulsionSofteningEpsilon !== undefined) {
          validatedParams.repulsionSofteningEpsilon = Math.max(0.001, Math.min(1.0, validatedParams.repulsionSofteningEpsilon));
        }
        if (validatedParams.centerGravityK !== undefined) {
          validatedParams.centerGravityK = Math.max(-1.0, Math.min(1.0, validatedParams.centerGravityK));
        }
        if (validatedParams.gridCellSize !== undefined) {
          validatedParams.gridCellSize = Math.max(1.0, Math.min(100.0, validatedParams.gridCellSize));
        }
        if (validatedParams.featureFlags !== undefined) {
          validatedParams.featureFlags = Math.max(0, Math.min(255, Math.floor(validatedParams.featureFlags)));
        }
        
        // Validate existing parameters with updated ranges
        if (validatedParams.springK !== undefined) {
          validatedParams.springK = Math.max(0.001, Math.min(10.0, validatedParams.springK));
        }
        if (validatedParams.repelK !== undefined) {
          validatedParams.repelK = Math.max(0.001, Math.min(100.0, validatedParams.repelK));
        }
        if (validatedParams.attractionK !== undefined) {
          validatedParams.attractionK = Math.max(0.0, Math.min(1.0, validatedParams.attractionK));
        }
        if (validatedParams.gravity !== undefined) {
          validatedParams.gravity = Math.max(-1.0, Math.min(1.0, validatedParams.gravity));
        }
        if (validatedParams.warmupIterations !== undefined) {
          validatedParams.warmupIterations = Math.max(0, Math.min(1000, Math.floor(validatedParams.warmupIterations)));
        }
        if (validatedParams.coolingRate !== undefined) {
          validatedParams.coolingRate = Math.max(0.0001, Math.min(1.0, validatedParams.coolingRate));
        }
        
        state.updateSettings((draft) => {
          const graphSettings = draft.visualisation.graphs[graphName as keyof typeof draft.visualisation.graphs];
          if (graphSettings && graphSettings.physics) {
            Object.assign(graphSettings.physics, validatedParams);
            
            if (debugState.isEnabled()) {
              logger.info('GPU Physics parameters updated:', {
                graphName,
                updatedParams: validatedParams,
                newPhysicsState: graphSettings.physics
              });
            }
          }
        });
        
        // Trigger WebSocket notification for real-time updates
        state.notifyPhysicsUpdate(graphName, validatedParams);
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

      // Alias method for updateGPUPhysics for backward compatibility
      updatePhysics: (graphName: string, params: Partial<GPUPhysicsParams>) => {
        const state = get();
        state.updateGPUPhysics(graphName, params);
      },

      // WebSocket notification for physics updates
      notifyPhysicsUpdate: (graphName: string, params: Partial<GPUPhysicsParams>) => {
        if (typeof window !== 'undefined') {
          try {
            // Try to get WebSocket service from window
            const wsService = (window as any).webSocketService;
            if (wsService && wsService.isConnected && wsService.isConnected()) {
              const message = {
                type: 'physics_parameter_update',
                timestamp: Date.now(),
                graph: graphName,
                parameters: params
              };
              
              wsService.send(message);
              
              if (debugState.isEnabled()) {
                logger.info('Physics update sent via WebSocket:', {
                  graphName,
                  parameters: params,
                  messageType: 'physics_parameter_update'
                });
              }
            } else {
              if (debugState.isEnabled()) {
                logger.info('WebSocket not connected, physics update queued for next connection');
              }
            }
          } catch (error) {
            logger.warn('Failed to notify physics update via WebSocket:', createErrorMetadata(error));
          }
          
          // Also dispatch a custom event for other components to listen to
          const event = new CustomEvent('physicsParametersUpdated', {
            detail: { graphName, params }
          });
          window.dispatchEvent(event);
        }
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
