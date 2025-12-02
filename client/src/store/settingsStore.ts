import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { Settings, SettingsPath, DeepPartial } from '../features/settings/config/settings'
import { createLogger } from '../utils/loggerConfig'
import { createErrorMetadata } from '../utils/loggerConfig'
import { debugState } from '../utils/clientDebugState'
import { produce } from 'immer';
import { toast } from '../features/design-system/components/Toast';
import { isViewportSetting } from '../features/settings/config/viewportSettings';
import { settingsApi, BatchOperation } from '../api/settingsApi';
import { nostrAuth } from '../services/nostrAuthService';
import { autoSaveManager } from './autoSaveManager';



const logger = createLogger('SettingsStore')

// Helper to wait for authentication to be ready
async function waitForAuthReady(maxWaitMs: number = 3000): Promise<void> {
  const startTime = Date.now();

  
  if (!nostrAuth['initialized']) {
    logger.info('Waiting for nostrAuth to initialize...');
    await nostrAuth.initialize();
  }

  
  return new Promise((resolve) => {
    const checkAuth = () => {
      const elapsed = Date.now() - startTime;

      
      if (elapsed >= maxWaitMs || !localStorage.getItem('nostr_session_token')) {
        logger.info('Proceeding with settings initialization', {
          authenticated: nostrAuth.isAuthenticated(),
          elapsed
        });
        resolve();
        return;
      }

      
      if (nostrAuth.isAuthenticated()) {
        logger.info('Auth ready, proceeding with settings initialization');
        resolve();
        return;
      }

      
      setTimeout(checkAuth, 100);
    };

    checkAuth();
  });
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
  'xr.mode',

  'visualisation.graphs.logseq.physics',
  'visualisation.graphs.visionflow.physics',

  // Node filtering settings - needed for visibility filtering
  'nodeFilter.enabled',
  'nodeFilter.qualityThreshold',
  'nodeFilter.authorityThreshold',
  'nodeFilter.filterByQuality',
  'nodeFilter.filterByAuthority',
  'nodeFilter.filterMode'
];


// Helper function to find changed paths between two objects
function findChangedPaths(oldObj: any, newObj: any, path: string = ''): string[] {
  const changedPaths: string[] = [];
  
  
  if (oldObj === newObj) return changedPaths;
  if (oldObj == null || newObj == null) {
    if (path) changedPaths.push(path);
    return changedPaths;
  }
  
  
  if (typeof oldObj !== 'object' || typeof newObj !== 'object') {
    if (oldObj !== newObj && path) {
      changedPaths.push(path);
    }
    return changedPaths;
  }
  
  
  const allKeys = new Set([...Object.keys(oldObj), ...Object.keys(newObj)]);
  
  for (const key of allKeys) {
    const currentPath = path ? `${path}.${key}` : key;
    const oldValue = oldObj[key];
    const newValue = newObj[key];
    
    if (typeof oldValue === 'object' && typeof newValue === 'object' && oldValue !== null && newValue !== null) {
      
      changedPaths.push(...findChangedPaths(oldValue, newValue, currentPath));
    } else if (oldValue !== newValue) {
      
      changedPaths.push(currentPath);
    }
  }
  
  return changedPaths;
}

interface SettingsState {
  
  partialSettings: DeepPartial<Settings>
  loadedPaths: Set<string> 
  loadingSections: Set<string> 
  
  
  settings: DeepPartial<Settings> 
  
  initialized: boolean
  authenticated: boolean
  user: { isPowerUser: boolean; pubkey: string } | null
  isPowerUser: boolean 
  subscribers: Map<string, Set<() => void>>

  
  initialize: () => Promise<void>
  setAuthenticated: (authenticated: boolean) => void
  setUser: (user: { isPowerUser: boolean; pubkey: string } | null) => void
  get: <T>(path: SettingsPath) => T
  set: <T>(path: SettingsPath, value: T) => void
  subscribe: (path: SettingsPath, callback: () => void, immediate?: boolean) => () => void;
  unsubscribe: (path: SettingsPath, callback: () => void) => void;
  updateSettings: (updater: (draft: Settings) => void) => void;
  notifyViewportUpdate: (path: SettingsPath) => void; 
  
  
  ensureLoaded: (paths: string[]) => Promise<void>
  loadSection: (section: string) => Promise<void>
  isLoaded: (path: SettingsPath) => boolean
  
  
  getByPath: <T>(path: SettingsPath) => Promise<T>; 
  setByPath: <T>(path: SettingsPath, value: T) => void; 
  batchUpdate: (updates: Array<{path: SettingsPath, value: any}>) => void; 
  flushPendingUpdates: () => Promise<void>; 
  
  
  resetSettings: () => Promise<void>; 
  exportSettings: () => Promise<string>; 
  importSettings: (jsonString: string) => Promise<void>; 
  
  
  updateComputeMode: (mode: string) => void;
  updateClustering: (config: ClusteringConfig) => void;
  updateConstraints: (constraints: ConstraintConfig[]) => void;
  updatePhysics: (graphName: string, params: Partial<GPUPhysicsParams>) => void;
  updateWarmupSettings: (settings: WarmupSettings) => void;
  
  
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
  
  
  restLength: number;
  repulsionCutoff: number;
  repulsionSofteningEpsilon: number;
  centerGravityK: number;
  gridCellSize: number;
  featureFlags: number;
  
  
  warmupIterations: number;
  coolingRate: number;
  
  
  enableBounds?: boolean;
  boundsSize?: number;
  boundaryDamping?: number;
  collisionRadius?: number;
  
  
  iterations?: number;
  massScale?: number;
  updateThreshold?: number;
  
  
  
  boundaryExtremeMultiplier?: number;
  
  boundaryExtremeForceMultiplier?: number;
  
  boundaryVelocityDamping?: number;
  
  maxForce?: number;
  
  seed?: number;
  
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
  warmupIterations?: number; 
  coolingRate?: number; 
}


export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      partialSettings: {},
      settings: {}, 
      loadedPaths: new Set(),
      loadingSections: new Set(),
      initialized: false,
      authenticated: false,
      user: null,
      isPowerUser: false,
      subscribers: new Map(),

      initialize: async () => {
        try {
          console.log('[SettingsStore] Starting initialization with essential paths');
          if (debugState.isEnabled()) {
            logger.info('Initializing settings store with essential paths only')
          }

          
          await waitForAuthReady();

          const isAuthenticated = nostrAuth.isAuthenticated();
          const user = nostrAuth.getCurrentUser();

          logger.info('Settings initialization with auth state', {
            authenticated: isAuthenticated,
            user: user?.pubkey?.slice(0, 8) + '...',
          });

          
          console.log('[SettingsStore] Calling settingsApi.getSettingsByPaths');
          const essentialSettings = await settingsApi.getSettingsByPaths(ESSENTIAL_PATHS);
          console.log('[SettingsStore] Essential settings loaded successfully');

          if (debugState.isEnabled()) {
            logger.info('Essential settings loaded:', { essentialSettings })
          }

          set(state => ({
            partialSettings: essentialSettings as DeepPartial<Settings>,
            settings: essentialSettings as DeepPartial<Settings>, 
            loadedPaths: new Set(ESSENTIAL_PATHS),
            initialized: true,
            authenticated: isAuthenticated,
            user: user ? { isPowerUser: user.isPowerUser, pubkey: user.pubkey } : null,
            isPowerUser: user?.isPowerUser || false
          }));

          
          autoSaveManager.setInitialized(true);

          if (debugState.isEnabled()) {
            logger.info('Settings store initialized with essential paths')
          }

        } catch (error) {
          console.error('[SettingsStore] Failed to initialize:', error);
          logger.error('Failed to initialize settings store:', createErrorMetadata(error))
          set({ initialized: false })
          throw error
        }
      },

      setAuthenticated: (authenticated: boolean) => set({ authenticated }),

      setUser: (user: { isPowerUser: boolean; pubkey: string } | null) => set({
        user,
        isPowerUser: user?.isPowerUser || false
      }),

      notifyViewportUpdate: (path: SettingsPath) => {
        
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

      
      get: <T>(path: SettingsPath): T | undefined => {
        const { partialSettings, loadedPaths } = get();

        if (!path?.trim()) {
          return partialSettings as unknown as T;
        }

        
        const isPathLoaded = loadedPaths.has(path) || 
          [...loadedPaths].some(loadedPath => 
            path.startsWith(loadedPath + '.') || loadedPath.startsWith(path + '.')
          );

        if (!isPathLoaded) {
          if (debugState.isEnabled()) {
            logger.warn(`Accessing unloaded path: ${path} - path should be loaded before access`);
          }
          
          
          return undefined as unknown as T;
        }

        
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

      
      set: <T>(path: SettingsPath, value: T) => {
        if (!path?.trim()) {
          throw new Error('Path cannot be empty');
        }

        
        set(state => {
          const newPartialSettings = { ...state.partialSettings };
          setNestedValue(newPartialSettings, path, value);
          const newLoadedPaths = new Set(state.loadedPaths);
          newLoadedPaths.add(path);

          return {
            partialSettings: newPartialSettings,
            settings: newPartialSettings, 
            loadedPaths: newLoadedPaths
          };
        });

        
        settingsApi.updateSettingByPath(path, value).catch(error => {
          logger.error(`Failed to update setting ${path}:`, createErrorMetadata(error));
        });

        if (debugState.isEnabled()) {
          logger.info('Setting updated:', { path, value });
        }
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

        
        if (immediate && get().initialized) {
          callback()
        }

        
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

      
      ensureLoaded: async (paths: string[]): Promise<void> => {
        const { loadedPaths } = get();
        const unloadedPaths = paths.filter(path => !loadedPaths.has(path));
        
        if (unloadedPaths.length === 0) {
          return; 
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
              settings: newPartialSettings, 
              loadedPaths: newLoadedPaths
            };
          });

          if (debugState.isEnabled()) {
            logger.info('Paths loaded on demand:', { paths: unloadedPaths });
          }
        } catch (error) {
          logger.error('Failed to load paths:', createErrorMetadata(error));
          throw error;
        }
      },

      
      loadSection: async (section: string): Promise<void> => {
        const { loadingSections } = get();
        if (loadingSections.has(section)) {
          return; 
        }

        const sectionPaths = getSectionPaths(section);
        if (sectionPaths.length === 0) {
          logger.warn(`Unknown section: ${section}`);
          return;
        }

        
        set(state => ({
          loadingSections: new Set(state.loadingSections).add(section)
        }));

        try {
          await get().ensureLoaded(sectionPaths);

          if (debugState.isEnabled()) {
            logger.info(`Section loaded: ${section}`, { paths: sectionPaths });
          }
        } finally {
          
          set(state => {
            const newLoadingSections = new Set(state.loadingSections);
            newLoadingSections.delete(section);
            return { loadingSections: newLoadingSections };
          });
        }
      },

      
      isLoaded: (path: SettingsPath): boolean => {
        const { loadedPaths } = get();
        return loadedPaths.has(path);
      },

      
      updateSettings: (updater: (draft: DeepPartial<Settings>) => void): void => {
        const { partialSettings } = get();
        
        
        const newSettings = produce(partialSettings, updater);
        
        
        const changedPaths = findChangedPaths(partialSettings, newSettings);
        
        if (changedPaths.length === 0) {
          return; 
        }

        
        set(state => {
          return {
            partialSettings: newSettings,
            settings: newSettings, 
            
            loadedPaths: new Set([...state.loadedPaths, ...changedPaths])
          };
        });

        
        changedPaths.forEach(path => {
          const pathParts = path.split('.');
          let current: any = newSettings;
          for (const part of pathParts) {
            current = current[part];
          }
          settingsApi.updateSettingByPath(path, current).catch(error => {
            logger.error(`Failed to update setting ${path}:`, createErrorMetadata(error));
          });
        });

        if (debugState.isEnabled()) {
          logger.info('Settings updated via updateSettings:', { changedPaths });
        }

        
        const state = get();
        
        
        const viewportUpdated = changedPaths.some(path => isViewportSetting(path));
        
        if (viewportUpdated) {
          
          state.notifyViewportUpdate('viewport.update');
          
          if (debugState.isEnabled()) {
            logger.info('Viewport settings updated, triggering immediate update', { 
              viewportPaths: changedPaths.filter(path => isViewportSetting(path))
            });
          }
        }

        
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
      },

      
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

      updatePhysics: (graphName: string, params: Partial<GPUPhysicsParams>) => {
        const state = get();
        
        
        const validatedParams = { ...params };
        
        
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
        
        
        if (validatedParams.arrow_size !== undefined) {
          validatedParams.arrow_size = Math.max(0.01, Math.min(5.0, validatedParams.arrow_size));
        }
        if (validatedParams.arrowSize !== undefined) {
          validatedParams.arrowSize = Math.max(0.01, Math.min(5.0, validatedParams.arrowSize));
        }
        if (validatedParams.base_width !== undefined) {
          validatedParams.base_width = Math.max(0.01, Math.min(5.0, validatedParams.base_width));
        }
        if (validatedParams.baseWidth !== undefined) {
          validatedParams.baseWidth = Math.max(0.01, Math.min(5.0, validatedParams.baseWidth));
        }
        
        
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
          if (!draft.visualisation) draft.visualisation = {};
          if (!draft.visualisation.graphs) draft.visualisation.graphs = {};
          
          const graphs = draft.visualisation.graphs as any;
          if (!graphs[graphName]) graphs[graphName] = {};
          if (!graphs[graphName].physics) graphs[graphName].physics = {};
          
          const graphSettings = graphs[graphName];
          if (graphSettings && graphSettings.physics) {
            Object.assign(graphSettings.physics, validatedParams);
            
            if (debugState.isEnabled()) {
              logger.info('Physics parameters updated:', {
                graphName,
                updatedParams: validatedParams,
                newPhysicsState: graphSettings.physics
              });
            }
          }
        });
        
        
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

      
      notifyPhysicsUpdate: (graphName: string, params: Partial<GPUPhysicsParams>) => {
        if (typeof window !== 'undefined') {
          try {
            
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
          
          
          const event = new CustomEvent('physicsParametersUpdated', {
            detail: { graphName, params }
          });
          window.dispatchEvent(event);
        }
      },

      
      
      getByPath: async <T>(path: SettingsPath): Promise<T> => {
        try {
          const value = await settingsApi.getSettingByPath(path);
          return value;
        } catch (error) {
          logger.error(`Failed to get setting by path ${path}:`, createErrorMetadata(error));
          
          return get().get(path);
        }
      },
      
      setByPath: <T>(path: SettingsPath, value: T) => {
        const state = get();

        
        state.set(path, value);

        
        settingsApi.updateSettingByPath(path, value).catch(error => {
          logger.error(`Failed to update setting ${path}:`, createErrorMetadata(error));
        });
      },
      
      batchUpdate: (updates: Array<{path: SettingsPath, value: any}>) => {
        const state = get();

        
        updates.forEach(({ path, value }) => {
          state.set(path, value);
        });

        
        settingsApi.updateSettingsByPaths(updates.map(u => ({ path: u.path, value: u.value }))).catch(error => {
          logger.error('Failed to batch update settings:', createErrorMetadata(error));
        });
      },
      
      flushPendingUpdates: async (): Promise<void> => {
        
        await settingsApi.flushPendingUpdates();
      },
      
      
      resetSettings: async (): Promise<void> => {
        try {
          
          await settingsApi.resetSettings();
          
          
          set({
            partialSettings: {},
            settings: {}, 
            loadedPaths: new Set()
          });
          
          
          await get().initialize();
          
          logger.info('Settings reset to defaults and essential paths reloaded');
        } catch (error) {
          logger.error('Failed to reset settings:', createErrorMetadata(error));
          throw error;
        }
      },
      
      exportSettings: async (): Promise<string> => {
        const { partialSettings, loadedPaths } = get();
        
        try {
          
          if (loadedPaths.size === ESSENTIAL_PATHS.length) {
            logger.info('Only essential settings loaded, fetching all settings for export...');
            
            
            const allPaths = getAllAvailableSettingsPaths();
            const allSettings = await settingsApi.getSettingsByPaths(allPaths);
            
            return settingsApi.exportSettings(allSettings as Settings);
          } else {
            
            return settingsApi.exportSettings(partialSettings as Settings);
          }
        } catch (error) {
          logger.error('Failed to export settings:', createErrorMetadata(error));
          throw error;
        }
      },
      
      importSettings: async (jsonString: string): Promise<void> => {
        try {
          
          const importedSettings = settingsApi.importSettings(jsonString);
          
          
          const allPaths = getAllSettingsPaths(importedSettings);
          const updates: Array<{path: string, value: any}> = [];
          
          for (const path of allPaths) {
            const value = path.split('.').reduce((obj, key) => obj?.[key], importedSettings);
            if (value !== undefined) {
              updates.push({ path, value });
            }
          }
          
          
          get().batchUpdate(updates);
          
          logger.info(`Successfully imported ${updates.length} settings using path-based updates`);
        } catch (error) {
          logger.error('Failed to import settings:', createErrorMetadata(error));
          throw error;
        }
      },
    }),
    {
      name: 'graph-viz-settings-v2',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        
        authenticated: state.authenticated,
        user: state.user,
        isPowerUser: state.isPowerUser,
        
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
        
        paths.push(...getAllSettingsPaths(value, currentPath));
      } else {
        
        paths.push(currentPath);
      }
    }
  }
  
  return paths;
}

// Helper function to get all available settings paths for comprehensive operations
function getAllAvailableSettingsPaths(): string[] {
  
  
  return [
    
    ...ESSENTIAL_PATHS,
    
    
    'visualisation.rendering.ambientLightIntensity',
    'visualisation.rendering.backgroundColor', 
    'visualisation.rendering.directionalLightIntensity',
    'visualisation.rendering.enableAmbientOcclusion',
    'visualisation.rendering.enableAntialiasing',
    'visualisation.rendering.enableShadows',
    'visualisation.rendering.environmentIntensity',
    'visualisation.rendering.shadowMapSize',
    'visualisation.rendering.shadowBias',
    
    
    'visualisation.graphs.logseq.nodes',
    'visualisation.graphs.logseq.edges', 
    'visualisation.graphs.logseq.labels',
    'visualisation.graphs.logseq.physics',
    'visualisation.graphs.visionflow.nodes',
    'visualisation.graphs.visionflow.edges',
    'visualisation.graphs.visionflow.labels', 
    'visualisation.graphs.visionflow.physics',
    
    
    'visualisation.glow.enabled',
    'visualisation.glow.intensity',
    'visualisation.glow.radius',
    'visualisation.glow.threshold',
    'visualisation.hologram.ringCount',
    'visualisation.hologram.ringColor',
    'visualisation.hologram.globalRotationSpeed',
    
    
    'xr.enableHandTracking',
    'xr.enableHaptics',
    'xr.quality',
    
    
    'system.performance.maxFPS',
    'system.performance.enableVSync',
    'system.websocket.url',
    'system.websocket.protocol',
    
    
  ];
}

// Export for testing and direct access
export const settingsStoreUtils = {
  getSectionPaths,
  setNestedValue,
  getAllSettingsPaths,
  getAllAvailableSettingsPaths
};
