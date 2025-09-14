import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { Settings, SettingsPath, DeepPartial } from '../features/settings/config/settings'
import { createLogger, createErrorMetadata } from '../utils/logger'
import { debugState } from '../utils/clientDebugState'
import { produce } from 'immer';
import { toast } from '../features/design-system/components/Toast';
import { isViewportSetting } from '../features/settings/config/viewportSettings';
import { settingsApi, BatchOperation } from '../api/settingsApi';
import { AutoSaveManager } from './autoSaveManager';



const logger = createLogger('SettingsStore')

// Create AutoSaveManager instance for debounced batch saving with retry logic
const autoSaveManager = new AutoSaveManager();

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
  // Add physics settings so control center has them on startup
  'visualisation.graphs.logseq.physics',
  'visualisation.graphs.visionflow.physics'
];


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

interface SettingsState {
  // Partial state management - only holds what's been loaded
  partialSettings: DeepPartial<Settings>
  loadedPaths: Set<string> // Track which paths have been loaded
  loadingSections: Set<string> // Track sections currently being loaded
  
  // Computed getter for backward compatibility with components
  settings: DeepPartial<Settings> // Maps to partialSettings for component access
  
  initialized: boolean
  authenticated: boolean
  user: { isPowerUser: boolean; pubkey: string } | null
  isPowerUser: boolean // Direct access to power user state
  subscribers: Map<string, Set<() => void>>

  // Actions
  initialize: () => Promise<void>
  setAuthenticated: (authenticated: boolean) => void
  setUser: (user: { isPowerUser: boolean; pubkey: string } | null) => void
  get: <T>(path: SettingsPath) => T
  set: <T>(path: SettingsPath, value: T) => void
  subscribe: (path: SettingsPath, callback: () => void, immediate?: boolean) => () => void;
  unsubscribe: (path: SettingsPath, callback: () => void) => void;
  updateSettings: (updater: (draft: Settings) => void) => void;
  notifyViewportUpdate: (path: SettingsPath) => void; // For real-time viewport updates
  
  // Lazy loading - load settings on demand
  ensureLoaded: (paths: string[]) => Promise<void>
  loadSection: (section: string) => Promise<void>
  isLoaded: (path: SettingsPath) => boolean
  
  // New path-based methods for better performance
  getByPath: <T>(path: SettingsPath) => Promise<T>; // Async get from server
  setByPath: <T>(path: SettingsPath, value: T) => void; // Immediate local + debounced server
  batchUpdate: (updates: Array<{path: SettingsPath, value: any}>) => void; // Batch operations
  flushPendingUpdates: () => Promise<void>; // Force immediate server sync
  
  // Settings management methods using path-based API
  resetSettings: () => Promise<void>; // Reset to defaults and reload essential paths
  exportSettings: () => Promise<string>; // Export current loaded settings as JSON
  importSettings: (jsonString: string) => Promise<void>; // Import settings using path-based updates
  
  // GPU-specific methods
  updateComputeMode: (mode: string) => void;
  updateClustering: (config: ClusteringConfig) => void;
  updateConstraints: (constraints: ConstraintConfig[]) => void;
  updatePhysics: (graphName: string, params: Partial<GPUPhysicsParams>) => void;
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


export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      partialSettings: {},
      settings: {}, // Initialize settings as alias to partialSettings
      loadedPaths: new Set(),
      loadingSections: new Set(),
      initialized: false,
      authenticated: false,
      user: null,
      isPowerUser: false,
      subscribers: new Map(),

      initialize: async () => {
        try {
          if (debugState.isEnabled()) {
            logger.info('Initializing settings store with essential paths only')
          }

          // Load only essential settings for fast startup
          const essentialSettings = await settingsApi.getSettingsByPaths(ESSENTIAL_PATHS);

          if (debugState.isEnabled()) {
            logger.info('Essential settings loaded:', { essentialSettings })
          }

          set(state => ({
            partialSettings: essentialSettings as DeepPartial<Settings>,
            settings: essentialSettings as DeepPartial<Settings>, // Keep settings in sync
            loadedPaths: new Set(ESSENTIAL_PATHS),
            initialized: true
          }));

          // Initialize AutoSaveManager now that store is ready
          autoSaveManager.setInitialized(true);

          if (debugState.isEnabled()) {
            logger.info('Settings store initialized with essential paths')
          }

        } catch (error) {
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

      // Path-based setter - updates partial settings and marks paths as loaded
      set: <T>(path: SettingsPath, value: T) => {
        if (!path?.trim()) {
          throw new Error('Path cannot be empty');
        }

        // Update local state immediately
        set(state => {
          const newPartialSettings = { ...state.partialSettings };
          setNestedValue(newPartialSettings, path, value);
          const newLoadedPaths = new Set(state.loadedPaths);
          newLoadedPaths.add(path);

          return {
            partialSettings: newPartialSettings,
            settings: newPartialSettings, // Keep settings in sync
            loadedPaths: newLoadedPaths
          };
        });

        // Schedule server update (will be batched and debounced)
        autoSaveManager.queueChange(path, value);

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
              settings: newPartialSettings, // Keep settings in sync
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
            logger.info(`Section loaded: ${section}`, { paths: sectionPaths });
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

      // Update settings using immer-style updater (synchronous for immediate UI updates)
      updateSettings: (updater: (draft: DeepPartial<Settings>) => void): void => {
        const { partialSettings } = get();
        
        // Use produce to create immutable update
        const newSettings = produce(partialSettings, updater);
        
        // Check what paths changed
        const changedPaths = findChangedPaths(partialSettings, newSettings);
        
        if (changedPaths.length === 0) {
          return; // No changes
        }

        // Update state and queue for batch update
        set(state => {
          return {
            partialSettings: newSettings,
            settings: newSettings, // Keep settings in sync
            // Add changed paths to loaded paths
            loadedPaths: new Set([...state.loadedPaths, ...changedPaths])
          };
        });

        // Schedule batch updates to server for all changed paths
        changedPaths.forEach(path => {
          const pathParts = path.split('.');
          let current: any = newSettings;
          for (const part of pathParts) {
            current = current[part];
          }
          autoSaveManager.queueChange(path, current);
        });

        if (debugState.isEnabled()) {
          logger.info('Settings updated via updateSettings:', { changedPaths });
        }

        // Handle viewport updates and subscribers...
        const state = get();
        
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

      updatePhysics: (graphName: string, params: Partial<GPUPhysicsParams>) => {
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
        
        // Validate edge/arrow settings to prevent server validation errors
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

      
      // New path-based methods for enhanced performance
      getByPath: async <T>(path: SettingsPath): Promise<T> => {
        try {
          const value = await settingsApi.getSettingByPath(path);
          return value;
        } catch (error) {
          logger.error(`Failed to get setting by path ${path}:`, createErrorMetadata(error));
          // Fallback to local state
          return get().get(path);
        }
      },
      
      setByPath: <T>(path: SettingsPath, value: T) => {
        const state = get();
        
        // Update local state immediately for responsive UI
        state.set(path, value);
        
        // Schedule server update (will be batched and debounced)
        autoSaveManager.queueChange(path, value);
      },
      
      batchUpdate: (updates: Array<{path: SettingsPath, value: any}>) => {
        const state = get();
        
        // Update all local state immediately
        updates.forEach(({ path, value }) => {
          state.set(path, value);
        });
        
        // Schedule all server updates (will be batched and debounced)
        const changes = new Map();
        updates.forEach(({ path, value }) => {
          changes.set(path, value);
        });
        autoSaveManager.queueChanges(changes);
      },
      
      flushPendingUpdates: async (): Promise<void> => {
        await autoSaveManager.forceFlush();
      },
      
      // Settings management methods using path-based API
      resetSettings: async (): Promise<void> => {
        try {
          // Call the server's reset endpoint
          await settingsApi.resetSettings();
          
          // Clear local state
          set({
            partialSettings: {},
            settings: {}, // Keep settings in sync
            loadedPaths: new Set()
          });
          
          // Reload essential settings
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
          // If we only have partial settings, fetch all settings paths first
          if (loadedPaths.size === ESSENTIAL_PATHS.length) {
            logger.info('Only essential settings loaded, fetching all settings for export...');
            
            // Get all available settings paths by fetching a comprehensive list
            const allPaths = getAllAvailableSettingsPaths();
            const allSettings = await settingsApi.getSettingsByPaths(allPaths);
            
            return settingsApi.exportSettings(allSettings as Settings);
          } else {
            // Export currently loaded settings
            return settingsApi.exportSettings(partialSettings as Settings);
          }
        } catch (error) {
          logger.error('Failed to export settings:', createErrorMetadata(error));
          throw error;
        }
      },
      
      importSettings: async (jsonString: string): Promise<void> => {
        try {
          // Parse and validate the imported settings
          const importedSettings = settingsApi.importSettings(jsonString);
          
          // Extract all paths and values from the imported settings
          const allPaths = getAllSettingsPaths(importedSettings);
          const updates: Array<{path: string, value: any}> = [];
          
          for (const path of allPaths) {
            const value = path.split('.').reduce((obj, key) => obj?.[key], importedSettings);
            if (value !== undefined) {
              updates.push({ path, value });
            }
          }
          
          // Apply all updates using our batch update system
          get().batchUpdate(updates);
          
          // Force immediate flush to server
          await get().flushPendingUpdates();
          
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

// Helper function to get all available settings paths for comprehensive operations
function getAllAvailableSettingsPaths(): string[] {
  // This is a comprehensive list of all known settings paths in the system
  // These should be kept in sync with the actual settings structure
  return [
    // System settings
    ...ESSENTIAL_PATHS,
    
    // Visualization settings
    'visualisation.rendering.ambientLightIntensity',
    'visualisation.rendering.backgroundColor', 
    'visualisation.rendering.directionalLightIntensity',
    'visualisation.rendering.enableAmbientOcclusion',
    'visualisation.rendering.enableAntialiasing',
    'visualisation.rendering.enableShadows',
    'visualisation.rendering.environmentIntensity',
    'visualisation.rendering.shadowMapSize',
    'visualisation.rendering.shadowBias',
    
    // Graph-specific settings
    'visualisation.graphs.logseq.nodes',
    'visualisation.graphs.logseq.edges', 
    'visualisation.graphs.logseq.labels',
    'visualisation.graphs.logseq.physics',
    'visualisation.graphs.visionflow.nodes',
    'visualisation.graphs.visionflow.edges',
    'visualisation.graphs.visionflow.labels', 
    'visualisation.graphs.visionflow.physics',
    
    // Effects
    'visualisation.glow.enabled',
    'visualisation.glow.intensity',
    'visualisation.glow.radius',
    'visualisation.glow.threshold',
    'visualisation.hologram.ringCount',
    'visualisation.hologram.ringColor',
    'visualisation.hologram.globalRotationSpeed',
    
    // XR settings
    'xr.enableHandTracking',
    'xr.enableHaptics',
    'xr.quality',
    
    // Additional system and performance settings
    'system.performance.maxFPS',
    'system.performance.enableVSync',
    'system.websocket.url',
    'system.websocket.protocol',
    
    // Add more paths as needed based on the actual settings structure
  ];
}

// Export for testing and direct access
export const settingsStoreUtils = {
  autoSaveManager,
  getSectionPaths,
  setNestedValue,
  getAllSettingsPaths,
  getAllAvailableSettingsPaths
};
