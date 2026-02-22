import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { Settings, SettingsPath, DeepPartial } from '../features/settings/config/settings'
import { createLogger } from '../utils/loggerConfig'
import { createErrorMetadata } from '../utils/loggerConfig'
import { debugState } from '../utils/clientDebugState'
import { produce } from 'immer';
import { toast } from '../features/design-system/components/Toast';
import { isViewportSetting } from '../features/settings/config/viewportSettings';
import { settingsApi } from '../api/settingsApi';
import { nostrAuth } from '../services/nostrAuthService';
import { autoSaveManager } from './autoSaveManager';



const logger = createLogger('SettingsStore')

// Module-level subscribers map (kept outside Zustand to avoid serialization/diffing overhead)
const subscribersMap = new Map<string, Set<() => void>>();

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


      if (elapsed >= maxWaitMs || !localStorage.getItem('nostr_user')) {
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

  // Graph-type visual settings - needed for per-type rendering
  'visualisation.graphTypeVisuals',

  // Node filtering settings - needed for visibility filtering
  'nodeFilter.enabled',
  'nodeFilter.qualityThreshold',
  'nodeFilter.authorityThreshold',
  'nodeFilter.filterByQuality',
  'nodeFilter.filterByAuthority',
  'nodeFilter.filterMode'
];


// Deep merge two settings objects. overlay values win over base values.
// Used during initialization to merge server defaults (base) with localStorage (overlay)
// so that user customizations survive page reloads.
function deepMergeSettings(
  base: Record<string, unknown>,
  overlay: Record<string, unknown>,
): Record<string, unknown> {
  const result = { ...base };
  for (const [key, value] of Object.entries(overlay)) {
    if (value === undefined) continue;
    const baseVal = result[key];
    if (
      value !== null && typeof value === 'object' && !Array.isArray(value) &&
      baseVal !== null && typeof baseVal === 'object' && !Array.isArray(baseVal)
    ) {
      result[key] = deepMergeSettings(
        baseVal as Record<string, unknown>,
        value as Record<string, unknown>,
      );
    } else {
      result[key] = value;
    }
  }
  return result;
}

// Helper function to find changed paths between two objects
function findChangedPaths(oldObj: unknown, newObj: unknown, path: string = ''): string[] {
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


  const oldRecord = oldObj as Record<string, unknown>;
  const newRecord = newObj as Record<string, unknown>;
  const allKeys = new Set([...Object.keys(oldRecord), ...Object.keys(newRecord)]);

  for (const key of allKeys) {
    const currentPath = path ? `${path}.${key}` : key;
    const oldValue = oldRecord[key];
    const newValue = newRecord[key];

    if (typeof oldValue === 'object' && typeof newValue === 'object' && oldValue !== null && newValue !== null) {

      changedPaths.push(...findChangedPaths(oldValue, newValue, currentPath));
    } else if (oldValue !== newValue) {

      changedPaths.push(currentPath);
    }
  }

  return changedPaths;
}

export interface SettingsState {

  partialSettings: DeepPartial<Settings>
  loadedPaths: Set<string>
  loadingSections: Set<string>


  settings: DeepPartial<Settings>

  initialized: boolean
  authenticated: boolean
  user: { isPowerUser: boolean; pubkey: string } | null
  isPowerUser: boolean


  initialize: () => Promise<void>
  setAuthenticated: (authenticated: boolean) => void
  setUser: (user: { isPowerUser: boolean; pubkey: string } | null) => void
  get: <T>(path: SettingsPath) => T | undefined
  set: <T>(path: SettingsPath, value: T, skipServerSync?: boolean) => void
  subscribe: (path: SettingsPath, callback: () => void, immediate?: boolean) => () => void;
  unsubscribe: (path: SettingsPath, callback: () => void) => void;
  updateSettings: (updater: (draft: Settings) => void) => void;
  notifyViewportUpdate: (path: SettingsPath) => void;


  ensureLoaded: (paths: string[]) => Promise<void>
  loadSection: (section: string) => Promise<void>
  isLoaded: (path: SettingsPath) => boolean


  getByPath: <T>(path: SettingsPath) => Promise<T>;
  setByPath: <T>(path: SettingsPath, value: T) => void;
  batchUpdate: (updates: Array<{path: SettingsPath, value: unknown}>) => void;
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

      initialize: async () => {
        try {
          logger.info('[SettingsStore] Starting initialization with essential paths');
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


          logger.info('[SettingsStore] Calling settingsApi.getSettingsByPaths');
          const essentialSettings = await settingsApi.getSettingsByPaths(ESSENTIAL_PATHS);
          logger.info('[SettingsStore] Essential settings loaded successfully');

          if (debugState.isEnabled()) {
            logger.info('Essential settings loaded:', { essentialSettings })
          }

          set(state => {
            // Deep merge: server-fetched settings provide defaults and authoritative
            // values as the base. localStorage-persisted customizations overlay on top
            // so user tweaks (graphTypeVisuals, glow, edges, etc.) survive reloads.
            const merged = deepMergeSettings(
              essentialSettings as Record<string, unknown>,
              state.partialSettings as Record<string, unknown>,
            ) as DeepPartial<Settings>;
            return {
              partialSettings: merged,
              settings: merged,
              loadedPaths: new Set([...state.loadedPaths, ...ESSENTIAL_PATHS]),
              initialized: true,
              authenticated: isAuthenticated,
              user: user ? { isPowerUser: user.isPowerUser, pubkey: user.pubkey } : null,
              isPowerUser: user?.isPowerUser || false,
            };
          });


          autoSaveManager.setInitialized(true);

          if (debugState.isEnabled()) {
            logger.info('Settings store initialized with essential paths')
          }

        } catch (error) {
          logger.error('[SettingsStore] Failed to initialize:', createErrorMetadata(error));
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
        const callbacks = subscribersMap.get('viewport.update');
        if (callbacks) {
          Array.from(callbacks).forEach(callback => {
            try {
              callback();
            } catch (error) {
              logger.error(`Error in viewport update subscriber:`, createErrorMetadata(error));
            }
          });
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
        let current: unknown = partialSettings;

        for (const part of pathParts) {
          if (current == null || typeof current !== 'object' || !(part in (current as Record<string, unknown>))) {
            return undefined;
          }
          current = (current as Record<string, unknown>)[part];
        }

        return current as T;
      },


      set: <T>(path: SettingsPath, value: T, skipServerSync: boolean = false) => {
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

        if (!skipServerSync) {
          // Route through autoSaveManager for 500ms debounce — prevents flooding
          // the backend when sliders fire onValueChange 60+/sec during drag.
          autoSaveManager.queueChange(path, value);
        }

        if (debugState.isEnabled()) {
          logger.info('Setting updated:', { path, value });
        }
      },

      subscribe: (path: SettingsPath, callback: () => void, immediate: boolean = true) => {
        if (!subscribersMap.has(path)) {
          subscribersMap.set(path, new Set());
        }
        subscribersMap.get(path)!.add(callback);

        if (immediate && get().initialized) {
          callback();
        }

        return () => get().unsubscribe(path, callback);
      },

      unsubscribe: (path: SettingsPath, callback: () => void) => {
        if (subscribersMap.has(path)) {
          const callbacks = subscribersMap.get(path)!;
          callbacks.delete(callback);
          if (callbacks.size === 0) {
            subscribersMap.delete(path);
          }
        }
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


      updateSettings: (updater: (draft: Settings) => void): void => {
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


        // Route through autoSaveManager for 500ms debounce — prevents flooding
        // the backend when sliders fire onValueChange 60+/sec during drag.
        const batchChanges = new Map<string, any>();
        changedPaths.forEach(path => {
          const pathParts = path.split('.');
          let current: unknown = newSettings;
          for (const part of pathParts) {
            current = (current as Record<string, unknown>)[part];
          }
          batchChanges.set(path, current);
        });
        autoSaveManager.queueChanges(batchChanges);

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


        // Only notify subscribers registered to paths that actually changed.
        // For each changedPath, fire callbacks on exact matches and on any
        // registered prefix (parent path) so that e.g. a subscriber on
        // "visualisation" is notified when "visualisation.glow.intensity" changes.
        const matchedCallbacks = new Set<() => void>();
        for (const changedPath of changedPaths) {
          // Exact match
          const exact = subscribersMap.get(changedPath);
          if (exact) {
            exact.forEach(cb => matchedCallbacks.add(cb));
          }
          // Prefix/parent matches: a subscriber on "a.b" should fire for "a.b.c.d"
          // and a subscriber on "a.b.c.d" should fire for "a.b" (child matches parent)
          subscribersMap.forEach((callbacks, subscribedPath) => {
            if (
              subscribedPath !== changedPath &&
              (changedPath.startsWith(subscribedPath + '.') ||
               subscribedPath.startsWith(changedPath + '.'))
            ) {
              callbacks.forEach(cb => matchedCallbacks.add(cb));
            }
          });
        }

        matchedCallbacks.forEach(callback => {
          try {
            callback();
          } catch (error) {
            logger.error('Error in settings subscriber during updateSettings:', createErrorMetadata(error));
          }
        });
      },


      updateComputeMode: (mode: string) => {
        const state = get();
        state.updateSettings((draft: Settings) => {
          const d = draft as unknown as Record<string, unknown>;
          if (!d.dashboard) {
            d.dashboard = { computeMode: '' };
          }
          (d.dashboard as Record<string, unknown>).computeMode = mode;
        });
      },

      updateClustering: (config: ClusteringConfig) => {
        const state = get();
        state.updateSettings((draft: Settings) => {
          const d = draft as unknown as Record<string, unknown>;
          if (!d.analytics) {
            d.analytics = {};
          }
          const analytics = d.analytics as Record<string, unknown>;
          if (!analytics.clustering) {
            analytics.clustering = {};
          }
          Object.assign(analytics.clustering as Record<string, unknown>, config);
        });
      },

      updateConstraints: (constraints: ConstraintConfig[]) => {
        const state = get();
        state.updateSettings((draft: Settings) => {
          const d = draft as unknown as Record<string, unknown>;
          if (!d.developer) {
            d.developer = {};
          }
          const developer = d.developer as Record<string, unknown>;
          if (!developer.constraints) {
            developer.constraints = {};
          }
          (developer.constraints as Record<string, unknown>).active = constraints;
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


        const extParams = validatedParams as Record<string, unknown>;
        if (extParams.arrow_size !== undefined) {
          extParams.arrow_size = Math.max(0.01, Math.min(5.0, extParams.arrow_size as number));
        }
        if (extParams.arrowSize !== undefined) {
          extParams.arrowSize = Math.max(0.01, Math.min(5.0, extParams.arrowSize as number));
        }
        if (extParams.base_width !== undefined) {
          extParams.base_width = Math.max(0.01, Math.min(5.0, extParams.base_width as number));
        }
        if (extParams.baseWidth !== undefined) {
          extParams.baseWidth = Math.max(0.01, Math.min(5.0, extParams.baseWidth as number));
        }


        if (validatedParams.springK !== undefined) {
          validatedParams.springK = Math.max(0.001, Math.min(10.0, validatedParams.springK));
        }
        if (validatedParams.repelK !== undefined) {
          validatedParams.repelK = Math.max(0.001, Math.min(500.0, validatedParams.repelK));
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

        state.updateSettings((draft: Settings) => {
          const d = draft as unknown as Record<string, unknown>;
          if (!d.visualisation) d.visualisation = { graphs: {} };
          const vis = d.visualisation as Record<string, unknown>;
          if (!vis.graphs) vis.graphs = {};

          const graphs = vis.graphs as Record<string, unknown>;
          if (!graphs[graphName]) graphs[graphName] = {};
          const graph = graphs[graphName] as Record<string, unknown>;
          if (!graph.physics) graph.physics = {};

          const graphSettings = graphs[graphName] as Record<string, unknown> | undefined;
          if (graphSettings && graphSettings.physics) {
            Object.assign(graphSettings.physics as Record<string, unknown>, validatedParams);

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
        state.updateSettings((draft: Settings) => {
          const d = draft as unknown as Record<string, unknown>;
          if (!d.performance) {
            d.performance = {};
          }
          Object.assign(d.performance as Record<string, unknown>, settings);
        });
      },


      notifyPhysicsUpdate: (graphName: string, params: Partial<GPUPhysicsParams>) => {
        if (typeof window !== 'undefined') {
          const event = new CustomEvent('physicsParametersUpdated', {
            detail: { graphName, params }
          });
          window.dispatchEvent(event);
        }
      },



      getByPath: async <T>(path: SettingsPath): Promise<T> => {
        try {
          const value = await settingsApi.getSettingByPath(path);
          return value as T;
        } catch (error) {
          logger.error(`Failed to get setting by path ${path}:`, createErrorMetadata(error));
          const localValue = get().get<T>(path);
          return localValue as T;
        }
      },

      setByPath: <T>(path: SettingsPath, value: T) => {
        const state = get();

        // Update Zustand state without triggering server sync from set()
        state.set(path, value, true);

        // Single server write from setByPath
        settingsApi.updateSettingByPath(path, value).catch(error => {
          logger.error(`Failed to update setting ${path}:`, createErrorMetadata(error));
        });
      },

      batchUpdate: (updates: Array<{path: SettingsPath, value: unknown}>) => {
        // Accumulate all changes locally without triggering per-key server writes
        set(state => {
          const newPartialSettings = { ...state.partialSettings };
          const newLoadedPaths = new Set(state.loadedPaths);
          for (const { path, value } of updates) {
            setNestedValue(newPartialSettings, path, value);
            newLoadedPaths.add(path);
          }
          return {
            partialSettings: newPartialSettings,
            settings: newPartialSettings,
            loadedPaths: newLoadedPaths
          };
        });

        // Single batched server write
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

            return settingsApi.exportSettings(allSettings);
          } else {

            return settingsApi.exportSettings(partialSettings as unknown as Record<string, unknown>);
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
          const updates: Array<{path: string, value: unknown}> = [];

          for (const path of allPaths) {
            const value = path.split('.').reduce<unknown>((obj, key) => (obj as Record<string, unknown>)?.[key], importedSettings);
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
        // Auth state
        authenticated: state.authenticated,
        user: state.user,
        isPowerUser: state.isPowerUser,
        // Persist ALL settings so visual settings (edges, glow, hologram, graphTypeVisuals, etc.)
        // survive page reloads. The server only handles a subset of categories, so localStorage
        // is the primary persistence layer for client-side visual settings.
        partialSettings: state.partialSettings,
      }),
      merge: (persistedState: unknown, currentState: SettingsState): SettingsState => {
        if (!persistedState) return currentState;
        const persisted = persistedState as Record<string, unknown>;

        // Deep-merge persisted partialSettings into the current state so that
        // server-fetched values take priority during initialize(), but any settings
        // that were only stored locally are restored from localStorage.
        const persistedSettings = (persisted.partialSettings as DeepPartial<Settings>) || {};

        // Reconstruct loadedPaths from the persisted settings keys
        const restoredPaths = new Set<string>();
        const collectPaths = (obj: unknown, prefix: string = '') => {
          if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
            for (const [key, value] of Object.entries(obj as Record<string, unknown>)) {
              const currentPath = prefix ? `${prefix}.${key}` : key;
              if (value && typeof value === 'object' && !Array.isArray(value)) {
                collectPaths(value, currentPath);
              } else {
                restoredPaths.add(currentPath);
              }
            }
          }
        };
        collectPaths(persistedSettings);

        return {
          ...currentState,
          authenticated: (persisted.authenticated as boolean) || false,
          user: (persisted.user as SettingsState['user']) || null,
          isPowerUser: (persisted.isPowerUser as boolean) || false,
          partialSettings: persistedSettings,
          settings: persistedSettings,
          loadedPaths: restoredPaths,
        };
      },
      onRehydrateStorage: () => (state) => {
        if (state) {
          if (debugState.isEnabled()) {
            logger.info('Settings store rehydrated from storage with persisted visual settings');
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
      'visualisation.graphs.logseq.physics'
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
    'graphTypeVisuals': [
      'visualisation.graphTypeVisuals.knowledgeGraph',
      'visualisation.graphTypeVisuals.ontology',
      'visualisation.graphTypeVisuals.agent'
    ],
    'nodes': [
      'visualisation.graphs.logseq.nodes'
    ],
    'edges': [
      'visualisation.graphs.logseq.edges'
    ],
    'labels': [
      'visualisation.graphs.logseq.labels'
    ],
    'gemMaterial': [
      'visualisation.gemMaterial'
    ],
    'sceneEffects': [
      'visualisation.sceneEffects'
    ],
    'clusterHulls': [
      'visualisation.clusterHulls'
    ],
    'hologram': [
      'visualisation.hologram'
    ],
    'animations': [
      'visualisation.animations'
    ],
    'interaction': [
      'visualisation.interaction'
    ],
    'analytics': [
      'analytics.enableMetrics',
      'analytics.updateInterval',
      'analytics.showDegreeDistribution',
      'analytics.showClusteringCoefficient',
      'analytics.showCentrality',
      'analytics.clustering'
    ],
    'qualityGates': [
      'qualityGates'
    ],
    'nodeFilter': [
      'nodeFilter'
    ],
    'constraints': [
      'constraints'
    ]
  };

  return sectionPathMap[section] || [];
}

// Helper function to set nested value by dot notation path
function setNestedValue(obj: Record<string, unknown>, path: string, value: unknown): void {
  const keys = path.split('.');
  let current: Record<string, unknown> = obj;

  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];
    if (!(key in current) || typeof current[key] !== 'object' || current[key] === null) {
      current[key] = {};
    }
    current = current[key] as Record<string, unknown>;
  }

  current[keys[keys.length - 1]] = value;
}

// Helper function to extract all paths from a settings object
function getAllSettingsPaths(obj: unknown, prefix: string = ''): string[] {
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


    'visualisation.glow.enabled',
    'visualisation.glow.intensity',
    'visualisation.glow.radius',
    'visualisation.glow.threshold',
    'visualisation.hologram.ringCount',
    'visualisation.hologram.ringColor',
    'visualisation.hologram.globalRotationSpeed',

    // Graph-type visual settings
    'visualisation.graphTypeVisuals.knowledgeGraph',
    'visualisation.graphTypeVisuals.ontology',
    'visualisation.graphTypeVisuals.agent',

    'xr.enableHandTracking',
    'xr.enableHaptics',
    'xr.quality',

    // Gem material
    'visualisation.gemMaterial',

    // Scene effects (WASM ambient)
    'visualisation.sceneEffects',

    // Cluster hulls
    'visualisation.clusterHulls',

    // Animations
    'visualisation.animations',

    // Interaction / selection highlighting
    'visualisation.interaction',

    // Analytics
    'analytics',

    // Quality gates
    'qualityGates',

    // Node filter
    'nodeFilter',

    // Constraints / LOD
    'constraints',

  ];
}

// Export for testing and direct access
export const settingsStoreUtils = {
  getSectionPaths,
  setNestedValue,
  getAllSettingsPaths,
  getAllAvailableSettingsPaths
};
