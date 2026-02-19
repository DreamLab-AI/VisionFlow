// frontend/src/api/settingsApi.ts
// REAL API client for settings management - NO MOCKS

import axios, { AxiosResponse } from 'axios';
import { nostrAuth } from '../services/nostrAuthService';

// Always use relative paths for API requests. In dev mode Vite proxies /api
// to the backend (http://127.0.0.1:4000). In production the serving proxy
// (nginx / HTTPS bridge) must also proxy /api to the backend.
const API_BASE = '';

// Global NIP-98 auth interceptor for all axios requests
axios.interceptors.request.use(async (config) => {
  if (!nostrAuth.isAuthenticated()) return config;

  const user = nostrAuth.getCurrentUser();
  if (!config.headers) return config;

  if (nostrAuth.isDevMode()) {
    config.headers['Authorization'] = 'Bearer dev-session-token';
    if (user?.pubkey) {
      config.headers['X-Nostr-Pubkey'] = user.pubkey;
    }
  } else if (user?.pubkey) {
    try {
      const fullUrl = new URL(config.url || '', config.baseURL || window.location.origin).href;
      const method = (config.method || 'GET').toUpperCase();
      const body = config.data
        ? (typeof config.data === 'string' ? config.data : JSON.stringify(config.data))
        : undefined;
      const token = await nostrAuth.signRequest(fullUrl, method, body);
      config.headers['Authorization'] = `Nostr ${token}`;
    } catch (e) {
      console.warn('[settingsApi] NIP-98 signing failed:', e);
    }
  }
  return config;
});

// ============================================================================
// Type Definitions (matching Rust backend exactly)
// ============================================================================

export interface PhysicsSettings {
  autoBalance: boolean;
  autoBalanceIntervalMs: number;
  autoBalanceConfig: {
    maxIterations: number;
    threshold: number;
  };
  autoPause: {
    enabled: boolean;
    inactivityThresholdMs: number;
  };
  boundsSize: number;
  separationRadius: number;
  damping: number;
  enableBounds: boolean;
  enabled: boolean;
  iterations: number;
  maxVelocity: number;
  maxForce: number;
  repelK: number;
}

export type PriorityWeighting = 'linear' | 'exponential' | 'quadratic';

export interface ConstraintSettings {
  lodEnabled: boolean;
  farThreshold: number;
  mediumThreshold: number;
  nearThreshold: number;
  priorityWeighting: PriorityWeighting;
  progressiveActivation: boolean;
  activationFrames: number;
}

export interface RenderingSettings {
  ambientLightIntensity: number;
  backgroundColor: string;
  directionalLightIntensity: number;
  enableAmbientOcclusion: boolean;
  enableAntialiasing: boolean;
  enableShadows: boolean;
  environmentIntensity: number;
  shadowMapSize?: string;
  shadowBias?: number;
  context?: string;
}

export interface NodeFilterSettings {
  enabled: boolean;
  qualityThreshold: number;
  authorityThreshold: number;
  filterByQuality: boolean;
  filterByAuthority: boolean;
  filterMode: 'or' | 'and';
}

export interface QualityGateSettings {
  gpuAcceleration: boolean;
  ontologyPhysics: boolean;
  semanticForces: boolean;
  layoutMode: 'force-directed' | 'dag-topdown' | 'dag-radial' | 'dag-leftright' | 'type-clustering';
  showClusters: boolean;
  showAnomalies: boolean;
  showCommunities: boolean;
  ruvectorEnabled: boolean;
  gnnPhysics: boolean;
  minFpsThreshold: number;
  maxNodeCount: number;
  autoAdjust: boolean;
}

export interface AllSettings {
  physics: PhysicsSettings;
  constraints: ConstraintSettings;
  rendering: RenderingSettings;
  nodeFilter: NodeFilterSettings;
  qualityGates: QualityGateSettings;
  visual?: Record<string, any>;
}

export interface SettingsProfile {
  id: number;
  name: string;
  createdAt: string;
  updatedAt: string;
}

export interface SaveProfileRequest {
  name: string;
}

export interface ProfileIdResponse {
  id: number;
}

export interface ErrorResponse {
  error: string;
}

// ============================================================================
// Default settings for visualisation effects (used when API doesn't provide them)
// ============================================================================

const DEFAULT_GLOW_SETTINGS = {
  enabled: true,
  intensity: 0.5,
  radius: 0.3,
  threshold: 0.3,
  diffuseStrength: 0.3,
  atmosphericDensity: 0.2,
  volumetricIntensity: 0.3,
  baseColor: '#ffffff',
  emissionColor: '#00ffff',
  opacity: 1.0,
  pulseSpeed: 1.0,
  flowSpeed: 0.5,
  nodeGlowStrength: 0.6,
  edgeGlowStrength: 0.3,
  environmentGlowStrength: 0.2
};

const DEFAULT_BLOOM_SETTINGS = {
  enabled: true,
  intensity: 0.4,
  threshold: 0.3,
  radius: 0.3,
  strength: 0.4
};

const DEFAULT_HOLOGRAM_SETTINGS = {
  ringCount: 3,
  ringColor: '#00ffff',
  ringOpacity: 0.5,
  sphereSizes: [100, 150] as [number, number],
  globalRotationSpeed: 0.5,
  ringRotationSpeed: 0.5,
};

const DEFAULT_GEM_MATERIAL = {
  ior: 2.42,
  transmission: 0.6,
  clearcoat: 1.0,
  clearcoatRoughness: 0.02,
  emissiveIntensity: 0.3,
  iridescence: 0.3,
};

const DEFAULT_SCENE_EFFECTS = {
  enabled: true,
  particleCount: 128,
  particleOpacity: 0.3,
  particleDrift: 0.5,
  wispsEnabled: true,
  wispCount: 32,
  wispOpacity: 0.4,
  wispDriftSpeed: 1.0,
  fogEnabled: false,
  fogOpacity: 0.05,
  atmosphereResolution: 128,
};

const DEFAULT_CLUSTER_HULLS = {
  enabled: false,
  opacity: 0.08,
  padding: 0.15,
};

const DEFAULT_ANIMATION_SETTINGS = {
  enableMotionBlur: false,
  enableNodeAnimations: true,
  motionBlurStrength: 0.5,
  selectionWaveEnabled: true,
  pulseEnabled: true,
  pulseSpeed: 1.0,
  pulseStrength: 0.5,
  waveSpeed: 1.0,
};

const DEFAULT_INTERACTION_SETTINGS = {
  selectionHighlightColor: '#ffff00',
  selectionEdgeFlow: false,
  selectionEdgeFlowSpeed: 1.0,
  selectionEdgeWidth: 0.5,
  selectionEdgeOpacity: 0.8,
};

const DEFAULT_NODES_SETTINGS = {
  baseColor: '#202724',
  metalness: 0.1,
  opacity: 0.88,
  roughness: 0.6,
  nodeSize: 1.7,
  quality: 'high' as const,
  enableInstancing: true,
  enableMetadataShape: false,
  enableMetadataVisualisation: true
};

const DEFAULT_EDGES_SETTINGS = {
  arrowSize: 0.02,
  baseWidth: 0.61,
  color: '#ff0000',
  enableArrows: false,
  opacity: 0.5,
  widthRange: [0.3, 1.5] as [number, number],
  quality: 'high' as const,
  enableFlowEffect: false,
  flowSpeed: 1.0,
  flowIntensity: 0.5,
  glowStrength: 0.3,
  distanceIntensity: 0.5,
  useGradient: false,
  gradientColors: ['#4a9eff', '#ff4a9e'] as [string, string]
};

const DEFAULT_LABELS_SETTINGS = {
  desktopFontSize: 1.41,
  enableLabels: true,
  labelDistanceThreshold: 500,
  textColor: '#676565',
  textOutlineColor: '#00ff40',
  textOutlineWidth: 0.0074725277,
  textResolution: 32,
  textPadding: 0.3,
  billboardMode: 'camera' as const,
  showMetadata: true,
  maxLabelWidth: 5.0
};

const DEFAULT_GRAPH_TYPE_VISUALS = {
  knowledgeGraph: {
    metalness: 0.6,
    roughness: 0.15,
    glowStrength: 2.5,
    innerGlowIntensity: 0.3,
    facetDetail: 2,
    authorityScaleFactor: 0.5,
    connectionInfluence: 0.4,
    globalScaleMultiplier: 2.5,
    showDomainBadge: true,
    showQualityStars: true,
    showRecencyIndicator: true,
    showConnectionDensity: false,
  },
  ontology: {
    glowStrength: 1.8,
    orbitalRingCount: 8,
    orbitalRingSpeed: 0.5,
    hierarchyScaleFactor: 0.15,
    minScale: 0.4,
    instanceCountInfluence: 0.1,
    depthColorGradient: true,
    showHierarchyBreadcrumb: true,
    showInstanceCount: true,
    showConstraintStatus: false,
    nebulaGlowIntensity: 0.7,
  },
  agent: {
    membraneOpacity: 0.7,
    nucleusGlowIntensity: 0.6,
    breathingSpeed: 1.5,
    breathingAmplitude: 0.4,
    workloadInfluence: 0.3,
    tokenRateInfluence: 100,
    tokenRateCap: 0.5,
    showHealthBar: true,
    showTokenRate: true,
    showTaskCount: false,
    bioluminescentIntensity: 0.6,
  },
};

// ============================================================================
// Transform flat API response to nested client structure
// ============================================================================

/** Deep merge stored server values over local defaults. Stored values win. */
function deepMergeVisual(defaults: Record<string, any>, stored: Record<string, any>): Record<string, any> {
  const result = { ...defaults };
  for (const [key, value] of Object.entries(stored)) {
    if (value && typeof value === 'object' && !Array.isArray(value) &&
        result[key] && typeof result[key] === 'object' && !Array.isArray(result[key])) {
      result[key] = deepMergeVisual(result[key], value);
    } else {
      result[key] = value;
    }
  }
  return result;
}

function transformApiToClientSettings(apiResponse: AllSettings): any {
  // Server-stored visual settings blob — defaults are used as fallback base
  const v = apiResponse.visual || {};
  return {
    visualisation: {
      rendering: apiResponse.rendering || {},
      glow: deepMergeVisual(DEFAULT_GLOW_SETTINGS, v.glow || {}),
      bloom: deepMergeVisual(DEFAULT_BLOOM_SETTINGS, v.bloom || {}),
      hologram: deepMergeVisual(DEFAULT_HOLOGRAM_SETTINGS, v.hologram || {}),
      graphTypeVisuals: deepMergeVisual(DEFAULT_GRAPH_TYPE_VISUALS, v.graphTypeVisuals || {}),
      gemMaterial: deepMergeVisual(DEFAULT_GEM_MATERIAL, v.gemMaterial || {}),
      sceneEffects: deepMergeVisual(DEFAULT_SCENE_EFFECTS, v.sceneEffects || {}),
      clusterHulls: deepMergeVisual(DEFAULT_CLUSTER_HULLS, v.clusterHulls || {}),
      animations: deepMergeVisual(DEFAULT_ANIMATION_SETTINGS, v.animations || {}),
      interaction: deepMergeVisual(DEFAULT_INTERACTION_SETTINGS, v.interaction || {}),
      graphs: {
        logseq: {
          physics: apiResponse.physics || {},
          nodes: deepMergeVisual(DEFAULT_NODES_SETTINGS, v.nodes || {}),
          edges: deepMergeVisual(DEFAULT_EDGES_SETTINGS, v.edges || {}),
          labels: deepMergeVisual(DEFAULT_LABELS_SETTINGS, v.labels || {})
        }
      }
    },
    system: {
      debug: {
        enabled: false,
        enableDataDebug: false,
        enableWebsocketDebug: false,
        logBinaryHeaders: false,
        logFullJson: false
      },
      websocket: {
        reconnectAttempts: 5,
        reconnectDelay: 1000,
        binaryChunkSize: 1024,
        compressionEnabled: true,
        compressionThreshold: 1024,
        updateRate: 60
      },
      persistSettings: true
    },
    xr: {
      enabled: false,
      mode: 'inline' as const,
      enableHandTracking: false,
      enableHaptics: false,
      quality: 'medium' as const
    },
    auth: {
      enabled: false,
      provider: 'nostr' as const,
      required: false
    },
    qualityGates: apiResponse.qualityGates || {
      gpuAcceleration: true,
      ontologyPhysics: false,
      semanticForces: false,
      layoutMode: 'force-directed' as const,
      showClusters: true,
      showAnomalies: true,
      showCommunities: false,
      ruvectorEnabled: false,
      gnnPhysics: false,
      minFpsThreshold: 30,
      maxNodeCount: 100000,
      autoAdjust: true
    },
    nodeFilter: apiResponse.nodeFilter || {
      enabled: true,
      qualityThreshold: 0.7,
      authorityThreshold: 0.5,
      filterByQuality: true,
      filterByAuthority: false,
      filterMode: 'or' as const
    }
  };
}

// ============================================================================
// Simple cache for getAll() to avoid redundant fetches
// ============================================================================

let _cachedAllSettings: any = null;
let _cachedAllTimestamp = 0;
const CACHE_TTL_MS = 2000;

function getNestedValue(obj: any, path: string): any {
  const parts = path.split('.');
  let current = obj;
  for (const part of parts) {
    if (current === undefined || current === null) return undefined;
    current = current[part];
  }
  return current;
}

// ============================================================================
// API Client
// ============================================================================

/**
 * Check if a settings path targets a visual setting that should be routed
 * to the server `/api/settings/visual` endpoint.
 *
 * Excludes paths already handled by dedicated endpoints (rendering, physics).
 */
function isVisualSettingsPath(path: string): boolean {
  if (path.startsWith('visualisation.rendering')) return false;
  if (path.startsWith('visualisation.graphs.') && path.includes('.physics')) return false;
  if (path.startsWith('visualisation.')) return true;
  return false;
}

/**
 * Convert a client settings path to its key within the visual blob stored on the server.
 *
 * Mappings:
 *   visualisation.graphs.logseq.nodes.X  → nodes.X
 *   visualisation.graphs.logseq.edges.X  → edges.X
 *   visualisation.graphs.logseq.labels.X → labels.X
 *   visualisation.<category>.X           → <category>.X
 */
function toVisualKey(path: string): string {
  if (path.startsWith('visualisation.graphs.logseq.nodes')) {
    return path.replace('visualisation.graphs.logseq.nodes', 'nodes');
  }
  if (path.startsWith('visualisation.graphs.logseq.edges')) {
    return path.replace('visualisation.graphs.logseq.edges', 'edges');
  }
  if (path.startsWith('visualisation.graphs.logseq.labels')) {
    return path.replace('visualisation.graphs.logseq.labels', 'labels');
  }
  return path.replace('visualisation.', '');
}

/** Build a nested object from a dot-notation path and a leaf value. */
function setNestedFromDotPath(obj: Record<string, any>, dotPath: string, value: any): void {
  const parts = dotPath.split('.');
  let current = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    if (!(parts[i] in current) || typeof current[parts[i]] !== 'object') {
      current[parts[i]] = {};
    }
    current = current[parts[i]];
  }
  current[parts[parts.length - 1]] = value;
}

export const settingsApi = {

  getPhysics: (): Promise<AxiosResponse<PhysicsSettings>> =>
    axios.get(`${API_BASE}/api/settings/physics`),

  updatePhysics: async (
    settings: Partial<PhysicsSettings>
  ): Promise<AxiosResponse<void>> => {
    console.warn('[SETTINGS-DIAG] updatePhysics called with:', settings);
    console.warn('[SETTINGS-DIAG] auth: authenticated=', nostrAuth.isAuthenticated());
    // GET-merge-PUT: backend requires full struct, not partial
    try {
      const current = await axios.get(`${API_BASE}/api/settings/physics`);
      console.warn('[SETTINGS-DIAG] updatePhysics GET current:', current.status, current.data);
      const currentData = current.data?.data ?? current.data ?? {};
      const merged = { ...currentData, ...settings };
      console.warn('[SETTINGS-DIAG] updatePhysics PUT merged:', merged);
      const result = await axios.put(`${API_BASE}/api/settings/physics`, merged);
      console.warn('[SETTINGS-DIAG] updatePhysics PUT response:', result.status, result.data);
      return result;
    } catch (err) {
      console.warn('[SETTINGS-DIAG] updatePhysics FAILED:', err);
      throw err;
    }
  },


  getConstraints: (): Promise<AxiosResponse<ConstraintSettings>> =>
    axios.get(`${API_BASE}/api/settings/constraints`),

  updateConstraints: async (
    settings: Partial<ConstraintSettings>
  ): Promise<AxiosResponse<void>> => {
    const current = await axios.get(`${API_BASE}/api/settings/constraints`);
    const currentData = current.data?.data ?? current.data ?? {};
    const merged = { ...currentData, ...settings };
    return axios.put(`${API_BASE}/api/settings/constraints`, merged);
  },


  getRendering: (): Promise<AxiosResponse<RenderingSettings>> =>
    axios.get(`${API_BASE}/api/settings/rendering`),

  updateRendering: async (
    settings: Partial<RenderingSettings>
  ): Promise<AxiosResponse<void>> => {
    const current = await axios.get(`${API_BASE}/api/settings/rendering`);
    const currentData = current.data?.data ?? current.data ?? {};
    const merged = { ...currentData, ...settings };
    return axios.put(`${API_BASE}/api/settings/rendering`, merged);
  },


  // NOTE: Over-fetches all settings sections. The backend does not currently support
  // fetching individual sections in a single call. Consider adding a query parameter
  // (e.g., ?sections=physics,rendering) if per-section fetching becomes available.
  getAll: (): Promise<AxiosResponse<AllSettings>> =>
    axios.get(`${API_BASE}/api/settings/all`),

  // Transform API response to client-expected nested structure, filtered to requested paths
  getSettingsByPaths: async (paths: string[]): Promise<any> => {
    try {
      let allSettings: any;
      const now = Date.now();

      // Use cached result if fresh enough
      if (_cachedAllSettings && (now - _cachedAllTimestamp) < CACHE_TTL_MS) {
        allSettings = _cachedAllSettings;
      } else {
        const response = await axios.get(`${API_BASE}/api/settings/all`);
        allSettings = transformApiToClientSettings(response.data);
        _cachedAllSettings = allSettings;
        _cachedAllTimestamp = now;
      }

      // Filter to only requested paths (for callers that use path-keyed results)
      const result: Record<string, any> = {};
      for (const path of paths) {
        const value = getNestedValue(allSettings, path);
        if (value !== undefined) {
          result[path] = value;
        }
      }

      // Return the full nested structure so callers that expect it still work
      // (the settingsStore initialize() treats the return as a full settings object)
      return allSettings;
    } catch (error) {
      console.error('[settingsApi] Failed to get settings:', error);
      // Return default settings on error
      return transformApiToClientSettings({
        physics: {} as PhysicsSettings,
        constraints: {} as ConstraintSettings,
        rendering: {} as RenderingSettings,
        nodeFilter: {} as NodeFilterSettings,
        qualityGates: {} as QualityGateSettings
      });
    }
  },

  // Get a single setting by dot-notation path
  getSettingByPath: async <T>(path: string): Promise<T> => {
    const allSettings = await settingsApi.getSettingsByPaths([path]);
    const parts = path.split('.');
    let current: any = allSettings;
    for (const part of parts) {
      if (current === undefined) break;
      current = current[part];
    }
    return current as T;
  },

  // Update a single setting by dot-notation path
  updateSettingByPath: async <T>(path: string, value: T): Promise<void> => {
    try {
      // Map client paths to API endpoints
      if (path.startsWith('visualisation.graphs.') && path.includes('.physics')) {
        await settingsApi.updatePhysics({ [path.split('.').pop()!]: value } as any);
      } else if (path.startsWith('visualisation.rendering')) {
        await settingsApi.updateRendering({ [path.split('.').pop()!]: value } as any);
      } else if (path.startsWith('qualityGates')) {
        await settingsApi.updateQualityGates({ [path.split('.').pop()!]: value } as any);
      } else if (path.startsWith('nodeFilter')) {
        await settingsApi.updateNodeFilter({ [path.split('.').pop()!]: value } as any);
      } else if (path.startsWith('constraints')) {
        const key = path.split('.').pop()!;
        await settingsApi.updateConstraints({ [key]: value });
      } else if (isVisualSettingsPath(path)) {
        // Route visual settings to the server visual endpoint
        const visualKey = toVisualKey(path);
        const nested: Record<string, any> = {};
        setNestedFromDotPath(nested, visualKey, value);
        await settingsApi.updateVisualSettings(nested);
      } else {
        // Non-visual, non-server paths (system, auth ephemeral state, etc.)
        console.debug(`[settingsApi] Path "${path}" persisted to localStorage only (no server endpoint)`);
      }
    } catch (error) {
      // Log but don't throw -- the value is already saved in settingsStore/localStorage.
      // Server-side persistence failure should not block the UI.
      console.warn(`[settingsApi] Server update failed for "${path}", value persisted locally`, error);
    }
  },

  // Update multiple settings by paths
  updateSettingsByPaths: async (updates: Array<{ path: string; value: any }>): Promise<void> => {
    console.warn(`[SETTINGS-DIAG] updateSettingsByPaths called with ${updates.length} updates:`, updates.map(u => u.path));
    // Group updates by API endpoint
    const physicsUpdates: Record<string, any> = {};
    const renderingUpdates: Record<string, any> = {};
    const qualityGatesUpdates: Record<string, any> = {};
    const nodeFilterUpdates: Record<string, any> = {};
    const constraintsUpdates: Record<string, any> = {};
    const visualUpdates: Record<string, any> = {};
    const localOnlyPaths: string[] = [];

    for (const { path, value } of updates) {
      if (path.startsWith('visualisation.graphs.') && path.includes('.physics.')) {
        const key = path.split('.').pop()!;
        physicsUpdates[key] = value;
        console.warn(`[SETTINGS-DIAG] routing ${path} → physics.${key} = ${value}`);
      } else if (path.startsWith('visualisation.rendering.')) {
        const key = path.split('.').pop()!;
        renderingUpdates[key] = value;
        console.warn(`[SETTINGS-DIAG] routing ${path} → rendering.${key} = ${value}`);
      } else if (path.startsWith('qualityGates.')) {
        const key = path.split('.').pop()!;
        qualityGatesUpdates[key] = value;
      } else if (path.startsWith('nodeFilter.')) {
        const key = path.split('.').pop()!;
        nodeFilterUpdates[key] = value;
      } else if (path.startsWith('constraints.')) {
        const key = path.split('.').pop()!;
        constraintsUpdates[key] = value;
      } else if (isVisualSettingsPath(path)) {
        // Batch all visual paths into a single nested object for the visual endpoint
        const visualKey = toVisualKey(path);
        setNestedFromDotPath(visualUpdates, visualKey, value);
        console.warn(`[SETTINGS-DIAG] routing ${path} → visual.${visualKey} = ${value}`);
      } else {
        // Non-visual, non-server paths (system debug, auth ephemeral state, etc.)
        localOnlyPaths.push(path);
        console.warn(`[SETTINGS-DIAG] routing ${path} → LOCAL ONLY (no server endpoint)`);
      }
    }

    if (localOnlyPaths.length > 0) {
      console.debug(`[settingsApi] ${localOnlyPaths.length} paths persisted to localStorage only:`, localOnlyPaths);
    }

    // Send batched updates to server for supported categories
    const promises: Promise<any>[] = [];
    if (Object.keys(physicsUpdates).length > 0) {
      promises.push(settingsApi.updatePhysics(physicsUpdates as any));
    }
    if (Object.keys(renderingUpdates).length > 0) {
      promises.push(settingsApi.updateRendering(renderingUpdates as any));
    }
    if (Object.keys(qualityGatesUpdates).length > 0) {
      promises.push(settingsApi.updateQualityGates(qualityGatesUpdates as any));
    }
    if (Object.keys(nodeFilterUpdates).length > 0) {
      promises.push(settingsApi.updateNodeFilter(nodeFilterUpdates as any));
    }
    if (Object.keys(constraintsUpdates).length > 0) {
      promises.push(settingsApi.updateConstraints(constraintsUpdates));
    }
    if (Object.keys(visualUpdates).length > 0) {
      promises.push(settingsApi.updateVisualSettings(visualUpdates));
    }

    if (promises.length > 0) {
      await Promise.all(promises);
    }
  },

  // Flush any pending updates (no-op for now, updates are immediate)
  flushPendingUpdates: async (): Promise<void> => {
    // Currently updates are synchronous, this is for future batching
  },

  // Reset settings to defaults
  resetSettings: async (): Promise<void> => {
    // Clear localStorage
    localStorage.removeItem('graph-viz-settings-v2');
    // Invalidate local cache
    _cachedAllSettings = null;
    _cachedAllTimestamp = 0;
    // Also reset server-side settings to defaults
    try {
      const defaultPhysics: Partial<PhysicsSettings> = {
        enabled: true,
        damping: 0.5,
        boundsSize: 1000,
        enableBounds: true,
        maxVelocity: 10,
        maxForce: 50,
        repelK: 1.0,
        iterations: 1,
        separationRadius: 50,
        autoBalance: false,
        autoBalanceIntervalMs: 5000,
        autoBalanceConfig: { maxIterations: 100, threshold: 0.01 },
        autoPause: { enabled: false, inactivityThresholdMs: 5000 },
      };
      await settingsApi.updatePhysics(defaultPhysics);
    } catch (e) {
      console.warn('[settingsApi] Failed to reset server settings:', e);
    }
  },

  // Export settings as JSON string
  exportSettings: (settings: any): string => {
    return JSON.stringify(settings, null, 2);
  },

  // Import settings from JSON string with schema validation
  importSettings: (jsonString: string): any => {
    const parsed = JSON.parse(jsonString);

    // Validate that parsed object is a non-null object (not array/primitive)
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      throw new Error('Invalid settings format: expected a JSON object');
    }

    // Validate expected top-level structure - must contain at least one known section
    const knownSections = ['physics', 'constraints', 'rendering', 'nodeFilter', 'qualityGates',
      'visualisation', 'system', 'xr', 'auth'];
    const parsedKeys = Object.keys(parsed);
    const hasKnownSection = parsedKeys.some(key => knownSections.includes(key));
    if (!hasKnownSection) {
      throw new Error(
        `Invalid settings structure: expected at least one of [${knownSections.join(', ')}]`
      );
    }

    return parsed;
  },


  saveProfile: (
    request: SaveProfileRequest
  ): Promise<AxiosResponse<ProfileIdResponse>> =>
    axios.post(`${API_BASE}/api/settings/profiles`, request),

  listProfiles: (): Promise<AxiosResponse<SettingsProfile[]>> =>
    axios.get(`${API_BASE}/api/settings/profiles`),

  loadProfile: (id: number): Promise<AxiosResponse<AllSettings>> =>
    axios.get(`${API_BASE}/api/settings/profiles/${id}`),

  deleteProfile: (id: number): Promise<AxiosResponse<void>> =>
    axios.delete(`${API_BASE}/api/settings/profiles/${id}`),

  // Node filter settings
  getNodeFilter: (): Promise<AxiosResponse<NodeFilterSettings>> =>
    axios.get(`${API_BASE}/api/settings/node-filter`),

  updateNodeFilter: async (
    settings: Partial<NodeFilterSettings>
  ): Promise<AxiosResponse<void>> => {
    const current = await axios.get(`${API_BASE}/api/settings/node-filter`);
    const currentData = current.data?.data ?? current.data ?? {};
    const merged = { ...currentData, ...settings };
    return axios.put(`${API_BASE}/api/settings/node-filter`, merged);
  },

  // Quality gate settings
  getQualityGates: (): Promise<AxiosResponse<QualityGateSettings>> =>
    axios.get(`${API_BASE}/api/settings/quality-gates`),

  updateQualityGates: async (
    settings: Partial<QualityGateSettings>
  ): Promise<AxiosResponse<void>> => {
    const current = await axios.get(`${API_BASE}/api/settings/quality-gates`);
    const currentData = current.data?.data ?? current.data ?? {};
    const merged = { ...currentData, ...settings };
    return axios.put(`${API_BASE}/api/settings/quality-gates`, merged);
  },

  // Visual settings (glow, hologram, graphTypeVisuals, gemMaterial, sceneEffects,
  // clusterHulls, animations, interaction, nodes, edges, labels)
  getVisualSettings: (): Promise<AxiosResponse<Record<string, any>>> =>
    axios.get(`${API_BASE}/api/settings/visual`),

  updateVisualSettings: async (
    patch: Record<string, any>
  ): Promise<AxiosResponse<Record<string, any>>> =>
    axios.put(`${API_BASE}/api/settings/visual`, patch),
};

// ============================================================================
// Utility Functions
// ============================================================================

export const clamp = (value: number, min: number, max: number): number => {
  return Math.max(min, Math.min(max, value));
};

export const validatePhysicsSettings = (
  settings: Partial<PhysicsSettings>
): string | null => {
  if (settings.damping !== undefined) {
    if (settings.damping < 0 || settings.damping > 1) {
      return 'Damping must be between 0 and 1';
    }
  }
  if (settings.boundsSize !== undefined) {
    if (settings.boundsSize <= 0) {
      return 'Bounds size must be positive';
    }
  }
  if (settings.maxVelocity !== undefined) {
    if (settings.maxVelocity <= 0) {
      return 'Max velocity must be positive';
    }
  }
  return null;
};

export const validateConstraintSettings = (
  settings: Partial<ConstraintSettings>
): string | null => {
  if (settings.activationFrames !== undefined) {
    if (settings.activationFrames < 1 || settings.activationFrames > 600) {
      return 'Activation frames must be between 1 and 600';
    }
  }
  if (settings.farThreshold !== undefined) {
    if (settings.farThreshold < 0) {
      return 'Far threshold must be non-negative';
    }
  }
  if (settings.mediumThreshold !== undefined) {
    if (settings.mediumThreshold < 0) {
      return 'Medium threshold must be non-negative';
    }
  }
  if (settings.nearThreshold !== undefined) {
    if (settings.nearThreshold < 0) {
      return 'Near threshold must be non-negative';
    }
  }
  return null;
};
