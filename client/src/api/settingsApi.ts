// frontend/src/api/settingsApi.ts
// REAL API client for settings management - NO MOCKS

import axios, { AxiosResponse } from 'axios';
import { nostrAuth } from '../services/nostrAuthService';

// Always use relative paths for API requests. In dev mode Vite proxies /api
// to the backend (http://127.0.0.1:4000). In production the serving proxy
// (nginx / HTTPS bridge) must also proxy /api to the backend.
// VITE_API_URL is only used for non-browser contexts (SSR, tests).
const API_BASE = '';

// Helper to get auth headers (Nostr NIP-07 + Bearer token)
const getAuthHeaders = () => {
  const token = nostrAuth.getSessionToken();
  const user = nostrAuth.getCurrentUser();
  if (!token) return {};
  const headers: Record<string, string> = {
    Authorization: `Bearer ${token}`,
  };
  if (user?.pubkey) {
    headers['X-Nostr-Pubkey'] = user.pubkey;
    headers['X-Nostr-Token'] = token;
  }
  return headers;
};

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

const DEFAULT_GRAPH_TYPE_VISUALS = {
  knowledgeGraph: {
    metalness: 0.6,
    roughness: 0.15,
    glowStrength: 2.5,
    innerGlowIntensity: 0.3,
    facetDetail: 2,
    authorityScaleFactor: 0.4,
    showDomainBadge: true,
    showQualityStars: true,
    showRecencyIndicator: true,
    showConnectionDensity: false,
  },
  ontology: {
    glowStrength: 1.8,
    orbitalRingCount: 8,
    orbitalRingSpeed: 0.5,
    hierarchyScaleFactor: 0.02,
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
    showHealthBar: true,
    showTokenRate: true,
    showTaskCount: false,
    bioluminescentIntensity: 0.6,
  },
};

// ============================================================================
// Transform flat API response to nested client structure
// ============================================================================

function transformApiToClientSettings(apiResponse: AllSettings): any {
  // Transform the flat API response into the nested structure the client expects
  return {
    visualisation: {
      rendering: apiResponse.rendering || {},
      glow: DEFAULT_GLOW_SETTINGS,
      bloom: DEFAULT_BLOOM_SETTINGS,
      hologram: DEFAULT_HOLOGRAM_SETTINGS,
      graphTypeVisuals: DEFAULT_GRAPH_TYPE_VISUALS,
      animations: {
        enableMotionBlur: false,
        enableNodeAnimations: true,
        motionBlurStrength: 0.5,
        selectionWaveEnabled: true,
        pulseEnabled: true,
        pulseSpeed: 1.0,
        pulseStrength: 0.5,
        waveSpeed: 1.0
      },
      graphs: {
        logseq: {
          physics: apiResponse.physics || {},
          nodes: {
            baseColor: '#202724',
            metalness: 0.1,
            opacity: 0.88,
            roughness: 0.6,
            nodeSize: 1.7,
            quality: 'high' as const,
            enableInstancing: true,
            enableMetadataShape: false,
            enableMetadataVisualisation: true
          },
          edges: {
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
          },
          labels: {
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
          }
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

export const settingsApi = {

  getPhysics: (): Promise<AxiosResponse<PhysicsSettings>> =>
    axios.get(`${API_BASE}/api/settings/physics`, { headers: getAuthHeaders() }),

  updatePhysics: async (
    settings: Partial<PhysicsSettings>
  ): Promise<AxiosResponse<void>> => {
    // GET-merge-PUT: backend requires full struct, not partial
    const current = await axios.get(`${API_BASE}/api/settings/physics`, { headers: getAuthHeaders() });
    const currentData = current.data?.data ?? current.data ?? {};
    const merged = { ...currentData, ...settings };
    return axios.put(`${API_BASE}/api/settings/physics`, merged, { headers: getAuthHeaders() });
  },


  getConstraints: (): Promise<AxiosResponse<ConstraintSettings>> =>
    axios.get(`${API_BASE}/api/settings/constraints`, { headers: getAuthHeaders() }),

  updateConstraints: async (
    settings: Partial<ConstraintSettings>
  ): Promise<AxiosResponse<void>> => {
    const current = await axios.get(`${API_BASE}/api/settings/constraints`, { headers: getAuthHeaders() });
    const currentData = current.data?.data ?? current.data ?? {};
    const merged = { ...currentData, ...settings };
    return axios.put(`${API_BASE}/api/settings/constraints`, merged, { headers: getAuthHeaders() });
  },


  getRendering: (): Promise<AxiosResponse<RenderingSettings>> =>
    axios.get(`${API_BASE}/api/settings/rendering`, { headers: getAuthHeaders() }),

  updateRendering: async (
    settings: Partial<RenderingSettings>
  ): Promise<AxiosResponse<void>> => {
    const current = await axios.get(`${API_BASE}/api/settings/rendering`, { headers: getAuthHeaders() });
    const currentData = current.data?.data ?? current.data ?? {};
    const merged = { ...currentData, ...settings };
    return axios.put(`${API_BASE}/api/settings/rendering`, merged, { headers: getAuthHeaders() });
  },


  // NOTE: Over-fetches all settings sections. The backend does not currently support
  // fetching individual sections in a single call. Consider adding a query parameter
  // (e.g., ?sections=physics,rendering) if per-section fetching becomes available.
  getAll: (): Promise<AxiosResponse<AllSettings>> =>
    axios.get(`${API_BASE}/api/settings/all`, { headers: getAuthHeaders() }),

  // Transform API response to client-expected nested structure, filtered to requested paths
  getSettingsByPaths: async (paths: string[]): Promise<any> => {
    try {
      let allSettings: any;
      const now = Date.now();

      // Use cached result if fresh enough
      if (_cachedAllSettings && (now - _cachedAllTimestamp) < CACHE_TTL_MS) {
        allSettings = _cachedAllSettings;
      } else {
        const response = await axios.get(`${API_BASE}/api/settings/all`, { headers: getAuthHeaders() });
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
      } else {
        // For paths without a dedicated backend endpoint (glow, edges, hologram,
        // graphTypeVisuals, nodes, labels, etc.), persist to localStorage via
        // the settingsStore partialize. The Zustand persist middleware handles
        // this automatically since partialSettings is now included in partialize.
        // Log at debug level so developers can track unhandled server paths.
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
    // Group updates by API endpoint
    const physicsUpdates: Record<string, any> = {};
    const renderingUpdates: Record<string, any> = {};
    const qualityGatesUpdates: Record<string, any> = {};
    const nodeFilterUpdates: Record<string, any> = {};
    const localOnlyPaths: string[] = [];

    for (const { path, value } of updates) {
      if (path.includes('.physics.')) {
        const key = path.split('.').pop()!;
        physicsUpdates[key] = value;
      } else if (path.startsWith('visualisation.rendering.')) {
        const key = path.split('.').pop()!;
        renderingUpdates[key] = value;
      } else if (path.startsWith('qualityGates.')) {
        const key = path.split('.').pop()!;
        qualityGatesUpdates[key] = value;
      } else if (path.startsWith('nodeFilter.')) {
        const key = path.split('.').pop()!;
        nodeFilterUpdates[key] = value;
      } else {
        // Paths without server endpoints (glow, edges, hologram, graphTypeVisuals, etc.)
        // are persisted to localStorage by the settingsStore partialize.
        localOnlyPaths.push(path);
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
    axios.post(`${API_BASE}/api/settings/profiles`, request, { headers: getAuthHeaders() }),

  listProfiles: (): Promise<AxiosResponse<SettingsProfile[]>> =>
    axios.get(`${API_BASE}/api/settings/profiles`, { headers: getAuthHeaders() }),

  loadProfile: (id: number): Promise<AxiosResponse<AllSettings>> =>
    axios.get(`${API_BASE}/api/settings/profiles/${id}`, { headers: getAuthHeaders() }),

  deleteProfile: (id: number): Promise<AxiosResponse<void>> =>
    axios.delete(`${API_BASE}/api/settings/profiles/${id}`, { headers: getAuthHeaders() }),

  // Node filter settings
  getNodeFilter: (): Promise<AxiosResponse<NodeFilterSettings>> =>
    axios.get(`${API_BASE}/api/settings/node-filter`, { headers: getAuthHeaders() }),

  updateNodeFilter: async (
    settings: Partial<NodeFilterSettings>
  ): Promise<AxiosResponse<void>> => {
    const current = await axios.get(`${API_BASE}/api/settings/node-filter`, { headers: getAuthHeaders() });
    const currentData = current.data?.data ?? current.data ?? {};
    const merged = { ...currentData, ...settings };
    return axios.put(`${API_BASE}/api/settings/node-filter`, merged, { headers: getAuthHeaders() });
  },

  // Quality gate settings
  getQualityGates: (): Promise<AxiosResponse<QualityGateSettings>> =>
    axios.get(`${API_BASE}/api/settings/quality-gates`, { headers: getAuthHeaders() }),

  updateQualityGates: async (
    settings: Partial<QualityGateSettings>
  ): Promise<AxiosResponse<void>> => {
    const current = await axios.get(`${API_BASE}/api/settings/quality-gates`, { headers: getAuthHeaders() });
    const currentData = current.data?.data ?? current.data ?? {};
    const merged = { ...currentData, ...settings };
    return axios.put(`${API_BASE}/api/settings/quality-gates`, merged, { headers: getAuthHeaders() });
  },
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
