// frontend/src/api/settingsApi.ts
// REAL API client for settings management - NO MOCKS

import axios, { AxiosResponse } from 'axios';
import { nostrAuth } from '../services/nostrAuthService';

// Use Vite proxy for API requests in development
// In production, this will be served from the same origin
// NOTE: Empty string because the function paths already include '/api/'
const API_BASE = '';

// Helper to get auth headers
const getAuthHeaders = () => {
  const token = nostrAuth.getSessionToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
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
  intensity: 1.0,
  radius: 0.5,
  threshold: 0.1,
  diffuseStrength: 0.5,
  atmosphericDensity: 0.3,
  volumetricIntensity: 0.5,
  baseColor: '#ffffff',
  emissionColor: '#00ffff',
  opacity: 1.0,
  pulseSpeed: 1.0,
  flowSpeed: 0.5,
  nodeGlowStrength: 1.0,
  edgeGlowStrength: 0.5,
  environmentGlowStrength: 0.3
};

const DEFAULT_BLOOM_SETTINGS = {
  enabled: true,
  intensity: 1.5,
  threshold: 0.1,
  radius: 0.5,
  strength: 1.5
};

const DEFAULT_HOLOGRAM_SETTINGS = {
  ringCount: 3,
  ringColor: '#00ffff',
  ringOpacity: 0.5,
  sphereSizes: [100, 150] as [number, number],
  globalRotationSpeed: 0.5,
  enableBuckminster: true,
  buckminsterSize: 120,
  buckminsterOpacity: 0.3,
  enableGeodesic: true,
  geodesicSize: 100,
  geodesicOpacity: 0.4,
  enableTriangleSphere: false,
  triangleSphereSize: 80,
  triangleSphereOpacity: 0.5,
  enableQuantumField: false,
  quantumFieldIntensity: 0.5,
  enablePlasmaEffects: false,
  plasmaIntensity: 0.5,
  enableEnergyFlow: false,
  energyFlowSpeed: 1.0,
  ringRotationSpeed: 0.5,
  enableRingParticles: false,
  particleDensity: 100
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
            baseColor: '#4a9eff',
            metalness: 0.3,
            opacity: 1.0,
            roughness: 0.7,
            nodeSize: 1.0,
            quality: 'medium' as const,
            enableInstancing: true,
            enableHologram: false,
            enableMetadataShape: false,
            enableMetadataVisualisation: false
          },
          edges: {
            arrowSize: 0.5,
            baseWidth: 0.2,
            color: '#56b6c2',
            enableArrows: true,
            opacity: 0.9,
            widthRange: [0.1, 0.3] as [number, number],
            quality: 'medium' as const,
            enableFlowEffect: false,
            flowSpeed: 1.0,
            flowIntensity: 0.5,
            glowStrength: 0.3,
            distanceIntensity: 0.5,
            useGradient: false,
            gradientColors: ['#4a9eff', '#ff4a9e'] as [string, string]
          },
          labels: {
            desktopFontSize: 1.4,
            enableLabels: true,
            textColor: '#ffffff',
            textOutlineColor: '#000000',
            textOutlineWidth: 0.5,
            textResolution: 256,
            textPadding: 4,
            billboardMode: 'camera' as const
          }
        },
        visionflow: {
          physics: apiResponse.physics || {},
          nodes: {
            baseColor: '#4a9eff',
            metalness: 0.3,
            opacity: 1.0,
            roughness: 0.7,
            nodeSize: 1.0,
            quality: 'medium' as const,
            enableInstancing: true,
            enableHologram: false,
            enableMetadataShape: false,
            enableMetadataVisualisation: false
          },
          edges: {
            arrowSize: 0.5,
            baseWidth: 0.2,
            color: '#56b6c2',
            enableArrows: true,
            opacity: 0.9,
            widthRange: [0.1, 0.3] as [number, number],
            quality: 'medium' as const,
            enableFlowEffect: false,
            flowSpeed: 1.0,
            flowIntensity: 0.5,
            glowStrength: 0.3,
            distanceIntensity: 0.5,
            useGradient: false,
            gradientColors: ['#4a9eff', '#ff4a9e'] as [string, string]
          },
          labels: {
            desktopFontSize: 1.4,
            enableLabels: true,
            textColor: '#ffffff',
            textOutlineColor: '#000000',
            textOutlineWidth: 0.5,
            textResolution: 256,
            textPadding: 4,
            billboardMode: 'camera' as const
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
      maxNodeCount: 10000,
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
// API Client
// ============================================================================

export const settingsApi = {

  getPhysics: (): Promise<AxiosResponse<PhysicsSettings>> =>
    axios.get(`${API_BASE}/api/settings/physics`, { headers: getAuthHeaders() }),

  updatePhysics: (
    settings: Partial<PhysicsSettings>
  ): Promise<AxiosResponse<void>> =>
    axios.put(`${API_BASE}/api/settings/physics`, settings, { headers: getAuthHeaders() }),


  getConstraints: (): Promise<AxiosResponse<ConstraintSettings>> =>
    axios.get(`${API_BASE}/api/settings/constraints`, { headers: getAuthHeaders() }),

  updateConstraints: (
    settings: Partial<ConstraintSettings>
  ): Promise<AxiosResponse<void>> =>
    axios.put(`${API_BASE}/api/settings/constraints`, settings, { headers: getAuthHeaders() }),


  getRendering: (): Promise<AxiosResponse<RenderingSettings>> =>
    axios.get(`${API_BASE}/api/settings/rendering`, { headers: getAuthHeaders() }),

  updateRendering: (
    settings: Partial<RenderingSettings>
  ): Promise<AxiosResponse<void>> =>
    axios.put(`${API_BASE}/api/settings/rendering`, settings, { headers: getAuthHeaders() }),


  getAll: (): Promise<AxiosResponse<AllSettings>> =>
    axios.get(`${API_BASE}/api/settings/all`, { headers: getAuthHeaders() }),

  // Transform API response to client-expected nested structure
  getSettingsByPaths: async (paths: string[]): Promise<any> => {
    try {
      const response = await axios.get(`${API_BASE}/api/settings/all`, { headers: getAuthHeaders() });
      const transformed = transformApiToClientSettings(response.data);
      return transformed;
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
    // Map client paths to API endpoints
    if (path.startsWith('visualisation.graphs.') && path.includes('.physics')) {
      await settingsApi.updatePhysics({ [path.split('.').pop()!]: value } as any);
    } else if (path.startsWith('visualisation.rendering')) {
      await settingsApi.updateRendering({ [path.split('.').pop()!]: value } as any);
    } else if (path.startsWith('qualityGates')) {
      await settingsApi.updateQualityGates({ [path.split('.').pop()!]: value } as any);
    }
    // For paths without backend support, store locally (already handled by settingsStore)
  },

  // Update multiple settings by paths
  updateSettingsByPaths: async (updates: Array<{ path: string; value: any }>): Promise<void> => {
    // Group updates by API endpoint
    const physicsUpdates: Record<string, any> = {};
    const renderingUpdates: Record<string, any> = {};
    const qualityGatesUpdates: Record<string, any> = {};

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
      }
    }

    // Send batched updates
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

    await Promise.all(promises);
  },

  // Flush any pending updates (no-op for now, updates are immediate)
  flushPendingUpdates: async (): Promise<void> => {
    // Currently updates are synchronous, this is for future batching
  },

  // Reset settings to defaults
  resetSettings: async (): Promise<void> => {
    // Clear localStorage and reload defaults
    localStorage.removeItem('graph-viz-settings-v2');
  },

  // Export settings as JSON string
  exportSettings: (settings: any): string => {
    return JSON.stringify(settings, null, 2);
  },

  // Import settings from JSON string
  importSettings: (jsonString: string): any => {
    return JSON.parse(jsonString);
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

  updateNodeFilter: (
    settings: Partial<NodeFilterSettings>
  ): Promise<AxiosResponse<void>> =>
    axios.put(`${API_BASE}/api/settings/node-filter`, settings, { headers: getAuthHeaders() }),

  // Quality gate settings
  getQualityGates: (): Promise<AxiosResponse<QualityGateSettings>> =>
    axios.get(`${API_BASE}/api/settings/quality-gates`, { headers: getAuthHeaders() }),

  updateQualityGates: (
    settings: Partial<QualityGateSettings>
  ): Promise<AxiosResponse<void>> =>
    axios.put(`${API_BASE}/api/settings/quality-gates`, settings, { headers: getAuthHeaders() }),
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
