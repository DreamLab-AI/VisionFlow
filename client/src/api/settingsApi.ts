// frontend/src/api/settingsApi.ts
// REAL API client for settings management - NO MOCKS

import axios, { AxiosResponse } from 'axios';

// Use Vite proxy for API requests in development
// In production, this will be served from the same origin
// NOTE: Empty string because the function paths already include '/api/'
const API_BASE = '';

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
// API Client
// ============================================================================

export const settingsApi = {
  
  getPhysics: (): Promise<AxiosResponse<PhysicsSettings>> =>
    axios.get(`${API_BASE}/api/settings/physics`),

  updatePhysics: (
    settings: Partial<PhysicsSettings>
  ): Promise<AxiosResponse<void>> =>
    axios.put(`${API_BASE}/api/settings/physics`, settings),

  
  getConstraints: (): Promise<AxiosResponse<ConstraintSettings>> =>
    axios.get(`${API_BASE}/api/settings/constraints`),

  updateConstraints: (
    settings: Partial<ConstraintSettings>
  ): Promise<AxiosResponse<void>> =>
    axios.put(`${API_BASE}/api/settings/constraints`, settings),

  
  getRendering: (): Promise<AxiosResponse<RenderingSettings>> =>
    axios.get(`${API_BASE}/api/settings/rendering`),

  updateRendering: (
    settings: Partial<RenderingSettings>
  ): Promise<AxiosResponse<void>> =>
    axios.put(`${API_BASE}/api/settings/rendering`, settings),

  
  getAll: (): Promise<AxiosResponse<AllSettings>> =>
    axios.get(`${API_BASE}/api/settings/all`),

  
  
  getSettingsByPaths: async (paths: string[]): Promise<AxiosResponse<AllSettings>> => {
    return axios.get(`${API_BASE}/api/settings/all`);
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

  updateNodeFilter: (
    settings: Partial<NodeFilterSettings>
  ): Promise<AxiosResponse<void>> =>
    axios.put(`${API_BASE}/api/settings/node-filter`, settings),

  // Quality gate settings
  getQualityGates: (): Promise<AxiosResponse<QualityGateSettings>> =>
    axios.get(`${API_BASE}/api/settings/quality-gates`),

  updateQualityGates: (
    settings: Partial<QualityGateSettings>
  ): Promise<AxiosResponse<void>> =>
    axios.put(`${API_BASE}/api/settings/quality-gates`, settings),
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
