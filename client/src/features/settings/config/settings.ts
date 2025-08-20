// Type definitions for settings

export type SettingsPath = string | '';

// Node settings
export interface NodeSettings {
  baseColor: string;
  metalness: number;
  opacity: number;
  roughness: number;
  nodeSize: number; // Changed from sizeRange: [number, number]
  quality: 'low' | 'medium' | 'high';
  enableInstancing: boolean;
  enableHologram: boolean;
  enableMetadataShape: boolean;
  enableMetadataVisualisation: boolean;
}

// Edge settings
export interface EdgeSettings {
  arrowSize: number;
  baseWidth: number;
  color: string;
  enableArrows: boolean;
  opacity: number;
  widthRange: [number, number];
  quality: 'low' | 'medium' | 'high';
  enableFlowEffect: boolean;
  flowSpeed: number;
  flowIntensity: number;
  glowStrength: number;
  distanceIntensity: number;
  useGradient: boolean;
  gradientColors: [string, string];
}

// Physics settings - using camelCase for client
export interface PhysicsSettings {
  enabled: boolean;
  
  // Core GPU Forces
  springK: number;
  repelK: number;
  attractionK: number;
  gravity: number;
  
  // Dynamics
  dt: number;
  maxVelocity: number;
  damping: number;
  temperature: number;
  
  // Boundary & Collision
  enableBounds: boolean;
  boundsSize: number;
  boundaryDamping: number;
  separationRadius: number;
  collisionRadius?: number; // For compatibility
  
  // New CUDA kernel parameters
  restLength: number;
  repulsionCutoff: number;
  repulsionSofteningEpsilon: number;
  centerGravityK: number;
  gridCellSize: number;
  featureFlags: number;
  
  // GPU-specific parameters
  stressWeight: number;
  stressAlpha: number;
  boundaryLimit: number;
  alignmentStrength: number;
  clusterStrength: number;
  computeMode: number;
  minDistance: number;
  maxRepulsionDist: number;
  boundaryMargin: number;
  boundaryForceStrength: number;
  
  // Advanced Parameters
  iterations: number;
  massScale: number;
  updateThreshold: number;
  
  // Warmup System
  warmupIterations: number;
  warmupCurve?: string; // Optional for compatibility
  zeroVelocityIterations?: number; // Optional for compatibility  
  coolingRate: number;
  
  // Clustering
  clusteringAlgorithm?: string; // Optional for compatibility
  clusterCount?: number; // Optional for compatibility
  clusteringResolution?: number; // Optional for compatibility
  clusteringIterations?: number; // Optional for compatibility
}

// Rendering settings
export interface RenderingSettings {
  ambientLightIntensity: number;
  backgroundColor: string;
  directionalLightIntensity: number;
  enableAmbientOcclusion: boolean;
  enableAntialiasing: boolean;
  enableShadows: boolean;
  environmentIntensity: number;
  shadowMapSize: string;
  shadowBias: number;
  context: 'desktop' | 'ar';
}

// Animation settings
export interface AnimationSettings {
  enableMotionBlur: boolean;
  enableNodeAnimations: boolean;
  motionBlurStrength: number;
  selectionWaveEnabled: boolean;
  pulseEnabled: boolean;
  pulseSpeed: number;
  pulseStrength: number;
  waveSpeed: number;
}

// Label settings
export interface LabelSettings {
  desktopFontSize: number;
  enableLabels: boolean;
  textColor: string;
  textOutlineColor: string;
  textOutlineWidth: number;
  textResolution: number;
  textPadding: number;
  billboardMode: 'camera' | 'vertical';
  showMetadata?: boolean; // Display metadata as secondary label
  maxLabelWidth?: number; // Maximum width for label text
}

// Bloom settings
export interface BloomSettings {
  edgeBloomStrength: number;
  enabled: boolean;
  environmentBloomStrength: number;
  nodeBloomStrength: number;
  radius: number;
  strength: number;
  threshold: number;
}

// Hologram settings
export interface HologramSettings {
  ringCount: number;
  ringColor: string;
  ringOpacity: number;
  sphereSizes: [number, number];
  ringRotationSpeed: number;
  enableBuckminster: boolean;
  buckminsterSize: number;
  buckminsterOpacity: number;
  enableGeodesic: boolean;
  geodesicSize: number;
  geodesicOpacity: number;
  enableTriangleSphere: boolean;
  triangleSphereSize: number;
  triangleSphereOpacity: number;
  globalRotationSpeed: number;
}

// WebSocket settings
export interface WebSocketSettings {
  reconnectAttempts: number;
  reconnectDelay: number;
  binaryChunkSize: number;
  binaryUpdateRate?: number;
  minUpdateRate?: number;
  maxUpdateRate?: number;
  motionThreshold?: number;
  motionDamping?: number;
  binaryMessageVersion?: number;
  compressionEnabled: boolean;
  compressionThreshold: number;
  heartbeatInterval?: number;
  heartbeatTimeout?: number;
  maxConnections?: number;
  maxMessageSize?: number;
  updateRate: number;
}

// Debug settings
export interface DebugSettings {
  enabled: boolean;
  logLevel?: 'debug' | 'info' | 'warn' | 'error'; // Added for client-side logging
  logFormat?: 'json' | 'text';
  enableDataDebug: boolean;
  enableWebsocketDebug: boolean;
  logBinaryHeaders: boolean;
  logFullJson: boolean;
  enablePhysicsDebug?: boolean;
  enableNodeDebug?: boolean;
  enableShaderDebug?: boolean;
  enableMatrixDebug?: boolean;
  enablePerformanceDebug?: boolean;
}

// SpacePilot settings
export interface SpacePilotConfig {
  enabled: boolean;
  mode: 'camera' | 'object' | 'navigation';
  sensitivity: {
    translation: number;
    rotation: number;
  };
  smoothing: number;
  deadzone: number;
  buttonFunctions?: Record<number, string>;
}

// XR settings
export interface XRSettings {
  enabled: boolean;
  clientSideEnableXR?: boolean; // Client-side XR toggle
  mode?: 'inline' | 'immersive-vr' | 'immersive-ar'; // Added from YAML
  roomScale?: number;
  spaceType?: 'local-floor' | 'bounded-floor' | 'unbounded';
  quality?: 'low' | 'medium' | 'high';
  enableHandTracking: boolean;
  handMeshEnabled?: boolean;
  handMeshColor?: string;
  handMeshOpacity?: number;
  handPointSize?: number;
  handRayEnabled?: boolean;
  handRayColor?: string;
  handRayWidth?: number;
  gestureSmoothing?: number;
  enableHaptics: boolean;
  hapticIntensity?: number;
  dragThreshold?: number;
  pinchThreshold?: number;
  rotationThreshold?: number;
  interactionRadius?: number;
  movementSpeed?: number;
  deadZone?: number;
  movementAxes?: {
    horizontal: number;
    vertical: number;
  };
  enableLightEstimation?: boolean;
  enablePlaneDetection?: boolean;
  enableSceneUnderstanding?: boolean;
  planeColor?: string;
  planeOpacity?: number;
  planeDetectionDistance?: number;
  showPlaneOverlay?: boolean;
  snapToFloor?: boolean;
  enablePassthroughPortal?: boolean;
  passthroughOpacity?: number;
  passthroughBrightness?: number;
  passthroughContrast?: number;
  portalSize?: number;
  portalEdgeColor?: string;
  portalEdgeWidth?: number;
  controllerModel?: string;
  renderScale?: number;
  interactionDistance?: number;
  locomotionMethod?: 'teleport' | 'continuous';
  teleportRayColor?: string;
  controllerRayColor?: string;
}

// Visualisation settings
export interface CameraSettings {
  fov: number;
  near: number;
  far: number;
  position: { x: number; y: number; z: number };
  lookAt?: { x: number; y: number; z: number }; // lookAt is often dynamic
}

// Graph-specific settings namespace
export interface GraphSettings {
  nodes: NodeSettings;
  edges: EdgeSettings;
  labels: LabelSettings;
  physics: PhysicsSettings;
}

// Multi-graph namespace structure
export interface GraphsSettings {
  logseq: GraphSettings;
  visionflow: GraphSettings;
}

export interface VisualisationSettings {
  // Global visualisation settings (shared across graphs)
  rendering: RenderingSettings;
  animations: AnimationSettings;
  bloom: BloomSettings;
  hologram: HologramSettings;
  spacePilot?: SpacePilotConfig;
  camera?: CameraSettings;
  
  // Graph-specific settings
  graphs: GraphsSettings;
  
}

// System settings
export interface SystemSettings {
  websocket: WebSocketSettings;
  debug: DebugSettings;
  persistSettings: boolean; // Added to control server-side persistence
  customBackendUrl?: string; // Add if missing
}

// RAGFlow settings
export interface RAGFlowSettings {
  apiKey?: string;
  agentId?: string;
  apiBaseUrl?: string;
  timeout?: number;
  maxRetries?: number;
  chatId?: string;
}

// Perplexity settings
export interface PerplexitySettings {
  apiKey?: string;
  model?: string;
  apiUrl?: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  timeout?: number;
  rateLimit?: number;
}

// OpenAI settings
export interface OpenAISettings {
  apiKey?: string;
  baseUrl?: string;
  timeout?: number;
  rateLimit?: number;
}

// Kokoro TTS settings
export interface KokoroSettings {
  apiUrl?: string;
  defaultVoice?: string;
  defaultFormat?: string;
  defaultSpeed?: number;
  timeout?: number;
  stream?: boolean;
  returnTimestamps?: boolean;
  sampleRate?: number;
}

// Auth settings
export interface AuthSettings {
  enabled: boolean;
  provider: 'nostr' | string; // Allow other providers potentially
  required: boolean;
}

// Whisper speech recognition settings
export interface WhisperSettings {
  apiUrl?: string;
  defaultModel?: string;
  defaultLanguage?: string;
  timeout?: number;
  temperature?: number;
  returnTimestamps?: boolean;
  vadFilter?: boolean;
  wordTimestamps?: boolean;
  initialPrompt?: string;
}

// Dashboard GPU status settings
export interface DashboardSettings {
  autoRefresh: boolean;
  refreshInterval: number;
  computeMode: 'Basic Force-Directed' | 'Dual Graph' | 'Constraint-Enhanced' | 'Visual Analytics';
  iterationCount: number;
  showConvergence: boolean;
  activeConstraints: number;
  clusteringActive: boolean;
}

// Analytics settings with GPU clustering
export interface AnalyticsSettings {
  updateInterval: number;
  showDegreeDistribution: boolean;
  showClusteringCoefficient: boolean;
  showCentrality: boolean;
  clustering: {
    algorithm: 'none' | 'kmeans' | 'spectral' | 'louvain';
    clusterCount: number;
    resolution: number;
    iterations: number;
    exportEnabled: boolean;
    importEnabled: boolean;
  };
}

// Performance settings with warmup controls
export interface PerformanceSettings {
  enableAdaptiveQuality: boolean;
  warmupDuration: number;
  convergenceThreshold: number;
  enableAdaptiveCooling: boolean;
}

// Developer GPU debug settings
export interface DeveloperSettings {
  gpu: {
    showForceVectors: boolean;
    showConstraints: boolean;
    showBoundaryForces: boolean;
    showConvergenceGraph: boolean;
  };
  constraints: {
    active: Array<{
      id: string;
      name: string;
      enabled: boolean;
      description?: string;
      icon?: string;
    }>;
  };
}

// XR GPU optimization settings
export interface XRGPUSettings {
  enableOptimizedCompute: boolean;
  performance: {
    preset: 'Battery Saver' | 'Balanced' | 'Performance';
  };
  physics: {
    scale: number;
  };
}

// Main settings interface - Single source of truth matching server AppFullSettings
export interface Settings {
  visualisation: VisualisationSettings;
  system: SystemSettings;
  xr: XRSettings & { gpu?: XRGPUSettings };
  auth: AuthSettings;
  ragflow?: RAGFlowSettings;
  perplexity?: PerplexitySettings;
  openai?: OpenAISettings;
  kokoro?: KokoroSettings;
  whisper?: WhisperSettings;
  dashboard?: DashboardSettings;
  analytics?: AnalyticsSettings;
  performance?: PerformanceSettings;
  developer?: DeveloperSettings;
}

// Partial update types for settings mutations
export type DeepPartial<T> = T extends object ? {
  [P in keyof T]?: DeepPartial<T[P]>;
} : T;

export type SettingsUpdate = DeepPartial<Settings>;
