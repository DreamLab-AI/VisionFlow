import { z } from 'zod';

// Regex patterns for validation
const HEX_COLOR_REGEX = /^#[0-9A-Fa-f]{6}$/;
const URL_REGEX = /^https?:\/\/[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})?(?::[0-9]+)?(?:\/.*)?$/;
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const DOMAIN_REGEX = /^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

// Custom validation functions
const hexColor = () => z.string().regex(HEX_COLOR_REGEX, 'Must be a valid hex color (e.g., #ff0000)');
const validUrl = () => z.string().regex(URL_REGEX, 'Must be a valid URL (e.g., https://example.com)');
const email = () => z.string().regex(EMAIL_REGEX, 'Must be a valid email address');
const domain = () => z.string().regex(DOMAIN_REGEX, 'Must be a valid domain');
const port = () => z.number().int().min(1, 'Port must be at least 1').max(65535, 'Port must be at most 65535');
const percentage = () => z.number().min(0, 'Must be at least 0%').max(100, 'Must be at most 100%');
const opacity = () => z.number().min(0.0, 'Opacity must be at least 0.0').max(1.0, 'Opacity must be at most 1.0');

// Movement axes validation
export const movementAxesSchema = z.object({
  horizontal: z.number().int().min(-100, 'Horizontal movement must be between -100 and 100').max(100, 'Horizontal movement must be between -100 and 100'),
  vertical: z.number().int().min(-100, 'Vertical movement must be between -100 and 100').max(100, 'Vertical movement must be between -100 and 100'),
});

// Node settings validation
export const nodeSettingsSchema = z.object({
  baseColor: hexColor(),
  metalness: opacity(),
  opacity: opacity(),
  roughness: opacity(),
  nodeSize: z.number().min(0.1, 'Node size must be at least 0.1').max(100.0, 'Node size must be at most 100.0'),
  quality: z.string().min(1, 'Quality cannot be empty'),
  enableInstancing: z.boolean(),
  enableHologram: z.boolean(),
  enableMetadataShape: z.boolean(),
  enableMetadataVisualisation: z.boolean(),
});

// Edge settings validation
export const edgeSettingsSchema = z.object({
  arrowSize: z.number().min(0.1, 'Arrow size must be between 0.1 and 10.0').max(10.0, 'Arrow size must be between 0.1 and 10.0'),
  baseWidth: z.number().min(0.1, 'Base width must be between 0.1 and 20.0').max(20.0, 'Base width must be between 0.1 and 20.0'),
  color: hexColor(),
  enableArrows: z.boolean(),
  opacity: opacity(),
  widthRange: z.array(z.number())
    .length(2, 'Width range must have exactly 2 values [min, max]')
    .refine(([min, max]) => min < max, 'Minimum width must be less than maximum width')
    .refine(([min, max]) => min >= 0 && max >= 0, 'Width values must be positive'),
  quality: z.string().min(1, 'Quality cannot be empty'),
});

// Physics settings validation
export const physicsSettingsSchema = z.object({
  autoBalance: z.boolean(),
  autoBalanceIntervalMs: z.number().int().min(100, 'Auto balance interval must be at least 100ms').max(10000, 'Auto balance interval must be at most 10 seconds'),
  attractionK: z.number().min(0.0, 'Attraction k must be between 0.0 and 10.0').max(10.0, 'Attraction k must be between 0.0 and 10.0'),
  boundsSize: z.number().min(10.0, 'Bounds size must be between 10.0 and 10000.0').max(10000.0, 'Bounds size must be between 10.0 and 10000.0'),
  separationRadius: z.number().min(0.1, 'Separation radius must be between 0.1 and 100.0').max(100.0, 'Separation radius must be between 0.1 and 100.0'),
  damping: opacity(),
  enableBounds: z.boolean(),
  enabled: z.boolean(),
  iterations: z.number().int().min(1, 'Iterations must be between 1 and 10000').max(10000, 'Iterations must be between 1 and 10000'),
  maxVelocity: z.number().min(0.1, 'Max velocity must be between 0.1 and 1000.0').max(1000.0, 'Max velocity must be between 0.1 and 1000.0'),
  maxForce: z.number().min(0.1, 'Max force must be between 0.1 and 10000.0').max(10000.0, 'Max force must be between 0.1 and 10000.0'),
  repelK: z.number().min(0.0, 'Repel k must be between 0.0 and 1000.0').max(1000.0, 'Repel k must be between 0.0 and 1000.0'),
  springK: z.number().min(0.0, 'Spring k must be between 0.0 and 10.0').max(10.0, 'Spring k must be between 0.0 and 10.0'),
  massScale: z.number().min(0.1, 'Mass scale must be between 0.1 and 10.0').max(10.0, 'Mass scale must be between 0.1 and 10.0'),
  boundaryDamping: opacity(),
  updateThreshold: z.number().min(0.001, 'Update threshold must be between 0.001 and 1.0').max(1.0, 'Update threshold must be between 0.001 and 1.0'),
  dt: z.number().min(0.001, 'Delta time must be between 0.001 and 1.0').max(1.0, 'Delta time must be between 0.001 and 1.0'),
  temperature: z.number().min(0.0, 'Temperature must be between 0.0 and 10.0').max(10.0, 'Temperature must be between 0.0 and 10.0'),
  gravity: z.number().min(0.0, 'Gravity must be between 0.0 and 10.0').max(10.0, 'Gravity must be between 0.0 and 10.0'),
});

// Network settings validation
export const networkSettingsSchema = z.object({
  bindAddress: z.string().min(7, 'Bind address must be between 7 and 45 characters').max(45, 'Bind address must be between 7 and 45 characters'),
  domain: z.string().min(1, 'Domain must be between 1 and 253 characters').max(253, 'Domain must be between 1 and 253 characters'),
  enableHttp2: z.boolean(),
  enableRateLimiting: z.boolean(),
  enableTls: z.boolean(),
  maxRequestSize: z.number().int().min(1024, 'Max request size must be between 1KB and 100MB').max(104857600, 'Max request size must be between 1KB and 100MB'),
  minTlsVersion: z.string().min(1, 'TLS version cannot be empty'),
  port: port(),
  rateLimitRequests: z.number().int().min(1, 'Rate limit requests must be between 1 and 10000').max(10000, 'Rate limit requests must be between 1 and 10000'),
  rateLimitWindow: z.number().int().min(1, 'Rate limit window must be between 1 and 3600 seconds').max(3600, 'Rate limit window must be between 1 and 3600 seconds'),
  tunnelId: z.string(),
  apiClientTimeout: z.number().int().min(1, 'API client timeout must be between 1 and 300 seconds').max(300, 'API client timeout must be between 1 and 300 seconds'),
  enableMetrics: z.boolean(),
  maxConcurrentRequests: z.number().int().min(1, 'Max concurrent requests must be between 1 and 1000').max(1000, 'Max concurrent requests must be between 1 and 1000'),
  maxRetries: z.number().int().min(0, 'Max retries must be between 0 and 10').max(10, 'Max retries must be between 0 and 10'),
  metricsPort: port(),
  retryDelay: z.number().int().min(100, 'Retry delay must be between 100ms and 30 seconds').max(30000, 'Retry delay must be between 100ms and 30 seconds'),
});

// WebSocket settings validation
export const webSocketSettingsSchema = z.object({
  binaryChunkSize: z.number().int().min(64, 'Binary chunk size must be between 64 bytes and 64KB').max(65536, 'Binary chunk size must be between 64 bytes and 64KB'),
  binaryUpdateRate: z.number().int().min(1, 'Binary update rate must be between 1 and 120 FPS').max(120, 'Binary update rate must be between 1 and 120 FPS'),
  minUpdateRate: z.number().int().min(1, 'Min update rate must be between 1 and 30 FPS').max(30, 'Min update rate must be between 1 and 30 FPS'),
  maxUpdateRate: z.number().int().min(30, 'Max update rate must be between 30 and 120 FPS').max(120, 'Max update rate must be between 30 and 120 FPS'),
  motionThreshold: z.number().min(0.001, 'Motion threshold must be between 0.001 and 1.0').max(1.0, 'Motion threshold must be between 0.001 and 1.0'),
  motionDamping: z.number().min(0.1, 'Motion damping must be between 0.1 and 1.0').max(1.0, 'Motion damping must be between 0.1 and 1.0'),
  binaryMessageVersion: z.number().int().min(1, 'Binary message version must be between 1 and 255').max(255, 'Binary message version must be between 1 and 255'),
  compressionEnabled: z.boolean(),
  compressionThreshold: z.number().int().min(64, 'Compression threshold must be between 64 bytes and 8KB').max(8192, 'Compression threshold must be between 64 bytes and 8KB'),
  heartbeatInterval: z.number().int().min(1000, 'Heartbeat interval must be between 1 and 60 seconds').max(60000, 'Heartbeat interval must be between 1 and 60 seconds'),
  heartbeatTimeout: z.number().int().min(5000, 'Heartbeat timeout must be between 5 seconds and 10 minutes').max(600000, 'Heartbeat timeout must be between 5 seconds and 10 minutes'),
  maxConnections: z.number().int().min(1, 'Max connections must be between 1 and 10000').max(10000, 'Max connections must be between 1 and 10000'),
  maxMessageSize: z.number().int().min(1024, 'Max message size must be between 1KB and 100MB').max(104857600, 'Max message size must be between 1KB and 100MB'),
  reconnectAttempts: z.number().int().min(0, 'Reconnect attempts must be between 0 and 20').max(20, 'Reconnect attempts must be between 0 and 20'),
  reconnectDelay: z.number().int().min(100, 'Reconnect delay must be between 100ms and 30 seconds').max(30000, 'Reconnect delay must be between 100ms and 30 seconds'),
  updateRate: z.number().int().min(1, 'Update rate must be between 1 and 120 FPS').max(120, 'Update rate must be between 1 and 120 FPS'),
});

// Security settings validation
export const securitySettingsSchema = z.object({
  allowedOrigins: z.array(z.string()),
  auditLogPath: z.string(),
  cookieHttponly: z.boolean(),
  cookieSamesite: z.string().min(1, 'Cookie samesite cannot be empty'),
  cookieSecure: z.boolean(),
  csrfTokenTimeout: z.number().int().min(60, 'CSRF token timeout must be between 1 minute and 24 hours').max(86400, 'CSRF token timeout must be between 1 minute and 24 hours'),
  enableAuditLogging: z.boolean(),
  enableRequestValidation: z.boolean(),
  sessionTimeout: z.number().int().min(300, 'Session timeout must be between 5 minutes and 24 hours').max(86400, 'Session timeout must be between 5 minutes and 24 hours'),
});

// Full settings schema
export const appFullSettingsSchema = z.object({
  visualisation: z.object({
    rendering: z.object({
      ambientLightIntensity: z.number().min(0, 'Ambient light intensity must be positive').max(10, 'Ambient light intensity must be at most 10'),
      backgroundColor: hexColor(),
      directionalLightIntensity: z.number().min(0, 'Directional light intensity must be positive').max(10, 'Directional light intensity must be at most 10'),
      enableAmbientOcclusion: z.boolean(),
      enableAntialiasing: z.boolean(),
      enableShadows: z.boolean(),
      environmentIntensity: z.number().min(0, 'Environment intensity must be positive').max(5, 'Environment intensity must be at most 5'),
      shadowMapSize: z.string().optional(),
      shadowBias: z.number().optional(),
      context: z.string().optional(),
    }),
    animations: z.object({
      enableMotionBlur: z.boolean(),
      enableNodeAnimations: z.boolean(),
      motionBlurStrength: z.number().min(0, 'Motion blur strength must be positive').max(2, 'Motion blur strength must be at most 2'),
      selectionWaveEnabled: z.boolean(),
      pulseEnabled: z.boolean(),
      pulseSpeed: z.number().min(0.1, 'Pulse speed must be between 0.1 and 5.0').max(5.0, 'Pulse speed must be between 0.1 and 5.0'),
      pulseStrength: z.number().min(0.1, 'Pulse strength must be between 0.1 and 2.0').max(2.0, 'Pulse strength must be between 0.1 and 2.0'),
      waveSpeed: z.number().min(0.1, 'Wave speed must be between 0.1 and 10.0').max(10.0, 'Wave speed must be between 0.1 and 10.0'),
    }),
    graphs: z.object({
      logseq: z.object({
        nodes: nodeSettingsSchema,
        edges: edgeSettingsSchema,
        physics: physicsSettingsSchema,
      }),
      visionflow: z.object({
        nodes: nodeSettingsSchema,
        edges: edgeSettingsSchema,
        physics: physicsSettingsSchema,
      }),
    }),
  }),
  system: z.object({
    network: networkSettingsSchema,
    websocket: webSocketSettingsSchema,
    security: securitySettingsSchema,
    debug: z.object({
      enabled: z.boolean(),
    }),
    persistSettings: z.boolean(),
    customBackendUrl: z.string().optional(),
  }),
  xr: z.object({
    enabled: z.boolean().optional(),
    clientSideEnableXr: z.boolean().optional(),
    mode: z.string().optional(),
    roomScale: z.number().min(0.1, 'Room scale must be at least 0.1').max(10.0, 'Room scale must be at most 10.0'),
    spaceType: z.string().min(1, 'Space type cannot be empty'),
    quality: z.string().min(1, 'Quality cannot be empty'),
    renderScale: z.number().min(0.1, 'Render scale must be between 0.1 and 2.0').max(2.0, 'Render scale must be between 0.1 and 2.0').optional(),
    interactionDistance: z.number().min(0.1, 'Interaction distance must be at least 0.1').max(10.0, 'Interaction distance must be at most 10.0'),
    locomotionMethod: z.string().min(1, 'Locomotion method cannot be empty'),
    teleportRayColor: hexColor(),
    controllerRayColor: hexColor(),
    controllerModel: z.string().optional(),
    enableHandTracking: z.boolean(),
    handMeshEnabled: z.boolean(),
    handMeshColor: hexColor(),
    handMeshOpacity: opacity(),
    movementAxes: movementAxesSchema,
  }),
  auth: z.object({
    enabled: z.boolean(),
    provider: z.string().min(1, 'Provider cannot be empty'),
    required: z.boolean(),
  }),
  ragflow: z.object({
    apiKey: z.string().optional(),
    agentId: z.string().optional(),
    apiBaseUrl: validUrl().optional(),
    timeout: z.number().int().min(1000, 'Timeout must be at least 1 second').max(300000, 'Timeout must be at most 5 minutes').optional(),
    maxRetries: z.number().int().min(0, 'Max retries must be at least 0').max(10, 'Max retries must be at most 10').optional(),
    chatId: z.string().optional(),
  }).optional(),
  perplexity: z.object({
    apiKey: z.string().optional(),
    model: z.string().optional(),
    apiUrl: validUrl().optional(),
    maxTokens: z.number().int().min(1, 'Max tokens must be at least 1').max(100000, 'Max tokens must be at most 100000').optional(),
    temperature: z.number().min(0.0, 'Temperature must be between 0.0 and 2.0').max(2.0, 'Temperature must be between 0.0 and 2.0').optional(),
    topP: z.number().min(0.0, 'Top P must be between 0.0 and 1.0').max(1.0, 'Top P must be between 0.0 and 1.0').optional(),
    presencePenalty: z.number().min(-2.0, 'Presence penalty must be between -2.0 and 2.0').max(2.0, 'Presence penalty must be between -2.0 and 2.0').optional(),
    frequencyPenalty: z.number().min(-2.0, 'Frequency penalty must be between -2.0 and 2.0').max(2.0, 'Frequency penalty must be between -2.0 and 2.0').optional(),
    timeout: z.number().int().min(1000, 'Timeout must be at least 1 second').max(300000, 'Timeout must be at most 5 minutes').optional(),
    rateLimit: z.number().int().min(1, 'Rate limit must be at least 1').max(1000, 'Rate limit must be at most 1000').optional(),
  }).optional(),
  openai: z.object({
    apiKey: z.string().optional(),
    baseUrl: validUrl().optional(),
    timeout: z.number().int().min(1000, 'Timeout must be at least 1 second').max(300000, 'Timeout must be at most 5 minutes').optional(),
    rateLimit: z.number().int().min(1, 'Rate limit must be at least 1').max(1000, 'Rate limit must be at most 1000').optional(),
  }).optional(),
});

// Export types
export type AppFullSettings = z.infer<typeof appFullSettingsSchema>;
export type NetworkSettings = z.infer<typeof networkSettingsSchema>;
export type PhysicsSettings = z.infer<typeof physicsSettingsSchema>;
export type NodeSettings = z.infer<typeof nodeSettingsSchema>;
export type EdgeSettings = z.infer<typeof edgeSettingsSchema>;
export type WebSocketSettings = z.infer<typeof webSocketSettingsSchema>;
export type SecuritySettings = z.infer<typeof securitySettingsSchema>;

// Validation helper function
export function validateSettingsPath(path: string, value: any): { valid: boolean; error?: string } {
  try {
    // Parse the path to determine which schema to use
    const pathParts = path.split('.');
    
    // Route to appropriate schema based on path
    if (pathParts[0] === 'system' && pathParts[1] === 'network') {
      const fieldName = pathParts[2];
      const field = networkSettingsSchema.shape[fieldName as keyof typeof networkSettingsSchema.shape];
      if (field) {
        field.parse(value);
      }
    } else if (pathParts[0] === 'visualisation' && pathParts[1] === 'graphs' && pathParts[3] === 'physics') {
      const fieldName = pathParts[4];
      const field = physicsSettingsSchema.shape[fieldName as keyof typeof physicsSettingsSchema.shape];
      if (field) {
        field.parse(value);
      }
    }
    // Add more path routing as needed
    
    return { valid: true };
  } catch (error) {
    if (error instanceof z.ZodError) {
      return { valid: false, error: error.errors[0]?.message || 'Validation failed' };
    }
    return { valid: false, error: 'Unknown validation error' };
  }
}

// Validate entire settings object
export function validateFullSettings(settings: any): { valid: boolean; errors?: z.ZodError } {
  try {
    appFullSettingsSchema.parse(settings);
    return { valid: true };
  } catch (error) {
    if (error instanceof z.ZodError) {
      return { valid: false, errors: error };
    }
    return { valid: false };
  }
}