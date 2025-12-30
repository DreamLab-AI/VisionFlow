/**
 * Services Index
 *
 * Central export point for all client services
 */

// WebSocket Services
export { webSocketService, default as WebSocketService } from './WebSocketService';
export type {
  WebSocketAdapter,
  LegacyWebSocketMessage,
  WebSocketErrorFrame,
  QueuedMessage,
  ConnectionState,
  SolidNotification,
  SolidNotificationCallback
} from './WebSocketService';

export { VoiceWebSocketService } from './VoiceWebSocketService';

// Binary Protocol
export { binaryProtocol, MessageType, GraphTypeFlag } from './BinaryWebSocketProtocol';

// Authentication
export { nostrAuth } from './nostrAuthService';

// Solid/JSS Pod Service
export { default as solidPodService } from './SolidPodService';
export type {
  PodInfo,
  PodCreationResult,
  JsonLdDocument,
  SolidNotification as SolidPodNotification
} from './SolidPodService';

// Audio Services
export { AudioContextManager } from './AudioContextManager';
export { AudioInputService } from './AudioInputService';
export { AudioOutputService } from './AudioOutputService';

// Platform & Hardware
export { platformManager } from './platformManager';
export { quest3AutoDetector, Quest3AutoDetector } from './quest3AutoDetector';
export { SpaceDriver } from './SpaceDriverService';

// API Services
export { interactionApi } from './interactionApi';
export { remoteLogger, createRemoteLogger } from './remoteLogger';
