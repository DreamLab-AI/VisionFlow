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

export { default as VoiceWebSocketService } from './VoiceWebSocketService';

// Binary Protocol
export { binaryProtocol, MessageType, GraphTypeFlag } from './BinaryWebSocketProtocol';

// Authentication
export { nostrAuth, default as nostrAuthService } from './nostrAuthService';

// Solid/JSS Pod Service
export { default as solidPodService } from './SolidPodService';
export type {
  PodInfo,
  PodCreationResult,
  JsonLdDocument,
  SolidNotification as SolidPodNotification
} from './SolidPodService';

// Audio Services
export { default as AudioContextManager } from './AudioContextManager';
export { default as AudioInputService } from './AudioInputService';
export { default as AudioOutputService } from './AudioOutputService';

// Platform & Hardware
export { default as platformManager } from './platformManager';
export { default as Quest3AutoDetector } from './quest3AutoDetector';
export { default as SpaceDriverService } from './SpaceDriverService';

// API Services
export { default as interactionApi } from './interactionApi';
export { default as remoteLogger } from './remoteLogger';
