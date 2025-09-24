/**
 * Telemetry System - Client-side Agent Monitoring and Debug Overlay
 *
 * This module provides comprehensive telemetry for the hive mind visualization:
 * - Agent spawn/state tracking
 * - WebSocket message flow monitoring
 * - Three.js rendering and position debugging
 * - Performance metrics and frame rate monitoring
 * - Real-time debug overlay with position tracking
 * - Offline logging buffer with localStorage backup
 * - Log upload to backend for analysis
 */

export { AgentTelemetryService, agentTelemetry } from './AgentTelemetry';
export type {
  TelemetryMetrics,
  TelemetryUploadPayload
} from './AgentTelemetry';

export {
  useTelemetry,
  useThreeJSTelemetry,
  useWebSocketTelemetry
} from './useTelemetry';

export {
  DebugOverlay,
  useDebugOverlay
} from './DebugOverlay';

// Re-export enhanced logger types
export type {
  AgentTelemetryData,
  WebSocketTelemetryData,
  ThreeJSTelemetryData
} from '../utils/loggerConfig';

export {
  createAgentLogger,
  agentTelemetryLogger
} from '../utils/loggerConfig';

/**
 * Quick setup helper for adding telemetry to any React component
 */
export function setupTelemetry(componentName: string) {
  const telemetry = useTelemetry(componentName);

  // Return telemetry methods with component name context
  return {
    ...telemetry,
    logStart: () => telemetry.startRender(),
    logEnd: () => telemetry.endRender(),
    logClick: (metadata?: Record<string, any>) =>
      telemetry.logInteraction('click', metadata),
    logHover: (metadata?: Record<string, any>) =>
      telemetry.logInteraction('hover', metadata)
  };
}

/**
 * Initialize telemetry system - call once in your app
 */
export function initializeTelemetry() {
  // Auto-start the telemetry service
  agentTelemetry.logAgentAction('system', 'telemetry', 'initialized', {
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    viewport: {
      width: window.innerWidth,
      height: window.innerHeight
    }
  });

  console.log('🔍 Agent Telemetry System Initialized');
  console.log('📊 Use Ctrl+Shift+D to toggle debug overlay');
  console.log('🤖 All agent spawns, WebSocket messages, and Three.js operations are being logged');
}