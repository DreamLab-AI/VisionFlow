

export { AgentTelemetryService, agentTelemetry } from './AgentTelemetry';
import { agentTelemetry } from './AgentTelemetry';
import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('telemetry');
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
  dynamicAgentTelemetryLogger
} from '../utils/loggerConfig';


export function setupTelemetry(_componentName: string) {
  // Note: useTelemetry is a React hook and cannot be called outside of React components
  // This function provides a stub interface for non-React usage
  return {
    startRender: () => {},
    endRender: () => {},
    logInteraction: (_type: string, _metadata?: Record<string, any>) => {},
    logError: (_error: Error, _context?: string) => {},
    logStart: () => {},
    logEnd: () => {},
    logClick: (_metadata?: Record<string, any>) => {},
    logHover: (_metadata?: Record<string, any>) => {}
  };
}


export function initializeTelemetry() {
  agentTelemetry.logAgentSpawn('system', 'telemetry', {
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    viewport: {
      width: window.innerWidth,
      height: window.innerHeight
    }
  });

  logger.info('Agent Telemetry System Initialized');
  logger.info('Use Ctrl+Shift+D to toggle debug overlay');
  logger.info('All agent spawns, WebSocket messages, and Three.js operations are being logged');
}