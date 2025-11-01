

export { AgentTelemetryService, agentTelemetry } from './AgentTelemetry';
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
  agentTelemetryLogger
} from '../utils/loggerConfig';


export function setupTelemetry(componentName: string) {
  const telemetry = useTelemetry(componentName);

  
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


export function initializeTelemetry() {
  
  agentTelemetry.logAgentAction('system', 'telemetry', 'initialized', {
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