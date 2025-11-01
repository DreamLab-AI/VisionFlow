import { createAgentLogger } from '../utils/loggerConfig';
import { AgentTelemetryData, WebSocketTelemetryData, ThreeJSTelemetryData } from '../utils/loggerConfig';
import { unifiedApiClient } from '../services/api/UnifiedApiClient';

export interface TelemetryMetrics {
  agentSpawns: number;
  webSocketMessages: number;
  threeJSOperations: number;
  renderCycles: number;
  averageFrameTime: number;
  memoryUsage?: number;
  errorCount: number;
}

export interface TelemetryUploadPayload {
  sessionId: string;
  timestamp: Date;
  metrics: TelemetryMetrics;
  agentTelemetry: AgentTelemetryData[];
  webSocketTelemetry: WebSocketTelemetryData[];
  threeJSTelemetry: ThreeJSTelemetryData[];
  systemInfo: {
    userAgent: string;
    viewport: { width: number; height: number };
    pixelRatio: number;
    webglRenderer?: string;
  };
}


export class AgentTelemetryService {
  private static instance: AgentTelemetryService;
  private logger = createAgentLogger('AgentTelemetryService');
  private sessionId: string;
  private metrics: TelemetryMetrics;
  private uploadInterval: NodeJS.Timeout | null = null;
  private frameTimeBuffer: number[] = [];
  private lastFrameTime = 0;

  private constructor() {
    this.sessionId = this.generateSessionId();
    this.metrics = this.initializeMetrics();
    this.setupPerformanceObserver();
    this.startAutoUpload();
  }

  static getInstance(): AgentTelemetryService {
    if (!AgentTelemetryService.instance) {
      AgentTelemetryService.instance = new AgentTelemetryService();
    }
    return AgentTelemetryService.instance;
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private initializeMetrics(): TelemetryMetrics {
    return {
      agentSpawns: 0,
      webSocketMessages: 0,
      threeJSOperations: 0,
      renderCycles: 0,
      averageFrameTime: 0,
      errorCount: 0
    };
  }

  private setupPerformanceObserver() {
    
    if ('memory' in performance && (performance as any).memory) {
      const updateMemory = () => {
        const memory = (performance as any).memory;
        this.metrics.memoryUsage = memory.usedJSHeapSize;
      };
      setInterval(updateMemory, 5000);
    }

    
    window.addEventListener('error', () => {
      this.metrics.errorCount++;
    });

    window.addEventListener('unhandledrejection', () => {
      this.metrics.errorCount++;
    });
  }

  private startAutoUpload() {
    
    
    
    this.uploadInterval = setInterval(() => {
      this.fetchAgentTelemetry().catch(error => {
        this.logger.error('Failed to fetch agent telemetry:', error);
      });
    }, 30000); 
  }

  
  logAgentSpawn(agentId: string, agentType: string, metadata?: Record<string, any>) {
    this.metrics.agentSpawns++;
    this.logger.logAgentAction(agentId, agentType, 'spawn', metadata);

    this.logger.debug('Agent Spawned', {
      agentType,
      agentId,
      metadata,
      totalSpawned: this.metrics.agentSpawns
    });
  }

  logAgentAction(agentId: string, agentType: string, action: string, metadata?: Record<string, any>, position?: { x: number; y: number; z: number }) {
    this.logger.logAgentAction(agentId, agentType, action, metadata, position);
  }

  logWebSocketMessage(messageType: string, direction: 'incoming' | 'outgoing', data?: any, size?: number) {
    this.metrics.webSocketMessages++;

    const metadata = {
      hasData: !!data,
      dataKeys: data && typeof data === 'object' ? Object.keys(data) : []
    };

    this.logger.logWebSocketMessage(messageType, direction, metadata, size);

    this.logger.debug('WebSocket Message', {
      messageType,
      direction,
      size: size ? `${size} bytes` : 'unknown',
      data
    });
  }

  logThreeJSOperation(action: ThreeJSTelemetryData['action'], objectId: string, position?: { x: number; y: number; z: number }, rotation?: { x: number; y: number; z: number }, metadata?: Record<string, any>) {
    this.metrics.threeJSOperations++;
    this.logger.logThreeJSAction(action, objectId, position, rotation, metadata);
  }

  logRenderCycle(frameTime: number) {
    this.metrics.renderCycles++;

    
    this.frameTimeBuffer.push(frameTime);
    if (this.frameTimeBuffer.length > 60) { 
      this.frameTimeBuffer.shift();
    }

    
    this.metrics.averageFrameTime = this.frameTimeBuffer.reduce((a, b) => a + b, 0) / this.frameTimeBuffer.length;

    
    if (frameTime > 50) { 
      console.warn(`⚡ PERFORMANCE: Slow frame detected - ${frameTime.toFixed(2)}ms`);
    }

    this.logger.logPerformance('render_cycle', frameTime);
  }

  logUserInteraction(interactionType: string, target: string, metadata?: Record<string, any>) {
    this.logger.debug('User Interaction', {
      interactionType,
      target,
      metadata
    });

    this.logger.logAgentAction('user', 'interaction', interactionType, { target, ...metadata });
  }

  
  getDebugOverlayData() {
    return {
      sessionId: this.sessionId,
      metrics: { ...this.metrics },
      recentFrameTimes: [...this.frameTimeBuffer.slice(-10)],
      agentTelemetry: this.logger.getAgentTelemetry().slice(-10),
      webSocketTelemetry: this.logger.getWebSocketTelemetry().slice(-10),
      threeJSTelemetry: this.logger.getThreeJSTelemetry().slice(-10)
    };
  }

  
  
  
  async fetchAgentTelemetry(): Promise<any> {
    try {
      
      const [statusResponse, dataResponse] = await Promise.all([
        unifiedApiClient.get('/api/bots/status'),
        unifiedApiClient.get('/api/bots/data')
      ]);

      
      const telemetryData = statusResponse.data;
      const agentData = dataResponse.data;

        
        const mergedData = {
          ...telemetryData,
          agents: agentData.agents || telemetryData.agents || []
        };

        this.logger.info(`Fetched telemetry for ${mergedData.agents?.length || 0} agents`);

        
        if (mergedData.agents) {
          this.processAgentTelemetry(mergedData.agents);
          this.cacheAgentTelemetry(mergedData);
        }

        return mergedData;
    } catch (error) {
      this.logger.error('Failed to fetch agent telemetry:', error);
      
      return this.getCachedTelemetry();
    }
  }

  
  private cacheAgentTelemetry(data: any) {
    try {
      const cacheKey = `agent-telemetry-cache-${this.sessionId}`;
      localStorage.setItem(cacheKey, JSON.stringify({
        timestamp: Date.now(),
        data: data
      }));
    } catch (e) {
      
    }
  }

  
  private processAgentTelemetry(agents: any[]) {
    agents.forEach(agent => {
      
      this.logger.logAgentMessage({
        type: 'telemetry-update',
        agentId: agent.id,
        agentType: agent.type,
        status: agent.status,
        metrics: {
          cpuUsage: agent.cpuUsage,
          memoryUsage: agent.memoryUsage,
          health: agent.health,
          workload: agent.workload
        },
        timestamp: new Date()
      });
    });
  }

  
  private getCachedTelemetry(): any {
    try {
      const cacheKey = `agent-telemetry-cache-${this.sessionId}`;
      const cached = localStorage.getItem(cacheKey);
      return cached ? JSON.parse(cached) : null;
    } catch (e) {
      return null;
    }
  }

  private getWebGLRenderer(): string | undefined {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      if (!gl) return undefined;

      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      if (debugInfo) {
        return gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
      }
      return undefined;
    } catch (e) {
      return undefined;
    }
  }

  private storeOfflineTelemetry() {
    try {
      const offlineKey = `offline-telemetry-${this.sessionId}`;
      const data = {
        metrics: this.metrics,
        agentTelemetry: this.logger.getAgentTelemetry(),
        webSocketTelemetry: this.logger.getWebSocketTelemetry(),
        threeJSTelemetry: this.logger.getThreeJSTelemetry(),
        timestamp: new Date().toISOString()
      };
      localStorage.setItem(offlineKey, JSON.stringify(data));
    } catch (e) {
      this.logger.warn('Failed to store offline telemetry:', e);
    }
  }

  
  destroy() {
    if (this.uploadInterval) {
      clearInterval(this.uploadInterval);
    }
    
  }
}

// Export singleton instance
export const agentTelemetry = AgentTelemetryService.getInstance();