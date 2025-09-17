import { createAgentLogger, AgentTelemetryData, WebSocketTelemetryData, ThreeJSTelemetryData } from '../utils/logger';

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

/**
 * Centralized telemetry service for agent monitoring and debugging
 */
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
    // Monitor memory usage if available
    if ('memory' in performance && (performance as any).memory) {
      const updateMemory = () => {
        const memory = (performance as any).memory;
        this.metrics.memoryUsage = memory.usedJSHeapSize;
      };
      setInterval(updateMemory, 5000);
    }

    // Monitor error count
    window.addEventListener('error', () => {
      this.metrics.errorCount++;
    });

    window.addEventListener('unhandledrejection', () => {
      this.metrics.errorCount++;
    });
  }

  private startAutoUpload() {
    // Poll REST endpoint for all agent telemetry/metadata every 10 seconds
    // WebSocket handles high-speed position/velocity/SSSP data
    // REST handles metadata, telemetry, and agent details
    this.uploadInterval = setInterval(() => {
      this.fetchAgentTelemetry().catch(error => {
        this.logger.error('Failed to fetch agent telemetry:', error);
      });
    }, 10000); // Poll every 10 seconds for telemetry updates
  }

  // Public telemetry methods
  logAgentSpawn(agentId: string, agentType: string, metadata?: Record<string, any>) {
    this.metrics.agentSpawns++;
    this.logger.logAgentAction(agentId, agentType, 'spawn', metadata);

    console.group(`🤖 Agent Spawned: ${agentType}:${agentId}`);
    console.log('Agent Type:', agentType);
    console.log('Agent ID:', agentId);
    console.log('Metadata:', metadata);
    console.log('Total Spawned:', this.metrics.agentSpawns);
    console.groupEnd();
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

    if (direction === 'incoming') {
      console.group(`📥 WebSocket Message: ${messageType}`);
    } else {
      console.group(`📤 WebSocket Message: ${messageType}`);
    }
    console.log('Type:', messageType);
    console.log('Direction:', direction);
    console.log('Size:', size ? `${size} bytes` : 'unknown');
    console.log('Data:', data);
    console.groupEnd();
  }

  logThreeJSOperation(action: ThreeJSTelemetryData['action'], objectId: string, position?: { x: number; y: number; z: number }, rotation?: { x: number; y: number; z: number }, metadata?: Record<string, any>) {
    this.metrics.threeJSOperations++;
    this.logger.logThreeJSAction(action, objectId, position, rotation, metadata);
  }

  logRenderCycle(frameTime: number) {
    this.metrics.renderCycles++;

    // Track frame time for performance analysis
    this.frameTimeBuffer.push(frameTime);
    if (this.frameTimeBuffer.length > 60) { // Keep last 60 frames
      this.frameTimeBuffer.shift();
    }

    // Update average frame time
    this.metrics.averageFrameTime = this.frameTimeBuffer.reduce((a, b) => a + b, 0) / this.frameTimeBuffer.length;

    // Log performance issues
    if (frameTime > 50) { // Slower than 20fps
      console.warn(`⚡ PERFORMANCE: Slow frame detected - ${frameTime.toFixed(2)}ms`);
    }

    this.logger.logPerformance('render_cycle', frameTime);
  }

  logUserInteraction(interactionType: string, target: string, metadata?: Record<string, any>) {
    console.group(`👆 User Interaction: ${interactionType}`);
    console.log('Target:', target);
    console.log('Metadata:', metadata);
    console.groupEnd();

    this.logger.logAgentAction('user', 'interaction', interactionType, { target, ...metadata });
  }

  // Debug overlay data
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

  // Fetch all agent telemetry and metadata from REST endpoints
  // This includes: agent status, CPU/memory usage, health, workload, tasks, etc.
  // Position/velocity data comes via WebSocket binary protocol separately
  async fetchAgentTelemetry(): Promise<any> {
    try {
      // Fetch both telemetry status and full agent data
      const [statusResponse, dataResponse] = await Promise.all([
        fetch('/api/bots/status', {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        }),
        fetch('/api/bots/data', {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        })
      ]);

      if (statusResponse.ok && dataResponse.ok) {
        const [telemetryData, agentData] = await Promise.all([
          statusResponse.json(),
          dataResponse.json()
        ]);

        // Merge telemetry and agent metadata
        const mergedData = {
          ...telemetryData,
          agents: agentData.agents || telemetryData.agents || []
        };

        this.logger.info(`Fetched telemetry for ${mergedData.agents?.length || 0} agents`);

        // Process and cache the telemetry data
        if (mergedData.agents) {
          this.processAgentTelemetry(mergedData.agents);
          this.cacheAgentTelemetry(mergedData);
        }

        return mergedData;
      } else {
        throw new Error(`Failed to fetch telemetry: status=${statusResponse.status}, data=${dataResponse.status}`);
      }
    } catch (error) {
      this.logger.error('Failed to fetch agent telemetry:', error);
      // Use cached telemetry if available
      return this.getCachedTelemetry();
    }
  }

  // Cache telemetry data for offline use
  private cacheAgentTelemetry(data: any) {
    try {
      const cacheKey = `agent-telemetry-cache-${this.sessionId}`;
      localStorage.setItem(cacheKey, JSON.stringify({
        timestamp: Date.now(),
        data: data
      }));
    } catch (e) {
      // Ignore cache errors
    }
  }

  // Process telemetry received from server
  private processAgentTelemetry(agents: any[]) {
    agents.forEach(agent => {
      // Store agent telemetry for local visualization
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

  // Get cached telemetry for offline use
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

  // Cleanup
  destroy() {
    if (this.uploadInterval) {
      clearInterval(this.uploadInterval);
    }
    // No final upload needed - telemetry flows from agents to server to client
  }
}

// Export singleton instance
export const agentTelemetry = AgentTelemetryService.getInstance();