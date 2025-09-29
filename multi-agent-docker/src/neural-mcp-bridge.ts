/**
 * Neural MCP Bridge
 * Connects codex-syntaptic neural processing with MCP protocol bridges
 */

import { EventEmitter } from 'events';
import { spawn, ChildProcess } from 'child_process';
import pino from 'pino';
import { CodexSyntaptic } from 'codex-syntaptic';

interface MCPBridgeConfig {
  enabled: boolean;
  bridges: string[];
  neuralEnhancement: boolean;
  adaptiveRouting: boolean;
}

interface MCPConnection {
  name: string;
  process: ChildProcess;
  port: number;
  status: 'connecting' | 'connected' | 'disconnected' | 'error';
  lastPing: Date;
  neuralContext: any;
}

interface NeuralMCPMessage {
  id: string;
  type: 'request' | 'response' | 'notification';
  method?: string;
  params?: any;
  result?: any;
  error?: any;
  neuralEnhanced?: boolean;
  contextVector?: number[];
  priority?: 'low' | 'medium' | 'high' | 'critical';
}

class NeuralMCPBridge extends EventEmitter {
  private config: MCPBridgeConfig;
  private logger: pino.Logger;
  private connections: Map<string, MCPConnection>;
  private codexSyntaptic?: CodexSyntaptic;
  private neuralContext: Map<string, any>;
  private messageQueue: NeuralMCPMessage[];
  private processingQueue: boolean;

  constructor(config: MCPBridgeConfig, logger: pino.Logger) {
    super();
    this.config = config;
    this.logger = logger.child({ component: 'neural-mcp-bridge' });
    this.connections = new Map();
    this.neuralContext = new Map();
    this.messageQueue = [];
    this.processingQueue = false;

    this.initializeBridges();
  }

  private async initializeBridges(): Promise<void> {
    try {
      this.logger.info('Initializing neural MCP bridges', { bridges: this.config.bridges });

      for (const bridgeName of this.config.bridges) {
        await this.createBridge(bridgeName);
      }

      if (this.config.neuralEnhancement) {
        this.initializeNeuralProcessing();
      }

      this.startMessageProcessor();
      this.logger.info('Neural MCP bridges initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize neural MCP bridges', { error });
      throw error;
    }
  }

  private async createBridge(bridgeName: string): Promise<void> {
    try {
      const port = this.getPortForBridge(bridgeName);
      let command: string;
      let args: string[];

      switch (bridgeName) {
        case 'claude-flow':
          command = 'npx';
          args = ['claude-flow@alpha', 'mcp', 'start', '--port', port.toString()];
          break;
        case 'ruv-swarm':
          command = 'npx';
          args = ['ruv-swarm', 'mcp', 'start', '--port', port.toString()];
          break;
        case 'flow-nexus':
          command = 'npx';
          args = ['flow-nexus@latest', 'mcp', 'start', '--port', port.toString()];
          break;
        default:
          throw new Error(`Unknown bridge: ${bridgeName}`);
      }

      const process = spawn(command, args, {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: {
          ...process.env,
          NEURAL_ENHANCED: 'true',
          MCP_BRIDGE_MODE: 'neural'
        }
      });

      const connection: MCPConnection = {
        name: bridgeName,
        process,
        port,
        status: 'connecting',
        lastPing: new Date(),
        neuralContext: {}
      };

      this.connections.set(bridgeName, connection);

      process.stdout?.on('data', (data) => {
        this.handleBridgeOutput(bridgeName, data.toString());
      });

      process.stderr?.on('data', (data) => {
        this.logger.warn('Bridge stderr', { bridge: bridgeName, data: data.toString() });
      });

      process.on('close', (code) => {
        this.handleBridgeClose(bridgeName, code);
      });

      process.on('error', (error) => {
        this.handleBridgeError(bridgeName, error);
      });

      // Wait for bridge to be ready
      await this.waitForBridgeReady(bridgeName);

      this.logger.info('Bridge created successfully', { bridge: bridgeName, port });
    } catch (error) {
      this.logger.error('Failed to create bridge', { bridge: bridgeName, error });
      throw error;
    }
  }

  private getPortForBridge(bridgeName: string): number {
    const basePorts = {
      'claude-flow': 9700,
      'ruv-swarm': 9710,
      'flow-nexus': 9720
    };
    return basePorts[bridgeName] || 9730;
  }

  private async waitForBridgeReady(bridgeName: string, timeout = 30000): Promise<void> {
    return new Promise((resolve, reject) => {
      const connection = this.connections.get(bridgeName);
      if (!connection) {
        reject(new Error(`Bridge ${bridgeName} not found`));
        return;
      }

      const startTime = Date.now();
      const checkReady = () => {
        if (connection.status === 'connected') {
          resolve();
        } else if (Date.now() - startTime > timeout) {
          reject(new Error(`Bridge ${bridgeName} failed to connect within timeout`));
        } else {
          setTimeout(checkReady, 100);
        }
      };

      checkReady();
    });
  }

  private handleBridgeOutput(bridgeName: string, data: string): void {
    const connection = this.connections.get(bridgeName);
    if (!connection) return;

    try {
      // Parse potential JSON messages
      const lines = data.split('\n').filter(line => line.trim());
      for (const line of lines) {
        try {
          const message = JSON.parse(line);
          this.handleBridgeMessage(bridgeName, message);
        } catch {
          // Not JSON, treat as log output
          if (line.includes('ready') || line.includes('listening')) {
            connection.status = 'connected';
            this.emit('bridge-connected', bridgeName);
          }
        }
      }
    } catch (error) {
      this.logger.error('Error handling bridge output', { bridge: bridgeName, error });
    }
  }

  private async handleBridgeMessage(bridgeName: string, message: any): Promise<void> {
    try {
      const neuralMessage: NeuralMCPMessage = {
        id: message.id || `${bridgeName}-${Date.now()}`,
        type: message.type || 'notification',
        method: message.method,
        params: message.params,
        result: message.result,
        error: message.error,
        neuralEnhanced: false
      };

      // Apply neural enhancement if configured
      if (this.config.neuralEnhancement && this.codexSyntaptic) {
        await this.enhanceMessageWithNeural(neuralMessage, bridgeName);
      }

      this.messageQueue.push(neuralMessage);
      this.emit('message-received', { bridge: bridgeName, message: neuralMessage });
    } catch (error) {
      this.logger.error('Error handling bridge message', { bridge: bridgeName, error });
    }
  }

  private async enhanceMessageWithNeural(message: NeuralMCPMessage, bridgeName: string): Promise<void> {
    try {
      if (!this.codexSyntaptic) return;

      // Create context vector for the message
      const contextData = {
        bridge: bridgeName,
        method: message.method,
        params: message.params,
        timestamp: new Date().toISOString()
      };

      const contextVector = await this.codexSyntaptic.generateEmbedding(JSON.stringify(contextData));
      message.contextVector = contextVector;

      // Analyze message priority using neural processing
      const priorityAnalysis = await this.codexSyntaptic.analyzePriority(message);
      message.priority = priorityAnalysis.priority;

      // Store neural context for future reference
      this.neuralContext.set(message.id, {
        vector: contextVector,
        priority: message.priority,
        bridge: bridgeName,
        timestamp: new Date()
      });

      message.neuralEnhanced = true;
    } catch (error) {
      this.logger.error('Error enhancing message with neural processing', { messageId: message.id, error });
    }
  }

  private handleBridgeClose(bridgeName: string, code: number | null): void {
    const connection = this.connections.get(bridgeName);
    if (connection) {
      connection.status = 'disconnected';
      this.logger.warn('Bridge process closed', { bridge: bridgeName, code });
      this.emit('bridge-disconnected', bridgeName);

      // Attempt to reconnect after a delay
      setTimeout(() => {
        this.reconnectBridge(bridgeName);
      }, 5000);
    }
  }

  private handleBridgeError(bridgeName: string, error: Error): void {
    const connection = this.connections.get(bridgeName);
    if (connection) {
      connection.status = 'error';
      this.logger.error('Bridge process error', { bridge: bridgeName, error });
      this.emit('bridge-error', { bridge: bridgeName, error });
    }
  }

  private async reconnectBridge(bridgeName: string): Promise<void> {
    try {
      this.logger.info('Attempting to reconnect bridge', { bridge: bridgeName });
      await this.createBridge(bridgeName);
    } catch (error) {
      this.logger.error('Failed to reconnect bridge', { bridge: bridgeName, error });
    }
  }

  private initializeNeuralProcessing(): void {
    try {
      this.codexSyntaptic = new CodexSyntaptic({
        mode: 'mcp-bridge',
        enableContextVectors: true,
        enablePriorityAnalysis: true,
        enableAdaptiveRouting: this.config.adaptiveRouting
      });

      this.logger.info('Neural processing initialized for MCP bridge');
    } catch (error) {
      this.logger.error('Failed to initialize neural processing', { error });
    }
  }

  private startMessageProcessor(): void {
    if (this.processingQueue) return;

    this.processingQueue = true;
    this.processMessageQueue();
  }

  private async processMessageQueue(): Promise<void> {
    while (this.processingQueue) {
      try {
        if (this.messageQueue.length > 0) {
          const message = this.messageQueue.shift();
          if (message) {
            await this.processMessage(message);
          }
        }
        await new Promise(resolve => setTimeout(resolve, 10)); // Small delay to prevent tight loop
      } catch (error) {
        this.logger.error('Error processing message queue', { error });
      }
    }
  }

  private async processMessage(message: NeuralMCPMessage): Promise<void> {
    try {
      // Apply adaptive routing if configured
      if (this.config.adaptiveRouting && message.neuralEnhanced) {
        const optimalBridge = await this.selectOptimalBridge(message);
        if (optimalBridge) {
          await this.routeMessageToBridge(message, optimalBridge);
        }
      }

      this.emit('message-processed', message);
    } catch (error) {
      this.logger.error('Error processing message', { messageId: message.id, error });
    }
  }

  private async selectOptimalBridge(message: NeuralMCPMessage): Promise<string | null> {
    try {
      if (!this.codexSyntaptic || !message.contextVector) return null;

      const bridgeScores = new Map<string, number>();

      for (const [bridgeName, connection] of this.connections) {
        if (connection.status === 'connected') {
          const score = await this.calculateBridgeScore(bridgeName, message);
          bridgeScores.set(bridgeName, score);
        }
      }

      if (bridgeScores.size === 0) return null;

      // Return bridge with highest score
      const sortedBridges = Array.from(bridgeScores.entries()).sort((a, b) => b[1] - a[1]);
      return sortedBridges[0][0];
    } catch (error) {
      this.logger.error('Error selecting optimal bridge', { error });
      return null;
    }
  }

  private async calculateBridgeScore(bridgeName: string, message: NeuralMCPMessage): Promise<number> {
    try {
      // This would use neural analysis to determine the best bridge for the message
      // For now, using a simple heuristic
      let score = 0.5; // Base score

      // Priority-based scoring
      if (message.priority === 'critical') score += 0.3;
      else if (message.priority === 'high') score += 0.2;
      else if (message.priority === 'medium') score += 0.1;

      // Bridge-specific scoring based on message type
      if (message.method?.includes('swarm') && bridgeName === 'ruv-swarm') score += 0.3;
      if (message.method?.includes('flow') && bridgeName === 'claude-flow') score += 0.3;
      if (message.method?.includes('nexus') && bridgeName === 'flow-nexus') score += 0.3;

      return Math.min(score, 1.0);
    } catch (error) {
      this.logger.error('Error calculating bridge score', { bridge: bridgeName, error });
      return 0.0;
    }
  }

  private async routeMessageToBridge(message: NeuralMCPMessage, bridgeName: string): Promise<void> {
    try {
      const connection = this.connections.get(bridgeName);
      if (!connection || connection.status !== 'connected') {
        throw new Error(`Bridge ${bridgeName} not available`);
      }

      const messageData = JSON.stringify(message) + '\n';
      connection.process.stdin?.write(messageData);

      this.logger.debug('Message routed to bridge', { messageId: message.id, bridge: bridgeName });
    } catch (error) {
      this.logger.error('Error routing message to bridge', { messageId: message.id, bridge: bridgeName, error });
    }
  }

  public async connect(): Promise<void> {
    if (!this.config.enabled) {
      this.logger.info('MCP bridge disabled, skipping connection');
      return;
    }

    try {
      this.logger.info('Connecting neural MCP bridges...');
      // Bridges are already created during initialization
      // This method ensures all are connected
      const connectionPromises = Array.from(this.connections.keys()).map(bridgeName =>
        this.waitForBridgeReady(bridgeName)
      );

      await Promise.all(connectionPromises);
      this.logger.info('All neural MCP bridges connected successfully');
    } catch (error) {
      this.logger.error('Failed to connect neural MCP bridges', { error });
      throw error;
    }
  }

  public async disconnect(): Promise<void> {
    try {
      this.logger.info('Disconnecting neural MCP bridges...');
      this.processingQueue = false;

      for (const [bridgeName, connection] of this.connections) {
        if (connection.process && !connection.process.killed) {
          connection.process.kill('SIGTERM');

          // Wait for graceful shutdown or force kill after timeout
          setTimeout(() => {
            if (!connection.process.killed) {
              connection.process.kill('SIGKILL');
            }
          }, 5000);
        }
      }

      this.connections.clear();
      this.neuralContext.clear();
      this.messageQueue.length = 0;

      this.logger.info('Neural MCP bridges disconnected successfully');
    } catch (error) {
      this.logger.error('Error disconnecting neural MCP bridges', { error });
      throw error;
    }
  }

  public isConnected(): boolean {
    if (!this.config.enabled) return false;

    return Array.from(this.connections.values()).some(conn => conn.status === 'connected');
  }

  public getConnectedBridges(): string[] {
    return Array.from(this.connections.entries())
      .filter(([_, conn]) => conn.status === 'connected')
      .map(([name, _]) => name);
  }

  public getBridgeStatus(bridgeName: string): string | null {
    const connection = this.connections.get(bridgeName);
    return connection ? connection.status : null;
  }

  public async sendMessage(bridgeName: string, message: any): Promise<void> {
    const connection = this.connections.get(bridgeName);
    if (!connection || connection.status !== 'connected') {
      throw new Error(`Bridge ${bridgeName} not connected`);
    }

    const neuralMessage: NeuralMCPMessage = {
      id: `send-${Date.now()}`,
      type: 'request',
      ...message
    };

    if (this.config.neuralEnhancement) {
      await this.enhanceMessageWithNeural(neuralMessage, bridgeName);
    }

    await this.routeMessageToBridge(neuralMessage, bridgeName);
  }
}

export { NeuralMCPBridge, MCPBridgeConfig, NeuralMCPMessage };