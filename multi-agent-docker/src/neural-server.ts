#!/usr/bin/env tsx

/**
 * Neural-Enhanced Multi-Agent Server
 * Integrates codex-syntaptic with advanced neural processing capabilities
 */

import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import pino from 'pino';
import { v4 as uuidv4 } from 'uuid';
import { CodexSyntaptic } from 'codex-syntaptic';
import { NeuralMCPBridge } from './neural-mcp-bridge';
import { NeuralWebSocketHandler } from './neural-websocket';
import { NeuralAPIGateway } from './neural-api-gateway';
import { NeuralResourceManager } from './neural-resource-manager';
import { NeuralMonitoring } from './neural-monitoring';

interface ServerConfig {
  port: number;
  host: string;
  neural: {
    enabled: boolean;
    models: string[];
    computeUnits: number;
    memoryLimit: string;
  };
  mcp: {
    enabled: boolean;
    bridges: string[];
  };
  monitoring: {
    enabled: boolean;
    metrics: string[];
    interval: number;
  };
}

class NeuralServer {
  private app: express.Application;
  private server: any;
  private wss: WebSocketServer;
  private logger: pino.Logger;
  private config: ServerConfig;
  private codexSyntaptic: CodexSyntaptic;
  private mcpBridge: NeuralMCPBridge;
  private wsHandler: NeuralWebSocketHandler;
  private apiGateway: NeuralAPIGateway;
  private resourceManager: NeuralResourceManager;
  private monitoring: NeuralMonitoring;
  private sessionId: string;

  constructor(config: Partial<ServerConfig> = {}) {
    this.sessionId = uuidv4();
    this.logger = pino({
      name: 'neural-server',
      level: process.env.LOG_LEVEL || 'info',
      transport: {
        target: 'pino-pretty',
        options: {
          colorize: true,
          translateTime: 'SYS:standard'
        }
      }
    });

    this.config = {
      port: config.port || parseInt(process.env.NEURAL_PORT || '9600'),
      host: config.host || process.env.NEURAL_HOST || '0.0.0.0',
      neural: {
        enabled: config.neural?.enabled ?? true,
        models: config.neural?.models || ['gpt-4', 'claude-3', 'gemini-pro'],
        computeUnits: config.neural?.computeUnits || 8,
        memoryLimit: config.neural?.memoryLimit || '16GB'
      },
      mcp: {
        enabled: config.mcp?.enabled ?? true,
        bridges: config.mcp?.bridges || ['claude-flow', 'ruv-swarm', 'flow-nexus']
      },
      monitoring: {
        enabled: config.monitoring?.enabled ?? true,
        metrics: config.monitoring?.metrics || ['cpu', 'memory', 'gpu', 'neural'],
        interval: config.monitoring?.interval || 5000
      }
    };

    this.initializeComponents();
  }

  private async initializeComponents(): Promise<void> {
    try {
      this.logger.info('Initializing neural server components...', { sessionId: this.sessionId });

      // Initialize Express app
      this.app = express();
      this.app.use(cors());
      this.app.use(express.json({ limit: '100mb' }));
      this.app.use(express.urlencoded({ extended: true, limit: '100mb' }));

      // Initialize HTTP server
      this.server = createServer(this.app);

      // Initialize WebSocket server
      this.wss = new WebSocketServer({
        server: this.server,
        path: '/neural-ws',
        clientTracking: true
      });

      // Initialize codex-syntaptic
      this.codexSyntaptic = new CodexSyntaptic({
        models: this.config.neural.models,
        computeUnits: this.config.neural.computeUnits,
        memoryLimit: this.config.neural.memoryLimit,
        neuralProcessing: true,
        distributedCompute: true,
        realTimeOptimization: true
      });

      // Initialize neural components
      this.resourceManager = new NeuralResourceManager(this.logger);
      this.monitoring = new NeuralMonitoring(this.config.monitoring, this.logger);
      this.mcpBridge = new NeuralMCPBridge(this.config.mcp, this.logger);
      this.wsHandler = new NeuralWebSocketHandler(this.wss, this.codexSyntaptic, this.logger);
      this.apiGateway = new NeuralAPIGateway(this.app, this.codexSyntaptic, this.mcpBridge, this.logger);

      await this.initializeRoutes();
      this.logger.info('Neural server components initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize neural server components', { error });
      throw error;
    }
  }

  private async initializeRoutes(): Promise<void> {
    // Health check endpoint
    this.app.get('/health', (req, res) => {
      const health = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        sessionId: this.sessionId,
        components: {
          neural: this.codexSyntaptic.isReady(),
          mcp: this.mcpBridge.isConnected(),
          monitoring: this.monitoring.isActive(),
          resources: this.resourceManager.getStatus()
        }
      };
      res.json(health);
    });

    // Neural processing endpoint
    this.app.post('/neural/process', async (req, res) => {
      try {
        const { input, config } = req.body;
        const result = await this.codexSyntaptic.process(input, config);
        res.json({ success: true, result });
      } catch (error) {
        this.logger.error('Neural processing error', { error });
        res.status(500).json({ success: false, error: error.message });
      }
    });

    // Neural model management
    this.app.get('/neural/models', async (req, res) => {
      try {
        const models = await this.codexSyntaptic.getAvailableModels();
        res.json({ models });
      } catch (error) {
        this.logger.error('Error fetching neural models', { error });
        res.status(500).json({ success: false, error: error.message });
      }
    });

    // Resource monitoring endpoint
    this.app.get('/resources', async (req, res) => {
      try {
        const resources = await this.resourceManager.getCurrentUsage();
        res.json(resources);
      } catch (error) {
        this.logger.error('Error fetching resource data', { error });
        res.status(500).json({ success: false, error: error.message });
      }
    });

    // Metrics endpoint
    this.app.get('/metrics', async (req, res) => {
      try {
        const metrics = await this.monitoring.getMetrics();
        res.json(metrics);
      } catch (error) {
        this.logger.error('Error fetching metrics', { error });
        res.status(500).json({ success: false, error: error.message });
      }
    });

    this.logger.info('Neural server routes initialized');
  }

  public async start(): Promise<void> {
    try {
      // Start monitoring
      if (this.config.monitoring.enabled) {
        await this.monitoring.start();
        this.logger.info('Neural monitoring started');
      }

      // Start MCP bridges
      if (this.config.mcp.enabled) {
        await this.mcpBridge.connect();
        this.logger.info('MCP bridges connected');
      }

      // Start neural processing
      await this.codexSyntaptic.initialize();
      this.logger.info('Codex-syntaptic initialized');

      // Start server
      return new Promise((resolve, reject) => {
        this.server.listen(this.config.port, this.config.host, (error: any) => {
          if (error) {
            this.logger.error('Failed to start neural server', { error });
            reject(error);
            return;
          }

          this.logger.info('Neural-enhanced multi-agent server started', {
            host: this.config.host,
            port: this.config.port,
            sessionId: this.sessionId,
            neural: this.config.neural.enabled,
            mcp: this.config.mcp.enabled,
            monitoring: this.config.monitoring.enabled
          });

          resolve();
        });
      });
    } catch (error) {
      this.logger.error('Failed to start neural server', { error });
      throw error;
    }
  }

  public async stop(): Promise<void> {
    this.logger.info('Stopping neural server...');

    try {
      // Stop monitoring
      if (this.monitoring) {
        await this.monitoring.stop();
      }

      // Disconnect MCP bridges
      if (this.mcpBridge) {
        await this.mcpBridge.disconnect();
      }

      // Close WebSocket connections
      if (this.wss) {
        this.wss.close();
      }

      // Close HTTP server
      if (this.server) {
        await new Promise<void>((resolve) => {
          this.server.close(() => resolve());
        });
      }

      this.logger.info('Neural server stopped successfully');
    } catch (error) {
      this.logger.error('Error stopping neural server', { error });
      throw error;
    }
  }

  public getSessionId(): string {
    return this.sessionId;
  }

  public getLogger(): pino.Logger {
    return this.logger;
  }
}

// Main execution
if (require.main === module) {
  const server = new NeuralServer();

  // Graceful shutdown
  process.on('SIGTERM', async () => {
    console.log('SIGTERM received, shutting down gracefully...');
    await server.stop();
    process.exit(0);
  });

  process.on('SIGINT', async () => {
    console.log('SIGINT received, shutting down gracefully...');
    await server.stop();
    process.exit(0);
  });

  // Start server
  server.start().catch((error) => {
    console.error('Failed to start neural server:', error);
    process.exit(1);
  });
}

export { NeuralServer };