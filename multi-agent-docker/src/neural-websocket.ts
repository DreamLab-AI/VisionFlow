/**
 * Neural WebSocket Handler
 * Real-time neural processing communication via WebSockets
 */

import { WebSocketServer, WebSocket } from 'ws';
import { EventEmitter } from 'events';
import pino from 'pino';
import { v4 as uuidv4 } from 'uuid';
import { CodexSyntaptic } from 'codex-syntaptic';

interface NeuralWebSocketClient {
  id: string;
  ws: WebSocket;
  sessionId: string;
  authenticated: boolean;
  subscriptions: Set<string>;
  neuralContext: any;
  lastActivity: Date;
  metadata: {
    userAgent?: string;
    ip?: string;
    capabilities?: string[];
  };
}

interface NeuralMessage {
  id: string;
  type: 'request' | 'response' | 'notification' | 'subscription' | 'neural-result';
  action?: string;
  data?: any;
  error?: any;
  clientId?: string;
  sessionId?: string;
  timestamp: string;
  neuralProcessed?: boolean;
  priority?: 'low' | 'medium' | 'high' | 'critical';
}

interface NeuralProcessingRequest {
  input: any;
  model?: string;
  config?: any;
  streaming?: boolean;
  contextId?: string;
}

interface NeuralProcessingResult {
  output: any;
  confidence: number;
  processingTime: number;
  model: string;
  contextVector?: number[];
  metadata?: any;
}

class NeuralWebSocketHandler extends EventEmitter {
  private wss: WebSocketServer;
  private codexSyntaptic: CodexSyntaptic;
  private logger: pino.Logger;
  private clients: Map<string, NeuralWebSocketClient>;
  private subscriptions: Map<string, Set<string>>; // topic -> clientIds
  private neuralSessions: Map<string, any>;
  private heartbeatInterval: NodeJS.Timeout;

  constructor(wss: WebSocketServer, codexSyntaptic: CodexSyntaptic, logger: pino.Logger) {
    super();
    this.wss = wss;
    this.codexSyntaptic = codexSyntaptic;
    this.logger = logger.child({ component: 'neural-websocket' });
    this.clients = new Map();
    this.subscriptions = new Map();
    this.neuralSessions = new Map();

    this.initializeWebSocketServer();
    this.startHeartbeat();
  }

  private initializeWebSocketServer(): void {
    this.wss.on('connection', (ws: WebSocket, request) => {
      this.handleNewConnection(ws, request);
    });

    this.wss.on('error', (error) => {
      this.logger.error('WebSocket server error', { error });
    });

    this.logger.info('Neural WebSocket handler initialized');
  }

  private handleNewConnection(ws: WebSocket, request: any): void {
    const clientId = uuidv4();
    const sessionId = uuidv4();

    const client: NeuralWebSocketClient = {
      id: clientId,
      ws,
      sessionId,
      authenticated: false,
      subscriptions: new Set(),
      neuralContext: {},
      lastActivity: new Date(),
      metadata: {
        userAgent: request.headers['user-agent'],
        ip: request.socket.remoteAddress,
        capabilities: []
      }
    };

    this.clients.set(clientId, client);

    this.logger.info('New WebSocket connection', {
      clientId,
      sessionId,
      ip: client.metadata.ip,
      userAgent: client.metadata.userAgent
    });

    ws.on('message', async (data) => {
      await this.handleMessage(clientId, data);
    });

    ws.on('close', (code, reason) => {
      this.handleDisconnection(clientId, code, reason);
    });

    ws.on('error', (error) => {
      this.handleClientError(clientId, error);
    });

    ws.on('pong', () => {
      this.handlePong(clientId);
    });

    // Send welcome message
    this.sendMessage(clientId, {
      id: uuidv4(),
      type: 'notification',
      action: 'welcome',
      data: {
        clientId,
        sessionId,
        neuralCapabilities: this.codexSyntaptic.getCapabilities(),
        serverTime: new Date().toISOString()
      },
      timestamp: new Date().toISOString()
    });
  }

  private async handleMessage(clientId: string, data: any): Promise<void> {
    const client = this.clients.get(clientId);
    if (!client) return;

    try {
      client.lastActivity = new Date();

      let message: NeuralMessage;
      try {
        message = JSON.parse(data.toString());
      } catch (error) {
        this.sendError(clientId, 'Invalid JSON message', 'PARSE_ERROR');
        return;
      }

      this.logger.debug('Received message', { clientId, messageId: message.id, action: message.action });

      // Route message based on action
      switch (message.action) {
        case 'authenticate':
          await this.handleAuthentication(clientId, message);
          break;
        case 'neural-process':
          await this.handleNeuralProcessing(clientId, message);
          break;
        case 'neural-stream':
          await this.handleNeuralStreaming(clientId, message);
          break;
        case 'subscribe':
          await this.handleSubscription(clientId, message);
          break;
        case 'unsubscribe':
          await this.handleUnsubscription(clientId, message);
          break;
        case 'get-models':
          await this.handleGetModels(clientId, message);
          break;
        case 'get-context':
          await this.handleGetContext(clientId, message);
          break;
        case 'clear-context':
          await this.handleClearContext(clientId, message);
          break;
        case 'ping':
          await this.handlePing(clientId, message);
          break;
        default:
          this.sendError(clientId, `Unknown action: ${message.action}`, 'UNKNOWN_ACTION');
      }
    } catch (error) {
      this.logger.error('Error handling message', { clientId, error });
      this.sendError(clientId, 'Internal server error', 'INTERNAL_ERROR');
    }
  }

  private async handleAuthentication(clientId: string, message: NeuralMessage): Promise<void> {
    const client = this.clients.get(clientId);
    if (!client) return;

    try {
      const { token, capabilities } = message.data || {};

      // Basic authentication (in production, validate token properly)
      if (token) {
        client.authenticated = true;
        client.metadata.capabilities = capabilities || [];

        this.sendMessage(clientId, {
          id: message.id,
          type: 'response',
          data: {
            authenticated: true,
            capabilities: client.metadata.capabilities
          },
          timestamp: new Date().toISOString()
        });

        this.logger.info('Client authenticated', { clientId });
      } else {
        this.sendError(clientId, 'Authentication failed', 'AUTH_FAILED');
      }
    } catch (error) {
      this.logger.error('Authentication error', { clientId, error });
      this.sendError(clientId, 'Authentication error', 'AUTH_ERROR');
    }
  }

  private async handleNeuralProcessing(clientId: string, message: NeuralMessage): Promise<void> {
    const client = this.clients.get(clientId);
    if (!client || !client.authenticated) {
      this.sendError(clientId, 'Authentication required', 'AUTH_REQUIRED');
      return;
    }

    try {
      const request: NeuralProcessingRequest = message.data;
      if (!request.input) {
        this.sendError(clientId, 'Input data required', 'MISSING_INPUT');
        return;
      }

      const startTime = Date.now();

      // Create neural session context
      const contextId = request.contextId || uuidv4();
      if (!this.neuralSessions.has(contextId)) {
        this.neuralSessions.set(contextId, {
          clientId,
          created: new Date(),
          history: []
        });
      }

      // Process with codex-syntaptic
      const result = await this.codexSyntaptic.process(request.input, {
        model: request.model,
        contextId,
        streaming: request.streaming || false,
        ...request.config
      });

      const processingTime = Date.now() - startTime;

      // Create neural result
      const neuralResult: NeuralProcessingResult = {
        output: result.output,
        confidence: result.confidence || 0.95,
        processingTime,
        model: result.model || request.model || 'default',
        contextVector: result.contextVector,
        metadata: {
          tokensUsed: result.tokensUsed,
          computeUnits: result.computeUnits,
          sessionId: client.sessionId
        }
      };

      // Update neural session
      const session = this.neuralSessions.get(contextId);
      if (session) {
        session.history.push({
          input: request.input,
          output: neuralResult.output,
          timestamp: new Date(),
          processingTime
        });
      }

      // Send result
      this.sendMessage(clientId, {
        id: message.id,
        type: 'response',
        action: 'neural-result',
        data: neuralResult,
        timestamp: new Date().toISOString(),
        neuralProcessed: true
      });

      this.logger.info('Neural processing completed', {
        clientId,
        contextId,
        processingTime,
        confidence: neuralResult.confidence
      });

    } catch (error) {
      this.logger.error('Neural processing error', { clientId, error });
      this.sendError(clientId, 'Neural processing failed', 'PROCESSING_ERROR');
    }
  }

  private async handleNeuralStreaming(clientId: string, message: NeuralMessage): Promise<void> {
    const client = this.clients.get(clientId);
    if (!client || !client.authenticated) {
      this.sendError(clientId, 'Authentication required', 'AUTH_REQUIRED');
      return;
    }

    try {
      const request: NeuralProcessingRequest = message.data;
      if (!request.input) {
        this.sendError(clientId, 'Input data required', 'MISSING_INPUT');
        return;
      }

      const contextId = request.contextId || uuidv4();

      // Start streaming processing
      const stream = await this.codexSyntaptic.processStream(request.input, {
        model: request.model,
        contextId,
        ...request.config
      });

      // Send streaming start notification
      this.sendMessage(clientId, {
        id: uuidv4(),
        type: 'notification',
        action: 'stream-start',
        data: { contextId, messageId: message.id },
        timestamp: new Date().toISOString()
      });

      // Handle stream data
      stream.on('data', (chunk) => {
        this.sendMessage(clientId, {
          id: uuidv4(),
          type: 'notification',
          action: 'stream-data',
          data: {
            contextId,
            messageId: message.id,
            chunk: chunk.toString(),
            partial: true
          },
          timestamp: new Date().toISOString()
        });
      });

      stream.on('end', () => {
        this.sendMessage(clientId, {
          id: message.id,
          type: 'response',
          action: 'stream-complete',
          data: { contextId },
          timestamp: new Date().toISOString()
        });
      });

      stream.on('error', (error) => {
        this.logger.error('Streaming error', { clientId, contextId, error });
        this.sendError(clientId, 'Streaming failed', 'STREAM_ERROR');
      });

    } catch (error) {
      this.logger.error('Neural streaming error', { clientId, error });
      this.sendError(clientId, 'Neural streaming failed', 'STREAMING_ERROR');
    }
  }

  private async handleSubscription(clientId: string, message: NeuralMessage): Promise<void> {
    const client = this.clients.get(clientId);
    if (!client) return;

    try {
      const { topic } = message.data || {};
      if (!topic) {
        this.sendError(clientId, 'Topic required', 'MISSING_TOPIC');
        return;
      }

      client.subscriptions.add(topic);

      if (!this.subscriptions.has(topic)) {
        this.subscriptions.set(topic, new Set());
      }
      this.subscriptions.get(topic)!.add(clientId);

      this.sendMessage(clientId, {
        id: message.id,
        type: 'response',
        data: { subscribed: true, topic },
        timestamp: new Date().toISOString()
      });

      this.logger.debug('Client subscribed to topic', { clientId, topic });
    } catch (error) {
      this.logger.error('Subscription error', { clientId, error });
      this.sendError(clientId, 'Subscription failed', 'SUBSCRIPTION_ERROR');
    }
  }

  private async handleUnsubscription(clientId: string, message: NeuralMessage): Promise<void> {
    const client = this.clients.get(clientId);
    if (!client) return;

    try {
      const { topic } = message.data || {};
      if (!topic) {
        this.sendError(clientId, 'Topic required', 'MISSING_TOPIC');
        return;
      }

      client.subscriptions.delete(topic);

      const topicSubscribers = this.subscriptions.get(topic);
      if (topicSubscribers) {
        topicSubscribers.delete(clientId);
        if (topicSubscribers.size === 0) {
          this.subscriptions.delete(topic);
        }
      }

      this.sendMessage(clientId, {
        id: message.id,
        type: 'response',
        data: { unsubscribed: true, topic },
        timestamp: new Date().toISOString()
      });

      this.logger.debug('Client unsubscribed from topic', { clientId, topic });
    } catch (error) {
      this.logger.error('Unsubscription error', { clientId, error });
      this.sendError(clientId, 'Unsubscription failed', 'UNSUBSCRIPTION_ERROR');
    }
  }

  private async handleGetModels(clientId: string, message: NeuralMessage): Promise<void> {
    try {
      const models = await this.codexSyntaptic.getAvailableModels();

      this.sendMessage(clientId, {
        id: message.id,
        type: 'response',
        data: { models },
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      this.logger.error('Get models error', { clientId, error });
      this.sendError(clientId, 'Failed to get models', 'GET_MODELS_ERROR');
    }
  }

  private async handleGetContext(clientId: string, message: NeuralMessage): Promise<void> {
    try {
      const { contextId } = message.data || {};
      const session = this.neuralSessions.get(contextId);

      this.sendMessage(clientId, {
        id: message.id,
        type: 'response',
        data: { context: session || null },
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      this.logger.error('Get context error', { clientId, error });
      this.sendError(clientId, 'Failed to get context', 'GET_CONTEXT_ERROR');
    }
  }

  private async handleClearContext(clientId: string, message: NeuralMessage): Promise<void> {
    try {
      const { contextId } = message.data || {};
      if (contextId) {
        this.neuralSessions.delete(contextId);
      }

      this.sendMessage(clientId, {
        id: message.id,
        type: 'response',
        data: { cleared: true, contextId },
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      this.logger.error('Clear context error', { clientId, error });
      this.sendError(clientId, 'Failed to clear context', 'CLEAR_CONTEXT_ERROR');
    }
  }

  private async handlePing(clientId: string, message: NeuralMessage): Promise<void> {
    this.sendMessage(clientId, {
      id: message.id,
      type: 'response',
      action: 'pong',
      data: { timestamp: new Date().toISOString() },
      timestamp: new Date().toISOString()
    });
  }

  private handleDisconnection(clientId: string, code: number, reason: Buffer): void {
    const client = this.clients.get(clientId);
    if (!client) return;

    this.logger.info('WebSocket disconnection', {
      clientId,
      sessionId: client.sessionId,
      code,
      reason: reason.toString()
    });

    // Clean up subscriptions
    for (const topic of client.subscriptions) {
      const topicSubscribers = this.subscriptions.get(topic);
      if (topicSubscribers) {
        topicSubscribers.delete(clientId);
        if (topicSubscribers.size === 0) {
          this.subscriptions.delete(topic);
        }
      }
    }

    // Clean up neural sessions
    for (const [contextId, session] of this.neuralSessions) {
      if (session.clientId === clientId) {
        this.neuralSessions.delete(contextId);
      }
    }

    this.clients.delete(clientId);
    this.emit('client-disconnected', clientId);
  }

  private handleClientError(clientId: string, error: Error): void {
    this.logger.error('WebSocket client error', { clientId, error });
    this.emit('client-error', { clientId, error });
  }

  private handlePong(clientId: string): void {
    const client = this.clients.get(clientId);
    if (client) {
      client.lastActivity = new Date();
    }
  }

  private sendMessage(clientId: string, message: NeuralMessage): void {
    const client = this.clients.get(clientId);
    if (!client || client.ws.readyState !== WebSocket.OPEN) return;

    try {
      client.ws.send(JSON.stringify(message));
    } catch (error) {
      this.logger.error('Error sending message to client', { clientId, error });
    }
  }

  private sendError(clientId: string, errorMessage: string, errorCode: string): void {
    this.sendMessage(clientId, {
      id: uuidv4(),
      type: 'response',
      error: {
        message: errorMessage,
        code: errorCode
      },
      timestamp: new Date().toISOString()
    });
  }

  public broadcast(topic: string, data: any): void {
    const subscribers = this.subscriptions.get(topic);
    if (!subscribers) return;

    const message: NeuralMessage = {
      id: uuidv4(),
      type: 'notification',
      action: 'broadcast',
      data: { topic, ...data },
      timestamp: new Date().toISOString()
    };

    for (const clientId of subscribers) {
      this.sendMessage(clientId, message);
    }

    this.logger.debug('Broadcast sent', { topic, subscriberCount: subscribers.size });
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      this.performHeartbeat();
    }, 30000); // 30 seconds
  }

  private performHeartbeat(): void {
    const now = new Date();
    const timeout = 60000; // 1 minute timeout

    for (const [clientId, client] of this.clients) {
      if (now.getTime() - client.lastActivity.getTime() > timeout) {
        this.logger.warn('Client timeout, closing connection', { clientId });
        client.ws.close(1000, 'Timeout');
      } else if (client.ws.readyState === WebSocket.OPEN) {
        client.ws.ping();
      }
    }
  }

  public getConnectedClients(): number {
    return this.clients.size;
  }

  public getSubscriptionCount(topic: string): number {
    const subscribers = this.subscriptions.get(topic);
    return subscribers ? subscribers.size : 0;
  }

  public stop(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }

    for (const [clientId, client] of this.clients) {
      client.ws.close(1000, 'Server shutdown');
    }

    this.clients.clear();
    this.subscriptions.clear();
    this.neuralSessions.clear();

    this.logger.info('Neural WebSocket handler stopped');
  }
}

export { NeuralWebSocketHandler, NeuralMessage, NeuralProcessingRequest, NeuralProcessingResult };