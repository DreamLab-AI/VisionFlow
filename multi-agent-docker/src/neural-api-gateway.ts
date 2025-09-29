/**
 * Neural API Gateway
 * Unified API access point for neural processing and MCP integration
 */

import { Express, Request, Response, NextFunction } from 'express';
import pino from 'pino';
import { v4 as uuidv4 } from 'uuid';
import { CodexSyntaptic } from 'codex-syntaptic';
import { NeuralMCPBridge } from './neural-mcp-bridge';

interface APIRequest extends Request {
  requestId?: string;
  startTime?: number;
  clientInfo?: {
    ip: string;
    userAgent: string;
    apiKey?: string;
  };
}

interface APIResponse {
  success: boolean;
  data?: any;
  error?: {
    message: string;
    code: string;
    details?: any;
  };
  metadata?: {
    requestId: string;
    processingTime: number;
    model?: string;
    tokensUsed?: number;
    timestamp: string;
  };
}

interface NeuralAPIConfig {
  rateLimit: {
    enabled: boolean;
    maxRequests: number;
    windowMs: number;
  };
  authentication: {
    enabled: boolean;
    apiKeyHeader: string;
    validKeys?: string[];
  };
  caching: {
    enabled: boolean;
    ttl: number;
  };
  monitoring: {
    enabled: boolean;
    logRequests: boolean;
    trackMetrics: boolean;
  };
}

class NeuralAPIGateway {
  private app: Express;
  private codexSyntaptic: CodexSyntaptic;
  private mcpBridge: NeuralMCPBridge;
  private logger: pino.Logger;
  private config: NeuralAPIConfig;
  private requestCache: Map<string, any>;
  private rateLimitStore: Map<string, { count: number; resetTime: number }>;

  constructor(
    app: Express,
    codexSyntaptic: CodexSyntaptic,
    mcpBridge: NeuralMCPBridge,
    logger: pino.Logger,
    config?: Partial<NeuralAPIConfig>
  ) {
    this.app = app;
    this.codexSyntaptic = codexSyntaptic;
    this.mcpBridge = mcpBridge;
    this.logger = logger.child({ component: 'neural-api-gateway' });
    this.requestCache = new Map();
    this.rateLimitStore = new Map();

    this.config = {
      rateLimit: {
        enabled: config?.rateLimit?.enabled ?? true,
        maxRequests: config?.rateLimit?.maxRequests ?? 100,
        windowMs: config?.rateLimit?.windowMs ?? 60000
      },
      authentication: {
        enabled: config?.authentication?.enabled ?? false,
        apiKeyHeader: config?.authentication?.apiKeyHeader ?? 'x-api-key',
        validKeys: config?.authentication?.validKeys || []
      },
      caching: {
        enabled: config?.caching?.enabled ?? true,
        ttl: config?.caching?.ttl ?? 300000 // 5 minutes
      },
      monitoring: {
        enabled: config?.monitoring?.enabled ?? true,
        logRequests: config?.monitoring?.logRequests ?? true,
        trackMetrics: config?.monitoring?.trackMetrics ?? true
      }
    };

    this.initializeMiddleware();
    this.initializeRoutes();
  }

  private initializeMiddleware(): void {
    // Request ID middleware
    this.app.use((req: APIRequest, res: Response, next: NextFunction) => {
      req.requestId = uuidv4();
      req.startTime = Date.now();
      req.clientInfo = {
        ip: req.ip || req.connection.remoteAddress || 'unknown',
        userAgent: req.get('User-Agent') || 'unknown',
        apiKey: req.get(this.config.authentication.apiKeyHeader)
      };
      next();
    });

    // Authentication middleware
    if (this.config.authentication.enabled) {
      this.app.use('/api/', (req: APIRequest, res: Response, next: NextFunction) => {
        this.authenticateRequest(req, res, next);
      });
    }

    // Rate limiting middleware
    if (this.config.rateLimit.enabled) {
      this.app.use('/api/', (req: APIRequest, res: Response, next: NextFunction) => {
        this.rateLimitRequest(req, res, next);
      });
    }

    // Request logging middleware
    if (this.config.monitoring.logRequests) {
      this.app.use('/api/', (req: APIRequest, res: Response, next: NextFunction) => {
        this.logRequest(req, res, next);
      });
    }

    this.logger.info('API Gateway middleware initialized');
  }

  private authenticateRequest(req: APIRequest, res: Response, next: NextFunction): void {
    const apiKey = req.clientInfo?.apiKey;

    if (!apiKey) {
      this.sendError(res, 'API key required', 'AUTH_REQUIRED', 401);
      return;
    }

    if (this.config.authentication.validKeys &&
        this.config.authentication.validKeys.length > 0 &&
        !this.config.authentication.validKeys.includes(apiKey)) {
      this.sendError(res, 'Invalid API key', 'AUTH_INVALID', 401);
      return;
    }

    next();
  }

  private rateLimitRequest(req: APIRequest, res: Response, next: NextFunction): void {
    const clientId = req.clientInfo?.ip || 'unknown';
    const now = Date.now();
    const windowStart = now - this.config.rateLimit.windowMs;

    // Clean expired entries
    for (const [key, data] of this.rateLimitStore) {
      if (data.resetTime < now) {
        this.rateLimitStore.delete(key);
      }
    }

    // Check current client
    const clientData = this.rateLimitStore.get(clientId);

    if (!clientData || clientData.resetTime < now) {
      // Reset or create new entry
      this.rateLimitStore.set(clientId, {
        count: 1,
        resetTime: now + this.config.rateLimit.windowMs
      });
    } else {
      // Increment count
      clientData.count++;

      if (clientData.count > this.config.rateLimit.maxRequests) {
        this.sendError(res, 'Rate limit exceeded', 'RATE_LIMIT_EXCEEDED', 429, {
          resetTime: clientData.resetTime,
          maxRequests: this.config.rateLimit.maxRequests
        });
        return;
      }
    }

    // Add rate limit headers
    const remaining = Math.max(0, this.config.rateLimit.maxRequests - (clientData?.count || 1));
    res.set({
      'X-RateLimit-Limit': this.config.rateLimit.maxRequests.toString(),
      'X-RateLimit-Remaining': remaining.toString(),
      'X-RateLimit-Reset': Math.ceil((clientData?.resetTime || now) / 1000).toString()
    });

    next();
  }

  private logRequest(req: APIRequest, res: Response, next: NextFunction): void {
    const originalSend = res.send;

    res.send = function(body: any) {
      const processingTime = Date.now() - (req.startTime || 0);

      logger.info('API Request', {
        requestId: req.requestId,
        method: req.method,
        path: req.path,
        statusCode: res.statusCode,
        processingTime,
        clientIp: req.clientInfo?.ip,
        userAgent: req.clientInfo?.userAgent
      });

      return originalSend.call(this, body);
    };

    const logger = this.logger;
    next();
  }

  private initializeRoutes(): void {
    // Neural processing routes
    this.app.post('/api/neural/process', this.handleNeuralProcess.bind(this));
    this.app.post('/api/neural/batch', this.handleNeuralBatch.bind(this));
    this.app.post('/api/neural/stream', this.handleNeuralStream.bind(this));
    this.app.get('/api/neural/models', this.handleGetModels.bind(this));
    this.app.get('/api/neural/capabilities', this.handleGetCapabilities.bind(this));

    // MCP bridge routes
    this.app.post('/api/mcp/:bridge/send', this.handleMCPSend.bind(this));
    this.app.get('/api/mcp/bridges', this.handleGetBridges.bind(this));
    this.app.get('/api/mcp/:bridge/status', this.handleGetBridgeStatus.bind(this));

    // Context management routes
    this.app.post('/api/context/create', this.handleCreateContext.bind(this));
    this.app.get('/api/context/:contextId', this.handleGetContext.bind(this));
    this.app.put('/api/context/:contextId', this.handleUpdateContext.bind(this));
    this.app.delete('/api/context/:contextId', this.handleDeleteContext.bind(this));

    // Analysis and optimization routes
    this.app.post('/api/analysis/sentiment', this.handleSentimentAnalysis.bind(this));
    this.app.post('/api/analysis/embedding', this.handleGenerateEmbedding.bind(this));
    this.app.post('/api/analysis/similarity', this.handleSimilarityAnalysis.bind(this));
    this.app.post('/api/optimization/recommend', this.handleOptimizationRecommendation.bind(this));

    // Admin routes
    this.app.get('/api/admin/stats', this.handleGetStats.bind(this));
    this.app.post('/api/admin/cache/clear', this.handleClearCache.bind(this));
    this.app.get('/api/admin/health', this.handleHealthCheck.bind(this));

    this.logger.info('API Gateway routes initialized');
  }

  private async handleNeuralProcess(req: APIRequest, res: Response): Promise<void> {
    try {
      const { input, model, config } = req.body;

      if (!input) {
        this.sendError(res, 'Input is required', 'MISSING_INPUT');
        return;
      }

      // Check cache first
      const cacheKey = this.generateCacheKey('neural-process', { input, model, config });
      if (this.config.caching.enabled) {
        const cached = this.getFromCache(cacheKey);
        if (cached) {
          this.sendSuccess(res, cached, req.requestId!, Date.now() - req.startTime!);
          return;
        }
      }

      const result = await this.codexSyntaptic.process(input, {
        model,
        ...config
      });

      // Cache result
      if (this.config.caching.enabled) {
        this.setCache(cacheKey, result);
      }

      this.sendSuccess(res, result, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Neural processing error', { requestId: req.requestId, error });
      this.sendError(res, 'Neural processing failed', 'PROCESSING_ERROR');
    }
  }

  private async handleNeuralBatch(req: APIRequest, res: Response): Promise<void> {
    try {
      const { inputs, model, config } = req.body;

      if (!Array.isArray(inputs) || inputs.length === 0) {
        this.sendError(res, 'Inputs array is required', 'MISSING_INPUTS');
        return;
      }

      const results = await Promise.all(
        inputs.map(async (input, index) => {
          try {
            return await this.codexSyntaptic.process(input, {
              model,
              batchIndex: index,
              ...config
            });
          } catch (error) {
            this.logger.error('Batch item processing error', { requestId: req.requestId, index, error });
            return { error: error.message, index };
          }
        })
      );

      this.sendSuccess(res, { results }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Neural batch processing error', { requestId: req.requestId, error });
      this.sendError(res, 'Neural batch processing failed', 'BATCH_ERROR');
    }
  }

  private async handleNeuralStream(req: APIRequest, res: Response): Promise<void> {
    try {
      const { input, model, config } = req.body;

      if (!input) {
        this.sendError(res, 'Input is required', 'MISSING_INPUT');
        return;
      }

      res.writeHead(200, {
        'Content-Type': 'text/plain',
        'Transfer-Encoding': 'chunked',
        'X-Request-ID': req.requestId!
      });

      const stream = await this.codexSyntaptic.processStream(input, {
        model,
        ...config
      });

      stream.on('data', (chunk) => {
        res.write(chunk);
      });

      stream.on('end', () => {
        res.end();
      });

      stream.on('error', (error) => {
        this.logger.error('Neural streaming error', { requestId: req.requestId, error });
        res.end();
      });
    } catch (error) {
      this.logger.error('Neural streaming setup error', { requestId: req.requestId, error });
      this.sendError(res, 'Neural streaming failed', 'STREAMING_ERROR');
    }
  }

  private async handleGetModels(req: APIRequest, res: Response): Promise<void> {
    try {
      const models = await this.codexSyntaptic.getAvailableModels();
      this.sendSuccess(res, { models }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Get models error', { requestId: req.requestId, error });
      this.sendError(res, 'Failed to get models', 'GET_MODELS_ERROR');
    }
  }

  private async handleGetCapabilities(req: APIRequest, res: Response): Promise<void> {
    try {
      const capabilities = this.codexSyntaptic.getCapabilities();
      this.sendSuccess(res, { capabilities }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Get capabilities error', { requestId: req.requestId, error });
      this.sendError(res, 'Failed to get capabilities', 'GET_CAPABILITIES_ERROR');
    }
  }

  private async handleMCPSend(req: APIRequest, res: Response): Promise<void> {
    try {
      const { bridge } = req.params;
      const message = req.body;

      if (!this.mcpBridge.isConnected()) {
        this.sendError(res, 'MCP bridge not connected', 'BRIDGE_NOT_CONNECTED');
        return;
      }

      await this.mcpBridge.sendMessage(bridge, message);
      this.sendSuccess(res, { sent: true }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('MCP send error', { requestId: req.requestId, error });
      this.sendError(res, 'Failed to send MCP message', 'MCP_SEND_ERROR');
    }
  }

  private async handleGetBridges(req: APIRequest, res: Response): Promise<void> {
    try {
      const bridges = this.mcpBridge.getConnectedBridges();
      this.sendSuccess(res, { bridges }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Get bridges error', { requestId: req.requestId, error });
      this.sendError(res, 'Failed to get bridges', 'GET_BRIDGES_ERROR');
    }
  }

  private async handleGetBridgeStatus(req: APIRequest, res: Response): Promise<void> {
    try {
      const { bridge } = req.params;
      const status = this.mcpBridge.getBridgeStatus(bridge);
      this.sendSuccess(res, { bridge, status }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Get bridge status error', { requestId: req.requestId, error });
      this.sendError(res, 'Failed to get bridge status', 'GET_BRIDGE_STATUS_ERROR');
    }
  }

  private async handleCreateContext(req: APIRequest, res: Response): Promise<void> {
    try {
      const contextId = uuidv4();
      const context = await this.codexSyntaptic.createContext(contextId, req.body);
      this.sendSuccess(res, { contextId, context }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Create context error', { requestId: req.requestId, error });
      this.sendError(res, 'Failed to create context', 'CREATE_CONTEXT_ERROR');
    }
  }

  private async handleGetContext(req: APIRequest, res: Response): Promise<void> {
    try {
      const { contextId } = req.params;
      const context = await this.codexSyntaptic.getContext(contextId);

      if (!context) {
        this.sendError(res, 'Context not found', 'CONTEXT_NOT_FOUND', 404);
        return;
      }

      this.sendSuccess(res, { context }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Get context error', { requestId: req.requestId, error });
      this.sendError(res, 'Failed to get context', 'GET_CONTEXT_ERROR');
    }
  }

  private async handleUpdateContext(req: APIRequest, res: Response): Promise<void> {
    try {
      const { contextId } = req.params;
      const context = await this.codexSyntaptic.updateContext(contextId, req.body);
      this.sendSuccess(res, { context }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Update context error', { requestId: req.requestId, error });
      this.sendError(res, 'Failed to update context', 'UPDATE_CONTEXT_ERROR');
    }
  }

  private async handleDeleteContext(req: APIRequest, res: Response): Promise<void> {
    try {
      const { contextId } = req.params;
      await this.codexSyntaptic.deleteContext(contextId);
      this.sendSuccess(res, { deleted: true }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Delete context error', { requestId: req.requestId, error });
      this.sendError(res, 'Failed to delete context', 'DELETE_CONTEXT_ERROR');
    }
  }

  private async handleSentimentAnalysis(req: APIRequest, res: Response): Promise<void> {
    try {
      const { text } = req.body;

      if (!text) {
        this.sendError(res, 'Text is required', 'MISSING_TEXT');
        return;
      }

      const sentiment = await this.codexSyntaptic.analyzeSentiment(text);
      this.sendSuccess(res, { sentiment }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Sentiment analysis error', { requestId: req.requestId, error });
      this.sendError(res, 'Sentiment analysis failed', 'SENTIMENT_ERROR');
    }
  }

  private async handleGenerateEmbedding(req: APIRequest, res: Response): Promise<void> {
    try {
      const { text, model } = req.body;

      if (!text) {
        this.sendError(res, 'Text is required', 'MISSING_TEXT');
        return;
      }

      const embedding = await this.codexSyntaptic.generateEmbedding(text, model);
      this.sendSuccess(res, { embedding }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Generate embedding error', { requestId: req.requestId, error });
      this.sendError(res, 'Embedding generation failed', 'EMBEDDING_ERROR');
    }
  }

  private async handleSimilarityAnalysis(req: APIRequest, res: Response): Promise<void> {
    try {
      const { texts, model } = req.body;

      if (!Array.isArray(texts) || texts.length < 2) {
        this.sendError(res, 'At least two texts are required', 'MISSING_TEXTS');
        return;
      }

      const similarities = await this.codexSyntaptic.calculateSimilarities(texts, model);
      this.sendSuccess(res, { similarities }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Similarity analysis error', { requestId: req.requestId, error });
      this.sendError(res, 'Similarity analysis failed', 'SIMILARITY_ERROR');
    }
  }

  private async handleOptimizationRecommendation(req: APIRequest, res: Response): Promise<void> {
    try {
      const { data, type } = req.body;

      if (!data) {
        this.sendError(res, 'Data is required', 'MISSING_DATA');
        return;
      }

      const recommendations = await this.codexSyntaptic.generateOptimizationRecommendations(data, type);
      this.sendSuccess(res, { recommendations }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Optimization recommendation error', { requestId: req.requestId, error });
      this.sendError(res, 'Optimization recommendation failed', 'OPTIMIZATION_ERROR');
    }
  }

  private async handleGetStats(req: APIRequest, res: Response): Promise<void> {
    try {
      const stats = {
        cacheSize: this.requestCache.size,
        rateLimitEntries: this.rateLimitStore.size,
        mcpBridgesConnected: this.mcpBridge.getConnectedBridges().length,
        neuralModelsAvailable: (await this.codexSyntaptic.getAvailableModels()).length
      };

      this.sendSuccess(res, { stats }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Get stats error', { requestId: req.requestId, error });
      this.sendError(res, 'Failed to get stats', 'GET_STATS_ERROR');
    }
  }

  private async handleClearCache(req: APIRequest, res: Response): Promise<void> {
    try {
      const clearedEntries = this.requestCache.size;
      this.requestCache.clear();

      this.sendSuccess(res, {
        cleared: true,
        clearedEntries
      }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Clear cache error', { requestId: req.requestId, error });
      this.sendError(res, 'Failed to clear cache', 'CLEAR_CACHE_ERROR');
    }
  }

  private async handleHealthCheck(req: APIRequest, res: Response): Promise<void> {
    try {
      const health = {
        status: 'healthy',
        neural: this.codexSyntaptic.isReady(),
        mcp: this.mcpBridge.isConnected(),
        cache: this.config.caching.enabled,
        rateLimit: this.config.rateLimit.enabled,
        authentication: this.config.authentication.enabled
      };

      this.sendSuccess(res, { health }, req.requestId!, Date.now() - req.startTime!);
    } catch (error) {
      this.logger.error('Health check error', { requestId: req.requestId, error });
      this.sendError(res, 'Health check failed', 'HEALTH_CHECK_ERROR');
    }
  }

  private sendSuccess(res: Response, data: any, requestId: string, processingTime: number): void {
    const response: APIResponse = {
      success: true,
      data,
      metadata: {
        requestId,
        processingTime,
        timestamp: new Date().toISOString()
      }
    };
    res.json(response);
  }

  private sendError(res: Response, message: string, code: string, statusCode = 400, details?: any): void {
    const response: APIResponse = {
      success: false,
      error: {
        message,
        code,
        details
      }
    };
    res.status(statusCode).json(response);
  }

  private generateCacheKey(prefix: string, data: any): string {
    const hash = require('crypto').createHash('md5').update(JSON.stringify(data)).digest('hex');
    return `${prefix}:${hash}`;
  }

  private getFromCache(key: string): any {
    const cached = this.requestCache.get(key);
    if (cached && cached.expires > Date.now()) {
      return cached.data;
    }
    this.requestCache.delete(key);
    return null;
  }

  private setCache(key: string, data: any): void {
    this.requestCache.set(key, {
      data,
      expires: Date.now() + this.config.caching.ttl
    });
  }
}

export { NeuralAPIGateway, APIRequest, APIResponse, NeuralAPIConfig };