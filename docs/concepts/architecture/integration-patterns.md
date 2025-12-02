# Integration Patterns in VisionFlow

## Table of Contents

1. [Overview](#overview)
2. [Multi-Agent Integration](#multi-agent-integration)
3. [Service Integration](#service-integration)
4. 
5. 
6. 
7. 
8. 
9. 
10. [Integration Testing](#integration-testing)

---

## Overview

### Integration Architecture

VisionFlow uses a **hybrid integration architecture** that combines:

- **Orchestration**: Centralized workflow coordination
- **Choreography**: Decentralized event-driven communication
- **Message Queuing**: Asynchronous task processing
- **Direct API Calls**: Synchronous request-response patterns

```
┌────────────────────────────────────────────────────────────────┐
│                     Integration Layer                          │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │  Event   │   │ Message  │   │   API    │   │  Stream  │  │
│  │   Bus    │   │  Queue   │   │ Gateway  │   │   Hub    │  │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘  │
│       │              │              │              │         │
└───────┼──────────────┼──────────────┼──────────────┼─────────┘
        │              │              │              │
        └──────────────┴──────────────┴──────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
┌───────▼────────┐                   ┌───────▼────────┐
│  Agent Swarm   │                   │    Services    │
│   (54+ types)  │                   │  (API, XR, AI) │
└────────────────┘                   └────────────────┘
```

### Key Integration Principles

1. **Loose Coupling**: Services communicate through well-defined interfaces
2. **High Cohesion**: Related functionality grouped together
3. **Fault Isolation**: Failures don't cascade across system boundaries
4. **Observability**: All integrations are traceable and monitorable
5. **Idempotency**: Operations can be safely retried
6. **Eventual Consistency**: Accept temporary inconsistencies for availability

---

## Multi-Agent Integration

### Agent Communication Protocol

VisionFlow supports 54+ specialized agents that communicate through a standardized protocol:

```typescript
/**
 * Agent message envelope
 */
interface AgentMessage {
  id: string;
  type: AgentMessageType;
  from: AgentIdentity;
  to: AgentIdentity | 'broadcast';
  timestamp: Date;
  payload: any;
  correlationId?: string;
  replyTo?: string;
  headers?: Record<string, string>;
}

/**
 * Message types
 */
enum AgentMessageType {
  COMMAND = 'command',
  QUERY = 'query',
  EVENT = 'event',
  RESPONSE = 'response',
  ERROR = 'error'
}

/**
 * Agent identity
 */
interface AgentIdentity {
  id: string;
  type: string; // 'researcher', 'coder', 'tester', etc.
  capabilities: string[];
  status: AgentStatus;
}

/**
 * Agent status
 */
enum AgentStatus {
  IDLE = 'idle',
  BUSY = 'busy',
  ERROR = 'error',
  OFFLINE = 'offline'
}
```

### Message Bus Implementation

```typescript
/**
 * Message bus for agent communication
 */
class AgentMessageBus {
  private subscribers = new Map<string, Set<MessageHandler>>();
  private agents = new Map<string, AgentIdentity>();
  private messageHistory: AgentMessage[] = [];
  private logger: Logger;

  constructor() {
    this.logger = createLogger('agent-message-bus');
  }

  /**
   * Register an agent
   */
  registerAgent(agent: AgentIdentity): void {
    this.agents.set(agent.id, agent);
    this.logger.info('Agent registered', { agentId: agent.id, type: agent.type });
  }

  /**
   * Unregister an agent
   */
  unregisterAgent(agentId: string): void {
    this.agents.delete(agentId);
    this.subscribers.delete(agentId);
    this.logger.info('Agent unregistered', { agentId });
  }

  /**
   * Subscribe to messages
   */
  subscribe(
    agentId: string,
    handler: MessageHandler,
    filter?: MessageFilter
  ): () => void {
    if (!this.subscribers.has(agentId)) {
      this.subscribers.set(agentId, new Set());
    }

    const wrappedHandler: MessageHandler = (msg) => {
      if (!filter || this.matchesFilter(msg, filter)) {
        handler(msg);
      }
    };

    this.subscribers.get(agentId)!.add(wrappedHandler);

    // Return unsubscribe function
    return () => {
      this.subscribers.get(agentId)?.delete(wrappedHandler);
    };
  }

  /**
   * Publish a message
   */
  async publish(message: AgentMessage): Promise<void> {
    // Validate message
    if (!this.agents.has(message.from.id)) {
      throw new Error(`Unknown sender: ${message.from.id}`);
    }

    // Add to history
    this.messageHistory.push(message);
    if (this.messageHistory.length > 1000) {
      this.messageHistory.shift();
    }

    this.logger.debug('Publishing message', {
      messageId: message.id,
      type: message.type,
      from: message.from.id,
      to: message.to
    });

    // Route message
    if (message.to === 'broadcast') {
      await this.broadcast(message);
    } else {
      await this.sendDirect(message);
    }
  }

  /**
   * Request-response pattern
   */
  async request(
    from: AgentIdentity,
    to: AgentIdentity,
    payload: any,
    timeout: number = 30000
  ): Promise<any> {
    const requestId = generateId();

    const request: AgentMessage = {
      id: requestId,
      type: AgentMessageType.QUERY,
      from,
      to,
      timestamp: new Date(),
      payload,
      replyTo: from.id
    };

    // Create promise that resolves when response arrives
    const responsePromise = new Promise((resolve, reject) => {
      const timeoutHandle = setTimeout(() => {
        unsubscribe();
        reject(new Error('Request timeout'));
      }, timeout);

      const unsubscribe = this.subscribe(from.id, (msg) => {
        if (msg.correlationId === requestId && msg.type === AgentMessageType.RESPONSE) {
          clearTimeout(timeoutHandle);
          unsubscribe();
          resolve(msg.payload);
        } else if (msg.correlationId === requestId && msg.type === AgentMessageType.ERROR) {
          clearTimeout(timeoutHandle);
          unsubscribe();
          reject(new Error(msg.payload.message));
        }
      });
    });

    await this.publish(request);
    return responsePromise;
  }

  /**
   * Get message history
   */
  getHistory(filter?: MessageFilter): AgentMessage[] {
    if (!filter) {
      return [...this.messageHistory];
    }

    return this.messageHistory.filter(msg => this.matchesFilter(msg, filter));
  }

  /**
   * Get registered agents
   */
  getAgents(type?: string): AgentIdentity[] {
    const agents = Array.from(this.agents.values());
    return type ? agents.filter(a => a.type === type) : agents;
  }

  private async broadcast(message: AgentMessage): Promise<void> {
    const deliveries: Promise<void>[] = [];

    for (const [agentId, handlers] of this.subscribers) {
      if (agentId !== message.from.id) {
        for (const handler of handlers) {
          deliveries.push(
            Promise.resolve().then(() => handler(message))
          );
        }
      }
    }

    await Promise.all(deliveries);
  }

  private async sendDirect(message: AgentMessage): Promise<void> {
    const recipientId = typeof message.to === 'string' ? message.to : message.to.id;
    const handlers = this.subscribers.get(recipientId);

    if (!handlers || handlers.size === 0) {
      this.logger.warn('No handlers for recipient', { recipientId });
      return;
    }

    const deliveries = Array.from(handlers).map(handler =>
      Promise.resolve().then(() => handler(message))
    );

    await Promise.all(deliveries);
  }

  private matchesFilter(message: AgentMessage, filter: MessageFilter): boolean {
    if (filter.type && message.type !== filter.type) {
      return false;
    }

    if (filter.fromType && message.from.type !== filter.fromType) {
      return false;
    }

    if (filter.correlationId && message.correlationId !== filter.correlationId) {
      return false;
    }

    return true;
  }
}

/**
 * Message handler function
 */
type MessageHandler = (message: AgentMessage) => void | Promise<void>;

/**
 * Message filter
 */
interface MessageFilter {
  type?: AgentMessageType;
  fromType?: string;
  correlationId?: string;
}
```

### Agent Coordination Patterns

#### 1. Request-Response Pattern

```typescript
/**
 * Request-response coordination
 */
class RequestResponseCoordinator {
  constructor(private messageBus: AgentMessageBus) {}

  /**
   * Coordinate a task between agents
   */
  async coordinateTask(
    requester: AgentIdentity,
    provider: AgentIdentity,
    task: any
  ): Promise<any> {
    this.logger.info('Coordinating task', {
      requester: requester.id,
      provider: provider.id,
      task
    });

    try {
      const result = await this.messageBus.request(
        requester,
        provider,
        { task, timestamp: Date.now() },
        60000 // 60 second timeout
      );

      this.logger.info('Task completed', { result });
      return result;
    } catch (error) {
      this.logger.error('Task failed', { error });
      throw error;
    }
  }
}
```

#### 2. Pub-Sub Pattern

```typescript
/**
 * Publish-subscribe coordination
 */
class PubSubCoordinator {
  constructor(private messageBus: AgentMessageBus) {}

  /**
   * Publish event to interested agents
   */
  async publishEvent(
    publisher: AgentIdentity,
    eventType: string,
    data: any
  ): Promise<void> {
    await this.messageBus.publish({
      id: generateId(),
      type: AgentMessageType.EVENT,
      from: publisher,
      to: 'broadcast',
      timestamp: new Date(),
      payload: { eventType, data },
      headers: { 'event-type': eventType }
    });
  }

  /**
   * Subscribe to events
   */
  subscribeToEvents(
    subscriber: AgentIdentity,
    eventTypes: string[],
    handler: (event: any) => void
  ): () => void {
    return this.messageBus.subscribe(
      subscriber.id,
      (msg) => {
        if (msg.type === AgentMessageType.EVENT) {
          const eventType = msg.payload.eventType;
          if (eventTypes.includes(eventType)) {
            handler(msg.payload.data);
          }
        }
      }
    );
  }
}
```

#### 3. Pipeline Pattern

```typescript
/**
 * Pipeline coordination for sequential processing
 */
class PipelineCoordinator {
  constructor(private messageBus: AgentMessageBus) {}

  /**
   * Execute a pipeline of agents
   */
  async executePipeline(
    initiator: AgentIdentity,
    pipeline: AgentIdentity[],
    input: any
  ): Promise<any> {
    let currentData = input;

    for (let i = 0; i < pipeline.length; i++) {
      const agent = pipeline[i];
      const stepName = `Step ${i + 1}: ${agent.type}`;

      this.logger.info(`Executing ${stepName}`, { agentId: agent.id });

      try {
        currentData = await this.messageBus.request(
          initiator,
          agent,
          { step: i, data: currentData },
          120000 // 2 minute timeout per step
        );

        this.logger.info(`${stepName} completed`, { result: currentData });
      } catch (error) {
        this.logger.error(`${stepName} failed`, { error });
        throw new Error(`Pipeline failed at ${stepName}: ${error.message}`);
      }
    }

    return currentData;
  }
}
```

#### 4. Scatter-Gather Pattern

```typescript
/**
 * Scatter-gather coordination for parallel processing
 */
class ScatterGatherCoordinator {
  constructor(private messageBus: AgentMessageBus) {}

  /**
   * Scatter work to multiple agents and gather results
   */
  async scatterGather<T>(
    coordinator: AgentIdentity,
    workers: AgentIdentity[],
    task: any,
    aggregator?: (results: T[]) => T
  ): Promise<T | T[]> {
    this.logger.info('Scattering task to workers', {
      coordinator: coordinator.id,
      workerCount: workers.length
    });

    // Scatter: Send task to all workers
    const requests = workers.map(worker =>
      this.messageBus.request(coordinator, worker, task, 60000)
        .catch(error => {
          this.logger.error('Worker failed', { workerId: worker.id, error });
          return null; // Continue with partial results
        })
    );

    // Gather: Wait for all results
    const results = await Promise.all(requests);
    const successfulResults = results.filter(r => r !== null) as T[];

    this.logger.info('Gathered results', {
      total: workers.length,
      successful: successfulResults.length
    });

    // Aggregate if aggregator provided
    if (aggregator) {
      return aggregator(successfulResults);
    }

    return successfulResults;
  }
}
```

### Multi-Agent Workflow Example

```typescript
/**
 * Complex multi-agent workflow
 */
class CodeReviewWorkflow {
  constructor(
    private messageBus: AgentMessageBus,
    private agents: {
      analyzer: AgentIdentity;
      securityScanner: AgentIdentity;
      performanceAnalyzer: AgentIdentity;
      reviewer: AgentIdentity;
    }
  ) {}

  /**
   * Execute complete code review workflow
   */
  async executeReview(
    coordinator: AgentIdentity,
    codeFiles: string[]
  ): Promise<ReviewReport> {
    // Step 1: Parallel analysis
    const scatterGather = new ScatterGatherCoordinator(this.messageBus);

    const analysisResults = await scatterGather.scatterGather<AnalysisResult>(
      coordinator,
      [
        this.agents.analyzer,
        this.agents.securityScanner,
        this.agents.performanceAnalyzer
      ],
      { files: codeFiles }
    ) as AnalysisResult[];

    // Step 2: Aggregate and review
    const pipeline = new PipelineCoordinator(this.messageBus);

    const finalReport = await pipeline.executePipeline(
      coordinator,
      [this.agents.reviewer],
      {
        codeFiles,
        analyses: analysisResults
      }
    );

    return finalReport;
  }
}

/**
 * Analysis result structure
 */
interface AnalysisResult {
  agentType: string;
  findings: Finding[];
  metrics: Record<string, number>;
}

/**
 * Review report structure
 */
interface ReviewReport {
  overallScore: number;
  analyses: AnalysisResult[];
  recommendations: string[];
  issues: Issue[];
}
```

---

## Service Integration

### Service Registry Pattern

```typescript
/**
 * Service registration info
 */
interface ServiceInfo {
  id: string;
  name: string;
  version: string;
  endpoint: string;
  healthCheckUrl: string;
  capabilities: string[];
  metadata: Record<string, any>;
}

/**
 * Service registry for service discovery
 */
class ServiceRegistry {
  private services = new Map<string, ServiceInfo>();
  private healthChecks = new Map<string, NodeJS.Timeout>();
  private logger: Logger;

  constructor() {
    this.logger = createLogger('service-registry');
  }

  /**
   * Register a service
   */
  async register(service: ServiceInfo): Promise<void> {
    this.services.set(service.id, service);

    // Start health checking
    this.startHealthCheck(service);

    this.logger.info('Service registered', {
      id: service.id,
      name: service.name,
      endpoint: service.endpoint
    });
  }

  /**
   * Unregister a service
   */
  async unregister(serviceId: string): Promise<void> {
    this.services.delete(serviceId);

    // Stop health checking
    const healthCheck = this.healthChecks.get(serviceId);
    if (healthCheck) {
      clearInterval(healthCheck);
      this.healthChecks.delete(serviceId);
    }

    this.logger.info('Service unregistered', { serviceId });
  }

  /**
   * Discover services by capability
   */
  discover(capability: string): ServiceInfo[] {
    return Array.from(this.services.values()).filter(
      service => service.capabilities.includes(capability)
    );
  }

  /**
   * Get service by ID
   */
  getService(serviceId: string): ServiceInfo | undefined {
    return this.services.get(serviceId);
  }

  /**
   * Get all services
   */
  getAllServices(): ServiceInfo[] {
    return Array.from(this.services.values());
  }

  private startHealthCheck(service: ServiceInfo): void {
    const checkInterval = setInterval(async () => {
      try {
        const response = await fetch(service.healthCheckUrl, {
          method: 'GET',
          timeout: 5000
        });

        if (!response.ok) {
          this.logger.warn('Service health check failed', {
            serviceId: service.id,
            status: response.status
          });
        }
      } catch (error) {
        this.logger.error('Service health check error', {
          serviceId: service.id,
          error
        });
        // Consider unregistering service after multiple failures
      }
    }, 30000); // Check every 30 seconds

    this.healthChecks.set(service.id, checkInterval);
  }
}
```

### API Gateway Pattern

```typescript
/**
 * API Gateway for unified service access
 */
class APIGateway {
  private registry: ServiceRegistry;
  private router: Router;
  private rateLimiter: RateLimiter;
  private logger: Logger;

  constructor(registry: ServiceRegistry) {
    this.registry = registry;
    this.router = new Router();
    this.rateLimiter = new RateLimiter();
    this.logger = createLogger('api-gateway');
    this.setupRoutes();
  }

  /**
   * Setup gateway routes
   */
  private setupRoutes(): void {
    // Health check
    this.router.get('/health', async (req, res) => {
      res.json({ status: 'healthy', timestamp: new Date() });
    });

    // Service discovery
    this.router.get('/services', async (req, res) => {
      const capability = req.query.capability as string;
      const services = capability
        ? this.registry.discover(capability)
        : this.registry.getAllServices();

      res.json({ services });
    });

    // Proxy to services
    this.router.all('/api/:serviceId/*', async (req, res) => {
      const serviceId = req.params.serviceId;
      const path = req.params[0];

      await this.proxyRequest(serviceId, path, req, res);
    });
  }

  /**
   * Proxy request to service
   */
  private async proxyRequest(
    serviceId: string,
    path: string,
    req: Request,
    res: Response
  ): Promise<void> {
    // Rate limiting
    const clientId = this.getClientId(req);
    if (!this.rateLimiter.checkLimit(clientId)) {
      res.status(429).json({ error: 'Rate limit exceeded' });
      return;
    }

    // Get service
    const service = this.registry.getService(serviceId);
    if (!service) {
      res.status(404).json({ error: 'Service not found' });
      return;
    }

    // Forward request
    try {
      const targetUrl = `${service.endpoint}/${path}`;
      const response = await fetch(targetUrl, {
        method: req.method,
        headers: this.prepareHeaders(req.headers),
        body: req.method !== 'GET' ? req.body : undefined
      });

      // Forward response
      res.status(response.status);
      for (const [key, value] of response.headers) {
        res.setHeader(key, value);
      }
      res.send(await response.text());

    } catch (error) {
      this.logger.error('Proxy request failed', {
        serviceId,
        path,
        error
      });
      res.status(502).json({ error: 'Bad gateway' });
    }
  }

  private getClientId(req: Request): string {
    return req.headers['x-client-id'] as string || req.ip;
  }

  private prepareHeaders(headers: Headers): Record<string, string> {
    const prepared: Record<string, string> = {};
    for (const [key, value] of headers) {
      if (!key.startsWith('x-gateway-')) {
        prepared[key] = value;
      }
    }
    return prepared;
  }
}
```

---

## Circuit Breakers & Resilience

### Circuit Breaker Implementation

```typescript
/**
 * Circuit breaker states
 */
enum CircuitState {
  CLOSED = 'closed',
  OPEN = 'open',
  HALF_OPEN = 'half_open'
}

/**
 * Circuit breaker configuration
 */
interface CircuitBreakerConfig {
  failureThreshold: number;
  successThreshold: number;
  timeout: number;
  resetTimeout: number;
}

/**
 * Circuit breaker for fault tolerance
 */
class CircuitBreaker {
  private state: CircuitState = CircuitState.CLOSED;
  private failureCount = 0;
  private successCount = 0;
  private nextAttempt = 0;
  private logger: Logger;

  constructor(
    private name: string,
    private config: CircuitBreakerConfig
  ) {
    this.logger = createLogger(`circuit-breaker:${name}`);
  }

  /**
   * Execute operation with circuit breaker protection
   */
  async execute<T>(operation: () => Promise<T>): Promise<T> {
    if (this.state === CircuitState.OPEN) {
      if (Date.now() < this.nextAttempt) {
        throw new CircuitBreakerOpenError(
          `Circuit breaker ${this.name} is OPEN`
        );
      }

      // Try half-open state
      this.state = CircuitState.HALF_OPEN;
      this.logger.info('Circuit breaker entering HALF_OPEN state');
    }

    try {
      const result = await this.executeWithTimeout(operation);

      // Success
      this.onSuccess();
      return result;

    } catch (error) {
      // Failure
      this.onFailure();
      throw error;
    }
  }

  /**
   * Get current state
   */
  getState(): CircuitState {
    return this.state;
  }

  /**
   * Get metrics
   */
  getMetrics() {
    return {
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount
    };
  }

  /**
   * Reset circuit breaker
   */
  reset(): void {
    this.state = CircuitState.CLOSED;
    this.failureCount = 0;
    this.successCount = 0;
    this.nextAttempt = 0;
    this.logger.info('Circuit breaker reset');
  }

  private async executeWithTimeout<T>(
    operation: () => Promise<T>
  ): Promise<T> {
    return Promise.race([
      operation(),
      new Promise<never>((_, reject) =>
        setTimeout(
          () => reject(new Error('Operation timeout')),
          this.config.timeout
        )
      )
    ]);
  }

  private onSuccess(): void {
    this.failureCount = 0;

    if (this.state === CircuitState.HALF_OPEN) {
      this.successCount++;

      if (this.successCount >= this.config.successThreshold) {
        this.state = CircuitState.CLOSED;
        this.successCount = 0;
        this.logger.info('Circuit breaker CLOSED');
      }
    }
  }

  private onFailure(): void {
    this.failureCount++;
    this.successCount = 0;

    if (this.failureCount >= this.config.failureThreshold) {
      this.state = CircuitState.OPEN;
      this.nextAttempt = Date.now() + this.config.resetTimeout;
      this.logger.warn('Circuit breaker OPEN', {
        failures: this.failureCount,
        resetAt: new Date(this.nextAttempt)
      });
    }
  }
}

/**
 * Circuit breaker open error
 */
class CircuitBreakerOpenError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CircuitBreakerOpenError';
  }
}
```

### Retry with Exponential Backoff

```typescript
/**
 * Retry policy configuration
 */
interface RetryPolicy {
  maxAttempts: number;
  initialDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
  retryableErrors?: string[];
}

/**
 * Retry with exponential backoff
 */
class RetryManager {
  /**
   * Execute operation with retry
   */
  static async withRetry<T>(
    operation: () => Promise<T>,
    policy: RetryPolicy,
    logger?: Logger
  ): Promise<T> {
    let lastError: Error;
    let delay = policy.initialDelay;

    for (let attempt = 1; attempt <= policy.maxAttempts; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;

        // Check if error is retryable
        if (policy.retryableErrors && !this.isRetryable(error, policy.retryableErrors)) {
          throw error;
        }

        // Last attempt
        if (attempt === policy.maxAttempts) {
          break;
        }

        // Log retry
        logger?.warn('Operation failed, retrying...', {
          attempt,
          maxAttempts: policy.maxAttempts,
          delay,
          error: error.message
        });

        // Wait before retry
        await sleep(delay);

        // Calculate next delay
        delay = Math.min(
          delay * policy.backoffMultiplier,
          policy.maxDelay
        );
      }
    }

    throw new Error(
      `Operation failed after ${policy.maxAttempts} attempts: ${lastError!.message}`
    );
  }

  private static isRetryable(error: Error, retryableErrors: string[]): boolean {
    return retryableErrors.some(pattern =>
      error.name.includes(pattern) || error.message.includes(pattern)
    );
  }
}
```

### Bulkhead Pattern

```typescript
/**
 * Bulkhead for resource isolation
 */
class Bulkhead {
  private activeRequests = 0;
  private queue: Array<() => void> = [];
  private logger: Logger;

  constructor(
    private name: string,
    private maxConcurrent: number,
    private maxQueue: number
  ) {
    this.logger = createLogger(`bulkhead:${name}`);
  }

  /**
   * Execute operation with bulkhead protection
   */
  async execute<T>(operation: () => Promise<T>): Promise<T> {
    // Check if we can execute immediately
    if (this.activeRequests < this.maxConcurrent) {
      return this.executeOperation(operation);
    }

    // Check queue capacity
    if (this.queue.length >= this.maxQueue) {
      throw new BulkheadRejectedError(
        `Bulkhead ${this.name} queue full`
      );
    }

    // Queue the operation
    return new Promise((resolve, reject) => {
      const execute = () => {
        this.executeOperation(operation)
          .then(resolve)
          .catch(reject);
      };

      this.queue.push(execute);
    });
  }

  /**
   * Get metrics
   */
  getMetrics() {
    return {
      activeRequests: this.activeRequests,
      queuedRequests: this.queue.length,
      maxConcurrent: this.maxConcurrent,
      maxQueue: this.maxQueue
    };
  }

  private async executeOperation<T>(operation: () => Promise<T>): Promise<T> {
    this.activeRequests++;

    try {
      return await operation();
    } finally {
      this.activeRequests--;

      // Process next queued operation
      const next = this.queue.shift();
      if (next) {
        next();
      }
    }
  }
}

/**
 * Bulkhead rejected error
 */
class BulkheadRejectedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'BulkheadRejectedError';
  }
}
```

---

## Integration Testing

### Integration Test Framework

```typescript
/**
 * Integration test suite
 */
class IntegrationTestSuite {
  private messageBus: AgentMessageBus;
  private serviceRegistry: ServiceRegistry;
  private testAgents: Map<string, AgentIdentity>;

  async setup(): Promise<void> {
    // Initialize message bus
    this.messageBus = new AgentMessageBus();

    // Initialize service registry
    this.serviceRegistry = new ServiceRegistry();

    // Create test agents
    this.testAgents = new Map();
    this.testAgents.set('researcher', {
      id: 'test-researcher-1',
      type: 'researcher',
      capabilities: ['research', 'analysis'],
      status: AgentStatus.IDLE
    });
    this.testAgents.set('coder', {
      id: 'test-coder-1',
      type: 'coder',
      capabilities: ['coding', 'implementation'],
      status: AgentStatus.IDLE
    });

    // Register agents
    for (const agent of this.testAgents.values()) {
      this.messageBus.registerAgent(agent);
    }
  }

  async teardown(): Promise<void> {
    // Unregister agents
    for (const agent of this.testAgents.values()) {
      this.messageBus.unregisterAgent(agent.id);
    }

    // Clear registries
    this.testAgents.clear();
  }
}

/**
 * Integration test examples
 */
describe('VisionFlow Integration Tests', () => {
  let suite: IntegrationTestSuite;

  beforeEach(async () => {
    suite = new IntegrationTestSuite();
    await suite.setup();
  });

  afterEach(async () => {
    await suite.teardown();
  });

  it('should coordinate agents via message bus', async () => {
    const researcher = suite.testAgents.get('researcher')!;
    const coder = suite.testAgents.get('coder')!;

    // Setup response handler for coder
    suite.messageBus.subscribe(coder.id, async (msg) => {
      if (msg.type === AgentMessageType.QUERY) {
        await suite.messageBus.publish({
          id: generateId(),
          type: AgentMessageType.RESPONSE,
          from: coder,
          to: msg.from,
          timestamp: new Date(),
          payload: { result: 'Code implemented' },
          correlationId: msg.id
        });
      }
    });

    // Send request
    const result = await suite.messageBus.request(
      researcher,
      coder,
      { task: 'implement feature' }
    );

    expect(result).toEqual({ result: 'Code implemented' });
  });

  it('should handle circuit breaker', async () => {
    const breaker = new CircuitBreaker('test-service', {
      failureThreshold: 3,
      successThreshold: 2,
      timeout: 1000,
      resetTimeout: 5000
    });

    // Simulate failures
    for (let i = 0; i < 3; i++) {
      try {
        await breaker.execute(() => Promise.reject(new Error('Failure')));
      } catch {
        // Expected
      }
    }

    expect(breaker.getState()).toBe(CircuitState.OPEN);

    // Circuit should be open
    await expect(
      breaker.execute(() => Promise.resolve('success'))
    ).rejects.toThrow(CircuitBreakerOpenError);
  });
});
```

---

**End of Integration Patterns Documentation**

This comprehensive guide provides real-world patterns for integrating VisionFlow's 54+ agents, services, and external systems with fault-tolerance, observability, and testability.
