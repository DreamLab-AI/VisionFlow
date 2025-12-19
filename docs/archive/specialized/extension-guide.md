---
title: VisionFlow Extension Guide
description: 1. [Overview](#overview) 2. [Extension Architecture](#extension-architecture) 3. [Custom Handlers](#custom-handlers)
category: explanation
tags:
  - tutorial
  - api
  - documentation
  - reference
  - visionflow
updated-date: 2025-12-18
difficulty-level: advanced
---


# VisionFlow Extension Guide

## Table of Contents

1. [Overview](#overview)
2. [Extension Architecture](#extension-architecture)
3. [Custom Handlers](#custom-handlers)
4. [Custom Workers](#custom-workers)
5. 
6. [Custom Components](#custom-components)
7. 
8. 
9. [Testing Extensions](#testing-extensions)
10. [Publishing Extensions](#publishing-extensions)

---

## Overview

### What are VisionFlow Extensions?

VisionFlow extensions allow you to add custom functionality to the platform without modifying core code. Extensions can:

- Add new API endpoints (Handlers)
- Implement background jobs (Workers)
- Integrate external systems (Adapters)
- Create custom UI components (Components)
- Add domain-specific logic (Services)
- Implement specialized agents (Agents)

### Extension Benefits

```
┌──────────────────────────────────────────────────────────────┐
│                     VisionFlow Core                          │
│                    (Stable, Version 1.x)                     │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   │ Extension API (Stable Contract)
                   │
    ┌──────────────┼──────────────┬──────────────┬────────────┐
    │              │              │              │            │
┌───▼────┐  ┌─────▼─────┐  ┌────▼─────┐  ┌─────▼──────┐  ┌─▼────┐
│Payment │  │Analytics  │  │  Custom  │  │ Enterprise │  │ More │
│  Ext   │  │    Ext    │  │  AI Ext  │  │   SSO Ext  │  │ ...  │
└────────┘  └───────────┘  └──────────┘  └────────────┘  └──────┘

Benefits:
✓ No core code modification
✓ Independent versioning
✓ Hot reloading in development
✓ Easy distribution
✓ Isolated testing
```

### Extension Types

| Type | Purpose | Examples |
|------|---------|----------|
| **Handler** | HTTP API endpoints | REST APIs, webhooks, GraphQL |
| **Worker** | Background processing | Email sending, data sync, cleanup |
| **Adapter** | External integration | Payment gateways, cloud storage |
| **Component** | UI functionality | Custom forms, visualizations |
| **Service** | Business logic | Domain services, calculations |
| **Agent** | AI capabilities | Custom LLM agents, specialized tasks |

---

## Extension Architecture

### Extension Manifest

Every extension must include a `manifest.json` file:

```json
{
  "name": "payment-extension",
  "version": "1.0.0",
  "displayName": "Payment Gateway Extension",
  "description": "Integrate payment processing into VisionFlow",
  "author": "Your Name <email@example.com>",
  "license": "MIT",

  "visionflow": {
    "minVersion": "1.0.0",
    "maxVersion": "2.0.0"
  },

  "main": "dist/index.js",
  "types": "dist/index.d.ts",

  "provides": {
    "handlers": ["PaymentHandler", "WebhookHandler"],
    "workers": ["PaymentProcessorWorker"],
    "adapters": ["StripeAdapter", "PayPalAdapter"],
    "components": ["PaymentForm"],
    "services": ["PaymentService"],
    "agents": []
  },

  "dependencies": {
    "stripe": "^12.0.0",
    "paypal-rest-sdk": "^1.8.1"
  },

  "config": {
    "stripe": {
      "type": "object",
      "required": true,
      "properties": {
        "apiKey": {
          "type": "string",
          "required": true,
          "secret": true
        },
        "webhookSecret": {
          "type": "string",
          "required": true,
          "secret": true
        }
      }
    }
  },

  "permissions": [
    "database:write",
    "api:external",
    "events:publish"
  ]
}
```

### Extension Structure

```
payment-extension/
├── manifest.json              # Extension metadata
├── package.json               # NPM package config
├── tsconfig.json              # TypeScript config
├── README.md                  # Documentation
├── LICENSE                    # License file
│
├── src/
│   ├── index.ts              # Main entry point
│   ├── handlers/             # HTTP handlers
│   │   ├── PaymentHandler.ts
│   │   └── WebhookHandler.ts
│   ├── workers/              # Background workers
│   │   └── PaymentProcessorWorker.ts
│   ├── adapters/             # External integrations
│   │   ├── StripeAdapter.ts
│   │   └── PayPalAdapter.ts
│   ├── components/           # UI components
│   │   └── PaymentForm.vue
│   ├── services/             # Business logic
│   │   └── PaymentService.ts
│   └── types/                # TypeScript types
│       └── index.ts
│
├── tests/                     # Test files
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
└── dist/                      # Built files (gitignored)
```

### Extension Entry Point

```typescript
// src/index.ts
import { ExtensionContext, Extension } from '@visionflow/extension-api';
import { PaymentHandler, WebhookHandler } from './handlers';
import { PaymentProcessorWorker } from './workers';
import { StripeAdapter, PayPalAdapter } from './adapters';
import { PaymentService } from './services';

/**
 * Extension entry point
 */
export default class PaymentExtension implements Extension {
  private context?: ExtensionContext;
  private paymentService?: PaymentService;

  /**
   * Extension activation
   */
  async activate(context: ExtensionContext): Promise<void> {
    this.context = context;

    // Get configuration
    const config = context.getConfig();
    context.logger.info('Payment extension activating', {
      version: context.manifest.version
    });

    // Initialize service
    this.paymentService = new PaymentService(
      context.database,
      context.logger.child('payment-service')
    );

    // Register adapters
    const stripeAdapter = new StripeAdapter(
      'stripe',
      config.stripe.apiKey,
      { enabled: true }
    );
    await stripeAdapter.initialize();
    context.registerAdapter(stripeAdapter);

    // Register handlers
    context.registerHandler(
      new PaymentHandler(this.paymentService, context.logger)
    );
    context.registerHandler(
      new WebhookHandler(this.paymentService, config.stripe.webhookSecret)
    );

    // Register workers
    context.registerWorker(
      new PaymentProcessorWorker(this.paymentService, context.logger)
    );

    // Subscribe to events
    context.events.on('order.created', async (order) => {
      await this.paymentService.processPayment(order);
    });

    context.logger.info('Payment extension activated');
  }

  /**
   * Extension deactivation
   */
  async deactivate(): Promise<void> {
    if (this.context) {
      this.context.logger.info('Payment extension deactivating');
      // Cleanup resources
      await this.paymentService?.dispose();
      this.context.logger.info('Payment extension deactivated');
    }
  }

  /**
   * Extension health check
   */
  async healthCheck(): Promise<boolean> {
    if (!this.paymentService) {
      return false;
    }
    return await this.paymentService.healthCheck();
  }
}
```

---

## Custom Handlers

### Handler Interface

```typescript
import { Request, Response, NextFunction } from 'express';

/**
 * HTTP handler interface
 */
interface Handler {
  /**
   * Handler route path
   */
  readonly path: string;

  /**
   * HTTP method(s)
   */
  readonly method: HttpMethod | HttpMethod[];

  /**
   * Handler middleware
   */
  readonly middleware?: RequestMiddleware[];

  /**
   * Handle the request
   */
  handle(req: Request, res: Response, next: NextFunction): Promise<void> | void;
}

/**
 * HTTP methods
 */
type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH' | 'OPTIONS';

/**
 * Request middleware
 */
type RequestMiddleware = (
  req: Request,
  res: Response,
  next: NextFunction
) => Promise<void> | void;
```

### Example: REST API Handler

```typescript
import { Handler, HttpMethod } from '@visionflow/extension-api';

/**
 * Payment API handler
 */
export class PaymentHandler implements Handler {
  readonly path = '/api/payments';
  readonly method: HttpMethod[] = ['GET', 'POST'];

  readonly middleware = [
    this.authenticate.bind(this),
    this.validateRequest.bind(this)
  ];

  constructor(
    private paymentService: PaymentService,
    private logger: Logger
  ) {}

  /**
   * Handle payment requests
   */
  async handle(req: Request, res: Response, next: NextFunction): Promise<void> {
    try {
      switch (req.method) {
        case 'GET':
          await this.handleGet(req, res);
          break;
        case 'POST':
          await this.handlePost(req, res);
          break;
        default:
          res.status(405).json({ error: 'Method not allowed' });
      }
    } catch (error) {
      this.logger.error('Payment handler error', { error });
      next(error);
    }
  }

  /**
   * Get payment by ID
   */
  private async handleGet(req: Request, res: Response): Promise<void> {
    const { id } = req.query;

    if (!id) {
      res.status(400).json({ error: 'Payment ID required' });
      return;
    }

    const payment = await this.paymentService.getPayment(id as string);

    if (!payment) {
      res.status(404).json({ error: 'Payment not found' });
      return;
    }

    res.json({ payment });
  }

  /**
   * Create new payment
   */
  private async handlePost(req: Request, res: Response): Promise<void> {
    const { amount, currency, source, description } = req.body;

    // Validate request
    if (!amount || !currency || !source) {
      res.status(400).json({ error: 'Invalid payment request' });
      return;
    }

    // Create payment
    const payment = await this.paymentService.createPayment({
      amount,
      currency,
      source,
      description,
      userId: req.user.id
    });

    res.status(201).json({ payment });
  }

  /**
   * Authentication middleware
   */
  private async authenticate(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    const token = req.headers.authorization?.replace('Bearer ', '');

    if (!token) {
      res.status(401).json({ error: 'Authentication required' });
      return;
    }

    try {
      const user = await this.verifyToken(token);
      req.user = user;
      next();
    } catch (error) {
      res.status(401).json({ error: 'Invalid token' });
    }
  }

  /**
   * Request validation middleware
   */
  private validateRequest(
    req: Request,
    res: Response,
    next: NextFunction
  ): void {
    if (req.method === 'POST') {
      const errors = this.validatePaymentRequest(req.body);
      if (errors.length > 0) {
        res.status(400).json({ errors });
        return;
      }
    }
    next();
  }

  private validatePaymentRequest(body: any): string[] {
    const errors: string[] = [];

    if (typeof body.amount !== 'number' || body.amount <= 0) {
      errors.push('Invalid amount');
    }

    if (!['USD', 'EUR', 'GBP'].includes(body.currency)) {
      errors.push('Invalid currency');
    }

    if (!body.source) {
      errors.push('Payment source required');
    }

    return errors;
  }

  private async verifyToken(token: string): Promise<any> {
    // Token verification logic
    return { id: 'user-123', email: 'user@example.com' };
  }
}
```

### Example: GraphQL Handler

```typescript
import { buildSchema } from 'graphql';
import { graphqlHTTP } from 'express-graphql';

/**
 * GraphQL handler
 */
export class GraphQLHandler implements Handler {
  readonly path = '/graphql';
  readonly method = 'POST';

  private schema = buildSchema(`
    type Payment {
      id: ID!
      amount: Float!
      currency: String!
      status: String!
      createdAt: String!
    }

    type Query {
      payment(id: ID!): Payment
      payments(limit: Int, offset: Int): [Payment!]!
    }

    type Mutation {
      createPayment(
        amount: Float!
        currency: String!
        source: String!
        description: String
      ): Payment!
    }
  `);

  constructor(private paymentService: PaymentService) {}

  async handle(req: Request, res: Response, next: NextFunction): Promise<void> {
    const handler = graphqlHTTP({
      schema: this.schema,
      rootValue: this.getRootResolvers(),
      graphiql: process.env.NODE_ENV === 'development'
    });

    return handler(req, res);
  }

  private getRootResolvers() {
    return {
      payment: async ({ id }: { id: string }) => {
        return await this.paymentService.getPayment(id);
      },

      payments: async ({ limit = 10, offset = 0 }) => {
        return await this.paymentService.listPayments({ limit, offset });
      },

      createPayment: async (args: any) => {
        return await this.paymentService.createPayment(args);
      }
    };
  }
}
```

---

## Custom Workers

### Worker Interface

```typescript
/**
 * Background worker interface
 */
interface Worker {
  /**
   * Worker name
   */
  readonly name: string;

  /**
   * Worker schedule (cron expression)
   */
  readonly schedule?: string;

  /**
   * Worker concurrency
   */
  readonly concurrency?: number;

  /**
   * Execute worker task
   */
  execute(context: WorkerContext): Promise<void>;
}

/**
 * Worker execution context
 */
interface WorkerContext {
  logger: Logger;
  database: DatabaseAdapter;
  redis: RedisAdapter;
  signal: AbortSignal;
}
```

### Example: Scheduled Worker

```typescript
/**
 * Payment processor worker (runs every 5 minutes)
 */
export class PaymentProcessorWorker implements Worker {
  readonly name = 'payment-processor';
  readonly schedule = '*/5 * * * *'; // Every 5 minutes
  readonly concurrency = 3;

  constructor(
    private paymentService: PaymentService,
    private logger: Logger
  ) {}

  /**
   * Process pending payments
   */
  async execute(context: WorkerContext): Promise<void> {
    this.logger.info('Processing pending payments');

    try {
      // Get pending payments
      const pendingPayments = await this.paymentService.getPendingPayments();

      this.logger.info('Found pending payments', {
        count: pendingPayments.length
      });

      // Process each payment
      for (const payment of pendingPayments) {
        // Check if worker should stop
        if (context.signal.aborted) {
          this.logger.info('Worker stopped');
          break;
        }

        await this.processPayment(payment, context);
      }

      this.logger.info('Payment processing completed');
    } catch (error) {
      this.logger.error('Payment processing failed', { error });
      throw error;
    }
  }

  private async processPayment(
    payment: Payment,
    context: WorkerContext
  ): Promise<void> {
    this.logger.debug('Processing payment', { paymentId: payment.id });

    try {
      // Process the payment
      const result = await this.paymentService.processPayment(payment);

      // Update payment status
      await this.paymentService.updatePaymentStatus(
        payment.id,
        result.status
      );

      // Send notification
      await this.sendPaymentNotification(payment, result);

      this.logger.info('Payment processed', {
        paymentId: payment.id,
        status: result.status
      });
    } catch (error) {
      this.logger.error('Payment processing error', {
        paymentId: payment.id,
        error
      });

      // Mark payment as failed
      await this.paymentService.updatePaymentStatus(
        payment.id,
        'failed',
        error.message
      );
    }
  }

  private async sendPaymentNotification(
    payment: Payment,
    result: PaymentResult
  ): Promise<void> {
    // Send email/webhook notification
  }
}
```

### Example: Queue-Based Worker

```typescript
import Bull from 'bull';

/**
 * Email sender worker (processes queue)
 */
export class EmailWorker implements Worker {
  readonly name = 'email-sender';
  readonly concurrency = 5;

  private queue: Bull.Queue;

  constructor(
    private emailService: EmailService,
    private logger: Logger,
    redisUrl: string
  ) {
    this.queue = new Bull('email-queue', redisUrl);
    this.setupQueue();
  }

  /**
   * Process email queue
   */
  async execute(context: WorkerContext): Promise<void> {
    // Queue processing handled by Bull
    // This method can be used for health checks
    const stats = await this.queue.getJobCounts();
    this.logger.info('Email queue stats', stats);
  }

  private setupQueue(): void {
    this.queue.process(this.concurrency!, async (job) => {
      const { to, subject, body, template } = job.data;

      this.logger.debug('Sending email', { to, subject });

      try {
        await this.emailService.sendEmail({
          to,
          subject,
          body,
          template
        });

        this.logger.info('Email sent', { to, subject });
      } catch (error) {
        this.logger.error('Email sending failed', { to, subject, error });
        throw error;
      }
    });

    this.queue.on('failed', (job, error) => {
      this.logger.error('Email job failed', {
        jobId: job.id,
        error
      });
    });
  }

  /**
   * Add email to queue
   */
  async queueEmail(email: EmailData): Promise<void> {
    await this.queue.add(email, {
      attempts: 3,
      backoff: {
        type: 'exponential',
        delay: 2000
      }
    });
  }
}
```

---

## Custom Components

### Vue Component Structure

```vue
<!-- src/components/PaymentForm.vue -->
<template>
  <div class="payment-form">
    <h2>Payment Information</h2>

    <form @submit.prevent="handleSubmit">
      <!-- Amount -->
      <div class="form-group">
        <label for="amount">Amount</label>
        <input
          id="amount"
          v-model.number="form.amount"
          type="number"
          min="0"
          step="0.01"
          required
          class="form-control"
        />
        <span v-if="errors.amount" class="error">{{ errors.amount }}</span>
      </div>

      <!-- Currency -->
      <div class="form-group">
        <label for="currency">Currency</label>
        <select
          id="currency"
          v-model="form.currency"
          required
          class="form-control"
        >
          <option value="USD">USD</option>
          <option value="EUR">EUR</option>
          <option value="GBP">GBP</option>
        </select>
      </div>

      <!-- Card Information -->
      <div class="form-group">
        <label for="cardNumber">Card Number</label>
        <div id="card-element" ref="cardElement"></div>
        <span v-if="errors.card" class="error">{{ errors.card }}</span>
      </div>

      <!-- Submit Button -->
      <button
        type="submit"
        :disabled="processing"
        class="btn btn-primary"
      >
        {{ processing ? 'Processing...' : 'Pay Now' }}
      </button>
    </form>

    <!-- Success Message -->
    <div v-if="paymentSuccess" class="alert alert-success">
      Payment successful! Transaction ID: {{ transactionId }}
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, onMounted } from 'vue';
import { loadStripe, Stripe, StripeElements } from '@stripe/stripe-js';

export default defineComponent({
  name: 'PaymentForm',

  props: {
    publicKey: {
      type: String,
      required: true
    }
  },

  emits: ['payment-success', 'payment-error'],

  setup(props, { emit }) {
    const form = ref({
      amount: 0,
      currency: 'USD'
    });

    const errors = ref<Record<string, string>>({});
    const processing = ref(false);
    const paymentSuccess = ref(false);
    const transactionId = ref('');

    const cardElement = ref<HTMLElement>();
    let stripe: Stripe | null = null;
    let elements: StripeElements | null = null;
    let card: any = null;

    onMounted(async () => {
      // Initialize Stripe
      stripe = await loadStripe(props.publicKey);
      if (stripe) {
        elements = stripe.elements();
        card = elements.create('card');
        card.mount(cardElement.value);

        card.on('change', (event: any) => {
          if (event.error) {
            errors.value.card = event.error.message;
          } else {
            delete errors.value.card;
          }
        });
      }
    });

    const validateForm = (): boolean => {
      errors.value = {};

      if (form.value.amount <= 0) {
        errors.value.amount = 'Amount must be greater than 0';
      }

      if (!['USD', 'EUR', 'GBP'].includes(form.value.currency)) {
        errors.value.currency = 'Invalid currency';
      }

      return Object.keys(errors.value).length === 0;
    };

    const handleSubmit = async () => {
      if (!validateForm() || !stripe || !card) {
        return;
      }

      processing.value = true;

      try {
        // Create payment intent
        const response = await fetch('/api/payments/intent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            amount: form.value.amount * 100, // Convert to cents
            currency: form.value.currency
          })
        });

        const { clientSecret } = await response.json();

        // Confirm payment
        const result = await stripe.confirmCardPayment(clientSecret, {
          payment_method: {
            card: card
          }
        });

        if (result.error) {
          errors.value.card = result.error.message || 'Payment failed';
          emit('payment-error', result.error);
        } else {
          paymentSuccess.value = true;
          transactionId.value = result.paymentIntent.id;
          emit('payment-success', result.paymentIntent);

          // Reset form
          form.value.amount = 0;
          card.clear();
        }
      } catch (error) {
        errors.value.card = 'Payment processing error';
        emit('payment-error', error);
      } finally {
        processing.value = false;
      }
    };

    return {
      form,
      errors,
      processing,
      paymentSuccess,
      transactionId,
      cardElement,
      handleSubmit
    };
  }
});
</script>

<style scoped>
.payment-form {
  max-width: 500px;
  margin: 0 auto;
  padding: 2rem;
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-control {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
}

#card-element {
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.error {
  color: #dc3545;
  font-size: 0.875rem;
  margin-top: 0.25rem;
}

.btn {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
}

.btn-primary {
  background-color: #007bff;
  color: white;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.alert {
  padding: 1rem;
  margin-top: 1rem;
  border-radius: 4px;
}

.alert-success {
  background-color: #d4edda;
  color: #155724;
  border: 1px solid #c3e6cb;
}
</style>
```

### Component Registration

```typescript
// Register component in extension
import PaymentForm from './components/PaymentForm.vue';

export default class PaymentExtension implements Extension {
  async activate(context: ExtensionContext): Promise<void> {
    // Register Vue component
    context.registerComponent('PaymentForm', PaymentForm);

    // Make component available globally
    context.app.component('PaymentForm', PaymentForm);
  }
}
```

---

## Testing Extensions

### Unit Tests

```typescript
// tests/unit/PaymentService.test.ts
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { PaymentService } from '../../src/services/PaymentService';
import { MockDatabaseAdapter } from '../mocks/DatabaseAdapter';

describe('PaymentService', () => {
  let service: PaymentService;
  let mockDb: MockDatabaseAdapter;

  beforeEach(() => {
    mockDb = new MockDatabaseAdapter();
    service = new PaymentService(mockDb, createMockLogger());
  });

  it('should create payment', async () => {
    const payment = await service.createPayment({
      amount: 100,
      currency: 'USD',
      source: 'tok_visa',
      userId: 'user-123'
    });

    expect(payment).toBeDefined();
    expect(payment.amount).toBe(100);
    expect(payment.currency).toBe('USD');
    expect(payment.status).toBe('pending');
  });

  it('should process payment', async () => {
    const payment = await service.createPayment({
      amount: 100,
      currency: 'USD',
      source: 'tok_visa',
      userId: 'user-123'
    });

    const result = await service.processPayment(payment);

    expect(result.status).toBe('succeeded');
    expect(result.transactionId).toBeDefined();
  });

  it('should handle payment failure', async () => {
    const payment = await service.createPayment({
      amount: 100,
      currency: 'USD',
      source: 'tok_chargeDeclined',
      userId: 'user-123'
    });

    await expect(service.processPayment(payment)).rejects.toThrow();
  });
});
```

### Integration Tests

```typescript
// tests/integration/PaymentFlow.test.ts
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createTestApp } from '../helpers/test-app';
import request from 'supertest';

describe('Payment Flow Integration', () => {
  let app: any;
  let authToken: string;

  beforeAll(async () => {
    app = await createTestApp();
    authToken = await app.generateTestToken();
  });

  afterAll(async () => {
    await app.cleanup();
  });

  it('should complete payment flow', async () => {
    // Create payment intent
    const intentResponse = await request(app.server)
      .post('/api/payments/intent')
      .set('Authorization', `Bearer ${authToken}`)
      .send({
        amount: 10000, // $100.00
        currency: 'USD'
      })
      .expect(200);

    expect(intentResponse.body.clientSecret).toBeDefined();

    // Confirm payment (simulated)
    const confirmResponse = await request(app.server)
      .post('/api/payments/confirm')
      .set('Authorization', `Bearer ${authToken}`)
      .send({
        paymentIntentId: intentResponse.body.id,
        paymentMethod: 'pm_card_visa'
      })
      .expect(200);

    expect(confirmResponse.body.status).toBe('succeeded');

    // Verify payment was recorded
    const paymentResponse = await request(app.server)
      .get(`/api/payments/${confirmResponse.body.id}`)
      .set('Authorization', `Bearer ${authToken}`)
      .expect(200);

    expect(paymentResponse.body.payment.status).toBe('succeeded');
  });
});
```

---

---

---

## Related Documentation

- [Integration Patterns in VisionFlow](../../explanations/architecture/integration-patterns.md)
- [DeepSeek User Setup - Complete](../reports/documentation-alignment-2025-12-02/DEEPSEEK_SETUP_COMPLETE.md)
- [Rust Type Correction Guide](../fixes/rust-type-correction-guide.md)
- [Borrow Checker Error Fixes - Documentation](../fixes/README.md)
- [VisionFlow Test Suite](../tests/test_README.md)

## Publishing Extensions

### Package Preparation

```json
// package.json
{
  "name": "@visionflow/payment-extension",
  "version": "1.0.0",
  "description": "Payment gateway integration for VisionFlow",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "files": [
    "dist",
    "manifest.json",
    "README.md",
    "LICENSE"
  ],
  "scripts": {
    "build": "tsc",
    "test": "vitest run",
    "test:watch": "vitest",
    "lint": "eslint src",
    "prepare": "npm run build"
  },
  "keywords": [
    "visionflow",
    "extension",
    "payment",
    "stripe",
    "paypal"
  ],
  "peerDependencies": {
    "@visionflow/extension-api": "^1.0.0"
  }
}
```

### Publishing Checklist

- [ ] Update version in `manifest.json` and `package.json`
- [ ] Write comprehensive README.md
- [ ] Add LICENSE file
- [ ] Build distribution files (`npm run build`)
- [ ] Run all tests (`npm test`)
- [ ] Update CHANGELOG.md
- [ ] Create git tag (`git tag v1.0.0`)
- [ ] Publish to npm (`npm publish`)
- [ ] Create GitHub release

### Extension Registry

```bash
# Publish to VisionFlow extension registry
visionflow extension publish

# Install extension
visionflow extension install @visionflow/payment-extension

# List installed extensions
visionflow extension list

# Update extension
visionflow extension update @visionflow/payment-extension

# Remove extension
visionflow extension remove @visionflow/payment-extension
```

---

**End of Extension Guide**

This comprehensive guide provides everything needed to create, test, and publish VisionFlow extensions with real-world examples and best practices.
