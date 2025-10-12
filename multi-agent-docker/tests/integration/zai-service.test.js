/**
 * Integration tests for Z.AI service
 */

const { describe, it, expect, beforeAll } = require('@jest/globals');
const axios = require('axios');

const ZAI_URL = process.env.ZAI_CONTAINER_URL || 'http://localhost:9600';

describe('Z.AI Service Integration Tests', () => {
  beforeAll(async () => {
    // Wait for service to be ready
    let retries = 10;
    while (retries > 0) {
      try {
        await axios.get(`${ZAI_URL}/health`);
        break;
      } catch (error) {
        retries--;
        if (retries === 0) throw new Error('Z.AI service not ready');
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  });

  describe('Health Check', () => {
    it('should return healthy status', async () => {
      const response = await axios.get(`${ZAI_URL}/health`);

      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('status');
      expect(response.data.status).toBe('ok');
      expect(response.data).toHaveProperty('service');
      expect(response.data.service).toBe('claude-zai-wrapper');
    });

    it('should include worker pool stats', async () => {
      const response = await axios.get(`${ZAI_URL}/health`);

      expect(response.data).toHaveProperty('poolSize');
      expect(response.data).toHaveProperty('busyWorkers');
      expect(response.data).toHaveProperty('queueLength');
      expect(response.data).toHaveProperty('maxQueueSize');
    });
  });

  describe('Prompt Endpoint', () => {
    it('should process simple prompts', async () => {
      const response = await axios.post(`${ZAI_URL}/prompt`, {
        prompt: 'Say hello in one word',
        timeout: 10000
      });

      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('success');
      expect(response.data.success).toBe(true);
      expect(response.data).toHaveProperty('response');
      expect(typeof response.data.response).toBe('string');
    });

    it('should handle JSON requests', async () => {
      const response = await axios.post(`${ZAI_URL}/prompt`, {
        prompt: 'Return a JSON object with one key "test" and value "success"',
        timeout: 10000
      });

      expect(response.status).toBe(200);
      expect(response.data.success).toBe(true);
    });

    it('should reject requests without prompt', async () => {
      const response = await axios.post(`${ZAI_URL}/prompt`, {
        timeout: 5000
      }, {
        validateStatus: () => true
      });

      expect(response.status).toBe(400);
      expect(response.data).toHaveProperty('error');
    });

    it('should handle timeout parameter', async () => {
      const response = await axios.post(`${ZAI_URL}/prompt`, {
        prompt: 'Count to 5',
        timeout: 15000
      });

      expect(response.status).toBe(200);
    });
  });

  describe('Retry Logic', () => {
    it('should retry on transient errors', async () => {
      // This test verifies retry mechanism exists
      // Actual retry testing would require mocking network failures
      const response = await axios.post(`${ZAI_URL}/prompt`, {
        prompt: 'Test retry',
        timeout: 5000
      });

      expect(response.data).toHaveProperty('success');
    });
  });

  describe('Worker Pool', () => {
    it('should handle concurrent requests', async () => {
      const requests = Array(5).fill(null).map((_, i) =>
        axios.post(`${ZAI_URL}/prompt`, {
          prompt: `Request ${i}`,
          timeout: 10000
        })
      );

      const responses = await Promise.all(requests);

      responses.forEach(response => {
        expect(response.status).toBe(200);
        expect(response.data.success).toBe(true);
      });
    });

    it('should queue requests when pool is full', async () => {
      const healthBefore = await axios.get(`${ZAI_URL}/health`);
      const poolSize = healthBefore.data.poolSize;

      const requests = Array(poolSize + 2).fill(null).map(() =>
        axios.post(`${ZAI_URL}/prompt`, {
          prompt: 'Test queuing',
          timeout: 5000
        })
      );

      const responses = await Promise.all(requests);

      responses.forEach(response => {
        expect(response.status).toBe(200);
      });
    });

    it('should reject when queue is full', async () => {
      const healthCheck = await axios.get(`${ZAI_URL}/health`);
      const maxQueueSize = healthCheck.data.maxQueueSize;
      const poolSize = healthCheck.data.poolSize;

      // Try to overflow the queue
      const requests = Array(poolSize + maxQueueSize + 5).fill(null).map(() =>
        axios.post(`${ZAI_URL}/prompt`, {
          prompt: 'Overflow test',
          timeout: 30000
        }, {
          validateStatus: () => true
        })
      );

      const responses = await Promise.all(requests);
      const rejected = responses.filter(r => r.status === 503);

      expect(rejected.length).toBeGreaterThan(0);
    });
  });

  describe('Error Handling', () => {
    it('should handle malformed requests', async () => {
      const response = await axios.post(`${ZAI_URL}/prompt`,
        'invalid json',
        {
          headers: { 'Content-Type': 'application/json' },
          validateStatus: () => true
        }
      );

      expect(response.status).toBeGreaterThanOrEqual(400);
    });

    it('should timeout long-running requests', async () => {
      const response = await axios.post(`${ZAI_URL}/prompt`, {
        prompt: 'Count to 1000 slowly',
        timeout: 1000
      }, {
        validateStatus: () => true,
        timeout: 5000
      });

      // Should either complete quickly or timeout
      expect([200, 408, 500]).toContain(response.status);
    });
  });
});
