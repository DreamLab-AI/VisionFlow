/**
 * Integration tests for Management API
 */

const { describe, it, expect, beforeAll, afterAll } = require('@jest/globals');
const axios = require('axios');

const BASE_URL = process.env.API_BASE_URL || 'http://localhost:9090';
const API_KEY = process.env.MANAGEMENT_API_KEY || 'change-this-secret-key';

const client = axios.create({
  baseURL: BASE_URL,
  headers: {
    'X-API-Key': API_KEY
  },
  validateStatus: () => true // Don't throw on any status
});

describe('Management API Integration Tests', () => {
  describe('Health Endpoints', () => {
    it('should return health status', async () => {
      const response = await axios.get(`${BASE_URL}/health`);

      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('status');
      expect(response.data.status).toBe('ok');
    });

    it('should return ready status', async () => {
      const response = await axios.get(`${BASE_URL}/ready`);

      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('status');
    });
  });

  describe('Metrics Endpoint', () => {
    it('should return Prometheus metrics', async () => {
      const response = await axios.get(`${BASE_URL}/metrics`);

      expect(response.status).toBe(200);
      expect(response.headers['content-type']).toContain('text/plain');
      expect(response.data).toContain('# HELP');
      expect(response.data).toContain('# TYPE');
    });

    it('should include custom metrics', async () => {
      const response = await axios.get(`${BASE_URL}/metrics`);

      expect(response.data).toContain('http_request_duration_seconds');
      expect(response.data).toContain('http_requests_total');
      expect(response.data).toContain('active_tasks_total');
    });
  });

  describe('API Documentation', () => {
    it('should serve Swagger UI', async () => {
      const response = await axios.get(`${BASE_URL}/docs`);

      expect(response.status).toBe(200);
      expect(response.headers['content-type']).toContain('text/html');
    });

    it('should provide OpenAPI JSON spec', async () => {
      const response = await axios.get(`${BASE_URL}/docs/json`);

      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('openapi');
      expect(response.data).toHaveProperty('info');
      expect(response.data).toHaveProperty('paths');
    });
  });

  describe('Authentication', () => {
    it('should reject requests without API key', async () => {
      const response = await axios.get(`${BASE_URL}/v1/status`);

      expect(response.status).toBe(401);
    });

    it('should reject requests with invalid API key', async () => {
      const response = await axios.get(`${BASE_URL}/v1/status`, {
        headers: { 'X-API-Key': 'invalid-key' }
      });

      expect(response.status).toBe(401);
    });

    it('should accept requests with valid API key', async () => {
      const response = await client.get('/v1/status');

      expect(response.status).toBe(200);
    });
  });

  describe('Rate Limiting', () => {
    it('should enforce rate limits', async () => {
      const requests = Array(150).fill(null).map(() =>
        client.get('/v1/status')
      );

      const responses = await Promise.all(requests);
      const rateLimited = responses.filter(r => r.status === 429);

      expect(rateLimited.length).toBeGreaterThan(0);
    });
  });

  describe('CORS', () => {
    it('should include CORS headers', async () => {
      const response = await axios.options(`${BASE_URL}/health`);

      expect(response.headers).toHaveProperty('access-control-allow-origin');
    });
  });

  describe('Root Endpoint', () => {
    it('should return API information', async () => {
      const response = await axios.get(BASE_URL);

      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('name');
      expect(response.data).toHaveProperty('version');
      expect(response.data).toHaveProperty('endpoints');
      expect(response.data).toHaveProperty('documentation');
      expect(response.data.documentation).toBe('/docs');
    });
  });
});
