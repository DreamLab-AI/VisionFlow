/**
 * Unit tests for Prometheus metrics
 */

const { describe, it, expect, beforeEach } = require('@jest/globals');
const metrics = require('../../management-api/utils/metrics');

describe('Prometheus Metrics', () => {
  beforeEach(() => {
    // Reset metrics before each test
    metrics.register.resetMetrics();
  });

  describe('HTTP Request Metrics', () => {
    it('should record HTTP request duration', () => {
      metrics.recordHttpRequest('GET', '/api/v1/tasks', 200, 0.5);

      const metricsOutput = metrics.register.metrics();
      expect(metricsOutput).toContain('http_request_duration_seconds');
      expect(metricsOutput).toContain('method="GET"');
      expect(metricsOutput).toContain('route="/api/v1/tasks"');
      expect(metricsOutput).toContain('status_code="200"');
    });

    it('should increment request counter', () => {
      metrics.recordHttpRequest('POST', '/api/v1/tasks', 201, 1.0);
      metrics.recordHttpRequest('POST', '/api/v1/tasks', 201, 1.2);

      const metricsOutput = metrics.register.metrics();
      expect(metricsOutput).toContain('http_requests_total');
      expect(metricsOutput).toMatch(/http_requests_total{.*} 2/);
    });
  });

  describe('Task Metrics', () => {
    it('should record completed tasks', () => {
      metrics.recordTask('agent-task', 'success', 5.5);

      const metricsOutput = metrics.register.metrics();
      expect(metricsOutput).toContain('completed_tasks_total');
      expect(metricsOutput).toContain('task_duration_seconds');
    });

    it('should track active tasks', () => {
      metrics.setActiveTasks(5);

      const metricsOutput = metrics.register.metrics();
      expect(metricsOutput).toMatch(/active_tasks_total \d+/);
    });
  });

  describe('MCP Tool Metrics', () => {
    it('should record MCP tool invocations', () => {
      metrics.recordMCPTool('web-summary', 'success', 2.3);

      const metricsOutput = metrics.register.metrics();
      expect(metricsOutput).toContain('mcp_tool_invocations_total');
      expect(metricsOutput).toContain('tool_name="web-summary"');
      expect(metricsOutput).toContain('mcp_tool_duration_seconds');
    });

    it('should track failed tool invocations', () => {
      metrics.recordMCPTool('playwright', 'error', undefined);

      const metricsOutput = metrics.register.metrics();
      expect(metricsOutput).toContain('status="error"');
    });
  });

  describe('Error Metrics', () => {
    it('should record API errors', () => {
      metrics.recordError('ValidationError', '/api/v1/tasks');

      const metricsOutput = metrics.register.metrics();
      expect(metricsOutput).toContain('api_errors_total');
      expect(metricsOutput).toContain('error_type="ValidationError"');
    });
  });

  describe('Worker Session Metrics', () => {
    it('should track worker session count', () => {
      metrics.setWorkerSessions(3);

      const metricsOutput = metrics.register.metrics();
      expect(metricsOutput).toMatch(/worker_sessions_total \d+/);
    });
  });
});
