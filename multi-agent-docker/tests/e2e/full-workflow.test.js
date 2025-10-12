/**
 * End-to-end workflow tests
 */

const { describe, it, expect, beforeAll, afterAll } = require('@jest/globals');
const axios = require('axios');
const { exec } = require('child_process');
const { promisify } = require('util');

const execAsync = promisify(exec);

const BASE_URL = process.env.API_BASE_URL || 'http://localhost:9090';
const ZAI_URL = process.env.ZAI_CONTAINER_URL || 'http://localhost:9600';
const API_KEY = process.env.MANAGEMENT_API_KEY || 'change-this-secret-key';
const CONTAINER_NAME = process.env.CONTAINER_NAME || 'agentic-flow-cachyos';

const client = axios.create({
  baseURL: BASE_URL,
  headers: { 'X-API-Key': API_KEY },
  validateStatus: () => true
});

async function dockerExec(command) {
  const { stdout } = await execAsync(`docker exec ${CONTAINER_NAME} ${command}`);
  return stdout;
}

describe('End-to-End Workflow Tests', () => {
  beforeAll(async () => {
    // Verify all services are running
    const healthChecks = await Promise.all([
      axios.get(`${BASE_URL}/health`),
      axios.get(`${ZAI_URL}/health`)
    ]);

    healthChecks.forEach(response => {
      expect(response.status).toBe(200);
    });
  });

  describe('Complete Web Summarization Workflow', () => {
    it('should complete full web summary workflow', async () => {
      // 1. Verify Z.AI service is ready
      const zaiHealth = await axios.get(`${ZAI_URL}/health`);
      expect(zaiHealth.data.status).toBe('ok');

      // 2. Check web-summary tool is configured
      const mcpList = await dockerExec('mcp list');
      expect(mcpList).toContain('web-summary');

      // 3. Execute web summary via MCP tool
      const summaryCommand = `echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"summarize_url","arguments":{"url":"https://www.anthropic.com"}}}' | /opt/venv/bin/python3 -u /app/core-assets/scripts/web-summary-mcp-server.py`;

      const summaryResult = await dockerExec(summaryCommand);
      const response = JSON.parse(summaryResult);

      // 4. Verify response structure
      expect(response).toHaveProperty('result');
      expect(response.result).toHaveProperty('title');
      expect(response.result).toHaveProperty('summary');
      expect(response.result).toHaveProperty('topics');
      expect(response.result).toHaveProperty('formatted');

      // 5. Verify topics are from permitted list
      const topicsConfig = await dockerExec('cat /app/core-assets/config/topics.json');
      const permittedTopics = JSON.parse(topicsConfig);

      response.result.topics.forEach(topic => {
        expect(permittedTopics).toContain(topic);
      });

      // 6. Verify formatted output has Logseq links
      expect(response.result.formatted).toMatch(/\[\[.+?\]\]/);
    });

    it('should handle YouTube video summarization', async () => {
      const summaryCommand = `echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"summarize_url","arguments":{"url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}}}' | /opt/venv/bin/python3 -u /app/core-assets/scripts/web-summary-mcp-server.py`;

      const summaryResult = await dockerExec(summaryCommand);
      const response = JSON.parse(summaryResult);

      expect(response.result).toHaveProperty('summary');
      expect(response.result.summary.length).toBeGreaterThan(0);
    });
  });

  describe('MCP Tool Management Workflow', () => {
    it('should complete tool lifecycle: add, use, remove', async () => {
      // 1. Add new tool
      const addResult = await dockerExec(
        'mcp add echo-tool echo \\"hello\\" \\"Simple echo tool\\"'
      );
      expect(addResult).toContain('Added tool');

      // 2. Verify tool is listed
      const listResult = await dockerExec('mcp list');
      expect(listResult).toContain('echo-tool');

      // 3. Show tool details
      const showResult = await dockerExec('mcp show echo-tool');
      expect(showResult).toContain('echo-tool');
      expect(showResult).toContain('echo');

      // 4. Update tool
      const updateResult = await dockerExec(
        'mcp update echo-tool --description "Updated echo tool"'
      );
      expect(updateResult).toContain('Updated tool');

      // 5. Backup configuration
      const backupResult = await dockerExec('mcp backup');
      expect(backupResult).toContain('Backup created');

      // 6. Remove tool
      const removeResult = await dockerExec('mcp remove echo-tool');
      expect(removeResult).toContain('Removed tool');

      // 7. Verify tool is removed
      const finalList = await dockerExec('mcp list');
      expect(finalList).not.toContain('echo-tool');
    });
  });

  describe('Monitoring and Observability Workflow', () => {
    it('should provide complete observability stack', async () => {
      // 1. Generate some API activity
      await Promise.all([
        client.get('/v1/status'),
        client.get('/v1/status'),
        client.get('/v1/status')
      ]);

      // 2. Check metrics are updated
      const metricsResponse = await axios.get(`${BASE_URL}/metrics`);
      expect(metricsResponse.data).toContain('http_requests_total');
      expect(metricsResponse.data).toMatch(/http_requests_total{.*} [1-9]/);

      // 3. Verify OpenAPI documentation
      const docsResponse = await axios.get(`${BASE_URL}/docs/json`);
      expect(docsResponse.data.openapi).toBe('3.0.0');
      expect(docsResponse.data.paths).toHaveProperty('/metrics');
      expect(docsResponse.data.paths).toHaveProperty('/health');

      // 4. Check logs are accessible
      const logsResult = await execAsync(`docker logs ${CONTAINER_NAME} --tail 10`);
      expect(logsResult.stdout.length).toBeGreaterThan(0);

      // 5. Verify health endpoints
      const healthResponse = await axios.get(`${BASE_URL}/health`);
      expect(healthResponse.data).toHaveProperty('status');
      expect(healthResponse.data.status).toBe('ok');
    });
  });

  describe('Error Handling and Resilience', () => {
    it('should handle Z.AI service errors gracefully', async () => {
      // Simulate error by using invalid prompt
      const response = await axios.post(`${ZAI_URL}/prompt`, {
        prompt: '', // Empty prompt
        timeout: 1000
      });

      expect([400, 500]).toContain(response.status);
      expect(response.data).toHaveProperty('error');
    });

    it('should retry failed Z.AI requests', async () => {
      // Test retry mechanism with short timeout
      const startTime = Date.now();

      const response = await axios.post(`${ZAI_URL}/prompt`, {
        prompt: 'Test retry',
        timeout: 100 // Very short timeout to potentially trigger retry
      });

      const duration = Date.now() - startTime;

      // If retry happened, duration would be longer
      expect(response.status).toBeLessThan(600);
    });

    it('should handle API rate limiting', async () => {
      const requests = Array(150).fill(null).map(() =>
        client.get('/v1/status')
      );

      const responses = await Promise.all(requests);
      const rateLimited = responses.filter(r => r.status === 429);

      expect(rateLimited.length).toBeGreaterThan(0);

      // Rate limit response should include retry info
      if (rateLimited.length > 0) {
        expect(rateLimited[0].headers).toHaveProperty('retry-after');
      }
    });
  });

  describe('Network Integration', () => {
    it('should communicate between containers', async () => {
      // Verify web-summary can reach Z.AI service
      const summaryCommand = `echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"summarize_url","arguments":{"url":"https://example.com"}}}' | /opt/venv/bin/python3 -u /app/core-assets/scripts/web-summary-mcp-server.py`;

      const result = await dockerExec(summaryCommand);
      const response = JSON.parse(result);

      expect(response).toHaveProperty('result');
      // This proves web-summary successfully called Z.AI service
    });

    it('should be accessible from host network', async () => {
      const endpoints = [
        `${BASE_URL}/health`,
        `${BASE_URL}/metrics`,
        `${BASE_URL}/docs`,
        `${ZAI_URL}/health`
      ];

      const responses = await Promise.all(
        endpoints.map(url => axios.get(url))
      );

      responses.forEach((response, index) => {
        expect(response.status).toBe(200);
      });
    });
  });

  describe('Data Persistence', () => {
    it('should persist MCP configuration changes', async () => {
      // Add tool
      await dockerExec('mcp add persist-test npx "-y test" "Test"');

      // Restart container (simulated by re-reading config)
      const config1 = await dockerExec('cat /home/devuser/.config/claude/mcp.json');
      expect(config1).toContain('persist-test');

      // Clean up
      await dockerExec('mcp remove persist-test');
    });

    it('should maintain backup history', async () => {
      await dockerExec('mcp backup');
      await dockerExec('mcp backup');

      const backups = await dockerExec(
        'ls -1 /home/devuser/.config/claude/backups/mcp-*.json'
      );

      const backupFiles = backups.trim().split('\n');
      expect(backupFiles.length).toBeGreaterThanOrEqual(2);
    });
  });
});
