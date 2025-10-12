/**
 * Integration tests for MCP tools
 */

const { describe, it, expect } = require('@jest/globals');
const { exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;

const execAsync = promisify(exec);

const CONTAINER_NAME = process.env.CONTAINER_NAME || 'agentic-flow-cachyos';

async function dockerExec(command) {
  const { stdout, stderr } = await execAsync(
    `docker exec ${CONTAINER_NAME} ${command}`
  );
  return { stdout, stderr };
}

describe('MCP Tools Integration Tests', () => {
  describe('MCP CLI', () => {
    it('should list all configured tools', async () => {
      const { stdout } = await dockerExec('mcp list');

      expect(stdout).toContain('claude-flow');
      expect(stdout).toContain('web-summary');
      expect(stdout).toContain('playwright');
    });

    it('should show tool details', async () => {
      const { stdout } = await dockerExec('mcp show web-summary');

      expect(stdout).toContain('web-summary');
      expect(stdout).toContain('command');
      expect(stdout).toContain('type');
    });

    it('should validate configuration', async () => {
      const { stdout } = await dockerExec('mcp validate');

      expect(stdout).toContain('valid');
    });

    it('should backup configuration', async () => {
      const { stdout } = await dockerExec('mcp backup');

      expect(stdout).toContain('Backup created');
    });

    it('should add new tool', async () => {
      const { stdout } = await dockerExec(
        'mcp add test-tool npx "-y test-package" "Test tool"'
      );

      expect(stdout).toContain('Added tool');

      // Clean up
      await dockerExec('mcp remove test-tool');
    });

    it('should remove tool', async () => {
      // Add test tool first
      await dockerExec('mcp add temp-tool npx "-y temp" "Temp"');

      const { stdout } = await dockerExec('mcp remove temp-tool');

      expect(stdout).toContain('Removed tool');
    });

    it('should update tool configuration', async () => {
      const { stdout } = await dockerExec(
        'mcp update web-summary --description "Updated description"'
      );

      expect(stdout).toContain('Updated tool');
    });
  });

  describe('Web Summary Tool', () => {
    it('should summarize web pages', async () => {
      const command = `echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"summarize_url","arguments":{"url":"https://example.com"}}}' | /opt/venv/bin/python3 -u /app/core-assets/scripts/web-summary-mcp-server.py`;

      const { stdout } = await dockerExec(command);

      const response = JSON.parse(stdout);
      expect(response).toHaveProperty('result');
      expect(response.result).toHaveProperty('summary');
      expect(response.result).toHaveProperty('formatted');
    });

    it('should handle YouTube URLs', async () => {
      const command = `echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"summarize_url","arguments":{"url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}}}' | /opt/venv/bin/python3 -u /app/core-assets/scripts/web-summary-mcp-server.py`;

      const { stdout } = await dockerExec(command);

      const response = JSON.parse(stdout);
      expect(response).toHaveProperty('result');
    });

    it('should include semantic topics', async () => {
      const command = `echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"summarize_url","arguments":{"url":"https://www.anthropic.com"}}}' | /opt/venv/bin/python3 -u /app/core-assets/scripts/web-summary-mcp-server.py`;

      const { stdout } = await dockerExec(command);

      const response = JSON.parse(stdout);
      expect(response.result).toHaveProperty('topics');
      expect(Array.isArray(response.result.topics)).toBe(true);
    });

    it('should handle invalid URLs gracefully', async () => {
      const command = `echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"summarize_url","arguments":{"url":"not-a-valid-url"}}}' | /opt/venv/bin/python3 -u /app/core-assets/scripts/web-summary-mcp-server.py`;

      const { stdout } = await dockerExec(command);

      const response = JSON.parse(stdout);
      expect(response).toHaveProperty('error');
    });
  });

  describe('MCP Configuration', () => {
    it('should have valid JSON configuration', async () => {
      const { stdout } = await dockerExec(
        'cat /home/devuser/.config/claude/mcp.json'
      );

      const config = JSON.parse(stdout);
      expect(config).toHaveProperty('mcpServers');
      expect(config).toHaveProperty('toolCategories');
    });

    it('should have all required tools configured', async () => {
      const { stdout } = await dockerExec(
        'cat /home/devuser/.config/claude/mcp.json'
      );

      const config = JSON.parse(stdout);
      const tools = Object.keys(config.mcpServers);

      expect(tools).toContain('claude-flow');
      expect(tools).toContain('web-summary');
      expect(tools).toContain('playwright');
      expect(tools).toContain('filesystem');
      expect(tools).toContain('git');
    });

    it('should have environment variables configured', async () => {
      const { stdout } = await dockerExec(
        'cat /home/devuser/.config/claude/mcp.json'
      );

      const config = JSON.parse(stdout);
      const webSummary = config.mcpServers['web-summary'];

      expect(webSummary.env).toHaveProperty('GOOGLE_API_KEY');
      expect(webSummary.env).toHaveProperty('ZAI_CONTAINER_URL');
    });
  });

  describe('Topics Database', () => {
    it('should have topics configuration', async () => {
      const { stdout } = await dockerExec(
        'cat /app/core-assets/config/topics.json'
      );

      const topics = JSON.parse(stdout);
      expect(Array.isArray(topics)).toBe(true);
      expect(topics.length).toBeGreaterThan(0);
    });

    it('should contain expected topics', async () => {
      const { stdout } = await dockerExec(
        'cat /app/core-assets/config/topics.json'
      );

      const topics = JSON.parse(stdout);
      expect(topics).toContain('Artificial Intelligence');
      expect(topics).toContain('Machine Learning');
    });
  });
});
