# Getting Started with Agentic Flow

---
**Version:** 1.0.0
**Last Updated:** 2025-10-12
**Status:** Active
**Category:** Guide
**Tags:** [quickstart, installation, tutorial]
---

## Overview

This guide will help you get Agentic Flow up and running in under 10 minutes. By the end, you'll have a fully functional AI agent orchestration system with multi-model routing and observability.

## Prerequisites

### System Requirements
- **Node.js:** v20.0.0 or higher
- **RAM:** 4GB minimum, 8GB recommended
- **Disk Space:** 2GB for installation
- **OS:** Linux, macOS, or Windows (WSL2)

### Optional Requirements
- **Docker:** v20.10+ (for containerised deployment)
- **Git:** v2.30+ (for source installation)

### API Keys

You'll need at least one of the following:
- **Anthropic API Key** (Claude models)
- **OpenRouter API Key** (100+ models)
- **Google AI API Key** (Gemini models)
- **OpenAI API Key** (GPT models)

Get your API keys:
- Anthropic: [https://console.anthropic.com/](https://console.anthropic.com/)
- OpenRouter: [https://openrouter.ai/keys](https://openrouter.ai/keys)
- Google AI: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
- OpenAI: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

---

## Installation Methods

### Method 1: NPM Installation (Recommended)

```bash
# Install globally
npm install -g agentic-flow

# Or install locally in your project
npm install agentic-flow

# Verify installation
agentic-flow --version
```

### Method 2: Source Installation

```bash
# Clone the repository
git clone https://github.com/ruvnet/agentic-flow.git
cd agentic-flow

# Install dependencies
npm install

# Build the project
npm run build

# Link for global usage
npm link
```

### Method 3: Docker Deployment

```bash
# Clone the repository
git clone https://github.com/ruvnet/agentic-flow.git
cd agentic-flow/docker/cachyos

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start services
./start-agentic-flow.sh

# Verify containers
docker ps
```

---

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# Core Configuration
NODE_ENV=production
PORT=9090

# API Keys
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
GOOGLE_API_KEY=your_google_key_here
OPENAI_API_KEY=your_openai_key_here

# Z.AI Configuration (optional)
ZAI_API_KEY=your_zai_key_here
ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic

# Router Configuration
DEFAULT_MODEL=claude-sonnet-4
ROUTING_POLICY=cost-optimised
ENABLE_FALLBACK=true

# Management API
MANAGEMENT_API_KEY=your_secure_api_key_here

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=info
```

### Configuration File

Alternatively, create `config.json`:

```json
{
  "router": {
    "policy": "cost-optimised",
    "fallbacks": ["claude", "gemini", "openrouter"],
    "timeout": 30000,
    "retries": 3
  },
  "models": {
    "default": "claude-sonnet-4",
    "fallback": "gemini-1.5-pro"
  },
  "monitoring": {
    "enabled": true,
    "metrics": {
      "port": 9090,
      "path": "/metrics"
    },
    "logging": {
      "level": "info",
      "format": "json"
    }
  },
  "agents": {
    "poolSize": 4,
    "maxQueue": 50
  }
}
```

---

## First Run

### Start the Server

```bash
# Using NPM package
agentic-flow start

# Using source
npm start

# Using Docker
./start-agentic-flow.sh
```

### Verify Installation

```bash
# Check health endpoint
curl http://localhost:9090/health

# Expected response:
{
  "status": "ok",
  "service": "agentic-flow",
  "version": "1.3.0",
  "uptime": 42.5,
  "memory": {
    "heapUsed": 52428800,
    "heapTotal": 104857600
  }
}
```

### Access API Documentation

Open your browser to:
```
http://localhost:9090/docs
```

You'll see the interactive Swagger UI with all available endpoints.

---

## Basic Usage

### Example 1: Simple Completion

```javascript
const { AgenticFlow } = require('agentic-flow');

const client = new AgenticFlow({
  apiKey: process.env.ANTHROPIC_API_KEY
});

const response = await client.complete({
  prompt: 'Explain quantum computing in simple terms',
  model: 'claude-sonnet-4',
  maxTokens: 500
});

console.log(response.content);
```

### Example 2: Multi-Model Routing

```javascript
const { MultiModelRouter } = require('agentic-flow');

const router = new MultiModelRouter({
  policy: 'cost-optimised',
  fallbacks: ['claude', 'gemini', 'openai']
});

const response = await router.route({
  prompt: 'Write a Python function to calculate Fibonacci numbers',
  task: 'code-generation',
  budget: 0.01 // USD
});

console.log(`Model used: ${response.model}`);
console.log(`Cost: $${response.cost.toFixed(4)}`);
console.log(`Response: ${response.content}`);
```

### Example 3: Agent Execution

```javascript
const { AgentSystem } = require('agentic-flow');

const agents = new AgentSystem();

const result = await agents.execute({
  agent: 'coder',
  task: 'refactor-code',
  input: './src/legacy.js',
  output: './src/modern.js',
  options: {
    style: 'functional',
    tests: true
  }
});

console.log(`Refactored: ${result.success}`);
console.log(`Changes: ${result.changes.length}`);
```

---

## Testing Your Installation

### Run System Tests

```bash
# Run all tests
npm test

# Run specific test suites
npm run test:unit
npm run test:integration
npm run test:e2e

# With coverage
npm run test:coverage
```

### Performance Benchmark

```bash
# Run benchmarks
npm run benchmark

# Expected output:
📊 Benchmark: Health Check Latency
  Mean:    12.34ms
  P95:     18.90ms
  P99:     32.11ms
  Ops/sec: 81.04
```

### Load Testing

```bash
# Install Artillery (if not already)
npm install -g artillery

# Run load tests
cd docker/cachyos/tests
artillery run load-tests/api-load-test.yml
```

---

## Next Steps

### 1. Configure Multi-Model Routing
Learn how to optimise costs and performance:
→ [Multi-Model Routing Guide](architecture/multi-model-routing.md)

### 2. Explore Agent System
Discover 66+ specialised agents:
→ [Agent System Overview](agents/overview.md)

### 3. Set Up Monitoring
Implement observability:
→ [Monitoring Guide](operations/monitoring.md)

### 4. Deploy to Production
Production deployment strategies:
→ [Deployment Guide](guides/deployment.md)

### 5. Integrate MCP Servers
Extend functionality with custom tools:
→ [MCP Integration Guide](integrations/mcp-servers.md)

---

## Common Issues

### Issue: Port Already in Use

```bash
# Error: EADDRINUSE: address already in use :::9090

# Solution: Change port in .env
PORT=9091

# Or kill process using port 9090
lsof -ti:9090 | xargs kill -9
```

### Issue: API Key Not Found

```bash
# Error: ANTHROPIC_API_KEY is required

# Solution: Ensure .env file exists and is loaded
cat .env | grep ANTHROPIC_API_KEY

# Verify environment variables
node -e "console.log(process.env.ANTHROPIC_API_KEY)"
```

### Issue: Docker Container Won't Start

```bash
# Check container logs
docker logs agentic-flow-cachyos

# Common fix: Remove old containers
docker-compose down -v
docker system prune -f

# Rebuild and restart
./start-agentic-flow.sh --clean
```

### Issue: Connection Timeout

```bash
# Check service health
curl -v http://localhost:9090/health

# Verify services are running
docker ps
ps aux | grep node

# Check firewall rules
sudo ufw status
```

---

## Development Mode

### Hot Reload

```bash
# Start with nodemon
npm run dev

# Or using Docker
./start-agentic-flow.sh --dev
```

### Debug Mode

```bash
# Enable debug logging
DEBUG=* npm start

# Or with specific namespace
DEBUG=agentic-flow:* npm start

# Using VS Code debugger
code .
# Press F5 to start debugging
```

---

## Resources

### Documentation
- [Architecture Overview](ARCHITECTURE.md)
- [API Reference](api/README.md)
- [Agent Guide](agents/README.md)
- [Command Reference](commands/README.md)

### Examples
- [Code Examples](examples/README.md)
- [Integration Patterns](integrations/README.md)
- [Deployment Scenarios](guides/deployment.md)

### Support
- [GitHub Issues](https://github.com/ruvnet/agentic-flow/issues)
- [Discussions](https://github.com/ruvnet/agentic-flow/discussions)
- [Discord Community](https://discord.gg/agentic-flow)

---

## Feedback

Found an issue with this guide? Please let us know:
- [Report Documentation Issue](https://github.com/ruvnet/agentic-flow/issues/new?labels=documentation)
- [Suggest Improvements](https://github.com/ruvnet/agentic-flow/discussions/new)

---

**Next:** [Configuration Guide](guides/configuration.md) | [Architecture Overview](ARCHITECTURE.md)
