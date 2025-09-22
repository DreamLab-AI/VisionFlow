# AI Agents - Claude-Flow v110

The Multi-Agent Docker environment includes Claude-Flow v110, providing advanced AI agent orchestration with Goal Planner and Neural agents.

## Overview

Claude-Flow enables intelligent task planning and execution through two main agent types:

### 🎯 Goal Planner Agent
- Uses Goal-Oriented Action Planning (GOAP) with A* pathfinding
- Dynamically calculates optimal action sequences
- Adapts when conditions change or steps fail
- Perfect for complex workflows and deployments

### 🧠 SAFLA Neural Agent
- Self-Aware Four-Layer Architecture
- Vector, episodic, semantic, and working memory
- Learns from interactions and improves over time
- Enables distributed swarm intelligence

## Getting Started

### Initial Setup

```bash
# Inside the container, initialize agents
claude-flow-init-agents

# Or manually:
npx claude-flow@alpha goal init
npx claude-flow@alpha neural init
```

### Verify Installation

```bash
# Check Claude-Flow version
npx claude-flow@alpha --version

# Check agent status
cf-status

# View agent logs
cf-logs
```

## Using Goal Planner

### Creating Goals

```bash
# CLI usage
npx claude-flow@alpha goal create --name "Deploy application" --description "Deploy to production"

# Through Claude Code
claude
> Use mcp__claude-flow__goal_create with name "Deploy application"
```

### Goal Components

Goals consist of:
- **Preconditions**: What must be true before starting
- **Effects**: What will be true after completion
- **Actions**: Steps to achieve the goal
- **Cost**: Resource estimation

### Example Workflow

```javascript
// Goal: Set up development environment
{
  "name": "setup-dev-env",
  "preconditions": {
    "docker_installed": true,
    "workspace_exists": true
  },
  "effects": {
    "environment_ready": true,
    "dependencies_installed": true
  },
  "actions": [
    "check_requirements",
    "install_dependencies",
    "configure_services",
    "run_tests"
  ]
}
```

## Using Neural Agent

### Training the Agent

```bash
# Train on current codebase
npx claude-flow@alpha neural train --path ./src

# Learn from interactions
npx claude-flow@alpha neural learn --session last
```

### Memory Layers

1. **Vector Memory**: Fast similarity search
2. **Episodic Memory**: Interaction history
3. **Semantic Memory**: Conceptual understanding
4. **Working Memory**: Current context

### Querying Knowledge

```bash
# Ask about learned patterns
npx claude-flow@alpha neural query "How do we handle authentication?"

# Get recommendations
npx claude-flow@alpha neural suggest --context "building API endpoint"
```

## Dual Access Architecture

Claude-Flow is accessible through multiple channels:

### 1. Local MCP (Inside Container)
- Direct access via Claude Code
- Shared database at `/workspace/.swarm/memory.db`
- Best for interactive development

### 2. TCP Port 9500 (Shared Instance)
- External access to same instance
- Maintains state with local operations
- Good for CI/CD integration

### 3. TCP Port 9502 (Isolated Sessions)
- Fresh instance per connection
- Complete isolation between sessions
- Ideal for testing and multi-tenant scenarios

## Advanced Features

### Swarm Coordination

```bash
# Create agent swarm
npx claude-flow@alpha swarm create --name "dev-team"

# Add specialized agents
npx claude-flow@alpha swarm add-agent --type "coder" --name "backend-dev"
npx claude-flow@alpha swarm add-agent --type "tester" --name "qa-bot"
```

### Task Distribution

```bash
# Distribute task to swarm
npx claude-flow@alpha swarm assign --task "implement-feature" --strategy "parallel"

# Monitor progress
npx claude-flow@alpha swarm status
```

### Memory Persistence

All agent memories are stored in:
- `/workspace/.swarm/memory.db` - Shared database
- `/workspace/.swarm/sessions/` - Isolated session data
- `/workspace/.hive-mind/` - Swarm configuration

## Monitoring and Debugging

### Check Agent Status

```bash
# View all agents
cf-status

# Check specific agent
npx claude-flow@alpha goal status
npx claude-flow@alpha neural status
```

### View Logs

```bash
# Real-time logs
cf-logs

# TCP server logs (includes agent activity)
mcp-tcp-logs

# Isolated session logs
cf-tcp-logs
```

### Test Connections

```bash
# Test shared TCP
echo '{"jsonrpc":"2.0","id":"1","method":"tools/list","params":{}}' | nc localhost 9500

# Test isolated TCP
cf-test-tcp
```

## Best Practices

1. **Initialize Once**: Agents persist across sessions
2. **Use Appropriate Access**: Local for dev, TCP for external
3. **Monitor Memory**: Check database size periodically
4. **Backup State**: Regular backups of `.swarm` directory

## Troubleshooting

### Agents Not Responding

```bash
# Restart MCP services
mcp-restart

# Reinitialize if needed
claude-flow-init-agents
```

### Memory Issues

```bash
# Check database size
du -h /workspace/.swarm/memory.db

# Compact database
npx claude-flow@alpha db compact
```

### Connection Problems

```bash
# Verify services
mcp-tcp-status
cf-tcp-status

# Check ports
ss -tlnp | grep -E '9500|9502'
```

## Integration Examples

### Python Client

```python
import socket
import json

# Connect to shared instance
sock = socket.create_connection(('localhost', 9500))

# Create goal
goal_request = {
    "jsonrpc": "2.0",
    "id": "1",
    "method": "tools/call",
    "params": {
        "name": "goal_create",
        "arguments": {
            "name": "Process data",
            "description": "ETL pipeline"
        }
    }
}

sock.send(json.dumps(goal_request).encode() + b'\n')
response = sock.recv(4096)
```

### Node.js Client

```javascript
const net = require('net');

// Connect to isolated instance
const client = net.createConnection({ port: 9502 });

client.on('connect', () => {
    const request = {
        jsonrpc: "2.0",
        id: "1",
        method: "initialize",
        params: { protocolVersion: "2024-11-05" }
    };
    client.write(JSON.stringify(request) + '\n');
});
```

## Next Steps

1. Explore goal templates in `/workspace/.swarm/templates/`
2. Train neural agent on your codebase
3. Create custom agent types
4. Build automated workflows with goals