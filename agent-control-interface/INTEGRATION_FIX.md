# Integration Fix for VisionFlow Connection

## Problem
The VisionFlow backend is not connecting to our Agent Control Interface because the `AGENT_CONTROL_URL` environment variable is not set in the docker-compose configuration.

## Solution

Add the following environment variable to your docker-compose.dev.yml:

```yaml
environment:
  # ... existing environment variables ...
  
  # Agent Control System connection
  - AGENT_CONTROL_URL=multi-agent-container:9500
  # or use IP directly:
  # - AGENT_CONTROL_URL=172.18.0.10:9500
```

## Quick Fix Commands

From the VisionFlow container, run:

```bash
# Set the environment variable
export AGENT_CONTROL_URL=multi-agent-container:9500

# Or use the IP directly
export AGENT_CONTROL_URL=172.18.0.10:9500

# Restart your Rust backend for changes to take effect
```

## Verification

After setting the environment variable and restarting, you should see in the logs:

```
[AppState::new] Starting AgentControlActor
[AppState::new] Agent Control URL: multi-agent-container:9500
```

Instead of:
```
[AppState::new] AgentControlActor not configured (AGENT_CONTROL_URL not set)
```

## Testing the Connection

Once configured, test from the VisionFlow container:

```bash
# Test initialization endpoint
curl -X POST http://localhost:4000/api/bots/swarm/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "topology": "hierarchical",
    "agentTypes": ["coordinator", "coder", "tester"]
  }'
```

## API Endpoints Available

- `POST /api/bots/swarm/initialize` - Initialize a swarm
- `GET /api/bots/agents` - Get all agents
- `GET /api/bots/visualization` - Get visualization snapshot