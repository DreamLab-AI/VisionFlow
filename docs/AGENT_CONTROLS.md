# Agent Control Panel User Guide

## Overview

The Agent Control Panel provides comprehensive orchestration and management of AI agents within the system. Access it through **Settings → Agents** tab in the Control Center.

## Features

### 1. Agent Spawner

Quickly spawn different types of AI agents with pre-configured settings:

- **Researcher** 🔍 - Information gathering and analysis
- **Coder** 💻 - Code generation and implementation
- **Analyzer** 📊 - Data analysis and insights
- **Tester** 🧪 - Test creation and validation
- **Optimizer** ⚡ - Performance optimization
- **Coordinator** 🎯 - Multi-agent coordination

**Usage:**
1. Click on any agent type button to spawn
2. Agent spawns with default settings (priority, strategy, provider)
3. Monitor spawn status in real-time
4. Max concurrent agents enforced automatically

### 2. Active Agents Monitor

Real-time view of all active agents with status indicators:

- **Status Icons:**
  - ✅ Green Check: Agent active and healthy
  - 🟠 Orange Pulse: Agent busy processing
  - ❌ Red X: Agent error state
  - ⚠️ Gray: Agent idle

- **Agent Information:**
  - Agent ID (truncated for readability)
  - Agent type
  - Health percentage
  - Tasks completed count
  - Uptime duration

### 3. Agent Settings (20+ Configuration Options)

#### Spawning Settings
Configure how agents are created and scaled:

| Setting | Type | Range | Description |
|---------|------|-------|-------------|
| Auto-Scale | Toggle | On/Off | Automatically spawn agents based on workload |
| Max Concurrent Agents | Number | 1-50 | Maximum number of agents running simultaneously |
| AI Provider | Select | gemini/openai/claude | AI model provider for agent intelligence |
| Default Priority | Select | low/medium/high/critical | Default task priority for new agents |
| Default Strategy | Select | parallel/sequential/adaptive | Default execution strategy |

**Best Practices:**
- Start with 5-10 concurrent agents for most workloads
- Use `adaptive` strategy for dynamic task handling
- Enable auto-scale for variable workloads
- Set priority based on task urgency

#### Lifecycle Settings
Control agent lifespan and health:

| Setting | Type | Range | Description |
|---------|------|-------|-------------|
| Idle Timeout | Number | 60-600s | Auto-terminate idle agents after this duration |
| Auto-Restart Failed | Toggle | On/Off | Automatically restart agents that encounter errors |
| Health Check Interval | Number | 10-120s | Frequency of agent health monitoring |

**Best Practices:**
- Set idle timeout to 300s (5 min) for balanced resource usage
- Enable auto-restart for production environments
- Use 30s health checks for critical operations

#### Monitoring Settings
Configure telemetry and logging:

| Setting | Type | Range | Description |
|---------|------|-------|-------------|
| Enable Telemetry | Toggle | On/Off | Real-time agent telemetry streaming |
| Poll Interval | Number | 1-30s | Telemetry update frequency |
| Log Level | Select | debug/info/warn/error | Agent logging verbosity |

**Best Practices:**
- Enable telemetry for development and debugging
- Use 5s poll interval for real-time monitoring
- Set log level to `info` for production
- Use `debug` for troubleshooting

#### Visualization Settings
Control how agents appear in the main graph:

| Setting | Type | Range | Description |
|---------|------|-------|-------------|
| Show in Main Graph | Toggle | On/Off | Display agents as nodes in visualization |
| Agent Node Size | Slider | 0.5-3.0 | Visual size of agent nodes |
| Agent Node Color | Color | Any hex | Color of agent nodes in graph |

**Best Practices:**
- Enable graph visualization for small agent counts (<20)
- Use distinct colors to differentiate agent types
- Size 1.0-1.5 for balanced visibility
- Disable for large swarms to improve performance

### 4. Agent Telemetry Stream

Real-time telemetry with dual display modes:

#### Telemetry Mode
- **7-Segment DSEG Font Display**
- Black text on orange background
- Timestamp, Agent ID, Status metrics
- Auto-scrolling with 50-message buffer
- Color-coded status levels:
  - Red: Errors
  - Orange: Warnings
  - Green: Success
  - Black: Info

#### GOAP Mode (Goal-Oriented Action Planning)
- Interactive AI planning widget
- A* pathfinding visualization
- Dynamic agent coordination
- Research goal tracking
- Double-click GOAP button to open full interface

**Usage:**
1. Switch between TELEMETRY and GOAP tabs
2. Monitor agent status in real-time
3. Identify performance bottlenecks
4. Track task completion
5. Debug agent behavior

## Common Workflows

### Starting a New Swarm
1. Configure max concurrent agents (5-10 recommended)
2. Select AI provider based on task requirements
3. Enable auto-scale if workload is variable
4. Spawn initial agents using spawner buttons
5. Monitor telemetry for health and performance

### Debugging Agent Issues
1. Set log level to `debug`
2. Reduce poll interval to 1-2s
3. Monitor telemetry stream for errors
4. Check agent health percentages
5. Review failed task counts
6. Enable auto-restart if issues persist

### Performance Optimization
1. Monitor active agent count vs. workload
2. Adjust max concurrent agents based on system resources
3. Use `adaptive` strategy for dynamic load balancing
4. Increase idle timeout if agents restart frequently
5. Disable graph visualization for large swarms (>20 agents)

### Production Deployment
1. Set appropriate max concurrent agents
2. Enable auto-scale for variable loads
3. Set idle timeout to 300-600s
4. Enable auto-restart for resilience
5. Set log level to `info` or `warn`
6. Use 5-10s poll interval
7. Monitor health metrics regularly

## Agent Types & Use Cases

### Researcher 🔍
- **Use Case:** Information gathering, research, analysis
- **Recommended Provider:** Gemini (fast, cost-effective)
- **Suggested Priority:** Medium-High
- **Typical Workload:** Parallel searches, data collection

### Coder 💻
- **Use Case:** Code generation, implementation, refactoring
- **Recommended Provider:** Claude (high quality code)
- **Suggested Priority:** High
- **Typical Workload:** Sequential complex tasks

### Analyzer 📊
- **Use Case:** Data analysis, pattern recognition, insights
- **Recommended Provider:** Gemini (strong analytics)
- **Suggested Priority:** Medium
- **Typical Workload:** Batch processing, parallel analysis

### Tester 🧪
- **Use Case:** Test generation, validation, QA
- **Recommended Provider:** OpenAI (thorough testing)
- **Suggested Priority:** High
- **Typical Workload:** Sequential test execution

### Optimizer ⚡
- **Use Case:** Performance tuning, efficiency improvements
- **Recommended Provider:** Claude (optimization expertise)
- **Suggested Priority:** Medium
- **Typical Workload:** Iterative optimization cycles

### Coordinator 🎯
- **Use Case:** Multi-agent orchestration, task distribution
- **Recommended Provider:** Gemini (fast coordination)
- **Suggested Priority:** Critical
- **Typical Workload:** Continuous coordination

## Settings Persistence

All agent settings are automatically saved to:
- **Local State:** Immediate updates via Zustand store
- **Server State:** Batched updates via REST API
- **Persistence:** Settings survive page refreshes

**Path Structure:**
```
settings.agents.spawn.*
settings.agents.lifecycle.*
settings.agents.monitoring.*
settings.agents.visualization.*
```

## API Integration

The Agent Control Panel integrates with the following endpoints:

### Spawn Agent
```bash
POST /api/bots/spawn-agent-hybrid
{
  "agent_type": "researcher",
  "swarm_id": "main-swarm",
  "method": "mcp-fallback",
  "priority": "medium",
  "strategy": "adaptive",
  "config": { ... }
}
```

### List Agents
```bash
GET /api/bots/agents
```

### Agent Telemetry
```bash
GET /api/bots/agents
# Polls every N seconds based on telemetry_poll_interval
```

## Troubleshooting

### Agents Won't Spawn
- **Check:** Max concurrent agents limit reached
- **Solution:** Terminate idle agents or increase limit

### High CPU Usage
- **Check:** Too many concurrent agents
- **Solution:** Reduce max concurrent agents
- **Check:** Poll interval too aggressive
- **Solution:** Increase to 10-15s

### Telemetry Not Updating
- **Check:** Telemetry enabled in settings
- **Solution:** Toggle on in monitoring settings
- **Check:** Network connectivity
- **Solution:** Check browser console for errors

### Agents Keep Failing
- **Check:** Health check interval too aggressive
- **Solution:** Increase to 30-60s
- **Check:** Idle timeout too short
- **Solution:** Increase to 300-600s
- **Solution:** Enable auto-restart

### Settings Not Saving
- **Check:** Browser local storage
- **Solution:** Clear cache and reload
- **Check:** Server connectivity
- **Solution:** Verify API endpoint accessible

## Performance Tips

1. **Optimal Agent Count:** 5-10 for most workloads
2. **Poll Interval:** 5-10s balances real-time vs. performance
3. **Graph Visualization:** Disable for >20 agents
4. **Log Level:** Use `info` in production, `debug` only when needed
5. **Auto-Scale:** Enable for variable workloads
6. **Health Checks:** 30s interval sufficient for most cases

## Advanced Features

### Custom Agent Configuration
Settings can be customized per agent type through the API:
```typescript
{
  agent_type: "custom-agent",
  config: {
    memory_limit: "4GB",
    timeout: 300,
    custom_params: { ... }
  }
}
```

### Swarm Coordination
Agents automatically coordinate through:
- Shared memory (MCP memory tools)
- Message passing (WebSocket)
- Task queues (Priority-based)
- Resource pooling (Auto-scale)

### Integration with Other Systems
The agent panel integrates with:
- **Control Center:** Centralized settings management
- **Graph Visualization:** Real-time agent nodes
- **Telemetry Stream:** Live monitoring
- **GOAP Widget:** Goal planning
- **WebSocket:** Real-time updates

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+A` | Focus agent spawner |
| `Ctrl+T` | Switch to telemetry |
| `Ctrl+G` | Switch to GOAP |
| `Ctrl+R` | Refresh agent list |

## Version History

- **v1.0.0:** Initial release with 20+ settings
- **Phase 3:** Comprehensive agent orchestration UI
- **Integration:** Settings Panel Redesign

## Related Documentation

- [Control Center Guide](./CONTROL_CENTER.md)
- [Settings Management](./SETTINGS.md)
- [Agent API Reference](./API_REFERENCE.md)
- [GOAP Widget](https://goal.ruv.io/)

## Support

For issues or questions:
1. Check telemetry for error messages
2. Review settings configuration
3. Verify API connectivity
4. Check browser console logs
5. Open GitHub issue with details
