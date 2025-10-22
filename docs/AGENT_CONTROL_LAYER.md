# Agent Control Layer - Settings & Integration
**Date**: 2025-10-22
**Purpose**: User-controllable agent orchestration, not internal agent mechanics

---

## 🎯 Overview

This document defines the **user-facing control layer** for the agentic system (Claude with claude-flow orchestration). This is NOT about exposing internal agent container settings, but rather the **language-based interfaces** and **visualization data structures** that users interact with.

**Key Principle**: The agentic system (Claude/claude-flow) is intelligent enough to return structured data that the Rust app can display in graphs and panels.

---

## 📡 Current Agent Control APIs

### Existing Endpoints (`/api/bots/*`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/bots/agents` | GET | List all active agents with telemetry |
| `/api/bots/data` | GET | Get agent graph visualization data |
| `/api/bots/data` | POST | Update agent graph data |
| `/api/bots/status` | GET | Get connection status and health |
| `/api/bots/initialize-swarm` | POST | Start a new swarm coordinator |
| `/api/bots/spawn-agent-hybrid` | POST | Spawn a new agent with config |
| `/api/bots/remove-task/{id}` | DELETE | Cancel/remove an agent task |

###  Agent Data Structures

#### 1. **Agent Telemetry** (Polled every 5s by frontend)

```typescript
interface AgentTelemetry {
  id: string;                    // Agent ID
  type: string;                  // "researcher", "coder", "analyzer", etc.
  status: "active" | "idle" | "error" | "warning";
  health: number;                // 0-100%
  cpuUsage: number;              // Percentage
  memoryUsage: number;           // MB
  workload: number;              // Arbitrary workload metric
  currentTask?: string;          // Current task description
  position?: {x, y, z};          // For visualization (optional)
  metadata?: Record<string, any>; // Custom agent data
}
```

#### 2. **Agent Spawn Request**

```typescript
interface SpawnAgentRequest {
  agent_type: string;            // Agent role: "researcher", "coder", etc.
  swarm_id: string;              // Swarm to join
  method: "docker" | "mcp-fallback";
  priority?: "low" | "medium" | "high" | "critical";
  strategy?: "parallel" | "sequential" | "adaptive";
  config?: {
    auto_scale?: boolean;
    monitor?: boolean;
    max_workers?: number;
  };
}
```

#### 3. **Agent Graph Data** (For Visualization)

```typescript
interface AgentGraphData {
  nodes: Array<{
    id: string;
    label: string;
    type: string;
    position: {x, y, z};
    metadata: {
      status: string;
      health: number;
      taskCount: number;
    };
  }>;
  edges: Array<{
    source: string;
    target: string;
    type: "communication" | "coordination" | "dependency";
    weight?: number;
  }>;
}
```

---

## 🎛️ User-Controllable Settings (Agent Orchestration)

### 1. **Agent Spawning Controls**

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `agents.spawn.auto_scale` | boolean | true | Automatically spawn agents based on workload |
| `agents.spawn.max_concurrent` | number | 10 | Maximum concurrent agents |
| `agents.spawn.default_provider` | string | "gemini" | AI provider: "gemini", "openai", "claude" |
| `agents.spawn.default_strategy` | string | "adaptive" | "parallel", "sequential", "adaptive" |
| `agents.spawn.default_priority` | string | "medium" | "low", "medium", "high", "critical" |

### 2. **Agent Lifecycle Management**

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `agents.lifecycle.idle_timeout` | number | 300 | Seconds before idle agent cleanup |
| `agents.lifecycle.health_check_interval` | number | 5 | Seconds between health checks |
| `agents.lifecycle.auto_restart` | boolean | true | Restart failed agents automatically |
| `agents.lifecycle.max_retries` | number | 3 | Max restart attempts |
| `agents.lifecycle.graceful_shutdown_timeout` | number | 30 | Seconds to wait for graceful shutdown |

### 3. **Agent Monitoring Settings**

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `agents.monitoring.telemetry_enabled` | boolean | true | Enable telemetry collection |
| `agents.monitoring.telemetry_poll_interval` | number | 5 | Seconds between telemetry polls |
| `agents.monitoring.log_level` | string | "info" | "error", "warn", "info", "debug" |
| `agents.monitoring.track_performance` | boolean | true | Track CPU/memory usage |
| `agents.monitoring.alert_on_failure` | boolean | true | Show alerts for agent failures |

### 4. **Agent Visualization Settings**

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `agents.visualization.show_in_graph` | boolean | true | Display agents in main graph |
| `agents.visualization.node_size` | number | 1.5 | Agent node size multiplier |
| `agents.visualization.node_color` | string | "#ff8800" | Agent node color |
| `agents.visualization.show_connections` | boolean | true | Show agent-to-agent edges |
| `agents.visualization.connection_color` | string | "#fbbf24" | Connection edge color |
| `agents.visualization.animate_activity` | boolean | true | Pulse animation for active agents |

### 5. **Task Orchestration Settings**

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `agents.tasks.queue_size` | number | 100 | Max queued tasks |
| `agents.tasks.timeout` | number | 600 | Task timeout in seconds |
| `agents.tasks.retry_failed` | boolean | true | Retry failed tasks |
| `agents.tasks.priority_scheduling` | boolean | true | Use priority-based scheduling |

---

## 🔌 Agent Integration Points

### 1. **Telemetry Stream → Visualization**

**Data Flow**: Claude Agents → MCP TCP → Rust `/api/bots/agents` → Frontend Polling → Graph Display

**Structured Data Returned**:
```json
{
  "agents": [
    {
      "id": "agent-123",
      "type": "researcher",
      "status": "active",
      "health": 95,
      "cpuUsage": 45,
      "memoryUsage": 256,
      "workload": 7,
      "currentTask": "Analyzing codebase structure",
      "position": {"x": 10, "y": 5, "z": 0}
    }
  ],
  "count": 1
}
```

**Visualization Use**:
- Display as nodes in main knowledge graph
- Color-code by status (active=green, idle=yellow, error=red)
- Size nodes by workload
- Show connections between collaborating agents

### 2. **Task Commands → Agent System**

**Language-Based Control**:
Users submit natural language tasks that Claude/agents execute:

```typescript
// POST /api/bots/spawn-agent-hybrid
{
  "agent_type": "researcher",
  "swarm_id": "main-swarm",
  "method": "mcp-fallback",
  "priority": "high",
  "config": {
    "auto_scale": true,
    "monitor": true
  }
}
```

**Agent System Response**:
```json
{
  "success": true,
  "swarm_id": "main-swarm",
  "method_used": "mcp-fallback",
  "message": "Agent spawned successfully"
}
```

### 3. **Agent Lifecycle Control**

**Control Commands** (via REST API):

| Command | Endpoint | Description |
|---------|----------|-------------|
| **Spawn** | `POST /api/bots/spawn-agent-hybrid` | Create new agent |
| **List** | `GET /api/bots/agents` | Get all active agents |
| **Status** | `GET /api/bots/status` | Check health/connection |
| **Stop** | `DELETE /api/bots/remove-task/{id}` | Cancel agent task |
| **Initialize Swarm** | `POST /api/bots/initialize-swarm` | Start coordination |

---

## 🎨 UI Control Panel Design

### Agent Control Panel Sections

#### 1. **Agent Dashboard** (Monitor Tab)
- Grid of active agents with status
- Real-time telemetry display (DSEG7 font)
- Health indicators (traffic light: green/yellow/red)
- Current task descriptions
- CPU/Memory usage bars

#### 2. **Agent Spawner** (Deploy Tab)
- Agent type selector (dropdown)
- Priority slider (low → critical)
- Strategy selector (parallel/sequential/adaptive)
- Auto-scale toggle
- "Spawn Agent" button

#### 3. **Task Queue** (Tasks Tab)
- List of pending/active/completed tasks
- Task progress bars
- Cancel/retry buttons
- Filter by status/priority

#### 4. **Swarm Coordinator** (Swarms Tab)
- Active swarms list
- Swarm topology visualization
- "Initialize New Swarm" button
- Swarm health metrics

#### 5. **Agent Graph** (Visualization Tab)
- Agents displayed as nodes in main graph
- Toggle: "Show Agents in Main Graph"
- Color settings for agent nodes
- Connection visibility toggle
- Animation settings

---

## 🔧 Settings Integration Examples

### Example 1: Enable Agent Visualization

```typescript
// User toggles setting in Control Panel
settingsStore.set('agents.visualization.show_in_graph', true);

// Frontend adds agents to Three.js scene
agents.forEach(agent => {
  scene.add(createAgentNode(agent, {
    size: settings.agents.visualization.node_size,
    color: settings.agents.visualization.node_color,
    animate: settings.agents.visualization.animate_activity
  }));
});
```

### Example 2: Adjust Telemetry Polling

```typescript
// User changes poll interval
settingsStore.set('agents.monitoring.telemetry_poll_interval', 10); // 10 seconds

// Frontend updates polling interval
if (pollIntervalRef.current) {
  clearInterval(pollIntervalRef.current);
}
pollIntervalRef.current = setInterval(
  pollTelemetry,
  settings.agents.monitoring.telemetry_poll_interval * 1000
);
```

### Example 3: Spawn Agent with Custom Config

```typescript
// User clicks "Spawn Agent" with settings
const request = {
  agent_type: settings.agents.spawn.default_agent_type || 'researcher',
  swarm_id: 'main-swarm',
  method: 'mcp-fallback',
  priority: settings.agents.spawn.default_priority,
  strategy: settings.agents.spawn.default_strategy,
  config: {
    auto_scale: settings.agents.spawn.auto_scale,
    monitor: settings.agents.monitoring.telemetry_enabled,
    max_workers: settings.agents.spawn.max_concurrent
  }
};

await fetch('/api/bots/spawn-agent-hybrid', {
  method: 'POST',
  body: JSON.stringify(request)
});
```

---

## 📋 Settings Database Schema (Agent Settings Only)

```sql
-- Agent-specific settings in unified settings database
INSERT INTO settings (key, value, value_type, category, priority, user_visible) VALUES
  -- Spawning
  ('agents.spawn.auto_scale', 'true', 'boolean', 'agents', 'medium', 1),
  ('agents.spawn.max_concurrent', '10', 'number', 'agents', 'high', 1),
  ('agents.spawn.default_provider', '"gemini"', 'string', 'agents', 'high', 1),
  ('agents.spawn.default_strategy', '"adaptive"', 'string', 'agents', 'medium', 1),
  ('agents.spawn.default_priority', '"medium"', 'string', 'agents', 'medium', 1),

  -- Lifecycle
  ('agents.lifecycle.idle_timeout', '300', 'number', 'agents', 'medium', 1),
  ('agents.lifecycle.health_check_interval', '5', 'number', 'agents', 'low', 1),
  ('agents.lifecycle.auto_restart', 'true', 'boolean', 'agents', 'high', 1),
  ('agents.lifecycle.max_retries', '3', 'number', 'agents', 'medium', 1),

  -- Monitoring
  ('agents.monitoring.telemetry_enabled', 'true', 'boolean', 'agents', 'medium', 1),
  ('agents.monitoring.telemetry_poll_interval', '5', 'number', 'agents', 'medium', 1),
  ('agents.monitoring.log_level', '"info"', 'string', 'agents', 'low', 1),
  ('agents.monitoring.track_performance', 'true', 'boolean', 'agents', 'low', 1),
  ('agents.monitoring.alert_on_failure', 'true', 'boolean', 'agents', 'high', 1),

  -- Visualization
  ('agents.visualization.show_in_graph', 'true', 'boolean', 'agents', 'medium', 1),
  ('agents.visualization.node_size', '1.5', 'number', 'agents', 'low', 1),
  ('agents.visualization.node_color', '"#ff8800"', 'string', 'agents', 'low', 1),
  ('agents.visualization.show_connections', 'true', 'boolean', 'agents', 'medium', 1),
  ('agents.visualization.connection_color', '"#fbbf24"', 'string', 'agents', 'low', 1),
  ('agents.visualization.animate_activity', 'true', 'boolean', 'agents', 'low', 1),

  -- Task Orchestration
  ('agents.tasks.queue_size', '100', 'number', 'agents', 'medium', 0),
  ('agents.tasks.timeout', '600', 'number', 'agents', 'medium', 1),
  ('agents.tasks.retry_failed', 'true', 'boolean', 'agents', 'medium', 1),
  ('agents.tasks.priority_scheduling', 'true', 'boolean', 'agents', 'low', 0);
```

---

## 🎯 Summary: What Goes Into Settings

### ✅ INCLUDE (User-Controllable Agent Settings)
- Agent spawning configuration (provider, strategy, priority)
- Agent lifecycle management (timeouts, retries, auto-restart)
- Telemetry polling intervals
- Visualization settings (colors, sizes, animations)
- Task queue configuration
- Monitoring and alerting preferences

### ❌ EXCLUDE (Internal Agent Mechanics)
- Docker container configuration
- Supervisord service settings
- MCP server internal settings
- Z.AI worker pool internals
- tmux workspace configuration
- Skills system internals
- User credential management

**Rationale**: The agentic system (Claude/claude-flow) manages its own infrastructure. Users control WHAT agents do and HOW they're visualized, not HOW agents are implemented.

---

## 🚀 Implementation Priority

### Phase 1: Telemetry Integration (✅ DONE)
- Frontend already polls `/api/bots/agents`
- Telemetry stream component exists
- Data structures defined

### Phase 2: Control Panel UI (⏳ TODO)
- Add "Agents" tab to settings panel
- Agent spawner interface
- Task queue viewer
- Swarm coordinator controls

### Phase 3: Visualization Integration (⏳ TODO)
- Add agents as nodes to main graph
- Apply visualization settings
- Show agent connections
- Activity animations

### Phase 4: Settings Database Integration (⏳ TODO)
- Add agent settings to database schema
- Wire settings to spawn API
- Connect to telemetry polling
- Save/load visualization preferences

---

## 📚 Reference

**Telemetry Files**:
- `client/src/telemetry/AgentTelemetry.ts` - Telemetry service
- `client/src/features/bots/components/AgentTelemetryStream.tsx` - UI component

**Backend APIs**:
- `src/handlers/api_handler/bots/mod.rs` - Agent API routes
- `src/handlers/bots_handler.rs` - Agent lifecycle handlers

**Data Structures**:
- `SpawnAgentRequest` - Agent spawn configuration
- `AgentTelemetry` - Real-time agent data
- `AgentGraphData` - Visualization data

---

*This document focuses on user-facing controls, not internal agent infrastructure.*
*Agent container details (Docker, supervisord, MCP, etc.) are implementation details managed by Claude/claude-flow.*
