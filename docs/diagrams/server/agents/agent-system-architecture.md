---
title: Agent/Bot System Architecture
description: 1.  [Overview](#overview) 2.
category: explanation
tags:
  - architecture
  - design
  - patterns
  - structure
  - api
related-docs:
  - diagrams/README.md
updated-date: 2025-12-18
difficulty-level: advanced
---

# Agent/Bot System Architecture

**Version:** 2.0
**Last Updated:** December 5, 2025
**Status:** Production

---

## Table of Contents

1. [Overview](#overview)
2. [Agent Type Hierarchy](#agent-type-hierarchy)
3. [Agent Lifecycle Management](#agent-lifecycle-management)
4. [Task Assignment and Scheduling](#task-assignment-and-scheduling)
5. [Inter-Agent Communication](#inter-agent-communication)
6. [Swarm Coordination Patterns](#swarm-coordination-patterns)
7. [Agent State Machine](#agent-state-machine)
8. [Resource Allocation and Limits](#resource-allocation-and-limits)
9. [Monitoring and Metrics](#monitoring-and-metrics)
10. [Error Handling and Recovery](#error-handling-and-recovery)
11. [MCP Protocol Integration](#mcp-protocol-integration)

---

## Overview

VisionFlow's agent system enables distributed AI orchestration through multiple providers (Claude, Gemini, OpenAI) with sophisticated lifecycle management, real-time monitoring, and intelligent coordination patterns. The system supports 15+ specialized agent types with hive-mind and hierarchical topologies.

### Architecture Layers

```
┌─────────────────────────────────────────────────────┐
│              Client Layer (TypeScript)              │
│  • AgentPollingService (smart polling)              │
│  • BotsWebSocketIntegration (real-time updates)     │
│  • AgentTelemetry (monitoring)                      │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│           Server Layer (Rust/Actix)                 │
│  • AgentVisualizationWs (WebSocket handler)         │
│  • BotsHandler (HTTP endpoints)                     │
│  • AgentMonitorActor (state tracking)               │
│  • TaskOrchestratorActor (task execution)           │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│           Integration Layer                         │
│  • Management API (port 9090)                       │
│  • MCP TCP Client (claude-flow, ruv-swarm)          │
│  • BotsClient (agent cache/snapshot)                │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│         External Agent Providers                    │
│  • Claude Flow (claude-flow MCP)                    │
│  • RuvSwarm (ruv-swarm MCP)                         │
│  • Management API (spawn tasks)                     │
└─────────────────────────────────────────────────────┘
```

---

## Agent Type Hierarchy

### Core Agent Types

The system supports 17 specialized agent types with distinct roles, capabilities, and visual representations:

```typescript
type AgentType =
  // Coordination & Management (Tier 1)
  | 'queen'              // Hive-mind central coordinator
  | 'coordinator'        // Swarm coordination and task distribution

  // Specialized Agents (Tier 2)
  | 'researcher'         // Research, analysis, information gathering
  | 'analyst'            // Data analysis and insights
  | 'architect'          // System design and architecture
  | 'tester'             // Testing and quality assurance
  | 'reviewer'           // Code review and validation
  | 'optimizer'          // Performance optimization
  | 'documenter'         // Documentation generation
  | 'monitor'            // System monitoring and health checks
  | 'coder'              // Code implementation

  // SPARC Methodology Agents (Tier 3)
  | 'requirements_analyst'   // Requirements gathering
  | 'design_architect'       // Design and planning
  | 'task_planner'           // Task breakdown and planning
  | 'implementation_coder'   // Implementation execution
  | 'quality_reviewer'       // Quality assurance
  | 'steering_documenter'    // Documentation and steering

  // Generic Worker (Tier 4)
  | 'specialist'         // General-purpose agent
```

### Agent Type Properties

| Type | Color | Size | Vertical Offset | Priority | Typical Use Case |
|------|-------|------|----------------|----------|------------------|
| **queen** | `#FFD700` (Gold) | 25.0 | 0.0 | Highest | Hive-mind central coordination |
| **coordinator** | `#FF6B6B` (Red) | 20.0 | 2.0 | Very High | Swarm orchestration |
| **researcher** | `#4ECDC4` (Cyan) | 18.0 | 0.0 | High | Information gathering |
| **analyst** | `#45B7D1` (Blue) | 18.0 | 0.0 | High | Data analysis |
| **coder** | `#95E1D3` (Mint) | 16.0 | -1.0 | Medium | Code implementation |
| **optimizer** | `#F38181` (Pink) | 16.0 | -1.0 | Medium | Performance tuning |
| **tester** | `#F6B93B` (Yellow) | 14.0 | -2.0 | Medium | Testing and QA |
| **specialist** | `#DFE4EA` (Gray) | 10.0 | -3.0 | Low | General tasks |

### Agent Capabilities

Each agent type has specific capabilities that define its role:

```rust
pub struct AgentInit {
    pub id: String,
    pub name: String,
    pub agent_type: String,
    pub status: String,

    // Visual properties
    pub color: String,
    pub shape: String,  // sphere, cube, octahedron, cylinder, torus, cone, pyramid
    pub size: f32,

    // Performance metrics
    pub health: f32,         // 0.0-1.0 (100%)
    pub cpu: f32,            // CPU usage percentage
    pub memory: f32,         // Memory usage (MB or %)
    pub activity: f32,       // Current activity level 0.0-1.0

    // Task tracking
    pub tasks_active: u32,
    pub tasks_completed: u32,
    pub success_rate: f32,   // 0.0-1.0

    // Token tracking
    pub tokens: u64,         // Total tokens consumed
    pub token_rate: f32,     // Tokens per second

    // Metadata
    pub capabilities: Vec<String>,
    pub created_at: i64,     // Unix timestamp
}
```

---

## Agent Lifecycle Management

### Lifecycle States

Agents transition through well-defined states during their lifecycle:

```
┌──────────────┐
│ INITIALIZING │ (Spawning, configuring)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     IDLE     │ (Waiting for tasks)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    ACTIVE    │ (Processing tasks)
└──────┬───────┘
       │
       ├───────► BUSY (High workload)
       │
       ├───────► ERROR (Failure state)
       │
       ▼
┌──────────────┐
│ TERMINATING  │ (Cleanup, shutdown)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   OFFLINE    │ (Terminated)
└──────────────┘
```

### Spawn Lifecycle

```rust
// 1. Client initiates spawn request
POST /api/bots/agent/spawn
{
  "agent_type": "coder",
  "swarm_id": "swarm-001",
  "method": "management-api",
  "priority": "high"
}

// 2. Server creates task via TaskOrchestratorActor
CreateTask {
    agent: "coder",
    task: "Spawn coder agent for swarm swarm-001",
    provider: "gemini"  // or "claude-flow"
}

// 3. Management API spawns process
POST http://localhost:9090/v1/tasks
{
  "agent": "coder",
  "task": "...",
  "provider": "claude-flow"
}

// 4. Process manager creates isolated task
- Task ID: uuid
- Working directory: /workspace/tasks/{uuid}
- Log file: /logs/tasks/{uuid}.log
- Environment: API keys, TASK_ID

// 5. Agent registers with MCP server
- Agent announces capabilities
- Joins swarm topology
- Begins polling for tasks

// 6. AgentMonitorActor tracks status
- Polls MCP servers every 2-10 seconds
- Caches agent state
- Broadcasts updates via WebSocket
```

### Pause/Resume (Not Supported)

The Management API does **not** support pause/resume operations. Agents can only be:
- **Spawned** (created and started)
- **Stopped** (SIGTERM → SIGKILL after 10s)

### Termination Protocol

```rust
// 1. Client requests termination
DELETE /api/bots/tasks/{taskId}

// 2. Server sends SIGTERM to process
process.kill(processInfo.pid, 'SIGTERM');

// 3. Process attempts graceful shutdown (10s timeout)
- Finish current operation
- Save state
- Cleanup resources
- Exit with code 0

// 4. If timeout exceeded, force kill
process.kill(processInfo.pid, 'SIGKILL');

// 5. Update agent status
status: 'stopped'
exitTime: Date.now()
exitCode: null or signal
```

---

## Task Assignment and Scheduling

### Task Creation Flow

```
┌──────────────────────────────────────────────────┐
│  1. User creates task via API                    │
│     POST /api/bots/swarm/initialize              │
└─────────────────┬────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────┐
│  2. Server selects agent type by strategy        │
│     • strategic → planner                        │
│     • tactical → coder                           │
│     • adaptive → researcher                      │
└─────────────────┬────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────┐
│  3. TaskOrchestratorActor sends to Mgmt API      │
│     CreateTask { agent, task, provider }         │
└─────────────────┬────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────┐
│  4. Management API spawns process                │
│     • Creates isolated directory                 │
│     • Spawns with environment variables          │
│     • Streams logs to file                       │
└─────────────────┬────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────┐
│  5. Process executes (claude/agentic-flow)       │
│     • Reads task from environment                │
│     • Connects to MCP servers                    │
│     • Executes task logic                        │
│     • Writes output to working directory         │
└─────────────────┬────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────┐
│  6. Task completes (exit code 0 or error)        │
│     • status: 'completed' or 'failed'            │
│     • exitCode: 0 or error code                  │
│     • logs captured in file                      │
└──────────────────────────────────────────────────┘
```

### Scheduling Algorithms

**Priority-Based Scheduling:**
```typescript
type Priority = 'low' | 'medium' | 'high' | 'critical';

interface TaskPriority {
  low: 1,      // Background tasks, cleanup
  medium: 2,   // Normal development tasks
  high: 3,     // User-requested features
  critical: 4  // System-critical operations
}
```

**Strategy-Based Agent Selection:**
```rust
fn select_agent_type(strategy: &str) -> &str {
    match strategy {
        "strategic" => "planner",    // High-level planning
        "tactical" => "coder",       // Implementation focused
        "adaptive" => "researcher",  // Research and exploration
        _ => "coder"                 // Default fallback
    }
}
```

### Task Queue Management

The Management API maintains a simple queue:

```javascript
class ProcessManager {
  constructor() {
    this.processes = new Map(); // taskId -> processInfo
  }

  spawnTask(agent, task, provider) {
    const taskId = uuidv4();
    const taskDir = path.join(this.workspaceRoot, 'tasks', taskId);
    const logFile = path.join(this.logsRoot, `${taskId}.log`);

    // Spawn isolated process
    const childProcess = spawn(command, args, {
      cwd: taskDir,
      env: taskEnv,
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe']
    });

    this.processes.set(taskId, {
      pid: childProcess.pid,
      taskId,
      agent,
      task,
      provider,
      startTime: Date.now(),
      status: 'running'
    });

    return processInfo;
  }
}
```

---

## Inter-Agent Communication

### Communication Protocols

**1. Direct Agent-to-Agent (via MCP)**
```
Agent A ──┐
          ├──► MCP Server ──► Agent B
Agent C ──┘
```

**2. Broadcast (via Coordinator)**
```
Coordinator
    ├──► Agent A
    ├──► Agent B
    ├──► Agent C
    └──► Agent D
```

**3. Hierarchical (Queen → Coordinators → Workers)**
```
       Queen
    ┌────┴────┐
Coord A    Coord B
  ├─┴─┐      ├─┴─┐
  W W W      W W W
```

### Message Types

```rust
pub enum AgentVisualizationMessage {
    // Initial swarm setup
    Initialize(InitializeMessage),

    // Real-time position updates (60 FPS)
    PositionUpdate(PositionUpdateMessage),

    // Agent state changes (status, health, metrics)
    StateUpdate(StateUpdateMessage),

    // Connection topology changes
    ConnectionUpdate(ConnectionUpdateMessage),

    // Performance and token metrics
    MetricsUpdate(MetricsUpdateMessage),
}
```

### Communication Flow

```
┌─────────────────────────────────────────────────┐
│  1. Agent A produces message                    │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  2. Message sent to MCP server                  │
│     • JSON-RPC 2.0 format                       │
│     • Tool invocation or state update           │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  3. MCP server routes message                   │
│     • Target agent identified                   │
│     • Queue message if agent busy               │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│  4. Agent B receives and processes              │
│     • Updates internal state                    │
│     • May produce response message              │
└─────────────────────────────────────────────────┘
```

### BotsCommunication Model

```typescript
interface BotsCommunication {
  id: string;
  type: 'communication';
  timestamp: string;
  sender: string;            // Agent ID
  receivers: string[];       // Target agent IDs
  metadata: {
    size: number;           // Message size in bytes
    type?: string;          // 'task', 'status', 'data', etc.
  };
}
```

---

## Swarm Coordination Patterns

### Supported Topologies

**1. Mesh Topology**
```
Every agent connected to every other agent
Pros: Low latency, no single point of failure
Cons: O(n²) connections, high overhead

A ──── B
│ \  / │
│  \/  │
│  /\  │
│ /  \ │
C ──── D
```

**2. Hierarchical Topology**
```
Tree structure with coordinators
Pros: Scalable, clear authority, efficient
Cons: Coordinator can be bottleneck

      Queen
    ┌───┴───┐
  Coord   Coord
  ├─┴─┐   ├─┴─┐
  W W W   W W W
```

**3. Ring Topology**
```
Each agent connected to next in ring
Pros: Simple, predictable routing
Cons: High latency, failure sensitive

A → B → C
↑       ↓
F ← E ← D
```

**4. Star Topology**
```
All agents connected to central coordinator
Pros: Simple management, easy monitoring
Cons: Central coordinator is SPOF

   B   C   D
    \ | /
      A (Central)
    / | \
   E   F   G
```

### Swarm Initialization

```rust
// Client request
POST /api/bots/swarm/initialize
{
  "topology": "hierarchical",
  "max_agents": 12,
  "strategy": "adaptive",
  "enable_neural": true,
  "agent_types": ["researcher", "coder", "tester"],
  "custom_prompt": "Build a REST API with full test coverage"
}

// Server processing
pub async fn initialize_hive_mind_swarm(
    request: web::Json<InitializeSwarmRequest>,
    state: web::Data<AppState>,
) -> Result<impl Responder> {
    // 1. Build task description
    let task = format!(
        "Initialize {} swarm with {} strategy and {} agents.
         Agent types: {}. Neural enabled: {}

         **IMPORTANT:** Messages will be displayed in telemetry panel.",
        request.topology,
        request.strategy,
        request.max_agents,
        request.agent_types.join(", "),
        request.enable_neural
    );

    // 2. Select agent type by strategy
    let agent_type = match request.strategy.as_str() {
        "strategic" => "planner",
        "tactical" => "coder",
        "adaptive" => "researcher",
        _ => "coder",
    };

    // 3. Create task via TaskOrchestratorActor
    let create_task_msg = CreateTask {
        agent: agent_type.to_string(),
        task,
        provider: "gemini",  // or from env
    };

    state.get_task_orchestrator_addr()
         .send(create_task_msg)
         .await
}
```

### Coordination Patterns

**Queen-Led Hive Mind:**
```rust
pub struct HiveMindPattern {
    queen_id: String,              // Central coordinator
    coordinators: Vec<String>,     // Sub-coordinators
    workers: Vec<String>,          // Worker agents
    coordination_efficiency: f32,  // 0.0-1.0
}

impl HiveMindPattern {
    // Queen broadcasts to all coordinators
    fn broadcast_task(&self, task: Task) {
        for coord in &self.coordinators {
            send_message(coord, task.clone());
        }
    }

    // Coordinators delegate to workers
    fn delegate_to_workers(&self, coordinator_id: &str, subtasks: Vec<Task>) {
        let workers = self.get_managed_workers(coordinator_id);
        for (worker, subtask) in workers.iter().zip(subtasks) {
            send_message(worker, subtask);
        }
    }
}
```

**Peer-to-Peer Mesh:**
```rust
pub struct MeshPattern {
    agents: HashMap<String, AgentInfo>,
    connections: Vec<(String, String)>,  // (source, target) pairs
}

impl MeshPattern {
    // Any agent can communicate with any other
    fn send_direct(&self, from: &str, to: &str, msg: Message) {
        if self.has_connection(from, to) {
            send_message(to, msg);
        }
    }

    // Broadcast to all connected agents
    fn broadcast(&self, from: &str, msg: Message) {
        for conn in &self.connections {
            if conn.0 == from {
                send_message(&conn.1, msg.clone());
            }
        }
    }
}
```

---

## Agent State Machine

### State Definitions

```rust
pub enum AgentStatus {
    // Initialization phase
    Initializing {
        start_time: i64,
        progress: f32,  // 0.0-1.0
    },

    // Ready but not working
    Idle {
        last_active: i64,
        idle_duration: i64,
    },

    // Processing tasks
    Active {
        current_task: String,
        task_started: i64,
        progress: f32,
    },

    // High workload
    Busy {
        active_tasks: Vec<String>,
        queue_length: u32,
        estimated_completion: i64,
    },

    // Error state
    Error {
        error_type: String,
        error_message: String,
        timestamp: i64,
        recoverable: bool,
    },

    // Shutting down
    Terminating {
        reason: String,
        cleanup_progress: f32,
    },

    // Terminated
    Offline {
        exit_time: i64,
        exit_code: Option<i32>,
    },
}
```

### State Transitions

```
INITIALIZING
    │
    ├─► [Success] ──────► IDLE
    │
    └─► [Failure] ──────► ERROR
                             │
IDLE                         │
    │                        │
    ├─► [Task assigned] ──► ACTIVE
    │                        │
    ├─► [Timeout] ────────► OFFLINE
    │                        │
    └─► [Error] ──────────► ERROR
                             │
ACTIVE                       │
    │                        │
    ├─► [Task complete] ──► IDLE
    │                        │
    ├─► [High load] ──────► BUSY
    │                        │
    ├─► [Error] ──────────► ERROR
    │                        │
    └─► [Stop signal] ────► TERMINATING
                             │
BUSY                         │
    │                        │
    ├─► [Load reduced] ───► ACTIVE
    │                        │
    ├─► [Error] ──────────► ERROR
    │                        │
    └─► [Stop signal] ────► TERMINATING
                             │
ERROR                        │
    │                        │
    ├─► [Recoverable] ────► IDLE (retry)
    │                        │
    └─► [Fatal] ──────────► TERMINATING
                             │
TERMINATING                  │
    │                        │
    └─► [Cleanup done] ───► OFFLINE
```

### State Update Protocol

```typescript
// Client-side state updates (smart polling)
export class AgentPollingService {
  private currentInterval: number;
  private activityLevel: 'active' | 'idle' = 'idle';

  // Adaptive polling based on activity
  private updateActivityLevel(data: AgentSwarmData, hasChanged: boolean) {
    const activeAgents = data.metadata?.active_agents || 0;
    const totalAgents = data.metadata?.total_agents || 0;
    const activeRatio = totalAgents > 0 ? activeAgents / totalAgents : 0;

    // Intelligent thresholds
    if (activeRatio > 0.2 || hasChanged ||
        data.metadata?.total_tasks > data.metadata?.completed_tasks) {
      this.activityLevel = 'active';
      this.currentInterval = 2000;  // 2s for active
    } else {
      this.activityLevel = 'idle';
      this.currentInterval = 10000; // 10s for idle
    }
  }
}
```

---

## Resource Allocation and Limits

### Per-Agent Resource Limits

```rust
pub struct AgentResourceLimits {
    // Compute resources
    cpu_limit: f32,        // Max CPU % (0.0-1.0)
    memory_limit: u64,     // Max memory in MB

    // Task limits
    max_active_tasks: u32,     // Concurrent task limit
    task_queue_size: u32,      // Max queued tasks

    // Token limits
    max_tokens_per_request: u64,   // Single request limit
    tokens_per_minute: u64,        // Rate limit
    total_token_budget: Option<u64>,  // Optional total budget

    // Time limits
    max_task_duration: Duration,   // Single task timeout
    max_idle_time: Duration,       // Auto-terminate after idle

    // Health thresholds
    min_health_score: f32,     // Min 0.0-1.0 before intervention
    error_threshold: u32,      // Max errors before restart
}

impl Default for AgentResourceLimits {
    fn default() -> Self {
        Self {
            cpu_limit: 0.8,                    // 80% CPU
            memory_limit: 4096,                // 4GB RAM
            max_active_tasks: 3,
            task_queue_size: 10,
            max_tokens_per_request: 100_000,
            tokens_per_minute: 200_000,
            total_token_budget: None,          // Unlimited
            max_task_duration: Duration::from_secs(3600),  // 1 hour
            max_idle_time: Duration::from_secs(1800),      // 30 min
            min_health_score: 0.3,             // 30% health
            error_threshold: 5,
        }
    }
}
```

### System-Wide Resource Management

```javascript
// Management API tracks all active processes
class ProcessManager {
  // Cleanup old completed tasks
  cleanup(maxAge = 3600000) {  // 1 hour default
    const now = Date.now();
    for (const [taskId, info] of this.processes.entries()) {
      if (info.status !== 'running' &&
          info.exitTime &&
          (now - info.exitTime) > maxAge) {
        this.processes.delete(taskId);
      }
    }
  }

  // Get resource utilization
  getActiveTasks() {
    return Array.from(this.processes.values())
      .filter(p => p.status === 'running');
  }
}

// Periodic cleanup (every 10 minutes)
setInterval(() => {
  processManager.cleanup(3600000);  // 1 hour retention
}, 600000);
```

---

## Monitoring and Metrics

### Real-Time Metrics Collection

**Server-Side (Rust):**
```rust
pub struct AgentMetrics {
    pub id: String,
    pub tokens: u64,
    pub token_rate: f32,           // Tokens/sec
    pub tasks_completed: u32,
    pub success_rate: f32,         // 0.0-1.0
}

pub struct SwarmMetrics {
    pub total_agents: u32,
    pub active_agents: u32,
    pub health_avg: f32,
    pub cpu_total: f32,
    pub memory_total: f32,
    pub tokens_total: u64,
    pub tokens_per_second: f32,
}
```

**Client-Side Polling (TypeScript):**
```typescript
export class AgentPollingService {
  private performanceMonitor: PollingPerformanceMonitor;

  async poll(): Promise<void> {
    const startTime = Date.now();

    // Fetch data from server
    const data = await unifiedApiClient.getData<AgentSwarmData>('/graph/data');

    const pollDuration = Date.now() - startTime;

    // Track performance
    const dataHash = this.hashData(data);
    const hasChanged = dataHash !== this.lastDataHash;
    this.performanceMonitor.recordPoll(pollDuration, hasChanged);

    // Update activity level for adaptive polling
    this.updateActivityLevel(data, hasChanged);
  }

  getPerformanceMetrics() {
    return {
      avgPollTime: number,
      minPollTime: number,
      maxPollTime: number,
      totalPolls: number,
      changedPolls: number,
      changeRate: number,
      errorCount: number,
      errorRate: number
    };
  }
}
```

### Prometheus Metrics (Management API)

```javascript
// Metrics exposed at http://localhost:9090/metrics
const metrics = {
  // HTTP metrics
  http_request_duration_seconds: Histogram,
  http_requests_total: Counter,

  // Task metrics
  active_tasks_total: Gauge,
  completed_tasks_total: Counter,
  task_duration_seconds: Histogram,

  // MCP metrics
  mcp_tool_invocations_total: Counter,
  mcp_tool_duration_seconds: Histogram,

  // Worker metrics
  worker_sessions_total: Gauge,

  // Error tracking
  api_errors_total: Counter,
};

// Record task completion
function recordTask(taskType, status, duration) {
  completedTasks.inc({ status });
  taskDuration.observe({ task_type: taskType, status }, duration);
}
```

### Health Monitoring

```rust
pub struct AgentHealthCheck {
    pub agent_id: String,
    pub health_score: f32,      // 0.0-1.0
    pub last_heartbeat: i64,
    pub response_time_ms: f32,
    pub error_count: u32,
    pub consecutive_failures: u32,
    pub is_healthy: bool,
}

impl AgentHealthCheck {
    pub fn evaluate_health(&mut self) -> bool {
        self.is_healthy =
            self.health_score >= 0.3 &&
            self.consecutive_failures < 5 &&
            (time::now().timestamp() - self.last_heartbeat) < 60;

        self.is_healthy
    }

    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.health_score = (self.health_score * 0.9 + 1.0 * 0.1).min(1.0);
    }

    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        self.health_score = (self.health_score * 0.9).max(0.0);
        self.error_count += 1;
    }
}
```

---

## Error Handling and Recovery

### Error Types

```rust
pub enum AgentError {
    // Initialization errors
    SpawnFailed {
        agent_type: String,
        reason: String,
    },

    // Runtime errors
    TaskExecutionFailed {
        task_id: String,
        error: String,
        is_recoverable: bool,
    },

    // Communication errors
    ConnectionLost {
        server_id: String,
        last_seen: i64,
    },

    // Resource errors
    ResourceExhausted {
        resource_type: String,  // "cpu", "memory", "tokens"
        current: f64,
        limit: f64,
    },

    // Timeout errors
    TaskTimeout {
        task_id: String,
        duration: Duration,
        max_duration: Duration,
    },
}
```

### Recovery Strategies

**1. Automatic Retry with Exponential Backoff**
```rust
pub struct RetryPolicy {
    max_retries: u32,
    initial_delay: Duration,
    max_delay: Duration,
    backoff_multiplier: f32,
}

impl RetryPolicy {
    pub async fn execute_with_retry<F, T, E>(
        &self,
        operation: F,
    ) -> Result<T, E>
    where
        F: Fn() -> Result<T, E>,
    {
        let mut attempt = 0;
        let mut delay = self.initial_delay;

        loop {
            match operation() {
                Ok(result) => return Ok(result),
                Err(e) if attempt >= self.max_retries => return Err(e),
                Err(_) => {
                    sleep(delay).await;
                    attempt += 1;
                    delay = (delay * self.backoff_multiplier).min(self.max_delay);
                }
            }
        }
    }
}
```

**2. Health-Based Auto-Restart**
```typescript
// Client-side error handling
export class AgentPollingService {
  private handlePollingError(error: Error): void {
    this.retryCount++;
    this.performanceMonitor.recordError();

    // Notify error callbacks
    this.errorCallbacks.forEach(callback => {
      callback(error);
    });

    // Stop after max retries
    if (this.retryCount >= this.config.maxRetries) {
      logger.error('Max retries reached, stopping polling');
      this.stop();
      return;
    }

    // Exponential backoff
    const retryDelay = this.config.retryDelay * Math.pow(2, this.retryCount - 1);
    setTimeout(() => this.poll(), retryDelay);
  }
}
```

**3. Graceful Degradation**
```rust
// If primary MCP server fails, fall back to cached data
pub async fn get_agents() -> Result<Vec<Agent>, Error> {
    match mcp_client.fetch_agents().await {
        Ok(agents) => {
            // Update cache
            cache.update(agents.clone());
            Ok(agents)
        }
        Err(e) => {
            warn!("MCP fetch failed: {}, using cache", e);
            // Return cached data
            Ok(cache.get_cached_agents())
        }
    }
}
```

### Circuit Breaker Pattern

```rust
pub struct CircuitBreaker {
    state: CircuitState,
    failure_threshold: u32,
    failure_count: u32,
    success_threshold: u32,
    success_count: u32,
    timeout: Duration,
    last_failure_time: Option<Instant>,
}

pub enum CircuitState {
    Closed,      // Normal operation
    Open,        // Failing, reject requests
    HalfOpen,    // Testing if service recovered
}

impl CircuitBreaker {
    pub async fn execute<F, T>(&mut self, operation: F) -> Result<T, Error>
    where
        F: Future<Output = Result<T, Error>>,
    {
        match self.state {
            CircuitState::Open => {
                // Check if timeout expired
                if let Some(last_fail) = self.last_failure_time {
                    if last_fail.elapsed() > self.timeout {
                        self.state = CircuitState::HalfOpen;
                    } else {
                        return Err(Error::CircuitOpen);
                    }
                }
            }
            _ => {}
        }

        // Execute operation
        match operation.await {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(e) => {
                self.on_failure();
                Err(e)
            }
        }
    }

    fn on_success(&mut self) {
        self.failure_count = 0;
        self.success_count += 1;

        if self.state == CircuitState::HalfOpen &&
           self.success_count >= self.success_threshold {
            self.state = CircuitState::Closed;
            self.success_count = 0;
        }
    }

    fn on_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());

        if self.failure_count >= self.failure_threshold {
            self.state = CircuitState::Open;
        }
    }
}
```

---

## MCP Protocol Integration

### MCP Server Architecture

```
┌───────────────────────────────────────────────┐
│         VisionFlow (Rust/Actix)               │
│                                               │
│  ┌──────────────────────────────────────┐    │
│  │     MCP TCP Client                   │    │
│  │  • Connects to port 9500 (TCP)       │    │
│  │  • Sends JSON-RPC 2.0 requests       │    │
│  │  • Handles responses/notifications   │    │
│  └──────────────┬───────────────────────┘    │
└─────────────────┼────────────────────────────┘
                  │
                  │ TCP Socket
                  │
┌─────────────────▼────────────────────────────┐
│       MCP Server (claude-flow/ruv-swarm)      │
│                                               │
│  ┌──────────────────────────────────────┐    │
│  │   MCP Server (port 9500)             │    │
│  │  • Receives JSON-RPC requests        │    │
│  │  • Invokes MCP tools                 │    │
│  │  • Returns agent status              │    │
│  └──────────────┬───────────────────────┘    │
└─────────────────┼────────────────────────────┘
                  │
                  │ Spawns/Manages
                  │
┌─────────────────▼────────────────────────────┐
│        Agent Processes (Gemini/Claude)        │
│                                               │
│  • Individual processes per agent             │
│  • Execute tasks in isolated directories      │
│  • Log to individual files                    │
│  • Report status to MCP server                │
└───────────────────────────────────────────────┘
```

### MCP Protocol Messages

**1. Tool Invocation (Agent List)**
```json
// Request
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "method": "tools/call",
  "params": {
    "name": "agent_list",
    "arguments": {
      "filter": "active"
    }
  }
}

// Response
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"agents\": [{\"id\": \"agent-001\", \"type\": \"coder\", \"status\": \"active\", ...}]}"
      }
    ]
  }
}
```

**2. Agent Status Query**
```json
// Request
{
  "jsonrpc": "2.0",
  "id": "req-002",
  "method": "tools/call",
  "params": {
    "name": "agent_metrics",
    "arguments": {
      "agentId": "agent-001"
    }
  }
}

// Response
{
  "jsonrpc": "2.0",
  "id": "req-002",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"cpu\": 45.2, \"memory\": 1024, \"tokens\": 15000, \"tasks_active\": 2}"
      }
    ]
  }
}
```

### MCP Client Implementation

```rust
pub struct MCPTcpClient {
    host: String,
    port: u16,
    stream: Option<TcpStream>,
    request_id_counter: AtomicU64,
}

impl MCPTcpClient {
    pub async fn connect(&mut self) -> Result<(), Error> {
        let addr = format!("{}:{}", self.host, self.port);
        self.stream = Some(TcpStream::connect(&addr).await?);
        Ok(())
    }

    pub async fn call_tool(
        &mut self,
        tool_name: &str,
        arguments: Value,
    ) -> Result<ToolResponse, Error> {
        let request_id = self.request_id_counter
            .fetch_add(1, Ordering::SeqCst)
            .to_string();

        let request = json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        });

        // Send request
        self.send_json(&request).await?;

        // Read response
        let response = self.read_json().await?;

        // Parse response
        Ok(serde_json::from_value(response)?)
    }

    pub async fn list_agents(&mut self) -> Result<Vec<AgentInfo>, Error> {
        let response = self.call_tool("agent_list", json!({})).await?;
        // Parse agent list from response
        Ok(parse_agent_list(response))
    }
}
```

### Agent Discovery Flow

```
┌────────────────────────────────────────────────┐
│  1. AgentMonitorActor starts                   │
│     • Initializes MCP TCP client               │
│     • Configures polling interval (2-10s)      │
└─────────────────┬──────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────┐
│  2. Poll MCP servers for agent status          │
│     • Call agent_list tool                     │
│     • Call agent_metrics tool                  │
└─────────────────┬──────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────┐
│  3. Update BotsClient cache                    │
│     • Store agent snapshots                    │
│     • Calculate metrics                        │
│     • Track changes                            │
└─────────────────┬──────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────┐
│  4. Broadcast updates via WebSocket            │
│     • Send AgentVisualizationMessage           │
│     • Update client visualizations             │
│     • Trigger telemetry logs                   │
└────────────────────────────────────────────────┘
```

---

---

---

## Related Documentation

- [Blender MCP Unified System Architecture](../../../architecture/blender-mcp-unified-architecture.md)
- [Agent Orchestration & Multi-Agent Systems](../../mermaid-library/04-agent-orchestration.md)
- [VisionFlow Client Architecture Analysis](../../../visionflow-architecture-analysis.md)
- [VisionFlow Architecture Diagrams - Complete Corpus](../../README.md)
- [VisionFlow Complete Architecture Documentation](../../../ARCHITECTURE_COMPLETE.md)

## Summary

The VisionFlow agent system provides:

✅ **17 specialized agent types** with distinct roles and capabilities
✅ **Robust lifecycle management** with spawn, monitor, and terminate operations
✅ **Intelligent task scheduling** with priority-based and strategy-based allocation
✅ **Multiple coordination patterns** (mesh, hierarchical, ring, star)
✅ **Real-time monitoring** with adaptive polling and performance tracking
✅ **Comprehensive error handling** with retry, circuit breaker, and graceful degradation
✅ **MCP protocol integration** for agent-to-agent and system-to-agent communication
✅ **Resource limits and health checks** to ensure system stability

### Key Integration Points

1. **Client → Server:** AgentPollingService with smart polling (2s active, 10s idle)
2. **Server → MCP:** MCPTcpClient for agent status queries
3. **Server → Management API:** TaskOrchestratorActor for task spawning
4. **Server → Client:** WebSocket for real-time state updates

### Performance Characteristics

- **Polling latency:** 2-10 seconds (adaptive based on activity)
- **WebSocket updates:** ~16ms (60 FPS for position updates)
- **Task spawn time:** ~100-500ms
- **Max agents per swarm:** Configurable (default 12-25)
- **Token throughput:** Provider-dependent (Claude: ~200k/min, Gemini: ~60k/min)

---

**Document Version:** 2.0
**Last Updated:** December 5, 2025
**Maintainer:** VisionFlow Architecture Team
