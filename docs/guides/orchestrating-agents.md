# Orchestrating Agents

[← Knowledge Base](../index.md) > [Guides](./index.md) > Orchestrating Agents

This comprehensive guide covers agent orchestration within the VisionFlow system, including practical examples, topology patterns, MCP protocol integration, and troubleshooting strategies for production multi-agent workflows.

## Table of Contents

1. [Agent Architecture Overview](#agent-architecture-overview)
2. [Agent Types and Roles](#agent-types-and-roles)
3. [Spawning and Managing Agents](#spawning-and-managing-agents)
4. [Coordination Topologies](#coordination-topologies)
5. [MCP Protocol Integration](#mcp-protocol-integration)
6. [Multi-Agent Container Integration](#multi-agent-container-integration)
7. [Inter-Agent Communication](#inter-agent-communication)
8. [Task Distribution](#task-distribution)
9. [Monitoring and Telemetry](#monitoring-and-telemetry)
10. [Error Handling and Recovery](#error-handling-and-recovery)
11. [Troubleshooting Agent Failures](#troubleshooting-agent-failures)
12. [Production Best Practices](#production-best-practices)

## Agent Architecture Overview

### System Components

```mermaid
graph TB
    subgraph "Control Plane"
        OM[Orchestration Manager]
        TQ[Task Queue]
        SM[State Manager]
        MCP[MCP Server :9500]
    end

    subgraph "Agent Pool"
        A1["Coordinator Agent"]
        A2["Researcher Agent"]
        A3["Coder Agent"]
        A4["Reviewer Agent"]
        A5["Tester Agent"]
        A6["Architect Agent"]
    end

    subgraph "Communication Layer"
        TCP[TCP :9500]
        WS[WebSocket :3002]
        MB[Message Bus]
    end

    subgraph "Monitoring"
        TM[Telemetry]
        LG[Logging]
        MT[Metrics]
    end

    OM --> MCP
    MCP --> TCP
    MCP --> WS
    OM --> TQ
    OM --> SM
    TQ --> A1
    TQ --> A2
    TQ --> A3
    TQ --> A4
    TQ --> A5
    TQ --> A6

    A1 --> MB
    A2 --> MB
    A3 --> MB
    A4 --> MB
    A5 --> MB
    A6 --> MB

    MB --> TM
    TM --> MT
    TM --> LG
```

### Core Concepts

1. **Agent**: Autonomous unit with specific capabilities and expertise
2. **Coordinator**: Meta-agent that orchestrates other agents
3. **Swarm**: Collection of agents working towards a common goal
4. **Task**: Unit of work with requirements and acceptance criteria
5. **Topology**: Communication and coordination pattern between agents
6. **MCP Protocol**: Model Context Protocol for agent communication (TCP port 9500)

## Agent Types and Roles

### Core Agent Types

| Agent Type | Role | Primary Capabilities | Use Cases |
|------------|------|---------------------|-----------|
| **Coordinator** | Orchestration | Task decomposition, agent supervision, workflow management | Complex project coordination, swarm management |
| **Researcher** | Information gathering | Web search, documentation analysis, data collection | Requirements analysis, technology research |
| **Coder** | Implementation | Code generation, refactoring, optimisation, API design | Feature development, code implementation |
| **Architect** | System design | Architecture planning, design patterns, scalability | System architecture, technical strategy |
| **Tester** | Quality assurance | Test creation, validation, automated testing | Testing, quality gates, validation |
| **Reviewer** | Code review | Quality assessment, best practices, security review | Code review, improvement suggestions |

### Practical Agent Examples

#### 1. Coordinator Agent

A coordinator manages high-level workflows and delegates to specialised agents:

```yaml
agent_type: coordinator
name: "project-coordinator"
capabilities:
  - task_decomposition
  - agent_supervision
  - workflow_management
  - performance_monitoring
  - conflict_resolution
config:
  max_workers: 10
  strategy: adaptive
  supervision_interval: 5000ms
```

**Example Workflow:**
```bash
# Initialize hierarchical swarm
mcp__claude-flow__swarm_init hierarchical \
  --maxAgents=10 \
  --strategy=adaptive

# Spawn specialised workers
mcp__claude-flow__agent_spawn researcher \
  --capabilities="research,analysis"

mcp__claude-flow__agent_spawn coder \
  --capabilities="implementation,testing"

# Coordinate task execution
mcp__claude-flow__task_orchestrate \
  "Build authentication service" \
  --strategy=sequential \
  --priority=high
```

See: Reference documentation for hierarchical coordination patterns

#### 2. Researcher Agent

Gathers information and analyses requirements:

```yaml
agent_type: researcher
name: "requirements-researcher"
capabilities:
  - information_gathering
  - market_research
  - competitive_analysis
  - documentation_analysis
config:
  search_depth: 5
  max_concurrent_tasks: 3
  timeout: 600
```

**Example Usage:**
```python
# Spawn researcher agent
researcher = await orchestrator.spawn_agent(
    agent_type="researcher",
    config={
        "search_depth": 5,
        "sources": ["web", "documentation", "code_repositories"],
        "timeout": 600
    }
)

# Assign research task
task = Task(
    type="requirements_analysis",
    description="Research authentication best practices",
    required_capabilities=["research", "security"]
)

result = await researcher.process_task(task)
```

See: Reference documentation for agent types and capabilities

#### 3. Coder Agent

Implements solutions following best practices:

```yaml
agent_type: coder
name: "backend-coder"
capabilities:
  - code_generation
  - refactoring
  - optimisation
  - api_design
  - error_handling
config:
  languages: ["python", "typescript", "rust"]
  linting: true
  formatting: true
  test_generation: true
```

**Example Implementation:**
```typescript
// Code generation with TDD approach
class CoderAgent {
  async implement(specification: Specification): Promise<Implementation> {
    // 1. Generate test cases first
    const tests = await this.generateTests(specification);

    // 2. Implement core functionality
    const code = await this.generateCode(specification);

    // 3. Refactor and optimise
    const optimised = await this.refactor(code);

    // 4. Run validation
    await this.validate(optimised, tests);

    return { code: optimised, tests };
  }
}
```

See: Reference documentation for coder agent capabilities

#### 4. Architect Agent

Designs system architecture and technical strategy:

```yaml
agent_type: architect
name: "system-architect"
capabilities:
  - system_design
  - architecture_patterns
  - scalability_planning
  - technical_strategy
config:
  design_methodology: "domain_driven_design"
  architecture_style: "microservices"
```

**Example Design Process:**
```bash
# Create architecture design
mcp__claude-flow__agent_spawn architect \
  --capabilities="system_design,patterns,scalability"

# Design authentication system
mcp__claude-flow__task_orchestrate \
  "Design scalable authentication architecture" \
  --requiredCapabilities="system_design,security" \
  --strategy=sequential
```

See: Reference documentation for architecture agent capabilities

#### 5. Tester Agent

Creates comprehensive test suites and validates functionality:

```yaml
agent_type: tester
name: "qa-tester"
capabilities:
  - test_creation
  - validation
  - quality_assurance
  - automated_testing
config:
  test_types: ["unit", "integration", "e2e"]
  coverage_threshold: 0.8
  test_framework: "pytest"
```

**Example Test Generation:**
```python
class TesterAgent:
    async def generate_test_suite(self, implementation):
        # Generate unit tests
        unit_tests = await self.generate_unit_tests(implementation)

        # Generate integration tests
        integration_tests = await self.generate_integration_tests(implementation)

        # Generate end-to-end tests
        e2e_tests = await self.generate_e2e_tests(implementation)

        # Run validation
        coverage = await self.run_tests_with_coverage([
            *unit_tests,
            *integration_tests,
            *e2e_tests
        ])

        return TestSuite(
            unit=unit_tests,
            integration=integration_tests,
            e2e=e2e_tests,
            coverage=coverage
        )
```

See: Reference documentation for tester agent capabilities

#### 6. Reviewer Agent

Performs code review and quality assessment:

```yaml
agent_type: reviewer
name: "code-reviewer"
capabilities:
  - code_review
  - quality_assessment
  - security_review
  - best_practices
config:
  review_depth: "thorough"
  focus_areas: ["security", "performance", "maintainability"]
```

**Example Review Process:**
```python
class ReviewerAgent:
    async def review_code(self, code_submission):
        # Analyse code quality
        quality_issues = await self.analyse_quality(code_submission)

        # Check security vulnerabilities
        security_issues = await self.check_security(code_submission)

        # Review best practices
        practice_issues = await self.check_best_practices(code_submission)

        # Performance analysis
        performance_issues = await self.analyse_performance(code_submission)

        return ReviewReport(
            quality=quality_issues,
            security=security_issues,
            practices=practice_issues,
            performance=performance_issues,
            recommendation=self.make_recommendation([
                quality_issues,
                security_issues,
                practice_issues,
                performance_issues
            ])
        )
```

See: Reference documentation for reviewer agent capabilities

## Spawning and Managing Agents

### Agent Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Created
    Created --> Initialising
    Initialising --> Ready
    Ready --> Processing
    Processing --> Ready
    Processing --> Error
    Error --> Recovering
    Recovering --> Ready
    Recovering --> Failed
    Ready --> Terminating
    Error --> Terminating
    Failed --> Terminating
    Terminating --> [*]
```

### Manual Agent Spawning

#### Using MCP TCP Protocol (Port 9500)

```bash
# Connect to MCP server at multi-agent-container:9500
# Initialize swarm first
curl -X POST http://localhost:9500 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "swarm_init",
      "arguments": {
        "topology": "hierarchical",
        "maxAgents": 10
      }
    }
  }'

# Spawn coder agent
curl -X POST http://localhost:9500 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "agent_spawn",
      "arguments": {
        "agentType": "coder",
        "swarmId": "swarm_1757880683494_yl81sece5",
        "config": {
          "model": "claude-3-opus",
          "capabilities": ["python", "rust", "typescript"],
          "maxTokens": 4096,
          "timeout": 300
        }
      }
    }
  }'
```

#### Using Docker Exec

```bash
# Spawn agent inside multi-agent-container
docker exec multi-agent-container mcp__claude-flow__agent_spawn \
  researcher \
  --swarmId="swarm_1757880683494_yl81sece5" \
  --capabilities="research,analysis,information_gathering"

# Spawn coder agent with specific configuration
docker exec multi-agent-container mcp__claude-flow__agent_spawn \
  coder \
  --swarmId="swarm_1757880683494_yl81sece5" \
  --config='{"language":"python","frameworks":["fastapi","sqlalchemy"]}'
```

### Programmatic Spawning

```python
from visionflow import AgentOrchestrator
import asyncio

async def spawn_development_team():
    orchestrator = AgentOrchestrator(
        mcp_host="multi-agent-container",
        mcp_port=9500
    )

    # Initialize swarm
    swarm = await orchestrator.initialize_swarm(
        topology="hierarchical",
        max_agents=10
    )

    # Spawn coordinator
    coordinator = await orchestrator.spawn_agent(
        agent_type="coordinator",
        swarm_id=swarm.id,
        config={
            "max_workers": 5,
            "strategy": "adaptive"
        }
    )

    # Spawn development team
    team = await orchestrator.spawn_agents([
        {
            "type": "researcher",
            "count": 1,
            "config": {"search_depth": 5}
        },
        {
            "type": "architect",
            "count": 1,
            "config": {"design_methodology": "ddd"}
        },
        {
            "type": "coder",
            "count": 2,
            "config": {"languages": ["python", "typescript"]}
        },
        {
            "type": "tester",
            "count": 1,
            "config": {"coverage_threshold": 0.8}
        },
        {
            "type": "reviewer",
            "count": 1,
            "config": {"review_depth": "thorough"}
        }
    ])

    return swarm, coordinator, team

# Run
swarm, coordinator, team = asyncio.run(spawn_development_team())
```

### Agent Configuration

```yaml
# config/agents.yaml
agent_templates:
  coordinator:
    class: visionflow.agents.CoordinatorAgent
    default_config:
      max_workers: 10
      supervision_interval: 5000
      strategy: adaptive
      escalation_threshold: 0.7
    resources:
      memory: "4Gi"
      cpu: "2.0"

  researcher:
    class: visionflow.agents.ResearcherAgent
    default_config:
      max_concurrent_tasks: 3
      search_depth: 5
      sources: ["web", "documentation", "code_repositories"]
      timeout: 600
    resources:
      memory: "2Gi"
      cpu: "1.0"

  coder:
    class: visionflow.agents.CoderAgent
    default_config:
      languages: ["python", "typescript", "rust"]
      linting: true
      formatting: true
      test_generation: true
      coverage_threshold: 0.8
    resources:
      memory: "4Gi"
      cpu: "2.0"

  architect:
    class: visionflow.agents.ArchitectAgent
    default_config:
      design_methodology: "domain_driven_design"
      architecture_style: "microservices"
      documentation_level: "comprehensive"
    resources:
      memory: "3Gi"
      cpu: "1.5"

  tester:
    class: visionflow.agents.TesterAgent
    default_config:
      test_types: ["unit", "integration", "e2e"]
      coverage_threshold: 0.8
      test_framework: "pytest"
      parallel_execution: true
    resources:
      memory: "3Gi"
      cpu: "1.5"

  reviewer:
    class: visionflow.agents.ReviewerAgent
    default_config:
      review_depth: "thorough"
      focus_areas: ["security", "performance", "maintainability"]
      automated_fixes: false
    resources:
      memory: "2Gi"
      cpu: "1.0"
```

## Coordination Topologies

### 1. Hierarchical Topology

**Best for:** Large projects with clear structure, mission-critical systems, complex workflows with dependencies.

```
    👑 Coordinator
   /   |   |   \
  🔬   💻   📊   🧪
Research Code Analyst Test
Workers Workers Workers Workers
```

**Characteristics:**
- Centralised command and control
- Clear reporting structure
- Efficient resource allocation
- Single point of coordination

**Example Implementation:**
```python
class HierarchicalCoordinator:
    def __init__(self):
        self.hierarchy = {
            "manager": None,
            "team_leads": [],
            "workers": {}
        }

    async def build_hierarchy(self, project_config):
        # Create manager (coordinator)
        self.hierarchy["manager"] = await self.spawn_agent(
            type="coordinator",
            config={
                "max_workers": 10,
                "strategy": "centralized"
            }
        )

        # Create team leads for each domain
        for team in project_config["teams"]:
            team_lead = await self.spawn_agent(
                type=team["lead_type"],
                config=team["config"]
            )
            self.hierarchy["team_leads"].append(team_lead)

            # Spawn workers for this team
            workers = []
            for worker_spec in team["workers"]:
                worker = await self.spawn_agent(
                    type=worker_spec["type"],
                    config=worker_spec["config"]
                )
                workers.append(worker)

            self.hierarchy["workers"][team_lead.id] = workers

    async def delegate_task(self, task):
        # Manager assigns to appropriate team
        manager = self.hierarchy["manager"]
        team_assignment = await manager.assign_to_team(task)

        # Team lead distributes to workers
        team_lead = self.get_team_lead(team_assignment)
        worker_tasks = await team_lead.distribute_work(task)

        # Workers execute in parallel
        results = await asyncio.gather(*[
            worker.process_task(subtask)
            for worker, subtask in worker_tasks
        ])

        # Team lead aggregates results
        team_result = await team_lead.aggregate_results(results)

        # Manager reviews final output
        final_result = await manager.review_result(team_result)

        return final_result
```

**MCP Commands:**
```bash
# Initialize hierarchical swarm
mcp__claude-flow__swarm_init hierarchical \
  --maxAgents=10 \
  --strategy=centralized

# Monitor hierarchy health
mcp__claude-flow__swarm_monitor --interval=5000

# Generate performance report
mcp__claude-flow__performance_report \
  --format=detailed \
  --timeframe=24h
```

See: [Hierarchical Coordinator Reference](../reference/agents/swarm/hierarchical-coordinator.md)

### 2. Mesh Topology

**Best for:** Highly distributed systems, fault-tolerant applications, collaborative development, democratic decision-making.

```
    🌐 MESH TOPOLOGY
   A ←→ B ←→ C
   ↕     ↕     ↕
   D ←→ E ←→ F
   ↕     ↕     ↕
   G ←→ H ←→ I
```

**Characteristics:**
- Peer-to-peer coordination
- No single point of failure
- Democratic consensus
- Full connectivity between agents

**Example Implementation:**
```python
class MeshCoordinator:
    def __init__(self):
        self.peers = []
        self.connections = {}
        self.consensus_protocol = "gossip"

    async def initialize_mesh(self, agent_count):
        # Spawn peer agents
        for i in range(agent_count):
            peer = await self.spawn_agent(
                type="peer_agent",
                config={
                    "peer_discovery": True,
                    "consensus_protocol": self.consensus_protocol
                }
            )
            self.peers.append(peer)

        # Establish full mesh connectivity
        for peer_a in self.peers:
            for peer_b in self.peers:
                if peer_a != peer_b:
                    await self.establish_connection(peer_a, peer_b)

    async def distribute_task(self, task):
        # Broadcast task to all peers
        bids = await self.broadcast_task_auction(task)

        # Peers vote on best candidate
        winner = await self.conduct_consensus_vote(bids)

        # Winning peer executes task
        result = await winner.execute_task(task)

        # Replicate result across mesh for fault tolerance
        await self.replicate_result(result, replication_factor=3)

        return result

    async def gossip_protocol(self):
        """Gossip algorithm for information dissemination"""
        while True:
            for peer in self.peers:
                # Select random peers for gossip
                targets = random.sample(self.peers, k=3)

                # Exchange state information
                for target in targets:
                    await peer.exchange_state(target)

            await asyncio.sleep(2)  # Gossip interval
```

**MCP Commands:**
```bash
# Initialize mesh network
mcp__claude-flow__swarm_init mesh \
  --maxAgents=12 \
  --strategy=distributed

# Establish peer connections
mcp__claude-flow__daa_communication \
  --from="mesh-coordinator" \
  --to="all" \
  --message='{"type":"network_init","topology":"mesh"}'

# Conduct consensus vote
mcp__claude-flow__daa_consensus \
  --agents="all" \
  --proposal='{"task_assignment":"auth-service","voting_mechanism":"weighted"}'

# Monitor network health
mcp__claude-flow__swarm_monitor \
  --interval=3000 \
  --metrics="connectivity,latency,throughput"
```

See: Reference documentation for mesh coordination patterns

### 3. Sequential (Pipeline) Topology

**Best for:** Workflows with clear stages, data processing pipelines, CI/CD automation.

```
Research → Plan → Code → Review → Test → Deploy
```

**Example Implementation:**
```python
class PipelineCoordinator:
    def __init__(self, stages):
        self.stages = stages
        self.pipeline = self.build_pipeline(stages)

    async def execute_pipeline(self, initial_input):
        result = initial_input

        for stage in self.pipeline:
            # Get appropriate agent for stage
            agent = await self.get_agent_for_stage(stage)

            # Create task
            task = Task(
                type=stage['task_type'],
                input=result,
                config=stage.get('config', {})
            )

            # Execute stage
            result = await agent.process_task(task)

            # Transform output if needed
            if stage.get('transform'):
                result = stage['transform'](result)

            # Validate stage output
            if stage.get('validation'):
                await self.validate_stage_output(result, stage['validation'])

        return result

# Usage example
pipeline = PipelineCoordinator([
    {
        "stage": "research",
        "agent_type": "researcher",
        "task_type": "gather_requirements",
        "validation": {"completeness": 0.9}
    },
    {
        "stage": "architecture",
        "agent_type": "architect",
        "task_type": "design_system",
        "validation": {"scalability_score": 0.8}
    },
    {
        "stage": "implementation",
        "agent_type": "coder",
        "task_type": "implement_solution",
        "validation": {"code_quality": 0.85}
    },
    {
        "stage": "review",
        "agent_type": "reviewer",
        "task_type": "review_code",
        "validation": {"approval_required": True}
    },
    {
        "stage": "testing",
        "agent_type": "tester",
        "task_type": "run_tests",
        "validation": {"coverage": 0.8, "pass_rate": 1.0}
    }
])

result = await pipeline.execute_pipeline(project_requirements)
```

**MCP Commands:**
```bash
# Execute sequential pipeline
mcp__claude-flow__task_orchestrate \
  "Build authentication service" \
  --strategy=sequential \
  --priority=high \
  --stages="research,architect,code,review,test"
```

## MCP Protocol Integration

### MCP Server Connection (TCP Port 9500)

The MCP server runs in `multi-agent-container` and provides agent orchestration capabilities via TCP.

**Connection Details:**
- **Host:** `multi-agent-container` (within Docker network)
- **Port:** `9500` (TCP)
- **Protocol:** JSON-RPC 2.0
- **Transport:** TCP with line-delimited JSON

### Establishing MCP Connection

#### Rust Client

```rust
use tokio::net::TcpStream;
use tokio_util::codec::{Framed, LinesCodec};
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};

pub struct MCPClient {
    connection: Framed<TcpStream, LinesCodec>,
    request_id: u64,
}

impl MCPClient {
    pub async fn connect() -> Result<Self> {
        // Connect to MCP TCP server
        let stream = TcpStream::connect("multi-agent-container:9500").await?;
        let mut connection = Framed::new(stream, LinesCodec::new());

        // Initialize connection
        let init_request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {
                    "name": "visionflow",
                    "version": "0.1.0"
                }
            }
        });

        connection.send(init_request.to_string()).await?;

        // Wait for initialization response
        if let Some(Ok(response)) = connection.next().await {
            let parsed: Value = serde_json::from_str(&response)?;
            if parsed.get("result").is_some() {
                return Ok(Self {
                    connection,
                    request_id: 2,
                });
            }
        }

        Err("MCP initialization failed".into())
    }

    pub async fn spawn_agent(
        &mut self,
        agent_type: &str,
        swarm_id: &str,
        config: Value
    ) -> Result<String> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": "agent_spawn",
                "arguments": {
                    "agentType": agent_type,
                    "swarmId": swarm_id,
                    "config": config
                }
            }
        });

        self.request_id += 1;
        self.connection.send(request.to_string()).await?;

        // Wait for response
        if let Some(Ok(response)) = self.connection.next().await {
            let parsed: Value = serde_json::from_str(&response)?;
            if let Some(result) = parsed.get("result") {
                return Ok(result["agentId"].as_str().unwrap().to_string());
            }
        }

        Err("Failed to spawn agent".into())
    }

    pub async fn orchestrate_task(
        &mut self,
        swarm_id: &str,
        task: Value
    ) -> Result<Value> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": "task_orchestrate",
                "arguments": {
                    "swarmId": swarm_id,
                    "task": task
                }
            }
        });

        self.request_id += 1;
        self.connection.send(request.to_string()).await?;

        // Wait for response
        if let Some(Ok(response)) = self.connection.next().await {
            let parsed: Value = serde_json::from_str(&response)?;
            if let Some(result) = parsed.get("result") {
                return Ok(result.clone());
            }
        }

        Err("Task orchestration failed".into())
    }
}
```

#### Python Client

```python
import asyncio
import json
from typing import Dict, Any

class MCPClient:
    def __init__(self):
        self.reader = None
        self.writer = None
        self.request_id = 1

    async def connect(self, host="multi-agent-container", port=9500):
        """Connect to MCP TCP server"""
        self.reader, self.writer = await asyncio.open_connection(host, port)

        # Initialize connection
        init_request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {
                    "name": "visionflow-python",
                    "version": "0.1.0"
                }
            }
        }

        await self._send_request(init_request)
        response = await self._read_response()

        if "result" not in response:
            raise Exception("MCP initialization failed")

        self.request_id += 1

    async def _send_request(self, request: Dict[str, Any]):
        """Send JSON-RPC request"""
        message = json.dumps(request) + "\n"
        self.writer.write(message.encode())
        await self.writer.drain()

    async def _read_response(self) -> Dict[str, Any]:
        """Read JSON-RPC response"""
        data = await self.reader.readline()
        return json.loads(data.decode())

    async def spawn_agent(
        self,
        agent_type: str,
        swarm_id: str,
        config: Dict[str, Any]
    ) -> str:
        """Spawn new agent"""
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": "agent_spawn",
                "arguments": {
                    "agentType": agent_type,
                    "swarmId": swarm_id,
                    "config": config
                }
            }
        }

        self.request_id += 1
        await self._send_request(request)
        response = await self._read_response()

        if "result" in response:
            return response["result"]["agentId"]
        else:
            raise Exception(f"Agent spawn failed: {response.get('error')}")

    async def orchestrate_task(
        self,
        swarm_id: str,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Orchestrate task execution"""
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": "task_orchestrate",
                "arguments": {
                    "swarmId": swarm_id,
                    "task": task
                }
            }
        }

        self.request_id += 1
        await self._send_request(request)
        response = await self._read_response()

        if "result" in response:
            return response["result"]
        else:
            raise Exception(f"Task orchestration failed: {response.get('error')}")

    async def close(self):
        """Close connection"""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()

# Usage example
async def main():
    client = MCPClient()

    try:
        # Connect to MCP server
        await client.connect()

        # Initialize swarm
        swarm_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "swarm_init",
                "arguments": {
                    "topology": "hierarchical",
                    "maxAgents": 10
                }
            }
        }

        # Spawn agents
        agent_id = await client.spawn_agent(
            agent_type="coder",
            swarm_id="swarm_1757880683494_yl81sece5",
            config={
                "model": "claude-3-opus",
                "capabilities": ["python", "rust"],
                "maxTokens": 4096
            }
        )

        # Orchestrate task
        result = await client.orchestrate_task(
            swarm_id="swarm_1757880683494_yl81sece5",
            task={
                "description": "Implement authentication service",
                "priority": "high",
                "requiredCapabilities": ["python", "security"]
            }
        )

        print(f"Task result: {result}")

    finally:
        await client.close()

asyncio.run(main())
```

### MCP Real-World Examples

#### Example 1: Spawning an Agent with Real TCP Connection

**Request:**
```json
{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
        "name": "agent_spawn",
        "arguments": {
            "agentType": "coder",
            "swarmId": "swarm_1757880683494_yl81sece5",
            "config": {
                "model": "claude-3-opus",
                "temperature": 0.7,
                "capabilities": ["python", "rust", "typescript"],
                "maxTokens": 4096,
                "timeout": 300,
                "retryAttempts": 3
            },
            "resources": {
                "cpuLimit": "2000m",
                "memoryLimit": "4Gi",
                "gpuAccess": true
            }
        }
    }
}
```

**Response:**
```json
{
    "jsonrpc": "2.0",
    "id": 3,
    "result": {
        "agentId": "agent_1757967065850_dv2zg7",
        "swarmId": "swarm_1757880683494_yl81sece5",
        "status": "spawning",
        "estimatedReadyTime": "2025-01-22T10:00:30Z",
        "tcpEndpoint": "multi-agent-container:9500",
        "capabilities": ["python", "rust", "typescript"],
        "resources": {
            "allocated": true,
            "cpu": "2000m",
            "memory": "4Gi",
            "gpuDevice": 0
        },
        "connectionPool": {
            "poolId": "pool_123",
            "connections": 3,
            "healthCheck": "passing"
        },
        "initialisationMetrics": {
            "spawnTime": 1247,
            "modelLoadTime": 892,
            "memoryAllocated": "3.2 GB"
        }
    }
}
```

#### Example 2: Task Orchestration with Distributed Execution

**Request:**
```json
{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {
        "name": "task_orchestrate",
        "arguments": {
            "swarmId": "swarm_1757880683494_yl81sece5",
            "task": {
                "description": "Analyse security vulnerabilities in auth module",
                "priority": "high",
                "strategy": "adaptive",
                "timeout": 300,
                "requiredCapabilities": ["security", "code", "review"],
                "parallelism": 3,
                "consensusRequired": true
            },
            "execution": {
                "mode": "distributed",
                "retryPolicy": "exponential_backoff",
                "resultAggregation": "majority_vote",
                "failureThreshold": 0.2
            }
        }
    }
}
```

**Response:**
```json
{
    "jsonrpc": "2.0",
    "id": 4,
    "result": {
        "taskId": "task_1757967065850_abc123",
        "swarmId": "swarm_1757880683494_yl81sece5",
        "status": "orchestrating",
        "assignedAgents": [
            {
                "agentId": "agent_1757967065850_dv2zg7",
                "role": "coordinator",
                "capabilities": ["security", "code"]
            },
            {
                "agentId": "agent_1757967065851_def456",
                "role": "worker",
                "capabilities": ["code", "review"]
            },
            {
                "agentId": "agent_1757967065852_ghi789",
                "role": "validator",
                "capabilities": ["security", "review"]
            }
        ],
        "orchestrationPlan": {
            "phases": [
                {
                    "name": "analysis",
                    "agents": ["agent_1757967065851_def456"],
                    "estimatedDuration": 120
                },
                {
                    "name": "security_scan",
                    "agents": ["agent_1757967065850_dv2zg7"],
                    "estimatedDuration": 180,
                    "dependencies": ["analysis"]
                },
                {
                    "name": "validation",
                    "agents": ["agent_1757967065852_ghi789"],
                    "estimatedDuration": 60,
                    "dependencies": ["security_scan"]
                }
            ]
        },
        "coordination": {
            "consensusThreshold": 0.7,
            "votingMechanism": "weighted",
            "conflictResolution": "expert_priority"
        },
        "estimatedCompletion": "2025-01-22T10:05:00Z",
        "mcpConnections": {
            "active": 3,
            "poolUtilization": 0.6
        }
    }
}
```

See: Reference documentation for MCP protocol details

## Multi-Agent Container Integration

### Container Architecture

The multi-agent environment runs in a dedicated Docker container with full MCP support:

```yaml
services:
  multi-agent:
    container_name: multi-agent-container
    ports:
      - "3000:3000"    # Claude Flow UI
      - "3002:3002"    # WebSocket Bridge
      - "9500:9500"    # MCP TCP Server (PRIMARY)
    environment:
      # MCP Configuration
      - MCP_TCP_PORT=9500
      - MCP_ENABLE_TCP=true
      - MCP_LOG_LEVEL=info

      # Agent Configuration
      - MAX_AGENTS=20
      - AGENT_TIMEOUT=300

      # Connection Pooling
      - MCP_POOL_SIZE=10
      - MCP_RETRY_ATTEMPTS=3

    networks:
      - docker_ragflow
```

### Accessing Multi-Agent Container

```bash
# Enter container shell
docker exec -it multi-agent-container /bin/bash

# Run agent commands
mcp__claude-flow__swarm_init hierarchical --maxAgents=10

# Check MCP server status
mcp-tcp-status

# View agent logs
docker logs -f multi-agent-container
```

### Container-to-Container Communication

Agents in the multi-agent-container can communicate with GUI tools in gui-tools-container:

```python
# Connect to Blender in GUI container
blender_client = MCPClient(
    host="gui-tools-service",
    port=9876
)

# Execute Blender command
await blender_client.execute_code(
    "import bpy; bpy.ops.mesh.primitive_cube_add()"
)
```

See: Multi-agent Docker documentation for container setup

## Inter-Agent Communication

### Communication Patterns

#### 1. Direct Messaging

```python
class Agent:
    async def send_message(self, target_agent_id: str, message: dict):
        """Send direct message to another agent"""
        await self.message_bus.publish(
            channel=f"agent.{target_agent_id}",
            message={
                "from": self.agent_id,
                "to": target_agent_id,
                "type": "direct",
                "content": message,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def handle_message(self, message: dict):
        """Process incoming messages"""
        if message["type"] == "task_request":
            await self.handle_task_request(message["content"])
        elif message["type"] == "status_update":
            await self.handle_status_update(message["content"])
        elif message["type"] == "data_share":
            await self.handle_data_share(message["content"])
```

#### 2. Broadcast Communication

```python
async def broadcast_task(task: Task, agent_filter=None):
    """Broadcast task to multiple agents"""
    message = {
        "type": "task_broadcast",
        "task": task.to_dict(),
        "requirements": task.requirements,
        "timestamp": datetime.utcnow().isoformat()
    }

    if agent_filter:
        channel = f"broadcast.{agent_filter}"
    else:
        channel = "broadcast.all"

    await message_bus.publish(channel, message)
```

#### 3. Event-Based Communication

```python
class EventDrivenAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_handlers = {
            "task.completed": self.on_task_completed,
            "task.failed": self.on_task_failed,
            "agent.joined": self.on_agent_joined,
            "agent.left": self.on_agent_left,
            "error.occurred": self.on_error_occurred
        }

    async def subscribe_to_events(self):
        """Subscribe to relevant events"""
        for event_type in self.event_handlers:
            await self.event_bus.subscribe(
                event_type,
                self.handle_event
            )

    async def handle_event(self, event):
        """Route events to appropriate handlers"""
        handler = self.event_handlers.get(event.type)
        if handler:
            await handler(event)
```

### Message Protocols

```python
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class AgentMessage(BaseModel):
    """Standard agent message format"""
    message_id: str
    from_agent: str
    to_agent: Optional[str]
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str]
    priority: int = 0

class TaskMessage(AgentMessage):
    """Task-specific message"""
    task_id: str
    task_type: str
    deadline: Optional[datetime]

class StatusMessage(AgentMessage):
    """Agent status update"""
    status: str
    metrics: Dict[str, float]
    capabilities: list[str]
```

## Task Distribution

### Task Queue Management

```python
from heapq import heappush, heappop
from dataclasses import dataclass, field
from typing import Optional

@dataclass(order=True)
class PrioritisedTask:
    priority: int
    task: Task = field(compare=False)
    created_at: datetime = field(default_factory=datetime.utcnow, compare=False)

class TaskQueue:
    def __init__(self):
        self.queue = []
        self.task_map = {}
        self.processing = set()

    async def add_task(self, task: Task, priority: int = 0):
        """Add task to queue with priority"""
        prioritised = PrioritisedTask(priority, task)
        heappush(self.queue, prioritised)
        self.task_map[task.id] = task

        # Notify available agents
        await self.notify_agents(task)

    async def get_task(self, agent_capabilities: list[str]) -> Optional[Task]:
        """Get next suitable task for agent"""
        # Find first matching task
        for i, prioritised in enumerate(self.queue):
            task = prioritised.task
            if (task.id not in self.processing and
                self.can_handle(task, agent_capabilities)):
                self.queue.pop(i)
                self.processing.add(task.id)
                return task
        return None

    def can_handle(self, task: Task, capabilities: list[str]) -> bool:
        """Check if agent can handle task"""
        required = set(task.required_capabilities)
        available = set(capabilities)
        return required.issubset(available)
```

### Capability-Based Assignment

```python
class CapabilityMatcher:
    def match_agent_to_task(
        self,
        task: Task,
        agents: list[Agent]
    ) -> Optional[Agent]:
        """Match task to most capable agent"""
        suitable_agents = []

        for agent in agents:
            if agent.is_available and self.has_capabilities(agent, task):
                score = self.calculate_match_score(agent, task)
                suitable_agents.append((agent, score))

        if suitable_agents:
            # Return agent with highest match score
            suitable_agents.sort(key=lambda x: x[1], reverse=True)
            return suitable_agents[0][0]

        return None

    def calculate_match_score(self, agent: Agent, task: Task) -> float:
        """Calculate how well agent matches task"""
        score = 0.0

        # Exact capability matches (40% weight)
        matching_caps = set(agent.capabilities) & set(task.required_capabilities)
        score += (len(matching_caps) / len(task.required_capabilities)) * 40

        # Past performance (30% weight)
        if agent.metrics.get('task_success_rate'):
            score += agent.metrics['task_success_rate'] * 30

        # Current workload (20% weight)
        workload = agent.current_tasks / agent.max_tasks
        score += (1 - workload) * 20

        # Response time (10% weight)
        if agent.metrics.get('avg_response_time'):
            response_score = max(0, 1 - (agent.metrics['avg_response_time'] / 300))
            score += response_score * 10

        return score
```

### Load Balancing

```python
class LoadBalancer:
    def __init__(self):
        self.agent_loads = {}

    async def assign_task(self, task: Task) -> Agent:
        """Assign task using load balancing"""
        available_agents = await self.get_available_agents(task)

        if not available_agents:
            raise NoAvailableAgentError()

        # Find least loaded agent
        min_load = float('inf')
        selected_agent = None

        for agent in available_agents:
            load = self.calculate_load(agent)
            if load < min_load:
                min_load = load
                selected_agent = agent

        # Update load tracking
        self.agent_loads[selected_agent.id] = min_load + task.estimated_duration

        return selected_agent

    def calculate_load(self, agent: Agent) -> float:
        """Calculate agent's current load"""
        base_load = self.agent_loads.get(agent.id, 0)

        # Add current task processing time
        for task in agent.active_tasks:
            base_load += task.remaining_time

        return base_load
```

## Monitoring and Telemetry

### Agent Telemetry System

```python
class AgentTelemetry:
    """Collect and process agent telemetry data"""

    def __init__(self):
        self.metrics = {}
        self.events = []
        self.prometheus_client = PrometheusClient()

    async def collect_metrics(self, agent_id: str):
        """Collect metrics from agent"""
        agent = await self.get_agent(agent_id)

        metrics = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "active_tasks": len(agent.active_tasks),
            "completed_tasks": agent.metrics["completed_tasks"],
            "error_rate": agent.metrics["error_rate"],
            "avg_task_duration": agent.metrics["avg_duration"],
            "success_rate": agent.metrics["success_rate"],
            "timestamp": datetime.utcnow()
        }

        self.metrics[agent_id] = metrics

        # Export to Prometheus
        await self.export_to_prometheus(agent_id, metrics)

        return metrics

    async def export_to_prometheus(self, agent_id: str, metrics: dict):
        """Export metrics to Prometheus"""
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.prometheus_client.gauge(
                    f"agent_{metric_name}",
                    value,
                    labels={"agent_id": agent_id}
                )
```

### Real-time Monitoring

```python
class RealtimeMonitor:
    """Real-time agent monitoring via WebSocket"""

    def __init__(self, websocket_url: str):
        self.ws_url = websocket_url
        self.connections = {}
        self.alert_thresholds = {
            "error_rate": 0.1,
            "cpu_usage": 90,
            "memory_usage": 90,
            "response_time": 10
        }

    async def monitor_agent(self, agent_id: str, callback):
        """Monitor specific agent in real-time"""
        ws = await websockets.connect(f"{self.ws_url}/agent/{agent_id}")

        try:
            async for message in ws:
                data = json.loads(message)
                await callback(data)

                # Check for alerts
                if self.should_alert(data):
                    await self.send_alert(agent_id, data)
        finally:
            await ws.close()

    def should_alert(self, data: dict) -> bool:
        """Check if metrics warrant an alert"""
        return any([
            data.get("error_rate", 0) > self.alert_thresholds["error_rate"],
            data.get("cpu_usage", 0) > self.alert_thresholds["cpu_usage"],
            data.get("memory_usage", 0) > self.alert_thresholds["memory_usage"],
            data.get("response_time", 0) > self.alert_thresholds["response_time"]
        ])

    async def send_alert(self, agent_id: str, data: dict):
        """Send alert notification"""
        alert = {
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": data,
            "severity": self.calculate_severity(data)
        }

        # Send to alerting system
        await self.alerting_system.send_alert(alert)
```

### Monitoring Dashboard

```bash
# Access Grafana dashboard
docker exec -it multi-agent-container grafana-server

# View agent metrics
curl http://localhost:3000/api/agents/metrics

# Generate performance report
docker exec multi-agent-container mcp__claude-flow__performance_report \
  --format=json \
  --timeframe=24h
```

## Error Handling and Recovery

### Failure Detection

```python
class FailureDetector:
    """Detect and handle agent failures"""

    def __init__(self, heartbeat_interval: int = 30):
        self.heartbeat_interval = heartbeat_interval
        self.last_heartbeat = {}
        self.failure_handlers = {}
        self.failure_history = defaultdict(list)

    async def monitor_heartbeats(self):
        """Monitor agent heartbeats"""
        while True:
            current_time = time.time()

            for agent_id, last_beat in self.last_heartbeat.items():
                time_since_heartbeat = current_time - last_beat

                if time_since_heartbeat > self.heartbeat_interval * 2:
                    # Agent may have failed
                    await self.handle_potential_failure(agent_id)

            await asyncio.sleep(self.heartbeat_interval / 2)

    async def handle_potential_failure(self, agent_id: str):
        """Handle potential agent failure"""
        # Verify failure with multiple checks
        if await self.verify_failure(agent_id):
            logger.warning(f"Agent {agent_id} failed")

            # Record failure
            self.failure_history[agent_id].append({
                "timestamp": time.time(),
                "type": "heartbeat_timeout"
            })

            # Get agent's active tasks
            agent = await self.get_agent(agent_id)
            active_tasks = agent.active_tasks if agent else []

            # Mark agent as failed
            await self.mark_agent_failed(agent_id)

            # Reassign tasks
            for task in active_tasks:
                await self.reassign_task(task)

            # Attempt recovery
            await self.attempt_recovery(agent_id)

    async def verify_failure(self, agent_id: str) -> bool:
        """Verify agent failure with multiple checks"""
        checks = [
            self.check_heartbeat(agent_id),
            self.check_tcp_connection(agent_id),
            self.check_process_status(agent_id)
        ]

        results = await asyncio.gather(*checks, return_exceptions=True)

        # Fail if majority of checks indicate failure
        failures = sum(1 for r in results if isinstance(r, Exception) or not r)
        return failures >= len(checks) // 2 + 1
```

### Recovery Strategies

```python
class RecoveryManager:
    """Manage agent recovery strategies"""

    async def recover_agent(self, agent_id: str, failure_type: str):
        """Recover failed agent based on failure type"""

        strategies = {
            "timeout": self.recover_from_timeout,
            "crash": self.recover_from_crash,
            "resource_exhaustion": self.recover_from_resource_exhaustion,
            "network_failure": self.recover_from_network_failure,
            "deadlock": self.recover_from_deadlock
        }

        recovery_fn = strategies.get(failure_type, self.default_recovery)
        return await recovery_fn(agent_id)

    async def recover_from_timeout(self, agent_id: str):
        """Recover from timeout failure"""
        # Kill stuck processes
        await self.kill_agent_processes(agent_id)

        # Restart agent with increased timeout
        agent_config = await self.get_agent_config(agent_id)
        agent_config["timeout"] *= 1.5  # Increase timeout by 50%

        agent = await self.restart_agent(agent_id, agent_config)

        # Restore state
        await self.restore_agent_state(agent)

        return agent

    async def recover_from_crash(self, agent_id: str):
        """Recover from agent crash"""
        # Analyse crash logs
        crash_info = await self.analyse_crash_logs(agent_id)

        # Fix configuration if needed
        if crash_info.get("config_error"):
            await self.fix_agent_config(agent_id, crash_info)

        # Restart with backoff
        return await self.restart_with_backoff(agent_id)

    async def recover_from_resource_exhaustion(self, agent_id: str):
        """Recover from resource exhaustion"""
        # Increase resource limits
        agent_config = await self.get_agent_config(agent_id)
        agent_config["resources"]["memory"] = str(
            int(agent_config["resources"]["memory"].rstrip("Gi")) * 1.5
        ) + "Gi"

        # Restart with increased resources
        return await self.restart_agent(agent_id, agent_config)

    async def restart_with_backoff(
        self,
        agent_id: str,
        max_attempts: int = 3
    ):
        """Restart agent with exponential backoff"""
        delay = 1

        for attempt in range(max_attempts):
            try:
                agent = await self.restart_agent(agent_id)
                return agent
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise

                logger.warning(
                    f"Restart attempt {attempt + 1} failed for {agent_id}: {e}"
                )
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
```

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    """Circuit breaker for agent operations"""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = defaultdict(int)
        self.last_failure_time = {}
        self.state = defaultdict(lambda: "closed")  # closed, open, half_open

    async def call(self, agent_id: str, operation: Callable):
        """Execute operation with circuit breaker protection"""
        if self.state[agent_id] == "open":
            # Check if timeout has elapsed
            if time.time() - self.last_failure_time[agent_id] > self.timeout:
                self.state[agent_id] = "half_open"
            else:
                raise CircuitOpenError(
                    f"Circuit open for agent {agent_id}. "
                    f"Retry after {self.timeout}s"
                )

        try:
            result = await operation()

            # Success - reset failures
            if self.state[agent_id] == "half_open":
                self.state[agent_id] = "closed"
            self.failures[agent_id] = 0

            return result

        except Exception as e:
            self.failures[agent_id] += 1
            self.last_failure_time[agent_id] = time.time()

            # Open circuit if threshold exceeded
            if self.failures[agent_id] >= self.failure_threshold:
                self.state[agent_id] = "open"
                logger.error(
                    f"Circuit opened for agent {agent_id} "
                    f"after {self.failures[agent_id]} failures"
                )

            raise e
```

## Troubleshooting Agent Failures

### Common Failure Scenarios

#### 1. Agent Not Responding

**Symptoms:**
- Heartbeat timeouts
- Tasks stuck in processing state
- No response to messages

**Diagnosis:**
```bash
# Check agent status
docker exec multi-agent-container mcp__claude-flow__swarm_status

# View agent logs
docker logs multi-agent-container | grep "agent_[agent_id]"

# Check TCP connection
docker exec multi-agent-container nc -zv multi-agent-container 9500

# Verify MCP server status
docker exec multi-agent-container mcp-tcp-status
```

**Resolution:**
```bash
# Restart specific agent
docker exec multi-agent-container mcp__claude-flow__agent_restart \
  --agentId="agent_1757967065850_dv2zg7"

# If restart fails, respawn agent
docker exec multi-agent-container mcp__claude-flow__agent_spawn \
  coder \
  --swarmId="swarm_1757880683494_yl81sece5" \
  --config='{"capabilities":["python","rust"]}'
```

#### 2. Task Queue Backlog

**Symptoms:**
- Growing queue depth
- Increasing task wait times
- Agent pool utilisation below 50%

**Diagnosis:**
```bash
# Check queue status
docker exec multi-agent-container mcp__claude-flow__swarm_status | \
  jq '.taskQueue'

# Check agent workload distribution
docker exec multi-agent-container mcp__claude-flow__performance_report \
  --format=json | jq '.agents[] | {id, active_tasks, utilization}'
```

**Resolution:**
```bash
# Clear stuck tasks
docker exec multi-agent-container mcp__claude-flow__task_queue_clear \
  --filter="status:stuck,age:>3600"

# Scale up agent pool
docker exec multi-agent-container mcp__claude-flow__agent_scale \
  --count=5 \
  --agentType="coder"

# Rebalance workload
docker exec multi-agent-container mcp__claude-flow__load_balance \
  --strategy="even_distribution"
```

#### 3. High Error Rate

**Symptoms:**
- Frequent task failures
- Increased retry attempts
- Error rate > 10%

**Diagnosis:**
```bash
# View error statistics
docker exec multi-agent-container mcp__claude-flow__performance_report \
  --format=json | jq '.errorMetrics'

# Analyse error patterns
docker logs multi-agent-container 2>&1 | \
  grep -i "error\|exception" | \
  awk '{print $5}' | \
  sort | uniq -c | sort -rn

# Check resource constraints
docker stats multi-agent-container --no-stream
```

**Resolution:**
```python
# Implement error recovery
async def handle_high_error_rate(agent_id: str):
    # Get error history
    errors = await get_agent_errors(agent_id)

    # Analyse error patterns
    error_types = Counter(e["type"] for e in errors)

    # Apply appropriate recovery strategy
    if error_types.most_common(1)[0][0] == "timeout":
        # Increase timeouts
        await update_agent_config(agent_id, {"timeout": 600})
    elif error_types.most_common(1)[0][0] == "resource_exhaustion":
        # Increase resources
        await scale_agent_resources(agent_id, {"memory": "8Gi"})
    else:
        # General recovery: restart agent
        await restart_agent(agent_id)
```

#### 4. Memory Leaks

**Symptoms:**
- Gradually increasing memory usage
- OOM kills
- Degraded performance over time

**Diagnosis:**
```bash
# Monitor memory trends
docker stats multi-agent-container --format \
  "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Check for memory leaks in agent processes
docker exec multi-agent-container ps aux --sort=-rss | head -10

# Profile agent memory usage
docker exec multi-agent-container python -m memory_profiler agent.py
```

**Resolution:**
```python
# Implement periodic agent recycling
class AgentRecycler:
    async def recycle_agents(self):
        """Periodically recycle agents to prevent memory leaks"""
        while True:
            agents = await self.get_all_agents()

            for agent in agents:
                # Check memory usage
                if agent.memory_usage > 0.9 * agent.memory_limit:
                    # Gracefully drain tasks
                    await agent.stop_accepting_tasks()
                    await agent.wait_for_tasks_complete()

                    # Respawn fresh agent
                    new_agent = await self.spawn_agent(
                        agent_type=agent.type,
                        config=agent.config
                    )

                    # Terminate old agent
                    await agent.terminate()

            # Check every hour
            await asyncio.sleep(3600)
```

#### 5. Network Partitioning

**Symptoms:**
- Agents unable to communicate
- Inconsistent swarm state
- Split-brain scenarios

**Diagnosis:**
```bash
# Check network connectivity
docker exec multi-agent-container ping -c 3 gui-tools-service

# Verify MCP connectivity between containers
docker exec multi-agent-container nc -zv gui-tools-service 9876

# Check Docker network status
docker network inspect docker_ragflow
```

**Resolution:**
```bash
# Restart Docker network
docker network disconnect docker_ragflow multi-agent-container
docker network connect docker_ragflow multi-agent-container

# Reconnect agents to MCP server
docker exec multi-agent-container mcp__claude-flow__swarm_reconnect \
  --swarmId="swarm_1757880683494_yl81sece5"

# Verify connectivity restored
docker exec multi-agent-container mcp-tcp-test
```

### Diagnostic Commands Reference

```bash
# Agent Status
mcp__claude-flow__swarm_status              # View swarm and agent status
mcp__claude-flow__agent_health --agentId=X  # Check specific agent health
mcp__claude-flow__performance_report        # Generate performance report

# Network Diagnostics
nc -zv multi-agent-container 9500           # Test TCP connectivity
mcp-tcp-status                              # Check MCP server status
docker network inspect docker_ragflow      # Inspect Docker network

# Resource Monitoring
docker stats multi-agent-container          # Real-time resource usage
docker logs -f multi-agent-container        # Follow logs in real-time
htop                                        # Interactive process viewer

# Troubleshooting
mcp__claude-flow__debug_enable              # Enable debug logging
mcp__claude-flow__agent_restart --agentId=X # Restart specific agent
mcp__claude-flow__swarm_reset               # Reset swarm state
```

## Production Best Practices

### 1. Agent Design Principles

- **Single Responsibility**: Each agent has one clear purpose
- **Stateless Operations**: Avoid storing state; use external state management
- **Idempotency**: Operations can be safely retried
- **Graceful Degradation**: Handle partial failures elegantly
- **Observable**: Emit comprehensive metrics and logs

### 2. Scaling Guidelines

```yaml
# Horizontal Scaling
min_agents_per_type:
  coordinator: 1
  researcher: 2
  architect: 1
  coder: 3
  tester: 2
  reviewer: 1

max_agents_per_type:
  coordinator: 2
  researcher: 5
  architect: 2
  coder: 10
  tester: 5
  reviewer: 3

# Vertical Scaling
resource_limits:
  coordinator:
    memory: "4Gi"
    cpu: "2.0"
  researcher:
    memory: "2Gi"
    cpu: "1.0"
  coder:
    memory: "4Gi"
    cpu: "2.0"
  architect:
    memory: "3Gi"
    cpu: "1.5"
  tester:
    memory: "3Gi"
    cpu: "1.5"
  reviewer:
    memory: "2Gi"
    cpu: "1.0"
```

### 3. Performance Optimisation

```python
# Efficient task batching
async def batch_process_tasks(agent, tasks: list[Task]):
    """Process multiple tasks efficiently"""
    # Group similar tasks
    task_groups = defaultdict(list)
    for task in tasks:
        task_groups[task.type].append(task)

    results = []
    for task_type, group in task_groups.items():
        # Process group in parallel if supported
        if agent.supports_batch(task_type):
            result = await agent.batch_process(group)
            results.extend(result)
        else:
            # Sequential processing with concurrency limit
            for batch in chunks(group, size=5):
                batch_results = await asyncio.gather(*[
                    agent.process_task(task) for task in batch
                ])
                results.extend(batch_results)

    return results
```

### 4. Monitoring Best Practices

- **Track Key Metrics**: Throughput, latency, error rate, resource utilisation
- **Set Up Alerts**: Anomaly detection for critical metrics
- **Log Structured Data**: Use JSON logging for analysis
- **Distributed Tracing**: Track requests across agent boundaries

### 5. Security Best Practices

- **Authentication**: Use tokens for MCP connections
- **Authorisation**: Implement role-based access control
- **Input Validation**: Sanitise all agent inputs
- **Rate Limiting**: Prevent abuse with rate limits
- **Audit Logging**: Log all security-relevant events

## Related Documentation

- [Development Workflow](./development-workflow.md)
- [Configuration Guide](./configuration.md)
- [Agent Control Panel](./agent-orchestration.md)
- [Reference Documentation](../reference/README.md)

---

*[Back to Guides](index.md) | [Development Workflow →](development-workflow.md)*
