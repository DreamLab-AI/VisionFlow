# Swarm Coordination Agents

*[Reference](../index.md) > [Agents](../../reference/agents/index.md) > [Swarm](../reference/agents/swarm/index.md)*

This directory contains specialised swarm coordination agents designed to work with the claude-code-flow hive-mind system. Each agent implements a different coordination topology and strategy.

## Available Agents

### 1. Hierarchical Coordinator (`hierarchical-coordinator.md`)
**Architecture**: Queen-led hierarchy with specialised workers
- **Use Cases**: Complex projects requiring central coordination
- **Strengths**: Clear command structure, efficient resource allocation
- **Best For**: Large-scale development, multi-team coordination

### 2. Mesh Coordinator (`mesh-coordinator.md`) 
**Architecture**: Peer-to-peer distributed network
- **Use Cases**: Fault-tolerant distributed processing
- **Strengths**: High resilience, no single point of failure
- **Best For**: Critical systems, high-availability requirements

### 3. Adaptive Coordinator (`adaptive-coordinator.md`)
**Architecture**: Dynamic topology switching with ML optimisation
- **Use Cases**: Variable workloads requiring optimisation
- **Strengths**: Self-optimising, learns from experience
- **Best For**: Production systems, long-running processes

## Coordination Patterns

### Topology Comparison

| Feature | Hierarchical | Mesh | Adaptive |
|---------|-------------|------|----------|
| **Fault Tolerance** | Medium | High | High |
| **Scalability** | High | Medium | High |
| **Coordination Overhead** | Low | High | Variable |
| **Learning Capability** | Low | Low | High |
| **Setup Complexity** | Low | High | Medium |
| **Best Use Case** | Structured projects | Critical systems | Variable workloads |

### Performance Characteristics

```
Hierarchical: ⭐⭐⭐⭐⭐ Coordination Efficiency
              ⭐⭐⭐⭐   Fault Tolerance  
              ⭐⭐⭐⭐⭐ Scalability

Mesh:         ⭐⭐⭐     Coordination Efficiency
              ⭐⭐⭐⭐⭐ Fault Tolerance
              ⭐⭐⭐     Scalability

Adaptive:     ⭐⭐⭐⭐⭐ Coordination Efficiency  
              ⭐⭐⭐⭐⭐ Fault Tolerance
              ⭐⭐⭐⭐⭐ Scalability
```

## MCP Tool Integration

All swarm coordinators leverage the following MCP tools:

### Core Coordination Tools
- `mcp__claude-flow__swarm_init` - Initialize swarm topology
- `mcp__claude-flow__agent_spawn` - Create specialised worker agents  
- `mcp__claude-flow__task_orchestrate` - Coordinate complex workflows
- `mcp__claude-flow__swarm_monitor` - Real-time performance monitoring

### Advanced Features
- `mcp__claude-flow__neural_patterns` - Pattern recognition and learning
- `mcp__claude-flow__daa_consensus` - Distributed decision making
- `mcp__claude-flow__topology_optimize` - Dynamic topology optimisation
- `mcp__claude-flow__performance_report` - Comprehensive analytics

## Usage Examples

### Hierarchical Coordination
```bash
# Initialize hierarchical swarm for development project
claude-flow agent spawn hierarchical-coordinator "Build authentication microservice"

# Agents will automatically:
# 1. Decompose project into tasks
# 2. Spawn specialised workers (research, code, test, docs)
# 3. Coordinate execution with central oversight
# 4. Generate comprehensive reports
```

### Mesh Coordination  
```bash
# Initialize mesh network for distributed processing
claude-flow agent spawn mesh-coordinator "Process user analytics data"

# Network will automatically:
# 1. Establish peer-to-peer connections
# 2. Distribute work across available nodes
# 3. Handle node failures gracefully
# 4. Maintain consensus on results
```

### Adaptive Coordination
```bash
# Initialize adaptive swarm for production optimisation
claude-flow agent spawn adaptive-coordinator "Optimise system performance"

# System will automatically:
# 1. Analyze current workload patterns
# 2. Select optimal topology (hierarchical/mesh/ring)
# 3. Learn from performance outcomes
# 4. Continuously adapt to changing conditions
```

## Architecture Decision Framework

### When to Use Hierarchical
- ✅ Well-defined project structure
- ✅ Clear resource hierarchy 
- ✅ Need for centralized decision making
- ✅ Large team coordination required
- ❌ High fault tolerance critical
- ❌ Network partitioning likely

### When to Use Mesh
- ✅ High availability requirements
- ✅ Distributed processing needs
- ✅ Network reliability concerns
- ✅ Peer collaboration model
- ❌ Simple coordination sufficient
- ❌ Resource constraints exist

### When to Use Adaptive
- ✅ Variable workload patterns
- ✅ Long-running production systems
- ✅ Performance optimisation critical
- ✅ Machine learning acceptable
- ❌ Predictable, stable workloads
- ❌ Simple requirements

## Performance Monitoring

Each coordinator provides comprehensive metrics:

### Key Performance Indicators
- **Task Completion Rate**: Percentage of successful task completion
- **Agent Utilization**: Efficiency of resource usage
- **Coordination Overhead**: Communication and management costs
- **Fault Recovery Time**: Speed of recovery from failures
- **Learning Convergence**: Adaptation effectiveness (adaptive only)

### Monitoring Dashboards
Real-time visibility into:
- Swarm topology and agent status
- Task queues and execution pipelines  
- Performance metrics and trends
- Error rates and failure patterns
- Resource utilization and capacity

## Best Practices

### Design Principles
1. **Start Simple**: Begin with hierarchical for well-understood problems
2. **Scale Gradually**: Add complexity as requirements grow
3. **Monitor Continuously**: Track performance and adapt strategies
4. **Plan for Failure**: Design fault tolerance from the beginning

### Operational Guidelines
1. **Agent Sizing**: Right-size swarms for workload (5-15 agents typical)
2. **Resource Planning**: Ensure adequate compute/memory for coordination overhead
3. **Network Design**: Consider latency and bandwidth for distributed topologies
4. **Security**: Implement proper authentication and authorization

### Troubleshooting
- **Poor Performance**: Check agent capability matching and load distribution
- **Coordination Failures**: Verify network connectivity and consensus thresholds
- **Resource Exhaustion**: Monitor and scale agent pools proactively
- **Learning Issues**: Validate training data quality and model convergence

## Integration with Claude-Flow

These agents integrate seamlessly with the broader claude-flow ecosystem:

- **Memory System**: All coordination state persisted in claude-flow memory bank
- **Terminal Management**: Agents can spawn and manage multiple terminal sessions
- **MCP Integration**: Full access to claude-flow's MCP tool ecosystem
- **Event System**: Real-time coordination through claude-flow event bus
- **Configuration**: Managed through claude-flow configuration system

For implementation details, see individual agent files and the claude-flow documentation.

## Related Topics

- [Agent Orchestration Architecture](../../../features/agent-orchestration.md)
- [Agent Type Conventions and Mapping](../../../AGENT_TYPE_CONVENTIONS.md)
- [Agent Visualisation Architecture](../../../agent-visualization-architecture.md)
- [Agentic Alliance](../../../archive/legacy/old_markdown/Agentic Alliance.md)
- [Agentic Metaverse for Global Creatives](../../../archive/legacy/old_markdown/Agentic Metaverse for Global Creatives.md)
- [Agentic Mycelia](../../../archive/legacy/old_markdown/Agentic Mycelia.md)
- [Agents](../../../archive/legacy/old_markdown/Agents.md)
- [Benchmark Suite Agent](../../../reference/agents/optimization/benchmark-suite.md)
- [Claude Code Agents Directory Structure](../../../reference/agents/README.md)
- [Claude Flow Commands to Agent System Migration Summary](../../../reference/agents/MIGRATION_SUMMARY.md)
- [Distributed Consensus Builder Agents](../../../reference/agents/consensus/README.md)
- [Financialised Agentic Memetics](../../../archive/legacy/old_markdown/Financialised Agentic Memetics.md)
- [Load Balancing Coordinator Agent](../../../reference/agents/optimization/load-balancer.md)
- [Multi Agent Orchestration](../../../server/agent-swarm.md)
- [Multi Agent RAG scrapbook](../../../archive/legacy/old_markdown/Multi Agent RAG scrapbook.md)
- [Multi-Agent Container Setup](../../../deployment/multi-agent-setup.md)
- [Multi-MCP Agent Visualisation System](../../../MCP_AGENT_VISUALIZATION.md)
- [Multi-MCP Agent Visualisation System](../../../multi-mcp-agent-visualization.md)
- [Performance Monitor Agent](../../../reference/agents/optimization/performance-monitor.md)
- [Performance Optimisation Agents](../../../reference/agents/optimization/README.md)
- [Resource Allocator Agent](../../../reference/agents/optimization/resource-allocator.md)
- [Topology Optimizer Agent](../../../reference/agents/optimization/topology-optimizer.md)
- [adaptive-coordinator](../../../reference/agents/swarm/adaptive-coordinator.md)
- [analyze-code-quality](../../../reference/agents/analysis/code-review/analyze-code-quality.md)
- [arch-system-design](../../../reference/agents/architecture/system-design/arch-system-design.md)
- [architecture](../../../reference/agents/sparc/architecture.md)
- [automation-smart-agent](../../../reference/agents/templates/automation-smart-agent.md)
- [base-template-generator](../../../reference/agents/base-template-generator.md)
- [byzantine-coordinator](../../../reference/agents/consensus/byzantine-coordinator.md)
- [code-analyzer](../../../reference/agents/analysis/code-analyzer.md)
- [code-review-swarm](../../../reference/agents/github/code-review-swarm.md)
- [coder](../../../reference/agents/core/coder.md)
- [coordinator-swarm-init](../../../reference/agents/templates/coordinator-swarm-init.md)
- [crdt-synchronizer](../../../reference/agents/consensus/crdt-synchronizer.md)
- [data-ml-model](../../../reference/agents/data/ml/data-ml-model.md)
- [dev-backend-api](../../../reference/agents/development/backend/dev-backend-api.md)
- [docs-api-openapi](../../../reference/agents/documentation/api-docs/docs-api-openapi.md)
- [github-modes](../../../reference/agents/github/github-modes.md)
- [github-pr-manager](../../../reference/agents/templates/github-pr-manager.md)
- [gossip-coordinator](../../../reference/agents/consensus/gossip-coordinator.md)
- [hierarchical-coordinator](../../../reference/agents/swarm/hierarchical-coordinator.md)
- [implementer-sparc-coder](../../../reference/agents/templates/implementer-sparc-coder.md)
- [issue-tracker](../../../reference/agents/github/issue-tracker.md)
- [memory-coordinator](../../../reference/agents/templates/memory-coordinator.md)
- [mesh-coordinator](../../../reference/agents/swarm/mesh-coordinator.md)
- [migration-plan](../../../reference/agents/templates/migration-plan.md)
- [multi-repo-swarm](../../../reference/agents/github/multi-repo-swarm.md)
- [ops-cicd-github](../../../reference/agents/devops/ci-cd/ops-cicd-github.md)
- [orchestrator-task](../../../reference/agents/templates/orchestrator-task.md)
- [performance-analyzer](../../../reference/agents/templates/performance-analyzer.md)
- [performance-benchmarker](../../../reference/agents/consensus/performance-benchmarker.md)
- [planner](../../../reference/agents/core/planner.md)
- [pr-manager](../../../reference/agents/github/pr-manager.md)
- [production-validator](../../../reference/agents/testing/validation/production-validator.md)
- [project-board-sync](../../../reference/agents/github/project-board-sync.md)
- [pseudocode](../../../reference/agents/sparc/pseudocode.md)
- [quorum-manager](../../../reference/agents/consensus/quorum-manager.md)
- [raft-manager](../../../reference/agents/consensus/raft-manager.md)
- [refinement](../../../reference/agents/sparc/refinement.md)
- [release-manager](../../../reference/agents/github/release-manager.md)
- [release-swarm](../../../reference/agents/github/release-swarm.md)
- [repo-architect](../../../reference/agents/github/repo-architect.md)
- [researcher](../../../reference/agents/core/researcher.md)
- [reviewer](../../../reference/agents/core/reviewer.md)
- [security-manager](../../../reference/agents/consensus/security-manager.md)
- [sparc-coordinator](../../../reference/agents/templates/sparc-coordinator.md)
- [spec-mobile-react-native](../../../reference/agents/specialized/mobile/spec-mobile-react-native.md)
- [specification](../../../reference/agents/sparc/specification.md)
- [swarm-issue](../../../reference/agents/github/swarm-issue.md)
- [swarm-pr](../../../reference/agents/github/swarm-pr.md)
- [sync-coordinator](../../../reference/agents/github/sync-coordinator.md)
- [tdd-london-swarm](../../../reference/agents/testing/unit/tdd-london-swarm.md)
- [tester](../../../reference/agents/core/tester.md)
- [workflow-automation](../../../reference/agents/github/workflow-automation.md)
