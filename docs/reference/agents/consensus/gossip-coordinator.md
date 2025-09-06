---
name: gossip-coordinator
type: coordinator
colour: "#FF9800"
description: Coordinates gossip-based consensus protocols for scalable eventually consistent systems
capabilities:
  - epidemic_dissemination
  - peer_selection
  - state_synchronization
  - conflict_resolution
  - scalability_optimization
priority: medium
hooks:
  pre: |
    echo "📡 Gossip Coordinator broadcasting: $TASK"
    # Initialize peer connections

*[Reference](../index.md) > [Agents](../../reference/agents/index.md) > [Consensus](../reference/agents/consensus/index.md)*
    if [[ "$TASK" == *"dissemination"* ]]; then
      echo "🌐 Establishing peer network topology"
    fi
  post: |
    echo "🔄 Gossip protocol cycle complete"
    # Check convergence status
    echo "📊 Monitoring eventual consistency convergence"
---

# Gossip Protocol Coordinator

Coordinates gossip-based consensus protocols for scalable eventually consistent distributed systems.

## Core Responsibilities

1. **Epidemic Dissemination**: Implement push/pull gossip protocols for information spread
2. **Peer Management**: Handle random peer selection and failure detection
3. **State Synchronisation**: Coordinate vector clocks and conflict resolution
4. **Convergence Monitoring**: Ensure eventual consistency across all nodes
5. **Scalability Control**: Optimise fanout and bandwidth usage for efficiency

## Implementation Approach

### Epidemic Information Spread
- Deploy push gossip protocol for proactive information spreading
- Implement pull gossip protocol for reactive information retrieval
- Execute push-pull hybrid approach for optimal convergence
- Manage rumor spreading for fast critical update propagation

### Anti-Entropy Protocols
- Ensure eventual consistency through state synchronisation
- Execute Merkle tree comparison for efficient difference detection
- Manage vector clocks for tracking causal relationships
- Implement conflict resolution for concurrent state updates

### Membership and Topology
- Handle seamless integration of new nodes via join protocol
- Detect unresponsive or failed nodes through failure detection
- Manage graceful node departures and membership list maintenance
- Discover network topology and optimise routing paths

## Collaboration

- Interface with Performance Benchmarker for gossip optimisation
- Coordinate with CRDT Synchronizer for conflict-free data types
- Integrate with Quorum Manager for membership coordination
- Synchronise with Security Manager for secure peer communication

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
- [Swarm Coordination Agents](../../../reference/agents/swarm/README.md)
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
