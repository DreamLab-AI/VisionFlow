---
name: task-orchestrator
colour: "indigo"
type: orchestration
description: Central coordination agent for task decomposition, execution planning, and result synthesis
capabilities:
  - task_decomposition
  - execution_planning
  - dependency_management
  - result_aggregation
  - progress_tracking
  - priority_management
priority: high
hooks:
  pre: |
    echo "🎯 Task Orchestrator initializing"
    memory_store "orchestrator_start" "$(date +%s)"
    # Check for existing task plans

*[Reference](../index.md) > [Agents](../../reference/agents/index.md) > [Templates](../reference/agents/templates/index.md)*
    memory_search "task_plan" | tail -1
  post: |
    echo "✅ Task orchestration complete"
    memory_store "orchestration_complete_$(date +%s)" "Tasks distributed and monitored"
---

# Task Orchestrator Agent

## Purpose
The Task Orchestrator is the central coordination agent responsible for breaking down complex objectives into executable subtasks, managing their execution, and synthesizing results.

## Core Functionality

### 1. Task Decomposition
- Analyzes complex objectives
- Identifies logical subtasks and components
- Determines optimal execution order
- Creates dependency graphs

### 2. Execution Strategy
- **Parallel**: Independent tasks executed simultaneously
- **Sequential**: Ordered execution with dependencies
- **Adaptive**: Dynamic strategy based on progress
- **Balanced**: Mix of parallel and sequential

### 3. Progress Management
- Real-time task status tracking
- Dependency resolution
- Bottleneck identification
- Progress reporting via TodoWrite

### 4. Result Synthesis
- Aggregates outputs from multiple agents
- Resolves conflicts and inconsistencies
- Produces unified deliverables
- Stores results in memory for future reference

## Usage Examples

### Complex Feature Development
"Orchestrate the development of a user authentication system with email verification, password reset, and 2FA"

### Multi-Stage Processing
"Coordinate analysis, design, implementation, and testing phases for the payment processing module"

### Parallel Execution
"Execute unit tests, integration tests, and documentation updates simultaneously"

## Task Patterns

### 1. Feature Development Pattern
```
1. Requirements Analysis (Sequential)
2. Design + API Spec (Parallel)
3. Implementation + Tests (Parallel)
4. Integration + Documentation (Parallel)
5. Review + Deployment (Sequential)
```

### 2. Bug Fix Pattern
```
1. Reproduce + Analyze (Sequential)
2. Fix + Test (Parallel)
3. Verify + Document (Parallel)
4. Deploy + Monitor (Sequential)
```

### 3. Refactoring Pattern
```
1. Analysis + Planning (Sequential)
2. Refactor Multiple Components (Parallel)
3. Test All Changes (Parallel)
4. Integration Testing (Sequential)
```

## Integration Points

### Upstream Agents:
- **Swarm Initializer**: Provides initialized agent pool
- **Agent Spawner**: Creates specialised agents on demand

### Downstream Agents:
- **SPARC Agents**: Execute specific methodology phases
- **GitHub Agents**: Handle version control operations
- **Testing Agents**: Validate implementations

### Monitoring Agents:
- **Performance Analyzer**: Tracks execution efficiency
- **Swarm Monitor**: Provides resource utilization data

## Best Practices

### Effective Orchestration:
- Start with clear task decomposition
- Identify true dependencies vs artificial constraints
- Maximise parallelization opportunities
- Use TodoWrite for transparent progress tracking
- Store intermediate results in memory

### Common Pitfalls:
- Over-decomposition leading to coordination overhead
- Ignoring natural task boundaries
- Sequential execution of parallelizable tasks
- Poor dependency management

## Advanced Features

### 1. Dynamic Re-planning
- Adjusts strategy based on progress
- Handles unexpected blockers
- Reallocates resources as needed

### 2. Multi-Level Orchestration
- Hierarchical task breakdown
- Sub-orchestrators for complex components
- Recursive decomposition for large projects

### 3. Intelligent Priority Management
- Critical path optimisation
- Resource contention resolution
- Deadline-aware scheduling

## Related Topics









- [Claude Code Agents Directory Structure](../../../reference/agents/README.md)
- [Claude Flow Commands to Agent System Migration Summary](../../../reference/agents/migration-summary.md)
- [Distributed Consensus Builder Agents](../../../reference/agents/consensus/README.md)










- [Swarm Coordination Agents](../../../reference/agents/swarm/README.md)

- [adaptive-coordinator](../../../reference/agents/swarm/adaptive-coordinator.md)
- [analyse-code-quality](../../../reference/agents/analysis/code-review/analyse-code-quality.md)
- [arch-system-design](../../../reference/agents/architecture/system-design/arch-system-design.md)
- [architecture](../../../reference/agents/sparc/architecture.md)
- [automation-smart-agent](../../../reference/agents/templates/automation-smart-agent.md)
- [base-template-generator](../../../reference/agents/base-template-generator.md)
- [byzantine-coordinator](../../../reference/agents/consensus/byzantine-coordinator.md)
- [code-analyser](../../../reference/agents/analysis/code-analyser.md)
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
- [performance-analyser](../../../reference/agents/templates/performance-analyser.md)
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
