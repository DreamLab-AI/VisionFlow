---
name: migration-planner
type: planning
colour: red
description: Comprehensive migration plan for converting commands to agent-based system
capabilities:
  - migration-planning
  - system-transformation
  - agent-mapping
  - compatibility-analysis
  - rollout-coordination
priority: medium
hooks:
  pre: |
    echo "📋 Agent System Migration Planner activated"
    echo "🔄 Analyzing current command structure for migration"
    # Check existing command structure

*[Reference](../index.md) > [Agents](../../reference/agents/index.md) > [Templates](../reference/agents/templates/index.md)*
    if [ -d ".claude/commands" ]; then
      echo "📁 Found existing command directory - will map to agents"
      find .claude/commands -name "*.md" | wc -l | xargs echo "Commands to migrate:"
    fi
  post: |
    echo "✅ Migration planning completed"
    echo "📊 Agent mapping strategy defined"
    echo "🚀 Ready for systematic agent system rollout"
---

# Claude Flow Commands to Agent System Migration Plan

## Overview
This document provides a comprehensive migration plan to convert existing .claude/commands to the new agent-based system. Each command is mapped to an equivalent agent with defined roles, responsibilities, capabilities, and tool access restrictions.

## Agent Definition Format
Each agent uses YAML frontmatter with the following structure:
```yaml
---
role: agent-type
name: Agent Display Name
responsibilities:
  - Primary responsibility
  - Secondary responsibility
capabilities:
  - capability-1
  - capability-2
tools:
  allowed:
    - tool-name
  restricted:
    - restricted-tool
triggers:
  - pattern: "regex pattern"
    priority: high|medium|low
  - keyword: "activation keyword"
---
```

## Migration Categories

### 1. Coordination Agents

#### Swarm Initializer Agent
**Command**: `.claude/commands/coordination/init.md`
```yaml
---
role: coordinator
name: Swarm Initializer
responsibilities:
  - Initialize agent swarms with optimal topology
  - Configure distributed coordination systems
  - Set up inter-agent communication channels
capabilities:
  - swarm-initialization
  - topology-optimisation
  - resource-allocation
  - network-configuration
tools:
  allowed:
    - mcp__claude-flow__swarm_init
    - mcp__claude-flow__topology_optimize
    - mcp__claude-flow__memory_usage
    - TodoWrite
  restricted:
    - Bash
    - Write
    - Edit
triggers:
  - pattern: "init.*swarm|create.*swarm|setup.*agents"
    priority: high
  - keyword: "swarm-init"
---
```

#### Agent Spawner
**Command**: `.claude/commands/coordination/spawn.md`
```yaml
---
role: coordinator
name: Agent Spawner
responsibilities:
  - Create specialised cognitive patterns for task execution
  - Assign capabilities to agents based on requirements
  - Manage agent lifecycle and resource allocation
capabilities:
  - agent-creation
  - capability-assignment
  - resource-management
  - pattern-recognition
tools:
  allowed:
    - mcp__claude-flow__agent_spawn
    - mcp__claude-flow__daa_agent_create
    - mcp__claude-flow__agent_list
    - mcp__claude-flow__memory_usage
  restricted:
    - Bash
    - Write
    - Edit
triggers:
  - pattern: "spawn.*agent|create.*agent|add.*agent"
    priority: high
  - keyword: "agent-spawn"
---
```

#### Task Orchestrator
**Command**: `.claude/commands/coordination/orchestrate.md`
```yaml
---
role: orchestrator
name: Task Orchestrator
responsibilities:
  - Decompose complex tasks into manageable subtasks
  - Coordinate parallel and sequential execution strategies
  - Monitor task progress and dependencies
  - Synthesize results from multiple agents
capabilities:
  - task-decomposition
  - execution-planning
  - dependency-management
  - result-aggregation
  - progress-tracking
tools:
  allowed:
    - mcp__claude-flow__task_orchestrate
    - mcp__claude-flow__task_status
    - mcp__claude-flow__task_results
    - mcp__claude-flow__parallel_execute
    - TodoWrite
    - TodoRead
  restricted:
    - Bash
    - Write
    - Edit
triggers:
  - pattern: "orchestrate|coordinate.*task|manage.*workflow"
    priority: high
  - keyword: "orchestrate"
---
```

### 2. GitHub Integration Agents

#### PR Manager Agent
**Command**: `.claude/commands/github/pr-manager.md`
```yaml
---
role: github-specialist
name: Pull Request Manager
responsibilities:
  - Manage complete pull request lifecycle
  - Coordinate multi-reviewer workflows
  - Handle merge strategies and conflict resolution
  - Track PR progress with issue integration
capabilities:
  - pr-creation
  - review-coordination
  - merge-management
  - conflict-resolution
  - status-tracking
tools:
  allowed:
    - Bash  # For gh CLI commands
    - mcp__claude-flow__swarm_init
    - mcp__claude-flow__agent_spawn
    - mcp__claude-flow__task_orchestrate
    - mcp__claude-flow__memory_usage
    - TodoWrite
    - Read
  restricted:
    - Write  # Should use gh CLI for GitHub operations
    - Edit
triggers:
  - pattern: "pr|pull.?request|merge.*request"
    priority: high
  - keyword: "pr-manager"
---
```

#### Code Review Swarm Agent
**Command**: `.claude/commands/github/code-review-swarm.md`
```yaml
---
role: reviewer
name: Code Review Coordinator
responsibilities:
  - Orchestrate multi-agent code reviews
  - Ensure code quality and standards compliance
  - Coordinate security and performance reviews
  - Generate comprehensive review reports
capabilities:
  - code-analysis
  - quality-assessment
  - security-scanning
  - performance-review
  - report-generation
tools:
  allowed:
    - Bash  # For gh CLI
    - Read
    - Grep
    - mcp__claude-flow__swarm_init
    - mcp__claude-flow__agent_spawn
    - mcp__claude-flow__github_code_review
    - mcp__claude-flow__memory_usage
  restricted:
    - Write
    - Edit
triggers:
  - pattern: "review.*code|code.*review|check.*pr"
    priority: high
  - keyword: "code-review"
---
```

#### Release Manager Agent
**Command**: `.claude/commands/github/release-manager.md`
```yaml
---
role: release-coordinator
name: Release Manager
responsibilities:
  - Coordinate release preparation and deployment
  - Manage version tagging and changelog generation
  - Orchestrate multi-repository releases
  - Handle rollback procedures
capabilities:
  - release-planning
  - version-management
  - changelog-generation
  - deployment-coordination
  - rollback-execution
tools:
  allowed:
    - Bash
    - Read
    - mcp__claude-flow__github_release_coord
    - mcp__claude-flow__swarm_init
    - mcp__claude-flow__task_orchestrate
    - TodoWrite
  restricted:
    - Write  # Use version control for releases
    - Edit
triggers:
  - pattern: "release|deploy|tag.*version|create.*release"
    priority: high
  - keyword: "release-manager"
---
```

### 3. SPARC Methodology Agents

#### SPARC Orchestrator Agent
**Command**: `.claude/commands/sparc/orchestrator.md`
```yaml
---
role: sparc-coordinator
name: SPARC Orchestrator
responsibilities:
  - Coordinate SPARC methodology phases
  - Manage task decomposition and agent allocation
  - Track progress across all SPARC phases
  - Synthesize results from specialised agents
capabilities:
  - sparc-coordination
  - phase-management
  - task-planning
  - resource-allocation
  - result-synthesis
tools:
  allowed:
    - mcp__claude-flow__sparc_mode
    - mcp__claude-flow__swarm_init
    - mcp__claude-flow__agent_spawn
    - mcp__claude-flow__task_orchestrate
    - TodoWrite
    - TodoRead
    - mcp__claude-flow__memory_usage
  restricted:
    - Bash
    - Write
    - Edit
triggers:
  - pattern: "sparc.*orchestrat|coordinate.*sparc"
    priority: high
  - keyword: "sparc-orchestrator"
---
```

#### SPARC Coder Agent
**Command**: `.claude/commands/sparc/coder.md`
```yaml
---
role: implementer
name: SPARC Implementation Specialist
responsibilities:
  - Transform specifications into working code
  - Implement TDD practices with parallel test creation
  - Ensure code quality and standards compliance
  - Optimise implementation for performance
capabilities:
  - code-generation
  - test-implementation
  - refactoring
  - optimisation
  - documentation
tools:
  allowed:
    - Read
    - Write
    - Edit
    - MultiEdit
    - Bash
    - mcp__claude-flow__sparc_mode
    - TodoWrite
  restricted:
    - mcp__claude-flow__swarm_init  # Focus on implementation
triggers:
  - pattern: "implement|code|develop|build.*feature"
    priority: high
  - keyword: "sparc-coder"
---
```

#### SPARC Tester Agent
**Command**: `.claude/commands/sparc/tester.md`
```yaml
---
role: quality-assurance
name: SPARC Testing Specialist
responsibilities:
  - Design comprehensive test strategies
  - Implement parallel test execution
  - Ensure coverage requirements are met
  - Coordinate testing across different levels
capabilities:
  - test-design
  - test-implementation
  - coverage-analysis
  - performance-testing
  - security-testing
tools:
  allowed:
    - Read
    - Write
    - Edit
    - Bash
    - mcp__claude-flow__sparc_mode
    - TodoWrite
    - mcp__claude-flow__parallel_execute
  restricted:
    - mcp__claude-flow__swarm_init
triggers:
  - pattern: "test|verify|validate|check.*quality"
    priority: high
  - keyword: "sparc-tester"
---
```

### 4. Analysis Agents

#### Performance Analyzer Agent
**Command**: `.claude/commands/analysis/performance-bottlenecks.md`
```yaml
---
role: analyst
name: Performance Bottleneck Analyzer
responsibilities:
  - Identify performance bottlenecks in workflows
  - Analyze execution patterns and resource usage
  - Recommend optimisation strategies
  - Monitor improvement metrics
capabilities:
  - performance-analysis
  - bottleneck-detection
  - metric-collection
  - pattern-recognition
  - optimisation-planning
tools:
  allowed:
    - mcp__claude-flow__bottleneck_analyze
    - mcp__claude-flow__performance_report
    - mcp__claude-flow__metrics_collect
    - mcp__claude-flow__trend_analysis
    - Read
    - Grep
  restricted:
    - Write
    - Edit
    - Bash
triggers:
  - pattern: "analyze.*performance|bottleneck|slow.*execution"
    priority: high
  - keyword: "performance-analyzer"
---
```

#### Token Efficiency Analyst Agent
**Command**: `.claude/commands/analysis/token-efficiency.md`
```yaml
---
role: analyst
name: Token Efficiency Analyzer
responsibilities:
  - Monitor token consumption across operations
  - Identify inefficient token usage patterns
  - Recommend optimisation strategies
  - Track cost implications
capabilities:
  - token-analysis
  - cost-optimisation
  - usage-tracking
  - pattern-detection
  - report-generation
tools:
  allowed:
    - mcp__claude-flow__token_usage
    - mcp__claude-flow__cost_analysis
    - mcp__claude-flow__usage_stats
    - mcp__claude-flow__memory_analytics
    - Read
  restricted:
    - Write
    - Edit
    - Bash
triggers:
  - pattern: "token.*usage|analyze.*cost|efficiency.*report"
    priority: medium
  - keyword: "token-analyzer"
---
```

### 5. Memory Management Agents

#### Memory Coordinator Agent
**Command**: `.claude/commands/memory/usage.md`
```yaml
---
role: memory-manager
name: Memory Coordination Specialist
responsibilities:
  - Manage persistent memory across sessions
  - Coordinate memory namespaces and TTL
  - Optimise memory usage and compression
  - Facilitate cross-agent memory sharing
capabilities:
  - memory-management
  - namespace-coordination
  - data-persistence
  - compression-optimisation
  - synchronisation
tools:
  allowed:
    - mcp__claude-flow__memory_usage
    - mcp__claude-flow__memory_search
    - mcp__claude-flow__memory_namespace
    - mcp__claude-flow__memory_compress
    - mcp__claude-flow__memory_sync
  restricted:
    - Write
    - Edit
    - Bash
triggers:
  - pattern: "memory|remember|store.*context|retrieve.*data"
    priority: high
  - keyword: "memory-manager"
---
```

#### Neural Pattern Agent
**Command**: `.claude/commands/memory/neural.md`
```yaml
---
role: ai-specialist
name: Neural Pattern Coordinator
responsibilities:
  - Train and manage neural patterns
  - Coordinate cognitive behaviour analysis
  - Implement adaptive learning strategies
  - Optimise AI model performance
capabilities:
  - neural-training
  - pattern-recognition
  - cognitive-analysis
  - model-optimisation
  - transfer-learning
tools:
  allowed:
    - mcp__claude-flow__neural_train
    - mcp__claude-flow__neural_patterns
    - mcp__claude-flow__neural_predict
    - mcp__claude-flow__cognitive_analyze
    - mcp__claude-flow__learning_adapt
  restricted:
    - Write
    - Edit
    - Bash
triggers:
  - pattern: "neural|ai.*pattern|cognitive|machine.*learning"
    priority: high
  - keyword: "neural-patterns"
---
```

### 6. Automation Agents

#### Smart Agent Coordinator
**Command**: `.claude/commands/automation/smart-agents.md`
```yaml
---
role: automation-specialist
name: Smart Agent Coordinator
responsibilities:
  - Automate agent spawning based on task requirements
  - Implement intelligent capability matching
  - Manage dynamic agent allocation
  - Optimise resource utilization
capabilities:
  - intelligent-spawning
  - capability-matching
  - resource-optimisation
  - pattern-learning
  - auto-scaling
tools:
  allowed:
    - mcp__claude-flow__daa_agent_create
    - mcp__claude-flow__daa_capability_match
    - mcp__claude-flow__daa_resource_alloc
    - mcp__claude-flow__swarm_scale
    - mcp__claude-flow__agent_metrics
  restricted:
    - Write
    - Edit
    - Bash
triggers:
  - pattern: "smart.*agent|auto.*spawn|intelligent.*coordination"
    priority: high
  - keyword: "smart-agents"
---
```

#### Self-Healing Coordinator Agent
**Command**: `.claude/commands/automation/self-healing.md`
```yaml
---
role: reliability-engineer
name: Self-Healing System Coordinator
responsibilities:
  - Detect and recover from system failures
  - Implement fault tolerance strategies
  - Coordinate automatic recovery procedures
  - Monitor system health continuously
capabilities:
  - fault-detection
  - automatic-recovery
  - health-monitoring
  - resilience-planning
  - error-analysis
tools:
  allowed:
    - mcp__claude-flow__daa_fault_tolerance
    - mcp__claude-flow__health_check
    - mcp__claude-flow__error_analysis
    - mcp__claude-flow__diagnostic_run
    - Bash  # For system commands
  restricted:
    - Write  # Prevent accidental file modifications during recovery
    - Edit
triggers:
  - pattern: "self.*heal|auto.*recover|fault.*toleran|system.*health"
    priority: high
  - keyword: "self-healing"
---
```

### 7. Optimisation Agents

#### Parallel Execution Optimizer Agent
**Command**: `.claude/commands/optimisation/parallel-execution.md`
```yaml
---
role: optimizer
name: Parallel Execution Optimizer
responsibilities:
  - Optimise task execution for parallelism
  - Identify parallelization opportunities
  - Coordinate concurrent operations
  - Monitor parallel execution efficiency
capabilities:
  - parallelization-analysis
  - execution-optimisation
  - load-balancing
  - performance-monitoring
  - bottleneck-removal
tools:
  allowed:
    - mcp__claude-flow__parallel_execute
    - mcp__claude-flow__load_balance
    - mcp__claude-flow__batch_process
    - mcp__claude-flow__performance_report
    - TodoWrite
  restricted:
    - Write
    - Edit
triggers:
  - pattern: "parallel|concurrent|simultaneous|batch.*execution"
    priority: high
  - keyword: "parallel-optimizer"
---
```

#### Auto-Topology Optimizer Agent
**Command**: `.claude/commands/optimisation/auto-topology.md`
```yaml
---
role: optimizer
name: Topology Optimisation Specialist
responsibilities:
  - Analyze and optimise swarm topology
  - Adapt topology based on workload
  - Balance communication overhead
  - Ensure optimal agent distribution
capabilities:
  - topology-analysis
  - graph-optimisation
  - network-design
  - load-distribution
  - adaptive-configuration
tools:
  allowed:
    - mcp__claude-flow__topology_optimize
    - mcp__claude-flow__swarm_monitor
    - mcp__claude-flow__coordination_sync
    - mcp__claude-flow__swarm_status
    - mcp__claude-flow__metrics_collect
  restricted:
    - Write
    - Edit
    - Bash
triggers:
  - pattern: "topology|optimise.*swarm|network.*structure"
    priority: medium
  - keyword: "topology-optimizer"
---
```

### 8. Monitoring Agents

#### Swarm Monitor Agent
**Command**: `.claude/commands/monitoring/status.md`
```yaml
---
role: monitor
name: Swarm Status Monitor
responsibilities:
  - Monitor swarm health and performance
  - Track agent status and utilization
  - Generate real-time status reports
  - Alert on anomalies or failures
capabilities:
  - health-monitoring
  - performance-tracking
  - status-reporting
  - anomaly-detection
  - alert-generation
tools:
  allowed:
    - mcp__claude-flow__swarm_status
    - mcp__claude-flow__swarm_monitor
    - mcp__claude-flow__agent_metrics
    - mcp__claude-flow__health_check
    - mcp__claude-flow__performance_report
  restricted:
    - Write
    - Edit
    - Bash
triggers:
  - pattern: "monitor|status|health.*check|swarm.*status"
    priority: medium
  - keyword: "swarm-monitor"
---
```

## Implementation Guidelines

### 1. Agent Activation
- Agents are activated by pattern matching in user messages
- Higher priority patterns take precedence
- Multiple agents can be activated for complex tasks

### 2. Tool Restrictions
- Each agent has specific allowed and restricted tools
- Restrictions ensure agents stay within their domain
- Critical operations require specialised agents

### 3. Inter-Agent Communication
- Agents communicate through shared memory
- Task orchestrator coordinates multi-agent workflows
- Results are aggregated by coordinator agents

### 4. Migration Steps
1. Create `.claude/agents/` directory structure
2. Convert each command to agent definition format
3. Update activation patterns for natural language
4. Test agent interactions and handoffs
5. Implement gradual rollout with fallbacks

### 5. Backwards Compatibility
- Keep command files during transition
- Map command invocations to agent activations
- Provide migration warnings for deprecated commands

## Monitoring Migration Success

### Key Metrics
- Agent activation accuracy
- Task completion rates
- Inter-agent coordination efficiency
- User satisfaction scores
- Performance improvements

### Validation Criteria
- All commands have equivalent agents
- No functionality loss during migration
- Improved natural language understanding
- Better task decomposition and parallelization
- Enhanced error handling and recovery

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
- [gossip-coordinator](../../../reference/agents/consensus/gossip-coordinator.md)
- [hierarchical-coordinator](../../../reference/agents/swarm/hierarchical-coordinator.md)
- [implementer-sparc-coder](../../../reference/agents/templates/implementer-sparc-coder.md)
- [issue-tracker](../../../reference/agents/github/issue-tracker.md)
- [memory-coordinator](../../../reference/agents/templates/memory-coordinator.md)
- [mesh-coordinator](../../../reference/agents/swarm/mesh-coordinator.md)
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
