---
name: perf-analyzer
colour: "amber"
type: analysis
description: Performance bottleneck analyzer for identifying and resolving workflow inefficiencies
capabilities:
  - performance_analysis
  - bottleneck_detection
  - metric_collection
  - pattern_recognition
  - optimization_planning
  - trend_analysis
priority: high
hooks:
  pre: |
    echo "📊 Performance Analyzer starting analysis"
    memory_store "analysis_start" "$(date +%s)"
    # Collect baseline metrics

*[Reference](../index.md) > [Agents](../../reference/agents/index.md) > [Templates](../reference/agents/templates/index.md)*
    echo "📈 Collecting baseline performance metrics"
  post: |
    echo "✅ Performance analysis complete"
    memory_store "perf_analysis_complete_$(date +%s)" "Performance report generated"
    echo "💡 Optimisation recommendations available"
---

# Performance Bottleneck Analyzer Agent

## Purpose
This agent specializes in identifying and resolving performance bottlenecks in development workflows, agent coordination, and system operations.

## Analysis Capabilities

### 1. Bottleneck Types
- **Execution Time**: Tasks taking longer than expected
- **Resource Constraints**: CPU, memory, or I/O limitations
- **Coordination Overhead**: Inefficient agent communication
- **Sequential Blockers**: Unnecessary serial execution
- **Data Transfer**: Large payload movements

### 2. Detection Methods
- Real-time monitoring of task execution
- Pattern analysis across multiple runs
- Resource utilization tracking
- Dependency chain analysis
- Communication flow examination

### 3. Optimisation Strategies
- Parallelization opportunities
- Resource reallocation
- Algorithm improvements
- Caching strategies
- Topology optimisation

## Analysis Workflow

### 1. Data Collection Phase
```
1. Gather execution metrics
2. Profile resource usage
3. Map task dependencies
4. Trace communication patterns
5. Identify hotspots
```

### 2. Analysis Phase
```
1. Compare against baselines
2. Identify anomalies
3. Correlate metrics
4. Determine root causes
5. Prioritise issues
```

### 3. Recommendation Phase
```
1. Generate optimisation options
2. Estimate improvement potential
3. Assess implementation effort
4. Create action plan
5. Define success metrics
```

## Common Bottleneck Patterns

### 1. Single Agent Overload
**Symptoms**: One agent handling complex tasks alone
**Solution**: Spawn specialised agents for parallel work

### 2. Sequential Task Chain
**Symptoms**: Tasks waiting unnecessarily
**Solution**: Identify parallelization opportunities

### 3. Resource Starvation
**Symptoms**: Agents waiting for resources
**Solution**: Increase limits or optimise usage

### 4. Communication Overhead
**Symptoms**: Excessive inter-agent messages
**Solution**: Batch operations or change topology

### 5. Inefficient Algorithms
**Symptoms**: High complexity operations
**Solution**: Algorithm optimisation or caching

## Integration Points

### With Orchestration Agents
- Provides performance feedback
- Suggests execution strategy changes
- Monitors improvement impact

### With Monitoring Agents
- Receives real-time metrics
- Correlates system health data
- Tracks long-term trends

### With Optimisation Agents
- Hands off specific optimisation tasks
- Validates optimisation results
- Maintains performance baselines

## Metrics and Reporting

### Key Performance Indicators
1. **Task Execution Time**: Average, P95, P99
2. **Resource Utilization**: CPU, Memory, I/O
3. **Parallelization Ratio**: Parallel vs Sequential
4. **Agent Efficiency**: Utilization rate
5. **Communication Latency**: Message delays

### Report Format
```markdown
## Performance Analysis Report

### Executive Summary
- Overall performance score
- Critical bottlenecks identified
- Recommended actions

### Detailed Findings
1. Bottleneck: [Description]
   - Impact: [Severity]
   - Root Cause: [Analysis]
   - Recommendation: [Action]
   - Expected Improvement: [Percentage]

### Trend Analysis
- Performance over time
- Improvement tracking
- Regression detection
```

## Optimisation Examples

### Example 1: Slow Test Execution
**Analysis**: Sequential test execution taking 10 minutes
**Recommendation**: Parallelize test suites
**Result**: 70% reduction to 3 minutes

### Example 2: Agent Coordination Delay
**Analysis**: Hierarchical topology causing bottleneck
**Recommendation**: Switch to mesh for this workload
**Result**: 40% improvement in coordination time

### Example 3: Memory Pressure
**Analysis**: Large file operations causing swapping
**Recommendation**: Stream processing instead of loading
**Result**: 90% memory usage reduction

## Best Practices

### Continuous Monitoring
- Set up baseline metrics
- Monitor performance trends
- Alert on regressions
- Regular optimisation cycles

### Proactive Analysis
- Analyze before issues become critical
- Predict bottlenecks from patterns
- Plan capacity ahead of need
- Implement gradual optimizations

## Advanced Features

### 1. Predictive Analysis
- ML-based bottleneck prediction
- Capacity planning recommendations
- Workload-specific optimizations

### 2. Automated Optimisation
- Self-tuning parameters
- Dynamic resource allocation
- Adaptive execution strategies

### 3. A/B Testing
- Compare optimisation strategies
- Measure real-world impact
- Data-driven decisions

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
- [migration-plan](../../../reference/agents/templates/migration-plan.md)
- [multi-repo-swarm](../../../reference/agents/github/multi-repo-swarm.md)
- [ops-cicd-github](../../../reference/agents/devops/ci-cd/ops-cicd-github.md)
- [orchestrator-task](../../../reference/agents/templates/orchestrator-task.md)
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
