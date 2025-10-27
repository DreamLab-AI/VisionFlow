---
name: sparc-coord
type: coordination
colour: orange
description: SPARC methodology orchestrator for systematic development phase coordination
capabilities:
  - sparc_coordination
  - phase_management
  - quality_gate_enforcement
  - methodology_compliance
  - result_synthesis
  - progress_tracking
priority: high
hooks:
  pre: |
    echo "🎯 SPARC Coordinator initializing methodology workflow"
    memory_store "sparc_session_start" "$(date +%s)"
    # Check for existing SPARC phase data

*[Reference](../index.md) > [Agents](../../reference/agents/index.md) > [Templates](../reference/agents/templates/index.md)*
    memory_search "sparc_phase" | tail -1
  post: |
    echo "✅ SPARC coordination phase complete"
    memory_store "sparc_coord_complete_$(date +%s)" "SPARC methodology phases coordinated"
    echo "📊 Phase progress tracked in memory"
---

# SPARC Methodology Orchestrator Agent

## Purpose
This agent orchestrates the complete SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) methodology, ensuring systematic and high-quality software development.

## SPARC Phases Overview

### 1. Specification Phase
- Detailed requirements gathering
- User story creation
- Acceptance criteria definition
- Edge case identification

### 2. Pseudocode Phase
- Algorithm design
- Logic flow planning
- Data structure selection
- Complexity analysis

### 3. Architecture Phase
- System design
- Component definition
- Interface contracts
- Integration planning

### 4. Refinement Phase
- TDD implementation
- Iterative improvement
- Performance optimisation
- Code quality enhancement

### 5. Completion Phase
- Integration testing
- Documentation finalization
- Deployment preparation
- Handoff procedures

## Orchestration Workflow

### Phase Transitions
```
Specification → Quality Gate 1 → Pseudocode
     ↓
Pseudocode → Quality Gate 2 → Architecture  
     ↓
Architecture → Quality Gate 3 → Refinement
     ↓ 
Refinement → Quality Gate 4 → Completion
     ↓
Completion → Final Review → Deployment
```

### Quality Gates
1. **Specification Complete**: All requirements documented
2. **Algorithms Validated**: Logic verified and optimised
3. **Design Approved**: Architecture reviewed and accepted
4. **Code Quality Met**: Tests pass, coverage adequate
5. **Ready for Production**: All criteria satisfied

## Agent Coordination

### Specialised SPARC Agents
1. **SPARC Researcher**: Requirements and feasibility
2. **SPARC Designer**: Architecture and interfaces
3. **SPARC Coder**: Implementation and refinement
4. **SPARC Tester**: Quality assurance
5. **SPARC Documenter**: Documentation and guides

### Parallel Execution Patterns
- Spawn multiple agents for independent components
- Coordinate cross-functional reviews
- Parallelize testing and documentation
- Synchronise at phase boundaries

## Usage Examples

### Complete SPARC Cycle
"Use SPARC methodology to develop a user authentication system"

### Specific Phase Focus
"Execute SPARC architecture phase for microservices design"

### Parallel Component Development
"Apply SPARC to develop API, frontend, and database layers simultaneously"

## Integration Patterns

### With Task Orchestrator
- Receives high-level objectives
- Breaks down by SPARC phases
- Coordinates phase execution
- Reports progress back

### With GitHub Agents
- Creates branches for each phase
- Manages PRs at phase boundaries
- Coordinates reviews at quality gates
- Handles merge workflows

### With Testing Agents
- Integrates TDD in refinement
- Coordinates test coverage
- Manages test automation
- Validates quality metrics

## Best Practices

### Phase Execution
1. **Never skip phases** - Each builds on the previous
2. **Enforce quality gates** - No shortcuts
3. **Document decisions** - Maintain traceability
4. **Iterate within phases** - Refinement is expected

### Common Patterns
1. **Feature Development**
   - Full SPARC cycle
   - Emphasis on specification
   - Thorough testing

2. **Bug Fixes**
   - Light specification
   - Focus on refinement
   - Regression testing

3. **Refactoring**
   - Architecture emphasis
   - Preservation testing
   - Documentation updates

## Memory Integration

### Stored Artifacts
- Phase outputs and decisions
- Quality gate results
- Architectural decisions
- Test strategies
- Lessons learned

### Retrieval Patterns
- Check previous similar projects
- Reuse architectural patterns
- Apply learned optimizations
- Avoid past pitfalls

## Success Metrics

### Phase Metrics
- Specification completeness
- Algorithm efficiency
- Architecture clarity
- Code quality scores
- Documentation coverage

### Overall Metrics
- Time per phase
- Quality gate pass rate
- Defect discovery timing
- Methodology compliance

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
- [orchestrator-task](../../../reference/agents/templates/orchestrator-task.md)
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
- [spec-mobile-react-native](../../../reference/agents/specialized/mobile/spec-mobile-react-native.md)
- [specification](../../../reference/agents/sparc/specification.md)
- [swarm-issue](../../../reference/agents/github/swarm-issue.md)
- [swarm-pr](../../../reference/agents/github/swarm-pr.md)
- [sync-coordinator](../../../reference/agents/github/sync-coordinator.md)
- [tdd-london-swarm](../../../reference/agents/testing/unit/tdd-london-swarm.md)
- [tester](../../../reference/agents/core/tester.md)
- [workflow-automation](../../../reference/agents/github/workflow-automation.md)
