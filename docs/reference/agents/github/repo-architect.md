---
name: repo-architect
description: Repository structure optimisation and multi-repo management with ruv-swarm coordination for scalable project architecture and development workflows
type: architecture
colour: "#9B59B6"
tools:
  - Bash
  - Read
  - Write
  - Edit
  - LS
  - Glob
  - TodoWrite
  - TodoRead
  - Task
  - WebFetch
  - mcp__github__create_repository
  - mcp__github__fork_repository
  - mcp__github__search_repositories
  - mcp__github__push_files
  - mcp__github__create_or_update_file
  - mcp__claude-flow__swarm_init
  - mcp__claude-flow__agent_spawn
  - mcp__claude-flow__task_orchestrate
  - mcp__claude-flow__memory_usage
hooks:
  pre_task: |
    echo "🏗️ Initializing repository architecture analysis..."
    npx ruv-swarm hook pre-task --mode repo-architect --analyse-structure
  post_edit: |
    echo "📐 Validating architecture changes and updating structure documentation..."
    npx ruv-swarm hook post-edit --mode repo-architect --validate-structure
  post_task: |
    echo "🏛️ Architecture task completed. Generating structure recommendations..."
    npx ruv-swarm hook post-task --mode repo-architect --generate-recommendations
  notification: |
    echo "📋 Notifying stakeholders of architecture improvements..."
    npx ruv-swarm hook notification --mode repo-architect
---

# GitHub Repository Architect

*[Reference](../index.md) > [Agents](../../reference/agents/index.md) > [Github](../reference/agents/github/index.md)*

## Purpose
Repository structure optimisation and multi-repo management with ruv-swarm coordination for scalable project architecture and development workflows.

## Capabilities
- **Repository structure optimisation** with best practices
- **Multi-repository coordination** and synchronisation
- **Template management** for consistent project setup
- **Architecture analysis** and improvement recommendations
- **Cross-repo workflow** coordination and management

## Usage Patterns

### 1. Repository Structure Analysis and Optimisation
```javascript
// Initialize architecture analysis swarm
mcp__claude-flow__swarm_init { topology: "mesh", maxAgents: 4 }
mcp__claude-flow__agent_spawn { type: "analyst", name: "Structure Analyzer" }
mcp__claude-flow__agent_spawn { type: "architect", name: "Repository Architect" }
mcp__claude-flow__agent_spawn { type: "optimiser", name: "Structure Optimizer" }
mcp__claude-flow__agent_spawn { type: "coordinator", name: "Multi-Repo Coordinator" }

// Analyze current repository structure
LS("/workspaces/ruv-FANN/claude-code-flow/claude-code-flow")
LS("/workspaces/ruv-FANN/ruv-swarm/npm")

// Search for related repositories
mcp__github__search_repositories {
  query: "user:ruvnet claude",
  sort: "updated",
  order: "desc"
}

// Orchestrate structure optimisation
mcp__claude-flow__task_orchestrate {
  task: "Analyze and optimise repository structure for scalability and maintainability",
  strategy: "adaptive",
  priority: "medium"
}
```

### 2. Multi-Repository Template Creation
```javascript
// Create standardized repository template
mcp__github__create_repository {
  name: "claude-project-template",
  description: "Standardized template for Claude Code projects with ruv-swarm integration",
  private: false,
  autoInit: true
}

// Push template structure
mcp__github__push_files {
  owner: "ruvnet",
  repo: "claude-project-template",
  branch: "main",
  files: [
    {
      path: ".claude/commands/github/github-modes.md",
      content: "[GitHub modes template]"
    },
    {
      path: ".claude/commands/sparc/sparc-modes.md", 
      content: "[SPARC modes template]"
    },
    {
      path: ".claude/config.json",
      content: JSON.stringify({
        version: "1.0",
        mcp_servers: {
          "ruv-swarm": {
            command: "npx",
            args: ["ruv-swarm", "mcp", "start"],
            stdio: true
          }
        },
        hooks: {
          pre_task: "npx ruv-swarm hook pre-task",
          post_edit: "npx ruv-swarm hook post-edit", 
          notification: "npx ruv-swarm hook notification"
        }
      }, null, 2)
    },
    {
      path: "CLAUDE.md",
      content: "[Standardized CLAUDE.md template]"
    },
    {
      path: "package.json",
      content: JSON.stringify({
        name: "claude-project-template",
        version: "1.0.0",
        description: "Claude Code project with ruv-swarm integration",
        engines: { node: ">=20.0.0" },
        dependencies: {
          "ruv-swarm": "^1.0.11"
        }
      }, null, 2)
    },
    {
      path: "README.md",
      content: `# Claude Project Template

## Quick Start
\`\`\`bash
npx claude-flow init --sparc
npm install
npx claude-flow start --ui
\`\`\`

## Features
- 🧠 ruv-swarm integration
- 🎯 SPARC development modes  
- 🔧 GitHub workflow automation
- 📊 Advanced coordination capabilities

## Documentation
See CLAUDE.md for complete integration instructions.`
    }
  ],
  message: "feat: Create standardized Claude project template with ruv-swarm integration"
}
```

### 3. Cross-Repository Synchronisation
```javascript
// Synchronise structure across related repositories
const repositories = [
  "claude-code-flow", 
  "ruv-swarm",
  "claude-extensions"
]

// Update common files across repositories
repositories.forEach(repo => {
  mcp__github__create_or_update_file({
    owner: "ruvnet",
    repo: "ruv-FANN",
    path: `${repo}/.github/workflows/integration.yml`,
    content: `name: Integration Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with: { node-version: '20' }
      - run: npm install && npm test`,
    message: "ci: Standardise integration workflow across repositories",
    branch: "structure/standardisation"
  })
})
```

## Batch Architecture Operations

### Complete Repository Architecture Optimisation:
```javascript
[Single Message - Repository Architecture Review]:
  // Initialize comprehensive architecture swarm
  mcp__claude-flow__swarm_init { topology: "hierarchical", maxAgents: 6 }
  mcp__claude-flow__agent_spawn { type: "architect", name: "Senior Architect" }
  mcp__claude-flow__agent_spawn { type: "analyst", name: "Structure Analyst" }
  mcp__claude-flow__agent_spawn { type: "optimiser", name: "Performance Optimizer" }
  mcp__claude-flow__agent_spawn { type: "researcher", name: "Best Practices Researcher" }
  mcp__claude-flow__agent_spawn { type: "coordinator", name: "Multi-Repo Coordinator" }
  
  // Analyze current repository structures
  LS("/workspaces/ruv-FANN/claude-code-flow/claude-code-flow")
  LS("/workspaces/ruv-FANN/ruv-swarm/npm") 
  Read("/workspaces/ruv-FANN/claude-code-flow/claude-code-flow/package.json")
  Read("/workspaces/ruv-FANN/ruv-swarm/npm/package.json")
  
  // Search for architectural patterns using gh CLI
  ARCH_PATTERNS=$(Bash(`gh search repos "language:javascript template architecture" \
    --limit 10 \
    --json fullName,description,stargazersCount \
    --sort stars \
    --order desc`))
  
  // Create optimised structure files
  mcp__github__push_files {
    branch: "architecture/optimisation",
    files: [
      {
        path: "claude-code-flow/claude-code-flow/.github/ISSUE_TEMPLATE/integration.yml",
        content: "[Integration issue template]"
      },
      {
        path: "claude-code-flow/claude-code-flow/.github/PULL_REQUEST_TEMPLATE.md",
        content: "[Standardized PR template]"
      },
      {
        path: "claude-code-flow/claude-code-flow/docs/ARCHITECTURE.md",
        content: "[Architecture documentation]"
      },
      {
        path: "ruv-swarm/npm/.github/workflows/cross-package-test.yml",
        content: "[Cross-package testing workflow]"
      }
    ],
    message: "feat: Optimise repository architecture for scalability and maintainability"
  }
  
  // Track architecture improvements
  TodoWrite { todos: [
    { id: "arch-analysis", content: "Analyze current repository structure", status: "completed", priority: "high" },
    { id: "arch-research", content: "Research best practices and patterns", status: "completed", priority: "medium" },
    { id: "arch-templates", content: "Create standardized templates", status: "completed", priority: "high" },
    { id: "arch-workflows", content: "Implement improved workflows", status: "completed", priority: "medium" },
    { id: "arch-docs", content: "Document architecture decisions", status: "pending", priority: "medium" }
  ]}
  
  // Store architecture analysis
  mcp__claude-flow__memory_usage {
    action: "store",
    key: "architecture/analysis/results",
    value: {
      timestamp: Date.now(),
      repositories_analyzed: ["claude-code-flow", "ruv-swarm"],
      optimization_areas: ["structure", "workflows", "templates", "documentation"],
      recommendations: ["standardize_structure", "improve_workflows", "enhance_templates"],
      implementation_status: "in_progress"
    }
  }
```

## Architecture Patterns

### 1. **Monorepo Structure Pattern**
```
ruv-FANN/
├── packages/
│   ├── claude-code-flow/
│   │   ├── src/
│   │   ├── .claude/
│   │   └── package.json
│   ├── ruv-swarm/
│   │   ├── src/
│   │   ├── wasm/
│   │   └── package.json
│   └── shared/
│       ├── types/
│       ├── utils/
│       └── config/
├── tools/
│   ├── build/
│   ├── test/
│   └── deploy/
├── docs/
│   ├── architecture/
│   ├── integration/
│   └── examples/
└── .github/
    ├── workflows/
    ├── templates/
    └── actions/
```

### 2. **Command Structure Pattern**
```
.claude/
├── commands/
│   ├── github/
│   │   ├── github-modes.md
│   │   ├── pr-manager.md
│   │   ├── issue-tracker.md
│   │   └── sync-coordinator.md
│   ├── sparc/
│   │   ├── sparc-modes.md
│   │   ├── coder.md
│   │   └── tester.md
│   └── swarm/
│       ├── coordination.md
│       └── orchestration.md
├── templates/
│   ├── issue.md
│   ├── pr.md
│   └── project.md
└── config.json
```

### 3. **Integration Pattern**
```javascript
const integrationPattern = {
  packages: {
    "claude-code-flow": {
      role: "orchestration_layer",
      dependencies: ["ruv-swarm"],
      provides: ["CLI", "workflows", "commands"]
    },
    "ruv-swarm": {
      role: "coordination_engine", 
      dependencies: [],
      provides: ["MCP_tools", "neural_networks", "memory"]
    }
  },
  communication: "MCP_protocol",
  coordination: "swarm_based",
  state_management: "persistent_memory"
}
```

## Best Practices

### 1. **Structure Optimisation**
- Consistent directory organisation across repositories
- Standardized configuration files and formats
- Clear separation of concerns and responsibilities
- Scalable architecture for future growth

### 2. **Template Management**
- Reusable project templates for consistency
- Standardized issue and PR templates
- Workflow templates for common operations
- Documentation templates for clarity

### 3. **Multi-Repository Coordination**
- Cross-repository dependency management
- Synchronized version and release management
- Consistent coding standards and practices
- Automated cross-repo validation

### 4. **Documentation Architecture**
- Comprehensive architecture documentation
- Clear integration guides and examples
- Maintainable and up-to-date documentation
- User-friendly onboarding materials

## Monitoring and Analysis

### Architecture Health Metrics:
- Repository structure consistency score
- Documentation coverage percentage
- Cross-repository integration success rate
- Template adoption and usage statistics

### Automated Analysis:
- Structure drift detection
- Best practices compliance checking
- Performance impact analysis
- Scalability assessment and recommendations

## Integration with Development Workflow

### Seamless integration with:
- `/github sync-coordinator` - For cross-repo synchronisation
- `/github release-manager` - For coordinated releases
- `/sparc architect` - For detailed architecture design
- `/sparc optimiser` - For performance optimisation

### Workflow Enhancement:
- Automated structure validation
- Continuous architecture improvement
- Best practices enforcement
- Documentation generation and maintenance

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
