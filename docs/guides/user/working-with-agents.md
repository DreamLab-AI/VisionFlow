---
title: Working with Agents
description: User guide for effectively working with AI agents in Turbo Flow Claude
category: user-guide
tags: [agents, coordination, workflow, collaboration]
difficulty: beginner
last_updated: 2025-10-27
related:
  - /docs/guides/user/xr-setup.md
  - /docs/getting-started/README.md
  - /docs/tutorials/first-swarm.md
---

# Working with Agents

> **⚠️ Work in Progress**: This guide is currently under development. Content will be expanded in future updates.

## Overview

This guide teaches you how to effectively work with AI agents in Turbo Flow Claude. You'll learn how to spawn agents, coordinate tasks, and leverage agent swarms for complex workflows.

## What Are Agents?

### Agent Types

Turbo Flow Claude provides 54 specialized agents across multiple categories:

**Core Development:**
- `coder` - Implements features and writes code
- `reviewer` - Reviews code quality and security
- `tester` - Creates comprehensive test suites
- `planner` - Breaks down tasks and creates plans
- `researcher` - Analyzes requirements and patterns

**Specialized Development:**
- `backend-dev` - Backend API development
- `system-architect` - System design and architecture
- `api-docs` - API documentation generation
- `ml-developer` - Machine learning implementations
- `cicd-engineer` - CI/CD pipeline setup

**SPARC Methodology:**
- `specification` - Requirements analysis
- `pseudocode` - Algorithm design
- `architecture` - System architecture
- `refinement` - TDD implementation

**GitHub Integration:**
- `pr-manager` - Pull request management
- `code-review-swarm` - Automated code review
- `issue-tracker` - Issue triage and tracking
- `release-manager` - Release coordination

**Performance & Analysis:**
- `perf-analyzer` - Performance analysis
- `performance-benchmarker` - Benchmarking
- `code-analyzer` - Code quality analysis

### How Agents Work

**Agent capabilities:**
- Specialized domain knowledge
- Autonomous task execution
- Inter-agent coordination
- Memory and context sharing
- Hook-based automation

**Agent coordination:**
- Agents communicate via memory storage
- Hooks enable automatic coordination
- Swarm topologies optimize workflows
- Neural patterns improve over time

## Getting Started with Agents

### Spawning Your First Agent

**Basic agent spawning:**
```bash
# Using Claude Code's Task tool (recommended)
Task("Coder Agent", "Implement user authentication", "coder")
```

**Using MCP for coordination setup (optional):**
```bash
# Initialize swarm topology
npx claude-flow mcp swarm_init --topology mesh

# Define agent types for coordination
npx claude-flow mcp agent_spawn --type coder --name "Backend Coder"
```

### Single Agent Workflow

**Example: Code review task**
```javascript
// Spawn a single reviewer agent
Task("Code Reviewer",
  "Review the authentication implementation in src/auth for security issues",
  "reviewer")
```

**What happens:**
1. Agent receives task description
2. Reviews specified files
3. Identifies issues and improvements
4. Provides detailed feedback
5. Stores findings in memory

### Multiple Agents Working Together

**Example: Full-stack feature development**
```javascript
// Spawn multiple agents concurrently
Task("Backend Developer",
  "Build REST API for user management with authentication",
  "backend-dev")

Task("Frontend Developer",
  "Create React components for user profile management",
  "coder")

Task("Database Architect",
  "Design PostgreSQL schema for user data and relationships",
  "system-architect")

Task("Tester",
  "Create comprehensive test suite for user management features",
  "tester")

Task("Documentation Writer",
  "Generate API documentation and user guides",
  "api-docs")
```

**Coordination:**
- Agents share decisions via memory
- Hooks coordinate file operations
- Dependencies managed automatically
- Progress tracked in real-time

## Agent Coordination

### Memory Sharing

**How agents coordinate:**
```bash
# Agent stores decision
npx claude-flow@alpha memory store \
  --key "swarm/backend/api-design" \
  --value '{"endpoints": ["/users", "/auth"], "auth": "JWT"}'

# Other agents retrieve decision
npx claude-flow@alpha memory retrieve \
  --key "swarm/backend/api-design"
```

**Common memory patterns:**
- `swarm/[agent]/[topic]` - Agent-specific data
- `swarm/shared/[topic]` - Shared decisions
- `swarm/dependencies/[name]` - Dependency info
- `swarm/status/[task]` - Task status

### Hooks Integration

**Hooks enable automatic coordination:**

**Pre-task hooks:**
```bash
# Automatically run before task starts
npx claude-flow@alpha hooks pre-task \
  --description "Implement user authentication"
```

**Post-edit hooks:**
```bash
# Automatically run after file edits
npx claude-flow@alpha hooks post-edit \
  --file "src/auth/login.ts" \
  --memory-key "swarm/coder/auth-implementation"
```

**Post-task hooks:**
```bash
# Automatically run after task completes
npx claude-flow@alpha hooks post-task \
  --task-id "auth-implementation"
```

**Benefits:**
- Automatic code formatting
- Memory coordination
- Neural pattern training
- Performance tracking
- Git integration

### Swarm Topologies

**Choose topology based on task:**

**Mesh (peer-to-peer):**
```bash
npx claude-flow mcp swarm_init --topology mesh
```
- Best for: Collaborative development
- All agents communicate directly
- Flexible and adaptive
- Good for brainstorming

**Hierarchical (tree structure):**
```bash
npx claude-flow mcp swarm_init --topology hierarchical
```
- Best for: Complex projects
- Clear delegation paths
- Coordinator manages sub-agents
- Good for large teams

**Star (centralized):**
```bash
npx claude-flow mcp swarm_init --topology star
```
- Best for: Simple coordination
- Central coordinator distributes tasks
- Clear single point of control
- Good for focused tasks

**Ring (circular):**
```bash
npx claude-flow mcp swarm_init --topology ring
```
- Best for: Sequential workflows
- Tasks pass from agent to agent
- Maintains order
- Good for pipelines

## SPARC Methodology with Agents

### What is SPARC?

**SPARC phases:**
1. **S**pecification - Requirements analysis
2. **P**seudocode - Algorithm design
3. **A**rchitecture - System design
4. **R**efinement - TDD implementation
5. **C**ompletion - Integration

### Using SPARC with Agents

**Run complete SPARC workflow:**
```bash
npx claude-flow sparc tdd "user authentication feature"
```

**This automatically:**
- Spawns specification agent
- Spawns architecture agent
- Spawns coder agents for TDD
- Spawns tester agents
- Coordinates integration

**Individual SPARC phases:**
```bash
# Specification and pseudocode
npx claude-flow sparc run spec-pseudocode "feature description"

# Architecture design
npx claude-flow sparc run architect "feature description"

# Integration and completion
npx claude-flow sparc run integration "feature description"
```

### SPARC Best Practices

**For best results:**
- Provide detailed feature descriptions
- Let agents complete each phase fully
- Review specifications before proceeding
- Use memory to track decisions
- Iterate based on feedback

## Practical Examples

### Example 1: Building a REST API

**Task:** Create a user management API

**Agent workflow:**
```javascript
// Step 1: Research and design
Task("API Researcher",
  "Research REST API best practices for user management",
  "researcher")

Task("System Architect",
  "Design API architecture with authentication and CRUD operations",
  "system-architect")

// Step 2: Implementation (after architecture complete)
Task("Backend Developer",
  "Implement REST API endpoints based on architecture stored in memory",
  "backend-dev")

Task("Database Developer",
  "Implement database schema and migrations",
  "coder")

// Step 3: Testing and documentation
Task("API Tester",
  "Create comprehensive API test suite with edge cases",
  "tester")

Task("Documentation Writer",
  "Generate OpenAPI spec and developer documentation",
  "api-docs")
```

### Example 2: Code Review

**Task:** Review pull request

**Agent workflow:**
```javascript
Task("Code Analyzer",
  "Analyze code quality and complexity in PR #123",
  "code-analyzer")

Task("Security Reviewer",
  "Review PR #123 for security vulnerabilities",
  "reviewer")

Task("Performance Analyzer",
  "Identify performance issues in PR #123",
  "perf-analyzer")

Task("Test Reviewer",
  "Verify test coverage and quality in PR #123",
  "tester")
```

### Example 3: Performance Optimization

**Task:** Optimize slow endpoint

**Agent workflow:**
```javascript
Task("Performance Benchmarker",
  "Benchmark /api/users endpoint and identify bottlenecks",
  "performance-benchmarker")

Task("Performance Analyzer",
  "Analyze bottleneck causes and propose solutions",
  "perf-analyzer")

Task("Optimizer Coder",
  "Implement performance optimizations based on analysis",
  "coder")

Task("Performance Tester",
  "Verify performance improvements meet targets",
  "tester")
```

## Monitoring Agents

### Check Agent Status

**View swarm status:**
```bash
npx claude-flow mcp swarm_status
```

**View specific agent metrics:**
```bash
npx claude-flow mcp agent_metrics --agentId "coder-1"
```

**Monitor task progress:**
```bash
npx claude-flow mcp task_status --taskId "task-123"
```

### Real-time Monitoring

**Enable swarm monitoring:**
```bash
npx claude-flow mcp swarm_monitor --interval 1
```

**View performance metrics:**
```bash
npx claude-flow mcp performance_report --format detailed
```

## Best Practices

### Spawning Agents

**DO:**
- ✅ Spawn all agents concurrently in one message
- ✅ Provide clear, specific task descriptions
- ✅ Choose appropriate agent types for tasks
- ✅ Enable hooks for coordination
- ✅ Use memory for shared context

**DON'T:**
- ❌ Spawn agents sequentially across messages
- ❌ Use vague task descriptions
- ❌ Mix unrelated tasks in one agent
- ❌ Ignore agent feedback
- ❌ Skip memory coordination

### Task Descriptions

**Good task descriptions:**
```javascript
// ✅ Specific and actionable
Task("Backend Developer",
  "Implement JWT authentication for /api/auth/login endpoint. \
   Use bcrypt for password hashing. Store tokens in Redis. \
   Follow REST API standards stored in memory.",
  "backend-dev")

// ❌ Too vague
Task("Developer", "Make auth work", "coder")
```

### Agent Selection

**Choose the right agent:**
- Use specialized agents when available
- Generic `coder` for general programming
- `system-architect` for design decisions
- `reviewer` for quality checks
- `tester` for comprehensive testing

### Coordination

**Effective coordination:**
- Use consistent memory key patterns
- Run hooks at appropriate times
- Monitor agent progress
- Review agent outputs
- Iterate based on results

## Troubleshooting

### Common Issues

**Agents not coordinating:**
```bash
# Check memory storage
npx claude-flow@alpha memory retrieve --key "swarm/shared/*"

# Verify hooks are running
npx claude-flow@alpha hooks post-edit --file "[file]"
```

**Task not completing:**
```bash
# Check task status
npx claude-flow mcp task_status --taskId "task-id"

# View agent logs
npx claude-flow mcp agent_metrics --agentId "agent-id"
```

**Poor agent performance:**
```bash
# Run performance analysis
npx claude-flow mcp bottleneck_analyze --component "swarm"

# Train neural patterns
npx claude-flow mcp neural_train --iterations 10
```

## Advanced Features

### Neural Pattern Training

**Improve agent performance:**
```bash
# Train from successful tasks
npx claude-flow mcp neural_train --pattern_type "coordination"

# View learned patterns
npx claude-flow mcp neural_patterns --action analyze
```

### Multi-Session Memory

**Persist agent knowledge:**
```bash
# Save session state
npx claude-flow@alpha hooks session-end --export-metrics true

# Restore in new session
npx claude-flow@alpha hooks session-restore --session-id "session-123"
```

### GitHub Integration

**Use GitHub-integrated agents:**
```javascript
Task("PR Manager",
  "Create pull request for authentication feature",
  "pr-manager")

Task("Code Review Swarm",
  "Review and approve PR #123",
  "code-review-swarm")
```

## Next Steps

After mastering basic agent coordination:
1. Explore [XR Setup](/docs/guides/user/xr-setup.md) for immersive development
2. Read [First Swarm Tutorial](/docs/tutorials/first-swarm.md)
3. Learn advanced [SPARC Methodology](/docs/reference/sparc-methodology.md)
4. Join the community and share experiences

## Related Resources

- [Getting Started](/docs/getting-started/README.md)
- [XR Setup Guide](/docs/guides/user/xr-setup.md)
- [First Swarm Tutorial](/docs/tutorials/first-swarm.md)
- [SPARC Methodology](/docs/reference/sparc-methodology.md)

## Support

- Documentation: https://github.com/ruvnet/claude-flow
- Issues: https://github.com/ruvnet/claude-flow/issues
- Community: https://github.com/ruvnet/claude-flow/discussions
- Agent Templates: `/home/devuser/agents/*.md`
