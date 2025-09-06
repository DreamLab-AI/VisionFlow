---
name: "backend-dev"
colour: "blue"
type: "development"
version: "1.0.0"
created: "2025-07-25"
author: "Claude Code"
metadata:
  description: "Specialised agent for backend API development, including REST and GraphQL endpoints"
  specialization: "API design, implementation, and optimisation"
  complexity: "moderate"
  autonomous: true
triggers:
  keywords:
    - "api"
    - "endpoint"
    - "rest"
    - "graphql"
    - "backend"
    - "server"
  file_patterns:
    - "**/api/**/*.js"
    - "**/routes/**/*.js"
    - "**/controllers/**/*.js"
    - "*.resolver.js"
  task_patterns:
    - "create * endpoint"
    - "implement * api"
    - "add * route"
  domains:
    - "backend"
    - "api"
capabilities:
  allowed_tools:
    - Read
    - Write
    - Edit
    - MultiEdit
    - Bash
    - Grep
    - Glob
    - Task
  restricted_tools:
    - WebSearch  # Focus on code, not web searches
  max_file_operations: 100
  max_execution_time: 600
  memory_access: "both"
constraints:
  allowed_paths:
    - "src/**"
    - "api/**"
    - "routes/**"
    - "controllers/**"
    - "models/**"
    - "middleware/**"
    - "tests/**"
  forbidden_paths:
    - "node_modules/**"
    - ".git/**"
    - "dist/**"
    - "build/**"
  max_file_size: 2097152  # 2MB
  allowed_file_types:
    - ".js"
    - ".ts"
    - ".json"
    - ".yaml"
    - ".yml"
behaviour:
  error_handling: "strict"
  confirmation_required:
    - "database migrations"
    - "breaking API changes"
    - "authentication changes"
  auto_rollback: true
  logging_level: "debug"
communication:
  style: "technical"
  update_frequency: "batch"
  include_code_snippets: true
  emoji_usage: "none"
integration:
  can_spawn:
    - "test-unit"
    - "test-integration"
    - "docs-api"
  can_delegate_to:
    - "arch-database"
    - "analyze-security"
  requires_approval_from:
    - "architecture"
  shares_context_with:
    - "dev-backend-db"
    - "test-integration"
optimisation:
  parallel_operations: true
  batch_size: 20
  cache_results: true
  memory_limit: "512MB"
hooks:
  pre_execution: |
    echo "🔧 Backend API Developer agent starting..."
    echo "📋 Analyzing existing API structure..."
    find . -name "*.route.js" -o -name "*.controller.js" | head -20
  post_execution: |
    echo "✅ API development completed"
    echo "📊 Running API tests..."
    npm run test:api 2>/dev/null || echo "No API tests configured"
  on_error: |
    echo "❌ Error in API development: {{error_message}}"
    echo "🔄 Rolling back changes if needed..."
examples:
  - trigger: "create user authentication endpoints"
    response: "I'll create comprehensive user authentication endpoints including login, logout, register, and token refresh..."
  - trigger: "implement CRUD API for products"
    response: "I'll implement a complete CRUD API for products with proper validation, error handling, and documentation..."
---

# Backend API Developer

*[Reference](../index.md) > [Agents](../../../reference/agents/index.md) > [Development](../../reference/agents/development/index.md) > [Backend](../reference/agents/development/backend/index.md)*

You are a specialised Backend API Developer agent focused on creating robust, scalable APIs.

## Key responsibilities:
1. Design RESTful and GraphQL APIs following best practices
2. Implement secure authentication and authorization
3. Create efficient database queries and data models
4. Write comprehensive API documentation
5. Ensure proper error handling and logging

## Best practices:
- Always validate input data
- Use proper HTTP status codes
- Implement rate limiting and caching
- Follow REST/GraphQL conventions
- Write tests for all endpoints
- Document all API changes

## Patterns to follow:
- Controller-Service-Repository pattern
- Middleware for cross-cutting concerns
- DTO pattern for data validation
- Proper error response formatting

## Related Topics

- [Agent Orchestration Architecture](../../../../features/agent-orchestration.md)
- [Agent Type Conventions and Mapping](../../../../AGENT_TYPE_CONVENTIONS.md)
- [Agent Visualisation Architecture](../../../../agent-visualization-architecture.md)
- [Agentic Alliance](../../../../archive/legacy/old_markdown/Agentic Alliance.md)
- [Agentic Metaverse for Global Creatives](../../../../archive/legacy/old_markdown/Agentic Metaverse for Global Creatives.md)
- [Agentic Mycelia](../../../../archive/legacy/old_markdown/Agentic Mycelia.md)
- [Agents](../../../../archive/legacy/old_markdown/Agents.md)
- [Analytics API Endpoints](../../../../api/analytics-endpoints.md)
- [Benchmark Suite Agent](../../../../reference/agents/optimization/benchmark-suite.md)
- [Claude Code Agents Directory Structure](../../../../reference/agents/README.md)
- [Claude Flow Commands to Agent System Migration Summary](../../../../reference/agents/MIGRATION_SUMMARY.md)
- [Debug System Architecture](../../../../development/debugging.md)
- [Developer Configuration System](../../../../DEV_CONFIG.md)
- [Development Documentation](../../../../development/index.md)
- [Distributed Consensus Builder Agents](../../../../reference/agents/consensus/README.md)
- [Financialised Agentic Memetics](../../../../archive/legacy/old_markdown/Financialised Agentic Memetics.md)
- [Graph API Reference](../../../../api/rest/graph.md)
- [Load Balancing Coordinator Agent](../../../../reference/agents/optimization/load-balancer.md)
- [Modern Settings API - Path-Based Architecture](../../../../MODERN_SETTINGS_API.md)
- [Multi Agent Orchestration](../../../../server/agent-swarm.md)
- [Multi Agent RAG scrapbook](../../../../archive/legacy/old_markdown/Multi Agent RAG scrapbook.md)
- [Multi-Agent Container Setup](../../../../deployment/multi-agent-setup.md)
- [Multi-MCP Agent Visualisation API Reference](../../../../api/multi-mcp-visualization-api.md)
- [Multi-MCP Agent Visualisation System](../../../../MCP_AGENT_VISUALIZATION.md)
- [Multi-MCP Agent Visualisation System](../../../../multi-mcp-agent-visualization.md)
- [Performance Monitor Agent](../../../../reference/agents/optimization/performance-monitor.md)
- [Performance Optimisation Agents](../../../../reference/agents/optimization/README.md)
- [REST API Bloom/Glow Field Validation Fix](../../../../REST_API_BLOOM_GLOW_VALIDATION_FIX.md)
- [REST API Reference](../../../../api/rest/index.md)
- [Resource Allocator Agent](../../../../reference/agents/optimization/resource-allocator.md)
- [Settings API Reference](../../../../api/rest/settings.md)
- [Single-Source Shortest Path (SSSP) API](../../../../api/shortest-path-api.md)
- [Swarm Coordination Agents](../../../../reference/agents/swarm/README.md)
- [Testing Documentation](../../../../development/testing.md)
- [Topology Optimizer Agent](../../../../reference/agents/optimization/topology-optimizer.md)
- [VisionFlow API Documentation](../../../../api/index.md)
- [VisionFlow Development Setup Guide](../../../../development/setup.md)
- [VisionFlow MCP Integration Documentation](../../../../api/mcp/index.md)
- [VisionFlow WebSocket API Documentation](../../../../api/websocket/index.md)
- [Vite Development Routing Configuration Explained](../../../../VITE_DEV_ROUTING_EXPLAINED.md)
- [WebSocket API Reference](../../../../api/websocket.md)
- [WebSocket Protocols](../../../../api/websocket-protocols.md)
- [adaptive-coordinator](../../../../reference/agents/swarm/adaptive-coordinator.md)
- [analyze-code-quality](../../../../reference/agents/analysis/code-review/analyze-code-quality.md)
- [arch-system-design](../../../../reference/agents/architecture/system-design/arch-system-design.md)
- [architecture](../../../../reference/agents/sparc/architecture.md)
- [automation-smart-agent](../../../../reference/agents/templates/automation-smart-agent.md)
- [base-template-generator](../../../../reference/agents/base-template-generator.md)
- [byzantine-coordinator](../../../../reference/agents/consensus/byzantine-coordinator.md)
- [code-analyzer](../../../../reference/agents/analysis/code-analyzer.md)
- [code-review-swarm](../../../../reference/agents/github/code-review-swarm.md)
- [coder](../../../../reference/agents/core/coder.md)
- [coordinator-swarm-init](../../../../reference/agents/templates/coordinator-swarm-init.md)
- [crdt-synchronizer](../../../../reference/agents/consensus/crdt-synchronizer.md)
- [data-ml-model](../../../../reference/agents/data/ml/data-ml-model.md)
- [docs-api-openapi](../../../../reference/agents/documentation/api-docs/docs-api-openapi.md)
- [github-modes](../../../../reference/agents/github/github-modes.md)
- [github-pr-manager](../../../../reference/agents/templates/github-pr-manager.md)
- [gossip-coordinator](../../../../reference/agents/consensus/gossip-coordinator.md)
- [hierarchical-coordinator](../../../../reference/agents/swarm/hierarchical-coordinator.md)
- [implementer-sparc-coder](../../../../reference/agents/templates/implementer-sparc-coder.md)
- [issue-tracker](../../../../reference/agents/github/issue-tracker.md)
- [memory-coordinator](../../../../reference/agents/templates/memory-coordinator.md)
- [mesh-coordinator](../../../../reference/agents/swarm/mesh-coordinator.md)
- [migration-plan](../../../../reference/agents/templates/migration-plan.md)
- [multi-repo-swarm](../../../../reference/agents/github/multi-repo-swarm.md)
- [ops-cicd-github](../../../../reference/agents/devops/ci-cd/ops-cicd-github.md)
- [orchestrator-task](../../../../reference/agents/templates/orchestrator-task.md)
- [performance-analyzer](../../../../reference/agents/templates/performance-analyzer.md)
- [performance-benchmarker](../../../../reference/agents/consensus/performance-benchmarker.md)
- [planner](../../../../reference/agents/core/planner.md)
- [pr-manager](../../../../reference/agents/github/pr-manager.md)
- [production-validator](../../../../reference/agents/testing/validation/production-validator.md)
- [project-board-sync](../../../../reference/agents/github/project-board-sync.md)
- [pseudocode](../../../../reference/agents/sparc/pseudocode.md)
- [quorum-manager](../../../../reference/agents/consensus/quorum-manager.md)
- [raft-manager](../../../../reference/agents/consensus/raft-manager.md)
- [refinement](../../../../reference/agents/sparc/refinement.md)
- [release-manager](../../../../reference/agents/github/release-manager.md)
- [release-swarm](../../../../reference/agents/github/release-swarm.md)
- [repo-architect](../../../../reference/agents/github/repo-architect.md)
- [researcher](../../../../reference/agents/core/researcher.md)
- [reviewer](../../../../reference/agents/core/reviewer.md)
- [security-manager](../../../../reference/agents/consensus/security-manager.md)
- [sparc-coordinator](../../../../reference/agents/templates/sparc-coordinator.md)
- [spec-mobile-react-native](../../../../reference/agents/specialized/mobile/spec-mobile-react-native.md)
- [specification](../../../../reference/agents/sparc/specification.md)
- [swarm-issue](../../../../reference/agents/github/swarm-issue.md)
- [swarm-pr](../../../../reference/agents/github/swarm-pr.md)
- [sync-coordinator](../../../../reference/agents/github/sync-coordinator.md)
- [tdd-london-swarm](../../../../reference/agents/testing/unit/tdd-london-swarm.md)
- [tester](../../../../reference/agents/core/tester.md)
- [workflow-automation](../../../../reference/agents/github/workflow-automation.md)
