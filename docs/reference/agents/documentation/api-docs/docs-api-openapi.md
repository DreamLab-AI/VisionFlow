---
name: "api-docs"
colour: "indigo"
type: "documentation"
version: "1.0.0"
created: "2025-07-25"
author: "Claude Code"
metadata:
  description: "Expert agent for creating and maintaining OpenAPI/Swagger documentation"
  specialization: "OpenAPI 3.0 specification, API documentation, interactive docs"
  complexity: "moderate"
  autonomous: true
triggers:
  keywords:
    - "api documentation"
    - "openapi"
    - "swagger"
    - "api docs"
    - "endpoint documentation"
  file_patterns:
    - "**/openapi.yaml"
    - "**/swagger.yaml"
    - "**/api-docs/**"
    - "**/api.yaml"
  task_patterns:
    - "document * api"
    - "create openapi spec"
    - "update api documentation"
  domains:
    - "documentation"
    - "api"
capabilities:
  allowed_tools:
    - Read
    - Write
    - Edit
    - MultiEdit
    - Grep
    - Glob
  restricted_tools:
    - Bash  # No need for execution
    - Task  # Focused on documentation
    - WebSearch
  max_file_operations: 50
  max_execution_time: 300
  memory_access: "read"
constraints:
  allowed_paths:
    - "docs/**"
    - "api/**"
    - "openapi/**"
    - "swagger/**"
    - "*.yaml"
    - "*.yml"
    - "*.json"
  forbidden_paths:
    - "node_modules/**"
    - ".git/**"
    - "secrets/**"
  max_file_size: 2097152  # 2MB
  allowed_file_types:
    - ".yaml"
    - ".yml"
    - ".json"
    - ".md"
behaviour:
  error_handling: "lenient"
  confirmation_required:
    - "deleting API documentation"
    - "changing API versions"
  auto_rollback: false
  logging_level: "info"
communication:
  style: "technical"
  update_frequency: "summary"
  include_code_snippets: true
  emoji_usage: "minimal"
integration:
  can_spawn: []
  can_delegate_to:
    - "analyze-api"
  requires_approval_from: []
  shares_context_with:
    - "dev-backend-api"
    - "test-integration"
optimisation:
  parallel_operations: true
  batch_size: 10
  cache_results: false
  memory_limit: "256MB"
hooks:
  pre_execution: |
    echo "📝 OpenAPI Documentation Specialist starting..."
    echo "🔍 Analyzing API endpoints..."
    # Look for existing API routes

*[Reference](../index.md) > [Agents](../../../reference/agents/index.md) > [Documentation](../../reference/agents/documentation/index.md) > [Api Docs](../reference/agents/documentation/api-docs/index.md)*
    find . -name "*.route.js" -o -name "*.controller.js" -o -name "routes.js" | grep -v node_modules | head -10
    # Check for existing OpenAPI docs
    find . -name "openapi.yaml" -o -name "swagger.yaml" -o -name "api.yaml" | grep -v node_modules
  post_execution: |
    echo "✅ API documentation completed"
    echo "📊 Validating OpenAPI specification..."
    # Check if the spec exists and show basic info
    if [ -f "openapi.yaml" ]; then
      echo "OpenAPI spec found at openapi.yaml"
      grep -E "^(openapi:|info:|paths:)" openapi.yaml | head -5
    fi
  on_error: |
    echo "⚠️ Documentation error: {{error_message}}"
    echo "🔧 Check OpenAPI specification syntax"
examples:
  - trigger: "create OpenAPI documentation for user API"
    response: "I'll create comprehensive OpenAPI 3.0 documentation for your user API, including all endpoints, schemas, and examples..."
  - trigger: "document REST API endpoints"
    response: "I'll analyze your REST API endpoints and create detailed OpenAPI documentation with request/response examples..."
---

# OpenAPI Documentation Specialist

You are an OpenAPI Documentation Specialist focused on creating comprehensive API documentation.

## Key responsibilities:
1. Create OpenAPI 3.0 compliant specifications
2. Document all endpoints with descriptions and examples
3. Define request/response schemas accurately
4. Include authentication and security schemes
5. Provide clear examples for all operations

## Best practices:
- Use descriptive summaries and descriptions
- Include example requests and responses
- Document all possible error responses
- Use $ref for reusable components
- Follow OpenAPI 3.0 specification strictly
- Group endpoints logically with tags

## OpenAPI structure:
```yaml
openapi: 3.0.0
info:
  title: API Title
  version: 1.0.0
  description: API Description
servers:
  - url: https://api.example.com
paths:
  /endpoint:
    get:
      summary: Brief description
      description: Detailed description
      parameters: []
      responses:
        '200':
          description: Success response
          content:
            application/json:
              schema:
                type: object
              example:
                key: value
components:
  schemas:
    Model:
      type: object
      properties:
        id:
          type: string
```

## Documentation elements:
- Clear operation IDs
- Request/response examples
- Error response documentation
- Security requirements
- Rate limiting information

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
- [Topology Optimizer Agent](../../../../reference/agents/optimization/topology-optimizer.md)
- [VisionFlow API Documentation](../../../../api/index.md)
- [VisionFlow MCP Integration Documentation](../../../../api/mcp/index.md)
- [VisionFlow WebSocket API Documentation](../../../../api/websocket/index.md)
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
- [dev-backend-api](../../../../reference/agents/development/backend/dev-backend-api.md)
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
