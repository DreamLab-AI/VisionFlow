# Content Migration Map

**Generated**: 2026-01-14
**Total Files**: 280 markdown files + 41 non-markdown files
**Target Structure**: Diataxis-compliant (tutorials, guides, reference, architecture)

---

## Migration Overview

| Category | Current Count | Action |
|----------|---------------|--------|
| Keep and Move | ~120 files | Reorganize to new paths |
| Consolidate | ~40 files | Merge duplicates |
| Delete | ~80 files | Remove after content extraction |
| Non-Markdown | ~41 files | Relocate or remove |

---

## Files to KEEP and MOVE

### Root Level Files (Move to Organized Locations)

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `01-GETTING_STARTED.md` | `getting-started/README.md` | Primary entry point, update links |
| `OVERVIEW.md` | `getting-started/overview.md` | Update navigation links |
| `ARCHITECTURE_OVERVIEW.md` | `architecture/overview.md` | Merge with ARCHITECTURE_COMPLETE.md |
| `TECHNOLOGY_CHOICES.md` | `architecture/technology-choices.md` | Update cross-references |
| `DEVELOPER_JOURNEY.md` | `development/developer-journey.md` | Update internal links |
| `CONTRIBUTION.md` | `development/contributing.md` | Merge with guides/contributing.md |
| `MAINTENANCE.md` | `development/maintenance.md` | Update links |
| `SOLID_POD_CREATION.md` | `guides/features/solid-pod-creation.md` | Move to features |
| `README.md` | `README.md` | Keep as main index, simplify |

### Tutorials Directory (Keep Structure)

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `tutorials/01-installation.md` | `getting-started/installation.md` | Update links |
| `tutorials/02-first-graph.md` | `getting-started/first-graph.md` | Update links |
| `tutorials/neo4j-quick-start.md` | `getting-started/neo4j-quickstart.md` | Update links |

### Guides Directory (Reorganize)

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `guides/README.md` | `guides/README.md` | Update index |
| `guides/configuration.md` | `guides/configuration.md` | Keep |
| `guides/deployment.md` | `guides/deployment.md` | Keep |
| `guides/security.md` | `guides/security.md` | Keep |
| `guides/testing-guide.md` | `development/testing.md` | Relocate to development |
| `guides/troubleshooting.md` | `guides/troubleshooting.md` | Keep |
| `guides/navigation-guide.md` | `guides/navigation.md` | Rename |
| `guides/ontology-parser.md` | `guides/ontology/parser.md` | Move to ontology subfolder |
| `guides/ontology-reasoning-integration.md` | `guides/ontology/reasoning-integration.md` | Move to ontology subfolder |
| `guides/ontology-semantic-forces.md` | `guides/ontology/semantic-forces.md` | Move to ontology subfolder |
| `guides/ontology-storage-guide.md` | `guides/ontology/storage.md` | Move to ontology subfolder |
| `guides/neo4j-integration.md` | `guides/database/neo4j-integration.md` | Create database subfolder |
| `guides/neo4j-migration.md` | `guides/database/neo4j-migration.md` | Move to database subfolder |
| `guides/neo4j-implementation-roadmap.md` | `guides/database/neo4j-roadmap.md` | Move to database subfolder |
| `guides/stress-majorization-guide.md` | `guides/visualization/stress-majorization.md` | Create viz subfolder |
| `guides/hierarchy-integration.md` | `guides/visualization/hierarchy-integration.md` | Move to viz subfolder |
| `guides/vircadia-xr-complete-guide.md` | `guides/xr/vircadia-complete.md` | Create XR subfolder |
| `guides/vircadia-multi-user-guide.md` | `guides/xr/vircadia-multiuser.md` | Move to XR subfolder |
| `guides/solid-integration.md` | `guides/integrations/solid.md` | Create integrations subfolder |
| `guides/docker-compose-guide.md` | `guides/infrastructure/docker-compose.md` | Keep in infrastructure |
| `guides/docker-environment-setup.md` | `guides/infrastructure/docker-environment.md` | Keep in infrastructure |
| `guides/telemetry-logging.md` | `guides/operations/telemetry-logging.md` | Create operations subfolder |
| `guides/semantic-features-implementation.md` | `guides/features/semantic-features.md` | Keep in features |
| `guides/agent-orchestration.md` | `guides/agents/orchestration.md` | Create agents subfolder |
| `guides/orchestrating-agents.md` | DELETE | Duplicate of above |
| `guides/multi-agent-skills.md` | `guides/agents/skills.md` | Move to agents subfolder |
| `guides/pipeline-admin-api.md` | `reference/api/pipeline-admin.md` | Move to reference |
| `guides/extending-the-system.md` | `development/extending.md` | Move to development |
| `guides/development-workflow.md` | `development/workflow.md` | Move to development |
| `guides/graphserviceactor-migration.md` | `architecture/migrations/graphserviceactor.md` | Move to architecture |
| `guides/contributing.md` | `development/contributing.md` | Merge with CONTRIBUTION.md |

### Guides/Developer Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `guides/developer/README.md` | `development/README.md` | Merge both readme.md files |
| `guides/developer/01-development-setup.md` | `development/setup.md` | Rename, update links |
| `guides/developer/02-project-structure.md` | `development/project-structure.md` | Update links |
| `guides/developer/04-adding-features.md` | `development/adding-features.md` | Update links |
| `guides/developer/06-contributing.md` | `development/contributing.md` | Merge with others |
| `guides/developer/json-serialization-patterns.md` | `development/patterns/json-serialization.md` | Create patterns subfolder |
| `guides/developer/test-execution.md` | `development/testing/test-execution.md` | Create testing subfolder |
| `guides/developer/websocket-best-practices.md` | `development/patterns/websocket-best-practices.md` | Move to patterns |

### Guides/Features Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `guides/features/auth-user-settings.md` | `guides/features/auth-settings.md` | Rename |
| `guides/features/filtering-nodes.md` | `guides/features/node-filtering.md` | Rename |
| `guides/features/github-pagination-fix.md` | DELETE | Implementation note, not guide |
| `guides/features/intelligent-pathfinding.md` | `guides/features/pathfinding.md` | Rename |
| `guides/features/local-file-sync-strategy.md` | `guides/features/file-sync.md` | Rename |
| `guides/features/natural-language-queries.md` | `guides/features/nlq.md` | Rename |
| `guides/features/nostr-auth.md` | `guides/features/nostr-auth.md` | Keep |
| `guides/features/ontology-sync-enhancement.md` | `guides/ontology/sync-enhancement.md` | Move to ontology |
| `guides/features/semantic-forces.md` | `guides/features/semantic-forces.md` | Keep |
| `guides/features/settings-authentication.md` | `guides/features/settings-auth.md` | Merge with auth-settings |
| `guides/features/MOVED.md` | DELETE | Placeholder file |

### Guides/AI-Models Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `guides/ai-models/README.md` | `guides/integrations/ai-models/README.md` | Move under integrations |
| `guides/ai-models/INTEGRATION_SUMMARY.md` | `guides/integrations/ai-models/summary.md` | Rename |
| `guides/ai-models/deepseek-deployment.md` | `guides/integrations/ai-models/deepseek.md` | Consolidate |
| `guides/ai-models/deepseek-verification.md` | DELETE | Merge into deepseek.md |
| `guides/ai-models/perplexity-integration.md` | `guides/integrations/ai-models/perplexity.md` | Rename |
| `guides/ai-models/ragflow-integration.md` | `guides/integrations/ai-models/ragflow.md` | Rename |

### Guides/Infrastructure Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `guides/infrastructure/README.md` | `guides/infrastructure/README.md` | Keep |
| `guides/infrastructure/architecture.md` | `architecture/infrastructure.md` | Move to architecture |
| `guides/infrastructure/docker-environment.md` | `guides/infrastructure/docker.md` | Merge with docker-compose |
| `guides/infrastructure/goalie-integration.md` | `guides/integrations/goalie.md` | Move to integrations |
| `guides/infrastructure/port-configuration.md` | `reference/configuration/ports.md` | Move to reference |
| `guides/infrastructure/tools.md` | `development/tools.md` | Move to development |
| `guides/infrastructure/troubleshooting.md` | `guides/troubleshooting.md` | Merge with root troubleshooting |

### Guides/Client Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `guides/client/state-management.md` | `architecture/client/state-management.md` | Move to architecture |
| `guides/client/three-js-rendering.md` | `architecture/client/threejs-rendering.md` | Move to architecture |
| `guides/client/xr-integration.md` | `architecture/client/xr-integration.md` | Move to architecture |

### Guides/Architecture Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `guides/architecture/actor-system.md` | `architecture/server/actor-system.md` | Move to architecture |

### Guides/Operations Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `guides/operations/pipeline-operator-runbook.md` | `guides/operations/pipeline-runbook.md` | Rename |

### Guides/Migration Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `guides/migration/json-to-binary-protocol.md` | `architecture/migrations/json-to-binary.md` | Move to architecture |

### Reference Directory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `reference/README.md` | `reference/README.md` | Update index |
| `reference/INDEX.md` | DELETE | Merge into README.md |
| `reference/API_REFERENCE.md` | `reference/api/README.md` | Move to api subfolder |
| `reference/CONFIGURATION_REFERENCE.md` | `reference/configuration/README.md` | Move to config subfolder |
| `reference/DATABASE_SCHEMA_REFERENCE.md` | `reference/database/README.md` | Move to database subfolder |
| `reference/ERROR_REFERENCE.md` | `reference/errors/README.md` | Move to errors subfolder |
| `reference/PROTOCOL_REFERENCE.md` | `reference/protocols/README.md` | Move to protocols subfolder |
| `reference/api-complete-reference.md` | DELETE | Merge into api/README.md |
| `reference/code-quality-status.md` | DELETE | Build artifact, not reference |
| `reference/implementation-status.md` | DELETE | Outdated status doc |
| `reference/performance-benchmarks.md` | `reference/benchmarks.md` | Keep |
| `reference/physics-implementation.md` | `architecture/physics-implementation.md` | Move to architecture |
| `reference/websocket-protocol.md` | `reference/protocols/websocket.md` | Keep in protocols |
| `reference/error-codes.md` | `reference/errors/error-codes.md` | Keep in errors |

### Reference/API Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `reference/api/README.md` | `reference/api/README.md` | Update |
| `reference/api/01-authentication.md` | `reference/api/authentication.md` | Rename |
| `reference/api/03-websocket.md` | `reference/api/websocket.md` | Rename |
| `reference/api/API_DESIGN_ANALYSIS.md` | DELETE | Internal analysis, not reference |
| `reference/api/API_IMPROVEMENT_TEMPLATES.md` | DELETE | Internal templates |
| `reference/api/pathfinding-examples.md` | `reference/api/pathfinding.md` | Rename |
| `reference/api/rest-api-complete.md` | DELETE | Merge into rest-api-reference |
| `reference/api/rest-api-reference.md` | `reference/api/rest.md` | Rename |
| `reference/api/semantic-features-api.md` | `reference/api/semantic-features.md` | Rename |
| `reference/api/solid-api.md` | `reference/api/solid.md` | Rename |

### Reference/Database Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `reference/database/neo4j-persistence-analysis.md` | DELETE | Analysis, not reference |
| `reference/database/ontology-schema-v2.md` | `reference/database/ontology-schema.md` | Rename |
| `reference/database/schemas.md` | `reference/database/schemas.md` | Keep |
| `reference/database/solid-pod-schema.md` | `reference/database/solid-schema.md` | Rename |
| `reference/database/user-settings-schema.md` | `reference/database/user-settings.md` | Rename |

### Reference/Protocols Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `reference/protocols/binary-websocket.md` | `reference/protocols/binary-websocket.md` | Keep |

### Explanations Directory (Move to Architecture)

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `explanations/system-overview.md` | `architecture/system-overview.md` | Move |
| `explanations/architecture/README.md` | DELETE | Merge into architecture/README.md |
| `explanations/architecture/adapter-patterns.md` | `architecture/patterns/adapters.md` | Move |
| `explanations/architecture/analytics-visualization.md` | `architecture/visualization/analytics.md` | Move |
| `explanations/architecture/api-handlers-reference.md` | `reference/api/handlers.md` | Move to reference |
| `explanations/architecture/cqrs-directive-template.md` | `architecture/patterns/cqrs.md` | Move |
| `explanations/architecture/data-flow-complete.md` | `architecture/data-flow.md` | Move |
| `explanations/architecture/database-architecture.md` | `architecture/database.md` | Move |
| `explanations/architecture/event-driven-architecture.md` | `architecture/patterns/event-driven.md` | Move |
| `explanations/architecture/github-sync-service-design.md` | `architecture/integrations/github-sync.md` | Move |
| `explanations/architecture/gpu-semantic-forces.md` | `architecture/gpu/semantic-forces.md` | Move |
| `explanations/architecture/hexagonal-cqrs.md` | `architecture/patterns/hexagonal-cqrs.md` | Move |
| `explanations/architecture/hierarchical-visualization.md` | `architecture/visualization/hierarchical.md` | Move |
| `explanations/architecture/integration-patterns.md` | `architecture/patterns/integrations.md` | Move |
| `explanations/architecture/multi-agent-system.md` | `architecture/agents/multi-agent.md` | Move |
| `explanations/architecture/ontology-analysis.md` | `architecture/ontology/analysis.md` | Move |
| `explanations/architecture/ontology-physics-integration.md` | `architecture/ontology/physics-integration.md` | Move |
| `explanations/architecture/ontology-reasoning-pipeline.md` | `architecture/ontology/reasoning-pipeline.md` | Move |
| `explanations/architecture/ontology-storage-architecture.md` | `architecture/ontology/storage.md` | Move |
| `explanations/architecture/pipeline-integration.md` | `architecture/pipelines/integration.md` | Move |
| `explanations/architecture/pipeline-sequence-diagrams.md` | `architecture/pipelines/sequence-diagrams.md` | Move |
| `explanations/architecture/quick-reference.md` | DELETE | Merge into README |
| `explanations/architecture/reasoning-data-flow.md` | `architecture/ontology/reasoning-data-flow.md` | Move |
| `explanations/architecture/reasoning-tests-summary.md` | DELETE | Test summary, not architecture |
| `explanations/architecture/ruvector-integration.md` | `architecture/integrations/ruvector.md` | Move |
| `explanations/architecture/semantic-forces-system.md` | `architecture/physics/semantic-forces-system.md` | Move |
| `explanations/architecture/semantic-physics-system.md` | `architecture/physics/semantic-physics.md` | Merge |
| `explanations/architecture/semantic-physics.md` | DELETE | Merge with above |
| `explanations/architecture/services-architecture.md` | `architecture/services.md` | Move |
| `explanations/architecture/services-layer.md` | DELETE | Merge with services.md |
| `explanations/architecture/stress-majorization.md` | `architecture/visualization/stress-majorization.md` | Move |
| `explanations/architecture/xr-immersive-system.md` | `architecture/xr/immersive-system.md` | Move |

### Explanations/Architecture/Core Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `explanations/architecture/core/client.md` | `architecture/client/overview.md` | Move |
| `explanations/architecture/core/server.md` | `architecture/server/overview.md` | Move |
| `explanations/architecture/core/visualization.md` | `architecture/visualization/overview.md` | Move |

### Explanations/Architecture/Components Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `explanations/architecture/components/websocket-protocol.md` | `architecture/protocols/websocket.md` | Move |

### Explanations/Architecture/Decisions Subdirectory (ADRs)

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `explanations/architecture/decisions/0001-neo4j-persistent-with-filesystem-sync.md` | `architecture/decisions/0001-neo4j-persistence.md` | Rename, keep ADR format |

### Explanations/Architecture/GPU Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `explanations/architecture/gpu/README.md` | `architecture/gpu/README.md` | Move |
| `explanations/architecture/gpu/communication-flow.md` | `architecture/gpu/communication-flow.md` | Move |
| `explanations/architecture/gpu/optimizations.md` | `architecture/gpu/optimizations.md` | Move |

### Explanations/Architecture/Ports Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `explanations/architecture/ports/01-overview.md` | `architecture/ports/overview.md` | Rename |
| `explanations/architecture/ports/02-settings-repository.md` | `architecture/ports/settings-repository.md` | Rename |
| `explanations/architecture/ports/03-knowledge-graph-repository.md` | `architecture/ports/knowledge-graph-repository.md` | Rename |
| `explanations/architecture/ports/04-ontology-repository.md` | `architecture/ports/ontology-repository.md` | Rename |
| `explanations/architecture/ports/05-inference-engine.md` | `architecture/ports/inference-engine.md` | Rename |
| `explanations/architecture/ports/06-gpu-physics-adapter.md` | `architecture/ports/gpu-physics-adapter.md` | Rename |
| `explanations/architecture/ports/07-gpu-semantic-analyzer.md` | `architecture/ports/gpu-semantic-analyzer.md` | Rename |

### Explanations/Ontology Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `explanations/ontology/client-side-hierarchical-lod.md` | `architecture/ontology/client-hierarchical-lod.md` | Move |
| `explanations/ontology/enhanced-parser.md` | `architecture/ontology/enhanced-parser.md` | Move |
| `explanations/ontology/hierarchical-visualization.md` | `architecture/ontology/hierarchical-visualization.md` | Move |
| `explanations/ontology/intelligent-pathfinding-system.md` | `architecture/ontology/pathfinding.md` | Move |
| `explanations/ontology/neo4j-integration.md` | `architecture/ontology/neo4j-integration.md` | Move |
| `explanations/ontology/ontology-pipeline-integration.md` | `architecture/ontology/pipeline-integration.md` | Move |
| `explanations/ontology/ontology-typed-system.md` | `architecture/ontology/typed-system.md` | Move |
| `explanations/ontology/reasoning-engine.md` | `architecture/ontology/reasoning-engine.md` | Move |

### Explanations/Physics Subdirectory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `explanations/physics/semantic-forces-actor.md` | `architecture/physics/semantic-forces-actor.md` | Move |
| `explanations/physics/semantic-forces.md` | `architecture/physics/semantic-forces.md` | Move |

### Concepts Directory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `concepts/architecture/core/client.md` | DELETE | Duplicate of explanations |
| `concepts/architecture/core/server.md` | DELETE | Duplicate of explanations |

### Architecture Directory (Root)

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `architecture/HEXAGONAL_ARCHITECTURE_STATUS.md` | DELETE | Status doc, not architecture |
| `architecture/PROTOCOL_MATRIX.md` | `reference/protocols/matrix.md` | Move to reference |
| `architecture/VIRCADIA_BABYLON_CONSOLIDATION_ANALYSIS.md` | DELETE | Analysis artifact |
| `architecture/blender-mcp-unified-architecture.md` | `architecture/integrations/blender-mcp.md` | Move |
| `architecture/phase1-completion.md` | DELETE | Sprint artifact |
| `architecture/skill-mcp-classification.md` | `architecture/agents/skill-mcp-classification.md` | Move |
| `architecture/skills-refactoring-plan.md` | DELETE | Planning artifact |
| `architecture/solid-sidecar-architecture.md` | `architecture/integrations/solid-sidecar.md` | Move |
| `architecture/user-agent-pod-design.md` | `architecture/agents/user-agent-pod.md` | Move |
| `architecture/visionflow-distributed-systems-assessment.md` | `architecture/distributed-systems.md` | Keep |

### Diagrams Directory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `diagrams/README.md` | `architecture/diagrams/README.md` | Move |
| `diagrams/ASCII-TO-MERMAID-CONVERSION-REPORT.md` | DELETE | Migration artifact |
| `diagrams/cross-reference-matrix.md` | `architecture/diagrams/cross-reference.md` | Move |
| `diagrams/mermaid-library/README.md` | `architecture/diagrams/mermaid-library/README.md` | Move |
| `diagrams/mermaid-library/00-mermaid-style-guide.md` | `development/diagrams/style-guide.md` | Move to development |
| `diagrams/mermaid-library/01-system-architecture-overview.md` | `architecture/diagrams/system-overview.md` | Move |
| `diagrams/mermaid-library/02-data-flow-diagrams.md` | `architecture/diagrams/data-flow.md` | Move |
| `diagrams/mermaid-library/03-deployment-infrastructure.md` | `architecture/diagrams/deployment.md` | Move |
| `diagrams/mermaid-library/04-agent-orchestration.md` | `architecture/diagrams/agent-orchestration.md` | Move |
| `diagrams/architecture/backend-api-architecture-complete.md` | `architecture/diagrams/backend-api.md` | Move |
| `diagrams/client/rendering/threejs-pipeline-complete.md` | `architecture/diagrams/threejs-pipeline.md` | Move |
| `diagrams/client/state/state-management-complete.md` | `architecture/diagrams/state-management.md` | Move |
| `diagrams/client/xr/xr-architecture-complete.md` | `architecture/diagrams/xr-architecture.md` | Move |
| `diagrams/data-flow/complete-data-flows.md` | `architecture/diagrams/data-flows.md` | Move |
| `diagrams/infrastructure/gpu/cuda-architecture-complete.md` | `architecture/diagrams/cuda-architecture.md` | Move |
| `diagrams/infrastructure/testing/test-architecture.md` | `development/testing/architecture.md` | Move |
| `diagrams/infrastructure/websocket/binary-protocol-complete.md` | `architecture/diagrams/binary-protocol.md` | Move |
| `diagrams/server/actors/actor-system-complete.md` | `architecture/diagrams/actor-system.md` | Move |
| `diagrams/server/agents/agent-system-architecture.md` | `architecture/diagrams/agent-system.md` | Move |
| `diagrams/server/api/rest-api-architecture.md` | `architecture/diagrams/rest-api.md` | Move |

### Testing Directory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `testing/PHASE8_COMPLETION.md` | DELETE | Sprint artifact |
| `testing/TESTING_GUIDE.md` | `development/testing/guide.md` | Move |

### Research Directory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `research/QUIC_HTTP3_ANALYSIS.md` | `architecture/research/quic-http3.md` | Move |
| `research/graph-visualization-sota-analysis.md` | `architecture/research/graph-visualization-sota.md` | Move |
| `research/threejs-vs-babylonjs-graph-visualization.md` | `architecture/research/threejs-vs-babylonjs.md` | Move |

### Analysis Directory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `analysis/DUAL_RENDERER_OVERHEAD_ANALYSIS.md` | DELETE | Internal analysis artifact |
| `analysis/ontology-knowledge-skills-analysis.md` | DELETE | Internal analysis artifact |
| `analysis/ontology-skills-cluster-analysis.md` | DELETE | Internal analysis artifact |

### Audits Directory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `audits/README.md` | DELETE | Build artifacts |
| `audits/ascii-diagram-deprecation-audit.md` | DELETE | Migration artifact |
| `audits/neo4j-migration-action-plan.md` | DELETE | Merge into neo4j-migration |
| `audits/neo4j-migration-summary.md` | DELETE | Merge into neo4j-migration |
| `audits/neo4j-settings-migration-audit.md` | DELETE | Merge into neo4j-migration |

### Sprints Directory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `sprints/heroic-refactor-sprint-2026-01.md` | DELETE | Sprint artifact |

### Multi-Agent-Docker Directory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `multi-agent-docker/ANTIGRAVITY.md` | `guides/agents/antigravity.md` | Move |
| `multi-agent-docker/SKILLS.md` | `guides/agents/skills.md` | Move |
| `multi-agent-docker/TERMINAL_GRID.md` | DELETE | Internal doc |
| `multi-agent-docker/comfyui-sam3d-setup.md` | `guides/integrations/comfyui-sam3d.md` | Move |
| `multi-agent-docker/hyprland-migration-summary.md` | DELETE | Migration artifact |
| `multi-agent-docker/upstream-analysis.md` | DELETE | Internal analysis |
| `multi-agent-docker/x-fluxagent-adaptation-plan.md` | DELETE | Planning artifact |

### Assets Directory

| Current Path | New Path | Changes Needed |
|--------------|----------|----------------|
| `assets/diagrams/sparc-turboflow-architecture.md` | `architecture/diagrams/sparc-turboflow.md` | Move |
| `assets/screenshots/.gitkeep` | `assets/screenshots/.gitkeep` | Keep |

---

## Files to CONSOLIDATE

### Navigation/Index Files (Merge into README.md)

| Files to Merge | Target File | Content from Each |
|----------------|-------------|-------------------|
| `INDEX.md`, `NAVIGATION.md`, `QUICK_NAVIGATION.md` | `README.md` | Best navigation structure from INDEX, quick links from NAVIGATION |

### Architecture Files

| Files to Merge | Target File | Content from Each |
|----------------|-------------|-------------------|
| `ARCHITECTURE_OVERVIEW.md`, `ARCHITECTURE_COMPLETE.md` | `architecture/overview.md` | OVERVIEW for summary, COMPLETE for details |
| `explanations/architecture/semantic-physics-system.md`, `explanations/architecture/semantic-physics.md` | `architecture/physics/semantic-physics.md` | Merge duplicates |
| `explanations/architecture/services-architecture.md`, `explanations/architecture/services-layer.md` | `architecture/services.md` | Merge service docs |

### Getting Started Files

| Files to Merge | Target File | Content from Each |
|----------------|-------------|-------------------|
| `01-GETTING_STARTED.md`, `GETTING_STARTED_WITH_UNIFIED_DOCS.md` | `getting-started/README.md` | Best elements of each |

### Contributing Files

| Files to Merge | Target File | Content from Each |
|----------------|-------------|-------------------|
| `CONTRIBUTION.md`, `guides/contributing.md`, `guides/developer/06-contributing.md` | `development/contributing.md` | All unique content |

### Troubleshooting Files

| Files to Merge | Target File | Content from Each |
|----------------|-------------|-------------------|
| `guides/troubleshooting.md`, `guides/infrastructure/troubleshooting.md` | `guides/troubleshooting.md` | All unique content |

### API Reference Files

| Files to Merge | Target File | Content from Each |
|----------------|-------------|-------------------|
| `reference/API_REFERENCE.md`, `reference/api-complete-reference.md`, `reference/api/rest-api-complete.md`, `reference/api/rest-api-reference.md` | `reference/api/README.md` | Comprehensive API reference |

### CUDA/GPU Files

| Files to Merge | Target File | Content from Each |
|----------------|-------------|-------------------|
| `CUDA_KERNEL_ANALYSIS_REPORT.md`, `CUDA_KERNEL_AUDIT_REPORT.md`, `CUDA_OPTIMIZATION_SUMMARY.md` | `architecture/gpu/cuda-optimization.md` | Consolidated GPU documentation |

### ComfyUI Files

| Files to Merge | Target File | Content from Each |
|----------------|-------------|-------------------|
| `comfyui-integration-design.md`, `comfyui-management-api-integration-summary.md`, `comfyui-service-integration.md` | `guides/integrations/comfyui.md` | Consolidated integration guide |

### Code Quality Files

| Files to Merge | Target File | Content from Each |
|----------------|-------------|-------------------|
| `CODE_QUALITY_ANALYSIS.md`, `code-quality-analysis-report.md` | DELETE | Build artifacts, not docs |

### Neo4j Migration Files

| Files to Merge | Target File | Content from Each |
|----------------|-------------|-------------------|
| `guides/neo4j-migration.md`, `audits/neo4j-migration-*.md` | `guides/database/neo4j-migration.md` | Complete migration guide |

### Developer README Files

| Files to Merge | Target File | Content from Each |
|----------------|-------------|-------------------|
| `guides/developer/README.md`, `guides/developer/readme.md` | `development/README.md` | Merge duplicates (case diff) |

### Infrastructure README Files

| Files to Merge | Target File | Content from Each |
|----------------|-------------|-------------------|
| `guides/infrastructure/README.md`, `guides/infrastructure/readme.md` | `guides/infrastructure/README.md` | Merge if different |

---

## Files to DELETE (After Content Extraction)

### Sprint/Phase Artifacts (Outdated)

| File | Reason | Valuable Content -> Destination |
|------|--------|-------------------------------|
| `phase6-integration-guide.md` | Outdated sprint doc | Integration patterns -> `guides/xr/vircadia-complete.md` |
| `phase6-multiuser-sync-implementation.md` | Outdated sprint doc | None |
| `phase7_broadcast_optimization.md` | Outdated sprint doc | None (implementation notes in code) |
| `phase7_implementation_summary.md` | Outdated sprint doc | None |
| `architecture/phase1-completion.md` | Sprint artifact | None |
| `testing/PHASE8_COMPLETION.md` | Sprint artifact | None |
| `sprints/heroic-refactor-sprint-2026-01.md` | Sprint artifact | None |

### Internal Analysis/Audit Artifacts

| File | Reason | Valuable Content -> Destination |
|------|--------|-------------------------------|
| `API_TEST_IMPLEMENTATION.md` | Internal test notes | None |
| `ASCII_DEPRECATION_COMPLETE.md` | Migration artifact | None |
| `CLIENT_CODE_ANALYSIS.md` | Internal analysis | Architecture -> `architecture/client/overview.md` |
| `CODE_QUALITY_ANALYSIS.md` | Build artifact | None |
| `QA_VALIDATION_FINAL.md` | QA artifact | None |
| `TEST_COVERAGE_ANALYSIS.md` | Build artifact | None |
| `VALIDATION_CHECKLIST.md` | Internal checklist | None |
| `PROJECT_CONSOLIDATION_PLAN.md` | Planning doc | None |
| `architecture_analysis_report.md` | Analysis artifact | None |
| `code-quality-analysis-report.md` | Build artifact | None |
| `conversion-report.md` | Migration artifact | None |
| `gpu-fix-summary.md` | Fix notes | None |
| `observability-analysis.md` | Analysis artifact | None |
| `refactoring_guide.md` | Internal guide | None |
| `VISIONFLOW_WARDLEY_ANALYSIS.md` | Strategy doc | None (not user-facing) |
| `visionflow-architecture-analysis.md` | Analysis artifact | None |
| `analysis/DUAL_RENDERER_OVERHEAD_ANALYSIS.md` | Analysis artifact | None |
| `analysis/ontology-knowledge-skills-analysis.md` | Analysis artifact | None |
| `analysis/ontology-skills-cluster-analysis.md` | Analysis artifact | None |
| `architecture/HEXAGONAL_ARCHITECTURE_STATUS.md` | Status doc | None |
| `architecture/VIRCADIA_BABYLON_CONSOLIDATION_ANALYSIS.md` | Analysis artifact | None |
| `architecture/skills-refactoring-plan.md` | Planning artifact | None |
| `audits/README.md` | Index for artifacts | None |
| `audits/ascii-diagram-deprecation-audit.md` | Migration artifact | None |
| `audits/neo4j-migration-action-plan.md` | Merged elsewhere | None |
| `audits/neo4j-migration-summary.md` | Merged elsewhere | None |
| `audits/neo4j-settings-migration-audit.md` | Merged elsewhere | None |
| `diagrams/ASCII-TO-MERMAID-CONVERSION-REPORT.md` | Migration artifact | None |
| `multi-agent-docker/TERMINAL_GRID.md` | Internal doc | None |
| `multi-agent-docker/hyprland-migration-summary.md` | Migration artifact | None |
| `multi-agent-docker/upstream-analysis.md` | Analysis artifact | None |
| `multi-agent-docker/x-fluxagent-adaptation-plan.md` | Planning artifact | None |

### Duplicate/Superseded Content

| File | Reason | Valuable Content -> Destination |
|------|--------|-------------------------------|
| `INDEX.md` | Merge into README | Best sections -> `README.md` |
| `NAVIGATION.md` | Merge into README | Best sections -> `README.md` |
| `QUICK_NAVIGATION.md` | Merge into README | Best sections -> `README.md` |
| `GETTING_STARTED_WITH_UNIFIED_DOCS.md` | Merge into getting-started | None |
| `explanations/architecture/README.md` | Duplicate index | None |
| `explanations/architecture/quick-reference.md` | Merge into README | None |
| `explanations/architecture/reasoning-tests-summary.md` | Test artifact | None |
| `explanations/architecture/semantic-physics.md` | Duplicate | Merge -> semantic-physics-system.md |
| `explanations/architecture/services-layer.md` | Duplicate | Merge -> services.md |
| `concepts/architecture/core/client.md` | Duplicate of explanations | None |
| `concepts/architecture/core/server.md` | Duplicate of explanations | None |
| `reference/INDEX.md` | Duplicate index | Merge -> README.md |
| `reference/api-complete-reference.md` | Duplicate | Merge -> api/README.md |
| `reference/api/rest-api-complete.md` | Duplicate | Merge -> rest.md |
| `reference/api/API_DESIGN_ANALYSIS.md` | Internal | None |
| `reference/api/API_IMPROVEMENT_TEMPLATES.md` | Internal | None |
| `reference/code-quality-status.md` | Build artifact | None |
| `reference/implementation-status.md` | Outdated | None |
| `reference/database/neo4j-persistence-analysis.md` | Analysis | None |
| `guides/orchestrating-agents.md` | Duplicate | Merge -> agent-orchestration.md |
| `guides/features/MOVED.md` | Placeholder | None |
| `guides/features/github-pagination-fix.md` | Implementation note | None |
| `guides/ai-models/deepseek-verification.md` | Merge into deepseek | None |

### Reports Directory (Build Artifacts)

| File | Reason | Valuable Content -> Destination |
|------|--------|-------------------------------|
| `reports/README.md` | Index for artifacts | None |
| `reports/CODE_COVERAGE_INDEX.md` | Build artifact | None |
| `reports/CONTENT-AUDIT-QUICK-REFERENCE.md` | Audit artifact | None |
| `reports/DIAGRAM_AUDIT_SUMMARY.md` | Audit artifact | None |
| `reports/LINK_FIX_CHECKLIST.md` | Audit artifact | None |
| `reports/LINK_VALIDATION_SUMMARY.md` | Audit artifact | None |
| `reports/README-AUDIT.md` | Audit artifact | None |
| `reports/UNDOCUMENTED_COMPONENTS.md` | Audit artifact | None |
| `reports/ascii-conversion-archive-batch-report.md` | Migration artifact | None |
| `reports/code-coverage.md` | Build artifact | None |
| `reports/consolidation-plan.md` | Planning artifact | None |
| `reports/content-audit.md` | Audit artifact | None |
| `reports/corpus-analysis.md` | Analysis artifact | None |
| `reports/diagram-audit.md` | Audit artifact | None |
| `reports/diataxis-compliance-final-report.md` | Migration artifact | None |
| `reports/frontmatter-quick-reference.md` | Migration artifact | None |
| `reports/frontmatter-remediation-action-items.md` | Migration artifact | None |
| `reports/frontmatter-validation.md` | Migration artifact | None |
| `reports/ia-proposal.md` | Planning artifact | None |
| `reports/link-validation.md` | Audit artifact | None |
| `reports/navigation-design.md` | Planning artifact | None |
| `reports/spelling-audit.md` | Audit artifact | None |

### Scripts Directory (Move Outside Docs)

| File | Reason | Valuable Content -> Destination |
|------|--------|-------------------------------|
| `scripts/README.md` | Scripts index | Move to /scripts |
| `scripts/AUTOMATION_COMPLETE.md` | Status doc | None |
| All `.py` files | Non-docs | Move to /scripts |
| All `.sh` files | Non-docs | Move to /scripts |

---

## Non-Markdown Files to Relocate

### Python Scripts (Move to /scripts)

| File | New Location |
|------|--------------|
| `validate_links.py` | `/scripts/docs/validate_links.py` |
| `validate_links_enhanced.py` | `/scripts/docs/validate_links_enhanced.py` |
| `visionflow_strategic_analysis.py` | DELETE (analysis artifact) |
| `visionflow_wardley_analysis.py` | DELETE (analysis artifact) |
| `scripts/add_frontmatter.py` | `/scripts/docs/add_frontmatter.py` |
| `scripts/diataxis-phase3-fix-links.py` | DELETE (migration artifact) |
| `scripts/diataxis-phase3-frontmatter.py` | DELETE (migration artifact) |
| `scripts/fix_unclosed_blocks.py` | `/scripts/docs/fix_unclosed_blocks.py` |
| `scripts/validate_code_blocks.py` | `/scripts/docs/validate_code_blocks.py` |

### Shell Scripts (Move to /scripts)

| File | New Location |
|------|--------------|
| `scripts/analyze-ascii-patterns.sh` | DELETE (migration artifact) |
| `scripts/check-diataxis-compliance.sh` | `/scripts/docs/check-diataxis-compliance.sh` |
| `scripts/detect-ascii.sh` | DELETE (migration artifact) |
| `scripts/diataxis-cleanup-remaining.sh` | DELETE (migration artifact) |
| `scripts/diataxis-migration.sh` | DELETE (migration artifact) |
| `scripts/extract-diagrams.sh` | DELETE (migration artifact) |
| `scripts/find-orphaned-files.sh` | `/scripts/docs/find-orphaned-files.sh` |
| `scripts/fix-diataxis-categories.sh` | DELETE (migration artifact) |
| `scripts/generate-index.sh` | `/scripts/docs/generate-index.sh` |
| `scripts/generate-reports.sh` | `/scripts/docs/generate-reports.sh` |
| `scripts/update-all-references.sh` | `/scripts/docs/update-all-references.sh` |
| `scripts/validate-all.sh` | `/scripts/docs/validate-all.sh` |
| `scripts/validate-coverage.sh` | `/scripts/docs/validate-coverage.sh` |
| `scripts/validate-frontmatter.sh` | `/scripts/docs/validate-frontmatter.sh` |
| `scripts/validate-links.sh` | `/scripts/docs/validate-links.sh` |
| `scripts/validate-mermaid.sh` | `/scripts/docs/validate-mermaid.sh` |
| `scripts/validate-spelling.sh` | `/scripts/docs/validate-spelling.sh` |
| `scripts/validate-structure.sh` | `/scripts/docs/validate-structure.sh` |

### JSON Files (Remove from Docs)

| File | Action |
|------|--------|
| `link-audit-categorized.json` | DELETE (build artifact) |
| `link-audit-fix-report.json` | DELETE (build artifact) |
| `reports/VALIDATION_METRICS.json` | DELETE (build artifact) |
| `reports/content-audit.json` | DELETE (build artifact) |
| `.claude-flow/metrics/*.json` | KEEP (internal tooling) |

### Text Files (Remove from Docs)

| File | Action |
|------|--------|
| `reports/COVERAGE_SUMMARY.txt` | DELETE (build artifact) |
| `reports/NAVIGATION-SUMMARY.txt` | DELETE (build artifact) |
| `reports/VALIDATION_SUMMARY.txt` | DELETE (build artifact) |
| `scripts/diataxis-compliance-report.txt` | DELETE (build artifact) |

### HTML Files

| File | Action |
|------|--------|
| `visionflow_wardley_map.html` | DELETE (analysis artifact) |

### Backup Files

| File | Action |
|------|--------|
| `guides/README-main.md.bak` | DELETE (backup file) |

---

## New Directory Structure (Target)

```
docs/
├── README.md                    # Main index (consolidated)
├── getting-started/             # Tutorials (learning-oriented)
│   ├── README.md               # Entry point
│   ├── overview.md             # What is VisionFlow
│   ├── installation.md         # Setup instructions
│   ├── first-graph.md          # Create first graph
│   └── neo4j-quickstart.md     # Database quick start
├── guides/                      # How-to guides (task-oriented)
│   ├── README.md               # Guide index
│   ├── configuration.md        # Configuration guide
│   ├── deployment.md           # Deployment guide
│   ├── security.md             # Security guide
│   ├── troubleshooting.md      # Troubleshooting guide
│   ├── navigation.md           # Navigation guide
│   ├── features/               # Feature guides
│   ├── ontology/               # Ontology guides
│   ├── database/               # Database guides
│   ├── visualization/          # Visualization guides
│   ├── xr/                     # XR/VR guides
│   ├── integrations/           # Integration guides
│   ├── infrastructure/         # Infrastructure guides
│   ├── operations/             # Operations guides
│   └── agents/                 # Agent guides
├── reference/                   # Reference documentation
│   ├── README.md               # Reference index
│   ├── benchmarks.md           # Performance benchmarks
│   ├── api/                    # API reference
│   ├── configuration/          # Config reference
│   ├── database/               # Schema reference
│   ├── errors/                 # Error reference
│   └── protocols/              # Protocol reference
├── architecture/                # Architecture explanations
│   ├── README.md               # Architecture overview
│   ├── overview.md             # System overview
│   ├── technology-choices.md   # Tech stack
│   ├── data-flow.md            # Data flow
│   ├── database.md             # Database architecture
│   ├── services.md             # Services architecture
│   ├── distributed-systems.md  # Distributed systems
│   ├── client/                 # Client architecture
│   ├── server/                 # Server architecture
│   ├── gpu/                    # GPU architecture
│   ├── physics/                # Physics system
│   ├── ontology/               # Ontology architecture
│   ├── visualization/          # Visualization architecture
│   ├── xr/                     # XR architecture
│   ├── agents/                 # Agent architecture
│   ├── protocols/              # Protocol architecture
│   ├── patterns/               # Design patterns
│   ├── ports/                  # Ports & adapters
│   ├── pipelines/              # Pipeline architecture
│   ├── integrations/           # Integration architecture
│   ├── migrations/             # Migration guides
│   ├── decisions/              # ADRs
│   ├── diagrams/               # Architecture diagrams
│   └── research/               # Research docs
├── development/                 # Development documentation
│   ├── README.md               # Developer index
│   ├── developer-journey.md    # Developer journey
│   ├── setup.md                # Development setup
│   ├── project-structure.md    # Project structure
│   ├── adding-features.md      # Adding features
│   ├── contributing.md         # Contributing guide
│   ├── maintenance.md          # Maintenance guide
│   ├── extending.md            # Extending guide
│   ├── workflow.md             # Development workflow
│   ├── tools.md                # Development tools
│   ├── testing/                # Testing docs
│   ├── patterns/               # Code patterns
│   └── diagrams/               # Diagram guidelines
└── assets/                      # Static assets
    └── screenshots/            # Screenshots
```

---

## Migration Priority

### Phase 1: Core Structure (Do First)
1. Create new directory structure
2. Consolidate navigation files into README.md
3. Move tutorials to getting-started/
4. Move key guides to their new locations

### Phase 2: Architecture Consolidation
1. Merge ARCHITECTURE_* files
2. Move all explanations/* to architecture/
3. Move diagrams/* to architecture/diagrams/
4. Organize by subsystem

### Phase 3: Reference Cleanup
1. Consolidate API reference files
2. Organize by type (api, database, protocols, errors)
3. Remove duplicate/outdated files

### Phase 4: Development Docs
1. Create development/ from guides/developer/
2. Merge contributing guides
3. Organize testing docs

### Phase 5: Cleanup
1. Delete all marked files
2. Move scripts to /scripts
3. Remove build artifacts
4. Update all internal links

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Files to Keep and Move | ~120 |
| Files to Consolidate | ~40 |
| Files to Delete | ~80 |
| Non-Markdown to Relocate | ~25 |
| Non-Markdown to Delete | ~16 |
| **Total Files Affected** | ~280 |

**Target Reduction**: From 280 markdown files to ~160 well-organized files (43% reduction)
