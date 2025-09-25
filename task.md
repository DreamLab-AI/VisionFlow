# Backend Refactoring Task List

## Cargo Check Status (Final Verification)
**✅ NO COMPILATION ERRORS** - System compiles successfully after all updates
**⚠️ 230 warnings** - Reduced from 242 (cleaned up GPU actor unused imports)
**📊 Stability Confirmed** - All documentation updates complete, no code breakage

## Priority Tasks (Ordered by Impact)

✅ **RESOLVED**: API Endpoint Analysis Complete
- The `/api/bots/spawn-agent-hybrid` endpoint **IS IMPLEMENTED** in `/workspace/ext/src/handlers/api_handler/bots/mod.rs:24`
- Implementation found in `/workspace/ext/src/handlers/bots_handler.rs:402`
- The documentation in `interface-layer.md:57` is **INCORRECT** - it states the endpoint is missing but it exists
- **Potential Issue**: Client sends `agentType` (camelCase) but server expects `agent_type` (snake_case), though Serde should handle conversion with `#[serde(rename_all = "camelCase")]`

Overly Complex Actor Implementations: Some actors, like graph_actor.rs, are extremely large (over 29,000 tokens). While feature-rich, this can hinder maintainability. Consider breaking down its responsibilities (e.g., into a separate PhysicsOrchestratorActor, SemanticAnalysisActor, etc.) to adhere more closely to the single-responsibility principle.

graph_actor.rs: This actor is the heart of the system but has grown too large. Its responsibilities (graph state, physics, semantic analysis, constraints, stress majorization) should be delegated to smaller, more focused child actors that it supervises.
settings_handler.rs & settings_paths.rs: The path-based API for granular settings updates is a high-performance approach that avoids sending large, redundant configuration objects. This is an excellent design choice.
src/gpu/ & src/utils/*.cu
✅ **RESOLVED**: GraphServiceActor Refactoring Plan Complete
- Created comprehensive refactoring blueprint at `/workspace/ext/docs/graphactor-refactoring-plan.md`
- Analyzed current state: 3,104 lines (~38,456 tokens) with 35+ message handlers
- Proposed decomposition into 5 focused actors:
  - GraphServiceSupervisor (coordinator, 500-800 lines)
  - GraphStateActor (data management, 800-1200 lines)
  - PhysicsOrchestratorActor (simulation & GPU, 1000-1500 lines)
  - SemanticProcessorActor (AI features, 800-1200 lines)
  - ClientCoordinatorActor (WebSocket communication, 600-1000 lines)
- Included 7-phase implementation plan with testing strategies and risk mitigation
- Defined success metrics: 75% complexity reduction, <1000 lines per actor
- Estimated effort: 16 days for complete refactoring with comprehensive testing
