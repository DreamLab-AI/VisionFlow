# Settings System Refactoring Project

## Overview
This project refactors the VisionFlow settings management system to address brittleness, verbosity, complexity, and network inefficiency while preserving all existing features and optimizing snake_case (Rust) to camelCase (TypeScript) conversion.

## Project Goals
1. **Establish Single Source of Truth**: Rust backend becomes definitive source with auto-generated TypeScript types
2. **Automate Case Conversion**: Use serde attributes to eliminate manual DTOs
3. **Optimize Network Communication**: Implement granular API endpoints for efficient data transfer
4. **Refactor Client-Side Architecture**: Improve performance and maintainability

## Current Architecture Analysis

### Backend Structure
- **Main Config**: `/workspace/ext/src/config/mod.rs` (991 lines)
- **Settings Handler**: `/workspace/ext/src/handlers/settings_handler.rs`
- **Settings Actor**: `/workspace/ext/src/actors/settings_actor.rs`

### Frontend Structure
- **Settings Store**: `/workspace/ext/client/src/store/settingsStore.ts`
- **Settings API**: `/workspace/ext/client/src/api/settingsApi.ts`
- **Settings Config**: `/workspace/ext/client/src/features/settings/config/settings.ts`
- **UI Definition**: `/workspace/ext/client/src/features/settings/config/settingsUIDefinition.ts`

### Key Settings Structures Identified

#### Main Settings Container
```rust
pub struct AppFullSettings {
    pub visualisation: VisualisationSettings,
    pub system: SystemSettings,
    pub xr: XRSettings,
    pub auth: AuthSettings,
    // Optional API service settings
    pub ragflow: Option<RagFlowSettings>,
    pub perplexity: Option<PerplexitySettings>,
    pub openai: Option<OpenAISettings>,
    pub kokoro: Option<KokoroSettings>,
    pub whisper: Option<WhisperSettings>,
}
```

#### Complex Nested Structures
- **VisualisationSettings**: Contains rendering, animations, glow, hologram, and graph settings
- **PhysicsSettings**: Extensive physics simulation parameters (96 fields)
- **SystemSettings**: Network, WebSocket, security, and debug configurations
- **XRSettings**: Virtual/Augmented reality settings
- **GraphsSettings**: Multi-graph container (logseq, visionflow)

## Project Documentation Structure

- `/docs/settings-refactoring/`
  - `README.md` (this file)
  - `architecture/` - System architecture documentation
  - `api/` - API documentation
  - `migration/` - Migration guides
  - `implementation/` - Implementation progress
  - `performance/` - Performance analysis
  - `changelog/` - Change tracking

## Implementation Phases

### Phase 1: Single Source of Truth
- [ ] Add type generation dependencies to Cargo.toml
- [ ] Annotate Rust structs with #[derive(specta::Type)]
- [ ] Create build script for TypeScript generation
- [ ] Update frontend to use generated types

### Phase 2: Automated Case Conversion
- [ ] Apply #[serde(rename_all = "camelCase")] attributes
- [ ] Remove manual DTO layer
- [ ] Simplify API handlers
- [ ] Update serialization logic

### Phase 3: Network Protocol Optimization
- [ ] Implement granular GET /api/settings/get endpoint
- [ ] Implement granular POST /api/settings/set endpoint
- [ ] Refactor frontend API service
- [ ] Update client-side state management

### Phase 4: Client-Side Refactoring
- [ ] Implement lazy-loading for settings panels
- [ ] Add selective state subscription
- [ ] Consolidate UI definitions
- [ ] Optimize component rendering

## Progress Tracking

**Overall Progress**: 0% Complete

**Current Phase**: Documentation and Analysis

**Next Steps**:
1. Complete system analysis
2. Create detailed implementation plans
3. Begin Phase 1 implementation

## Risk Assessment

### High Risk Items
- Breaking changes to existing API contracts
- Frontend/backend synchronization during transition
- Maintaining backward compatibility

### Mitigation Strategies
- Implement feature flags for gradual rollout
- Maintain parallel API versions during transition
- Comprehensive testing at each phase

## Performance Targets

### Current Issues
- Full settings object transferred on every request
- Manual case conversion overhead
- Duplicated type definitions between frontend/backend

### Expected Improvements
- 60-80% reduction in network payload size
- 40-50% reduction in backend code complexity
- Elimination of type synchronization errors
- Improved initial load times

## Testing Strategy

### Testing Phases
1. **Unit Tests**: Individual component validation
2. **Integration Tests**: API contract verification
3. **Performance Tests**: Network efficiency validation
4. **End-to-End Tests**: Full workflow verification

### Test Coverage Goals
- Backend: >90% code coverage
- Frontend: >85% code coverage
- API contracts: 100% coverage

---

**Last Updated**: 2025-09-01  
**Documentation Maintained By**: Documentation and Progress Tracking Agent