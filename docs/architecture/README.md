# Hexagonal/CQRS Architecture Documentation

**Complete architecture design for VisionFlow graph service migration**

---

## 📋 Documentation Index

### 1. [Hexagonal CQRS Architecture](./hexagonal-cqrs-architecture.md)
**Main architecture document** - Complete target architecture specification
- Executive summary and problem statement
- Current state analysis (monolithic GraphServiceActor)
- Target hexagonal architecture with CQRS
- Event sourcing design
- Repository ports and adapters
- GitHub sync integration fix (316 nodes bug)
- Success criteria and performance targets

**Key Sections**:
- ✅ Layer overview and component diagram
- ✅ CQRS architecture (commands + queries)
- ✅ Event sourcing with domain events
- ✅ Repository ports and adapter implementations
- ✅ API handler migration examples
- ✅ GitHub sync event integration (THE FIX!)
- ✅ Real-time updates via WebSocket events
- ✅ Directory structure

### 2. [Event Flow Diagrams](./event-flow-diagrams.md)
**Visual event flows** - Detailed ASCII diagrams showing event propagation
- GitHub sync event flow (before vs. after fix)
- Physics simulation event flow
- Node creation event flow
- WebSocket connection event flow
- Cache invalidation event flow
- Semantic analysis event flow
- Error handling event flow
- Event store replay (event sourcing)

**Use Case Scenarios**:
- ⭐ **GitHub Sync Fix**: Shows how `GitHubSyncCompletedEvent` triggers cache invalidation
- ⚡ **Real-time Physics**: Shows 60 FPS position updates via events
- 🌐 **WebSocket Sync**: Shows client initial sync and real-time updates
- 🔄 **Cache Management**: Shows when and how caches get invalidated

### 3. [Migration Strategy](./migration-strategy.md)
**4-phase migration plan** - Step-by-step implementation guide
- Phase 1: Read operations (1 week) - SAFEST
- Phase 2: Write operations + events (2 weeks)
- Phase 3: Real-time features (2 weeks) - COMPLEX
- Phase 4: Legacy removal (1 week)

**Each Phase Includes**:
- ✅ Time estimates
- ✅ Risk assessment
- ✅ Detailed steps
- ✅ Code examples
- ✅ Validation procedures
- ✅ Rollback strategies
- ✅ Success criteria

### 4. [Code Examples](./code-examples.md)
**Production-ready code** - Complete implementation examples
- Query handlers (`GetGraphDataQueryHandler`, etc.)
- Command handlers (`CreateNodeCommandHandler`, etc.)
- Event definitions (`GraphEvent` enum)
- Event bus implementation (`InMemoryEventBus`)
- Repository implementations (`SqliteGraphRepository`)
- WebSocket integration (`WebSocketEventSubscriber`)
- API handlers (before/after migration)
- Unit tests and integration tests

**Features**:
- ✅ Full type safety with Rust
- ✅ Async/await with tokio
- ✅ Error handling patterns
- ✅ Logging best practices
- ✅ Test examples with mocks

---

## 🎯 Quick Start Guide

### For Architects
1. Read [Hexagonal CQRS Architecture](./hexagonal-cqrs-architecture.md) first
2. Review [Event Flow Diagrams](./event-flow-diagrams.md) to understand event propagation
3. Study the **GitHub Sync Fix** section (critical for understanding the 316 nodes bug)

### For Developers
1. Start with [Migration Strategy](./migration-strategy.md) Phase 1
2. Reference [Code Examples](./code-examples.md) for implementation patterns
3. Follow the step-by-step migration plan
4. Run validation tests after each phase

### For Project Managers
1. Review the 4-phase timeline in [Migration Strategy](./migration-strategy.md)
2. Check success criteria for each phase
3. Understand risk mitigation strategies
4. Plan team coordination based on roles

---

## 🔥 Critical Problem Being Solved

### The GitHub Sync Cache Bug

**Problem**: After GitHub sync completes and writes 316 nodes to SQLite, the GraphServiceActor's in-memory cache remains stale with only 63 nodes. API calls return cached data instead of fresh database records.

**Root Cause**: No cache invalidation mechanism. GitHub sync writes to database but never notifies the actor.

**Solution**: Event-driven architecture where:
1. GitHub sync emits `GitHubSyncCompletedEvent`
2. Cache invalidation subscriber clears all caches
3. WebSocket clients receive reload notification
4. Next API call reads fresh data from database

**Result**: ✅ API returns 316 nodes after sync (bug fixed!)

---

## 📊 Architecture Overview

### Current State (Monolithic)
```
GraphServiceActor (48K tokens!)
├── In-memory cache (STALE!)
├── WebSocket broadcasting
├── Physics simulation
├── Semantic analysis
└── Settings management
```

**Problems**:
- ❌ Cache never invalidated
- ❌ Tight coupling
- ❌ Untestable
- ❌ Single bottleneck

### Target State (Hexagonal/CQRS)
```
HTTP/WebSocket Layer
    │
    ├──> Commands (Write) ──> EventBus ──> Subscribers
    │                              │
    │                              ├──> WebSocket
    │                              ├──> Cache Invalidation
    │                              └──> Logging/Metrics
    │
    └──> Queries (Read) ──> Repository ──> SQLite
```

**Benefits**:
- ✅ Cache invalidation via events
- ✅ Separation of concerns
- ✅ Testable components
- ✅ Horizontal scalability

---

## 🚀 Migration Timeline

| Phase | Duration | Risk | Goal |
|-------|----------|------|------|
| **Phase 1: Read Operations** | 1 week | Low | Migrate queries to CQRS |
| **Phase 2: Write Operations** | 2 weeks | Medium | Add commands + events |
| **Phase 3: Real-Time Features** | 2 weeks | High | Physics + GitHub sync via events |
| **Phase 4: Legacy Removal** | 1 week | Low | Delete old actors |
| **Total** | **6 weeks** | - | Complete migration |

---

## 📈 Success Criteria

### Functional Requirements
- ✅ All API endpoints migrated from actors to CQRS
- ✅ GitHub sync triggers `GitHubSyncCompletedEvent`
- ✅ Cache invalidation works after GitHub sync
- ✅ **API returns 316 nodes after sync (BUG FIXED!)**
- ✅ WebSocket clients receive real-time updates
- ✅ Physics simulation works via events
- ✅ Zero data loss during migration

### Non-Functional Requirements
- ✅ Query latency <50ms (p95)
- ✅ Command latency <100ms (p95)
- ✅ Event dispatch latency <10ms
- ✅ WebSocket broadcast latency <20ms
- ✅ Test coverage >80%
- ✅ Zero downtime during migration

---

## 🧪 Testing Strategy

### Unit Tests
- Query handlers (read operations)
- Command handlers (write operations)
- Event bus (publish/subscribe)
- Validators (input validation)

### Integration Tests
- End-to-end API tests
- GitHub sync integration
- WebSocket event delivery
- Database transactions

### Load Tests
- Concurrent query performance
- Event throughput (events/second)
- WebSocket broadcast scalability
- Database connection pooling

---

## 🔧 Technology Stack

### Core
- **Rust**: System programming language
- **Tokio**: Async runtime
- **Actix-web**: HTTP server
- **SQLite**: Database (via rusqlite)

### Architecture Patterns
- **Hexagonal Architecture**: Port/adapter pattern
- **CQRS**: Command/query separation
- **Event Sourcing**: Domain events
- **Repository Pattern**: Data access abstraction

### Infrastructure
- **Event Bus**: In-memory (can upgrade to Redis/RabbitMQ)
- **WebSocket**: Actix WebSocket server
- **Logging**: log + env_logger
- **Testing**: cargo test + tokio::test

---

## 📚 Additional Resources

### Hexagonal Architecture
- [Alistair Cockburn's Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [Netflix: Hexagonal Architecture](https://netflixtechblog.com/ready-for-changes-with-hexagonal-architecture-b315ec967749)

### CQRS
- [Martin Fowler: CQRS](https://martinfowler.com/bliki/CQRS.html)
- [Microsoft: CQRS Pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/cqrs)

### Event Sourcing
- [Martin Fowler: Event Sourcing](https://martinfowler.com/eaaDev/EventSourcing.html)
- [Event Sourcing Made Simple](https://kickstarter.engineering/event-sourcing-made-simple-4a2625113224)

---

## 👥 Team Roles

### Architecture Lead
- Review architecture design
- Approve major decisions
- Guide migration strategy

### Backend Developers
- Implement command/query handlers
- Write repository adapters
- Create event subscribers

### Infrastructure Engineers
- Set up event bus
- Configure monitoring
- Optimize database

### QA Engineers
- Write integration tests
- Perform load testing
- Validate migration phases

### DevOps
- Monitor production rollout
- Manage feature flags
- Handle rollbacks if needed

---

## 🎯 Next Steps

1. **Week 1**: Architecture review with team
2. **Week 2**: Begin Phase 1 (read operations migration)
3. **Week 4**: Complete Phase 2 (write operations + events)
4. **Week 6**: Deploy Phase 3 (GitHub sync fix goes live!)
5. **Week 7**: Cleanup and documentation

---

## 📝 Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-10-26 | 1.0.0 | Initial architecture design |
| - | - | Event flow diagrams created |
| - | - | Migration strategy documented |
| - | - | Code examples provided |

---

## 🏆 Goals Summary

**Primary Goal**: Fix GitHub sync cache bug (316 nodes)
**Secondary Goals**:
- Improve code maintainability
- Enable horizontal scalability
- Increase testability
- Reduce technical debt

**Expected Outcomes**:
- ✅ 316 nodes displayed after GitHub sync
- ✅ Clean, maintainable codebase
- ✅ 80%+ test coverage
- ✅ <50ms query latency
- ✅ Zero downtime migration

---

**Architecture designed by**: Hive Mind Architecture Planner Agent
**Date**: 2025-10-26
**Status**: Ready for Queen's approval 👑
**Memory Stored**: Yes (swarm coordination enabled)

---

## 🔗 Related Documents

- [Current GraphServiceActor](../../src/actors/graph_actor.rs) - Legacy implementation
- [GitHub Sync Service](../../src/services/github_sync_service.rs) - Current sync logic
- [Existing Repositories](../../src/adapters/) - Current adapter implementations
- [Application Layer](../../src/application/) - Existing CQRS patterns

---

**For questions or clarifications, consult the Hive Mind or contact the Architecture Planner agent.**
