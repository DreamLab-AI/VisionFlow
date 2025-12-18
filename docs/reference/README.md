---
title: Reference Documentation
description: Comprehensive unified reference documentation for VisionFlow extracted from 306-file corpus
type: reference
status: stable
version: 2.0
last_updated: 2025-12-18
---

# VisionFlow Reference Documentation

**Version**: 2.0
**Last Updated**: December 18, 2025
**Source**: 306-file corpus consolidated into unified reference guides

This directory contains comprehensive unified reference documentation for all VisionFlow APIs, protocols, configurations, database schemas, and error codes.

---

## Quick Access

### Core Reference Documents

| Document | Size | Lines | Description |
|----------|------|-------|-------------|
| **[INDEX.md](./INDEX.md)** | 19K | 430 | Master index with alphabetical and categorical navigation |
| **[API_REFERENCE.md](./API_REFERENCE.md)** | 18K | - | Complete REST API, WebSocket, and binary protocol reference |
| **[CONFIGURATION_REFERENCE.md](./CONFIGURATION_REFERENCE.md)** | 16K | - | All configuration options: ENV, YAML, runtime settings |
| **[DATABASE_SCHEMA_REFERENCE.md](./DATABASE_SCHEMA_REFERENCE.md)** | 20K | - | SQLite + Neo4j schema, tables, relationships, queries |
| **[ERROR_REFERENCE.md](./ERROR_REFERENCE.md)** | 17K | - | Error codes, solutions, troubleshooting procedures |
| **[PROTOCOL_REFERENCE.md](./PROTOCOL_REFERENCE.md)** | 17K | - | Binary WebSocket, REST HTTP, MCP protocol specifications |

**Total**: 107K of unified reference documentation

---

## What's Inside

### 1. API Reference ([API_REFERENCE.md](./API_REFERENCE.md))

**Complete API documentation covering**:
- ✅ REST API endpoints (50+ endpoints documented)
- ✅ WebSocket protocols (binary + JSON)
- ✅ Binary protocol specification (V2/V3/V4)
- ✅ Authentication methods (JWT, Nostr, API keys)
- ✅ Error responses and status codes
- ✅ Rate limiting and versioning
- ✅ Request/response examples with code snippets

**Key Sections**:
- Authentication & Authorization
- Graph, Ontology, Physics Endpoints
- Binary Protocol V2 (36-byte format)
- Error Handling
- Rate Limiting

---

### 2. Configuration Reference ([CONFIGURATION_REFERENCE.md](./CONFIGURATION_REFERENCE.md))

**Complete configuration options**:
- ✅ 100+ environment variables documented
- ✅ YAML configuration structure
- ✅ Runtime settings API
- ✅ Feature flags
- ✅ Performance tuning guides
- ✅ Security configuration

**Key Sections**:
- Environment Variables (Core, Network, Database, GPU, AI)
- YAML Settings (System, Visualization, XR, Auth)
- Runtime Configuration
- Feature Flags
- Recommended Configurations (Small/Medium/Large)

---

### 3. Database Schema Reference ([DATABASE_SCHEMA_REFERENCE.md](./DATABASE_SCHEMA_REFERENCE.md))

**Complete database schema**:
- ✅ 8 SQLite tables fully documented
- ✅ Neo4j graph schema
- ✅ Relationships and foreign keys
- ✅ Indexes and performance optimization
- ✅ Query patterns (SQLite + Cypher)
- ✅ Migration procedures

**Key Sections**:
- SQLite Schema (unified.db)
- Neo4j Graph Schema
- Entity Relationship Diagrams
- Common Query Patterns
- Performance Characteristics

---

### 4. Error Reference ([ERROR_REFERENCE.md](./ERROR_REFERENCE.md))

**Complete error documentation**:
- ✅ 100+ error codes documented
- ✅ Solutions and workarounds
- ✅ Diagnostic procedures
- ✅ Common issues and fixes
- ✅ Health check scripts
- ✅ Performance profiling

**Key Sections**:
- Error Code System (AP/DB/GR/GP/WS/AU/ST)
- API, Database, GPU, WebSocket Errors
- Common Issues & Solutions
- Diagnostic Procedures

---

### 5. Protocol Reference ([PROTOCOL_REFERENCE.md](./PROTOCOL_REFERENCE.md))

**Complete protocol specifications**:
- ✅ Binary WebSocket Protocol (V2/V3/V4)
- ✅ REST HTTP Protocol
- ✅ MCP Protocol (JSON-RPC 2.0)
- ✅ Authentication Protocols (JWT, Nostr)
- ✅ Wire formats with byte layouts
- ✅ Performance comparisons

**Key Sections**:
- Binary Protocol V2 (36 bytes/node)
- Protocol V3 (48 bytes with analytics)
- Protocol V4 (16 bytes delta encoding)
- JSON Control Messages
- Protocol Comparison Tables

---

### 6. Master Index ([INDEX.md](./INDEX.md))

**Comprehensive navigation**:
- ✅ Alphabetical index (A-Z)
- ✅ Type index (Configuration, Data, Protocol, Error types)
- ✅ Category index (API, Config, Database, Protocol, Error, Performance)
- ✅ Endpoint index (REST, WebSocket, MCP)
- ✅ Quick lookup tables
- ✅ Cross-references

---

## Documentation Coverage

### Source Material

**Extracted from**:
- 306 markdown files
- 86 directories
- Multiple documentation layers (guides, explanations, diagrams, reference)

**Consolidated into**:
- 6 unified reference documents
- 430 total lines of comprehensive documentation
- 107K of technical reference material

### Coverage Statistics

| Category | Files Analyzed | References Extracted |
|----------|----------------|---------------------|
| API Documentation | 45+ | 50+ endpoints, 3 protocols |
| Configuration | 25+ | 100+ variables, 50+ settings |
| Database Schema | 15+ | 8 tables, 20+ indexes |
| Error Codes | 20+ | 100+ error codes |
| Protocols | 10+ | 4 protocol versions |

---

## How to Use This Documentation

### For Developers

1. **Start with [INDEX.md](./INDEX.md)** - Alphabetical and categorical navigation
2. **API Integration** - See [API_REFERENCE.md](./API_REFERENCE.md)
3. **Configuration** - See [CONFIGURATION_REFERENCE.md](./CONFIGURATION_REFERENCE.md)
4. **Troubleshooting** - See [ERROR_REFERENCE.md](./ERROR_REFERENCE.md)

### For System Architects

1. **Database Design** - See [DATABASE_SCHEMA_REFERENCE.md](./DATABASE_SCHEMA_REFERENCE.md)
2. **Protocol Selection** - See [PROTOCOL_REFERENCE.md](./PROTOCOL_REFERENCE.md)
3. **Performance Optimization** - See [performance-benchmarks.md](./performance-benchmarks.md)

### For DevOps Engineers

1. **Configuration Management** - See [CONFIGURATION_REFERENCE.md](./CONFIGURATION_REFERENCE.md)
2. **Error Resolution** - See [ERROR_REFERENCE.md](./ERROR_REFERENCE.md)
3. **Health Checks** - See [ERROR_REFERENCE.md](./ERROR_REFERENCE.md#diagnostic-procedures)

---

## Quick Reference Lookups

### Common Tasks

| Task | Reference | Section |
|------|-----------|---------|
| Connect to API | [API_REFERENCE.md](./API_REFERENCE.md#authentication--authorization) | Authentication |
| Configure GPU | [CONFIGURATION_REFERENCE.md](./CONFIGURATION_REFERENCE.md#gpu-configuration) | Environment Variables |
| Fix database error | [ERROR_REFERENCE.md](./ERROR_REFERENCE.md#database-layer-errors) | DB Errors |
| Understand binary protocol | [PROTOCOL_REFERENCE.md](./PROTOCOL_REFERENCE.md#protocol-v2-specification) | Binary WebSocket |
| Query graph database | [DATABASE_SCHEMA_REFERENCE.md](./DATABASE_SCHEMA_REFERENCE.md#query-patterns) | Query Patterns |
| Check API endpoints | [INDEX.md](./INDEX.md#endpoint-index) | Endpoint Index |

### Quick Lookups

- **Port Numbers** → [INDEX.md](./INDEX.md#port-configuration)
- **Error Code Meanings** → [ERROR_REFERENCE.md](./ERROR_REFERENCE.md#error-code-system)
- **Protocol Versions** → [PROTOCOL_REFERENCE.md](./PROTOCOL_REFERENCE.md#protocol-versions)
- **Database Tables** → [DATABASE_SCHEMA_REFERENCE.md](./DATABASE_SCHEMA_REFERENCE.md#core-tables)
- **Environment Variables** → [CONFIGURATION_REFERENCE.md](./CONFIGURATION_REFERENCE.md#environment-variables)

---

## Additional Reference Documentation

### Specialized References

| Document | Location | Description |
|----------|----------|-------------|
| **Performance Benchmarks** | [performance-benchmarks.md](./performance-benchmarks.md) | Comprehensive performance metrics |
| **Error Codes** | [error-codes.md](./error-codes.md) | Legacy error code reference |
| **WebSocket Protocol** | [websocket-protocol.md](./websocket-protocol.md) | Detailed WebSocket specification |
| **Binary WebSocket** | [protocols/binary-websocket.md](./protocols/binary-websocket.md) | Binary protocol deep dive |

### API-Specific Documentation

| Document | Location | Description |
|----------|----------|-------------|
| **REST API** | [api/rest-api-complete.md](./api/rest-api-complete.md) | Complete REST reference |
| **WebSocket API** | [api/03-websocket.md](./api/03-websocket.md) | WebSocket guide |
| **Authentication** | [api/01-authentication.md](./api/01-authentication.md) | Auth methods |
| **Semantic Features** | [api/semantic-features-api.md](./api/semantic-features-api.md) | Analytics API |

### Database Documentation

| Document | Location | Description |
|----------|----------|-------------|
| **Database Schemas** | [database/schemas.md](./database/schemas.md) | Schema overview |
| **Ontology Schema** | [database/ontology-schema-v2.md](./database/ontology-schema-v2.md) | OWL schema |
| **User Settings** | [database/user-settings-schema.md](./database/user-settings-schema.md) | User data |

---

## Documentation Standards

### Frontmatter Format

All reference documents include standardized frontmatter:
```yaml
---
title: Document Title
description: Brief description
type: reference
status: stable
version: 2.0
last_updated: 2025-12-18
---
```

### Cross-Reference Format

Cross-references use consistent markdown links:
```markdown
See [API Reference](./API_REFERENCE.md#section-name) for details.
```

### Code Example Format

Code examples include language tags and descriptions:
````markdown
```typescript
// TypeScript example
const ws = new WebSocket('ws://localhost:9090/ws');
```
````

---

## Maintenance

### Version History

| Version | Date | Changes | Documents Updated |
|---------|------|---------|-------------------|
| 2.0 | 2025-12-18 | Unified reference documentation from 306-file corpus | All 6 documents |
| 1.0 | 2025-11-04 | Initial reference documentation | Original files |

### Update Process

1. Extract reference material from source files
2. Consolidate duplicates and conflicts
3. Create unified structure
4. Add cross-references
5. Validate all links
6. Update indexes

---

## Related Documentation

### Guides

- [Configuration Guide](../guides/configuration.md) - Practical configuration examples
- [Development Workflow](../guides/development-workflow.md) - Development best practices
- [Deployment Guide](../guides/deployment.md) - Production deployment

### Explanations

- [Architecture Overview](../explanations/architecture/README.md) - System architecture
- [Data Flow](../explanations/architecture/data-flow-complete.md) - Data flow diagrams

### Tutorials

- [Installation Guide](../tutorials/01-installation.md) - Getting started
- [First Graph](../tutorials/02-first-graph.md) - Create your first graph

---

## Support

### Getting Help

- **Documentation Issues**: Check [ERROR_REFERENCE.md](./ERROR_REFERENCE.md)
- **API Questions**: See [API_REFERENCE.md](./API_REFERENCE.md)
- **Configuration Help**: See [CONFIGURATION_REFERENCE.md](./CONFIGURATION_REFERENCE.md)

### Contributing

Found an error or want to improve documentation?
1. Check if issue already documented
2. Verify against source code
3. Submit documentation update
4. Update cross-references

---

**Reference Documentation Version**: 2.0
**VisionFlow Version**: v0.1.0
**Documentation Maintainer**: VisionFlow Documentation Team
**Last Updated**: December 18, 2025
**Source Files**: 306 files consolidated
**Output**: 6 unified reference documents (107K total)
