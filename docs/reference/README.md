---
title: Reference Documentation
description: Technical reference for APIs, configuration, protocols, and database schemas
category: reference
difficulty-level: intermediate
tags:
  - reference
  - api
  - configuration
  - protocols
  - database
updated-date: 2025-01-29
---

# VisionFlow Reference Documentation

Complete technical reference documentation for VisionFlow APIs, protocols, configurations, database schemas, and error codes.

---

## Reference Sections

| Section | Description |
|---------|-------------|
| **[API Reference](./api/README.md)** | REST API, WebSocket API, authentication methods |
| **[Configuration Reference](./configuration/README.md)** | Environment variables, Docker Compose options |
| **[Protocol Reference](./protocols/README.md)** | Binary WebSocket, MCP protocol specifications |
| **[Database Reference](./database/README.md)** | SQLite and Neo4j schema documentation |
| **[CLI Reference](./cli/README.md)** | Cargo and Docker command reference |
| **[Error Codes](./error-codes.md)** | Complete error code reference with solutions |
| **[Glossary](./glossary.md)** | Technical term definitions |

---

## Quick Access

### API Reference

| Document | Description |
|----------|-------------|
| [REST API](./api/rest-api.md) | Core REST endpoints |
| [WebSocket API](./api/websocket-api.md) | Real-time WebSocket protocol |
| [Authentication](./api/authentication.md) | JWT, API keys, Nostr NIP-98 |
| [Complete REST Reference](./api/rest-api-complete.md) | All 110+ endpoints |

### Configuration

| Document | Description |
|----------|-------------|
| [Environment Variables](./configuration/environment-variables.md) | All env var options |
| [Docker Compose](./configuration/docker-compose-options.md) | Container configuration |

### Protocols

| Document | Description |
|----------|-------------|
| [Binary WebSocket](./protocols/binary-websocket.md) | V2/V3/V4 wire formats |
| [MCP Protocol](./protocols/mcp-protocol.md) | Agent orchestration protocol |

### Database

| Document | Description |
|----------|-------------|
| [SQLite Schema](./database/schemas.md) | Unified.db table definitions |
| [Neo4j Schema](./database/neo4j-schema.md) | Graph database schema |
| [Ontology Schema](./database/ontology-schema-v2.md) | OWL ontology storage |

### CLI

| Document | Description |
|----------|-------------|
| [Cargo Commands](./cli/cargo-commands.md) | Rust build, test, run |
| [Docker Commands](./cli/docker-commands.md) | Docker Compose commands |

---

## Additional Documentation

### Specialized References

| Document | Description |
|----------|-------------|
| [Performance Benchmarks](./performance-benchmarks.md) | Performance metrics and targets |
| [Implementation Status](./implementation-status.md) | Feature implementation status |
| [Code Quality Status](./code-quality-status.md) | Code quality metrics |

### API Deep Dives

| Document | Description |
|----------|-------------|
| [Pathfinding Examples](./api/pathfinding-examples.md) | Graph pathfinding API examples |
| [Semantic Features API](./api/semantic-features-api.md) | Analytics and ML features |
| [Solid API](./api/solid-api.md) | Solid pod integration |

---

## Documentation Standards

### Frontmatter Format

All reference documents use standardised frontmatter:

```yaml
---
title: Document Title
description: Brief description
category: reference
difficulty-level: intermediate
updated-date: 2025-01-29
---
```

### Difficulty Levels

| Level | Audience |
|-------|----------|
| `beginner` | New users |
| `intermediate` | Experienced users |
| `advanced` | System architects, contributors |

---

## Related Documentation

### Guides

- [Configuration Guide](../guides/configuration.md) - Practical examples
- [Deployment Guide](../guides/deployment.md) - Production deployment
- [Troubleshooting Guide](../guides/troubleshooting.md) - Common issues

### Concepts

- [Architecture Overview](../concepts/README.md) - System architecture
- [Data Flow](../explanations/architecture/data-flow-complete.md) - Data flow diagrams

### Getting Started

- [Installation Guide](../getting-started/installation.md) - Setup instructions
- [First Graph](../getting-started/first-graph.md) - Quick start tutorial

---

**Last Updated**: January 29, 2025
