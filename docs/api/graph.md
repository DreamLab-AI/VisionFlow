# Graph API Reference

*This file redirects to the comprehensive graph API documentation.*

See [Graph API Reference](rest/graph.md) for complete graph API documentation.

## Quick Links

- [REST Graph Endpoints](rest/graph.md) - Complete graph API reference
- [Graph Data Operations](rest/graph.md#graph-operations) - CRUD operations
- [GraphQL Interface](rest/graph.md#graphql) - Query interface
- [WebSocket Graph Updates](websocket-streams.md#graph-updates) - Real-time updates

## Core Endpoints

### Graph Data
- `GET /api/graph/data` - Retrieve complete graph structure
- `GET /api/graph/data/paginated` - Paginated graph data
- `POST /api/graph/update` - Update graph from sources
- `POST /api/graph/refresh` - Force refresh from external sources

### Node Operations
- `GET /api/graph/nodes/{id}` - Get specific node
- `POST /api/graph/nodes` - Create new node
- `PUT /api/graph/nodes/{id}` - Update node
- `DELETE /api/graph/nodes/{id}` - Delete node

### Edge Operations
- `GET /api/graph/edges` - List all edges
- `POST /api/graph/edges` - Create new edge
- `DELETE /api/graph/edges/{id}` - Remove edge

### Analytics
- `POST /api/analytics/shortest-path` - Calculate shortest paths
- `GET /api/analytics/clustering` - Get graph clustering
- `GET /api/analytics/metrics` - Graph metrics

---

[← Back to API Documentation](README.md)