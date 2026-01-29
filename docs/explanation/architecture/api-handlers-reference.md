---
title: API Handlers Reference - Complete Endpoint Documentation
description: **Version:** 1.0.0 **Last Updated:** 2025-11-04 **Status:** Production
category: explanation
tags:
  - api
  - architecture
  - api
  - api
  - database
updated-date: 2025-12-18
difficulty-level: advanced
---


# API Handlers Reference - Complete Endpoint Documentation

**Version:** 1.0.0
**Last Updated:** 2025-11-04
**Status:** Production

## Table of Contents

1. [Overview](#overview)
2. [Core API Handlers](#core-api-handlers)
3. [Graph Handlers](#graph-handlers)
4. [Settings Handlers](#settings-handlers)
5. [File Management Handlers](#file-management-handlers)
6. [Ontology Handlers](#ontology-handlers)
7. [Bot Orchestration Handlers](#bot-orchestration-handlers)
8. [Analytics Handlers](#analytics-handlers)
9. [Visualization Handlers](#visualization-handlers)
10. [XR/Quest3 Handlers](#xrquest3-handlers)
11. [External Integration Handlers](#external-integration-handlers)
12. [WebSocket Handlers](#websocket-handlers)
13. [Admin Handlers](#admin-handlers)
14. [Utility Handlers](#utility-handlers)
15. 
16. [Error Handling](#error-handling)
17. [Rate Limiting](#rate-limiting)

---

## Overview

### Handler Categories

The platform exposes **52 handler files** organized into **14 functional categories**:

```
Total HTTP Handlers: 52 files
Total WebSocket Handlers: 8 endpoints
Total REST Endpoints: 100+ individual routes
Base Path: /api
```

### Handler File Structure

```
src/handlers/
├── Core (2 files)
│   └── api_handler/mod.rs
├── Graph (3 files)
│   ├── graph_state_handler.rs
│   ├── graph_export_handler.rs
│   └── api_handler/graph/mod.rs
├── Settings (5 files)
│   ├── settings_handler.rs [DEPRECATED]
│   ├── settings::api module (new)
│   ├── settings_validation_fix.rs
│   ├── websocket_settings_handler.rs
│   └── api_handler/settings/mod.rs
├── Files (1 file)
│   └── api_handler/files/mod.rs
├── Ontology (2 files)
│   ├── ontology_handler.rs
│   └── api_handler/ontology/mod.rs
├── Bots (3 files)
│   ├── bots_handler.rs
│   ├── bots_visualization_handler.rs
│   └── api_handler/bots/mod.rs
├── Analytics (4 files)
│   ├── clustering_handler.rs
│   └── api_handler/analytics/
│       ├── anomaly.rs
│       ├── clustering.rs
│       ├── community.rs
│       └── websocket_integration.rs
├── Visualization (1 file)
│   └── api_handler/visualisation/mod.rs
├── XR (1 file)
│   └── api_handler/quest3/mod.rs
├── External (6 files)
│   ├── ragflow_handler.rs
│   ├── perplexity_handler.rs
│   ├── nostr_handler.rs
│   ├── mcp_relay_handler.rs
│   ├── multi_mcp_websocket_handler.rs
│   └── speech_socket_handler.rs
├── WebSocket (6 files)
│   ├── socket_flow_handler.rs
│   ├── realtime_websocket_handler.rs
│   ├── client_messages_handler.rs
│   ├── websocket_utils.rs
│   └── [handlers above]
├── Admin (3 files)
│   ├── admin_sync_handler.rs
│   ├── consolidated_health_handler.rs
│   └── pipeline_admin_handler.rs [DEPRECATED]
└── Utility (10 files)
    ├── workspace_handler.rs
    ├── pages_handler.rs
    ├── constraints_handler.rs
    ├── physics_handler.rs
    ├── semantic_handler.rs
    ├── inference_handler.rs
    ├── validation_handler.rs
    ├── client_log_handler.rs
    └── utils.rs
```

---

## Core API Handlers

### Health Check

**Endpoint:** `GET /api/health`
**Handler:** `api_handler::health_check`
**Authentication:** None
**Rate Limit:** None

**Purpose:** Health check endpoint for monitoring and load balancers.

**Response Schema:**

```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": "2025-11-04T12:00:00Z"
}
```

**Status Codes:**
- `200 OK` - Service healthy

**Example:**

```bash
curl http://localhost:4000/api/health

# Response:
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": "2025-11-04T12:00:00.000Z"
}
```

---

### Application Configuration

**Endpoint:** `GET /api/config`
**Handler:** `api_handler::get_app_config`
**Authentication:** None
**Rate Limit:** 100/minute

**Purpose:** Get application configuration and feature flags.

**Response Schema:**

```json
{
  "version": "string",
  "features": {
    "ragflow": "boolean",
    "perplexity": "boolean",
    "openai": "boolean",
    "kokoro": "boolean",
    "whisper": "boolean"
  },
  "websocket": {
    "minUpdateRate": "number",
    "maxUpdateRate": "number",
    "motionThreshold": "number",
    "motionDamping": "number"
  },
  "rendering": {
    "ambientLightIntensity": "number",
    "enableAmbientOcclusion": "boolean",
    "backgroundColor": "string"
  },
  "xr": {
    "enabled": "boolean",
    "roomScale": "number",
    "spaceType": "string"
  }
}
```

**Status Codes:**
- `200 OK` - Configuration retrieved
- `500 Internal Server Error` - Failed to load configuration

**Example:**

```bash
curl http://localhost:4000/api/config

# Response:
{
  "version": "1.0.0",
  "features": {
    "ragflow": true,
    "perplexity": true,
    "openai": false,
    "kokoro": true,
    "whisper": true
  },
  "websocket": {
    "minUpdateRate": 10,
    "maxUpdateRate": 60,
    "motionThreshold": 0.01,
    "motionDamping": 0.95
  },
  "rendering": {
    "ambientLightIntensity": 0.5,
    "enableAmbientOcclusion": true,
    "backgroundColor": "#1a1a2e"
  },
  "xr": {
    "enabled": true,
    "roomScale": 3.0,
    "spaceType": "local-floor"
  }
}
```

---

## Graph Handlers

### Get Graph Data

**Endpoint:** `GET /api/graph/data`
**Handler:** `api_handler::graph::get_graph_data`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Retrieve complete knowledge graph with node positions and physics state.

**Response Schema:**

```json
{
  "nodes": [
    {
      "id": "number",
      "metadataId": "string",
      "label": "string",
      "position": {"x": "number", "y": "number", "z": "number"},
      "velocity": {"x": "number", "y": "number", "z": "number"},
      "metadata": {"key": "value"},
      "type": "string | null",
      "size": "number | null",
      "color": "string | null",
      "weight": "number | null",
      "group": "string | null"
    }
  ],
  "edges": [
    {
      "source": "number",
      "target": "number",
      "label": "string",
      "weight": "number | null",
      "type": "string | null"
    }
  ],
  "metadata": {
    "metadataId": {
      "id": "string",
      "file_path": "string",
      "title": "string",
      "tags": ["string"],
      "last_modified": "number"
    }
  },
  "settlementState": {
    "isSettled": "boolean",
    "stableFrameCount": "number",
    "kineticEnergy": "number"
  }
}
```

**Status Codes:**
- `200 OK` - Graph data retrieved
- `500 Internal Server Error` - Failed to retrieve graph data

**Implementation Details:**
- Uses CQRS `GetGraphData`, `GetNodeMap`, `GetPhysicsState` queries
- Executes queries in parallel via `tokio::join!`
- Enriches nodes with physics positions from actor state
- Blocking operations run in thread pool via `execute_in_thread()`

**Example:**

```bash
curl http://localhost:4000/api/graph/data

# Response (truncated):
{
  "nodes": [
    {
      "id": 1,
      "metadataId": "logseq-page-123",
      "label": "Knowledge Management",
      "position": {"x": 100.5, "y": 200.3, "z": 0.0},
      "velocity": {"x": 0.01, "y": -0.02, "z": 0.0},
      "metadata": {
        "file": "pages/knowledge-management.md",
        "tags": "productivity,learning"
      },
      "type": "page",
      "size": 10.0,
      "color": "#3498db",
      "weight": 1.5,
      "group": "core"
    }
  ],
  "edges": [
    {
      "source": 1,
      "target": 2,
      "label": "links to",
      "weight": 1.0,
      "type": "reference"
    }
  ],
  "settlementState": {
    "isSettled": true,
    "stableFrameCount": 120,
    "kineticEnergy": 0.0005
  }
}
```

---

### Get Paginated Graph Data

**Endpoint:** `GET /api/graph/data/paginated`
**Handler:** `api_handler::graph::get_paginated_graph_data`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Retrieve graph data with pagination for large graphs.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | integer | 1 | Page number (1-indexed) |
| page_size | integer | 100 | Items per page (1-1000) |
| query | string | null | Search query (not implemented) |
| sort | string | null | Sort field (not implemented) |
| filter | string | null | Filter criteria (not implemented) |

**Response Schema:**

```json
{
  "nodes": ["array of nodes (same as /api/graph/data)"],
  "edges": ["array of edges (filtered to page nodes)"],
  "metadata": {"object"},
  "totalPages": "number",
  "currentPage": "number",
  "totalItems": "number",
  "pageSize": "number"
}
```

**Status Codes:**
- `200 OK` - Paginated data retrieved
- `400 Bad Request` - Invalid page or page_size
- `500 Internal Server Error` - Failed to retrieve data

**Example:**

```bash
curl "http://localhost:4000/api/graph/data/paginated?page=2&page_size=50"

# Response:
{
  "nodes": [/* 50 nodes */],
  "edges": [/* edges connecting page nodes */],
  "metadata": {},
  "totalPages": 10,
  "currentPage": 2,
  "totalItems": 500,
  "pageSize": 50
}
```

---

### Refresh Graph

**Endpoint:** `POST /api/graph/refresh`
**Handler:** `api_handler::graph::refresh_graph`
**Authentication:** None
**Rate Limit:** 1/minute

**Purpose:** Trigger graph refresh (returns current state without re-fetching).

**Response Schema:**

```json
{
  "success": true,
  "message": "Graph data retrieved successfully",
  "data": {
    "nodes": ["array"],
    "edges": ["array"],
    "metadata": {"object"}
  }
}
```

**Status Codes:**
- `200 OK` - Graph refreshed
- `500 Internal Server Error` - Failed to refresh

**Note:** Despite the name, this endpoint does NOT re-fetch from GitHub. Use `/api/admin/sync` for that.

**Example:**

```bash
curl -X POST http://localhost:4000/api/graph/refresh

# Response:
{
  "success": true,
  "message": "Graph data retrieved successfully",
  "data": {
    "nodes": [/* current nodes */],
    "edges": [/* current edges */],
    "metadata": {}
  }
}
```

---

### Update Graph from GitHub

**Endpoint:** `POST /api/graph/update`
**Handler:** `api_handler::graph::update_graph`
**Authentication:** None
**Rate Limit:** 1/5 minutes

**Purpose:** Fetch new files from GitHub and update the knowledge graph.

**Response Schema:**

```json
{
  "success": true,
  "message": "Graph updated with {N} new files"
}
```

**Status Codes:**
- `200 OK` - Graph updated successfully
- `500 Internal Server Error` - Failed to fetch or process files

**Implementation Details:**
1. Load metadata from `FileService`
2. Fetch new files from GitHub via `ContentAPI`
3. Process markdown files (parse frontmatter, extract nodes/edges)
4. Send `AddNodesFromMetadata` message to `GraphServiceActor`
5. Update `MetadataActor` with new metadata

**Example:**

```bash
curl -X POST http://localhost:4000/api/graph/update

# Response:
{
  "success": true,
  "message": "Graph updated with 15 new files"
}
```

---

### Auto-Balance Notifications

**Endpoint:** `GET /api/graph/auto-balance-notifications`
**Handler:** `api_handler::graph::get_auto_balance_notifications`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Get notifications about auto-balance events in physics simulation.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| since | integer | null | Unix timestamp (seconds) to filter notifications |

**Response Schema:**

```json
{
  "success": true,
  "notifications": [
    {
      "timestamp": "number",
      "type": "string",
      "message": "string",
      "data": {"object"}
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Notifications retrieved
- `500 Internal Server Error` - Failed to retrieve notifications

**Example:**

```bash
curl "http://localhost:4000/api/graph/auto-balance-notifications?since=1699027200"

# Response:
{
  "success": true,
  "notifications": [
    {
      "timestamp": 1699030800,
      "type": "balance_applied",
      "message": "Auto-balance applied to cluster",
      "data": {
        "cluster_id": "main",
        "nodes_affected": 150
      }
    }
  ]
}
```

---

### Export Graph

**Endpoint:** `GET /api/graph/export`
**Handler:** `graph_export_handler::export_graph`
**Authentication:** None
**Rate Limit:** 10/minute

**Purpose:** Export graph in various formats (JSON, GraphML, GEXF, CSV).

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| format | string | json | Export format: json, graphml, gexf, csv |
| include_positions | boolean | true | Include 3D positions |
| include_metadata | boolean | false | Include metadata fields |

**Response Schema:**

Depends on format:
- `json` - Custom JSON format
- `graphml` - GraphML XML
- `gexf` - GEXF XML
- `csv` - Two CSV files (nodes.csv, edges.csv) in ZIP

**Status Codes:**
- `200 OK` - Export successful (Content-Type varies)
- `400 Bad Request` - Invalid format
- `500 Internal Server Error` - Export failed

**Example:**

```bash
curl "http://localhost:4000/api/graph/export?format=graphml" > graph.graphml

# GraphML response:
<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <graph id="G" edgedefault="directed">
    <node id="1">
      <data key="label">Knowledge Management</data>
    </node>
    <edge source="1" target="2">
      <data key="label">links to</data>
    </edge>
  </graph>
</graphml>
```

---

### Graph State Sync

**Endpoint:** `GET /api/graph/state`
**Handler:** `graph_state_handler::get_graph_state`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Get current graph state with version information for sync.

**Response Schema:**

```json
{
  "version": "number",
  "node_count": "number",
  "edge_count": "number",
  "last_updated": "number",
  "checksum": "string"
}
```

**Status Codes:**
- `200 OK` - State retrieved
- `500 Internal Server Error` - Failed to retrieve state

**Example:**

```bash
curl http://localhost:4000/api/graph/state

# Response:
{
  "version": 42,
  "node_count": 500,
  "edge_count": 1200,
  "last_updated": 1699030800,
  "checksum": "a3c5f9e2..."
}
```

---

## Settings Handlers

### Get All Settings

**Endpoint:** `GET /api/settings`
**Handler:** `settings::api::get_all_settings`
**Authentication:** Optional (Nostr)
**Rate Limit:** 10/second

**Purpose:** Retrieve complete application settings (YAML structure as JSON).

**Response Schema:**

```json
{
  "system": {
    "websocket": {
      "minUpdateRate": "number",
      "maxUpdateRate": "number",
      "motionThreshold": "number",
      "motionDamping": "number",
      "heartbeatInterval": "number",
      "heartbeatTimeout": "number"
    },
    "network": {
      "port": "number",
      "workers": "number"
    }
  },
  "visualisation": {
    "rendering": {
      "ambientLightIntensity": "number",
      "enableAmbientOcclusion": "boolean",
      "backgroundColor": "string"
    },
    "graphs": {
      "logseq": {
        "physics": {
          "enabled": "boolean",
          "gravity": "number",
          "linkStrength": "number",
          "nodeRepulsion": "number",
          "damping": "number",
          "timeStep": "number",
          "settlementThreshold": "number"
        }
      }
    }
  },
  "xr": {
    "enabled": "boolean",
    "roomScale": "number",
    "spaceType": "string"
  },
  "ragflow": {
    "apiUrl": "string",
    "apiKey": "string (masked)"
  },
  "perplexity": {
    "apiKey": "string (masked)"
  },
  "kokoro": {
    "apiUrl": "string"
  }
}
```

**Status Codes:**
- `200 OK` - Settings retrieved
- `500 Internal Server Error` - Failed to load settings

**Example:**

```bash
curl http://localhost:4000/api/settings

# Response (truncated):
{
  "system": {
    "websocket": {
      "minUpdateRate": 10,
      "maxUpdateRate": 60,
      "motionThreshold": 0.01,
      "motionDamping": 0.95
    }
  },
  "visualisation": {
    "rendering": {
      "ambientLightIntensity": 0.5,
      "enableAmbientOcclusion": true,
      "backgroundColor": "#1a1a2e"
    },
    "graphs": {
      "logseq": {
        "physics": {
          "enabled": true,
          "gravity": 0.1,
          "linkStrength": 1.0,
          "nodeRepulsion": 100.0,
          "damping": 0.95
        }
      }
    }
  }
}
```

---

### Update Setting by Path

**Endpoint:** `PUT /api/settings/{path}`
**Handler:** `settings::api::update_setting`
**Authentication:** Optional (Nostr)
**Rate Limit:** 10/second

**Purpose:** Update a single setting value using JSON path notation.

**Path Parameter:**
- `path` - Dot-separated path (e.g., `visualisation.rendering.ambientLightIntensity`)

**Request Body:**

```json
{
  "value": "any JSON value"
}
```

**Response Schema:**

```json
{
  "success": true,
  "message": "Setting updated successfully"
}
```

**Status Codes:**
- `200 OK` - Setting updated
- `400 Bad Request` - Invalid path or value
- `500 Internal Server Error` - Failed to update setting

**Example:**

```bash
curl -X PUT http://localhost:4000/api/settings/visualisation.rendering.ambientLightIntensity \
  -H "Content-Type: application/json" \
  -d '{"value": 0.75}'

# Response:
{
  "success": true,
  "message": "Setting updated successfully"
}
```

---

### Update Multiple Settings (Batch)

**Endpoint:** `POST /api/settings/batch`
**Handler:** `settings::api::update_settings_batch`
**Authentication:** Optional (Nostr)
**Rate Limit:** 5/minute

**Purpose:** Update multiple settings in a single transaction.

**Request Body:**

```json
{
  "updates": [
    {
      "path": "string",
      "value": "any JSON value"
    }
  ]
}
```

**Response Schema:**

```json
{
  "success": true,
  "updated": "number",
  "failed": [
    {
      "path": "string",
      "error": "string"
    }
  ]
}
```

**Status Codes:**
- `200 OK` - All settings updated
- `207 Multi-Status` - Partial success (some updates failed)
- `400 Bad Request` - Invalid request format
- `500 Internal Server Error` - Failed to update settings

**Example:**

```bash
curl -X POST http://localhost:4000/api/settings/batch \
  -H "Content-Type: application/json" \
  -d '{
    "updates": [
      {"path": "visualisation.rendering.ambientLightIntensity", "value": 0.75},
      {"path": "system.websocket.maxUpdateRate", "value": 120}
    ]
  }'

# Response:
{
  "success": true,
  "updated": 2,
  "failed": []
}
```

---

### Save All Settings

**Endpoint:** `POST /api/settings`
**Handler:** `settings::api::save_all_settings`
**Authentication:** Optional (Nostr)
**Rate Limit:** 1/minute

**Purpose:** Replace entire settings object (dangerous - use batch update instead).

**Request Body:**

Complete `AppFullSettings` object (same schema as GET response).

**Response Schema:**

```json
{
  "success": true,
  "message": "Settings saved successfully"
}
```

**Status Codes:**
- `200 OK` - Settings saved
- `400 Bad Request` - Invalid settings format
- `500 Internal Server Error` - Failed to save settings

**Example:**

```bash
curl -X POST http://localhost:4000/api/settings \
  -H "Content-Type: application/json" \
  -d @settings.json

# Response:
{
  "success": true,
  "message": "Settings saved successfully"
}
```

---

### Get Physics Settings

**Endpoint:** `GET /api/settings/physics`
**Handler:** `settings::api::get_physics_settings`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Get physics simulation settings for graph layout.

**Response Schema:**

```json
{
  "enabled": "boolean",
  "gravity": "number",
  "linkStrength": "number",
  "nodeRepulsion": "number",
  "damping": "number",
  "timeStep": "number",
  "settlementThreshold": "number",
  "maxVelocity": "number",
  "centerForce": "number"
}
```

**Status Codes:**
- `200 OK` - Physics settings retrieved
- `500 Internal Server Error` - Failed to load settings

**Example:**

```bash
curl http://localhost:4000/api/settings/physics

# Response:
{
  "enabled": true,
  "gravity": 0.1,
  "linkStrength": 1.0,
  "nodeRepulsion": 100.0,
  "damping": 0.95,
  "timeStep": 0.016,
  "settlementThreshold": 0.001,
  "maxVelocity": 10.0,
  "centerForce": 0.05
}
```

---

### Update Physics Settings

**Endpoint:** `PUT /api/settings/physics`
**Handler:** `settings::api::update_physics_settings`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Update physics settings and apply to running simulation.

**Request Body:**

Partial or complete physics settings object.

**Response Schema:**

```json
{
  "success": true,
  "message": "Physics settings updated and applied"
}
```

**Status Codes:**
- `200 OK` - Settings updated and applied
- `400 Bad Request` - Invalid physics parameters
- `500 Internal Server Error` - Failed to update settings

**Implementation Details:**
- Validates physics parameters (e.g., damping must be 0-1)
- Updates Neo4j settings repository
- Sends `UpdateSimulationParams` message to `GraphServiceActor`
- Resets simulation if parameters changed significantly

**Example:**

```bash
curl -X PUT http://localhost:4000/api/settings/physics \
  -H "Content-Type: application/json" \
  -d '{
    "damping": 0.98,
    "linkStrength": 1.5
  }'

# Response:
{
  "success": true,
  "message": "Physics settings updated and applied"
}
```

---

### List Physics Profiles

**Endpoint:** `GET /api/settings/physics/profiles`
**Handler:** `settings::api::list_physics_profiles`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** List saved physics configuration presets.

**Response Schema:**

```json
{
  "profiles": [
    {
      "id": "string",
      "name": "string",
      "description": "string",
      "settings": {"object"}
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Profiles retrieved
- `500 Internal Server Error` - Failed to load profiles

**Example:**

```bash
curl http://localhost:4000/api/settings/physics/profiles

# Response:
{
  "profiles": [
    {
      "id": "default",
      "name": "Default",
      "description": "Balanced physics for general use",
      "settings": {
        "gravity": 0.1,
        "linkStrength": 1.0,
        "damping": 0.95
      }
    },
    {
      "id": "tight",
      "name": "Tight Clustering",
      "description": "Strong attraction for compact layouts",
      "settings": {
        "gravity": 0.2,
        "linkStrength": 2.0,
        "damping": 0.98
      }
    }
  ]
}
```

---

## File Management Handlers

### List Files

**Endpoint:** `GET /api/files`
**Handler:** `api_handler::files::list_files`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** List markdown files from GitHub repository.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| path | string | "" | Filter by directory path |
| recursive | boolean | false | Include subdirectories |

**Response Schema:**

```json
{
  "files": [
    {
      "path": "string",
      "name": "string",
      "size": "number",
      "type": "file | directory",
      "last_modified": "string (ISO 8601)"
    }
  ],
  "total": "number"
}
```

**Status Codes:**
- `200 OK` - Files listed
- `500 Internal Server Error` - Failed to fetch file list

**Example:**

```bash
curl "http://localhost:4000/api/files?path=pages&recursive=true"

# Response:
{
  "files": [
    {
      "path": "pages/knowledge-management.md",
      "name": "knowledge-management.md",
      "size": 4520,
      "type": "file",
      "last_modified": "2025-11-03T10:30:00Z"
    },
    {
      "path": "pages/productivity.md",
      "name": "productivity.md",
      "size": 3210,
      "type": "file",
      "last_modified": "2025-11-02T15:45:00Z"
    }
  ],
  "total": 2
}
```

---

### Get File Content

**Endpoint:** `GET /api/files/{path}`
**Handler:** `api_handler::files::get_file_content`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Get raw markdown content of a file from GitHub.

**Path Parameter:**
- `path` - URL-encoded file path (e.g., `pages%2Fknowledge-management.md`)

**Response Schema:**

```json
{
  "path": "string",
  "content": "string (markdown)",
  "metadata": {
    "size": "number",
    "last_modified": "string",
    "sha": "string"
  }
}
```

**Status Codes:**
- `200 OK` - File content retrieved
- `404 Not Found` - File not found
- `500 Internal Server Error` - Failed to fetch file

**Example:**

```bash
curl http://localhost:4000/api/files/pages%2Fknowledge-management.md

# Response:
{
  "path": "pages/knowledge-management.md",
  "content": "---\ntitle: Knowledge Management\ntags: productivity, learning\n---\n\n# Knowledge Management\n\nContent here...",
  "metadata": {
    "size": 4520,
    "last_modified": "2025-11-03T10:30:00Z",
    "sha": "a3c5f9e2..."
  }
}
```

---

### Process Files

**Endpoint:** `POST /api/files/process`
**Handler:** `api_handler::files::process_files`
**Authentication:** None
**Rate Limit:** 1/minute

**Purpose:** Manually trigger file processing (parse and add to graph).

**Request Body:**

```json
{
  "paths": ["string"],
  "force": "boolean (default: false)"
}
```

**Response Schema:**

```json
{
  "success": true,
  "processed": "number",
  "failed": [
    {
      "path": "string",
      "error": "string"
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Files processed
- `207 Multi-Status` - Partial success
- `400 Bad Request` - Invalid paths
- `500 Internal Server Error` - Processing failed

**Example:**

```bash
curl -X POST http://localhost:4000/api/files/process \
  -H "Content-Type: application/json" \
  -d '{
    "paths": ["pages/new-page.md"],
    "force": true
  }'

# Response:
{
  "success": true,
  "processed": 1,
  "failed": []
}
```

---

## Ontology Handlers

### List OWL Classes

**Endpoint:** `GET /api/ontology/classes`
**Handler:** `api_handler::ontology::list_owl_classes`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** List all OWL classes from ontology database.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| parent_iri | string | null | Filter by parent class IRI |
| search | string | null | Search by label or IRI |
| limit | integer | 100 | Max results (1-1000) |
| offset | integer | 0 | Pagination offset |

**Response Schema:**

```json
{
  "classes": [
    {
      "iri": "string",
      "label": "string",
      "description": "string | null",
      "parent_iri": "string | null",
      "created_at": "number (Unix timestamp)"
    }
  ],
  "total": "number",
  "limit": "number",
  "offset": "number"
}
```

**Status Codes:**
- `200 OK` - Classes retrieved
- `500 Internal Server Error` - Failed to load classes

**Example:**

```bash
curl "http://localhost:4000/api/ontology/classes?limit=5"

# Response:
{
  "classes": [
    {
      "iri": "http://example.org/ontology#Concept",
      "label": "Concept",
      "description": "An abstract or general idea",
      "parent_iri": "http://www.w3.org/2002/07/owl#Thing",
      "created_at": 1699027200
    },
    {
      "iri": "http://example.org/ontology#Person",
      "label": "Person",
      "description": "A human being",
      "parent_iri": "http://example.org/ontology#Agent",
      "created_at": 1699027300
    }
  ],
  "total": 50,
  "limit": 5,
  "offset": 0
}
```

---

### Get OWL Class

**Endpoint:** `GET /api/ontology/classes/{iri}`
**Handler:** `api_handler::ontology::get_owl_class`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Get details of a specific OWL class.

**Path Parameter:**
- `iri` - URL-encoded class IRI

**Response Schema:**

```json
{
  "iri": "string",
  "label": "string",
  "description": "string | null",
  "parent_iri": "string | null",
  "subclasses": ["string (IRIs)"],
  "properties": [
    {
      "iri": "string",
      "label": "string",
      "type": "object | data | annotation"
    }
  ],
  "axioms": ["string"],
  "created_at": "number"
}
```

**Status Codes:**
- `200 OK` - Class retrieved
- `404 Not Found` - Class not found
- `500 Internal Server Error` - Failed to load class

**Example:**

```bash
curl http://localhost:4000/api/ontology/classes/http%3A%2F%2Fexample.org%2Fontology%23Person

# Response:
{
  "iri": "http://example.org/ontology#Person",
  "label": "Person",
  "description": "A human being",
  "parent_iri": "http://example.org/ontology#Agent",
  "subclasses": [
    "http://example.org/ontology#Student",
    "http://example.org/ontology#Teacher"
  ],
  "properties": [
    {
      "iri": "http://example.org/ontology#hasName",
      "label": "has name",
      "type": "data"
    }
  ],
  "axioms": [
    "Person subClassOf Agent"
  ],
  "created_at": 1699027300
}
```

---

### Create OWL Class

**Endpoint:** `POST /api/ontology/classes`
**Handler:** `api_handler::ontology::create_owl_class`
**Authentication:** Required (Power User)
**Rate Limit:** 10/minute

**Purpose:** Create a new OWL class in ontology.

**Request Body:**

```json
{
  "iri": "string (unique)",
  "label": "string",
  "description": "string | null",
  "parent_iri": "string | null"
}
```

**Response Schema:**

```json
{
  "success": true,
  "class_id": "number",
  "iri": "string"
}
```

**Status Codes:**
- `201 Created` - Class created
- `400 Bad Request` - Invalid IRI or missing required fields
- `409 Conflict` - IRI already exists
- `401 Unauthorized` - Not authenticated
- `403 Forbidden` - Not a power user
- `500 Internal Server Error` - Failed to create class

**Example:**

```bash
curl -X POST http://localhost:4000/api/ontology/classes \
  -H "Content-Type: application/json" \
  -H "Authorization: Nostr <nostr_token>" \
  -d '{
    "iri": "http://example.org/ontology#Student",
    "label": "Student",
    "description": "A person enrolled in education",
    "parent_iri": "http://example.org/ontology#Person"
  }'

# Response:
{
  "success": true,
  "class_id": 123,
  "iri": "http://example.org/ontology#Student"
}
```

---

### Update OWL Class

**Endpoint:** `PUT /api/ontology/classes/{iri}`
**Handler:** `api_handler::ontology::update_owl_class`
**Authentication:** Required (Power User)
**Rate Limit:** 10/minute

**Purpose:** Update an existing OWL class.

**Path Parameter:**
- `iri` - URL-encoded class IRI

**Request Body:**

```json
{
  "label": "string | null",
  "description": "string | null",
  "parent_iri": "string | null"
}
```

**Response Schema:**

```json
{
  "success": true,
  "message": "Class updated successfully"
}
```

**Status Codes:**
- `200 OK` - Class updated
- `404 Not Found` - Class not found
- `400 Bad Request` - Invalid update fields
- `401 Unauthorized` - Not authenticated
- `403 Forbidden` - Not a power user
- `500 Internal Server Error` - Failed to update class

**Example:**

```bash
curl -X PUT http://localhost:4000/api/ontology/classes/http%3A%2F%2Fexample.org%2Fontology%23Student \
  -H "Content-Type: application/json" \
  -H "Authorization: Nostr <nostr_token>" \
  -d '{
    "description": "A person enrolled in formal education"
  }'

# Response:
{
  "success": true,
  "message": "Class updated successfully"
}
```

---

### Delete OWL Class

**Endpoint:** `DELETE /api/ontology/classes/{iri}`
**Handler:** `api_handler::ontology::delete_owl_class`
**Authentication:** Required (Power User)
**Rate Limit:** 10/minute

**Purpose:** Delete an OWL class (cascades to subclasses if specified).

**Path Parameter:**
- `iri` - URL-encoded class IRI

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| cascade | boolean | false | Also delete subclasses |

**Response Schema:**

```json
{
  "success": true,
  "deleted_count": "number"
}
```

**Status Codes:**
- `200 OK` - Class deleted
- `404 Not Found` - Class not found
- `401 Unauthorized` - Not authenticated
- `403 Forbidden` - Not a power user
- `500 Internal Server Error` - Failed to delete class

**Example:**

```bash
curl -X DELETE "http://localhost:4000/api/ontology/classes/http%3A%2F%2Fexample.org%2Fontology%23Student?cascade=true" \
  -H "Authorization: Nostr <nostr_token>"

# Response:
{
  "success": true,
  "deleted_count": 3
}
```

---

### List OWL Properties

**Endpoint:** `GET /api/ontology/properties`
**Handler:** `api_handler::ontology::list_owl_properties`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** List all OWL properties (object, data, annotation).

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| type | string | null | Filter by property type: object, data, annotation |
| domain_iri | string | null | Filter by domain class IRI |
| range_iri | string | null | Filter by range class/datatype IRI |

**Response Schema:**

```json
{
  "properties": [
    {
      "iri": "string",
      "label": "string",
      "property_type": "object | data | annotation",
      "domain_iri": "string | null",
      "range_iri": "string | null"
    }
  ],
  "total": "number"
}
```

**Status Codes:**
- `200 OK` - Properties retrieved
- `500 Internal Server Error` - Failed to load properties

**Example:**

```bash
curl "http://localhost:4000/api/ontology/properties?type=object"

# Response:
{
  "properties": [
    {
      "iri": "http://example.org/ontology#knows",
      "label": "knows",
      "property_type": "object",
      "domain_iri": "http://example.org/ontology#Person",
      "range_iri": "http://example.org/ontology#Person"
    }
  ],
  "total": 1
}
```

---

### Load Ontology Graph

**Endpoint:** `GET /api/ontology/graph`
**Handler:** `ontology_handler::load_ontology_graph`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Get ontology as a graph (nodes = classes, edges = subclass relationships).

**Response Schema:**

```json
{
  "nodes": [
    {
      "id": "number",
      "iri": "string",
      "label": "string",
      "type": "owl:Class",
      "parent_iri": "string | null"
    }
  ],
  "edges": [
    {
      "source": "number",
      "target": "number",
      "label": "rdfs:subClassOf"
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Ontology graph retrieved
- `500 Internal Server Error` - Failed to load ontology

**Example:**

```bash
curl http://localhost:4000/api/ontology/graph

# Response:
{
  "nodes": [
    {
      "id": 1,
      "iri": "http://www.w3.org/2002/07/owl#Thing",
      "label": "Thing",
      "type": "owl:Class",
      "parent_iri": null
    },
    {
      "id": 2,
      "iri": "http://example.org/ontology#Agent",
      "label": "Agent",
      "type": "owl:Class",
      "parent_iri": "http://www.w3.org/2002/07/owl#Thing"
    }
  ],
  "edges": [
    {
      "source": 1,
      "target": 2,
      "label": "rdfs:subClassOf"
    }
  ]
}
```

---

## Bot Orchestration Handlers

### List Bots

**Endpoint:** `GET /api/bots`
**Handler:** `api_handler::bots::list_bots`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** List all registered bot agents.

**Response Schema:**

```json
{
  "bots": [
    {
      "id": "string",
      "name": "string",
      "type": "string",
      "status": "idle | running | stopped | error",
      "created_at": "number",
      "last_active": "number"
    }
  ],
  "total": "number"
}
```

**Status Codes:**
- `200 OK` - Bots retrieved
- `503 Service Unavailable` - Bot orchestration service unavailable

**Example:**

```bash
curl http://localhost:4000/api/bots

# Response:
{
  "bots": [
    {
      "id": "bot-123",
      "name": "Knowledge Collector",
      "type": "researcher",
      "status": "running",
      "created_at": 1699027200,
      "last_active": 1699030800
    }
  ],
  "total": 1
}
```

---

### Create Bot

**Endpoint:** `POST /api/bots`
**Handler:** `api_handler::bots::create_bot`
**Authentication:** Required
**Rate Limit:** 5/minute

**Purpose:** Create and register a new bot agent.

**Request Body:**

```json
{
  "name": "string",
  "type": "researcher | coder | analyst | optimizer | coordinator",
  "config": {
    "key": "value"
  }
}
```

**Response Schema:**

```json
{
  "success": true,
  "bot_id": "string",
  "status": "string"
}
```

**Status Codes:**
- `201 Created` - Bot created
- `400 Bad Request` - Invalid bot configuration
- `503 Service Unavailable` - Bot service unavailable

**Example:**

```bash
curl -X POST http://localhost:4000/api/bots \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Research Assistant",
    "type": "researcher",
    "config": {
      "max_queries": 10,
      "timeout": 300
    }
  }'

# Response:
{
  "success": true,
  "bot_id": "bot-456",
  "status": "idle"
}
```

---

### Execute Bot Command

**Endpoint:** `POST /api/bots/{bot_id}/execute`
**Handler:** `api_handler::bots::execute_bot_command`
**Authentication:** Required
**Rate Limit:** 10/minute

**Purpose:** Send command to a bot for execution.

**Path Parameter:**
- `bot_id` - Bot identifier

**Request Body:**

```json
{
  "command": "string",
  "parameters": {
    "key": "value"
  }
}
```

**Response Schema:**

```json
{
  "success": true,
  "execution_id": "string",
  "status": "queued | running | completed | failed"
}
```

**Status Codes:**
- `202 Accepted` - Command queued
- `404 Not Found` - Bot not found
- `400 Bad Request` - Invalid command
- `503 Service Unavailable` - Bot service unavailable

**Example:**

```bash
curl -X POST http://localhost:4000/api/bots/bot-456/execute \
  -H "Content-Type: application/json" \
  -d '{
    "command": "research",
    "parameters": {
      "query": "knowledge management",
      "depth": 3
    }
  }'

# Response:
{
  "success": true,
  "execution_id": "exec-789",
  "status": "queued"
}
```

---

### Get Bot Status

**Endpoint:** `GET /api/bots/{bot_id}/status`
**Handler:** `api_handler::bots::get_bot_status`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Get current status and metrics of a bot.

**Path Parameter:**
- `bot_id` - Bot identifier

**Response Schema:**

```json
{
  "bot_id": "string",
  "status": "idle | running | stopped | error",
  "current_task": "string | null",
  "metrics": {
    "tasks_completed": "number",
    "tasks_failed": "number",
    "average_duration": "number",
    "last_error": "string | null"
  }
}
```

**Status Codes:**
- `200 OK` - Status retrieved
- `404 Not Found` - Bot not found
- `503 Service Unavailable` - Bot service unavailable

**Example:**

```bash
curl http://localhost:4000/api/bots/bot-456/status

# Response:
{
  "bot_id": "bot-456",
  "status": "running",
  "current_task": "Researching: knowledge management",
  "metrics": {
    "tasks_completed": 15,
    "tasks_failed": 2,
    "average_duration": 45.5,
    "last_error": null
  }
}
```

---

### Delete Bot

**Endpoint:** `DELETE /api/bots/{bot_id}`
**Handler:** `api_handler::bots::delete_bot`
**Authentication:** Required
**Rate Limit:** 10/minute

**Purpose:** Stop and delete a bot.

**Path Parameter:**
- `bot_id` - Bot identifier

**Response Schema:**

```json
{
  "success": true,
  "message": "Bot stopped and deleted"
}
```

**Status Codes:**
- `200 OK` - Bot deleted
- `404 Not Found` - Bot not found
- `503 Service Unavailable` - Bot service unavailable

**Example:**

```bash
curl -X DELETE http://localhost:4000/api/bots/bot-456 \
  -H "Authorization: Nostr <token>"

# Response:
{
  "success": true,
  "message": "Bot stopped and deleted"
}
```

---

### Bot Visualization

**Endpoint:** `GET /api/bots/visualization`
**Handler:** `bots_visualization_handler::get_bots_visualization`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Get visualization data for bot network (nodes = bots, edges = communications).

**Response Schema:**

```json
{
  "nodes": [
    {
      "id": "string",
      "label": "string",
      "type": "bot type",
      "status": "string",
      "position": {"x": "number", "y": "number", "z": "number"}
    }
  ],
  "edges": [
    {
      "source": "string",
      "target": "string",
      "label": "communication",
      "weight": "number"
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Visualization data retrieved
- `503 Service Unavailable` - Bot service unavailable

**Example:**

```bash
curl http://localhost:4000/api/bots/visualization

# Response:
{
  "nodes": [
    {
      "id": "bot-456",
      "label": "Research Assistant",
      "type": "researcher",
      "status": "running",
      "position": {"x": 100.0, "y": 200.0, "z": 0.0}
    }
  ],
  "edges": []
}
```

---

## Analytics Handlers

### Anomaly Detection

**Endpoint:** `POST /api/analytics/anomaly`
**Handler:** `api_handler::analytics::anomaly::detect_anomalies`
**Authentication:** None
**Rate Limit:** 5/minute

**Purpose:** Detect anomalous nodes in graph using GPU-accelerated algorithms.

**Request Body:**

```json
{
  "method": "isolation_forest | local_outlier_factor | one_class_svm",
  "sensitivity": "number (0-1, default: 0.5)",
  "features": ["string (node features to use)"]
}
```

**Response Schema:**

```json
{
  "anomalies": [
    {
      "node_id": "number",
      "score": "number (anomaly score)",
      "reasons": ["string (explanation)"]
    }
  ],
  "method": "string",
  "total_analyzed": "number",
  "execution_time_ms": "number"
}
```

**Status Codes:**
- `200 OK` - Analysis complete
- `400 Bad Request` - Invalid method or parameters
- `503 Service Unavailable` - GPU not available

**Example:**

```bash
curl -X POST http://localhost:4000/api/analytics/anomaly \
  -H "Content-Type: application/json" \
  -d '{
    "method": "isolation_forest",
    "sensitivity": 0.7,
    "features": ["degree", "betweenness_centrality"]
  }'

# Response:
{
  "anomalies": [
    {
      "node_id": 42,
      "score": 0.89,
      "reasons": [
        "Unusually high degree (50 vs avg 5)",
        "Low betweenness centrality (0.01 vs avg 0.15)"
      ]
    }
  ],
  "method": "isolation_forest",
  "total_analyzed": 500,
  "execution_time_ms": 120
}
```

---

### Community Detection (Clustering)

**Endpoint:** `POST /api/analytics/community`
**Handler:** `api_handler::analytics::community::detect_communities`
**Authentication:** None
**Rate Limit:** 5/minute

**Purpose:** Detect communities/clusters in graph using GPU algorithms.

**Request Body:**

```json
{
  "algorithm": "louvain | leiden | label_propagation",
  "resolution": "number (default: 1.0)",
  "min_community_size": "number (default: 3)"
}
```

**Response Schema:**

```json
{
  "communities": [
    {
      "id": "number",
      "nodes": ["number (node IDs)"],
      "size": "number",
      "modularity": "number (quality metric)"
    }
  ],
  "algorithm": "string",
  "total_communities": "number",
  "modularity": "number (overall)",
  "execution_time_ms": "number"
}
```

**Status Codes:**
- `200 OK` - Communities detected
- `400 Bad Request` - Invalid algorithm or parameters
- `503 Service Unavailable` - GPU not available

**Example:**

```bash
curl -X POST http://localhost:4000/api/analytics/community \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "louvain",
    "resolution": 1.0,
    "min_community_size": 5
  }'

# Response:
{
  "communities": [
    {
      "id": 0,
      "nodes": [1, 2, 5, 8, 12, 15, 20],
      "size": 7,
      "modularity": 0.42
    },
    {
      "id": 1,
      "nodes": [3, 4, 6, 7, 9, 10],
      "size": 6,
      "modularity": 0.38
    }
  ],
  "algorithm": "louvain",
  "total_communities": 2,
  "modularity": 0.40,
  "execution_time_ms": 85
}
```

---

### Graph Clustering

**Endpoint:** `POST /api/clustering`
**Handler:** `clustering_handler::cluster_graph`
**Authentication:** None
**Rate Limit:** 5/minute

**Purpose:** Cluster graph nodes based on similarity (alternative to community detection).

**Request Body:**

```json
{
  "method": "kmeans | dbscan | hierarchical",
  "num_clusters": "number (for kmeans)",
  "eps": "number (for dbscan)",
  "min_samples": "number (for dbscan)"
}
```

**Response Schema:**

```json
{
  "clusters": [
    {
      "cluster_id": "number",
      "nodes": ["number"],
      "centroid": {"x": "number", "y": "number", "z": "number"}
    }
  ],
  "method": "string",
  "silhouette_score": "number (quality metric)"
}
```

**Status Codes:**
- `200 OK` - Clustering complete
- `400 Bad Request` - Invalid method or parameters
- `503 Service Unavailable` - GPU not available

**Example:**

```bash
curl -X POST http://localhost:4000/api/clustering \
  -H "Content-Type: application/json" \
  -d '{
    "method": "kmeans",
    "num_clusters": 5
  }'

# Response:
{
  "clusters": [
    {
      "cluster_id": 0,
      "nodes": [1, 5, 8, 12],
      "centroid": {"x": 150.2, "y": 230.5, "z": 10.1}
    }
  ],
  "method": "kmeans",
  "silhouette_score": 0.62
}
```

---

### Real-Time Analytics WebSocket

**Endpoint:** `WS /ws/analytics`
**Handler:** `api_handler::analytics::websocket_integration`
**Authentication:** None
**Rate Limit:** 1 connection/client

**Purpose:** Subscribe to real-time analytics updates.

**Connection:**

```javascript
const ws = new WebSocket('ws://localhost:4000/ws/analytics');

ws.onopen = () => {
  // Subscribe to analytics events
  ws.send(JSON.stringify({
    type: 'subscribe',
    events: ['anomaly_detected', 'community_updated', 'metrics_changed']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Analytics update:', data);
};
```

**Message Types:**

```json
// Anomaly detected
{
  "type": "anomaly_detected",
  "timestamp": "number",
  "data": {
    "node_id": "number",
    "score": "number",
    "reasons": ["string"]
  }
}

// Community updated
{
  "type": "community_updated",
  "timestamp": "number",
  "data": {
    "community_id": "number",
    "nodes_added": ["number"],
    "nodes_removed": ["number"]
  }
}

// Metrics changed
{
  "type": "metrics_changed",
  "timestamp": "number",
  "data": {
    "metric": "string",
    "old_value": "number",
    "new_value": "number"
  }
}
```

---

## Visualization Handlers

### Get Visualization Settings

**Endpoint:** `GET /api/visualisation/settings`
**Handler:** `api_handler::visualisation::get_visualisation_settings`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Get visualization rendering settings.

**Response Schema:**

```json
{
  "rendering": {
    "ambientLightIntensity": "number",
    "enableAmbientOcclusion": "boolean",
    "backgroundColor": "string (hex color)",
    "enableShadows": "boolean",
    "shadowIntensity": "number",
    "enableBloom": "boolean",
    "bloomIntensity": "number"
  },
  "camera": {
    "fov": "number",
    "near": "number",
    "far": "number",
    "position": {"x": "number", "y": "number", "z": "number"}
  },
  "nodes": {
    "defaultSize": "number",
    "minSize": "number",
    "maxSize": "number",
    "defaultColor": "string",
    "highlightColor": "string"
  },
  "edges": {
    "defaultWidth": "number",
    "minWidth": "number",
    "maxWidth": "number",
    "defaultColor": "string",
    "curveIntensity": "number"
  }
}
```

**Status Codes:**
- `200 OK` - Settings retrieved
- `500 Internal Server Error` - Failed to load settings

**Example:**

```bash
curl http://localhost:4000/api/visualisation/settings

# Response:
{
  "rendering": {
    "ambientLightIntensity": 0.5,
    "enableAmbientOcclusion": true,
    "backgroundColor": "#1a1a2e",
    "enableShadows": true,
    "shadowIntensity": 0.3,
    "enableBloom": true,
    "bloomIntensity": 0.8
  },
  "camera": {
    "fov": 75,
    "near": 0.1,
    "far": 10000,
    "position": {"x": 0, "y": 0, "z": 500}
  }
}
```

---

### Update Visualization Settings

**Endpoint:** `PUT /api/visualisation/settings`
**Handler:** `api_handler::visualisation::update_visualisation_settings`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Update visualization settings (applies immediately to connected clients).

**Request Body:**

Partial or complete visualization settings object.

**Response Schema:**

```json
{
  "success": true,
  "message": "Visualization settings updated"
}
```

**Status Codes:**
- `200 OK` - Settings updated
- `400 Bad Request` - Invalid settings
- `500 Internal Server Error` - Failed to update settings

**Example:**

```bash
curl -X PUT http://localhost:4000/api/visualisation/settings \
  -H "Content-Type: application/json" \
  -d '{
    "rendering": {
      "backgroundColor": "#2c3e50",
      "bloomIntensity": 1.2
    }
  }'

# Response:
{
  "success": true,
  "message": "Visualization settings updated"
}
```

---

## XR/Quest3 Handlers

### Initialize XR Session

**Endpoint:** `POST /api/quest3/session`
**Handler:** `api_handler::quest3::initialize_session`
**Authentication:** None
**Rate Limit:** 10/minute

**Purpose:** Initialize WebXR session for Quest 3 device.

**Request Body:**

```json
{
  "session_mode": "immersive-vr | immersive-ar | inline",
  "space_type": "local | local-floor | bounded-floor | unbounded",
  "features": ["hand-tracking", "eye-tracking", "passthrough"]
}
```

**Response Schema:**

```json
{
  "session_id": "string",
  "supported_features": ["string"],
  "reference_space": "string",
  "frame_rate": "number",
  "expires_at": "number"
}
```

**Status Codes:**
- `201 Created` - Session initialized
- `400 Bad Request` - Unsupported session mode or features
- `503 Service Unavailable` - XR not available

**Example:**

```bash
curl -X POST http://localhost:4000/api/quest3/session \
  -H "Content-Type: application/json" \
  -d '{
    "session_mode": "immersive-vr",
    "space_type": "local-floor",
    "features": ["hand-tracking"]
  }'

# Response:
{
  "session_id": "xr-session-789",
  "supported_features": ["hand-tracking"],
  "reference_space": "local-floor",
  "frame_rate": 72,
  "expires_at": 1699034400
}
```

---

### Get XR Session Status

**Endpoint:** `GET /api/quest3/session/{session_id}`
**Handler:** `api_handler::quest3::get_session_status`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Get status of active XR session.

**Path Parameter:**
- `session_id` - Session identifier

**Response Schema:**

```json
{
  "session_id": "string",
  "status": "active | paused | ended",
  "duration_seconds": "number",
  "frame_count": "number",
  "average_fps": "number"
}
```

**Status Codes:**
- `200 OK` - Status retrieved
- `404 Not Found` - Session not found

**Example:**

```bash
curl http://localhost:4000/api/quest3/session/xr-session-789

# Response:
{
  "session_id": "xr-session-789",
  "status": "active",
  "duration_seconds": 120,
  "frame_count": 8640,
  "average_fps": 72
}
```

---

### End XR Session

**Endpoint:** `DELETE /api/quest3/session/{session_id}`
**Handler:** `api_handler::quest3::end_session`
**Authentication:** None
**Rate Limit:** 10/minute

**Purpose:** End an active XR session.

**Path Parameter:**
- `session_id` - Session identifier

**Response Schema:**

```json
{
  "success": true,
  "message": "XR session ended"
}
```

**Status Codes:**
- `200 OK` - Session ended
- `404 Not Found` - Session not found

**Example:**

```bash
curl -X DELETE http://localhost:4000/api/quest3/session/xr-session-789

# Response:
{
  "success": true,
  "message": "XR session ended"
}
```

---

## External Integration Handlers

### RAGFlow Chat

**Endpoint:** `POST /api/ragflow/chat`
**Handler:** `ragflow_handler::send_message`
**Authentication:** None
**Rate Limit:** 10/minute

**Purpose:** Send message to RAGFlow AI chat with RAG.

**Request Body:**

```json
{
  "message": "string",
  "session_id": "string | null",
  "stream": "boolean (default: false)"
}
```

**Response Schema:**

```json
{
  "answer": "string",
  "references": [
    {
      "document": "string",
      "chunk": "string",
      "score": "number"
    }
  ],
  "session_id": "string"
}
```

**Status Codes:**
- `200 OK` - Response received
- `400 Bad Request` - Missing message
- `503 Service Unavailable` - RAGFlow service unavailable

**Example:**

```bash
curl -X POST http://localhost:4000/api/ragflow/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is knowledge management?",
    "stream": false
  }'

# Response:
{
  "answer": "Knowledge management is the process of creating, sharing, using and managing the knowledge and information of an organization...",
  "references": [
    {
      "document": "knowledge-management.md",
      "chunk": "Knowledge management is...",
      "score": 0.92
    }
  ],
  "session_id": "ragflow-session-123"
}
```

---

### Perplexity Search

**Endpoint:** `POST /api/perplexity/search`
**Handler:** `perplexity_handler::search`
**Authentication:** None
**Rate Limit:** 10/minute

**Purpose:** Search using Perplexity AI.

**Request Body:**

```json
{
  "query": "string",
  "model": "llama-3.1-sonar-large-128k-online | llama-3.1-sonar-small-128k-online"
}
```

**Response Schema:**

```json
{
  "answer": "string",
  "citations": [
    {
      "url": "string",
      "title": "string",
      "snippet": "string"
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Search complete
- `400 Bad Request` - Missing query
- `503 Service Unavailable` - Perplexity service unavailable

**Example:**

```bash
curl -X POST http://localhost:4000/api/perplexity/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest trends in knowledge graphs",
    "model": "llama-3.1-sonar-large-128k-online"
  }'

# Response:
{
  "answer": "Recent trends in knowledge graphs include...",
  "citations": [
    {
      "url": "https://example.com/article",
      "title": "Knowledge Graph Trends 2025",
      "snippet": "The latest developments..."
    }
  ]
}
```

---

### Nostr Authentication

**Endpoint:** `POST /api/nostr/auth`
**Handler:** `nostr_handler::authenticate`
**Authentication:** None
**Rate Limit:** 10/minute

**Purpose:** Authenticate using Nostr protocol (NIP-98).

**Request Body:**

```json
{
  "pubkey": "string (hex)",
  "event": {
    "id": "string",
    "pubkey": "string",
    "created_at": "number",
    "kind": "number",
    "tags": [["string"]],
    "content": "string",
    "sig": "string"
  }
}
```

**Response Schema:**

```json
{
  "success": true,
  "token": "string (JWT)",
  "expires_at": "number",
  "user": {
    "pubkey": "string",
    "display_name": "string | null",
    "features": ["string"]
  }
}
```

**Status Codes:**
- `200 OK` - Authenticated
- `400 Bad Request` - Invalid event
- `401 Unauthorized` - Authentication failed

**Example:**

```bash
curl -X POST http://localhost:4000/api/nostr/auth \
  -H "Content-Type: application/json" \
  -d '{
    "pubkey": "a3c5f9e2...",
    "event": {
      "id": "...",
      "pubkey": "a3c5f9e2...",
      "created_at": 1699030800,
      "kind": 27235,
      "tags": [["u", "http://localhost:4000/api/nostr/auth"]],
      "content": "",
      "sig": "..."
    }
  }'

# Response:
{
  "success": true,
  "token": "eyJhbGciOi...",
  "expires_at": 1699117200,
  "user": {
    "pubkey": "a3c5f9e2...",
    "display_name": "Alice",
    "features": ["power_user", "settings_sync"]
  }
}
```

---

### Nostr User Profile

**Endpoint:** `GET /api/nostr/user/{pubkey}`
**Handler:** `nostr_handler::get_user_profile`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Get Nostr user profile.

**Path Parameter:**
- `pubkey` - Nostr public key (hex)

**Response Schema:**

```json
{
  "pubkey": "string",
  "display_name": "string | null",
  "about": "string | null",
  "picture": "string (URL) | null",
  "nip05": "string | null",
  "features": ["string"],
  "created_at": "number"
}
```

**Status Codes:**
- `200 OK` - Profile retrieved
- `404 Not Found` - User not found

**Example:**

```bash
curl http://localhost:4000/api/nostr/user/a3c5f9e2...

# Response:
{
  "pubkey": "a3c5f9e2...",
  "display_name": "Alice",
  "about": "Knowledge enthusiast",
  "picture": "https://example.com/alice.jpg",
  "nip05": "alice@nostr.com",
  "features": ["power_user"],
  "created_at": 1699027200
}
```

---

## WebSocket Handlers

### Primary Graph WebSocket

**Endpoint:** `WS /wss`
**Handler:** `socket_flow_handler::socket_flow_handler`
**Protocol:** Binary (custom format with delta compression)
**Authentication:** None
**Heartbeat:** 5 seconds

**Purpose:** Real-time graph updates with optimized binary protocol.

**Connection:**

```javascript
const ws = new WebSocket('ws://localhost:4000/wss');

ws.binaryType = 'arraybuffer';

ws.onopen = () => {
  console.log('Connected to graph WebSocket');
};

ws.onmessage = (event) => {
  // Binary message parsing
  const data = new Uint8Array(event.data);
  const messageType = data[0];

  if (messageType === 0x01) {
    // Full graph update
    parseFullGraphUpdate(data.slice(1));
  } else if (messageType === 0x02) {
    // Delta update
    parseDeltaUpdate(data.slice(1));
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from graph WebSocket');
};
```

**Message Types:**

| Type | Code | Description |
|------|------|-------------|
| Full Update | 0x01 | Complete graph state |
| Delta Update | 0x02 | Incremental changes |
| Heartbeat | 0x03 | Connection keep-alive |
| Settings Update | 0x04 | Settings changed |
| Physics Update | 0x05 | Physics state changed |

**Binary Format (Delta Update):**

```
[Message Type: 1 byte]
[Timestamp: 8 bytes]
[Update Count: 4 bytes]
[Updates: variable]
  [Node ID: 4 bytes]
  [Position X: 4 bytes float]
  [Position Y: 4 bytes float]
  [Position Z: 4 bytes float]
  [Velocity X: 4 bytes float]
  [Velocity Y: 4 bytes float]
  [Velocity Z: 4 bytes float]
```

---

### Speech WebSocket

**Endpoint:** `WS /ws/speech`
**Handler:** `speech_socket_handler::speech_socket_handler`
**Protocol:** JSON
**Authentication:** None
**Heartbeat:** 10 seconds

**Purpose:** Real-time speech synthesis streaming.

**Connection:**

```javascript
const ws = new WebSocket('ws://localhost:4000/ws/speech');

ws.onopen = () => {
  // Request speech synthesis
  ws.send(JSON.stringify({
    type: 'synthesize',
    text: 'Hello, this is a test.',
    voice: 'default',
    format: 'pcm16'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'audio_chunk') {
    // Base64 audio data
    const audioData = atob(data.chunk);
    playAudio(audioData);
  } else if (data.type === 'synthesis_complete') {
    console.log('Synthesis complete');
  }
};
```

**Message Types:**

```json
// Client → Server: Request synthesis
{
  "type": "synthesize",
  "text": "string",
  "voice": "string",
  "format": "pcm16 | opus | mp3"
}

// Server → Client: Audio chunk
{
  "type": "audio_chunk",
  "chunk": "string (base64)",
  "sequence": "number"
}

// Server → Client: Synthesis complete
{
  "type": "synthesis_complete",
  "total_chunks": "number",
  "duration_seconds": "number"
}

// Server → Client: Error
{
  "type": "error",
  "error": "string"
}
```

---

### MCP Relay WebSocket

**Endpoint:** `WS /ws/mcp-relay`
**Handler:** `mcp_relay_handler::mcp_relay_handler`
**Protocol:** JSON
**Authentication:** None
**Heartbeat:** 30 seconds

**Purpose:** Relay Model Context Protocol messages for AI agents.

**Connection:**

```javascript
const ws = new WebSocket('ws://localhost:4000/ws/mcp-relay');

ws.onopen = () => {
  // Subscribe to agent messages
  ws.send(JSON.stringify({
    type: 'subscribe',
    agent_id: 'agent-123'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'agent_message') {
    console.log('Agent message:', data.message);
  }
};
```

**Message Types:**

```json
// Client → Server: Subscribe to agent
{
  "type": "subscribe",
  "agent_id": "string"
}

// Server → Client: Agent message
{
  "type": "agent_message",
  "agent_id": "string",
  "message": {
    "role": "assistant | user",
    "content": "string"
  },
  "timestamp": "number"
}

// Client → Server: Send message to agent
{
  "type": "send_message",
  "agent_id": "string",
  "message": "string"
}

// Server → Client: Agent status
{
  "type": "agent_status",
  "agent_id": "string",
  "status": "idle | thinking | responding"
}
```

---

### Client Messages WebSocket

**Endpoint:** `WS /ws/client-messages`
**Handler:** `client_messages_handler::websocket_client_messages`
**Protocol:** JSON
**Authentication:** None
**Heartbeat:** 15 seconds

**Purpose:** General-purpose client-to-server messaging bus.

**Connection:**

```javascript
const ws = new WebSocket('ws://localhost:4000/ws/client-messages');

ws.onopen = () => {
  // Send custom message
  ws.send(JSON.stringify({
    type: 'custom',
    payload: {
      action: 'do_something',
      data: {}
    }
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Server response:', data);
};
```

---

### Settings WebSocket

**Endpoint:** `WS /ws/settings`
**Handler:** `websocket_settings_handler::WebSocketSettingsHandler`
**Protocol:** JSON (with optional compression)
**Authentication:** None
**Heartbeat:** 5 seconds

**Purpose:** Real-time settings synchronization with delta updates.

**Connection:**

```javascript
const ws = new WebSocket('ws://localhost:4000/ws/settings');

ws.onopen = () => {
  // Request full sync
  ws.send(JSON.stringify({
    type: 'sync_request',
    last_sync: 0,
    client_id: 'client-123',
    compression_supported: true
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'full_sync') {
    // Apply full settings
    applySettings(data.data);
  } else if (data.type === 'delta_update') {
    // Apply delta
    applyDelta(data.updates);
  }
};
```

**Message Types:**

```json
// Client → Server: Sync request
{
  "type": "sync_request",
  "last_sync": "number (Unix timestamp)",
  "client_id": "string",
  "compression_supported": "boolean"
}

// Server → Client: Full sync
{
  "type": "full_sync",
  "data": {"object (full settings)"},
  "timestamp": "number",
  "checksum": "string"
}

// Server → Client: Delta update
{
  "type": "delta_update",
  "timestamp": "number",
  "updates": [
    {
      "path": "string",
      "value": "any",
      "old_value": "any",
      "operation": "set | delete"
    }
  ]
}

// Client → Server: Update setting
{
  "type": "update_setting",
  "path": "string",
  "value": "any"
}

// Server → Client: Performance delta
{
  "type": "performance_delta",
  "bandwidth_saved": "number (bytes)",
  "compression_ratio": "number",
  "message_count": "number"
}
```

---

## Admin Handlers

### Trigger GitHub Sync

**Endpoint:** `POST /api/admin/sync`
**Handler:** `admin_sync_handler::trigger_sync`
**Authentication:** Required (Power User)
**Rate Limit:** 1/5 minutes

**Purpose:** Manually trigger GitHub repository sync.

**Request Body:**

```json
{
  "force": "boolean (default: false)",
  "paths": ["string (optional)"]
}
```

**Response Schema:**

```json
{
  "success": true,
  "message": "Sync started",
  "sync_id": "string"
}
```

**Status Codes:**
- `202 Accepted` - Sync queued
- `401 Unauthorized` - Not authenticated
- `403 Forbidden` - Not a power user
- `429 Too Many Requests` - Rate limit exceeded

**Example:**

```bash
curl -X POST http://localhost:4000/api/admin/sync \
  -H "Authorization: Nostr <token>" \
  -H "Content-Type: application/json" \
  -d '{"force": true}'

# Response:
{
  "success": true,
  "message": "Sync started",
  "sync_id": "sync-789"
}
```

---

### Get Sync Status

**Endpoint:** `GET /api/admin/sync/{sync_id}`
**Handler:** `admin_sync_handler::get_sync_status`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Get status of GitHub sync job.

**Path Parameter:**
- `sync_id` - Sync job identifier

**Response Schema:**

```json
{
  "sync_id": "string",
  "status": "queued | running | completed | failed",
  "progress": {
    "total_files": "number",
    "processed_files": "number",
    "kg_files": "number",
    "ontology_files": "number",
    "errors": ["string"]
  },
  "started_at": "number",
  "completed_at": "number | null",
  "duration_seconds": "number | null"
}
```

**Status Codes:**
- `200 OK` - Status retrieved
- `404 Not Found` - Sync job not found

**Example:**

```bash
curl http://localhost:4000/api/admin/sync/sync-789

# Response:
{
  "sync_id": "sync-789",
  "status": "running",
  "progress": {
    "total_files": 100,
    "processed_files": 45,
    "kg_files": 30,
    "ontology_files": 15,
    "errors": []
  },
  "started_at": 1699030800,
  "completed_at": null,
  "duration_seconds": null
}
```

---

### Health Check (Consolidated)

**Endpoint:** `GET /api/health/detailed`
**Handler:** `consolidated_health_handler::detailed_health_check`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Comprehensive health check for all services.

**Response Schema:**

```json
{
  "status": "healthy | degraded | unhealthy",
  "timestamp": "number",
  "version": "string",
  "services": {
    "neo4j": {
      "status": "healthy | unhealthy",
      "latency_ms": "number",
      "error": "string | null"
    },
    "sqlite": {
      "status": "healthy | unhealthy",
      "latency_ms": "number",
      "error": "string | null"
    },
    "actors": {
      "status": "healthy | degraded | unhealthy",
      "graph_service": "boolean",
      "settings_service": "boolean",
      "client_coordinator": "boolean"
    },
    "external": {
      "github": "boolean",
      "ragflow": "boolean",
      "perplexity": "boolean",
      "kokoro": "boolean"
    }
  },
  "metrics": {
    "active_connections": "number",
    "total_nodes": "number",
    "total_edges": "number",
    "memory_usage_mb": "number",
    "uptime_seconds": "number"
  }
}
```

**Status Codes:**
- `200 OK` - All services healthy or degraded
- `503 Service Unavailable` - Critical services unhealthy

**Example:**

```bash
curl http://localhost:4000/api/health/detailed

# Response:
{
  "status": "healthy",
  "timestamp": 1699030800,
  "version": "1.0.0",
  "services": {
    "neo4j": {
      "status": "healthy",
      "latency_ms": 5,
      "error": null
    },
    "sqlite": {
      "status": "healthy",
      "latency_ms": 1,
      "error": null
    },
    "actors": {
      "status": "healthy",
      "graph_service": true,
      "settings_service": true,
      "client_coordinator": true
    },
    "external": {
      "github": true,
      "ragflow": true,
      "perplexity": false,
      "kokoro": true
    }
  },
  "metrics": {
    "active_connections": 5,
    "total_nodes": 500,
    "total_edges": 1200,
    "memory_usage_mb": 450,
    "uptime_seconds": 86400
  }
}
```

---

## Utility Handlers

### Workspace Operations

**Endpoints:**
- `GET /api/workspace/files` - List workspace files
- `POST /api/workspace/upload` - Upload file to workspace
- `GET /api/workspace/files/{path}` - Get file content
- `DELETE /api/workspace/files/{path}` - Delete file

**Handler:** `workspace_handler`

**Purpose:** Manage user workspace files (separate from GitHub sync).

---

### Static Pages

**Endpoint:** `GET /api/pages/{page_name}`
**Handler:** `pages_handler::get_page`
**Authentication:** None
**Rate Limit:** 10/second

**Purpose:** Serve static HTML/Markdown pages.

---

### Physics Constraints

**Endpoints:**
- `GET /api/constraints` - List constraints
- `POST /api/constraints` - Create constraint
- `PUT /api/constraints/{id}` - Update constraint
- `DELETE /api/constraints/{id}` - Delete constraint

**Handler:** `constraints_handler`

**Purpose:** Manage physics constraints (pin nodes, distance constraints, etc.).

---

### Client Error Logging

**Endpoint:** `POST /api/client-logs`
**Handler:** `client_log_handler::handle_client_logs`
**Authentication:** None
**Rate Limit:** 100/minute

**Purpose:** Log client-side errors to server for debugging.

**Request Body:**

```json
{
  "level": "error | warn | info | debug",
  "message": "string",
  "stack": "string | null",
  "timestamp": "number",
  "user_agent": "string",
  "url": "string"
}
```

**Response Schema:**

```json
{
  "success": true,
  "log_id": "string"
}
```

**Status Codes:**
- `200 OK` - Log received
- `400 Bad Request` - Invalid log format

---

## Authentication & Authorization

### Authentication Methods

1. **None (Public Endpoints)**
   - Most read endpoints
   - No authentication required

2. **Nostr (NIP-98)**
   - POST /api/nostr/auth
   - Returns JWT token
   - Token in Authorization header: `Authorization: Nostr <token>`

3. **Feature Flags**
   - Power User: Can modify ontology, trigger sync
   - Settings Sync: Can modify and sync settings

### Authorization Flow

```
Client
  │
  │ POST /api/nostr/auth
  │ { pubkey, event (signed) }
  │
  ▼
Nostr Handler
  │
  │ Verify event signature
  │ Check pubkey in database
  │ Check feature flags from env
  │
  ▼
Generate JWT
  │
  │ Claims: pubkey, features, expiry
  │
  ▼
Client
  │
  │ Store token
  │ Include in subsequent requests
  │ Authorization: Nostr <token>
  │
  ▼
Protected Endpoint
  │
  │ Extract token from header
  │ Verify JWT signature
  │ Check expiry
  │ Check required features
  │
  ▼
Execute Handler
```

### Feature Access

**Environment Variables:**

```bash
# Power users (comma-separated pubkeys)
POWER_USERS=pubkey1,pubkey2,pubkey3

# Settings sync users
SETTINGS_SYNC_USERS=pubkey1,pubkey2

# All features
ALL_FEATURES_USERS=pubkey1
```

**Feature Checks in Handlers:**

```rust
if !state.feature_access.is_power_user(&pubkey) {
    return Err(actix_web::error::ErrorForbidden(
        "Power user access required"
    ));
}
```

---

## Error Handling

### Error Response Format

**Standard Error Response:**

```json
{
  "error": "string (error message)",
  "code": "string (error code, optional)",
  "details": "string | object (additional info, optional)"
}
```

### HTTP Status Codes

| Code | Meaning | Usage |
|------|---------|-------|
| 200 OK | Success | Standard success response |
| 201 Created | Resource created | POST endpoints that create resources |
| 202 Accepted | Async operation queued | Long-running operations |
| 204 No Content | Success, no body | DELETE operations |
| 400 Bad Request | Invalid input | Validation errors, malformed JSON |
| 401 Unauthorized | Not authenticated | Missing or invalid auth token |
| 403 Forbidden | Not authorized | Insufficient permissions |
| 404 Not Found | Resource not found | Invalid ID, missing resource |
| 409 Conflict | Resource conflict | Duplicate IRI, concurrent modification |
| 413 Payload Too Large | Request too large | File upload size limit |
| 429 Too Many Requests | Rate limit exceeded | Too many requests |
| 500 Internal Server Error | Server error | Uncaught exceptions, database errors |
| 503 Service Unavailable | Service down | External service unavailable |

### Error Code Examples

```rust
// 400 Bad Request
bad_request!("Invalid page size: must be between 1 and 1000")

// 404 Not Found
not_found!("Node with ID {} not found", node_id)

// 500 Internal Server Error
error_json!("Failed to retrieve graph data")

// 503 Service Unavailable
service_unavailable!("RAGFlow service is currently unavailable")
```

---

## Rate Limiting

### Rate Limit Configuration

**Implementation:** In-memory token bucket (per-endpoint, per-IP)
**Headers:**

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1699030860
```

### Rate Limits by Endpoint Category

| Category | Limit | Window |
|----------|-------|--------|
| Health checks | Unlimited | - |
| Graph reads | 10/second | 1 second |
| Graph writes | 1/minute | 1 minute |
| Settings reads | 10/second | 1 second |
| Settings writes | 10/second | 1 second |
| Batch updates | 5/minute | 1 minute |
| File operations | 10/second | 1 second |
| Ontology reads | 10/second | 1 second |
| Ontology writes | 10/minute | 1 minute |
| Bot operations | 10/minute | 1 minute |
| Analytics | 5/minute | 1 minute |
| External integrations | 10/minute | 1 minute |
| Admin operations | 1/5 minutes | 5 minutes |

### Rate Limit Response

```json
{
  "error": "Rate limit exceeded",
  "code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 45
}
```

**Status Code:** `429 Too Many Requests`

---

## Cross-References

- **Services Architecture:** [services-architecture.md](services-architecture.md)
- **Database Schemas:** [schemas.md](schemas.md)
- **WebSocket Protocol Details:**  (TODO)
- **CQRS Commands/Queries:** [hexagonal-cqrs.md](hexagonal-cqrs.md)
- **Authentication Guide:**  (TODO)

---

**Document Status:** Production
**Maintainer:** API Team
**Review Cycle:** Monthly
